from dspy.sim.order import Order
from dspy.utils import ts_to_str  # Existing helper for formatting timestamps
from dspy.features.feature_utils import apply_batch_features, extract_features, flatten_features
import csv
from datetime import datetime
from pathlib import Path

class MarketSimulator:
    def __init__(
        self,
        book,
        feature_config,
        feature_columns,
        inventory_feature_flag,
        agent,
        latency_ns=200_000,
        inventory_penalty=0.001,
        max_inventory=5,
        min_order_size=0.001,
        tick_size= 0.01,
        initial_cash=0.0,
        cost_in_bps=0.0,
        fixed_cost=0.0,
        simulator_mode="eval",  # either "training" or "eval"

    ):
        """
        Initialize market simulator.

        Parameters:
        - book: polars DataFrame with LOB and feature columns
        - feature_config: dict from features.json
        - feature_columns: list of column names to use as features
        - inventory_feature_flag: bool, whether to include inventory as feature
        - agent: trading agent object
        - latency_ns: delay for order confirmation
        - inventory_penalty: penalty in reward shaping
        - max_inventory: position cap
        - initial_cash: starting capital
        - simulator_mode: "training" or "eval" (affects reward/logging)
        """

        self.book = book
        self.feature_config = feature_config
        self.feature_columns = feature_columns
        self.inventory_feature_flag = inventory_feature_flag
        self.agent = agent
        self.latency_ns = latency_ns
        self.inventory_penalty = inventory_penalty
        self.max_inventory = max_inventory
        self.simulator_mode = simulator_mode
        self.cost_in_bps= cost_in_bps
        self.fixed_cost = fixed_cost
        self.tick_size = tick_size
        self.min_order_size = min_order_size
        self.ptr = 0
        self.inventory = 0.0
        self.cash = self.prev_cash = initial_cash
        fixed_cost=0.0,
        self.reward = 0.0
        self.realized_pnl = 0.0
        self.state = None

        self.active_orders = {"bid": None, "ask": None}
        self.pending_orders = {"bid": None, "ask": None}
        self.pending_quotes = {"bid": None, "ask": None}
        self.t_ready = {"bid": 0, "ask": 0}

        # Eval mode tracking (log full pnl)
        self.eval_log = []

    def total_pnl(self, mid):
        """Return total PnL = cash + inventory * midprice"""
        return self.cash + self.inventory * mid

    def update_position(self, side, execution_price, qty):
        """Update inventory and cash after a fill."""
        trading_cost = execution_price * qty * (self.cost_in_bps / 10_000) + self.fixed_cost
        if side == "bid":
            self.inventory += qty
            self.cash -= execution_price * qty + trading_cost
        elif side == "ask":
            self.inventory -= qty
            self.cash += execution_price * qty - trading_cost

    def update_reward(self, mid):
        """
        RL reward function :
        Reward = ΔCash - Inventory Penalty
        """
        if self.simulator_mode == "training":
            delta_cash = self.cash - self.prev_cash
            penalty = self.inventory_penalty * (self.inventory ** 2)
            self.reward = delta_cash - penalty
            self.prev_cash = self.cash

    def log_eval_metrics(self, ts, mid, trade_side, trade_price, trade_qty,best_bid=None, best_ask=None):
        """
        Logs timestamp, PnL, inventory, and trade details on each fill (called only on trade).
        """
        self.eval_log.append({
            "timestamp": ts_to_str(ts),
            "mid": mid,
            "cash": self.cash,
            "inventory": self.inventory,
            "pnl": self.total_pnl(mid),
            "trade_side": trade_side,
            "trade_price": trade_price,
            "trade_quantity": trade_qty,
            "best_bid": best_bid,
            "best_ask": best_ask
        })


    def save_eval_log(self, run_start: str, run_end: str):
        """
        Saves the evaluation log to a CSV file.
        Creates a unique folder inside logs/ with current datetime.
        The filename is based on simulation interval start and end.
        """
        if not self.eval_log:
            print("No trade logs to save.")
            return

        # Format current datetime to use as folder name
        current_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = Path(__file__).parent.parent.parent.parent / "logs" / current_dt
        folder_path.mkdir(parents=True, exist_ok=True)

        # Create filename like 2025-04-01_00-00-00__2025-04-02_12-59-59.csv
        clean_start = run_start.replace(":", "-").replace(" ", "_")
        clean_end = run_end.replace(":", "-").replace(" ", "_")
        file_name = f"{clean_start}__{clean_end}.csv"

        log_path = folder_path / file_name

        # Save log
        keys = self.eval_log[0].keys()
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.eval_log)

        print(f"Saved evaluation log to {log_path}")


    def square_off(self, lob_row):
        """
        Force close remaining inventory at best bid/ask immediately (used at final step).
        """
        best_bid = lob_row["bids[0].price"].item()
        best_ask = lob_row["asks[0].price"].item()
        print('sqare off called')
        qty = abs(self.inventory)
        print('qty:', qty, 'min_order_size:', self.min_order_size)
        if qty < self.min_order_size:
            print('invemtory too small to square off:', self.inventory)
            return  # Ignore dust

        price = best_bid if self.inventory > 0 else best_ask
        print(f"Square off: {self.inventory} @ {price:.2f}")
        print('bestbid:',best_bid, 'bestask:',best_ask)
        side = "ask" if self.inventory > 0 else "bid"
        self.update_position(side, price, qty)
        self.inventory = 0  # fully closed
        ts_current = lob_row["ts"].item()
        mid = (best_bid + best_ask) / 2
        self.log_eval_metrics(
                    ts_current, mid,
                    trade_side=side,
                    trade_price=price,
                    trade_qty=qty,
                    best_bid=best_bid,
                    best_ask=best_ask
                )
        



    def send_order(self, side, quote, ts):
        """
        Send an order with the given side/price/qty.
        Places into `pending_orders`, activates after latency.
        """
        price, qty = quote
        self.pending_orders[side] = Order(
            side=side,
            price=price,
            qty=qty,
            ts_sent=ts,
            delay_ns=self.latency_ns,
        )
        self.t_ready[side] = ts + self.latency_ns

    def step(self):
        """
        Executes one time step (tick) of simulation.
        """
        ts_current = self.book["ts"][self.ptr]
        # lob_row = self.book.row(self.ptr)
        lob_row = self.book[self.ptr]
        best_bid = lob_row["bids[0].price"].item()
        best_ask = lob_row["asks[0].price"].item()
        mid = (best_bid + best_ask) / 2

        # Promote pending orders to active if delay passed
        for side in ["bid", "ask"]:
            pending = self.pending_orders[side]
            if pending and ts_current >= pending.ts_active:
                self.active_orders[side] = pending
                self.active_orders[side].activate(ts_current)
                self.pending_orders[side] = None

        # Fill logic
        for side in ["bid", "ask"]:
            order = self.active_orders[side]
            if order and not order.is_filled():
                if order.try_fill(best_bid, best_ask):
                    if side == "bid":
                        execution_price = min(order.price, best_ask)
                    else:
                        execution_price = max(order.price, best_bid)
                    self.update_position(side, execution_price, order.qty)
                     # Evaluation: log PnL
                    self.log_eval_metrics(
                                ts_current, mid,
                                trade_side=side,
                                trade_price=execution_price,
                                trade_qty=order.qty,
                                best_bid=best_bid,
                                best_ask=best_ask
                            )

                    self.active_orders[side] = None  # remove filled order

        # Training: update reward
        self.update_reward(mid)

        # Get agent's new quote
        agent_input = extract_features(
            row=lob_row,
            feature_columns=self.feature_columns,
            include_inventory=self.inventory_feature_flag,
            inventory=self.inventory)


        new_bid, new_ask = self.agent.get_quotes(agent_input)

        for side, new_quote in [("bid", new_bid), ("ask", new_ask)]:
            price, qty = new_quote

            # Skip if there's a pending order
            if self.pending_orders[side]:
                self.pending_quotes[side] = new_quote
                continue

            # Check inventory constraints
            if side == "bid" and self.inventory + qty > self.max_inventory:
                continue
            if side == "ask" and self.inventory - qty < -self.max_inventory:
                continue

            # Send new quote if it differs from current active
            current = self.active_orders[side]
            if current is None or current.quote != new_quote:
                self.send_order(side, new_quote, ts_current)
                self.pending_quotes[side] = None

        # Advance pointer to next tick
        self.ptr += 1
