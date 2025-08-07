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
        active_quotes_flag,
        agent,
        latency_ns=200_000,
        inventory_penalty=0.001,
        max_inventory=5,
        min_order_size=0.001,
        tick_size= 0.01,
        initial_cash=0.0,
        cost_in_bps=0.0,
        fixed_cost=0.0,
        simulator_mode="eval",  # either "train" or "eval"

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
        self.active_quotes_flag = active_quotes_flag 
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
        self.initial_cash = initial_cash
        self.cash = self.prev_cash = initial_cash
        fixed_cost=0.0,
        self.reward = 0.0
        self.realized_pnl = 0.0
        self.trade_count = 0  # Count of trades executed
        self.active_orders_count = 0  # Count of active orders
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

    # def update_reward(self, mid):
    #     """
    #     RL reward function :
    #     Reward = ΔCash - Inventory Penalty
    #     """
    #     if self.simulator_mode == "train":
    #         delta_cash = self.cash - self.prev_cash
    #         penalty = self.inventory_penalty * (self.inventory ** 2)
    #         self.reward = delta_cash - penalty
    #         self.prev_cash = self.cash
    def update_reward(self, mid):
        """
        RL reward function (only in training mode):
        Reward = ΔCash - Inventory Penalty

        - ΔCash = realized profit/loss from fills
        - Inventory Penalty = 0 if |inventory| ≤ 1
                            = λ * inventory² if |inventory| > 1
        """
    
        delta_cash = self.cash - self.prev_cash

        # Zero penalty for small inventory
        if abs(self.inventory) > 2:
            penalty = self.inventory_penalty * (self.inventory ** 2)
        else:
            penalty = 0.0

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
        folder_path = Path(__file__).parent.parent.parent.parent / "logs/eval_logs" / current_dt
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
        qty = abs(self.inventory)
        if qty < self.min_order_size:
            print('invemtory too small to square off:', self.inventory)
            return  # Ignore dust

        price = best_bid if self.inventory > 0 else best_ask
       
        side = "ask" if self.inventory > 0 else "bid"
        self.update_position(side, price, qty)
        self.inventory = 0  # fully closed
        ts_current = lob_row["ts"].item()
        mid = (best_bid + best_ask) / 2
        self.active_orders_count += 1  # Increment active orders count
        self.trade_count += 1  # Increment trade count
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
                self.active_orders_count += 1  # Increment active orders count
                self.pending_orders[side] = None

        # Fill logic
        for side in ["bid", "ask"]:
            order = self.active_orders[side]
            if order and not order.is_filled():
                if order.try_fill(best_bid, best_ask):
                    self.trade_count += 1  # Increment trade count
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

        # Update agent inventory
        self.agent.inventory = self.inventory

        # Training: update reward
        self.update_reward(mid)


        # === Quote logic ===
        if self.simulator_mode == "train":
            # Use quotes already injected by agent via SimEnv
            new_bid = self.pending_quotes["bid"]
            new_ask = self.pending_quotes["ask"]
        else:
            # Evaluation mode — generate quotes using agent's policy
            state = self.get_state_vector()

            # Get basic LOB snapshot (best ask and best bid)
            lob_state = [best_ask, best_bid]
            quotes = self.agent.get_quotes_eval(state, lob_state=lob_state)
            new_bid = (quotes["bid_px"], quotes["bid_qty"])
            new_ask = (quotes["ask_px"], quotes["ask_qty"])

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


    def reset_state(self):
        """
        Resets the internal state of the simulator to prepare for a new training episode.

        This function:
        - Moves the LOB pointer back to the start (like resetting time)
        - Resets inventory and cash to initial conditions
        - Clears evaluation logs used during simulation
        """
        self.ptr = 0                          # Reset order book pointer to beginning
        self.inventory = 0                    # Clear inventory
        self.trade_count = 0                   # Reset trade count
        self.active_orders_count = 0           # Reset active orders count
        self.cash = self.initial_cash         # Reset cash
        self.prev_cash = self.initial_cash    # Used for reward calculation
        self.eval_log = []                    # Empty the log used for tracking performance
        
        self.reward = 0.0                     # Reset reward to zero
        self.realized_pnl = 0.0               # Reset realized PnL
        self.state = None                     # Reset state to None (will be set by agent)

        # Reset all order states
        self.active_orders = {"bid": None, "ask": None}
        self.pending_orders = {"bid": None, "ask": None}
        self.pending_quotes = {"bid": None, "ask": None}
        self.t_ready = {"bid": 0, "ask": 0}

    def get_state_vector(self) -> list[float]:
        """
        Constructs the current normalized state vector for the agent using:
        - LOB features (e.g., spread, imbalance, etc.)
        - Active quote offsets and sizes (optional)
        - Inventory level (optional, added as last feature)

        Feature scaling:
        - Prices (offsets) are normalized using tick_size
        - Quantities and inventory are normalized using max_inventory

        Returns:
            list[float]: Flattened state vector as list of floats
        """
        lob_row = self.book[self.ptr]

        # Fallback to zero quotes if no active orders exist
        active_quotes = {
            "bid_px": self.active_orders["bid"].price if self.active_orders["bid"] else 0.0,
            "ask_px": self.active_orders["ask"].price if self.active_orders["ask"] else 0.0,
            "bid_qty": self.active_orders["bid"].qty if self.active_orders["bid"] else 0.0,
            "ask_qty": self.active_orders["ask"].qty if self.active_orders["ask"] else 0.0,
        }

        # Extract and scale features
        features_dict = extract_features(
            row=lob_row,
            feature_columns=self.feature_columns,
            include_inventory=self.inventory_feature_flag,
            inventory=self.inventory,
            include_active_quotes=self.active_quotes_flag,
            active_quotes=active_quotes,
            scale_features=True,
            max_inventory=self.max_inventory,
            tick_size=self.tick_size
        )
        feat_vect= [float(x) for x in features_dict if isinstance(x, (int, float))]
        return list(feat_vect)  # agent expects flat vector


    def is_done(self) -> bool:
        """
        Checks whether the simulation has reached the end of the time series.

        Returns:
            bool: True if no more LOB updates remain; otherwise False.
        """
        return self.ptr >= len(self.book)  # Simulation ends when pointer exceeds data length


