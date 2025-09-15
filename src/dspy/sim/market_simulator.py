from dspy.sim.order import Order
from dspy.utils import ts_to_str  # Existing helper for formatting timestamps
from dspy.features.feature_utils import apply_batch_features, extract_features, flatten_features
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import gc
import polars as pl

class MarketSimulator:
    def __init__(
        self,
        book,
        feature_config,
        feature_columns,
        inventory_feature_flag,
        active_quotes_flag,
        pending_quotes_flag,
        active_age_flag,
        pending_age_flag,
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
        eval_log_flag = False,
        folder_label = None

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

        self.feature_config = feature_config
        self.feature_columns = feature_columns
        self.inventory_feature_flag = inventory_feature_flag
        self.active_quotes_flag = active_quotes_flag
        self.pending_quotes_flag=pending_quotes_flag,
        self.active_age_flag = active_age_flag,
        self.pending_age_flag = pending_age_flag, 
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
        self.reward = 0.0
        self.realized_pnl = 0.0
        self.total_cost = 0.0
        self.max_drawdown = float("-inf")
        self.drawdown =0.0
        self.max_pnl = float("-inf")
        self.tpnl =0.0
        self.trade_count = 0  # Count of trades executed
        self.active_orders_count = 0  # Count of active orders
        self.zero_reward_count =0
        self.state = None
        self.eval_log_flag = eval_log_flag
        self.prev_reward_time = None  # Track time of last reward update

        self.active_orders = {"bid": None, "ask": None}
        self.pending_orders = {"bid": None, "ask": None}
        self.pending_quotes = {"bid": None, "ask": None}
        self.t_ready = {"bid": 0, "ask": 0}

        # Eval mode tracking (log full pnl)
        self.eval_log = []

        # Ensure eager and keep only needed cols
        if isinstance(book, pl.LazyFrame):
            book = book.collect(streaming=True)
        need = set(self.feature_columns) | {"ts","bids[0].price", "asks[0].price", "mid"}
        have = [c for c in book.columns if c in need]
        book = book.select(have).rechunk()

        # Cast features + prices to Float32 once; timestamps stay int64
        cast_cols = [c for c in self.feature_columns if c in book.columns] + \
                    [c for c in ("bids[0].price","asks[0].price", "mid") if c in book.columns]
        book = book.with_columns([pl.col(c).cast(pl.Float32) for c in cast_cols])

        # ---- FP32 caches (NumPy) ----
        self._best_bid = book["bids[0].price"].to_numpy().astype(np.float32, copy=False)
        self._best_ask = book["asks[0].price"].to_numpy().astype(np.float32, copy=False)
        # Save RAM by NOT caching mid; compute on the fly
        self._ts = book["ts"].to_numpy() if "ts" in book.columns else None  # int64

        # Base feature matrix (float32)
        feat_df = book.select([c for c in self.feature_columns if c in book.columns])
        self._feat_mat = feat_df.to_numpy().astype(np.float32, copy=False)

        self._n = self._feat_mat.shape[0] # number of ticks

        # Drop Polars DF to avoid 20–30 GB RSS bloat
        del book, feat_df
        gc.collect()

        # Preallocate state buffer (float32)
        self._base_dim = self._feat_mat.shape[1]
        
        #precompute scalers/caps (and state dim) once
        self._recompute_state_caches()
        self._state_buf = np.empty(self.state_dim, dtype=np.float32)
        # self._extra_dim = (4 if self.active_quotes_flag else 0) + (4 if self.pending_quotes_flag else 0) + (1 if self.inventory_feature_flag else 0) +(2 if self.active_age_flag else 0) +(2 if self.pending_age_flag else 0)  
        # self._state_buf = np.empty((self._base_dim + self._extra_dim,), dtype=np.float32)

        # Accumulators in float64
        self.cum_reward = np.float64(0.0)

        self.folder_label = folder_label


    def total_pnl(self, mid):
        """Return total PnL = cash + inventory * midprice"""
        return self.cash + self.inventory * mid

    def update_position(self, side, execution_price, qty):
        """Update inventory and cash after a fill."""
        trading_cost = execution_price * abs(qty) * (self.cost_in_bps / 10_000) + self.fixed_cost
        self.total_cost += trading_cost
        if side == "bid":
            self.inventory += qty
            self.cash -= execution_price * qty + trading_cost
        elif side == "ask":
            self.inventory -= qty
            self.cash += execution_price * qty - trading_cost
    
    def update_pnl_dd(self,mid):
        self.tpnl=self.total_pnl(mid)
        if self.tpnl>self.max_pnl:
            self.max_pnl =  self.total_pnl(mid)
        
        self.drawdown = max(self.max_pnl - self.tpnl,0)
        
        if self.drawdown > self.max_drawdown:
            self.max_drawdown = self.drawdown


    def update_reward(self, mid,time_r=None):
        """
        RL reward function (only in training mode):
        Reward = ΔCash - Inventory Penalty - normalized 

        - ΔCash = realized profit/loss from fills - normalized by mid*max_inventory
        - Inventory Penalty =  = λ * inventory²*dt  - normalized by max_inventory
        """
    
        delta_cash = (self.cash - self.prev_cash)/ (mid*self.max_inventory if mid !=0 else 1)  # Normalize by mid to keep reward scale consistent

        if abs(self.inventory) > 0: #2*self.min_order_size:
            dt = max(0.0,(time_r-self.prev_reward_time if self.prev_reward_time is not None else 0))
            penalty = self.inventory_penalty * ((self.inventory/self.max_inventory) ** 2)*dt # Scale penalty by time held
        else:
            penalty = 0.0
        self.reward = np.clip(delta_cash - penalty, -2.5, 2.5) # Clip reward for stability
        self.prev_cash = self.cash
        self.prev_reward_time = time_r



    def log_eval_metrics(self, ts, mid, trade_side, trade_price, trade_qty,best_bid=None, best_ask=None):
        """
        Logs timestamp, PnL, inventory, and trade details on each fill (called only on trade).
        """
        self.eval_log.append({
            "timestamp": ts_to_str(ts),
            "mid": mid,
            "cash": self.cash,
            "inventory": self.inventory,
            "total_cost":self.total_cost,
            "pnl": self.tpnl,
            "drawdown":self.drawdown,
            "max_drawdown":self.max_drawdown,
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
        folder_path = Path(__file__).parent.parent.parent.parent / "logs/eval_logs" / self.folder_label
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

    
    def square_off(self, idx: int):
        """Force close remaining inventory at best bid/ask immediately."""
        qty = abs(self.inventory)
        if qty < self.min_order_size:
            return

        best_bid = float(self._best_bid[idx])
        best_ask = float(self._best_ask[idx])
        mid      = (best_bid + best_ask)*0.5
        ts_now   = self._ts[idx] if self._ts is not None else None

        # If long, sell at best bid; if short, buy at best ask
        side  = "ask" if self.inventory > 0 else "bid"
        price = best_bid if self.inventory > 0 else best_ask

        self.update_position(side, price, qty)
        self.inventory = 0.0
        self.update_pnl_dd(mid)
        self.update_reward(mid,ts_now)

        if self.eval_log_flag:
            self.log_eval_metrics(
                ts_now, mid,
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

    def pre_step(self):
        """
        Prepares and processes fills for the current tick.
        Handles order activation, fills, reward update, and state update.
        """
        i = self.ptr

        best_bid = self._best_bid[i]
        best_ask = self._best_ask[i]
        mid = (best_bid + best_ask)*0.5  # FP32 math here is fine
        ts_current = self._ts[i] if self._ts is not None else None

        #Locals for faster process 
        active_orders = self.active_orders 
        
        # Promote pending orders to active if delay passed
        for side in ["bid", "ask"]:
            pending = self.pending_orders[side]
            if pending and ts_current >= pending.ts_active:
                active_orders[side] = pending
                active_orders[side].activate(ts_current)
                self.active_orders_count += 1  # Increment active orders count
                self.pending_orders[side] = None

        # Fill logic
        for side in ["bid", "ask"]:
            order = active_orders[side]
            if order and not order.is_filled():
                if order.try_fill(best_bid, best_ask):
                    self.trade_count += 1  # Increment trade count
                    if side == "bid":
                        execution_price = min(order.price, best_ask)
                    else:
                        execution_price = max(order.price, best_bid)
                    self.update_position(side, execution_price, order.qty)
                    self.update_pnl_dd(mid)
                     # Evaluation: log PnL
                    if self.eval_log_flag:
                        self.log_eval_metrics(
                                    ts_current, mid,
                                    trade_side=side,
                                    trade_price=execution_price,
                                    trade_qty=order.qty,
                                    best_bid=best_bid,
                                    best_ask=best_ask
                                )

                    active_orders[side] = None  # remove filled order

        # Update agent inventory
        self.agent.inventory = self.inventory

        # Training: update reward
        self.update_reward(mid,ts_current)
        if self.reward ==0:
            self.zero_reward_count +=1


        

    def step(self):
        """
        Executes one time step (tick) of simulation.
        """
        i = self.ptr

        best_bid = self._best_bid[i]
        best_ask = self._best_ask[i]
        # mid = (best_bid + best_ask)*0.5  # FP32 math here is fine
        ts_current = self._ts[i] if self._ts is not None else None
        
        # === Quote logic ===
        if self.simulator_mode == "train":
            # Use quotes already injected by agent via SimEnv
            new_bid = self.pending_quotes["bid"]
            new_ask = self.pending_quotes["ask"]
        else:
            # Evaluation mode — generate quotes using agent's policy
            state = self.get_state_vector(i)

            # Get basic LOB snapshot (best ask and best bid)
            quotes = self.agent.get_quotes_eval(state, best_ask, best_bid)
            new_bid = (quotes["bid_px"], quotes["bid_qty"])
            new_ask = (quotes["ask_px"], quotes["ask_qty"])

            # self.pending_quotes["bid"] = new_bid
            # self.pending_quotes["ask"] = new_ask

        for side, new_quote in [("bid", new_bid), ("ask", new_ask)]:
            
            if new_quote is None:
                continue

            price, qty = new_quote
            

            # Skip if there's a pending order
            if self.pending_orders[side] is not None:
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
        self.zero_reward_count =0
        self.cash = self.initial_cash         # Reset cash
        self.tpnl = 0.0
        self.max_pnl = 0.0
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.total_cost = 0.0
        self.prev_cash = self.initial_cash    # Used for reward calculation
        self.prev_reward_time = None 
        self.eval_log = []                    # Empty the log used for tracking performance
        
        self.reward = 0.0                     # Reset reward to zero
        self.realized_pnl = 0.0               # Reset realized PnL
        self.state = None                     # Reset state to None (will be set by agent)

        # Reset all order states
        self.active_orders = {"bid": None, "ask": None}
        self.pending_orders = {"bid": None, "ask": None}
        self.pending_quotes = {"bid": None, "ask": None}
        self.t_ready = {"bid": 0, "ask": 0}


    def get_state_vector(self, idx: int) -> np.ndarray:
        # ===== base features =====
        buf = self._state_buf
        buf[:self._base_dim] = self._feat_mat[idx]
        off = self._base_dim

        # ===== time & local scalers (bind cached to locals for speed) =====
        time_c = float(self._ts[idx]) if self._ts is not None else float(idx * self._late)
        A_MAX_TICKS = self._A_max

        # orders (local refs)
        active_bid  = self.active_orders["bid"]
        active_ask  = self.active_orders["ask"]
        pending_bid = self.pending_orders["bid"]
        pending_ask = self.pending_orders["ask"]

        # norms (keep your names, pull from caches)
        mid      = (self._best_bid[idx] + self._best_ask[idx]) * 0.5
        denom    = self._denom
        inv_norm = self._inv_norm
        late     = self._late

        # ===== ACTIVE quotes block =====
        if self.active_quotes_flag:
            # --- active bid ---
            if active_bid is None:
                m_b = 0.0; off_b = qty_b = age_b = 0.0
            else:
                m_b   = 1.0
                off_b = (float(active_bid.price) - mid) / denom
                qty_b = float(active_bid.qty) * inv_norm
                dt    = max(0.0, time_c - float(active_bid.ts_active))
                age_b = min(1.0, (dt / late) / A_MAX_TICKS)

            buf[off + 0] = np.float32(m_b)
            buf[off + 1] = np.float32(off_b)
            buf[off + 2] = np.float32(qty_b)
            if self.active_age_flag:
                buf[off + 3] = np.float32(age_b); off += 4
            else:
                off += 3

            # --- active ask ---
            if active_ask is None:
                m_a = 0.0; off_a = qty_a = age_a = 0.0
            else:
                m_a   = 1.0
                off_a = (float(active_ask.price) - mid) / denom
                qty_a = float(active_ask.qty) * inv_norm
                dt    = max(0.0, time_c - float(active_ask.ts_active))
                age_a = min(1.0, (dt / late) / A_MAX_TICKS)

            buf[off + 0] = np.float32(m_a)
            buf[off + 1] = np.float32(off_a)
            buf[off + 2] = np.float32(qty_a)
            if self.active_age_flag:
                buf[off + 3] = np.float32(age_a); off += 4
            else:
                off += 3

        # ===== PENDING quotes block =====
        if self.pending_quotes_flag:
            # --- pending bid ---
            if pending_bid is None:
                m_b = 0.0; off_b = qty_b = age_b = 0.0
            else:
                m_b   = 1.0
                off_b = (float(pending_bid.price) - mid) / denom
                qty_b = float(pending_bid.qty) * inv_norm
                dt    = max(0.0, time_c - float(pending_bid.ts_sent))
                age_b = min(1.0, (dt / late))  # normalize by one latency round-trip

            buf[off + 0] = np.float32(m_b)
            buf[off + 1] = np.float32(off_b)
            buf[off + 2] = np.float32(qty_b)
            if self.pending_age_flag:
                buf[off + 3] = np.float32(age_b); off += 4
            else:
                off += 3

            # --- pending ask ---
            if pending_ask is None:
                m_a = 0.0; off_a = qty_a = age_a = 0.0
            else:
                m_a   = 1.0
                off_a = (float(pending_ask.price) - mid) / denom
                qty_a = float(pending_ask.qty) * inv_norm
                dt    = max(0.0, time_c - float(pending_ask.ts_sent))
                age_a = min(1.0, (dt / late))

            buf[off + 0] = np.float32(m_a)
            buf[off + 1] = np.float32(off_a)
            buf[off + 2] = np.float32(qty_a)
            if self.pending_age_flag:
                buf[off + 3] = np.float32(age_a); off += 4
            else:
                off += 3

        # ===== inventory =====
        if self.inventory_feature_flag:
            buf[off] = np.float32(self.inventory * inv_norm)
            off += 1

        # Optional safety in debug mode:
        # assert off == self.state_dim, f"state mismatch: off={off}, expected={self.state_dim}"

        return buf



    def is_done(self) -> bool:
        """
        Checks whether the simulation has reached the end of the time series.

        Returns:
            bool: True if no more LOB updates remain; otherwise False.
        """
        return self.ptr >= (self._n-1)  # Simulation ends when pointer exceeds data length


    def _recompute_state_caches(self):
        # Clamp to avoid div-by-zero
        max_inv  =  float(self.max_inventory)
        tick     = float(self.tick_size)
        latency  =  float(self.latency_ns)

        # Cached scalers/caps (names match your code)
        self._inv_norm  = 1.0 / max_inv
        self._denom     = 10.0 * tick
        self._late      = latency
        self._A_max     = float(getattr(self, "_age_cap_ticks", 20.0))
        self._act_cap   = self._A_max * self._late   # active-age cap (ns)
        self._pend_cap  = self._late                 # pending-age cap (ns)

        # Compute state_dim from your flags (base+blocks+inventory)
        def _per_side_dims(include_age: bool):
            # mask + offset + qty + (age?)
            return 3 + (1 if include_age else 0)

        add = 0
        if self.active_quotes_flag:
            add += 2 * _per_side_dims(self.active_age_flag)      # bid+ask
        if self.pending_quotes_flag:
            add += 2 * _per_side_dims(self.pending_age_flag)     # bid+ask
        if self.inventory_feature_flag:
            add += 1

        # self._base_dim must already be set (size of self._feat_mat[idx])
        self.state_dim = int(self._base_dim + add)
        # print(f"Recomputed state_dim = {self.state_dim} (base={self._base_dim}, add={add})")