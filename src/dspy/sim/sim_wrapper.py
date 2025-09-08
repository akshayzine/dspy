class SimEnv:
    __slots__ = ("sim", "agent")  # tiny micro-opt; optional

    def __init__(self, sim):
        """
        Wraps the MarketSimulator instance to create an RL-compatible interface.

        Args:
            sim (MarketSimulator): The core simulator responsible for LOB updates,
                                   order matching, reward computation, etc.
        """
        self.sim = sim
        self.agent = sim.agent      # Convenience access to agent (not required but helpful)


# ---------- Episode control ----------
    def reset_state(self):
        """
        Reset simulator to beginning of episode.
        This:
        - Sets LOB pointer to 0
        - Resets inventory and cash
        - Clears evaluation log and reward
        """
        self.sim.reset_state()

    def is_done(self):
        """
        Check if simulator has reached the end of data.

        Returns:
            bool: True if no more LOB data remains, i.e., end of episode.
        """
        return self.sim.is_done()

 # ---------- State access ----------
    def get_state_vector(self,idx: int):
        """
        Returns the agent’s current state vector (features + inventory).

        Returns:
            list[float]: Flat feature vector at current time step.
        """
        return self.sim.get_state_vector(idx)
    
    # Fast LOB helpers (avoid touching Polars; no dicts)
    def best_bid(self, idx: int) -> float:
        return self.sim._best_bid[idx]

    def best_ask(self, idx: int) -> float:
        return self.sim._best_ask[idx]

    def mid(self, idx: int) -> float:
        return (self.sim._best_bid[idx] + self.sim._best_ask[idx])*0.5

    def total_pnl(self, mid: float) -> float:
        return float(self.sim.total_pnl(mid))

    def square_off(self, idx: int):
        # index-based square-off (no Polars row)
        return self.sim.square_off(idx)
    
    def current_time(self, idx: int) -> float:
        return self.sim._ts[idx]
    


    def inject_quotes(self, bid_px: float, bid_qty: float, ask_px: float, ask_qty: float):
        """
        Zero-allocation injector; prefer this in the trainer.
        """
        self.sim.pending_quotes["bid"] = (bid_px, bid_qty)
        self.sim.pending_quotes["ask"] = (ask_px, ask_qty)

    def step_with_injected_quotes(self):
        """
        Executes one step of simulation using injected quotes.

        This triggers the full simulation logic (fills, reward update,
        order placement, and advancing the time step) using the bid/ask
        quotes that were set by inject_quotes().
        """
        self.sim.step()  # Now handles both training and eval internally

    def pre_step(self):
        """
        Executes pre step of simulation 
        """
        self.sim.pre_step()  # Now handles both training and eval internally

    @property
    def reward(self):
        """
        Returns:
            float: Reward assigned after last step (used by trainer).
        """
        return self.sim.reward

    @property
    def inventory(self):
        """
        Returns:
            float: Current inventory held by the agent.
        """
        return self.sim.inventory

    @property
    def cash(self):
        """
        Returns:
            float: Current cash available after trade settlement.
        """
        return self.sim.cash
    
    @property
    def max_drawdown(self):
        """
        Returns:
            float: Current cash available after trade settlement.
        """
        return self.sim.max_drawdown

    @property
    def tpnl(self):
        """
        Returns:
            float: Current cash available after trade settlement.
        """
        return self.sim.tpnl
    
    @property
    def drawdown(self):
        """
        Returns:
            float: Current cash available after trade settlement.
        """
        return self.sim.drawdown

    @property
    def total_cost(self):
        """
        Returns:
            float: Current cash available after trade settlement.
        """
        return self.sim.total_cost
    
    @property
    def ptr(self):
        return self.sim.ptr
    
    @property
    def n_steps(self) -> int:
        # number of ticks available
        return int(self.sim._n)
    
    @property
    def trade_count(self):
        return self.sim.trade_count
    @property
    def zero_reward_count(self):
        return self.sim.zero_reward_count
    @property
    def state_dim(self):
        return len(self.sim._state_buf)