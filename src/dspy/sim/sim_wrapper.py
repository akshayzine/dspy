class SimEnv:
    def __init__(self, sim):
        """
        Wraps the MarketSimulator instance to create an RL-compatible interface.

        Args:
            sim (MarketSimulator): The core simulator responsible for LOB updates,
                                   order matching, reward computation, etc.
        """
        self.sim = sim
        self.agent = sim.agent      # Convenience access to agent (not required but helpful)
        self.book = sim.book        # Expose LOB for inspection if needed


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

    def get_state_vector(self):
        """
        Returns the agent’s current state vector (features + inventory).

        Returns:
            list[float]: Flat feature vector at current time step.
        """
        return self.sim.get_state_vector()
    
    def total_pnl(self,mid: float):
        """
        Returns the agent’s current state vector (features + inventory).

        Returns:
            list[float]: Flat feature vector at current time step.
        """
        return self.sim.total_pnl(mid)

    def inject_quotes(self, quotes: dict):
        """
        Push quotes into the simulator for training mode. These quotes
        will be consumed by the next call to sim.step().

        Args:
            quotes (dict): Must include:
                - "bid_px": Bid price
                - "bid_qty": Bid quantity
                - "ask_px": Ask price
                - "ask_qty": Ask quantity
        """
        self.sim.pending_quotes["bid"] = (quotes["bid_px"], quotes["bid_qty"])
        self.sim.pending_quotes["ask"] = (quotes["ask_px"], quotes["ask_qty"])

    def step_with_injected_quotes(self):
        """
        Executes one step of simulation using injected quotes.

        This triggers the full simulation logic (fills, reward update,
        order placement, and advancing the time step) using the bid/ask
        quotes that were set by inject_quotes().
        """
        self.sim.step()  # Now handles both training and eval internally

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
    def ptr(self):
        return self.sim.ptr
    
    @property
    def trade_count(self):
        return self.sim.trade_count