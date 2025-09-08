import numpy as np
import torch


class DQNAgent:
    def __init__(
        self,
        model,
        tick_size: float,
        min_order_size: float,
        max_inventory: float,
        device: torch.device | str = "cpu",
        base_quote_size: float = 0.1,
        epsilon: float = 0.1,
        max_quote_level: int =4,
        action_set: list[list[int]] = None
    ):
        """
        DQN Agent with discrete action space and inventory-aware quoting.

        Args:
            model: Trained Q-network
            tick_size: Exchange-defined minimum price increment
            min_order_size: Minimum quote quantity allowed by exchange
            max_inventory: Inventory limit used for quoting size
            base_quote_size: Default quote size when neutral
            epsilon: Epsilon for ε-greedy exploration in training
            device: "cpu" or "cuda"
            action_set: List of [ask_offset, bid_offset] actions
                        Default is 16-point grid: [3,3] to [0,0]
        """
        # Default: 4x4 grid of actions [ask_offset, bid_offset]
        if action_set is None:
            action_set = [[i, j] for i in reversed(range(max_quote_level)) for j in reversed(range(max_quote_level))]

        self.model = model
        self.action_set = action_set
        self.tick_size = tick_size
        self.min_order_size = min_order_size
        self.max_inventory = max_inventory
        self.base_quote_size = base_quote_size
        self.epsilon = epsilon
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # Will be updated each step
        self.action_idx = 0
        self.inventory = 0  # Must be injected from simulator before each step


    

    def act(self, state: list[float], explore=True) -> int:
        """
        Chooses an action index using ε-greedy policy.

        Args:
            state: Current state vector
            explore: Whether to use ε-greedy (True during training)

        Returns:
            action_idx (int): Index into self.action_set
        """
        if explore and np.random.rand() < self.epsilon:
            self.action_idx = np.random.randint(len(self.action_set))  # Random action
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state) if isinstance(state, np.ndarray) else torch.tensor(state, dtype=torch.float32)
                q_values = self.model(state_tensor.unsqueeze(0).to(self.device))  # Shape: [1, num_actions]
                self.action_idx = int(q_values.argmax(dim=1))

        return self.action_idx

    def set_action_idx(self, idx: int):
        """
        Allows external trainer to set the action index manually.
        (Used in DQN training loop.)
        """
        self.action_idx = idx

    def get_quotes(self, state: list[float], best_ask: float, best_bid: float) -> dict:
        """
        Generates final quote prices and sizes based on:
          - Discrete action
          - LOB snapshot (best ask/bid)
          - Current inventory (adjusts bid/ask size)

        Args:
            state: Feature vector (used for model input)
            best_ask
            best_bid

        Returns:
            dict: {
                "bid_px", "ask_px",    → price levels
                "bid_qty", "ask_qty"   → size per side
            }
        """
        ask_offset, bid_offset = self.action_set[self.action_idx]
        tick_s = self.tick_size
        inv = getattr(self, "inventory", 0)  # Get current inventory from sim

        # Action is defined in LOB ticks from best ask/bid
        bid_px = best_bid - (bid_offset - 1) * tick_s
        bid_px = round(min(bid_px, best_ask - tick_s),1)  # Ensure bid < ask
        

        ask_px = best_ask + (ask_offset - 1) * tick_s
        ask_px =round(max(ask_px, best_bid + tick_s),1)  # Ensure ask > bid
        # Ensure bid < ask    (necessary for spread of 0.2 spread)        
        if ask_px <= bid_px:
            if inv>=0:
                ask_px = round(ask_px + tick_s,1)
            else:
                bid_px = round(bid_px - tick_s,1)
                


        qty = self.min_order_size
        

        # Inventory-based quantity logic
        if inv > 0:
            # Long: try to sell more
            bid_qty = qty
            ask_qty = max(self.min_order_size, min(inv, self.max_inventory))
        elif inv < 0:
            # Short: try to buy more
            bid_qty = max(self.min_order_size, min(-inv, self.max_inventory))
            ask_qty = qty
        else:
            # Neutral
            bid_qty = ask_qty = qty

        return {
            "bid_px": bid_px,
            "ask_px": ask_px,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty
        }

    def set_quotes(self, quotes: dict):
        """
        Optionally store quotes for logging or evaluation.
        """
        self.last_quotes = quotes

    def get_quotes_eval(self, state: list[float], best_ask: float, best_bid: float) -> dict:
        """
        Evaluation-time quote generator:
        - Uses greedy action (no exploration)
        - Returns bid/ask price and quantity based on inventory

        Args:
            state: feature vector
            best_ask
            best_bid

        Returns:
            dict: {bid_px, ask_px, bid_qty, ask_qty}
        """
        self.act(state, explore=False)
        return self.get_quotes(state, best_ask,best_bid)
