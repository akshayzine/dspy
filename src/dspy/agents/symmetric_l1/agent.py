from typing import Optional
class SymmetricL1Agent:
    def __init__(
        self,
        tick_size: float,
        min_order_size: float = 0.00001,
        max_inventory: float = 0.0001,
        base_quote_size:  Optional[float] = None,
        
    ):
        """
        Symmetric L1 agent that always quotes at best level
=
        Args:
            tick_size (float): Minimum price increment.
            base_quote_size (float): Default quantity to quote when neutral.
            min_order_size (float): Minimum order size allowed.
            max_inventory (float): Inventory constraint for quoting.
        """
        if base_quote_size is None:
            base_quote_size = min_order_size
        self.tick_size = tick_size
        self.base_quote_size = base_quote_size
        self.min_order_size = min_order_size
        self.max_inventory = max_inventory
        self.inventory = 0  # Will be set by the simulator

    def get_quotes_eval(self, state, best_ask,best_bid):
        """
        Generate symmetric bid/ask quotes based on LOB and inventory.

        Args:
            state: Feature vector (not used).
            best_ask 
            best_bid

        Returns:
            dict: Quotes {"bid_px", "bid_qty", "ask_px", "ask_qty"}
        """
        
        bid_px = best_bid
        ask_px = best_ask

        qty = self.base_quote_size
        inv = getattr(self, "inventory", 0)

        # Inventory-aware quantity allocation
        if inv > 0:
            # Long: prefer to sell
            bid_qty = qty
            ask_qty = max(self.min_order_size, min(inv, self.max_inventory))
        elif inv < 0:
            # Short: prefer to buy
            bid_qty = max(self.min_order_size, min(-inv, self.max_inventory))
            ask_qty = qty
        else:
            # Neutral
            bid_qty = ask_qty = qty

        return {
            "bid_px": bid_px,
            "bid_qty": bid_qty,
            "ask_px": ask_px,
            "ask_qty": ask_qty,
        }
