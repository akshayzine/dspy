class SymmetricL2Agent:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.quote_qty = config.get("min_order_size", 0.01)  # fallback default
        self.max_inventory = config.get("max_inventory", 10)

    def get_quotes(self, state: list[float]) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Inputs:
            state = [mid, lob_bid_2, lob_ask_2, inventory]
        Outputs:
            ((bid_price, bid_qty), (ask_price, ask_qty))
        """
        mid, bid_l2, ask_l2, inventory = state

        # Base quantity to quote
        qty = self.quote_qty

        # Inventory-based scaling
        if inventory > 0:
            # Holding too much → favor selling
            bid_qty = qty
            ask_qty = min(inventory, self.max_inventory)  # sell more
        elif inventory < 0:
            # Short → favor buying
            bid_qty = min(-inventory, self.max_inventory)  # buy more
            ask_qty = qty
        else:
            # Neutral
            bid_qty = ask_qty = qty

        # Final quotes at 2nd L2 levels
        bid_quote = (bid_l2, bid_qty)
        ask_quote = (ask_l2, ask_qty)

        return bid_quote, ask_quote
