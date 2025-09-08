class Order:
    """
    Represents a single market order (bid or ask) in the simulation.
    Encapsulates lifecycle: from being sent → becoming active → potentially filled.
    """

    def __init__(self, side, price, qty, ts_sent, delay_ns):
        """
        Initialize the order.

        Args:
            side (str): "bid" or "ask"
            price (float): price level at which the order is quoted
            qty (float): quantity of the order
            ts_sent (int): exchange timestamp when the order was sent
            delay_ns (int): latency in nanoseconds for the order to become active
        """
        self.side = side                  # "bid" or "ask"
        self.price = price                # Quoted price
        self.qty = qty                    # Quantity
        self.ts_sent = ts_sent            # Timestamp when order was sent
        self.delay_ns = delay_ns          # Network or processing delay

        self.ts_active = ts_sent + delay_ns  # Timestamp when order becomes active
        self.active = False               # Will be set to True after ts_active
        self.filled = False              # True when fill occurs

    def activate(self, ts_current):
        """
        Promote order from pending to active, if delay has elapsed.

        Args:
            ts_current (int): Current exchange timestamp
        """
        if ts_current >= self.ts_active:
            self.active = True

    def is_pending(self):
        """
        Returns True if the order is not yet active (still in delay window).
        """
        return not self.active

    def is_active(self):
        """
        Returns True if the order is live and not yet filled.
        """
        return self.active and not self.filled

    def is_filled(self):
        """
        Returns True if the order has already been filled.
        """
        return self.filled

    @property
    def quote(self):
        """
        Returns the (price, quantity) tuple of the order.
        Useful for comparing or hashing quotes.
        """
        return (self.price, self.qty)

    def try_fill(self, best_bid, best_ask):
        """
        Attempts to fill the order based on current best bid and ask levels.

        Fill Conditions:
        - If bid order: fill if price >= best ask (we are aggressive or best)
        - If ask order: fill if price <= best bid

        Returns:
            bool: True if fill occurred, False otherwise
        """
        if not self.is_active():
            return False  # Can't fill until active

        if self.side == "bid" and self.price > best_bid:
            self.filled = True
            return True

        elif self.side == "ask" and self.price < best_ask:
            self.filled = True
            return True

        return False  # No fill occurred
