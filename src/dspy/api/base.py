"""
A simple interface for interacting with exchanges.
"""
import time
import logging

logger = logging.getLogger('DS.exchanges')

class Exchange:
    """
    Simple interface for interacting with exchanges.
    """
    
    def __init__(self):
        """
        Set up connector.
        """
        pass

    # Market
    def get_mid(self, _symbol: str) -> float:
        """
        Return best mid price.

        Arguments:
            symbol -- the product symbol
        """
        raise NotImplementedError("get_mid not implemented")
    
    def get_ask(self, _symbol: str, _depth: int = 1) -> list[float]:
        """
        Return best ask price and volume.

        Arguments:
            symbol -- the product symbol
        """
        raise NotImplementedError("get_ask not implemented")
    
    def get_bid(self, _symbol: str, _depth: int = 1) -> list[float]:
        """
        Return best bid price and volume.

        Arguments:
            symbol -- the product symbol
        """
        raise NotImplementedError("get_bid not implemented")
    
    def get_orderbook(self, _symbol: str, _depth: int = 25) -> list[float]:
        """
        Return orderbook for product.

        Arguments:
            symbol -- the product symbol
            depth -- the depth of the orderbook
        """
        raise NotImplementedError("get_orderbook not implemented")
    
    def get_trades(self, _symbol: str, _limit: int = 100) -> list[float]:
        """
        Return trades for product.

        Arguments:
            symbol -- the product symbol
            limit -- the number of trades to return
        """
        raise NotImplementedError("get_trades not implemented")
    
    def get_latency(self) -> float:
        """
        Return latency of exchange.
        """
        raise NotImplementedError("get_latency not implemented")
    
    # Account info
    def get_wallet_balance(self) -> float:
        """
        Return wallet balance.
        """
        raise NotImplementedError("get_wallet_balance not implemented")
    
    def get_fees(self, _symbol: str) -> list:
        """
        Return taker and maker fees for product.

        Arguments:
            symbol -- the product symbol
        """
        raise NotImplementedError("get_fees not implemented")

    # Position info
    def get_position(self, _symbol: str) -> dict:
        """
        Return positions in products specified by symbol.

        Arguments:
            symbol -- the product symbol
        """
        raise NotImplementedError("get_position not implemented")
    
    # Trading
    def place_order(self, _symbol: str, _qty: float, _price: float | None = None, _type: str = 'Market') -> dict:
        """
        Place limit order at given price or market order.

        Arguments:
            symbol -- the product symbol
            qty -- the quantity to trade
            price -- the price to trade at
            type -- the type of order to place
        """
        raise NotImplementedError("place_order not implemented")

    def replace_order(self, _symbol: str, _order_id: float, _qty: float, _price: float) -> dict:
        """
        Cancel specified limit order and place a new one.

        Arguments:
            symbol -- the product symbol
            order_id -- the order id to replace
            qty -- the quantity to trade
            price -- the price to trade at
        """
        raise NotImplementedError("replace_order not implemented")
    
    def place_batch_order(self, _symbol: str, _qtys: list, _prices: list) -> dict:
        """
        Place a collection of orders for a given product.

        Arguments:
            symbol -- the product symbol
            qtys -- the quantities to trade
        """
        raise NotImplementedError("place_batch_order not implemented")
    
    def cancel_order(self, symbol: str, order_id: str):
        """
        Cancel specific limit order based on order id.

        Arguments:
            symbol -- the product symbol
            order_id -- the order id to cancel
        """
        raise NotImplementedError("cancel_order not implemented")
    
    def cancel_batch_order(self, _symbol: str, _order_ids: list) -> dict:
        """
        Cancel a list of orders for one product based on order ids.
        
        Arguments:
            symbol -- the product symbol
            order_ids -- the list of order ids to cancel
        """
        raise NotImplementedError("cancel_batch_order not implemented")

    def cancel_all_orders(self, _symbol: str) -> dict:
        """
        Cancel all outstanding orders for a set of products.

        Arguments:
            symbol -- the product symbol
        """
        raise NotImplementedError("cancel_all_orders not implemented")

    def close_positions(self, _symbols: list) -> dict:
        """
        Close positions for a list of products.

        Arguments:
            symbols -- the list of product symbols
        """
        raise NotImplementedError("close_positions not implemented")

    def set_trading_stop(self, _symbol: str, _stop_price: int) -> dict:
        """
        Set stop loss.

        Arguments:
            symbol -- the product symbol
            stop_price -- the stop loss price
        """
        raise NotImplementedError("set_trading_stop not implemented")

    # Various helper and dummy methods
    def wait(self, timeout: float):
        """
        Wait for an indicated amount of time.

        Arguments:
            timeout -- the amount of time to wait
        """
        time.sleep(timeout)

    def next(self):
        """
        For compatibility with simulator.
        """
        return True
