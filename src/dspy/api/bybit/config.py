from collections import namedtuple

Config = namedtuple('Config', (
    'api_key', 
    'api_secret', 
    'symbol', 
    'coin', 
    'equity',
    'range', 
    'num_orders', 
    'polling_rate', 
    'tp_dist', 
    'stop_dist'))

# Fixed parameters
API_KEY = "StMMBpVlRHbiZF5tki"
API_SECRET = "DOe9UXJgeWLpTKWBibHBpjIDW0UpSON9FuUp"
SYMBOL = 'BTCUSDT'
COIN = 'USDT'

# Parameters to be optimized
EQUITY = 2.0
RANGE = 0.005
NUM_ORDERS = 5
POLLING_RATE = 1
TP_DIST = 0.0005
STOP_DIST = 0 #0.025

config = Config(API_KEY, 
                API_SECRET, 
                SYMBOL, 
                COIN, 
                EQUITY, 
                RANGE, 
                NUM_ORDERS, 
                POLLING_RATE, 
                TP_DIST, 
                STOP_DIST)