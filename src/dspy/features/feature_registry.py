# feature_registry.py

# feature_registry.py

# Import your feature functions
from dspy.features.book_features import (
    add_mid,
    add_spread,
    add_volume,
    add_vwap_l1,
    add_rel_returns,
    add_log_returns,
    add_lob_price_level,
    add_book_imbalance,
    add_vwap,
    add_cross_vwap,
    add_ret_tick,
    add_realized_vol_time,
    add_realized_vol_tick,
    add_ret_time,
    add_zscore_time,
    add_zscore_tick,
    add_avg_time,
    add_std_time,
    add_avg_tick,
    add_std_tick
)

# FEATURE_REGISTRY maps each feature name to:
# - type: the type of feature ("batch", "row", "external")
# - func: the function that implements it

FEATURE_REGISTRY = {
    "midprice": {
        "type": "batch",
        "func": add_mid,
    },
    "spread": {
        "type": "batch",
        "func": add_spread,
    },
    "volume": {
        "type": "batch",
        "func": add_volume,
    },
    "vwap_l1": {
        "type": "batch",
        "func": add_vwap_l1,
    },
    "rel_returns": {
        "type": "batch",
        "func": add_rel_returns,
    },
    "log_returns": {
        "type": "batch",
        "func": add_log_returns,
    },
    "lob_price_level": {
        "type": "batch",
        "func": add_lob_price_level,
    },
    "book_imbalance": {
        "type": "batch",
        "func": add_book_imbalance,
    },
    "vwap": {
        "type": "batch",
        "func": add_vwap,
    },
    "cross_vwap": {
        "type": "batch",
        "func": add_cross_vwap,
    },
    "ret_tick": {
        "type": "batch",
        "func": add_ret_tick,
    },
    "realized_vol_time": {
        "type": "batch",
        "func": add_realized_vol_time,
    },
    "realized_vol_tick": {
        "type": "batch",
        "func": add_realized_vol_tick,
    },
    "ret_time": {
        "type": "batch",
        "func": add_ret_time,
    },
    "zscore_time": {
        "type": "batch",
        "func": add_zscore_time,
    },
    "zscore_tick": {
        "type": "batch",
        "func": add_zscore_tick,
    },
    "avg_time": {
        "type": "batch",
        "func": add_avg_time,
    },
    "std_time": {
        "type": "batch",
        "func": add_std_time,
    },
    "avg_tick": {
        "type": "batch",
        "func": add_avg_tick,
    },
    "std_tick": {
        "type": "batch",
        "func": add_std_tick,
    },
    "inventory": {
        "type": "external",
        "func": None,
    },
    "active_quotes": {
        "type": "external",
        "func": None,
    }
}


# # Import your feature functions
# from dspy.features.book_features import add_mid, add_lob_price_level         

# # FEATURE_REGISTRY maps each feature name to:
# # - type: the type of feature ("batch", "row", "external")
# # - func: the function that implements it

# FEATURE_REGISTRY = {
#     # ---------------------------
#     # BATCH feature:
#     # Computed once on the entire Polars DataFrame
#     # Output is stored in a new column (e.g., 'mid')
#     # Accessed later via row["mid"]
#     # ---------------------------
#     "midprice": {
#         "type": "batch",
#         "func": add_mid,
#     },

#     # ---------------------------
#     # ROW feature:
#     # Extracted per step from the current LOB snapshot
#     # Input is a dictionary row from df.row(ptr)
#     # Output is a list of floats (e.g., [bid_p0, bid_q0, ..., ask_p0, ask_q0])
#     # ---------------------------
#      "lob_price_level": {
#         "type": "batch",
#         "func": add_lob_price_level
#     },

#     # ---------------------------
#     # EXTERNAL feature:
#     # Comes from simulator state (e.g., inventory)
#     # Passed directly during simulation step
#     # No computation function needed
#     # ---------------------------
#     "inventory": {
#         "type": "external",
#         "func": None,
#     }
# }
