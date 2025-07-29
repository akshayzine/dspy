# feature_registry.py

# Import your feature functions
from dspy.features.book_features import add_mid, add_lob_price_level         

# FEATURE_REGISTRY maps each feature name to:
# - type: the type of feature ("batch", "row", "external")
# - func: the function that implements it

FEATURE_REGISTRY = {
    # ---------------------------
    # BATCH feature:
    # Computed once on the entire Polars DataFrame
    # Output is stored in a new column (e.g., 'mid')
    # Accessed later via row["mid"]
    # ---------------------------
    "midprice": {
        "type": "batch",
        "func": add_mid,
    },

    # ---------------------------
    # ROW feature:
    # Extracted per step from the current LOB snapshot
    # Input is a dictionary row from df.row(ptr)
    # Output is a list of floats (e.g., [bid_p0, bid_q0, ..., ask_p0, ask_q0])
    # ---------------------------
     "lob_price_level": {
        "type": "batch",
        "func": add_lob_price_level
    },

    # ---------------------------
    # EXTERNAL feature:
    # Comes from simulator state (e.g., inventory)
    # Passed directly during simulation step
    # No computation function needed
    # ---------------------------
    "inventory": {
        "type": "external",
        "func": None,
    }
}
