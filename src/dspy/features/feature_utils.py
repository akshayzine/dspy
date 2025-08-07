# feature_utils.py

import polars as pl
from dspy.features.feature_registry import FEATURE_REGISTRY
import numpy as np

# ----------------------------------------------------------------------------
# apply_batch_features:
# Applies all batch-level features to the full Polars DataFrame BEFORE simulation
# ----------------------------------------------------------------------------
def apply_batch_features(df, feature_config: dict) -> tuple[pl.DataFrame, list[str]]:
    """
    Applies all batch features specified in the config and returns:
    - updated DataFrame
    - list of added feature column names
    """
    used_columns = []

    for key, config in feature_config.items():
        entry = FEATURE_REGISTRY.get(key)
        if not entry or entry["type"] != "batch":
            continue

        func = entry["func"]
        # If config is a list of param sets, apply each one
        if isinstance(config, list):
            for sub_cfg in config:
                df_before = df.columns
                df = func(df, **sub_cfg)
                new_cols = set(df.columns) - set(df_before)
                used_columns.extend(new_cols)
        else:
            df_before = df.columns
            df = func(df, **(config or {}))
            new_cols = set(df.columns) - set(df_before)
            used_columns.extend(new_cols)

    return df, used_columns




# ----------------------------------------------------------------------------
# extract_features:
# Called during each simulation step to collect agent inputs
# ----------------------------------------------------------------------------
def extract_features(
    row,
    feature_columns: list[str],
    include_inventory: bool = False,
    inventory: float = 0.0,
    include_active_quotes: bool = False,
    active_quotes: dict = None,
    scale_features: bool = False,
    max_inventory: float = 5.0,
    tick_size: float = 0.1
) -> list[float]:
    """
    Efficiently extracts a flat list of floats from a LOB snapshot row.
    Designed for high-frequency applications.

    Returns:
        list[float]: Feature vector.
    """
    # --- Fast extraction using polars row object ---
    if hasattr(row, "select") and callable(row.select):
        features = list(row.select(feature_columns).row(0))  # Efficient row extraction
    else:
        # fallback for dictionary-like rows
        features = [float(row[col]) for col in feature_columns]

    # --- Calculate midprice ---
    if "mid" in feature_columns:
        mid = row["mid"].item()
    else:
        try:
            mid = (row["asks[0].price"].item() + row["bids[0].price"].item()) / 2
        except KeyError:
            raise KeyError("Midprice not found and best bid/ask not present to compute it.")

    # --- Append active quote features ---
    if include_active_quotes and active_quotes is not None:
        bid_px = active_quotes["bid_px"]
        ask_px = active_quotes["ask_px"]
        bid_qty = active_quotes["bid_qty"]
        ask_qty = active_quotes["ask_qty"]
        if scale_features:
            features.append((bid_px - mid) / tick_size if bid_px != 0.0 else 0.0)
            features.append((ask_px - mid) / tick_size if ask_px != 0.0 else 0.0)
            features.append(bid_qty / max_inventory)
            features.append(ask_qty / max_inventory)
        else:
            features.append(bid_px)
            features.append(ask_px)
            features.append(bid_qty)
            features.append(ask_qty)

    # --- Append inventory ---
    if include_inventory:
        features.append(inventory / max_inventory if scale_features else inventory)

    return features  # Flat list[float] suitable for training


def flatten_features(feature_dict: dict) -> list[float]:
    vec = []
    for val in feature_dict.values():
        if isinstance(val, list):
            vec.extend(val)
        else:
            vec.append(val)
    return vec