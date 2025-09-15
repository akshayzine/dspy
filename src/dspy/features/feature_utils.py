# feature_utils.py

import polars as pl
from dspy.features.feature_registry import FEATURE_REGISTRY
import numpy as np

def apply_batch_features(df, feature_config, std_flag=False):
    used_columns = []
    ignore_base_list =['mid','vwap','cvwap']
    ignore_columns =[]
    scaling_map = {}  # col_name -> (mean, std)

    for key, config in feature_config.items():
        entry = FEATURE_REGISTRY.get(key)
        if not entry or entry["type"] != "batch":
            continue

        func = entry["func"]

        if isinstance(config, list):
            configs_to_apply = config
        else:
            configs_to_apply = [config]

        for sub_cfg in configs_to_apply:
            # Remove scaling params before calling the function
            func_args = {k: v for k, v in sub_cfg.items() if k not in ("scaling_mean", "scaling_std")}

            df_before = df.columns
            df = func(df, **func_args)
            new_cols = set(df.columns) - set(df_before)
            
            if sub_cfg.get("base_col"):
                if sub_cfg["base_col"] in ignore_base_list:
                    if sub_cfg["base_col"] == "mid":
                        col_to_ignore="mid"
                        # ignore_columns.extend("mid")
                    elif (sub_cfg["base_col"] == "vwap"):
                        col_to_ignore = f'vwap_level{sub_cfg["levels"]}'
                        # ignore_columns.extend(f"vwap_level{sub_cfg["levels"]}")
                    elif (sub_cfg["base_col"] == "cvwap"):
                        col_to_ignore = f'cvwap_level{sub_cfg["levels"]}'
                    # ignore_columns.extend(f"cvwap_level{sub_cfg["levels"]}")
                    new_cols = [col for col in new_cols if col != col_to_ignore]
            new_cols = [col for col in new_cols if df[col].dtype != pl.Datetime]
            used_columns.extend(new_cols)
    
            mean, std = 0, 1
            if std_flag:
                # Only set mean/std if BOTH are present, else default (0,1)
                if "scaling_mean" in sub_cfg and "scaling_std" in sub_cfg:
                    mean = sub_cfg["scaling_mean"]
                    std  = sub_cfg["scaling_std"]
                    

            for col in new_cols:
                if df[col].dtype != pl.Datetime:
                    scaling_map[col] = (mean, std)
    
    # remove datetime columns from used_columns
    used_columns = [col for col in used_columns if df[col].dtype != pl.Datetime]
    used_columns = [col for col in used_columns if col != "mid"]
    
    # deduplicate while preserving order
    used_columns = list(dict.fromkeys(used_columns))
    
    # Standardize features using scaling_map
    if std_flag:
        for col in used_columns:
            if col in scaling_map:
                mean, std = scaling_map[col]
                if std != 0:
                    df = df.with_columns(
                            ((pl.col(col) - mean) / std)
                            .clip(lower_bound=-5, upper_bound=5)  # clip after scaling
                            .alias(col)
                        )

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
            features.append((bid_px - mid) / (5*tick_size) if bid_px != 0.0 else 0.0)
            features.append((ask_px - mid) / (5*tick_size) if ask_px != 0.0 else 0.0)
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