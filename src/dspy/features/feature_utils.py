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
    inventory: float = 0.0
) -> np.ndarray:
    """
    Extracts the agent input vector from a Polars row.

    Parameters:
    - row: current row of the DataFrame (pl.Row)
    - feature_columns: list of feature column names to extract
    - include_inventory: whether to include inventory in input
    - inventory: current inventory value

    Returns:
    - np.ndarray: Input vector for the agent
    """
    # input_vec = [row[col] for col in feature_columns]
    input_vec = [(row[col].item()) for col in feature_columns] #more strick

    # input_vec = [row[col].item() if hasattr(row[col], "item") else row[col] for col in feature_columns]

    if include_inventory:
        input_vec.append(inventory)
    return np.array(input_vec, dtype=np.float32)


def flatten_features(feature_dict: dict) -> list[float]:
    vec = []
    for val in feature_dict.values():
        if isinstance(val, list):
            vec.extend(val)
        else:
            vec.append(val)
    return vec