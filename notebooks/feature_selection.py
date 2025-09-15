import polars as pl
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import gc

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dspy.hdb import get_dataset
from dspy.features.feature_utils import apply_batch_features
from dspy.utils import add_ts_dt



# ---------- Load run config file ----------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

config_path = project_root / "run/run_config.json"
config = load_config(config_path)

dataset_name     = config["dataset"]
product          = config["product"]
depth            = config["depth"]
latency_ns       = config["latency_micros"] * 1_000
max_inventory    = config["max_inventory"]
inv_penalty      = config["inventory_penalty"]
initial_cash     = config["initial_cash"]
agent_config     = config["agent"]
intervals        = config["intervals"]
min_order_size   = config["min_order_size"]
tick_size        = config["tick_size"]
initial_cash     = config["initial_cash"]
cost_in_bps      = config["cost_in_bps"]
fixed_cost       = config["fixed_cost"]
simulator_mode   = config["simulator_mode"]
system_pc        = config["comp_system"]

loader = get_dataset(dataset_name)
all_books, all_ts = [], []
feature_path = project_root / "run/features_list_selection.json"
feature_config = load_config(feature_path)
inventory_feature_flag = "inventory" in feature_config.keys()


from dspy.features.book_features import add_mid,add_vwap,add_ts_dt


#function to add targets
def add_target_ret_time(
    df: pl.DataFrame,
    delta: int = 50,  # in milliseconds
    base_col: str = "mid",  # 'mid' or 'vwap'
    levels: int = 1,
    depth: int = 1,
    time_col: str = "ts_dt",
    products: list[str] | None = None
) -> pl.DataFrame:

    if time_col not in df.columns:
        df = add_ts_dt(df)

    # Ensure time_col is datetime[ns]
    if df[time_col].dtype != pl.Datetime("ns"):
        df = df.with_columns(pl.col(time_col).cast(pl.Datetime("ns")))

    # Ensure price column exists
    if base_col == "mid":
        col_prefix = "mid"
        if col_prefix not in df.columns:
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if col_prefix not in df.columns:
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    else:
        raise ValueError("base_col must be 'mid' or 'vwap'.")

    

    price_col = col_prefix
    future_df = (
        df.select([time_col, price_col])
        .with_columns(
            (pl.col(time_col) + pl.duration(milliseconds=-delta))
            .cast(pl.Datetime("ns"))
            .alias(time_col)
        )
        .rename({price_col: f"{price_col}_fut"})
    )
    ret_col_name = (
            f"ret_{delta}ms_fut"
        )
    df = df.join_asof(
        future_df,
        left_on=time_col,
        right_on=time_col,
        strategy="backward",
        tolerance=timedelta(milliseconds=1000000)  # generous tolerance
    ).with_columns([
        (pl.col(f"{price_col}_fut")/ (pl.col(price_col) ) - 1).alias(ret_col_name)
    ]).drop([f"{price_col}_fut"])


    return df.drop_nulls()

# --- MAIN EXECUTION LOGIC (REVISED WITH CHUNKING STRATEGY) ---

# 1. Load the single, massive interval as you are doing now
print('Loading the full dataset...')
# ... (your existing code to load the single interval into `df`) ...
for interval in intervals:
        start_str = interval["start"]
        end_str   = interval["end"]
        print('dataframe from:', start_str,'to:',end_str)

        start_ts = datetime.strptime(interval["start"], "%Y-%m-%d %H:%M:%S").strftime("%y%m%d.%H%M%S")
        end_ts   = datetime.strptime(interval["end"],   "%Y-%m-%d %H:%M:%S").strftime("%y%m%d.%H%M%S")

        df = loader.load_book(
            product=product,
            times=[start_ts, end_ts],
            depth=depth,
            type="book_snapshot_25",
            lazy=False
        )
print('Full dataset loaded. Shape:', df.shape)
print( ' Estimated size of df in MB: ', df.estimated_size("mb"))

# 2. Define the number of chunks and the directory for processed files
num_chunks = 1
chunk_size = len(df) // num_chunks # Use integer division
processed_data_dir = Path(__file__).parent/ "chunk_data"
print(processed_data_dir)
processed_data_dir.mkdir(parents=True, exist_ok=True)
print(f"Splitting the DataFrame into {num_chunks} chunks of approximately {chunk_size} rows each.")

# 3. Loop through each chunk, process it, and save it to disk
feature_cols = []
target_list = []
for i in range(1):
    print(f"--- Processing chunk {i+1}/{num_chunks} ---")
    
    # Calculate the start offset for the slice
    offset = i * chunk_size
    
    # Get the current chunk using a simple slice
    df_chunk = df.slice(offset, chunk_size)

    # Apply feature engineering to the chunk
    print(f"  - Applying batch features...")
    df_processed_chunk, feature_cols = apply_batch_features(df_chunk, feature_config)
    feature_cols = [col for col in feature_cols if df_processed_chunk[col].dtype != pl.Datetime]
    
    # Add target variables to the chunk
    print(f"  - Adding targets...")
    time_horizons = [100,200, 500,1000, 5000]
    price = 'mid'
    for t in time_horizons:
        if i==0:
            target_list.append(f"ret_{t}ms_fut")
        df_processed_chunk = add_target_ret_time(df_processed_chunk, t, price)

    # Save the processed chunk to disk
    output_path = processed_data_dir / f"processed_chunk_{i}.parquet"
    print(f"  - Saving processed chunk to {output_path}")
    df_processed_chunk.write_parquet(output_path)

# 4. Release memory of the huge raw DataFrame
print("\nReleasing memory of the original raw DataFrame...")
del df
gc.collect()


# 5. Load all the processed chunks from disk
print("--- Loading all processed chunks from disk ---")
all_processed_files = list(processed_data_dir.glob("*.parquet"))
# df = pl.read_parquet(all_processed_files)
df = pl.scan_parquet(all_processed_files) 


# 6. Now run your feature selection as before
print('\n--- Starting Feature Selection ---')

def correlation_xgb_feature_selection_lazy(
    df_lazy: pl.LazyFrame,
    features: list,
    targets: list,
    num_feat: list,
    corr_threshold: float = 0.80,
    n_splits: int = 5,
    min_fold_count: int = 3
):
    """
    Performs feature selection on a LazyFrame using robust, memory-safe
    methods by sampling the data for all memory-intensive steps.
    """
    # Memory-safe all-zero column check (this part is correct)
    print("Checking for all-zero columns (one by one to save memory)...")
    cols_to_drop = ['mid']
    for feature in features:
        is_all_zero = df_lazy.select(
            (pl.col(feature) == 0).all()
        ).collect().item()
        if is_all_zero:
            cols_to_drop.append(feature)
            
    df_lazy = df_lazy.drop(cols_to_drop)
    reduced_features_0 = [f for f in features if f not in cols_to_drop]
    print(f"Dropped {len(cols_to_drop)} all-zero columns.")

    # --- Step 1: Create a large but manageable sample for ALL subsequent steps ---
    print("Creating a 2M row random sample to use for all calculations...")
    
    # We will perform all calculations on this manageable, in-memory sample.
    # Lazily slice a subset larger than our sample, then collect and sample it.
    df_sample = df_lazy.slice(0, 5_000_000).collect() \
                       .sample(n=2_000_000, shuffle=True, seed=42)

    # --- Step 2: Correlation filtering on the sample ---
    print("Calculating correlation matrix on the sample...")
    corr_matrix = df_sample.select(reduced_features_0) \
                           .to_pandas() \
                           .corr(method='spearman').abs()

    # --- Step 2a: Print feature–target correlations ---
    print("\nCalculating correlations of all features with each target...")
    # Only need features + targets subset from the sample
    feat_target_corr = (
        df_sample.select(reduced_features_0 + targets)
                 .to_pandas()
                 .corr(method='spearman')
    )
    
    for target in targets:
         
        corr_series = feat_target_corr[target].drop(target)  # remove self-correlation
        # Sort by absolute correlation, descending
        corr_series = corr_series.reindex(reduced_features_0).abs().sort_values(ascending=False).head(10)

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    reduced_features = [f for f in reduced_features_0 if f not in to_drop_corr]
    print(f'Dropped {len(to_drop_corr)} features via correlation.')
    
    # --- Step 3: XGBoost with stability selection ON THE SAMPLE ---
    all_importances = []
    stable_features_all_targets = set()

    for target, top_n in zip(targets, num_feat):
        print(f"Processing Target: {target}")

        # Prepare X and y directly from the in-memory sample DataFrame
        X = df_sample.select(reduced_features).to_pandas()
        y_raw = df_sample.get_column(target)

        # Clip the target variable
        lower_bound = y_raw.quantile(0.005)
        upper_bound = y_raw.quantile(0.995)
        y = y_raw.clip(lower_bound=lower_bound, upper_bound=upper_bound).to_pandas()

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_importances = []
        feature_counts = {f: 0 for f in reduced_features}

        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            model = XGBRegressor(
                objective='reg:pseudohubererror',
                n_estimators=500, max_depth=5, learning_rate=0.05,
                random_state=42, n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            importances = model.feature_importances_
            top_features_fold = [f for f, _ in sorted(zip(reduced_features, importances), key=lambda x: x[1], reverse=True)[:top_n]]
            for f in top_features_fold:
                feature_counts[f] += 1
            fold_importances.append(importances)
            
        mfc = min_fold_count
        if target == "ret_100ms_fut":
            mfc +=1
        stable_features = [f for f, count in feature_counts.items() if count >= mfc]
        stable_features_all_targets.update(stable_features)
        avg_importances = np.mean(fold_importances, axis=0)
        imp_df = pd.DataFrame({"feature": reduced_features, "importance": avg_importances, "target": target}).sort_values("importance", ascending=False)
        all_importances.append(imp_df)

    # --- Final combination step ---
    all_importances_df = pd.concat(all_importances)
    avg_importance = (
        all_importances_df.groupby("feature")["importance"]
        .mean().sort_values(ascending=False).reset_index()
    )
    final_features = [f for f in avg_importance["feature"].tolist() if f in stable_features_all_targets]
    
    return final_features, avg_importance






feature_length=[12,10,8,8,6]
price ='mid'

print('selection features')
final_feats, importance_df = correlation_xgb_feature_selection_lazy(
    df_lazy=df,
    features=feature_cols,
    targets=target_list,
    num_feat=feature_length,
    corr_threshold=0.9,
)


# --- Compute mean and std for final selected features ---
print("\n--- Computing mean/std for final selected features ---")

# Collect only needed columns from df_lazy to save memory
df_selected = df.select(final_feats).collect()

# Clip each column to its 0.01–0.99 quantiles
clipped_cols = []
# Compute per-column clipped mean/std independently
for col in final_feats:
    q_low = df_selected[col].quantile(0.005)
    q_high = df_selected[col].quantile(0.995)

    mean_val = (
        df_selected
        .select(
            pl.when(pl.col(col) < q_low).then(q_low)
            .when(pl.col(col) > q_high).then(q_high)
            .otherwise(pl.col(col))
        )
        .mean()
        .item()
    )

    std_val = (
        df_selected
        .select(
            pl.when(pl.col(col) < q_low).then(q_low)
            .when(pl.col(col) > q_high).then(q_high)
            .otherwise(pl.col(col))
        )
        .std()
        .item()
    )

    print(f"{col}: mean={mean_val}, std={std_val}")
    




