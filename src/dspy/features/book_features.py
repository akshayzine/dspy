"""
Module for generating features from book data.

This module provides functions to add spread, volume, and other features to order book data.
"""

import polars as pl
from datetime import timedelta
from dspy.features.utils import get_products
from dspy.utils import add_ts_dt
# Features for prices   

def add_mid(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['asks[0].price', 'bids[0].price']) -> pl.DataFrame:
    """
    Add a mid column to the DataFrame.
    """
    if products is None:
        products = get_products(df, cols)
        
    if products == []:
        return df.with_columns(
            ((pl.col(f"{cols[0]}") + pl.col(f"{cols[1]}"))/2).alias('mid'))
    for product in products:
        df = df.with_columns(
            ((pl.col(f"{cols[0]}_{product}") + pl.col(f"{cols[1]}_{product}"))/2).alias(f'mid_{product}'))
    return df

def add_spread(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['asks[0].price', 'bids[0].price']) -> pl.DataFrame:
    """
    Add a spread column to the DataFrame.
    """
    if products is None:
        products = get_products(df, cols)

    if products == []:
        return df.with_columns(
            (pl.col(f"{cols[0]}") - pl.col(f"{cols[1]}")).alias('spread'))
    for product in products:
        df = df.with_columns(
            (pl.col(f"{cols[0]}_{product}") - pl.col(f"{cols[1]}_{product}")).alias(f'spread_{product}'))
    return df

def add_volume(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['vol_s0', 'vol_s1']) -> pl.DataFrame:
    """
    Add a volume column to the DataFrame.
    """
    if products is None:
        products = get_products(df, cols)

    if products == []:
        return df.with_columns(
            (pl.col(f"{cols[0]}") + pl.col(f"{cols[1]}")).alias('volume'))
    for product in products:
        df = df.with_columns(
            (pl.col(f"{cols[0]}_{product}") + pl.col(f"{cols[1]}_{product}")).alias(f'volume_{product}'))
    return df

def add_vwap_l1(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['prc_s0', 'prc_s1', 'vol_s0', 'vol_s1']) -> pl.DataFrame:
    """
    Add a VWAP column to the DataFrame.
    """
    if products is None:
        products = get_products(df, cols)

    if products == []:
        df = df.with_columns(
            pl.when(pl.col(f"{cols[2]}") + pl.col(f"{cols[3]}") > 0)
            .then(
                (
                    (pl.col(f"{cols[0]}") * pl.col(f"{cols[2]}") + 
                        pl.col(f"{cols[1]}") * pl.col(f"{cols[3]}")) /
                    (pl.col(f"{cols[2]}") + pl.col(f"{cols[3]}"))
                )
            )
            .otherwise(pl.lit(0))
            .alias('vwap')
        )
    else:
        for product in products:
            df = df.with_columns(
                pl.when(pl.col(f"{cols[2]}_{product}") + pl.col(f"{cols[3]}_{product}") > 0)
                .then(
                    (
                        (pl.col(f"{cols[0]}_{product}") * pl.col(f"{cols[2]}_{product}") + 
                            pl.col(f"{cols[1]}_{product}") * pl.col(f"{cols[3]}_{product}")) /
                        (pl.col(f"{cols[2]}_{product}") + pl.col(f"{cols[3]}_{product}"))
                    )
                )
                .otherwise(pl.lit(0))
                .alias(f'vwap_{product}')
            )
    return df

def add_rel_returns(df: pl.DataFrame, cols: list[str]=['mid'], products: list[str] | None = None) -> pl.DataFrame:
    """
    Add a relative return column to the DataFrame.
    """
    print("Adding relative returns...")
    if products is None:
        products = get_products(df, cols)
    if isinstance(cols, str):
        cols = [cols]
    print("Products:", products)
    if products == []:
        df = df.with_columns(
            # pl.col(f"{cols[0]}").pct_change().alias("rel_return")
            (pl.col(f"{cols[0]}") / pl.col(f"{cols[0]}").shift(1) - 1).alias(f"rel_return_{cols[0]}")
        )
    else:
        df = df.with_columns(
            [
                #pl.col(f"{col}_{product}").pct_change().alias(f"rel_return_{col}_{product}")
                (pl.col(f"{col}_{product}") / pl.col(f"{col}_{product}").shift(1) - 1).alias(f"rel_return_{col}_{product}")
                for col in cols
                for product in products
            ]
        )
    return df.drop_nulls()
    
def add_log_returns(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['mid']) -> pl.DataFrame:
    """
    Add a log return column to the DataFrame.
    """
    if products is None:
        products = get_products(df, cols)
    if isinstance(cols, str):
        cols = [cols]

    if products == []:
        df = df.with_columns(
            pl.col(f"{cols[0]}").log().diff().alias("log_return")
        )
    else:
        df = df.with_columns(
            [
                pl.col(f"{col}_{product}").log().diff().alias(f"log_return_{col}_{product}")
                for col in cols
                for product in products
            ]
        )
    return df.drop_nulls()


def add_lob_price_level(
    df: pl.DataFrame,
    side: str = "bid",
    level: int = 0,
    products: list[str] | None = None,
) -> pl.DataFrame:
    """
    Adds a column for the LOB price at a given side and level.
    """
    col_name = f"{'bids' if side == 'bid' else 'asks'}[{level-1}].price"
    alias = f"lob_price_level_{side}_{level}"

    if products is None:
        products = get_products(df,col_name)


    if products == []:
        return df.with_columns(
            pl.col(col_name).alias(alias)
        )

    for product in products:
        df = df.with_columns(
            pl.col(f"{col_name}_{product}").alias(f"{alias}_{product}")
        )

    return df

def add_price_snapshot(
    df: pl.DataFrame,
    levels: int,
    depth: int,
    products: list[str] | None = None
) -> pl.DataFrame:
    """
    Adds price snapshot columns (bid/ask prices) for each level up to the given depth.

    Parameters:
        df (pl.DataFrame): Input LOB DataFrame.
        levels (int): Number of top levels to extract.
        depth (int): Total book depth available.
        products (list[str] | None): Optional product suffixes.

    Returns:
        pl.DataFrame: DataFrame with additional price columns.

    Raises:
        ValueError: If requested level exceeds depth.
    """
    if levels > depth:
        raise ValueError(f"Requested level {levels} exceeds book depth {depth}")

    if products is None:
        products = get_products(df, [f"asks[0].price", f"bids[0].price"])

    new_cols = []

    if products == []:
        for i in range(levels):
            new_cols.append(pl.col(f"asks[{i}].price").alias(f"ask_price_{i}"))
            new_cols.append(pl.col(f"bids[{i}].price").alias(f"bid_price_{i}"))
    else:
        for product in products:
            for i in range(levels):
                new_cols.append(pl.col(f"asks[{i}].price_{product}").alias(f"ask_price_{i}_{product}"))
                new_cols.append(pl.col(f"bids[{i}].price_{product}").alias(f"bid_price_{i}_{product}"))

    return df.with_columns(new_cols)


def add_book_imbalance(
    df: pl.DataFrame,
    levels: int,
    depth: int,
    products: list[str] | None = None
) -> pl.DataFrame:
    """
    Adds LOB imbalance over the top `levels` using bids[i].amount and asks[i].amount columns.

    If multiple products are present, computes imbalance per product using suffixes.
    Temporary dummy volume columns are dropped after use.

    Parameters:
        df (pl.DataFrame): Input DataFrame with bids/asks[i].amount.
        levels (int): Number of levels to include in imbalance.
        depth (int): Max depth of book data.
        products (list[str] | None): Product suffixes (e.g., ['btc', 'eth']).

    Returns:
        pl.DataFrame: Updated DataFrame with imbalance columns added.
    """
    if levels > depth:
        raise ValueError(f"Requested level {levels} exceeds book depth {depth}")

    # Infer products if not provided
    if products is None:
        products = get_products(df, ["bids[0].amount", "asks[0].amount"])

    if products == []:
        bid_cols = [f"amount_b_l{i}" for i in range(levels)]
        ask_cols = [f"amount_s_l{i}" for i in range(levels)]

        # Create dummy cols
        df = df.with_columns([
            *(pl.col(f"bids[{i}].amount").alias(f"amount_b_l{i}") for i in range(levels)),
            *(pl.col(f"asks[{i}].amount").alias(f"amount_s_l{i}") for i in range(levels)),
        ])

        # Compute imbalance
        imbalance_expr = (
            (pl.sum_horizontal([pl.col(c) for c in bid_cols]) -
             pl.sum_horizontal([pl.col(c) for c in ask_cols])) /
            (pl.sum_horizontal([pl.col(c) for c in bid_cols]) +
             pl.sum_horizontal([pl.col(c) for c in ask_cols]))
        ).alias(f"book_imbalance_{levels}")

        return df.with_columns([imbalance_expr]).drop(bid_cols + ask_cols)

    for product in products:
        bid_cols = [f"amount_b_l{i}_{product}" for i in range(levels)]
        ask_cols = [f"amount_s_l{i}_{product}" for i in range(levels)]

        # Create dummy cols per product
        df = df.with_columns([
            *(pl.col(f"bids[{i}].amount_{product}").alias(f"amount_b_l{i}_{product}") for i in range(levels)),
            *(pl.col(f"asks[{i}].amount_{product}").alias(f"amount_s_l{i}_{product}") for i in range(levels)),
        ])

        imbalance_expr = (
            (pl.sum_horizontal([pl.col(c) for c in bid_cols]) -
             pl.sum_horizontal([pl.col(c) for c in ask_cols])) /
            (pl.sum_horizontal([pl.col(c) for c in bid_cols]) +
             pl.sum_horizontal([pl.col(c) for c in ask_cols]))
        ).alias(f"book_imbalance_{levels}_{product}")

        df = df.with_columns([imbalance_expr]).drop(bid_cols + ask_cols)

    return df



def add_vwap(
    df: pl.DataFrame,
    levels: int,
    depth: int,
    products: list[str] | None = None
) -> pl.DataFrame:
    """
    Adds combined VWAP (bid+ask) up to a given number of levels for each product.

    Uses:
        VWAP = sum(price * volume) / sum(volume)
        across bids[i] and asks[i]

    Supports multi-product columns like bids[0].price_btc.

    Parameters:
        df (pl.DataFrame): Input LOB DataFrame.
        levels (int): Number of LOB levels to include.
        depth (int): Maximum depth of LOB snapshot.
        products (list[str] | None): List of product suffixes (e.g., ['btc']).

    Returns:
        pl.DataFrame: Updated with vwap_level{levels} or vwap_level{levels}_{product}.
    """
    if levels > depth:
        raise ValueError(f"Requested level {levels} exceeds book depth {depth}")

    if products is None:
        products = get_products(df, ["bids[0].price", "asks[0].price"])

    if products == []:
        # No product suffix
        bid_pv = [pl.col(f"bids[{i}].price") * pl.col(f"bids[{i}].amount") for i in range(levels)]
        ask_pv = [pl.col(f"asks[{i}].price") * pl.col(f"asks[{i}].amount") for i in range(levels)]
        bid_amt = [pl.col(f"bids[{i}].amount") for i in range(levels)]
        ask_amt = [pl.col(f"asks[{i}].amount") for i in range(levels)]

        numerator = pl.sum_horizontal(bid_pv + ask_pv)
        denominator = pl.sum_horizontal(bid_amt + ask_amt)

        vwap_expr = (numerator / denominator).alias(f"vwap_level{levels}")
        return df.with_columns(vwap_expr)

    # Multi-product case
    for product in products:
        bid_pv = [pl.col(f"bids[{i}].price_{product}") * pl.col(f"bids[{i}].amount_{product}") for i in range(levels)]
        ask_pv = [pl.col(f"asks[{i}].price_{product}") * pl.col(f"asks[{i}].amount_{product}") for i in range(levels)]
        bid_amt = [pl.col(f"bids[{i}].amount_{product}") for i in range(levels)]
        ask_amt = [pl.col(f"asks[{i}].amount_{product}") for i in range(levels)]

        numerator = pl.sum_horizontal(bid_pv + ask_pv)
        denominator = pl.sum_horizontal(bid_amt + ask_amt)

        vwap_expr = (numerator / denominator).alias(f"vwap_level{levels}_{product}")
        df = df.with_columns(vwap_expr)

    return df


def add_cross_vwap(
    df: pl.DataFrame,
    levels: int,
    depth: int,
    products: list[str] | None = None
) -> pl.DataFrame:
    """
    Adds a normalized cross VWAP column to the DataFrame.

    Formula (for each level i):
        numerator = bid[i].price * ask[i].amount + ask[i].price * bid[i].amount
        denominator = bid[i].amount + ask[i].amount

    Final:
        cross_vwap = sum(numerators) / sum(denominators)

    Parameters:
        df (pl.DataFrame): LOB snapshot DataFrame.
        levels (int): Number of levels to use.
        depth (int): Max depth of book available.
        products (list[str] | None): Optional product suffixes.

    Returns:
        pl.DataFrame: Updated with 'cross_vwap_level{levels}' column(s).
    """
    if levels > depth:
        raise ValueError(f"Requested level {levels} exceeds book depth {depth}")

    if products is None:
        products = get_products(df, [])

    if products == []:
        numerator_exprs = []
        denominator_exprs = []

        for i in range(levels):
            numerator_exprs.append(pl.col(f"bids[{i}].price") * pl.col(f"asks[{i}].amount"))
            numerator_exprs.append(pl.col(f"asks[{i}].price") * pl.col(f"bids[{i}].amount"))
            denominator_exprs.append(pl.col(f"bids[{i}].amount") + pl.col(f"asks[{i}].amount"))

        cvwap_expr = (
            (pl.sum_horizontal(numerator_exprs) / pl.sum_horizontal(denominator_exprs))
            .alias(f"cross_vwap_level{levels}")
        )

        df = df.with_columns(cvwap_expr)
        return df

    # Product-wise computation
    for product in products:
        numerator_exprs = []
        denominator_exprs = []

        for i in range(levels):
            numerator_exprs.append(pl.col(f"bids[{i}].price_{product}") * pl.col(f"asks[{i}].amount_{product}"))
            numerator_exprs.append(pl.col(f"asks[{i}].price_{product}") * pl.col(f"bids[{i}].amount_{product}"))
            denominator_exprs.append(pl.col(f"bids[{i}].amount_{product}") + pl.col(f"asks[{i}].amount_{product}"))

        cvwap_expr = (
            (pl.sum_horizontal(numerator_exprs) / pl.sum_horizontal(denominator_exprs))
            .alias(f"cross_vwap_level{levels}_{product}")
        )

        df = df.with_columns(cvwap_expr)

    return df



def add_ret_tick(
    df: pl.DataFrame,
    base_col: str = "mid",           # 'mid' or 'vwap'
    ticks: int = 1,
    levels: int = 1,
    depth: int = 1,
    products: list[str] | None = None
) -> pl.DataFrame:
    """
    Adds tick-based return of mid or VWAP price over given lag in ticks.

    Computes: (x_t - x_{t - ticks}) / x_t
    where x is mid or vwap_level{levels}.

    Parameters:
        df (pl.DataFrame): Input DataFrame.
        base_col (str): 'mid' or 'vwap'.
        ticks (int): Lag in rows (ticks).
        levels (int): VWAP levels (ignored for 'mid').
        depth (int): Book depth (used if VWAP needs to be created).
        products (list[str] | None): Optional product suffixes.

    Returns:
        pl.DataFrame: Updated DataFrame with return columns, nulls dropped.
    """
    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    else:
        raise ValueError("base_col must be 'mid' or 'vwap'.")

    if products is None:
        products = get_products(df, [col_prefix])

    if products == []:
        ret_col = (
            f"{col_prefix}_ret_t{ticks}" if base_col == "mid"
            else f"vwap_ret_t{ticks}_l{levels}"
        )
        df = df.with_columns(
            ((pl.col(col_prefix) - pl.col(col_prefix).shift(ticks)) / pl.col(col_prefix)).alias(ret_col)
        )
    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            ret_col = (
                f"{col_prefix}_ret_t{ticks}_{product}" if base_col == "mid"
                else f"vwap_ret_t{ticks}_l{levels}_{product}"
            )
            df = df.with_columns(
                ((pl.col(col) - pl.col(col).shift(ticks)) / pl.col(col)).alias(ret_col)
            )

    return df.drop_nulls()


def add_realized_vol_time(
    df: pl.DataFrame,
    window: int = 100,               # milliseconds
    base_col: str = "mid",           # 'mid' or 'vwap'
    levels: int = 1,
    depth: int = 1,
    time_col: str = "ts_dt",
    products: list[str] | None = None
) -> pl.DataFrame:
    if time_col not in df.columns:
        df = add_ts_dt(df)

    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    else:
        raise ValueError("base_col must be 'mid' or 'vwap'.")

    if products is None:
        products = get_products(df, [col_prefix])

    window_str = f"{window}ms"

    if products == []:
        ret_col = f"__ret_tmp_{col_prefix}"
        vol_col = f"realized_vol_{window}ms_{col_prefix}"

        df = df.with_columns(
            ((pl.col(col_prefix) / pl.col(col_prefix).shift(1))-1).alias(ret_col)
        )

        rolling_df = (
            df.rolling(index_column=time_col, period=window_str, closed="right")
            .agg([pl.col(ret_col).std().alias(vol_col)])
        )

        df = rolling_df.join(df.drop(ret_col), on=time_col, how="right")
        # df = df.select([col for col in df.columns if col != vol_col] + [vol_col])

    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            ret_col = f"__ret_tmp_{col}"
            vol_col = f"realized_vol_{window}ms_{col}"

            df = df.with_columns(
                ((pl.col(col) / pl.col(col).shift(1))-1).alias(ret_col)
            )

            rolling_df = (
                df.rolling(index_column=time_col, period=window_str, closed="right")
                .agg([pl.col(ret_col).std().alias(vol_col)])
            )

            df = rolling_df.join(df.drop(ret_col), on=time_col, how="right")
            # df = df.select([col for col in df.columns if col != vol_col] + [vol_col])

    return df.drop_nulls()

def add_realized_vol_tick(
    df: pl.DataFrame,
    ticks: int = 50,
    base_col: str = "mid",           # 'mid' or 'vwap'
    levels: int = 1,
    depth: int = 1,
    products: list[str] | None = None
) -> pl.DataFrame:
    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    else:
        raise ValueError("base_col must be 'mid' or 'vwap'.")

    if products is None:
        products = get_products(df, [col_prefix])

    if products == []:
        ret_col = f"__ret_tmp_{col_prefix}"
        vol_col = f"realized_vol_{ticks}_ticks_{col_prefix}"

        df = df.with_columns(
            (pl.col(col_prefix) / pl.col(col_prefix).shift(1) - 1).alias(ret_col)
        )
        df = df.with_columns(
            pl.col(ret_col).rolling_std(window_size=ticks).alias(vol_col)
        )
        df = df.drop([ret_col])

    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            ret_col = f"__ret_tmp_{col}"
            vol_col = f"realized_vol_{ticks}_ticks_{col}"

            df = df.with_columns(
                (pl.col(col) / pl.col(col).shift(1) - 1).alias(ret_col)
            )
            df = df.with_columns(
                pl.col(ret_col).rolling_std(window_size=ticks).alias(vol_col)
            )
            df = df.drop([ret_col])

    return df.drop_nulls()



#new
def add_ret_time(
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
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    else:
        raise ValueError("base_col must be 'mid' or 'vwap'.")

    if products is None:
        products = get_products(df, [col_prefix])

    if products == []:
        price_col = col_prefix
        lagged_df = (
            df.select([time_col, price_col])
            .with_columns(
                (pl.col(time_col) + pl.duration(milliseconds=delta))
                .cast(pl.Datetime("ns"))
                .alias(time_col)
            )
            .rename({price_col: f"{price_col}_past"})
        )
        ret_col_name = (
                f"ret_{delta}ms_{col_prefix}"
            )
        df = df.join_asof(
            lagged_df,
            left_on=time_col,
            right_on=time_col,
            strategy="backward",
            tolerance=timedelta(milliseconds=1000000)  # generous tolerance
        ).with_columns([
            ((pl.col(price_col) / pl.col(f"{price_col}_past")) - 1).alias(ret_col_name)
        ]).drop([f"{price_col}_past"])

    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            lagged_df = (
                df.select([time_col, col])
                .with_columns(
                    (pl.col(time_col) + pl.duration(milliseconds=delta))
                    .cast(pl.Datetime("ns"))
                    .alias(time_col)
                )
                .rename({col: f"{col}_past"})
            )

            ret_col_name = (
                f"ret_{delta}ms_{col_prefix}_{product}"
            )

            df = df.join_asof(
                lagged_df,
                left_on=time_col,
                right_on=time_col,
                strategy="backward",
                tolerance=timedelta(milliseconds=1000000)
            ).with_columns([
                ((pl.col(col) / pl.col(f"{col}_past")) - 1).alias(ret_col_name)
            ]).drop([f"{col}_past"])

    return df.drop_nulls()

def add_zscore_time(
    df: pl.DataFrame,
    window_ms: int = 500,
    base_col: str = "mid",            # 'mid' or 'vwap'
    levels: int = 1,
    depth: int = 5,
    time_col: str = "ts_dt",
    products: list[str] | None = None
) -> pl.DataFrame:
    if time_col not in df.columns:
        df = add_ts_dt(df)

    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    else:
        raise ValueError("base_col must be 'mid' or 'vwap'.")

    if products is None:
        products = get_products(df, [col_prefix])

    window_str = f"{window_ms}ms"

    if products == []:
        mean_col = f"__mean_tmp_{col_prefix}"
        std_col = f"__std_tmp_{col_prefix}"
        z_col = f"zscore_{window_ms}ms_{col_prefix}"

        rolling_stats = (
            df.rolling(index_column=time_col, period=window_str, closed="right")
            .agg([
                pl.col(col_prefix).mean().alias(mean_col),
                pl.col(col_prefix).std().alias(std_col),
            ])
        )

        df = df.join(rolling_stats, on=time_col, how="right")
        df = df.with_columns(
            ((pl.col(col_prefix) - pl.col(mean_col)) / (pl.col(std_col)+1e-10)).alias(z_col)
        ).drop([mean_col, std_col])

    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            mean_col = f"__mean_tmp_{col}"
            std_col = f"__std_tmp_{col}"
            z_col = f"zscore_{window_ms}ms_{col}"

            rolling_stats = (
                df.rolling(index_column=time_col, period=window_str, closed="right")
                .agg([
                    pl.col(col).mean().alias(mean_col),
                    pl.col(col).std().alias(std_col),
                ])
            )

            df = df.join(rolling_stats, on=time_col, how="right")

            df = df.with_columns(
                ((pl.col(col) - pl.col(mean_col)) / pl.col(std_col)).alias(z_col)
            ).drop([mean_col, std_col])

    return df.drop_nulls()


def add_zscore_tick(
    df: pl.DataFrame,
    ticks: int = 50,
    base_col: str = "mid",           # 'mid' or 'vwap'
    levels: int = 1,
    depth: int = 1,
    products: list[str] | None = None
) -> pl.DataFrame:
    """
    Adds Z-score over a rolling tick window for mid or VWAP price.

    Z = (x_t - mean_t) / std_t where mean and std are over last N ticks.

    Parameters:
        df (pl.DataFrame): Input data.
        base_col (str): 'mid' or 'vwap'.
        ticks (int): Number of rows to use in rolling window.
        levels (int): LOB levels for VWAP (ignored for 'mid').
        depth (int): Book depth (only used if VWAP is missing).
        products (list[str] | None): Optional product suffixes.

    Returns:
        pl.DataFrame: DataFrame with z-score columns, nulls dropped.
    """
    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    else:
        raise ValueError("base_col must be 'mid' or 'vwap'.")

    if products is None:
        products = get_products(df, [col_prefix])

    if products == []:
        z_col = f"zscore_t{ticks}_{col_prefix}"
        df = df.with_columns([
            (
                (pl.col(col_prefix) - pl.col(col_prefix).rolling_mean(window_size=ticks))
                / (pl.col(col_prefix).rolling_std(window_size=ticks)+1e-10)
            ).alias(z_col)
        ])
    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            z_col = f"zscore_t{ticks}_{col}"
            df = df.with_columns([
                (
                    (pl.col(col) - pl.col(col).rolling_mean(window_size=ticks))
                    / (pl.col(col).rolling_std(window_size=ticks)+1e-10)
                ).alias(z_col)
            ])

    return df.drop_nulls()

def add_avg_time(
    df: pl.DataFrame,
    window: int = 100,            # in milliseconds
    base_col: str = "mid",         # 'mid', 'vwap', or 'spread'
    levels: int = 1,
    depth: int = 1,
    time_col: str = "ts_dt",
    products: list[str] | None = None
) -> pl.DataFrame:
    if time_col not in df.columns:
        df = add_ts_dt(df)

    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    elif base_col == "spread":
        col_prefix = "spread"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_spread(df, products=products)
    else:
        raise ValueError("base_col must be 'mid', 'vwap', or 'spread'.")

    if products is None:
        products = get_products(df, [col_prefix])

    window_str = f"{window}ms"

    if products == []:
        avg_col = f"avg_{window}ms_{col_prefix}"
        rolling_df = (
            df.rolling(index_column=time_col, period=window_str, closed="right")
            .agg([pl.col(col_prefix).mean().alias(avg_col)])
        )
        df = df.join(rolling_df, on=time_col, how="right")
    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            avg_col = f"avg_{window}ms_{col}"
            rolling_df = (
                df.rolling(index_column=time_col, period=window_str, closed="right")
                .agg([pl.col(col).mean().alias(avg_col)])
            )
            df = df.join(rolling_df, on=time_col, how="right")

    return df.drop_nulls()

def add_std_time(
    df: pl.DataFrame,
    window: int = 100,             # in milliseconds
    base_col: str = "mid",         # 'mid', 'vwap', or 'spread'
    levels: int = 1,
    depth: int = 1,
    time_col: str = "ts_dt",
    products: list[str] | None = None
) -> pl.DataFrame:
    if time_col not in df.columns:
        df = add_ts_dt(df)

    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)
    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)
    elif base_col == "spread":
        col_prefix = "spread"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_spread(df, products=products)
    else:
        raise ValueError("base_col must be 'mid', 'vwap', or 'spread'.")

    if products is None:
        products = get_products(df, [col_prefix])

    window_str = f"{window}ms"

    if products == []:
        std_col = f"std_{window}ms_{col_prefix}"
        rolling_df = (
            df.rolling(index_column=time_col, period=window_str, closed="right")
            .agg([pl.col(col_prefix).std().alias(std_col)])
        )
        df = df.join(rolling_df, on=time_col, how="right")
    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            std_col = f"std_{window}ms_{col}"
            rolling_df = (
                df.rolling(index_column=time_col, period=window_str, closed="right")
                .agg([pl.col(col).std().alias(std_col)])
            )
            df = df.join(rolling_df, on=time_col, how="right")

    return df.drop_nulls()

def add_avg_tick(
    df: pl.DataFrame,
    ticks: int = 50,
    base_col: str = "mid",         # 'mid', 'vwap', or 'spread'
    levels: int = 1,
    depth: int = 1,
    products: list[str] | None = None
) -> pl.DataFrame:
    """
    Adds a rolling tick-based average of mid, VWAP, or spread over the past N ticks.

    Parameters:
        df (pl.DataFrame): Input DataFrame.
        base_col (str): 'mid', 'vwap', or 'spread'.
        ticks (int): Rolling window size in rows.
        levels (int): VWAP levels (only used for 'vwap').
        depth (int): Book depth if VWAP needs to be created.
        products (list[str] | None): Optional product suffixes.

    Returns:
        pl.DataFrame: Updated DataFrame with average columns.
    """
    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)

    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)

    elif base_col == "spread":
        col_prefix = "spread"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_spread(df, products=products)

    else:
        raise ValueError("base_col must be 'mid', 'vwap', or 'spread'.")

    if products is None:
        products = get_products(df, [col_prefix])

    if products == []:
        avg_col = f"avg_t{ticks}_{col_prefix}"
        df = df.with_columns([
            pl.col(col_prefix).rolling_mean(window_size=ticks).alias(avg_col)
        ])
    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            avg_col = f"avg_t{ticks}_{col}"
            df = df.with_columns([
                pl.col(col).rolling_mean(window_size=ticks).alias(avg_col)
            ])

    return df.drop_nulls()

def add_std_tick(
    df: pl.DataFrame,
    ticks: int = 50,
    base_col: str = "mid",         # 'mid', 'vwap', or 'spread'
    levels: int = 1,
    depth: int = 1,
    products: list[str] | None = None
) -> pl.DataFrame:
    """
    Adds a rolling tick-based standard deviation of mid, VWAP, or spread
    over the past N ticks (rows).

    Parameters:
        df (pl.DataFrame): Input DataFrame.
        base_col (str): 'mid', 'vwap', or 'spread'.
        ticks (int): Rolling window size in rows.
        levels (int): VWAP levels (only used for 'vwap').
        depth (int): Book depth if VWAP needs to be created.
        products (list[str] | None): Optional product suffixes.

    Returns:
        pl.DataFrame: Updated DataFrame with std columns.
    """
    if base_col == "mid":
        col_prefix = "mid"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_mid(df, products=products)

    elif base_col == "vwap":
        col_prefix = f"vwap_level{levels}"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_vwap(df, levels=levels, depth=depth, products=products)

    elif base_col == "spread":
        col_prefix = "spread"
        if not any(col.startswith(col_prefix) for col in df.columns):
            df = add_spread(df, products=products)

    else:
        raise ValueError("base_col must be 'mid', 'vwap', or 'spread'.")

    if products is None:
        products = get_products(df, [col_prefix])

    if products == []:
        std_col = f"std_t{ticks}_{col_prefix}"
        df = df.with_columns([
            pl.col(col_prefix).rolling_std(window_size=ticks).alias(std_col)
        ])
    else:
        for product in products:
            col = f"{col_prefix}_{product}"
            std_col = f"std_t{ticks}_{col}"
            df = df.with_columns([
                pl.col(col).rolling_std(window_size=ticks).alias(std_col)
            ])

    return df.drop_nulls()
