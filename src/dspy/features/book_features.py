"""
Module for generating features from book data.

This module provides functions to add spread, volume, and other features to order book data.
"""

import polars as pl

from dspy.features.utils import get_products
# Features for prices   

def add_mid(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['prc_s0', 'prc_s1']) -> pl.DataFrame:
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

def add_spread(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['prc_s0', 'prc_s1']) -> pl.DataFrame:
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

def add_vwap(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['prc_s0', 'prc_s1', 'vol_s0', 'vol_s1']) -> pl.DataFrame:
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

def add_rel_returns(df: pl.DataFrame, products: list[str] | None = None, cols: list[str]=['mid']) -> pl.DataFrame:
    """
    Add a relative return column to the DataFrame.
    """
    if products is None:
        products = get_products(df, cols)
    if isinstance(cols, str):
        cols = [cols]

    if products == []:
        df = df.with_columns(
            pl.col(f"{cols[0]}").pct_change().alias("rel_return")
        )
    else:
        df = df.with_columns(
            [
                pl.col(f"{col}_{product}").pct_change().alias(f"rel_return_{col}_{product}")
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

def lob_vector(row: dict, depth: int = 5) -> list[float]:
    """
    Extracts a flattened LOB vector from the row for top `depth` levels.
    Format: [bid_p0, bid_q0, ..., ask_p0, ask_q0, ...]
    """
    lob_features = []

    # Bids
    for i in range(depth):
        price_key = f"bids.price[{i}]"
        qty_key   = f"bids.amount[{i}]"
        lob_features.append(row.get(price_key, 0.0))
        lob_features.append(row.get(qty_key, 0.0))

    # Asks
    for i in range(depth):
        price_key = f"asks.price[{i}]"
        qty_key   = f"asks.amount[{i}]"
        lob_features.append(row.get(price_key, 0.0))
        lob_features.append(row.get(qty_key, 0.0))

    return lob_features

def add_lob_price_level(
    df: pl.DataFrame,
    side: str = "bid",
    level: int = 0,
    products: list[str] | None = None,
) -> pl.DataFrame:
    """
    Adds a column for the LOB price at a given side and level.
    """
    col_name = f"{'bids' if side == 'bid' else 'asks'}[{level}].price"
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
