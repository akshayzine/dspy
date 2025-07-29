"""
Module for generating features from trade data.

This module provides functions to add size, side, and other features to trade data.
"""

import polars as pl

# Freatures for trades

def agg_trades(df: pl.DataFrame, cols: list[str]=['ts', 'prc', 'product']) -> pl.DataFrame:
    """
    Aggregate trades by timestamp and price.
    """
    return df.group_by(cols, maintain_order=True).agg(pl.col('trade_id').first(), pl.col('qty').sum())
    
def add_side(df: pl.DataFrame, col: str='qty') -> pl.DataFrame:
    """
    Add a side column to the DataFrame.
    """
    df = df.with_columns(
        pl.when(pl.col(col) > 0).then(1).otherwise(-1).alias('side'))
    return df
    
def add_size(df: pl.DataFrame, col: str='qty') -> pl.DataFrame:
    """
    Add a size column to the DataFrame.
    """
    df = df.with_columns(
        pl.col(col).abs().alias('size'))
    return df
