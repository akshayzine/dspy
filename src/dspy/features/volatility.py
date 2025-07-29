"""
Module for generating volatility features.
"""
import polars as pl

from dspy.features.utils import get_products


def add_realized_vola(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a realized volatility column to the DataFrame.
    """
    return df