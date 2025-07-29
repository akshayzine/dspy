"""
This module provides additional functionality for Polars DataFrames.
"""
import polars as pl

from dspy.features import add_mid, add_spread, add_volume, add_vwap, add_rel_returns, add_log_returns, add_sig_pnl, agg_trades, add_side, add_size

@pl.api.register_dataframe_namespace("ds")
@pl.api.register_lazyframe_namespace("ds")
class DatetimeMethods:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def add_datetime(self, ts_col: str='ts') -> pl.DataFrame:
        """
        Add a datetime column to the DataFrame.
        """
        return self._df.with_columns([pl.from_epoch(ts_col, time_unit='ns').alias('dts')])
    
    def aggregate(self, cols: list[str]) -> pl.DataFrame:
        """
        Keep only the rows where a change happens in columns indexed by cols.
        """
        agg_func = [pl.col(col).first() for col in cols]
        return self._df.group_by(cols, maintain_order=True).agg(agg_func)

@pl.api.register_dataframe_namespace("feature")
@pl.api.register_lazyframe_namespace("feature")
class FeatureMethods:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def add_mid(self, products: list[str] | None = None, cols: list[str]=['prc_s0', 'prc_s1']) -> pl.DataFrame:
        """
        Add a mid column to the DataFrame.
        """
        return add_mid(self._df, products, cols)

    def add_spread(self, products: list[str] | None = None, cols: list[str]=['prc_s0', 'prc_s1']) -> pl.DataFrame:
        """
        Add a spread column to the DataFrame.
        """
        return add_spread(self._df, products, cols)
    
    def add_volume(self, products: list[str] | None = None, cols: list[str]=['vol_s0', 'vol_s1']) -> pl.DataFrame:
        """
        Add a volume column to the DataFrame.
        """
        return add_volume(self._df, products, cols)

    def add_vwap(self, products: list[str] | None = None, cols: list[str]=['prc_s0', 'prc_s1', 'vol_s0', 'vol_s1']) -> pl.DataFrame:
        """
        Add a VWAP column to the DataFrame.
        """
        return add_vwap(self._df, products, cols)

    def add_rel_returns(self, products: list[str] | None = None, cols: list[str]=['mid']) -> pl.DataFrame:
        """
        Add a relative return column to the DataFrame.
        """
        return add_rel_returns(self._df, products, cols)
    
    def add_log_returns(self, products: list[str] | None = None, cols: list[str]=['mid']) -> pl.DataFrame:
        """
        Add a log return column to the DataFrame.
        """
        return add_log_returns(self._df, products, cols)
    
@pl.api.register_dataframe_namespace("trade")
@pl.api.register_lazyframe_namespace("trade")
class TradeMethods:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def agg_trades(self, cols: list[str]=['ts', 'prc', 'product']) -> pl.DataFrame:
        """
        Aggregate trades by timestamp and price.
        """
        return agg_trades(self._df, cols)
    
    def add_side(self, col: str='qty') -> pl.DataFrame:
        """
        Add a side column to the DataFrame.
        """
        return add_side(self._df, col)
    
    def add_size(self, col: str='qty') -> pl.DataFrame:
        """
        Add a size column to the DataFrame.
        """
        return add_size(self._df, col)
    
@pl.api.register_dataframe_namespace("target")
@pl.api.register_lazyframe_namespace("target")
class TargetMethods:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def add_sig_pnl(self,
                    ts_col: str = 'ts', 
                    col: str = 'prc',
                    signal: str | None = None,
                    horizon: str = '1s',
                    in_bp: bool = True,
                    fee_in_bp: float = 0.0) -> pl.DataFrame:
        """ 
        Add a signal PnL column to the DataFrame.
        """
        return add_sig_pnl(self._df, ts_col, col, signal, horizon, in_bp, fee_in_bp)