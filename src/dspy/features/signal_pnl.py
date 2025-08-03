
import polars as pl
import numpy as np
from datetime import datetime, timedelta

from dspy.utils import str_to_timedelta, timedelta_to_nanoseconds

def add_sig_pnl(
        df: pl.DataFrame,
        ts_col: str = 'ts',
        col: str = 'prc',
        signal: str | None = None,
        horizon: str = '1s',
        in_bp: bool = True,
        fee_in_bp: float = 0.0
        ) -> pl.DataFrame:
    """
    Add a signal PnL column to the DataFrame.
    """

    tdelta = str_to_timedelta(horizon)
    if df[ts_col].dtype == pl.UInt64 or df[ts_col].dtype == pl.Int64:
        tdelta = timedelta_to_nanoseconds(tdelta)

    expr_diff = (pl.col(f"fut_{col}") - pl.col(col))
    if in_bp:
        expr_diff = (expr_diff * 10_000 / pl.col(col)) - fee_in_bp
    if signal is not None:
        expr_diff *= pl.col(signal) if not in_bp else pl.col(signal).sign()
    
    fut_df = df.select(pl.col(ts_col), pl.col(col).alias(f"fut_{col}"))
    df = df.filter(pl.col(ts_col) <= pl.col(ts_col).max() - tdelta)

    df_shift = df.select(
        [ (pl.col(ts_col)+tdelta).alias('ts').set_sorted(), pl.col(col), pl.col(signal) ]
    ).join_asof(
        fut_df, 
        on=ts_col, 
        strategy='backward'
    ).select(
        (expr_diff).alias(f'pnl_sig_{horizon}')
    )

    df_sig = pl.concat((df.select(pl.col(ts_col)), df_shift), how='horizontal')
    return df.join(df_sig, on=ts_col, how='left')

def sync_with_book(
        df: pl.DataFrame, 
        bdf: pl.DataFrame, 
        on: str = "ts", 
        cols: list[str] = ['prc_s0', 'prc_s1', 'vol_s0', 'vol_s1', 'sig_pnl']
        ) -> pl.DataFrame:
    """
    Sync a dataframe with the book dataframe on a given timestamp column.
    """
    bdf = bdf.select([on, *cols])
    df = df.join_asof(bdf, on=on, strategy="backward")
    return df

def add_signal(
        df: pl.DataFrame,
        signal_df: pl.DataFrame,
        signal_col: str,
        on: str = "ts",
        ) -> pl.DataFrame:
    """
    Add a signal column from signal_df to df by joining on the timestamp column.
    
    Args:
        df: The dataframe to add the signal to
        signal_df: The dataframe containing the signal
        signal_col: The name of the signal column in signal_df
        on: The timestamp column to join on (default: "ts")
        
    Returns:
        DataFrame with the signal column added
    """
    signal_df = signal_df.select([on, signal_col]).set_sorted(on)
    df = df.join_asof(signal_df, on=on, strategy="backward").fill_null(0)
    return df

def create_signal_test_dataframes():
    """
    Create two test dataframes for add_signal function.
    
    Returns:
        tuple: (df, signal_df) where:
            - df: DataFrame with 'ts' and 'prc' columns (20 entries)
            - signal_df: DataFrame with 'ts' and 'trade' columns (8 entries)
            with timestamps interleaved between df's timestamps
    """
    
    start_dt = datetime(2023, 1, 1, 0, 0, 0)

    base_ts = pl.DataFrame({
        "dt": [start_dt + timedelta(minutes=i) for i in range(20)]
    })["dt"]
    
    df = pl.DataFrame({
        "ts": base_ts,
        "prc": np.random.normal(100, 5, 20)
    })
    
    deltas = [2,3,2,3,2,3]
    signal_ts = [
        datetime(2023, 1, 1, 0, 0, 30) + timedelta(minutes=deltas[i] * i)
        for i in range(6)
    ]
    
    signal_df = pl.DataFrame({
        "ts": signal_ts,
        "signal": [1, -1, 2, -2, 3, -3]
    })
    signal_df = signal_df.sort("ts").set_sorted("ts")
    
    return df, signal_df
