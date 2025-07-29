
"""
Add position tracking columns to the DataFrame.
"""
import polars as pl
import datetime

def add_positions(
        df: pl.DataFrame,
        products: list[str] = ['BTCUSD'],
        price_type: str = 'mid',
        order_cols: list[str] = ['pos'],   
        fees_bps: list[float] = [0.0]
        ) -> pl.DataFrame:
    """
    Add position tracking columns to the DataFrame.
    
    Args:
        df: Input DataFrame
        price_cols: Column names for prices
        order_cols: Column names for orders
        fees_bps: List of fees in basis points
        
    Returns:
        DataFrame with added position tracking columns:
        - 'inventory': Current inventory in the product
        - f'pnl_{price_col}': Cumulative profit and loss for the price column
        - 'pnl': Cumulative profit and loss of portfolio
    """
    price_cols = [f'{price_type}_{product}' for product in products]
    fees = [fee_bps / 10_000 for fee_bps in fees_bps]
    for i, product in enumerate(products):
        df = df.with_columns(
            pl.col(order_cols[i]).cum_sum().alias(f'pos_{product}')
        )
        df = df.with_columns(
            (pl.col(f'pos_{product}').shift(1) * (pl.col(price_cols[i]) - pl.col(price_cols[i]).shift(1))).fill_null(0).alias('local_pnl')
        )
        df = df.with_columns(
            (pl.col('local_pnl')-pl.col(price_cols[i]).mul(fees[i]).mul(pl.col(order_cols[i]).abs())).alias(f'local_pnl_{product}')
        ).drop('local_pnl') 
        df = df.with_columns(
            pl.col(f'local_pnl_{product}').cum_sum().alias(f'pnl_{product}')
        ).drop([f'local_pnl_{product}'])

    pnl_cols = [f'pnl_{product}' for product in products]
    if len(pnl_cols) == 1:
        df = df.with_columns(
            pl.col(pnl_cols[0]).alias('pnl')
        )
    else:
        df = df.with_columns(
            pl.sum_horizontal(pnl_cols).alias('pnl')
        )
    return df

def rebalance_positions(
        df: pl.DataFrame,
        products: list[str] = ['BTCUSD'],
        price_type: str = 'mid',
        pos_cols: list[str] = ['pos'],
        fees_bps: list[float] = [0.0]
        ) -> pl.DataFrame:
    """
    Rebalance positions in the DataFrame.
    """
    df = df.with_columns(
        [
            pl.col(pos_cols[i]).diff(null_behavior='ignore').fill_null(df[pos_cols[i]][0]).alias(f'order_{product}')
            for i, product in enumerate(products)
        ]
    )
    order_cols = [f'order_{product}' for product in products]
    df = add_positions(df, products=products, price_type=price_type, order_cols=order_cols, fees_bps=fees_bps)
    return df

def create_test_positions_data() -> pl.DataFrame:
    """
    Create a small test dataset for testing position tracking functionality.
    
    Returns:
        A DataFrame with 10 rows containing test data with columns:
        - 'ts': Timestamp
        - 'prc': Price
        - 'pos_side': Position side (+1 for buy, -1 for sell)
        - 'pos_size': Position size
    """
    data = {
        'ts': [
            datetime.datetime(2023, 1, 1, 0, 0, i) 
            for i in range(10)
        ],
        'mid_BTCUSDT': [100.0, 101.0, 102.0, 101.5, 101.0, 100.5, 100.0, 101.0, 102.0, 103.0],
        'mid_ETHUSDT': [50.0, 50.5, 51.0, 51.5, 52.0, 51.5, 51.0, 52.0, 53.0, 52.5],
        'order_BTCUSDT': [1.0, 0.5, -1.0, 0.0, -0.5, 1.0, 0.0, 2.0, -1.5, 1.0],
        'order_ETHUSDT': [0.5, -0.5, 1.5, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5]
    }
    
    products = ['BTCUSDT', 'ETHUSDT']
    order_cols = ['order_BTCUSDT', 'order_ETHUSDT']
    price_type = 'mid'
    fees_bps = [2.0, 0.0]
    
    df = pl.DataFrame(data)
    df_with_positions = add_positions(df, products=products, price_type=price_type, order_cols=order_cols, fees_bps=fees_bps)
    
    return df_with_positions
