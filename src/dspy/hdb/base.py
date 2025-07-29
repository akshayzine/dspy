"""
Base class for loading data
"""

from pathlib import Path
import logging
from datetime import datetime
import polars as pl

# Local imports
from dspy.utils import str_to_timedelta, round_up_to_nearest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"

def get_months(start_date: datetime, end_date: datetime) -> list[str]:
    """
    Given two datetime objects, generate a list of months between them as strings in 'MM' format.
    """
    months = []
    current_date = start_date.replace(day=1)
    
    while current_date <= end_date:
        months.append(current_date.strftime('%y%m'))
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    return sorted(months)

class DataLoader:
    """
    Base class for loading and processing financial data.
    
    This class provides common functionality for downloading, processing,
    and accessing financial data from various sources.
    """
    def __init__(self, root: str | Path = DATA_PATH, cache: bool = True):
        """
        Initialize the DataLoader with a path to the data.
        """
        logger.info("Initializing DataLoader with path %s"%root)
        self.root = root
        self._raw_path = Path(root) / "raw"
        self._processed_path = Path(root) / "processed"
        # maintain a cache of dataframes
        if cache:
            self.cache = {}
        else:
            self.cache = None

    @property
    def raw_path(self):
        """
        Get the path to raw data files.
        
        Returns:
            Path: The directory containing raw data files.
        """
        return str(self._raw_path)
    
    @property
    def processed_path(self):
        """
        Get the path to processed data files.
        
        Returns:
            Path: The directory containing processed data files.
        """
        return str(self._processed_path)

    @raw_path.setter
    def raw_path(self, path: str | Path):
        """
        Set the path to raw data files.
        
        Args:
            path (str | Path): The directory to store raw data files.
        """
        self._raw_path = path

    @processed_path.setter
    def processed_path(self, path: str | Path):
        """
        Set the path to processed data files.
        
        Args:
            path (str | Path): The directory to store processed data files.
        """
        self._processed_path = path
    
    def load_trades(self, products: list[str] | str, times: list[str], lazy=False) -> pl.DataFrame:
        """
        Load trades data for a given product and times.
        """
        if isinstance(products, str):
            products = [products]
        dfs = []
        for product in products:
            df = self._load_data(product, times, "trade", lazy)
            df = df.with_columns(pl.lit(product).alias('product'))
            dfs.append(df)
        df = pl.concat(dfs).sort('ts')
        return df

    def load_book(self, _product: str, _times: list[str], _depth: int = 10, _lazy: bool = False) -> pl.DataFrame:
        """
        Load book data for a given product and times.
        """
        raise NotImplementedError
    
    def load(self, products: list[str], times: list[str], col: str = "mid", freq: str = "1s", lazy=False) -> pl.DataFrame:
        """
        Load data for a given product and times.
        """
        df = self.load_book(products, times, lazy=lazy)
        df = df.ds.add_datetime()
        dtimes = [datetime.strptime(t, "%y%m%d.%H%M%S") for t in times]
        try:
            td = str_to_timedelta(freq)
        except ValueError:
            raise ValueError(f"Invalid frequency: {freq}")
        
        min_dt = round_up_to_nearest(df["dts"][0], td)
        max_dt = dtimes[1]

        # Make sure that every timestamp is present in the dataframe
        rdf = pl.DataFrame(
            { "dts": pl.datetime_range(min_dt, max_dt, freq, time_unit="ns", eager=True) }
        )
        rdf = rdf.join_asof(df, on="dts", strategy="backward")

        if col == "mid":
            rdf = rdf.feature.add_mid(cols=["prc_s0", "prc_s1"])
            if len(products) > 1:
                rdf = rdf.select([pl.col("dts").alias("ts")] + [pl.col(f"mid_{product}") for product in products])
            else:
                rdf = rdf.select([pl.col("dts").alias("ts"), pl.col("mid")])
        elif col == "vwap":
            rdf = rdf.feature.add_vwap(cols=["prc_s0", "prc_s1", "vol_s0", "vol_s1"])
            if len(products) > 1:
                rdf= rdf.select([pl.col("dts").alias("ts")] + [pl.col(f"vwap_{product}") for product in products]) 
            else:
                rdf = rdf.select([pl.col("dts").alias("ts"), pl.col("vwap")])
        return rdf
        
    def download(self, _product: str, _month: str, _type: str):
        raise NotImplementedError

    def process(self, _product: str, _month: str, _type: str):
        raise NotImplementedError