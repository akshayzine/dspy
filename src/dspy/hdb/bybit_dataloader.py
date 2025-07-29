"""
Module for loading and processing Bybit data
This module provides functionality to access and manipulate Bybit datasets.
"""

from pathlib import Path
import logging
import polars as pl

# Local imports
from dspy.hdb.base import DataLoader
from dspy.hdb.registry import register_dataset

logger = logging.getLogger(__name__)    

BYBIT_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "bybit"
LOB_BASE_URL = "https://quote-saver.bycsi.com/orderbook/linear/"


@register_dataset("bybit")
class BybitData(DataLoader):
    """
    Dataloader for Bybit data
    """
    def __init__(self, root: str | Path = BYBIT_DATA_PATH):
        logger.info("Initializing BybitDataLoader with path %s", root)
        super().__init__(root)

    def load_book(self, products: list[str], times: list[str], depth: int = 1, lazy=False) -> pl.DataFrame:
        """
        Load book data for a given product and times.
        """
        df = super().load_book(products, times, depth, lazy)
        df = df.rename({"prc__s0": "prc_s0", "prc__s1": "prc_s1", "vol__s0": "vol_s0", "vol__s1": "vol_s1"})
        return df

def _load_data(self, product: str, times: list[str], type: str, lazy=False) -> pl.DataFrame:    
        """
        Load data for a given product and times.
        """
        if len(times) != 2:
            raise ValueError("Times must be a list of two strings in the format '%y%m%d.%H%M'")
        try:
            dtimes = [datetime.strptime(t, "%y%m%d.%H%M%S") for t in times]
        except ValueError:
            raise ValueError("Times must be in the format '%y%m%d.%H%M%S'")
        months = get_months(dtimes[0], dtimes[1])

        dfs = []
        for month in months:
            filename = "{}/{}_{}_{}.parquet".format(str(self.processed_path), product, month, type)
            if not Path(filename).exists():
                logger.info(f"File {filename} not found, downloading...")
                self.download(product, month, type)
                logger.info("File downloaded, processing...")
                df = self.process(product, month, type)
                if df is None:
                    logger.info(f"Product {product} with type {type} and month {month} is not available")
                    return None
            else:
                # check if the dataframe is already in the cache
                if self.cache is not None and filename in self.cache and not lazy:
                    df = self.cache[filename]
                else:
                    if lazy:
                        df = pl.scan_parquet(filename)
                    else:
                        df = pl.read_parquet(filename)
                        if self.cache is not None:
                            self.cache[filename] = df
            dfs.append(df)

        df = pl.concat(dfs)
        df = df.filter(pl.col('ts').is_between(nanoseconds(times[0]), nanoseconds(times[1])))
        return df