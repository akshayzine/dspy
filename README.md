# dspy

A Python data handling system for high-frequency data. Can handle both freely available data and act as a wrapper for proprietary packages.

## Installation

```zsh
git clone git@github.com:Tripudium/dspy.git
```

Install using the [uv](https://docs.astral.sh/uv/) package manager:

```zsh
uv python list
uv .venv --python 3.13.2
source .venv/bin/activate
uv sync
```

To make with work with the proprietary Terank ```trpy-data``` framework, this needs to be installed:

```zsh
uv pip install -e /path/to/trpy-data
```

Some further hacking may be necessary.

## Usage

Data is available in two forms: limit order book (LOB) and fixed frequency data (trade data will be included too). The available depth depends on the ultimate data source being used. The timestamps are given in nanosecond resolution as Unix timestamps. A simple dataloader and some helper function to convert Python datetime objects or strings of the form '240802.145010' into timestamps are provided.

```python
from dspy.hdb import get_dataset

dl = get_dataset("tardis") # uses data provided by tardis
```

To get book data:

```python
df = dl.load_book(product='BTCUSDT', times=['250120.000100', '250120.215000'], depth=10)
# Add human readable timestamp and mid prices
df = df.ds.add_datetime('ts').feature.add_mid(products=['BTCUSDT'])
```

Note: the data is expected as parquet files in the data/tardis/processed directory. If the files are not present, the loader will attempt to fetch them from tardis and preprocess them. This, however, requires a Tardis subscription and a corresponding API key. The API key can be set in the environment variable TARDIS_API_KEY. As this is not provided by default, the already preprocessed parquet files for BTCUSDT from Binance from April to June 2025 are provided on [huggingface](https://huggingface.co/datasets/tripudium/tardisdata/tree/main). This data should be downloaded and placed in the data/tardis/processed directory.

See the [example notebook](examples/dataloading.ipynb) for more.

## Additional packages

This package is used by downstream packages such as [cooc](https://github.com/Tripudium/cooc).





