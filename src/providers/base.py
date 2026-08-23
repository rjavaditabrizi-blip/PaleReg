"""Interface every data source implements, so the mining/combine/backtest
pipeline never needs to know where its data came from.

Everything downstream of data-loading only depends on two shapes:
- a universe table: columns [ticker, sector]
- an OHLCV panel: columns [date, ticker, open, high, low, close, volume]

Add a new source (a different market-data API, a fundamentals feed, an
alt-data provider) by implementing DataProvider and registering it in
providers/__init__.py -- nothing in features.py/train.py/combine.py/
backtest.py/predict_today.py needs to change.
"""
from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    name: str

    @abstractmethod
    def fetch_universe(self, refresh: bool = False) -> pd.DataFrame:
        """Returns a DataFrame with columns [ticker, sector]."""

    @abstractmethod
    def fetch_ohlcv(self, tickers: list[str], period: str) -> pd.DataFrame:
        """Returns a long panel: columns [date, ticker, open, high, low, close, volume]."""
