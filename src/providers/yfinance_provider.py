"""The current (and so far only) data source: free delayed OHLCV from
yfinance, universe/sector from the Wikipedia S&P 500 table. Thin wrapper
around the existing functions in src/data.py -- no logic duplicated here."""
import pandas as pd

from ..data import download_ohlcv, get_sp500_table
from .base import DataProvider


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def fetch_universe(self, refresh: bool = False) -> pd.DataFrame:
        return get_sp500_table(refresh=refresh)

    def fetch_ohlcv(self, tickers: list[str], period: str) -> pd.DataFrame:
        return download_ohlcv(tickers, period=period)
