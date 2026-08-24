"""Interface for Level-2 (order book depth) data sources.

Separate from DataProvider (base.py) because the shape is fundamentally
different: DataProvider deals in daily OHLCV panels; an order book is a
point-in-time snapshot of bid/ask price levels and sizes. Also, critically,
these two data types have different HISTORY availability: free crypto
exchange APIs give you a live snapshot of the order book *right now*, not
years of historical L2 data the way yfinance gives years of historical
OHLCV. There is no free lunch for historical L2 -- see collector.py for the
practical implication of that.
"""
from abc import ABC, abstractmethod

import pandas as pd


class OrderBookProvider(ABC):
    name: str

    @abstractmethod
    def fetch_order_book(self, symbol: str, depth: int = 50) -> dict:
        """Returns a live snapshot:
        {
            "symbol": str,
            "timestamp": pd.Timestamp (UTC),
            "bids": pd.DataFrame[price, size]  -- sorted best (highest) first
            "asks": pd.DataFrame[price, size]  -- sorted best (lowest) first
        }
        """
