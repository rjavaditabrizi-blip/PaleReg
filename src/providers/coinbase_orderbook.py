"""Coinbase Exchange public order book API. No API key/account needed --
this is public market data. Symbol format: "BTC-USD", "ETH-USD", etc."""
import pandas as pd
import requests

from .orderbook_base import OrderBookProvider

BASE_URL = "https://api.exchange.coinbase.com"


class CoinbaseOrderBookProvider(OrderBookProvider):
    name = "coinbase"

    def fetch_order_book(self, symbol: str, depth: int = 50) -> dict:
        # level=2 gives aggregated price levels (what we want for L2 depth);
        # level=3 would give every individual open order, which is far more
        # detail than a depth-imbalance feature needs.
        resp = requests.get(
            f"{BASE_URL}/products/{symbol}/book",
            params={"level": 2},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        bids = pd.DataFrame(data["bids"][:depth], columns=["price", "size", "n_orders"]).astype(
            {"price": float, "size": float}
        )
        asks = pd.DataFrame(data["asks"][:depth], columns=["price", "size", "n_orders"]).astype(
            {"price": float, "size": float}
        )
        return {
            "symbol": symbol,
            "timestamp": pd.Timestamp.utcnow(),
            "bids": bids[["price", "size"]],
            "asks": asks[["price", "size"]],
        }
