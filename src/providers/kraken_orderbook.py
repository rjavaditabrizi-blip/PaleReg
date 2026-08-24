"""Kraken public order book API. No API key/account needed. Symbol format
is Kraken's own pair naming, e.g. "XBTUSD" (BTC/USD), "ETHUSD"."""
import pandas as pd
import requests

from .orderbook_base import OrderBookProvider

BASE_URL = "https://api.kraken.com/0/public/Depth"


class KrakenOrderBookProvider(OrderBookProvider):
    name = "kraken"

    def fetch_order_book(self, symbol: str, depth: int = 50) -> dict:
        resp = requests.get(BASE_URL, params={"pair": symbol, "count": depth}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken API error: {data['error']}")

        # Kraken echoes back its own canonical pair name as the result key
        # (e.g. "XBTUSD" -> "XXBTZUSD"), which may not match what was passed in.
        book = next(iter(data["result"].values()))
        bids = pd.DataFrame(book["bids"], columns=["price", "size", "ts"]).astype(
            {"price": float, "size": float}
        )
        asks = pd.DataFrame(book["asks"], columns=["price", "size", "ts"]).astype(
            {"price": float, "size": float}
        )
        return {
            "symbol": symbol,
            "timestamp": pd.Timestamp.utcnow(),
            "bids": bids[["price", "size"]],
            "asks": asks[["price", "size"]],
        }
