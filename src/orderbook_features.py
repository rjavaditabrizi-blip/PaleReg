"""Compute order-book-imbalance-style features from a single L2 snapshot
(see providers/orderbook_base.py for the snapshot shape).

This is deliberately the crypto prototype for the "L2 data" idea discussed
for the equity pipeline: free crypto exchange APIs give a live order book
snapshot with no account/API key, so you can test whether these features
carry any signal before ever paying for equity-market L2 data. It does NOT
give you free historical L2 -- see collector.py for what that actually
requires.
"""
from pathlib import Path

import pandas as pd

DEPTH_TIERS = (5, 10, 25)


def compute_book_features(order_book: dict) -> dict:
    bids, asks = order_book["bids"], order_book["asks"]
    best_bid, best_bid_size = bids.iloc[0]["price"], bids.iloc[0]["size"]
    best_ask, best_ask_size = asks.iloc[0]["price"], asks.iloc[0]["size"]

    mid_price = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    microprice = (best_bid * best_ask_size + best_ask * best_bid_size) / (
        best_bid_size + best_ask_size
    )

    features = {
        "symbol": order_book["symbol"],
        "timestamp": order_book["timestamp"],
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread": spread,
        "spread_bps": spread / mid_price * 10_000,
        "microprice": microprice,
        "microprice_minus_mid_bps": (microprice - mid_price) / mid_price * 10_000,
    }

    for n in DEPTH_TIERS:
        bid_depth = bids["size"].head(n).sum()
        ask_depth = asks["size"].head(n).sum()
        total = bid_depth + ask_depth
        features[f"bid_depth_top{n}"] = bid_depth
        features[f"ask_depth_top{n}"] = ask_depth
        features[f"imbalance_top{n}"] = (bid_depth - ask_depth) / total if total > 0 else 0.0

    return features


def collect_snapshot(provider, symbol: str, out_path: str, depth: int = 50) -> dict:
    """Fetch one snapshot, compute its features, and append a row to a CSV.
    Run this periodically (cron, the `schedule`/`loop` skill, etc.) to start
    building an actual historical L2 dataset -- the only realistic way to
    get one for free, since there's no historical L2 API to backfill from."""
    book = provider.fetch_order_book(symbol, depth=depth)
    row = compute_book_features(book)
    df = pd.DataFrame([row])
    header = not Path(out_path).exists()
    df.to_csv(out_path, mode="a", header=header, index=False)
    return row
