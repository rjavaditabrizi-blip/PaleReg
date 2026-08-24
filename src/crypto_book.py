"""Fetch a live L2 order book snapshot from a crypto exchange and print its
computed imbalance/spread features. Optionally append it to a CSV -- run
this periodically (cron, the `schedule`/`loop` skill, etc.) to start
building an actual historical L2 dataset, since there's no free historical
L2 API to backfill from.

Usage:
    python -m src.crypto_book --provider coinbase --symbol BTC-USD
    python -m src.crypto_book --provider kraken --symbol XBTUSD --depth 100
    python -m src.crypto_book --provider coinbase --symbol ETH-USD --out data/eth_book.csv
"""
import argparse

from .orderbook_features import collect_snapshot, compute_book_features
from .providers import get_orderbook_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["coinbase", "kraken"], default="coinbase")
    parser.add_argument("--symbol", default="BTC-USD", help="coinbase: 'BTC-USD' style; kraken: 'XBTUSD' style")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--out", default=None, help="If set, append this snapshot's features as a row to this CSV")
    args = parser.parse_args()

    provider = get_orderbook_provider(args.provider)
    if args.out:
        row = collect_snapshot(provider, args.symbol, args.out, depth=args.depth)
        print(f"Appended snapshot -> {args.out}")
    else:
        book = provider.fetch_order_book(args.symbol, depth=args.depth)
        row = compute_book_features(book)

    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
