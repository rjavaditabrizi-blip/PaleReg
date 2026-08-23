"""Fetch and cache S&P 500 daily OHLCV data.

Usage:
    python -m src.data --years 5
"""
import argparse
import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKERS_CACHE = DATA_DIR / "sp500_tickers.csv"
PANEL_CACHE = DATA_DIR / "ohlcv_panel.parquet"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_table(refresh: bool = False) -> pd.DataFrame:
    """Returns a DataFrame with columns [ticker, sector] (GICS Sector)."""
    if TICKERS_CACHE.exists() and not refresh:
        return pd.read_csv(TICKERS_CACHE)

    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    out = pd.DataFrame(
        {
            "ticker": table["Symbol"].str.replace(".", "-", regex=False),
            "sector": table["GICS Sector"],
        }
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TICKERS_CACHE, index=False)
    return out


def get_sp500_tickers(refresh: bool = False) -> list[str]:
    return get_sp500_table(refresh=refresh)["ticker"].tolist()


def download_ohlcv(tickers: list[str], period: str, batch_size: int = 50) -> pd.DataFrame:
    """Download daily OHLCV for all tickers and return a long panel:
    columns = [date, ticker, open, high, low, close, volume]

    `period` is any yfinance period string, e.g. "5y" or "6mo".
    """
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Downloading {i + 1}-{i + len(batch)} / {len(tickers)}: {batch[0]}..{batch[-1]}")
        df = yf.download(
            batch,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in df.columns.get_level_values(0):
                    continue
                sub = df[ticker].dropna(how="all").copy()
                if sub.empty:
                    continue
                sub["ticker"] = ticker
                frames.append(sub)
        else:
            # single-ticker batch fallback
            sub = df.dropna(how="all").copy()
            sub["ticker"] = batch[0]
            frames.append(sub)

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    panel.columns = [c.lower() for c in panel.columns]
    panel = panel[["date", "ticker", "open", "high", "low", "close", "volume"]]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--refresh-tickers", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers (for quick tests)")
    args = parser.parse_args()

    tickers = get_sp500_tickers(refresh=args.refresh_tickers)
    if args.limit:
        tickers = tickers[: args.limit]

    panel = download_ohlcv(tickers, period=f"{args.years}y")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_CACHE, index=False)
    print(f"Saved {len(panel):,} rows across {panel['ticker'].nunique()} tickers -> {PANEL_CACHE}")


if __name__ == "__main__":
    main()
