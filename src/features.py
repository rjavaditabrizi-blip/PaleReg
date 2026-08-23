"""Build operand features and forward-return targets from the OHLCV panel.

Mirrors the "basic feature" style used in formulaic-alpha papers (AlphaForge /
AlphaFormer): open, high, low, close, volume, vwap-proxy, plus rolling-window
transforms (mean, std, sum, delta) over a few lookback windows. Everything is
cross-sectionally z-scored per trading day so gplearn sees comparable scales
across stocks of very different price levels (this is also what makes a mined
formula transferable across stocks, per the Symbolic Modeling framing).

Also computes a sector-neutralized target (fwd_ret - that day's sector-mean
forward return): mining against raw fwd_ret risks the GP search just finding
"which sector will do well this week" rather than genuine stock-level
selection skill. The raw fwd_ret is kept too, for realistic backtest P&L.

`_compute_operand_columns` is shared between `build_features` (training,
which also needs the target and can drop warmup/lookahead NaNs) and
`compute_live_features` (inference, which has no target and must keep the
most recent rows even though *their* fwd_ret would be NaN) so the two paths
can't silently drift apart -- that kind of train/serve skew is a classic way
for a live model to quietly stop matching what was validated.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import get_sp500_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL_CACHE = DATA_DIR / "ohlcv_panel.parquet"
FEATURES_CACHE = DATA_DIR / "features.parquet"

WINDOWS = (5, 10, 20)
HORIZON = 5  # trading days ahead for the forward return target


def _compute_operand_columns(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["ret_1d"] = df.groupby("ticker", sort=False)["close"].pct_change()

    close_g = df.groupby("ticker", sort=False)["close"]
    volume_g = df.groupby("ticker", sort=False)["volume"]
    for w in WINDOWS:
        # NOTE: sum_volume_w and mean_volume_w are proportional (sum = w * mean), which
        # become byte-identical columns after cross-sectional z-scoring below -- only
        # keep one (sum) to avoid wasting GP search space on a duplicate operand.
        df[f"mean_close_{w}"] = close_g.transform(lambda s, w=w: s.rolling(w).mean())
        df[f"std_close_{w}"] = close_g.transform(lambda s, w=w: s.rolling(w).std())
        df[f"sum_volume_{w}"] = volume_g.transform(lambda s, w=w: s.rolling(w).sum())
        df[f"delta_close_{w}"] = df["close"] - close_g.transform(lambda s, w=w: s.shift(w))
        df[f"ewma_close_{w}"] = close_g.transform(lambda s, w=w: s.ewm(halflife=w).mean())

    df["mom_60"] = df["close"] - close_g.transform(lambda s: s.shift(60))
    df["corr_close_volume_10"] = df.groupby("ticker", sort=False, group_keys=False)[
        ["close", "volume"]
    ].apply(lambda g: g["close"].rolling(10).corr(g["volume"]))

    feature_cols = [
        c
        for c in df.columns
        if c not in {"date", "ticker", "open", "high", "low", "close", "volume"}
    ]
    return df, feature_cols


def _zscore_cross_sectionally(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    date_g = df.groupby("date", sort=False)[feature_cols]
    means = date_g.transform("mean")
    stds = date_g.transform("std").replace(0, np.nan)
    df[feature_cols] = ((df[feature_cols] - means) / stds).fillna(0.0)
    return df


def _attach_sector(df: pd.DataFrame) -> pd.DataFrame:
    sector_table = get_sp500_table()[["ticker", "sector"]]
    df = df.merge(sector_table, on="ticker", how="left")
    df["sector"] = df["sector"].fillna("Unknown")
    return df


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Training path: adds fwd_ret + sector-neutralized fwd_ret_neutral, drops
    warmup/lookahead NaNs, then cross-sectionally z-scores the operand columns."""
    df, feature_cols = _compute_operand_columns(panel)
    df["fwd_ret"] = df.groupby("ticker", sort=False)["close"].transform(
        lambda s: s.shift(-HORIZON) / s - 1.0
    )
    df = _attach_sector(df)

    df = df.dropna(subset=feature_cols + ["fwd_ret"]).reset_index(drop=True)

    sector_mean_fwd_ret = df.groupby(["date", "sector"], sort=False)["fwd_ret"].transform("mean")
    df["fwd_ret_neutral"] = df["fwd_ret"] - sector_mean_fwd_ret

    df = _zscore_cross_sectionally(df, feature_cols)
    return df, feature_cols


def compute_live_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Inference path: same operand computation, no target (there's nothing
    to shift forward into yet), keep the most recent rows even though their
    fwd_ret would have been NaN. Caller should take the latest date's rows."""
    df, feature_cols = _compute_operand_columns(panel)
    df = _attach_sector(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df = _zscore_cross_sectionally(df, feature_cols)
    return df, feature_cols


def main():
    panel = pd.read_parquet(PANEL_CACHE)
    df, feature_cols = build_features(panel)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_CACHE, index=False)
    print(f"Saved {len(df):,} rows, {len(feature_cols)} features -> {FEATURES_CACHE}")
    print("Feature columns:", feature_cols)
    print("Sectors:", sorted(df["sector"].unique()))


if __name__ == "__main__":
    main()
