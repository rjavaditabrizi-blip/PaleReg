"""Backtests for a composite alpha score: decile long-short, and an
IC-weighted continuous-exposure alternative.

Both rebalance every HORIZON trading days (matching the forward-return
horizon used to train/combine the factors) to avoid double-counting
overlapping return windows. Each period is a fresh, fully-independent basket
(no position carryover tracked), so turnover is ~100% every rebalance --
--cost-bps charges that explicitly instead of silently assuming a free lunch.

Both also support sector_neutral=True, which forms the portfolio *within*
each GICS sector instead of across the whole universe -- otherwise a
"top decile" or "highest z-score" portfolio can just be a leveraged bet on
whichever sector is in favor that period, which isn't the stock-selection
skill the alpha is supposed to be measuring.

Usage:
    python -m src.backtest --method decile --n-buckets 10 --cost-bps 5
    python -m src.backtest --method ic_weighted --sector-neutral --cost-bps 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import HORIZON

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ALPHA_CACHE = RESULTS_DIR / "composite_alpha.npz"


def _periods_to_metrics(period_returns: list, period_long_only: list, horizon: int) -> dict:
    n_periods = len(period_returns)
    if n_periods < 2:
        return {
            "n_periods": n_periods, "ann_return": float("nan"), "ann_vol": float("nan"),
            "sharpe": float("nan"), "long_only_ann_return": float("nan"), "equity_multiple": float("nan"),
        }

    period_returns = np.array(period_returns)
    period_long_only = np.array(period_long_only)
    periods_per_year = 252 / horizon

    equity = np.cumprod(1 + period_returns)
    ann_return = equity[-1] ** (periods_per_year / n_periods) - 1
    ann_vol = period_returns.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

    lo_equity = np.cumprod(1 + period_long_only)
    lo_ann_return = lo_equity[-1] ** (periods_per_year / n_periods) - 1

    return {
        "n_periods": n_periods,
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "long_only_ann_return": float(lo_ann_return),
        "equity_multiple": float(equity[-1]),
    }


def decile_backtest(
    df: pd.DataFrame, n_buckets: int = 10, cost_bps: float = 5.0, horizon: int = HORIZON,
    sector_neutral: bool = False,
) -> dict:
    """df needs columns: date, alpha, fwd_ret (+ sector if sector_neutral=True)."""
    cost = cost_bps / 10_000.0
    unique_dates = np.sort(df["date"].unique())
    rebalance_dates = unique_dates[::horizon]

    period_returns, period_long_only = [], []
    for d in rebalance_dates:
        day = df[df["date"] == d]

        if sector_neutral:
            tops, bottoms = [], []
            for _, g in day.groupby("sector"):
                if len(g) < n_buckets * 2:
                    continue
                bucket = pd.qcut(g["alpha"], n_buckets, labels=False, duplicates="drop")
                tops.append(g[bucket == bucket.max()])
                bottoms.append(g[bucket == bucket.min()])
            if not tops:
                continue
            top, bottom = pd.concat(tops), pd.concat(bottoms)
        else:
            if len(day) < n_buckets * 2:
                continue
            bucket = pd.qcut(day["alpha"], n_buckets, labels=False, duplicates="drop")
            top, bottom = day[bucket == bucket.max()], day[bucket == bucket.min()]

        period_returns.append(top["fwd_ret"].mean() - bottom["fwd_ret"].mean() - 4 * cost)
        period_long_only.append(top["fwd_ret"].mean() - 2 * cost)

    return _periods_to_metrics(period_returns, period_long_only, horizon)


def ic_weighted_backtest(
    df: pd.DataFrame, cost_bps: float = 5.0, horizon: int = HORIZON, sector_neutral: bool = False,
    min_names: int = 20,
) -> dict:
    """Continuous-exposure alternative to decile bucketing: weight every name
    by its z-scored alpha (long positive, short negative) instead of only
    trading the extremes. Uses the full cross-section's information, which
    should be less noisy than a top/bottom-decile spread when the underlying
    IC is real but modest. Gross exposure is normalized to 2 (1x long + 1x
    short) each period. df needs columns: date, alpha, fwd_ret (+ sector if
    sector_neutral=True)."""
    cost = cost_bps / 10_000.0
    unique_dates = np.sort(df["date"].unique())
    rebalance_dates = unique_dates[::horizon]

    period_returns, period_long_only = [], []
    for d in rebalance_dates:
        day = df[df["date"] == d]
        if len(day) < min_names:
            continue

        if sector_neutral:
            grp = day.groupby("sector")["alpha"]
            std = grp.transform("std").replace(0, np.nan)
            z = ((day["alpha"] - grp.transform("mean")) / std).fillna(0.0)
        else:
            std = day["alpha"].std(ddof=0)
            z = (day["alpha"] - day["alpha"].mean()) / std if std > 0 else day["alpha"] * 0.0

        gross = z.abs().sum()
        if gross == 0:
            continue
        w = z / gross * 2.0  # 1x long + 1x short gross exposure

        period_returns.append(float((w * day["fwd_ret"]).sum()) - 4 * cost)
        long_mask = w > 0
        long_gross = w[long_mask].sum()
        period_long_only.append(
            float((w[long_mask] * day.loc[long_mask, "fwd_ret"]).sum() / long_gross) - 2 * cost
            if long_gross > 0 else 0.0
        )

    return _periods_to_metrics(period_returns, period_long_only, horizon)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["decile", "ic_weighted"], default="decile")
    parser.add_argument("--n-buckets", type=int, default=10, help="Decile count (decile method only)")
    parser.add_argument("--sector-neutral", action="store_true", help="Form the portfolio within each GICS sector")
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="One-way transaction cost in bps per trade. Long-short pays 4x this per period "
        "(open+close on both legs); long-only pays 2x. Set to 0 to reproduce the frictionless number.",
    )
    args = parser.parse_args()

    data = np.load(ALPHA_CACHE, allow_pickle=True)
    df = pd.DataFrame(
        {
            "date": data["dates_test"],
            "ticker": data["tickers_test"],
            "alpha": data["test_pred"],
            "fwd_ret": data["raw_y_test"] if "raw_y_test" in data else data["y_test"],
        }
    )
    if "sectors_test" in data:
        df["sector"] = data["sectors_test"]
    elif args.sector_neutral:
        raise SystemExit("No sector data in composite_alpha.npz -- re-run combine.py after re-mining with sector support.")

    if args.method == "decile":
        m = decile_backtest(df, n_buckets=args.n_buckets, cost_bps=args.cost_bps, sector_neutral=args.sector_neutral)
    else:
        m = ic_weighted_backtest(df, cost_bps=args.cost_bps, sector_neutral=args.sector_neutral)

    print(f"Method: {args.method}{' (sector-neutral)' if args.sector_neutral else ''}")
    print(f"Rebalance periods: {m['n_periods']} (every {HORIZON} trading days)")
    print(f"Long-short spread: ann. return={m['ann_return']:+.2%}  ann. vol={m['ann_vol']:.2%}  Sharpe={m['sharpe']:.2f}")
    print(f"Long-only:         ann. return={m['long_only_ann_return']:+.2%}")
    print(f"Final long-short equity multiple: {m['equity_multiple']:.3f}x")


if __name__ == "__main__":
    main()
