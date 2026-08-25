"""Screen candidate features for standalone predictive power and stability
BEFORE running an expensive SR search on them.

This automates what was previously a one-off manual diagnostic: computing
each feature's per-fold IC against forward returns directly (no mining
needed -- just correlation), which is exactly how high_low_range_10 was
caught as a variance-inducing feature after the fact. Running this first
means a bad feature gets caught in seconds instead of after a 1-2 hour
walk-forward validation.

Ranks by Information Ratio (mean IC / std IC across folds) rather than raw
mean IC, and separately flags features with anomalously large IC variance
relative to the rest of the pool.

Important caveat this tool can't fix by itself: a LOW standalone/univariate
IC does NOT mean a feature is useless to the search. SR mining looks for
NONLINEAR combinations, not just direct correlation -- delta_close_5 screens
near the bottom here on its own, but it's a core ingredient of the best
mined formula found so far ("sum_volume_10 - delta_close_5"). Use this to
flag candidates for review (especially the variance-outlier flag, which
would have caught high_low_range_10 by its raw std alone -- its IR looked
unremarkable, ~1.07, right in the middle of the pack), not as an automatic
hard filter that silently removes anything with a weak solo correlation.

Usage:
    python -m src.feature_screen
    python -m src.feature_screen --top-n 15
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fitness import mean_ic
from .folds import make_folds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_CACHE = DATA_DIR / "features.parquet"
SCREEN_CACHE = RESULTS_DIR / "feature_screen.json"

NON_FEATURE_COLS = {"date", "ticker", "sector", "open", "high", "low", "close", "volume", "fwd_ret", "fwd_ret_neutral"}


def screen_features(df: pd.DataFrame, feature_cols: list[str], n_folds: int = 5, min_train_frac: float = 0.4) -> pd.DataFrame:
    unique_dates = np.sort(df["date"].unique())
    folds = make_folds(unique_dates, n_folds, min_train_frac)

    rows = []
    for feat in feature_cols:
        ics = []
        for _, test_dates in folds:
            test_df = df[df["date"].isin(test_dates)]
            ic = mean_ic(test_df[feat].values, test_df["fwd_ret"].values, test_df["date"].values)
            ics.append(ic)
        ics = np.array(ics)
        mean_ic_val, std_ic = float(ics.mean()), float(ics.std(ddof=0))
        rows.append(
            {
                "feature": feat,
                "mean_ic": mean_ic_val,
                "std_ic": std_ic,
                "information_ratio": mean_ic_val / std_ic if std_ic > 1e-9 else 0.0,
                "hit_rate": float(np.mean(ics > 0)),
                "min_ic": float(ics.min()),
                "max_ic": float(ics.max()),
            }
        )
    result = pd.DataFrame(rows)
    median_std = result["std_ic"].median()
    result["variance_ratio"] = result["std_ic"] / median_std if median_std > 1e-9 else 0.0
    return result.sort_values("information_ratio", ascending=False).reset_index(drop=True)


def select_features(
    screen_df: pd.DataFrame, top_n: int | None = None, min_ir: float | None = None,
    max_variance_ratio: float | None = None,
) -> list[str]:
    df = screen_df
    if min_ir is not None:
        df = df[df["information_ratio"] >= min_ir]
    if max_variance_ratio is not None:
        df = df[df["variance_ratio"] <= max_variance_ratio]
    if top_n is not None:
        df = df.head(top_n)
    return df["feature"].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-frac", type=float, default=0.4)
    parser.add_argument("--top-n", type=int, default=None, help="Print/save only the top N features by information ratio")
    parser.add_argument("--min-ir", type=float, default=None, help="Flag features below this information ratio as screening candidates for removal")
    parser.add_argument(
        "--variance-flag-ratio", type=float, default=4.0,
        help="Flag features whose IC std is more than this many times the pool's median std "
        "(high_low_range_10 was ~5-6x; the volume factors, which are genuinely good, sit "
        "around ~2-3x -- default 4.0 catches the former without flagging the latter)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_CACHE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    screen_df = screen_features(df, feature_cols, args.n_folds, args.min_train_frac)
    if args.top_n:
        screen_df = screen_df.head(args.top_n)

    print(f"{'feature':<22} {'mean_ic':>9} {'std_ic':>8} {'IR':>7} {'var_ratio':>9} {'hit_rate':>9} {'range':>20}")
    for _, r in screen_df.iterrows():
        flags = []
        if args.min_ir is not None and r["information_ratio"] < args.min_ir:
            flags.append("LOW IR")
        if r["variance_ratio"] > args.variance_flag_ratio:
            flags.append(f"VARIANCE {r['variance_ratio']:.1f}x pool median -- review before mining")
        flag_str = f"  <-- {', '.join(flags)}" if flags else ""
        print(
            f"{r['feature']:<22} {r['mean_ic']:>+9.4f} {r['std_ic']:>8.4f} {r['information_ratio']:>7.2f} "
            f"{r['variance_ratio']:>8.1f}x {r['hit_rate']:>8.0%}  [{r['min_ic']:+.4f}, {r['max_ic']:+.4f}]{flag_str}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCREEN_CACHE, "w") as f:
        json.dump(screen_df.to_dict(orient="records"), f, indent=2)
    print(f"\nSaved -> {SCREEN_CACHE}")


if __name__ == "__main__":
    main()
