"""Proper walk-forward validation.

The single-static-split runs in this project swung from Sharpe +0.76 to
-0.35 between two otherwise-similar runs, because each mined a different
thin, borderline-IC formula and got scored on just one fixed test window.
That's not evidence of an edge (or its absence) -- it's evidence that one
split isn't enough data to tell a real weak signal apart from noise.

This script re-mines the factor pool from scratch at each fold, using only
data strictly before that fold (an expanding window -- no peeking), then
backtests purely out-of-sample on the fold itself. It reports the
DISTRIBUTION of outcomes across folds (mean/std/hit-rate), which is what
actually tells you whether there's a real, reproducible edge.

This reruns the full mining pipeline once per fold, so it costs roughly
n_folds times as much compute as a single train.py run -- scale down
--population/--generations/--n-seeds accordingly for a first pass.

Usage:
    python -m src.walkforward --n-folds 5 --min-train-frac 0.4 \
        --n-seeds 4 --population 300 --generations 15
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import decile_backtest, ic_weighted_backtest
from .combine import static_combine
from .engines.pysr_engine import mine_pool_pysr_multiseed
from .features import HORIZON
from .fitness import mean_rank_ic
from .train import mine_pool

METHODS = {"decile": decile_backtest, "ic_weighted": ic_weighted_backtest}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_CACHE = DATA_DIR / "features.parquet"


def make_folds(unique_dates: np.ndarray, n_folds: int, min_train_frac: float):
    n = len(unique_dates)
    min_train_end = int(n * min_train_frac)
    fold_bounds = np.linspace(min_train_end, n, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        start_idx, end_idx = fold_bounds[i], fold_bounds[i + 1]
        if start_idx >= end_idx or start_idx == 0:
            continue
        train_cutoff = unique_dates[start_idx - 1]
        test_dates = unique_dates[start_idx:end_idx]
        folds.append((train_cutoff, test_dates))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-frac", type=float, default=0.4, help="Minimum initial training history, as a fraction of total dates")
    parser.add_argument("--engine", choices=["gplearn", "pysr"], default="gplearn")
    parser.add_argument("--tag", default=None, help="Output filename suffix. Defaults to '' for gplearn, '_pysr' for pysr, so a pysr run can't silently overwrite the validated gplearn baseline the UI reads.")
    # gplearn-specific
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--population", type=int, default=300)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--hall-of-fame", type=int, default=30)
    parser.add_argument("--parsimony", type=float, default=0.001)
    parser.add_argument("--tournament-size", type=int, default=20)
    parser.add_argument("--p-crossover", type=float, default=0.7)
    parser.add_argument("--p-subtree-mutation", type=float, default=0.1)
    parser.add_argument("--p-hoist-mutation", type=float, default=0.05)
    parser.add_argument("--p-point-mutation", type=float, default=0.1)
    parser.add_argument("--metric", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--n-jobs", type=int, default=-1)
    # pysr-specific
    parser.add_argument("--pysr-niterations", type=int, default=30)
    parser.add_argument("--pysr-populations", type=int, default=10)
    parser.add_argument("--pysr-population-size", type=int, default=30)
    parser.add_argument("--pysr-maxsize", type=int, default=20)
    parser.add_argument("--pysr-procs", type=int, default=4)
    parser.add_argument("--pysr-batch-days", type=int, default=60, help="Whole trading days sampled fresh per loss evaluation (see pysr_engine.py)")
    parser.add_argument("--pysr-n-seeds", type=int, default=1, help="Independent PySR searches to pool together (mirrors gplearn's --n-seeds)")
    # shared
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1e-4, help="Lasso regularization strength")
    parser.add_argument(
        "--target",
        choices=["raw", "neutral"],
        default="neutral",
        help="fwd_ret (raw) or fwd_ret_neutral (sector-demeaned) for mining + IC. Backtest P&L always uses raw fwd_ret.",
    )
    parser.add_argument("--method", choices=list(METHODS), default="decile")
    parser.add_argument("--sector-neutral", action="store_true", help="Form the backtest portfolio within each GICS sector")
    parser.add_argument("--n-buckets", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    tag = args.tag if args.tag is not None else ("" if args.engine == "gplearn" else f"_{args.engine}")
    wf_results_cache = RESULTS_DIR / f"walkforward_results{tag}.json"
    oos_predictions_cache = RESULTS_DIR / f"oos_predictions{tag}.parquet"

    df = pd.read_parquet(FEATURES_CACHE)
    feature_cols = [
        c
        for c in df.columns
        if c not in {"date", "ticker", "sector", "open", "high", "low", "close", "volume", "fwd_ret", "fwd_ret_neutral"}
    ]
    target_col = "fwd_ret" if args.target == "raw" else "fwd_ret_neutral"
    backtest_fn = METHODS[args.method]
    unique_dates = np.sort(df["date"].unique())
    folds = make_folds(unique_dates, args.n_folds, args.min_train_frac)
    print(f"{len(folds)} folds from {len(unique_dates)} unique trading days (min_train_frac={args.min_train_frac})")

    fold_results = []
    all_oos = []
    for k, (train_cutoff, test_dates) in enumerate(folds):
        print(f"\n{'=' * 70}\nFold {k + 1}/{len(folds)}: train <= {train_cutoff}  |  test {test_dates[0]} .. {test_dates[-1]} ({len(test_dates)} days)\n{'=' * 70}")

        train_df = df[df["date"] <= train_cutoff]
        test_df = df[df["date"].isin(test_dates)]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        dates_train = train_df["date"].values

        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        dates_test = test_df["date"].values
        tickers_test = test_df["ticker"].values
        sectors_test = test_df["sector"].values
        raw_y_test = test_df["fwd_ret"].values

        if args.engine == "gplearn":
            pool_train, pool_test, formulas, _models = mine_pool(
                X_train, X_test, y_train, dates_train, feature_cols, args, verbose=False
            )
        else:
            pool_train, pool_test, formulas, _models, y_train, dates_train = mine_pool_pysr_multiseed(
                X_train, X_test, y_train, dates_train, feature_cols,
                n_seeds=args.pysr_n_seeds,
                seed=args.seed,
                verbose=False,
                niterations=args.pysr_niterations,
                populations=args.pysr_populations,
                population_size=args.pysr_population_size,
                maxsize=args.pysr_maxsize,
                procs=args.pysr_procs,
                batch_days=args.pysr_batch_days,
            )
        n_unique = len(formulas)

        weights, train_ic, test_ic, test_pred = static_combine(
            pool_train, y_train, pool_test, y_test, dates_train, dates_test, args.alpha
        )
        test_rank_ic = mean_rank_ic(test_pred, y_test, dates_test)

        bt_df = pd.DataFrame(
            {
                "date": dates_test, "ticker": tickers_test, "alpha": test_pred,
                "fwd_ret": raw_y_test, "sector": sectors_test,
            }
        )
        metrics = backtest_fn(
            bt_df, cost_bps=args.cost_bps, horizon=HORIZON, sector_neutral=args.sector_neutral,
            **({"n_buckets": args.n_buckets} if args.method == "decile" else {}),
        )
        oos_chunk = bt_df.copy()
        oos_chunk["fold"] = k + 1
        all_oos.append(oos_chunk)

        result = {
            "fold": k + 1,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_cutoff": str(train_cutoff),
            "test_start": str(test_dates[0]),
            "test_end": str(test_dates[-1]),
            "n_unique_formulas": n_unique,
            "formulas": formulas,
            "train_ic": train_ic,
            "test_ic": test_ic,
            "test_rank_ic": test_rank_ic,
            **metrics,
        }
        fold_results.append(result)
        print(
            f"Fold {k + 1}: IC={test_ic:+.4f} RankIC={test_rank_ic:+.4f} "
            f"Sharpe={metrics['sharpe']:.2f} ann_return={metrics['ann_return']:+.2%} "
            f"({n_unique} unique formulas)"
        )

    sharpes = np.array([r["sharpe"] for r in fold_results if not np.isnan(r["sharpe"])])
    ics = np.array([r["test_ic"] for r in fold_results])
    rank_ics = np.array([r["test_rank_ic"] for r in fold_results])
    ann_returns = np.array([r["ann_return"] for r in fold_results if not np.isnan(r["ann_return"])])

    print(f"\n{'=' * 70}\nWalk-forward summary across {len(fold_results)} folds\n{'=' * 70}")
    if len(sharpes) > 0:
        print(f"Sharpe:      mean={sharpes.mean():+.2f}  std={sharpes.std(ddof=0):.2f}  min={sharpes.min():+.2f}  max={sharpes.max():+.2f}  hit_rate(>0)={np.mean(sharpes > 0):.0%}")
        print(f"Ann. return: mean={ann_returns.mean():+.2%}  std={ann_returns.std(ddof=0):.2%}")
    print(f"IC:          mean={ics.mean():+.4f}  std={ics.std(ddof=0):.4f}  hit_rate(>0)={np.mean(ics > 0):.0%}")
    print(f"Rank IC:     mean={rank_ics.mean():+.4f}  std={rank_ics.std(ddof=0):.4f}  hit_rate(>0)={np.mean(rank_ics > 0):.0%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(wf_results_cache, "w") as f:
        json.dump(fold_results, f, indent=2, default=str)
    print(f"\nSaved per-fold results -> {wf_results_cache}")

    pd.concat(all_oos, ignore_index=True).to_parquet(oos_predictions_cache, index=False)
    print(f"Saved pooled out-of-sample predictions -> {oos_predictions_cache}")


if __name__ == "__main__":
    main()
