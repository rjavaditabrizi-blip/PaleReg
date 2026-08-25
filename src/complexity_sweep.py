"""Empirically validate PySR's complexity budget (maxsize) instead of
guessing it -- maxsize=20 was picked without tuning back when the PySR
engine was first built. Walk-forward-validates a few maxsize values and
compares fold hit-rate/variance, not just mean IC, per the same
"consistency over lucky means" standard applied throughout this project
(this is also directly the property the shared quant-literature summary
attributed to symbolic regression's complexity penalty: simpler equations
should generalize better out-of-sample -- worth actually testing that,
not assuming it).

Shares fold boundaries and feature data across all maxsize settings tested,
rather than re-loading per setting.

Usage:
    python -m src.complexity_sweep --maxsizes 10 20 30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import decile_backtest
from .combine import static_combine
from .engines.pysr_engine import mine_pool_pysr_multiseed
from .features import HORIZON
from .fitness import mean_rank_ic
from .folds import make_folds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_CACHE = DATA_DIR / "features.parquet"
SWEEP_CACHE = RESULTS_DIR / "complexity_sweep.json"

NON_FEATURE_COLS = {"date", "ticker", "sector", "open", "high", "low", "close", "volume", "fwd_ret", "fwd_ret_neutral"}


def run_one_maxsize(df, feature_cols, folds, maxsize, args):
    fold_results = []
    for k, (train_cutoff, test_dates) in enumerate(folds):
        train_df = df[df["date"] <= train_cutoff]
        test_df = df[df["date"].isin(test_dates)]
        X_train, y_train, dates_train = train_df[feature_cols].values, train_df["fwd_ret"].values, train_df["date"].values
        X_test, y_test, dates_test = test_df[feature_cols].values, test_df["fwd_ret"].values, test_df["date"].values

        pool_train, pool_test, formulas, _models, y_train_used, dates_train_used = mine_pool_pysr_multiseed(
            X_train, X_test, y_train, dates_train, feature_cols,
            n_seeds=args.n_seeds, seed=args.seed, verbose=False,
            niterations=args.niterations, populations=args.populations,
            population_size=args.population_size, maxsize=maxsize,
            procs=args.procs, batch_days=args.batch_days,
        )
        weights, train_ic, test_ic, test_pred = static_combine(
            pool_train, y_train_used, pool_test, y_test, dates_train_used, dates_test, args.alpha
        )
        test_rank_ic = mean_rank_ic(test_pred, y_test, dates_test)
        bt_df = pd.DataFrame({"date": dates_test, "alpha": test_pred, "fwd_ret": y_test})
        metrics = decile_backtest(bt_df, cost_bps=args.cost_bps, horizon=HORIZON)
        fold_results.append({"fold": k + 1, "n_unique_formulas": len(formulas), "test_ic": test_ic, "test_rank_ic": test_rank_ic, **metrics})
        print(f"  fold {k + 1}: IC={test_ic:+.4f} Sharpe={metrics['sharpe']:.2f} ({len(formulas)} formulas)")
    return fold_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxsizes", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-frac", type=float, default=0.4)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--niterations", type=int, default=20)
    parser.add_argument("--populations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=33)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--batch-days", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_CACHE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    unique_dates = np.sort(df["date"].unique())
    folds = make_folds(unique_dates, args.n_folds, args.min_train_frac)

    all_results = {}
    for maxsize in args.maxsizes:
        print(f"\n=== maxsize={maxsize} ===")
        fold_results = run_one_maxsize(df, feature_cols, folds, maxsize, args)
        sharpes = np.array([r["sharpe"] for r in fold_results if not np.isnan(r["sharpe"])])
        ics = np.array([r["test_ic"] for r in fold_results])
        all_results[str(maxsize)] = {
            "maxsize": maxsize,
            "sharpe_mean": float(sharpes.mean()) if len(sharpes) else float("nan"),
            "sharpe_std": float(sharpes.std(ddof=0)) if len(sharpes) else float("nan"),
            "sharpe_hit_rate": float(np.mean(sharpes > 0)) if len(sharpes) else float("nan"),
            "ic_mean": float(ics.mean()),
            "ic_std": float(ics.std(ddof=0)),
            "ic_hit_rate": float(np.mean(ics > 0)),
            "folds": fold_results,
        }

    print(f"\n{'=' * 66}\nComplexity sweep summary\n{'=' * 66}")
    print(f"{'maxsize':>8} {'SharpeMean':>11} {'SharpeStd':>10} {'SharpeHit':>10} {'ICMean':>9} {'ICHit':>7}")
    for s in all_results.values():
        print(f"{s['maxsize']:>8} {s['sharpe_mean']:>+11.2f} {s['sharpe_std']:>10.2f} {s['sharpe_hit_rate']:>9.0%} {s['ic_mean']:>+9.4f} {s['ic_hit_rate']:>6.0%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_CACHE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved -> {SWEEP_CACHE}")


if __name__ == "__main__":
    main()
