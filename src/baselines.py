"""Cheap linear/tree baselines to sanity-check whether the expensive SR
search is actually earning its cost, and (for the tree model) to screen
which features/interactions look worth SR's attention before committing to
another expensive walk-forward run.

Both are validated through the EXACT SAME walk-forward fold structure and
IC/Sharpe/decile-backtest machinery as the SR engines (make_folds,
decile_backtest, mean_ic/mean_rank_ic) -- a direct, honest, apples-to-apples
comparison against the validated production Sharpe +0.29 / IC +0.0218, not
a different methodology dressed up as one.

Usage:
    python -m src.baselines --model elasticnet
    python -m src.baselines --model lightgbm
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import decile_backtest
from .features import HORIZON
from .fitness import mean_ic, mean_rank_ic
from .folds import make_folds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_CACHE = DATA_DIR / "features.parquet"

NON_FEATURE_COLS = {"date", "ticker", "sector", "open", "high", "low", "close", "volume", "fwd_ret", "fwd_ret_neutral"}


def elasticnet_fit_predict(X_train, y_train, X_test, feature_cols):
    from sklearn.linear_model import ElasticNetCV

    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=3, n_jobs=-1, max_iter=5000)
    model.fit(X_train, y_train)
    coefs = dict(zip(feature_cols, model.coef_.tolist()))
    return model.predict(X_test), coefs


def lightgbm_fit_predict(X_train, y_train, X_test, feature_cols):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train)
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    return model.predict(X_test), importances


FIT_PREDICT_FNS = {"elasticnet": elasticnet_fit_predict, "lightgbm": lightgbm_fit_predict}


def run_baseline_walkforward(df, feature_cols, folds, fit_predict_fn, cost_bps=5.0):
    fold_results, importances_per_fold = [], []
    for k, (train_cutoff, test_dates) in enumerate(folds):
        train_df = df[df["date"] <= train_cutoff]
        test_df = df[df["date"].isin(test_dates)]
        X_train, y_train = train_df[feature_cols].values, train_df["fwd_ret"].values
        X_test, y_test = test_df[feature_cols].values, test_df["fwd_ret"].values
        dates_test = test_df["date"].values

        pred_test, importance = fit_predict_fn(X_train, y_train, X_test, feature_cols)
        importances_per_fold.append(importance)

        test_ic = mean_ic(pred_test, y_test, dates_test)
        test_rank_ic = mean_rank_ic(pred_test, y_test, dates_test)
        bt_df = pd.DataFrame({"date": dates_test, "alpha": pred_test, "fwd_ret": y_test})
        metrics = decile_backtest(bt_df, cost_bps=cost_bps, horizon=HORIZON)
        fold_results.append({"fold": k + 1, "test_ic": test_ic, "test_rank_ic": test_rank_ic, **metrics})
        print(f"  fold {k + 1}: IC={test_ic:+.4f} RankIC={test_rank_ic:+.4f} Sharpe={metrics['sharpe']:.2f}")
    return fold_results, importances_per_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(FIT_PREDICT_FNS), required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-frac", type=float, default=0.4)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_CACHE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    unique_dates = np.sort(df["date"].unique())
    folds = make_folds(unique_dates, args.n_folds, args.min_train_frac)
    print(f"{len(folds)} folds, model={args.model}, {len(feature_cols)} features\n")

    fold_results, importances_per_fold = run_baseline_walkforward(
        df, feature_cols, folds, FIT_PREDICT_FNS[args.model], args.cost_bps
    )

    sharpes = np.array([r["sharpe"] for r in fold_results if not np.isnan(r["sharpe"])])
    ics = np.array([r["test_ic"] for r in fold_results])
    rank_ics = np.array([r["test_rank_ic"] for r in fold_results])

    print(f"\n{'=' * 60}\n{args.model} walk-forward summary\n{'=' * 60}")
    print(f"Sharpe:  mean={sharpes.mean():+.2f}  std={sharpes.std(ddof=0):.2f}  hit_rate={np.mean(sharpes > 0):.0%}")
    print(f"IC:      mean={ics.mean():+.4f}  std={ics.std(ddof=0):.4f}  hit_rate={np.mean(ics > 0):.0%}")
    print(f"Rank IC: mean={rank_ics.mean():+.4f}  std={rank_ics.std(ddof=0):.4f}  hit_rate={np.mean(rank_ics > 0):.0%}")

    avg_importance = pd.DataFrame(importances_per_fold).mean().sort_values(key=abs, ascending=False)
    label = "coefficient" if args.model == "elasticnet" else "gain importance"
    print(f"\nFeatures ranked by mean {label} across folds:")
    for feat, val in avg_importance.items():
        print(f"  {feat:<20} {val:+.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"model": args.model, "fold_results": fold_results, "feature_importance": avg_importance.to_dict()}
    with open(RESULTS_DIR / f"baseline_{args.model}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {RESULTS_DIR / f'baseline_{args.model}.json'}")


if __name__ == "__main__":
    main()
