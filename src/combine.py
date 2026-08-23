"""Combine the mined factor pool into a single composite alpha ("Mega-Alpha"),
mirroring AlphaForge's combination stage: a sparse linear combination of
factors, optionally re-fit periodically over an expanding window so weights
can drift as factors go in and out of favor.

Usage:
    python -m src.combine --mode static
    python -m src.combine --mode walk-forward --refit-every 21
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso

from .fitness import mean_ic

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
POOL_CACHE = RESULTS_DIR / "factor_pool.npz"
FORMULAS_CACHE = RESULTS_DIR / "formulas.json"
WEIGHTS_CACHE = RESULTS_DIR / "combo_weights.json"
ALPHA_CACHE = RESULTS_DIR / "composite_alpha.npz"


def static_combine(pool_train, y_train, pool_test, y_test, dates_train, dates_test, alpha):
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000)
    model.fit(pool_train, y_train)
    train_pred = model.predict(pool_train)
    train_ic = mean_ic(train_pred, y_train, dates_train)
    if pool_test.shape[0] == 0:
        return model.coef_, train_ic, float("nan"), np.empty(0)
    test_pred = model.predict(pool_test)
    test_ic = mean_ic(test_pred, y_test, dates_test)
    return model.coef_, train_ic, test_ic, test_pred


def walk_forward_combine(pool_train, y_train, dates_train, pool_test, y_test, dates_test, alpha, refit_every):
    """Expanding-window refit: at each step, fit on all data up to a cutoff
    date and score the next `refit_every`-day chunk out-of-sample."""
    pool_all = np.concatenate([pool_train, pool_test])
    y_all = np.concatenate([y_train, y_test])
    dates_all = np.concatenate([dates_train, dates_test])

    unique_dates = np.sort(np.unique(dates_all))
    n_train_days = len(np.unique(dates_train))

    preds = np.full(len(y_all), np.nan)
    last_weights = None
    for start in range(n_train_days, len(unique_dates), refit_every):
        train_cutoff_date = unique_dates[start - 1]
        chunk_dates = unique_dates[start : start + refit_every]
        if len(chunk_dates) == 0:
            continue
        fit_mask = dates_all <= train_cutoff_date
        chunk_mask = np.isin(dates_all, chunk_dates)
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=20000)
        model.fit(pool_all[fit_mask], y_all[fit_mask])
        preds[chunk_mask] = model.predict(pool_all[chunk_mask])
        last_weights = model.coef_

    valid = ~np.isnan(preds)
    oos_ic = mean_ic(preds[valid], y_all[valid], dates_all[valid])
    test_pred = preds[len(y_train):]
    return last_weights, oos_ic, test_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1e-4, help="Lasso regularization strength")
    parser.add_argument("--mode", choices=["static", "walk-forward"], default="static")
    parser.add_argument("--refit-every", type=int, default=21, help="Trading days between refits (walk-forward mode)")
    args = parser.parse_args()

    data = np.load(POOL_CACHE, allow_pickle=True)
    with open(FORMULAS_CACHE) as f:
        formulas = json.load(f)

    if args.mode == "static":
        weights, train_ic, test_ic, test_pred = static_combine(
            data["pool_train"], data["y_train"], data["pool_test"], data["y_test"],
            data["dates_train"], data["dates_test"], args.alpha,
        )
        print(f"Static composite alpha: IC(train)={train_ic:+.4f} IC(test)={test_ic:+.4f}")
    else:
        weights, oos_ic, test_pred = walk_forward_combine(
            data["pool_train"], data["y_train"], data["dates_train"],
            data["pool_test"], data["y_test"], data["dates_test"],
            args.alpha, args.refit_every,
        )
        print(f"Walk-forward composite alpha: IC(oos, all refits)={oos_ic:+.4f}")

    print("\nFactor weights:")
    for f, w in zip(formulas, weights):
        if abs(w) < 1e-10:
            continue
        print(f"  w={w:+.4f}  [{f['index']}] {f['formula']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_CACHE, "w") as fh:
        json.dump({"mode": args.mode, "alpha": args.alpha, "weights": weights.tolist()}, fh, indent=2)

    extra = {}
    if "raw_y_test" in data:
        extra["raw_y_test"] = data["raw_y_test"]
    if "sectors_test" in data:
        extra["sectors_test"] = data["sectors_test"]

    np.savez(
        ALPHA_CACHE,
        test_pred=test_pred,
        y_test=data["y_test"],
        dates_test=data["dates_test"],
        tickers_test=data["tickers_test"],
        **extra,
    )
    print(f"\nSaved weights -> {WEIGHTS_CACHE}")
    print(f"Saved composite test-set alpha -> {ALPHA_CACHE}")


if __name__ == "__main__":
    main()
