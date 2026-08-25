"""Second-stage search: feed the current production model's validated
formulas back in as NEW input features for a follow-up SR round, letting
the search discover interaction effects between already-proven signals
instead of only ever combining raw OHLCV terms directly. This is the
"alpha generation" idea from the shared quant-literature summary taken one
step further -- SR mining its own prior output, not just raw features.

Usage:
    python -m src.second_stage --test-frac 0.2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .fitness import mean_ic, mean_rank_ic
from .predict_today import load_model
from .train import mine_pool, time_split
from .engines.pysr_engine import mine_pool_pysr_multiseed

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_CACHE = DATA_DIR / "features.parquet"

NON_FEATURE_COLS = {"date", "ticker", "sector", "open", "high", "low", "close", "volume", "fwd_ret", "fwd_ret_neutral"}


def add_formula_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Evaluates the production model's mined formulas on df and adds them
    as new columns (prefixed 'meta_'), cross-sectionally z-scored the same
    way every other operand column is -- so the second-stage search sees
    them on equal footing with the raw OHLCV-derived features."""
    programs, model_feature_cols, weights, formulas = load_model()
    X = df[model_feature_cols].values

    df = df.copy()
    new_cols = []
    for i, program in enumerate(programs):
        col = f"meta_formula_{i}"
        df[col] = program.execute(X)
        new_cols.append(col)

    date_g = df.groupby("date", sort=False)[new_cols]
    means = date_g.transform("mean")
    stds = date_g.transform("std").replace(0, np.nan)
    df[new_cols] = ((df[new_cols] - means) / stds).fillna(0.0)

    return df, new_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--engine", choices=["gplearn", "pysr"], default="pysr")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--niterations", type=int, default=20)
    parser.add_argument("--populations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=33)
    parser.add_argument("--maxsize", type=int, default=20)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--batch-days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_CACHE)
    base_feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    df, meta_cols = add_formula_features(df)
    all_feature_cols = base_feature_cols + meta_cols
    print(f"Base features: {len(base_feature_cols)}, meta (formula) features added: {len(meta_cols)}, total: {len(all_feature_cols)}")

    train_df, test_df, cutoff = time_split(df, args.test_frac)
    X_train, y_train, dates_train = train_df[all_feature_cols].values, train_df["fwd_ret"].values, train_df["date"].values
    X_test, y_test, dates_test = test_df[all_feature_cols].values, test_df["fwd_ret"].values, test_df["date"].values
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | cutoff: {cutoff}")

    pool_train, pool_test, formulas, _models, y_train, dates_train = mine_pool_pysr_multiseed(
        X_train, X_test, y_train, dates_train, all_feature_cols,
        n_seeds=args.n_seeds, seed=args.seed,
        niterations=args.niterations, populations=args.populations,
        population_size=args.population_size, maxsize=args.maxsize,
        procs=args.procs, batch_days=args.batch_days,
    )

    print(f"\nSecond-stage mined factors ({len(formulas)} unique):")
    uses_meta = 0
    for i, formula_str in enumerate(formulas):
        train_ic = mean_ic(pool_train[:, i], y_train, dates_train)
        test_ic = mean_ic(pool_test[:, i], y_test, dates_test) if len(y_test) else float("nan")
        uses_meta_feature = "meta_formula" in formula_str
        uses_meta += int(uses_meta_feature)
        tag = "  [uses a first-stage formula]" if uses_meta_feature else ""
        print(f"[{i}] IC(train)={train_ic:+.4f} IC(test)={test_ic:+.4f}  {formula_str}{tag}")

    print(f"\n{uses_meta}/{len(formulas)} second-stage formulas actually use a first-stage formula as an ingredient.")
    print(
        "If that number is 0, the search found nothing worth building on top of the existing "
        "signal -- the base features already cover what's discoverable. If it's >0, those "
        "formulas are testing genuine second-order interaction effects, worth a full "
        "walk-forward validation before considering production."
    )


if __name__ == "__main__":
    main()
