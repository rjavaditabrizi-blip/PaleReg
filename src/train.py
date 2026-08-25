"""Mine a pool of diverse alpha-factor formulas via genetic-programming
symbolic regression (gplearn's SymbolicTransformer), scored by mean daily IC
against forward returns.

This is the "factor zoo" stage from AlphaForge / the Symbolic Modeling
framing: one fixed operator/operand set, searched once, producing several
diverse interpretable formulas rather than a single black-box fit.

A single big SymbolicTransformer run tends to collapse: gplearn's default
crossover rate (0.9) means 90% of each new generation is bred from whichever
tree currently wins, so the population homogenizes around one formula within
a couple generations and its own hall-of-fame diversity selection has nothing
diverse left to pick from. Two independent countermeasures are applied here:
(1) crossover is turned down and mutation rates turned up so novel structure
keeps entering the population, and (2) --n-seeds runs several independent,
smaller searches and pools their distinct top formulas -- diversity by
construction rather than by hoping one run's selection saves it.

Usage:
    python -m src.train --n-seeds 8 --population 500 --generations 25 --n-components 5
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer

from .engines.pysr_engine import mine_pool_pysr_multiseed
from .fitness import make_ic_fitness, make_rank_ic_fitness, mean_ic, mean_rank_ic
from .preflight import resolve_feature_subset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_CACHE = DATA_DIR / "features.parquet"
POOL_CACHE = RESULTS_DIR / "factor_pool.npz"
FORMULAS_CACHE = RESULTS_DIR / "formulas.json"
PROGRAMS_CACHE = RESULTS_DIR / "programs.pkl"

DEFAULT_FUNCTION_SET = ("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv", "max", "min")


def time_split(df: pd.DataFrame, test_frac: float):
    dates = np.sort(df["date"].unique())
    if test_frac <= 0:
        # "Production" mode: train on everything, no held-out test. Only use this
        # for the final live-prediction model -- research validation should
        # always go through walkforward.py, never this.
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True), dates[-1]
    cutoff = dates[int(len(dates) * (1 - test_frac))]
    train_df = df[df["date"] < cutoff].reset_index(drop=True)
    test_df = df[df["date"] >= cutoff].reset_index(drop=True)
    return train_df, test_df, cutoff


def run_one_seed(seed, X_train, X_test, y_train, dates_train, feature_cols, args):
    ic_fitness = make_rank_ic_fitness(dates_train) if args.metric == "spearman" else make_ic_fitness(dates_train)
    transformer = SymbolicTransformer(
        population_size=args.population,
        generations=args.generations,
        n_components=args.n_components,
        hall_of_fame=args.hall_of_fame,
        function_set=DEFAULT_FUNCTION_SET,
        metric=ic_fitness,
        parsimony_coefficient=args.parsimony,
        tournament_size=args.tournament_size,
        p_crossover=args.p_crossover,
        p_subtree_mutation=args.p_subtree_mutation,
        p_hoist_mutation=args.p_hoist_mutation,
        p_point_mutation=args.p_point_mutation,
        max_samples=1.0,  # must stay 1.0 -- see fitness.py docstring
        low_memory=True,
        n_jobs=args.n_jobs,
        feature_names=feature_cols,
        random_state=seed,
        verbose=1,
    )
    transformer.fit(X_train, y_train)
    pool_train = transformer.transform(X_train)
    pool_test = transformer.transform(X_test) if X_test.shape[0] > 0 else np.empty((0, args.n_components))
    programs = list(transformer)
    formulas = [str(program) for program in programs]
    return pool_train, pool_test, formulas, programs


def mine_pool(X_train, X_test, y_train, dates_train, feature_cols, args, verbose=True):
    """Run --n-seeds independent GP searches and pool their distinct formulas.
    Returns (pool_train, pool_test, formulas, programs), all deduped and
    aligned with the pool columns. `programs` are the actual gplearn _Program
    objects (picklable) -- keep these around if you want to evaluate the
    mined formulas on new data later without re-parsing formula strings."""
    all_pool_train, all_pool_test, all_formulas, all_programs = [], [], [], []
    for i in range(args.n_seeds):
        seed = args.seed + i
        if verbose:
            print(f"\n=== Seed {i + 1}/{args.n_seeds} (random_state={seed}) ===")
        pool_train, pool_test, formulas, programs = run_one_seed(
            seed, X_train, X_test, y_train, dates_train, feature_cols, args
        )
        all_pool_train.append(pool_train)
        all_pool_test.append(pool_test)
        all_formulas.extend(formulas)
        all_programs.extend(programs)

    pool_train_raw = np.hstack(all_pool_train)
    pool_test_raw = np.hstack(all_pool_test) if X_test.shape[0] > 0 else np.empty((0, len(all_formulas)))

    seen = {}
    for i, f in enumerate(all_formulas):
        seen.setdefault(f, i)  # keep first occurrence's column index
    keep_idx = sorted(seen.values())
    if verbose:
        print(f"\nMined {len(all_formulas)} formulas across {args.n_seeds} seeds, {len(keep_idx)} unique.")

    pool_train = pool_train_raw[:, keep_idx]
    pool_test = pool_test_raw[:, keep_idx]
    formulas = [all_formulas[i] for i in keep_idx]
    programs = [all_programs[i] for i in keep_idx]
    return pool_train, pool_test, formulas, programs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["gplearn", "pysr"], default="gplearn")
    # pysr-specific (validated as the production engine via walkforward.py: 3-seed
    # day-aware-batched PySR beat gplearn on every metric -- Sharpe +0.29 vs +0.21
    # with lower variance, IC +0.0218 vs +0.0206, both at 100% fold hit-rate)
    parser.add_argument("--pysr-n-seeds", type=int, default=3)
    parser.add_argument("--pysr-niterations", type=int, default=20)
    parser.add_argument("--pysr-populations", type=int, default=8)
    parser.add_argument("--pysr-population-size", type=int, default=33)
    parser.add_argument("--pysr-maxsize", type=int, default=20)
    parser.add_argument("--pysr-procs", type=int, default=8)
    parser.add_argument("--pysr-batch-days", type=int, default=60)
    # gplearn-specific
    parser.add_argument("--population", type=int, default=500, help="Population size PER SEED")
    parser.add_argument("--generations", type=int, default=25, help="Generations PER SEED")
    parser.add_argument("--n-components", type=int, default=5, help="Diverse formulas kept PER SEED")
    parser.add_argument("--hall-of-fame", type=int, default=50, help="Hall of fame size PER SEED")
    parser.add_argument("--n-seeds", type=int, default=8, help="Independent GP searches to pool together")
    parser.add_argument("--parsimony", type=float, default=0.001)
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=20,
        help="Lower = weaker selection pressure = more population diversity (gplearn default 20)",
    )
    parser.add_argument("--p-crossover", type=float, default=0.7)
    parser.add_argument("--p-subtree-mutation", type=float, default=0.1)
    parser.add_argument("--p-hoist-mutation", type=float, default=0.05)
    parser.add_argument("--p-point-mutation", type=float, default=0.1)
    parser.add_argument(
        "--metric",
        choices=["pearson", "spearman"],
        default="spearman",
        help="Fitness metric for the GP search: mean daily Pearson IC or Rank IC. "
        "Papers favor Rank IC for robustness to outliers.",
    )
    parser.add_argument("--test-frac", type=float, default=0.2, help="Set to 0 for a production run (train on all data, no held-out test)")
    parser.add_argument(
        "--target",
        choices=["raw", "neutral"],
        default="raw",
        help="fwd_ret (raw, default) or fwd_ret_neutral (sector-demeaned). Walk-forward "
        "validation showed 'neutral' gutted the signal (mean IC +0.0206 -> +0.0066) because "
        "part of the real edge is a volume/attention effect that's inherently sector-correlated "
        "-- sector risk should be capped at the portfolio-construction step instead (see "
        "predict_today.py --max-per-sector), not baked into the mining target.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--feature-subset", default=None,
        help="'preflight' to use the last `python -m src.preflight --top-n N` recommendation, "
        "or a comma-separated feature list. Default: use all features. This is a "
        "prioritization from cheap baselines, not a validated exclusion -- some of SR's best "
        "mined formulas use features that scored low standalone, so only restrict when you "
        "specifically want a faster, narrower search.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed; seeds used are seed, seed+1, ...")
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_CACHE)
    feature_cols = [
        c
        for c in df.columns
        if c not in {"date", "ticker", "sector", "open", "high", "low", "close", "volume", "fwd_ret", "fwd_ret_neutral"}
    ]
    feature_cols = resolve_feature_subset(args.feature_subset, feature_cols)
    target_col = "fwd_ret" if args.target == "raw" else "fwd_ret_neutral"

    train_df, test_df, cutoff = time_split(df, args.test_frac)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | cutoff date: {cutoff} | target: {target_col}")

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    dates_train = train_df["date"].values
    tickers_train = train_df["ticker"].values
    sectors_train = train_df["sector"].values
    raw_y_train = train_df["fwd_ret"].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    dates_test = test_df["date"].values
    tickers_test = test_df["ticker"].values
    sectors_test = test_df["sector"].values
    raw_y_test = test_df["fwd_ret"].values

    if args.engine == "gplearn":
        pool_train, pool_test, deduped_formulas, programs = mine_pool(
            X_train, X_test, y_train, dates_train, feature_cols, args
        )
    else:
        pool_train, pool_test, deduped_formulas, programs, y_train, dates_train = mine_pool_pysr_multiseed(
            X_train, X_test, y_train, dates_train, feature_cols,
            n_seeds=args.pysr_n_seeds,
            seed=args.seed,
            niterations=args.pysr_niterations,
            populations=args.pysr_populations,
            population_size=args.pysr_population_size,
            maxsize=args.pysr_maxsize,
            procs=args.pysr_procs,
            batch_days=args.pysr_batch_days,
        )

    formulas = []
    print("\nMined factors (deduped):")
    for i, formula_str in enumerate(deduped_formulas):
        train_ic = mean_ic(pool_train[:, i], y_train, dates_train)
        test_ic = mean_ic(pool_test[:, i], y_test, dates_test) if len(y_test) else float("nan")
        train_rank_ic = mean_rank_ic(pool_train[:, i], y_train, dates_train)
        test_rank_ic = mean_rank_ic(pool_test[:, i], y_test, dates_test) if len(y_test) else float("nan")
        formulas.append(
            {
                "index": i,
                "formula": formula_str,
                "train_ic": train_ic,
                "test_ic": test_ic,
                "train_rank_ic": train_rank_ic,
                "test_rank_ic": test_rank_ic,
            }
        )
        print(
            f"[{i}] IC(train)={train_ic:+.4f} IC(test)={test_ic:+.4f} "
            f"RankIC(train)={train_rank_ic:+.4f} RankIC(test)={test_rank_ic:+.4f}  {formula_str}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        POOL_CACHE,
        pool_train=pool_train,
        pool_test=pool_test,
        y_train=y_train,
        y_test=y_test,
        raw_y_train=raw_y_train,
        raw_y_test=raw_y_test,
        dates_train=dates_train,
        dates_test=dates_test,
        tickers_train=tickers_train,
        tickers_test=tickers_test,
        sectors_train=sectors_train,
        sectors_test=sectors_test,
    )
    with open(FORMULAS_CACHE, "w") as f:
        json.dump(formulas, f, indent=2, default=str)
    with open(PROGRAMS_CACHE, "wb") as f:
        pickle.dump({"programs": programs, "feature_cols": feature_cols}, f)

    print(f"\nSaved factor pool -> {POOL_CACHE}")
    print(f"Saved formulas -> {FORMULAS_CACHE}")
    print(f"Saved programs -> {PROGRAMS_CACHE}")


if __name__ == "__main__":
    main()
