"""PySR-backed alternative to gplearn's SymbolicTransformer (src/train.py).

Motivation: across every gplearn run this project has done, the search
repeatedly collapsed to 1-3 near-duplicate formulas regardless of tuning
(crossover/mutation rates, tournament size, multi-seed ensembling). PySR's
search (SymbolicRegression.jl) is generally more capable of avoiding this
kind of premature convergence. This module lets train.py/walkforward.py use
it as a drop-in alternative search backend.

The tricky part: PySR's built-in losses operate on the pooled dataset, with
no notion of "day" -- a plain MSE/correlation loss would let the search
exploit day-to-day market-wide return swings instead of genuine
cross-sectional stock-picking skill (exactly the confound gplearn's custom
per-day-grouped IC fitness was built to avoid). This is worked around by
smuggling a per-day group id through PySR's `weights` argument (repurposed,
not used as real sample weights) and writing a custom Julia loss function
that groups by it internally to compute the same mean-daily-Pearson-IC
gplearn's fitness.py computes -- verified against synthetic data with a
large day-level confound before being used on the real pipeline.

Day-aware batching: the first version of this pre-filtered the training set
down to a small fixed set of random days in Python before ever calling PySR,
so the search only ever saw that one small slice of history for the whole
run -- and a head-to-head walk-forward test showed this made results *less*
consistent than gplearn (worse hit-rate, ~2x the IC variance), plausibly
because gplearn always trains on the full expanding-window history. Instead,
the loss function itself now samples a *fresh* random subset of whole
trading days on every single evaluation (every candidate tree, every
generation) while PySR is given the FULL training set. This mirrors
ordinary minibatch SGD: any one evaluation only pays for `batch_days` worth
of correlation math, but across thousands of evaluations over a run the
search sees most of the actual training history, rather than being
permanently restricted to one small slice chosen once at the start.
"""
import numpy as np
import pandas as pd

# `weights` (arg 2 to eval_tree_array's dataset) carries day-group ids here,
# not real per-row weights -- see module docstring. Manually mirrors
# fitness.py's _day_corr, in Julia, using only Base (no Statistics/Random
# imports, since it's not guaranteed those are in scope where this string
# gets evaluated inside SymbolicRegression.jl -- rand() alone IS part of
# Base and always available without any `using`).
_GROUPED_IC_LOSS_TEMPLATE = """
function eval_loss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    y = dataset.y
    g = dataset.weights
    n = dataset.n
    groups = Dict{{T, Vector{{Int}}}}()
    for i in 1:n
        gid = g[i]
        if haskey(groups, gid)
            push!(groups[gid], i)
        else
            groups[gid] = Int[i]
        end
    end

    all_days = collect(keys(groups))
    n_days = length(all_days)
    batch_days = min({batch_days}, n_days)
    keep_prob = batch_days / n_days

    total = zero(L)
    count = 0
    for gid in all_days
        if rand() >= keep_prob
            continue
        end
        idxs = groups[gid]
        m = length(idxs)
        if m < 2
            continue
        end
        p = prediction[idxs]
        yy = y[idxs]
        mp = sum(p) / m
        my = sum(yy) / m
        covpy = zero(L)
        varp = zero(L)
        vary = zero(L)
        for j in 1:m
            dp = p[j] - mp
            dy = yy[j] - my
            covpy += dp * dy
            varp += dp * dp
            vary += dy * dy
        end
        if varp < L(1e-12) || vary < L(1e-12)
            continue
        end
        c = covpy / sqrt(varp * vary)
        total += c
        count += 1
    end
    if count == 0
        return L(0)
    end
    return L(-(total / count))
end
"""

DEFAULT_BINARY_OPERATORS = ["+", "-", "*", "/"]
DEFAULT_UNARY_OPERATORS = ["sqrt", "log", "square", "abs"]


class PySRFormulaModel:
    """Wraps one row of a fitted PySRRegressor's Pareto front. Exposes
    .execute(X) to match gplearn's _Program naming, so predict_today.py can
    treat models from either engine the same way."""

    def __init__(self, fitted_model, index: int, formula_str: str):
        self._model = fitted_model
        self._index = index
        self._formula_str = formula_str

    def execute(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(X, index=self._index))

    def __str__(self) -> str:
        return self._formula_str


def mine_pool_pysr(
    X_train, X_test, y_train, dates_train, feature_cols,
    niterations: int = 40,
    populations: int = 15,
    population_size: int = 33,
    maxsize: int = 20,
    procs: int = 4,
    random_state: int = 42,
    tempdir: str | None = None,
    batch_days: int = 60,
):
    """PySR equivalent of train.mine_pool. Returns (pool_train, pool_test,
    formulas, models, y_train, dates_train) -- the same pool/formulas/models
    shape gplearn's mine_pool returns (y_train/dates_train are echoed back
    unchanged here since this version doesn't pre-filter them -- kept for a
    consistent call signature with any future engine that does subsample).

    batch_days: how many whole trading days the loss function samples fresh
    on every single evaluation (see module docstring). The FULL X_train is
    still given to PySR -- this only bounds the per-evaluation cost, not how
    much history the search can eventually see.
    """
    from pysr import PySRRegressor

    day_ids = pd.factorize(dates_train)[0].astype(np.float64)
    loss_fn = _GROUPED_IC_LOSS_TEMPLATE.format(batch_days=batch_days)

    kwargs = dict(
        niterations=niterations,
        binary_operators=DEFAULT_BINARY_OPERATORS,
        unary_operators=DEFAULT_UNARY_OPERATORS,
        maxsize=maxsize,
        populations=populations,
        population_size=population_size,
        loss_function=loss_fn,
        loss_scale="linear",
        parallelism="multithreading",
        procs=procs,
        random_state=random_state,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    if tempdir is not None:
        kwargs["tempdir"] = tempdir

    model = PySRRegressor(**kwargs)
    model.fit(X_train, y_train, weights=day_ids, variable_names=list(feature_cols))

    equations = model.equations_
    pool_train_cols, pool_test_cols, formulas, models = [], [], [], []
    for idx, row in equations.iterrows():
        formula_str = str(row["equation"])
        wrapped = PySRFormulaModel(model, idx, formula_str)
        pool_train_cols.append(wrapped.execute(X_train))
        if X_test.shape[0] > 0:
            pool_test_cols.append(wrapped.execute(X_test))
        formulas.append(formula_str)
        models.append(wrapped)

    pool_train = np.column_stack(pool_train_cols)
    pool_test = (
        np.column_stack(pool_test_cols) if X_test.shape[0] > 0 else np.empty((0, len(formulas)))
    )
    return pool_train, pool_test, formulas, models, y_train, dates_train


def mine_pool_pysr_multiseed(
    X_train, X_test, y_train, dates_train, feature_cols,
    n_seeds: int = 4,
    seed: int = 42,
    verbose: bool = True,
    **kwargs,
):
    """Run --n-seeds independent PySR searches and pool their distinct
    formulas, mirroring train.mine_pool's multi-seed gplearn ensembling.
    Since mine_pool_pysr no longer subsamples rows in Python (day-aware
    batching happens inside the Julia loss instead -- see module docstring),
    every seed's pool_train has the same row count as the original X_train,
    so pools from different seeds can be concatenated directly without the
    row-alignment risk a per-seed subsample would introduce."""
    all_pool_train, all_pool_test, all_formulas, all_models = [], [], [], []
    for i in range(n_seeds):
        s = seed + i
        if verbose:
            print(f"\n=== PySR seed {i + 1}/{n_seeds} (random_state={s}) ===")
        pool_train, pool_test, formulas, models, y_train_used, dates_train_used = mine_pool_pysr(
            X_train, X_test, y_train, dates_train, feature_cols, random_state=s, **kwargs
        )
        all_pool_train.append(pool_train)
        all_pool_test.append(pool_test)
        all_formulas.extend(formulas)
        all_models.extend(models)

    pool_train_raw = np.hstack(all_pool_train)
    pool_test_raw = (
        np.hstack(all_pool_test) if X_test.shape[0] > 0 else np.empty((0, len(all_formulas)))
    )

    seen = {}
    for i, f in enumerate(all_formulas):
        seen.setdefault(f, i)
    keep_idx = sorted(seen.values())
    if verbose:
        print(f"\nMined {len(all_formulas)} formulas across {n_seeds} PySR seeds, {len(keep_idx)} unique.")

    pool_train = pool_train_raw[:, keep_idx]
    pool_test = pool_test_raw[:, keep_idx]
    formulas = [all_formulas[i] for i in keep_idx]
    models = [all_models[i] for i in keep_idx]
    return pool_train, pool_test, formulas, models, y_train_used, dates_train_used
