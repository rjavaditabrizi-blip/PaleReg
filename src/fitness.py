"""Information-Coefficient fitness for gplearn.

Financial alpha factors are judged by cross-sectional rank correlation with
forward returns (IC / Rank IC), not by raw fit error (see AlphaForge Eq. 1 and
AlphaFormer Sec 2.2) -- fitting MSE directly tends to chase noise. This module
builds a gplearn-compatible fitness function that computes the average daily
Pearson IC between a candidate factor and forward returns.

Caveat: gplearn's fitness function only receives flat (y, y_pred, weight)
arrays, not the row dates. We close over a fixed `dates` array aligned to the
training row order. This only stays valid if gplearn evaluates every program
on the full, unshuffled training set -- i.e. keep `max_samples=1.0` in
SymbolicRegressor/SymbolicTransformer. If you want row subsampling later,
you'll need to subsample the dates array in lockstep.
"""
import numpy as np
import pandas as pd
from gplearn.fitness import make_fitness


def _day_corr(g: pd.DataFrame) -> float:
    if g["pred"].std(ddof=0) < 1e-12 or g["y"].std(ddof=0) < 1e-12:
        return 0.0
    return float(np.corrcoef(g["pred"], g["y"])[0, 1])


def _day_rank_corr(g: pd.DataFrame) -> float:
    """Spearman rank correlation for one day's cross-section (Rank IC). More
    robust to outliers than Pearson IC -- see AlphaForge / AlphaFormer, both
    of which report Rank IC alongside IC for this reason."""
    if g["pred"].nunique() < 2 or g["y"].nunique() < 2:
        return 0.0
    return float(np.corrcoef(g["pred"].rank(), g["y"].rank())[0, 1])


def _global_corr(y: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(y, y_pred)[0, 1])


def make_ic_fitness(dates):
    """Build a gplearn fitness callable that scores mean daily IC."""
    dates = np.asarray(dates)

    def _ic(y, y_pred, sample_weight):
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        if len(y) != len(dates):
            # Fallback if row count doesn't match (e.g. max_samples < 1.0).
            return _global_corr(y, y_pred)
        df = pd.DataFrame({"y": y, "pred": y_pred, "date": dates})
        ics = df.groupby("date", sort=False).apply(_day_corr, include_groups=False)
        if ics.empty:
            return 0.0
        return float(np.nan_to_num(ics.mean()))

    return make_fitness(function=_ic, greater_is_better=True)


def make_rank_ic_fitness(dates):
    """Build a gplearn fitness callable that scores mean daily Rank IC (Spearman)."""
    dates = np.asarray(dates)

    def _rank_ic(y, y_pred, sample_weight):
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        if len(y) != len(dates):
            return _global_corr(pd.Series(y_pred).rank().values, pd.Series(y).rank().values)
        df = pd.DataFrame({"y": y, "pred": y_pred, "date": dates})
        ics = df.groupby("date", sort=False).apply(_day_rank_corr, include_groups=False)
        if ics.empty:
            return 0.0
        return float(np.nan_to_num(ics.mean()))

    return make_fitness(function=_rank_ic, greater_is_better=True)


def mean_ic(pred: np.ndarray, y: np.ndarray, dates) -> float:
    """Scalar mean daily IC, for reporting on held-out data."""
    df = pd.DataFrame({"y": np.asarray(y), "pred": np.asarray(pred), "date": np.asarray(dates)})
    ics = df.groupby("date", sort=False).apply(_day_corr, include_groups=False)
    return float(np.nan_to_num(ics.mean())) if not ics.empty else 0.0


def mean_rank_ic(pred: np.ndarray, y: np.ndarray, dates) -> float:
    """Scalar mean daily Rank IC, for reporting on held-out data."""
    df = pd.DataFrame({"y": np.asarray(y), "pred": np.asarray(pred), "date": np.asarray(dates)})
    ics = df.groupby("date", sort=False).apply(_day_rank_corr, include_groups=False)
    return float(np.nan_to_num(ics.mean())) if not ics.empty else 0.0


def ic_series(pred: np.ndarray, y: np.ndarray, dates) -> pd.Series:
    """Per-day IC series, for plotting/inspection."""
    df = pd.DataFrame({"y": np.asarray(y), "pred": np.asarray(pred), "date": np.asarray(dates)})
    return df.groupby("date", sort=True).apply(_day_corr, include_groups=False)
