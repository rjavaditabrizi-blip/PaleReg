"""Expanding-window walk-forward fold boundaries. Standalone (no
dependencies on the rest of the package) so it can be imported by
walkforward.py, feature_screen.py, and preflight.py without creating an
import cycle between them."""
import numpy as np


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
