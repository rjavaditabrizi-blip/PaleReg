"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): corrects an
observed "best" Sharpe ratio for selection bias from trying multiple
configurations. The more things you test, the more likely *something* looks
good by chance alone, even with zero true skill -- exactly the failure mode
this project already got burned by once (the Sharpe 4.25 result that turned
out to be noise). This project has walk-forward-validated something like a
dozen engine/feature configurations; DSR asks how much of the best one's
Sharpe should actually be trusted after accounting for that search.

Usage:
    python -m src.significance
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GAMMA = 0.5772156649  # Euler-Mascheroni constant


def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark_sr: float = 0.0) -> tuple[float, float]:
    """PSR: probability the TRUE per-period Sharpe exceeds benchmark_sr,
    given the observed return sample. Accounts for skew/kurtosis rather than
    assuming Gaussian returns, which matters -- financial returns rarely are.
    Returns (psr, observed_per_period_sharpe)."""
    returns = np.asarray(returns)
    T = len(returns)
    sr_hat = np.mean(returns) / np.std(returns, ddof=1)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=False)  # Pearson convention: normal distribution = 3
    denom = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat**2)
    z = (sr_hat - benchmark_sr) * np.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z)), float(sr_hat)


def expected_max_sharpe_under_null(trial_sharpes: np.ndarray) -> float:
    """E[max Sharpe observed across N independent trials], assuming zero true
    skill in all of them, using the actual cross-trial Sharpe spread we
    observed as the null variance -- i.e. "given how much our own results
    have bounced around across configurations, how high would the best of
    N such noisy trials look purely by chance?"."""
    trial_sharpes = np.asarray(trial_sharpes)
    n = len(trial_sharpes)
    sigma_sr = np.std(trial_sharpes, ddof=1)
    return float(
        sigma_sr
        * ((1 - GAMMA) * stats.norm.ppf(1 - 1 / n) + GAMMA * stats.norm.ppf(1 - 1 / (n * np.e)))
    )


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: np.ndarray) -> dict:
    """Full DSR: PSR of the selected trial's actual returns against the
    expected max Sharpe you'd see from len(trial_sharpes) trials by luck
    alone. A DSR well above 0.5 means the result likely reflects real skill,
    not just having tried enough configurations."""
    benchmark = expected_max_sharpe_under_null(trial_sharpes)
    psr, sr_hat = probabilistic_sharpe_ratio(returns, benchmark_sr=benchmark)
    return {
        "n_trials": len(trial_sharpes),
        "trial_sharpes": trial_sharpes.tolist() if hasattr(trial_sharpes, "tolist") else list(trial_sharpes),
        "per_period_sharpe_of_selected": sr_hat,
        "expected_max_sharpe_under_null": benchmark,
        "deflated_sharpe_ratio": psr,
    }


PERIODS_PER_YEAR = 252 / 5  # HORIZON = 5 trading days


def load_trial_sharpes() -> dict[str, float]:
    """Every walkforward_results*.json in results/ is one walk-forward-
    validated configuration ("trial") this project actually ran. Returns
    {tag: mean_fold_sharpe}, converted from the annualized units everything
    is normally reported in down to PER-PERIOD Sharpe -- the DSR/PSR
    machinery needs consistent units throughout, and probabilistic_sharpe_
    ratio() computes its sr_hat directly from raw per-period returns, not
    annualized ones. Mixing the two understates DSR by roughly sqrt(periods
    per year) (~7x here) -- confirmed as a real bug during development by
    checking that dividing an annualized trial Sharpe by that factor lands
    in the same ballpark as the per-period Sharpe computed directly from
    returns, before this conversion was added."""
    trials = {}
    for path in sorted(RESULTS_DIR.glob("walkforward_results*.json")):
        with open(path) as f:
            folds = json.load(f)
        sharpes = [r["sharpe"] for r in folds if r.get("sharpe") is not None and not np.isnan(r["sharpe"])]
        if sharpes:
            trials[path.stem] = float(np.mean(sharpes)) / np.sqrt(PERIODS_PER_YEAR)
    return trials


def main():
    trials = load_trial_sharpes()
    # walkforward_results.json (no suffix) is a copy of whichever tagged
    # trial got promoted to production, not an independent trial -- keeping
    # it in the null-distribution estimate would double-count the selected
    # config as if it were also one of the "other things tried".
    trials.pop("walkforward_results", None)
    print(f"Found {len(trials)} independent walk-forward-validated trials (excludes the production promotion, which duplicates one of these):")
    for tag, sr in trials.items():
        print(f"  {tag}: per-period Sharpe {sr:+.4f}  (annualized {sr * np.sqrt(PERIODS_PER_YEAR):+.2f})")

    # The production config is whichever trial file backs the untagged
    # walkforward_results.json / oos_predictions.parquet the app reads.
    prod_path = RESULTS_DIR / "walkforward_results.json"
    with open(prod_path) as f:
        prod_folds = json.load(f)
    period_returns = [r for fold in prod_folds for r in fold.get("period_returns", [])]
    if not period_returns:
        raise SystemExit(
            "No period_returns in walkforward_results.json -- re-run walkforward.py "
            "with the updated backtest.py to populate this before computing DSR."
        )

    result = deflated_sharpe_ratio(np.array(period_returns), np.array(list(trials.values())))

    print(f"\n{'=' * 60}\nDeflated Sharpe Ratio for production config\n{'=' * 60}")
    print(f"Trials considered: {result['n_trials']}")
    print(f"Pooled OOS periods for production: {len(period_returns)}")
    print(f"Per-period Sharpe (production, unannualized): {result['per_period_sharpe_of_selected']:+.4f}")
    print(f"Expected max Sharpe from {result['n_trials']} trials under pure luck: {result['expected_max_sharpe_under_null']:+.4f}")
    print(f"Deflated Sharpe Ratio: {result['deflated_sharpe_ratio']:.3f}")
    print(
        "\n(DSR is a probability, not a Sharpe value -- 0.95 means 95% confidence the true "
        "skill-adjusted Sharpe is above the luck-adjusted benchmark, not a 95% Sharpe.)"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "significance.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {RESULTS_DIR / 'significance.json'}")


if __name__ == "__main__":
    main()
