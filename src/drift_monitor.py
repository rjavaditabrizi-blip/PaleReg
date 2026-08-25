"""Track the REALIZED performance of the currently-deployed model as new
weeks resolve, and flag when it's drifting below what walk-forward
validation showed. This is the operational answer to "markets are
non-stationary, regime shifts happen": don't just hope the complexity
penalty prevents overfitting to a regime that later ends -- watch for the
deployed model to actually stop working, and say so.

Two halves, meant to be called from a weekly job:
1. record_prediction() -- each time predict_today.py runs, save that
   week's scores (outcome isn't known yet, resolves HORIZON days later).
2. resolve_predictions() -- on a later run, fills in realized outcomes for
   any past predictions whose HORIZON window has now passed.

check_drift() then flags if the trailing window's realized IC has fallen
meaningfully below the validated baseline (a one-sided z-test using the
walk-forward fold-to-fold std as the reference variability).

This has never run in production (there's no real weekly history yet) --
demo_replay() validates the logic against the existing walk-forward
out-of-sample data instead of waiting real calendar weeks, treating every
HORIZON-th trading day as one simulated week.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LIVE_LOG_CACHE = RESULTS_DIR / "live_performance_log.csv"


def record_prediction(date, tickers, alpha_scores, log_path: Path = LIVE_LOG_CACHE) -> None:
    """Save one week's live prediction. Outcome isn't known yet."""
    df = pd.DataFrame(
        {"predict_date": date, "ticker": tickers, "alpha": alpha_scores, "realized_fwd_ret": np.nan, "resolved": False}
    )
    header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=header, index=False)


def resolve_predictions(realized_returns: pd.DataFrame, log_path: Path = LIVE_LOG_CACHE) -> pd.DataFrame:
    """realized_returns: DataFrame[date, ticker, fwd_ret]. Fills in outcomes
    for any unresolved past predictions whose date now has a known return."""
    if not log_path.exists():
        return pd.DataFrame()
    log = pd.read_csv(log_path, parse_dates=["predict_date"])
    lookup = realized_returns.rename(columns={"date": "predict_date", "fwd_ret": "_new_ret"})
    merged = log.merge(lookup, on=["predict_date", "ticker"], how="left")
    fill_mask = merged["_new_ret"].notna() & ~merged["resolved"]
    merged.loc[fill_mask, "realized_fwd_ret"] = merged.loc[fill_mask, "_new_ret"]
    merged.loc[fill_mask, "resolved"] = True
    merged = merged.drop(columns=["_new_ret"])
    merged.to_csv(log_path, index=False)
    return merged


def weekly_ic_series(log: pd.DataFrame) -> pd.Series:
    """One realized cross-sectional IC per predict_date, resolved rows only."""
    resolved = log[log["resolved"]]
    if resolved.empty:
        return pd.Series(dtype=float)

    def _corr(g):
        if g["alpha"].std() > 0 and g["realized_fwd_ret"].std() > 0:
            return g["alpha"].corr(g["realized_fwd_ret"])
        return np.nan

    return resolved.groupby("predict_date").apply(_corr, include_groups=False).dropna()


def compute_baseline_stats(oos_path: Path = RESULTS_DIR / "oos_predictions.parquet", horizon: int = 5) -> tuple[float, float]:
    """Mean and std of PERIOD-level (not fold-level) IC from the validated
    walk-forward OOS data -- must match the granularity of whatever
    ic_series check_drift() is called with, or the standard-error math is
    wrong. Fold-level IC std (as reported in walkforward.py's summary,
    e.g. 0.0127) is already averaged over ~59 periods per fold and is
    roughly 10x smaller than the true period-level std (~0.13) -- using it
    here understates the standard error by that factor and makes the
    monitor flag drift on nearly half of all normal weeks, confirmed
    empirically before this function existed."""
    oos = pd.read_parquet(oos_path)
    unique_dates = np.sort(oos["date"].unique())
    ics = []
    for d in unique_dates[::horizon]:
        day = oos[oos["date"] == d]
        if day["alpha"].std() > 0 and day["fwd_ret"].std() > 0:
            ics.append(day["alpha"].corr(day["fwd_ret"]))
    ics = np.array(ics)
    return float(ics.mean()), float(ics.std(ddof=0))


def check_drift(ic_series: pd.Series, baseline_mean: float, baseline_std: float, window: int = 8) -> dict:
    """Flags if the trailing `window` weeks' mean realized IC is
    significantly below the validated baseline. p_value is P(seeing a mean
    this low or lower | the baseline is still true) -- small means the
    recent run is surprising under "nothing changed". baseline_mean/std
    MUST be period-level (see compute_baseline_stats), not the fold-level
    numbers walkforward.py prints."""
    recent = ic_series.tail(window)
    if len(recent) < 3:
        return {"status": "insufficient_data", "n_weeks": int(len(recent))}
    recent_mean = float(recent.mean())
    se = baseline_std / np.sqrt(len(recent))
    z = (recent_mean - baseline_mean) / se
    p_value = float(stats.norm.cdf(z))
    return {
        "status": "drift_flagged" if p_value < 0.10 else "normal",
        "n_weeks": int(len(recent)),
        "recent_mean_ic": recent_mean,
        "baseline_mean_ic": baseline_mean,
        "z_score": float(z),
        "p_value": p_value,
    }


def demo_replay(oos_path: Path = RESULTS_DIR / "oos_predictions.parquet", horizon: int = 5, window: int = 8) -> dict:
    """Validates the monitor's mechanics using existing walk-forward OOS data
    instead of waiting real weeks: replays every `horizon`-th trading day as
    one simulated week's predict+resolve cycle, and reports what the drift
    monitor would have said at each point in that history.

    Caveat this demo can't avoid: baseline stats are computed from the same
    series being monitored (there's no separate pre-deployment validation
    period to draw them from yet, since this has never run live). A real
    deployment must compute baseline_mean/std once from walk-forward
    validation BEFORE going live, then hold them fixed while monitoring
    genuinely new weeks -- comparing a series against its own baseline, as
    this demo necessarily does, will understate how often true drift would
    get caught against an independently-set baseline."""
    baseline_mean, baseline_std = compute_baseline_stats(oos_path, horizon)

    oos = pd.read_parquet(oos_path)
    unique_dates = np.sort(oos["date"].unique())
    sim_dates = unique_dates[::horizon]

    rows = []
    for d in sim_dates:
        day = oos[oos["date"] == d]
        if day["alpha"].std() == 0 or day["fwd_ret"].std() == 0:
            continue
        rows.append({"predict_date": d, "ic": day["alpha"].corr(day["fwd_ret"])})
    ic_series = pd.Series([r["ic"] for r in rows], index=[r["predict_date"] for r in rows])

    timeline = [
        {"as_of_week": i, **check_drift(ic_series.iloc[:i], baseline_mean, baseline_std, window=window)}
        for i in range(window, len(ic_series) + 1)
    ]
    return {
        "n_simulated_weeks": len(ic_series), "ic_series": ic_series, "drift_timeline": timeline,
        "baseline_mean": baseline_mean, "baseline_std": baseline_std,
    }


def main():
    result = demo_replay()
    print(f"Baseline (period-level, from full OOS history): mean IC={result['baseline_mean']:+.4f} std={result['baseline_std']:.4f}")
    print(f"Replayed {result['n_simulated_weeks']} simulated weeks from oos_predictions.parquet\n")
    print(f"{'as_of_week':>10} {'status':>14} {'recent_IC':>10} {'z':>7} {'p_value':>8}")
    for row in result["drift_timeline"]:
        if row["status"] == "insufficient_data":
            continue
        print(
            f"{row['as_of_week']:>10} {row['status']:>14} {row['recent_mean_ic']:>+10.4f} "
            f"{row['z_score']:>7.2f} {row['p_value']:>8.3f}"
        )
    n_flagged = sum(1 for r in result["drift_timeline"] if r["status"] == "drift_flagged")
    print(f"\n{n_flagged}/{len(result['drift_timeline'])} simulated weeks would have flagged drift.")


if __name__ == "__main__":
    main()
