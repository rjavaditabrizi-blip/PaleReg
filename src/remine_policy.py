"""Decision policy for when to re-mine (expensive, ~1-2 hours) vs. just
re-score with the currently-deployed model (predict_today.py, seconds).

Full SR search doesn't need to run every week -- only when there's reason
to think the deployed formula has stopped working, or it's been long enough
that re-validating from scratch is warranted regardless. This encodes that
as an explicit, checkable policy instead of an ad hoc judgment call each
time, and persists the last-mine date so the policy survives between runs.

Usage:
    python -m src.remine_policy check          # ask whether to re-mine now
    python -m src.remine_policy record          # mark "just re-mined" (call after train.py finishes)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from .drift_monitor import check_drift, compute_baseline_stats, weekly_ic_series, LIVE_LOG_CACHE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
POLICY_STATE_CACHE = RESULTS_DIR / "remine_policy_state.json"

MAX_STALENESS_DAYS = 90  # re-mine at least quarterly regardless of drift status


def should_remine(last_mine_date: str, drift_status: dict | None, max_staleness_days: int = MAX_STALENESS_DAYS) -> dict:
    last = datetime.fromisoformat(last_mine_date)
    staleness_days = (datetime.now() - last).days
    reasons = []
    if staleness_days >= max_staleness_days:
        reasons.append(f"stale: {staleness_days} days since last mine (max {max_staleness_days})")
    if drift_status is not None and drift_status.get("status") == "drift_flagged":
        reasons.append(
            f"drift flagged: recent IC {drift_status.get('recent_mean_ic'):+.4f} vs baseline "
            f"{drift_status.get('baseline_mean_ic'):+.4f} (p={drift_status.get('p_value'):.3f})"
        )
    return {"should_remine": len(reasons) > 0, "reasons": reasons, "staleness_days": staleness_days}


def record_mine_date(date: str | None = None) -> None:
    date = date or datetime.now().isoformat()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(POLICY_STATE_CACHE, "w") as f:
        json.dump({"last_mine_date": date}, f, indent=2)


def get_last_mine_date() -> str | None:
    if not POLICY_STATE_CACHE.exists():
        return None
    with open(POLICY_STATE_CACHE) as f:
        return json.load(f).get("last_mine_date")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "record"])
    args = parser.parse_args()

    if args.action == "record":
        record_mine_date()
        print(f"Recorded last-mine date -> {POLICY_STATE_CACHE}")
        return

    last_mine_date = get_last_mine_date()
    if last_mine_date is None:
        print("No recorded last-mine date. Treating as stale -- run train.py, then `python -m src.remine_policy record`.")
        return

    drift_status = None
    if LIVE_LOG_CACHE.exists():
        import pandas as pd

        log = pd.read_csv(LIVE_LOG_CACHE, parse_dates=["predict_date"])
        ic_series = weekly_ic_series(log)
        if len(ic_series) >= 3:
            baseline_mean, baseline_std = compute_baseline_stats()
            drift_status = check_drift(ic_series, baseline_mean, baseline_std)

    result = should_remine(last_mine_date, drift_status)
    print(f"Last mined: {last_mine_date} ({result['staleness_days']} days ago)")
    if drift_status is not None:
        print(f"Drift status: {drift_status['status']}")
    else:
        print("Drift status: no live prediction log yet (this has never run in production)")
    print(f"\nShould re-mine: {result['should_remine']}")
    for reason in result["reasons"]:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
