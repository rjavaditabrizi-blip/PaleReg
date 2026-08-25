"""Pre-flight check: run the cheap ElasticNet + LightGBM baselines (and the
existing univariate IC screen) before committing to an expensive SR search,
and combine them into a GO/CAUTION signal plus a feature-priority ranking
SR can optionally restrict to.

Empirically validated tradeoffs this encodes (see project history):
- ElasticNet walk-forward IC is a near-free (~90s) floor check: if it can't
  find ANY linear signal, an expensive nonlinear search is a much longer
  shot at finding something a linear model couldn't.
- ElasticNet does NOT replace SR for portfolio construction -- it matched
  SR's IC (+0.0200 vs +0.0218) but had markedly worse decile-backtest
  Sharpe (-0.15 vs +0.29), so a healthy ElasticNet floor doesn't mean SR is
  pointless -- SR's nonlinearity earns something specifically in portfolio
  construction that IC alone doesn't capture.
- LightGBM's raw predictions were poor here (untuned, overfit to low-signal
  data -- IC ~0) but its feature importance still triangulated onto the
  same features ElasticNet and SR each found valuable independently --
  useful as a cheap feature-relevance signal even though its own
  predictions shouldn't be trusted.

This does NOT hard-block an SR run, and does NOT recommend excluding
low-ranked features by default -- feature_screen.py's own caveat applies
here too: some of SR's best mined formulas use features that score low
standalone (e.g. delta_close_5, only valuable combined with sum_volume_10).
Restricting to a subset is an explicit opt-in (--top-n), not automatic.

Usage:
    python -m src.preflight
    python -m src.preflight --top-n 12   # also writes a recommended subset
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import NON_FEATURE_COLS, elasticnet_fit_predict, lightgbm_fit_predict, run_baseline_walkforward
from .feature_screen import screen_features
from .folds import make_folds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FEATURES_CACHE = DATA_DIR / "features.parquet"
PREFLIGHT_CACHE = RESULTS_DIR / "preflight.json"


def resolve_feature_subset(spec: str | None, all_feature_cols: list[str]) -> list[str]:
    """spec is either None (use everything), "preflight" (read the saved
    preflight.json's recommended top-N), or a comma-separated feature list.
    Used by train.py/walkforward.py's --feature-subset flag."""
    if spec is None:
        return all_feature_cols
    if spec == "preflight":
        if not PREFLIGHT_CACHE.exists():
            raise SystemExit(f"{PREFLIGHT_CACHE} not found -- run `python -m src.preflight --top-n N` first.")
        with open(PREFLIGHT_CACHE) as f:
            recommended = json.load(f).get("recommended_top_n")
        if not recommended:
            raise SystemExit(f"{PREFLIGHT_CACHE} has no recommended_top_n -- re-run preflight.py with --top-n.")
        return recommended
    subset = [c.strip() for c in spec.split(",")]
    unknown = set(subset) - set(all_feature_cols)
    if unknown:
        raise SystemExit(f"Unknown feature(s) in --feature-subset: {unknown}")
    return subset


def run_preflight(df: pd.DataFrame, feature_cols: list[str], folds, cost_bps: float = 5.0) -> dict:
    print("=== ElasticNet floor check ===")
    en_folds, en_importance = run_baseline_walkforward(df, feature_cols, folds, elasticnet_fit_predict, cost_bps)
    en_ics = np.array([r["test_ic"] for r in en_folds])
    en_sharpes = np.array([r["sharpe"] for r in en_folds if not np.isnan(r["sharpe"])])

    print("\n=== LightGBM importance screen ===")
    lgb_folds, lgb_importance = run_baseline_walkforward(df, feature_cols, folds, lightgbm_fit_predict, cost_bps)
    lgb_ics = np.array([r["test_ic"] for r in lgb_folds])

    print("\n=== Univariate IC screen ===")
    screen_df = screen_features(df, feature_cols)

    lgb_rank = pd.DataFrame(lgb_importance).mean().rank(pct=True)
    ir_rank = screen_df.set_index("feature")["information_ratio"].abs().rank(pct=True)
    consensus = ((lgb_rank + ir_rank.reindex(lgb_rank.index)) / 2).sort_values(ascending=False)

    en_mean_ic = float(en_ics.mean())
    go = en_mean_ic > 0.005 and float(np.mean(en_ics > 0)) >= 0.6
    reasoning = (
        [
            f"ElasticNet mean IC ({en_mean_ic:+.4f}) is weak/inconsistent -- an expensive nonlinear "
            "search is a longer shot at finding signal a linear model can barely see. Consider "
            "richer/different data before another SR run, not just more search budget."
        ]
        if not go
        else [
            f"ElasticNet mean IC ({en_mean_ic:+.4f}) confirms real linear signal is present. SR is "
            "worth running specifically for the DECILE-BACKTEST SHARPE gain its nonlinearity has shown "
            "previously, not for IC improvement -- ElasticNet already gets ~the same IC far more cheaply."
        ]
    )

    return {
        "elasticnet": {
            "mean_ic": en_mean_ic,
            "mean_sharpe": float(en_sharpes.mean()) if len(en_sharpes) else float("nan"),
        },
        "lightgbm": {"mean_ic": float(lgb_ics.mean())},
        "go": go,
        "reasoning": reasoning,
        "consensus_feature_rank": consensus.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-frac", type=float, default=0.4)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--top-n", type=int, default=None, help="If set, also writes a recommended top-N feature subset by consensus rank")
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_CACHE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    unique_dates = np.sort(df["date"].unique())
    folds = make_folds(unique_dates, args.n_folds, args.min_train_frac)

    result = run_preflight(df, feature_cols, folds, args.cost_bps)

    print(f"\n{'=' * 60}\nPre-flight summary\n{'=' * 60}")
    print(f"ElasticNet: mean IC={result['elasticnet']['mean_ic']:+.4f}  mean Sharpe={result['elasticnet']['mean_sharpe']:+.2f}")
    print(f"LightGBM:   mean IC={result['lightgbm']['mean_ic']:+.4f}")
    print(f"\n{'GO' if result['go'] else 'CAUTION'}: " + " ".join(result["reasoning"]))

    print("\nConsensus feature ranking (LightGBM importance + |univariate IR|, averaged percentile):")
    for feat, score in result["consensus_feature_rank"].items():
        print(f"  {feat:<20} {score:.3f}")

    recommended = None
    if args.top_n:
        recommended = list(result["consensus_feature_rank"].keys())[: args.top_n]
        print(f"\nRecommended top-{args.top_n} feature subset for a scoped SR run:\n  {recommended}")
        print(
            "\nNOTE: this is a PRIORITIZATION, not a hard exclusion recommendation -- some of SR's best "
            "mined formulas use features that scored low standalone. Use this subset only when you "
            "specifically want a faster, narrower search, not as a default."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result["recommended_top_n"] = recommended
    with open(PREFLIGHT_CACHE, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved -> {PREFLIGHT_CACHE}")


if __name__ == "__main__":
    main()
