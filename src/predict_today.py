"""Generate today's cross-sectional stock ranking from the mined + combined
alpha model.

This is a RESEARCH SIGNAL, not investment advice. Two things worth
internalizing before acting on the output at all:

1. The validated edge (see walkforward.py results) is a mean IC of roughly
   +0.02 -- a *portfolio-level* statistic. It means the ranking works on
   average across hundreds of names. It says approximately nothing about
   whether any single top-ranked stock is a good bet in isolation. Use this
   as one input into a diversified basket, not as a single pick.
2. This pulls today's OHLCV, computes the same operand features used in
   training, evaluates the pickled GP programs on them, and combines them
   with the saved Lasso weights. If any of those artifacts (programs.pkl,
   combo_weights.json, formulas.json) came from different, inconsistent
   runs, the output is meaningless -- run train.py then combine.py together
   in one pass before using this.

Usage:
    python -m src.predict_today --top 15
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .features import compute_live_features
from .providers import get_provider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PROGRAMS_CACHE = RESULTS_DIR / "programs.pkl"
WEIGHTS_CACHE = RESULTS_DIR / "combo_weights.json"
FORMULAS_CACHE = RESULTS_DIR / "formulas.json"


def load_model():
    if not PROGRAMS_CACHE.exists():
        raise SystemExit(
            f"{PROGRAMS_CACHE} not found. Run a production mining pass first, e.g.:\n"
            "  python -m src.train --test-frac 0 ...\n"
            "  python -m src.combine --mode static"
        )
    with open(PROGRAMS_CACHE, "rb") as f:
        saved = pickle.load(f)
    with open(WEIGHTS_CACHE) as f:
        weights = np.array(json.load(f)["weights"])
    with open(FORMULAS_CACHE) as f:
        formulas = json.load(f)

    programs, feature_cols = saved["programs"], saved["feature_cols"]
    if not (len(programs) == len(weights) == len(formulas)):
        raise SystemExit(
            f"Mismatched artifacts: {len(programs)} programs, {len(weights)} weights, "
            f"{len(formulas)} formulas. These must come from the same train.py + combine.py pass."
        )
    return programs, feature_cols, weights, formulas


def score_universe(df_today: pd.DataFrame, feature_cols, programs, weights) -> pd.DataFrame:
    X = df_today[feature_cols].values
    pool = np.column_stack([program.execute(X) for program in programs])
    df_today = df_today.copy()
    df_today["alpha"] = pool @ weights
    return df_today


def select_diversified(ranked: pd.DataFrame, top_n: int, max_per_sector: int | None = None) -> pd.DataFrame:
    """Take the top-n rows of an already-priority-sorted DataFrame, optionally
    skipping any row that would push its sector's count above max_per_sector.

    This is the risk-control step, applied *after* scoring rather than baked
    into the mining target -- walk-forward validation showed sector-
    neutralizing the mining target gutted the signal (a real part of the
    edge is a volume/attention effect that's inherently sector-correlated).
    Capping sector concentration in the final picks gets the diversification
    benefit without touching what the model is actually allowed to find."""
    if not max_per_sector:
        return ranked.head(top_n)
    picked_idx, sector_counts = [], {}
    for idx, row in ranked.iterrows():
        if len(picked_idx) >= top_n:
            break
        sector = row["sector"]
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        picked_idx.append(idx)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    return ranked.loc[picked_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=15, help="How many long/short candidates to show")
    parser.add_argument("--period", default="8mo", help="How much recent history to pull (needs to cover mom_60 + rolling windows + warmup)")
    parser.add_argument("--refresh-tickers", action="store_true")
    parser.add_argument("--provider", default="yfinance", help="Data source name (see src/providers)")
    parser.add_argument("--max-per-sector", type=int, default=None, help="Cap picks per GICS sector, e.g. 3 -- applied after scoring, doesn't change the model")
    args = parser.parse_args()

    programs, feature_cols, weights, formulas = load_model()

    provider = get_provider(args.provider)
    universe = provider.fetch_universe(refresh=args.refresh_tickers)
    tickers = universe["ticker"].tolist()
    print(f"Downloading last {args.period} of OHLCV for {len(tickers)} tickers via {provider.name}...")
    panel = provider.fetch_ohlcv(tickers, period=args.period)

    df, live_feature_cols = compute_live_features(panel)
    missing = set(feature_cols) - set(live_feature_cols)
    if missing:
        raise SystemExit(f"Live feature set is missing columns the model expects: {missing}")

    latest_date = df["date"].max()
    df_today = df[df["date"] == latest_date]
    print(f"Scoring {len(df_today)} stocks as of {latest_date.date()}")

    scored = score_universe(df_today, feature_cols, programs, weights)

    longs = select_diversified(scored.sort_values("alpha", ascending=False), args.top, args.max_per_sector)
    shorts = select_diversified(scored.sort_values("alpha", ascending=True), args.top, args.max_per_sector)

    cap_note = f" (max {args.max_per_sector}/sector)" if args.max_per_sector else ""
    print(f"\n{'=' * 60}\nTop {args.top} (long candidates){cap_note}\n{'=' * 60}")
    print(longs[["ticker", "sector", "alpha"]].to_string(index=False))

    print(f"\n{'=' * 60}\nBottom {args.top} (short candidates){cap_note}\n{'=' * 60}")
    print(shorts[["ticker", "sector", "alpha"]].to_string(index=False))

    print(
        "\nNOTE: this is a research signal (validated mean IC ~ +0.02, a portfolio-level "
        "statistic), not investment advice, and not a recommendation for any single stock. "
        "See walkforward.py results before trusting any of this with real capital."
    )


if __name__ == "__main__":
    main()
