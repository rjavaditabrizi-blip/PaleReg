"""Streamlit UI for the symbolic-alpha research pipeline.

Reads whatever the CLI tools (train.py / combine.py / walkforward.py) have
already produced in results/ -- this app is a viewer + a trigger for
predict_today.py's live-scoring step, not a replacement for the mining
pipeline (that still runs from the command line; GP search takes too long
to run inside a web request).

Run with:
    streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.explain import breakdown_program, render_tree_lines
from src.features import compute_live_features
from src.predict_today import load_model, score_universe, select_diversified
from src.providers import get_provider

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WF_RESULTS_CACHE = RESULTS_DIR / "walkforward_results.json"
OOS_PREDICTIONS_CACHE = RESULTS_DIR / "oos_predictions.parquet"
FORMULAS_CACHE = RESULTS_DIR / "formulas.json"
WEIGHTS_CACHE = RESULTS_DIR / "combo_weights.json"

st.set_page_config(page_title="Symbolic Alpha", layout="wide")


@st.cache_resource
def cached_load_model():
    return load_model()


@st.cache_data(ttl=3600)
def cached_fetch_and_score(_provider_name: str, tickers: tuple, period: str, feature_cols: tuple, _programs, weights: tuple):
    provider = get_provider(_provider_name)
    panel = provider.fetch_ohlcv(list(tickers), period=period)
    df, live_feature_cols = compute_live_features(panel)
    missing = set(feature_cols) - set(live_feature_cols)
    if missing:
        raise RuntimeError(f"Live feature set is missing columns the model expects: {missing}")
    latest_date = df["date"].max()
    df_today = df[df["date"] == latest_date]
    scored = score_universe(df_today, list(feature_cols), _programs, np.array(weights))
    # Keep the raw feature row per ticker too, so the "why" breakdown can be
    # computed later without re-fetching -- indexed for O(1) lookup by ticker.
    feature_rows = df_today.set_index("ticker")[list(feature_cols)]
    return scored.sort_values("alpha", ascending=False), latest_date, feature_rows


@st.cache_data
def cached_load_oos_predictions():
    return pd.read_parquet(OOS_PREDICTIONS_CACHE)


st.title("Symbolic Alpha")
st.caption("Lightweight genetic-programming symbolic regression for S&P 500 alpha factors")

st.warning(
    "Research signal, not investment advice. Validated mean IC ~ +0.02 across a 5-fold "
    "walk-forward test -- a portfolio-level statistic, not a claim about any single stock. "
    "See the validated performance below before trusting any of this with real capital.",
    icon="⚠️",
)

tab_picks, tab_performance, tab_factors = st.tabs(["Today's Picks", "Validated Performance", "Mined Factors"])

with tab_picks:
    st.subheader("Generate today's cross-sectional ranking")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        provider_name = st.selectbox("Data provider", ["yfinance"], help="See src/providers for adding more sources")
    with col2:
        period = st.text_input("History window to pull", value="8mo", help="Needs to cover the longest lookback (60d momentum) plus warmup")
    with col3:
        top_n = st.slider("Top/bottom N to show", 5, 30, 15)
    with col4:
        max_per_sector = st.slider(
            "Max picks per sector", 0, 15, 0,
            help="0 = no cap. The model itself isn't sector-diversified (a real part of its edge "
            "is a volume/attention effect that's inherently sector-correlated) -- this caps "
            "concentration in the final picks without changing what the model is allowed to find.",
        )

    if st.button("Generate picks", type="primary"):
        try:
            model = cached_load_model()
        except SystemExit as e:
            st.error(str(e))
            st.stop()
        programs, feature_cols, weights, formulas = model

        provider = get_provider(provider_name)
        with st.spinner(f"Fetching universe and {period} of history via {provider_name}..."):
            universe = provider.fetch_universe()
            tickers = tuple(universe["ticker"].tolist())
            scored, latest_date, feature_rows = cached_fetch_and_score(
                provider_name, tickers, period, tuple(feature_cols), programs, tuple(weights.tolist())
            )

        # Stash everything the "why" breakdown needs so it survives the rerun
        # that happens when the ticker selectbox below changes.
        st.session_state["picks"] = {
            "scored": scored, "latest_date": latest_date, "feature_rows": feature_rows,
            "programs": programs, "feature_cols": feature_cols, "weights": weights, "formulas": formulas,
        }

    if "picks" in st.session_state:
        p = st.session_state["picks"]
        scored, latest_date = p["scored"], p["latest_date"]

        st.success(f"Scored {len(scored)} stocks as of {latest_date.date()}")

        cap = max_per_sector or None
        longs = select_diversified(scored.sort_values("alpha", ascending=False), top_n, cap)
        shorts = select_diversified(scored.sort_values("alpha", ascending=True), top_n, cap)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Top {top_n} (long candidates)**")
            st.dataframe(longs[["ticker", "sector", "alpha"]], hide_index=True, width="stretch")
        with c2:
            st.markdown(f"**Bottom {top_n} (short candidates)**")
            st.dataframe(shorts[["ticker", "sector", "alpha"]], hide_index=True, width="stretch")

        st.markdown("**Sector distribution of long candidates**")
        st.bar_chart(longs["sector"].value_counts())

        st.divider()
        st.markdown("### Why did it pick that? (formula breakdown)")
        candidates = pd.concat([longs, shorts])["ticker"].tolist()
        picked_ticker = st.selectbox("Inspect a candidate", candidates)

        if picked_ticker:
            row = scored[scored["ticker"] == picked_ticker].iloc[0]
            X_row = p["feature_rows"].loc[[picked_ticker]][p["feature_cols"]].values
            st.write(f"**{picked_ticker}** ({row['sector']}) -- composite alpha = `{row['alpha']:+.6f}`")

            nonzero = [
                (f, w, prog) for f, w, prog in zip(p["formulas"], p["weights"], p["programs"]) if abs(w) > 1e-10
            ]
            for f, w, prog in nonzero:
                val = prog.execute(X_row)[0]
                with st.expander(f"{f['formula']}  →  value={val:+.4f} × weight={w:+.4f} = contributes {val * w:+.6f}"):
                    tree = breakdown_program(prog, X_row, p["feature_cols"])
                    st.code("\n".join(render_tree_lines(tree)), language=None)
    else:
        st.caption("Click \"Generate picks\" to fetch live data and score the universe.")

with tab_performance:
    st.subheader("Walk-forward validation results")
    if not WF_RESULTS_CACHE.exists():
        st.info("No walk-forward results yet. Run `python -m src.walkforward` first.")
    else:
        with open(WF_RESULTS_CACHE) as f:
            fold_results = json.load(f)
        df_folds = pd.DataFrame(fold_results)

        sharpes = df_folds["sharpe"].dropna()
        ics = df_folds["test_ic"].dropna()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Sharpe", f"{sharpes.mean():+.2f}", f"std {sharpes.std(ddof=0):.2f}")
        m2.metric("Sharpe hit rate", f"{(sharpes > 0).mean():.0%}")
        m3.metric("Mean IC", f"{ics.mean():+.4f}", f"std {ics.std(ddof=0):.4f}")
        m4.metric("IC hit rate", f"{(ics > 0).mean():.0%}")

        if "train_ic" in df_folds.columns:
            st.markdown("**In-sample vs. out-of-sample IC, per fold**")
            ic_chart = df_folds.set_index("fold")[["train_ic", "test_ic"]].rename(
                columns={"train_ic": "in-sample (train)", "test_ic": "out-of-sample (test)"}
            )
            st.line_chart(ic_chart)

            mean_train_ic = df_folds["train_ic"].mean()
            mean_test_ic = df_folds["test_ic"].mean()
            retained = mean_test_ic / mean_train_ic if mean_train_ic > 0 else float("nan")
            if mean_train_ic > 0.005 and (retained < 0.3 or mean_test_ic < 0):
                st.error(
                    f"⚠️ WARNING: High risk of overfitting -- out-of-sample IC "
                    f"(+{mean_test_ic:.4f}) retains only {retained:.0%} of in-sample IC "
                    f"(+{mean_train_ic:.4f}).",
                    icon="🚨",
                )
            else:
                st.caption(
                    f"Out-of-sample IC retains {retained:.0%} of in-sample IC on average -- "
                    "no strong divergence between the two lines above."
                )

        st.markdown("**Per-fold results**")
        show_cols = [c for c in ["fold", "test_start", "test_end", "train_ic", "test_ic", "test_rank_ic", "sharpe", "ann_return", "n_unique_formulas"] if c in df_folds.columns]
        st.dataframe(df_folds[show_cols], hide_index=True, width="stretch")

        st.caption(
            "Sharpe estimated from only a handful of folds is noisy by construction -- the IC "
            "hit-rate (does the sign hold up out-of-sample, fold after fold) is the more "
            "trustworthy number here. See the project conversation history for why."
        )

        if OOS_PREDICTIONS_CACHE.exists():
            st.divider()
            st.markdown("**Out-of-sample alpha vs. realized forward return**")
            st.caption(
                "Every point here is a real out-of-sample prediction pooled across all "
                "walk-forward folds -- this is what a mean IC of ~0.02 actually looks like: "
                "a real but weak/noisy relationship, not a clean line."
            )
            oos = cached_load_oos_predictions()
            sample = oos.sample(min(5000, len(oos)), random_state=0)
            st.scatter_chart(sample, x="alpha", y="fwd_ret", size=10)
        else:
            st.info("No pooled out-of-sample predictions yet -- re-run `python -m src.walkforward` with the updated script to generate this.")

with tab_factors:
    st.subheader("Mined alpha formulas")
    if not FORMULAS_CACHE.exists():
        st.info("No mined formulas yet. Run `python -m src.train` first.")
    else:
        with open(FORMULAS_CACHE) as f:
            formulas = json.load(f)
        df_formulas = pd.DataFrame(formulas)

        if WEIGHTS_CACHE.exists():
            with open(WEIGHTS_CACHE) as f:
                weights_info = json.load(f)
            weights = weights_info["weights"]
            if len(weights) == len(df_formulas):
                df_formulas["combo_weight"] = weights

        st.dataframe(df_formulas, hide_index=True, width="stretch")
