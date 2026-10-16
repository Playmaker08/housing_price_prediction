"""Cook County HomeValue — streamlined public-facing Streamlit estimator."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cook County HomeValue",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {padding-top: 2.1rem; max-width: 1130px;}
.hero {background: linear-gradient(110deg, #0c2235, #174462); color: white;
       border-radius: 19px; padding: 26px 30px; margin-bottom: 18px;}
.hero h1 {color: white; margin: 0; font-size: 2.2rem; letter-spacing: -0.6px;}
.hero p {color: #d5e3eb; margin: 9px 0 0 0; font-size: 1rem;}
.result-price {font-size: clamp(2.5rem, 7vw, 4rem); font-weight: 750;
               letter-spacing: -1.2px; line-height: 1.15; margin: 8px 0 12px;}
.result-label {color: #64748b; font-size: .95rem; margin-bottom: 3px;}
/* One consistent type treatment for both dollar amounts; never Markdown-parse $ signs. */
.price-range {display: flex; flex-wrap: wrap; align-items: baseline; gap: .35rem;
              color: #f8fafc; font-size: clamp(1.15rem, 2.2vw, 1.55rem);
              font-weight: 700; font-variant-numeric: tabular-nums;
              letter-spacing: -.02em; line-height: 1.45;}
.price-range .bound, .price-range .separator {color: inherit; font: inherit;}
.price-range .separator {padding: 0 .1rem; opacity: .7;}
</style>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = Path(__file__).with_name("model.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Missing model.joblib. Keep app.py alongside the existing model file.")
    st.stop()

bundle = load_model()
metrics = bundle["metrics"]

# Empirical quantiles of log(actual sale / predicted sale) on 1,995 held-out
# Cook County transactions, using the SAME original model and test split.
# 80% = 10th–90th percentiles; 95% = 2.5th–97.5th percentiles.
# These are approximate marginal prediction intervals, NOT confidence intervals
# for mean home value. Coverage is not independently validated on fresh data.
# The regression model, model.joblib, and point estimates remain unchanged.
PRICE_INTERVALS = {
    "80%": (-0.6474121267651101, 0.417813268740971),
    "95%": (-1.2398231164701248, 0.6796212990919108),
}

st.markdown(
    """<div class="hero"><h1>Cook County HomeValue</h1>
<p>Data-driven residential price estimates · Cook County, Illinois</p></div>""",
    unsafe_allow_html=True,
)


def category_input(column, label, field, display=None, default=None):
    opts = bundle["category_options"][field]
    preferred = default or bundle["category_modes"][field]
    initial = opts.index(preferred) if preferred in opts else 0
    return column.selectbox(
        label,
        opts,
        index=initial,
        format_func=display if display else str,
    )


st.write("Enter your home's details to receive an estimated sale price. Fields marked * are required.")

with st.form("estimate"):
    st.subheader("1 · Location & property size")
    a, b, c = st.columns(3)
    township = category_input(
        a,
        "Township *",
        "township_name",
        default="Lake" if "Lake" in bundle["category_options"]["township_name"] else None,
    )
    building_sqft = b.number_input(
        "Building area (sq ft) *", min_value=200, max_value=30000, value=1500, step=50
