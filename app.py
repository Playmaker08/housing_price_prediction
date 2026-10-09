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

# Empirical 2.5th/97.5th percentiles of log(actual price / predicted price)
# on the original 1,995-row, random_state=42 held-out test split.
# These were calculated offline from Cook_County_Housing_Dataset.xlsx using
# the EXISTING model.joblib; no training or additional deployment file needed.
# This is an approximate PREDICTION interval for an individual sale, NOT a
# classical confidence interval for the conditional mean price.
LOG_RATIO_Q025 = -1.23982312
LOG_RATIO_Q975 = 0.67962130

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
    )
    lot_sqft = c.number_input(
        "Lot size (sq ft) *", min_value=0, max_value=250000, value=5500, step=100
    )

    st.subheader("2 · Interior details")
    c1, c2, c3, c4 = st.columns(4)
    bedrooms = c1.number_input("Bedrooms *", min_value=0, max_value=15, value=3, step=1)
    full_baths = c2.number_input("Full bathrooms *", min_value=0, max_value=12, value=2, step=1)
    half_baths = c3.number_input("Half bathrooms", min_value=0, max_value=8, value=0, step=1)
    rooms = c4.number_input("Total rooms *", min_value=1, max_value=35, value=6, step=1)
    c5, c6, c7 = st.columns(3)
    year_built = c5.number_input(
        "Year built *", min_value=1800, max_value=int(metrics["max_year"]), value=1960, step=1
    )
    fireplaces = c6.number_input("Fireplaces", min_value=0, max_value=10, value=0, step=1)
    sale_year = int(metrics["max_year"])
    c7.text_input(
        "Reference sale year",
        value=str(sale_year),
        disabled=True,
        help="The latest year available in the historical data, not a live market forecast.",
    )

    with st.expander("3 · More property characteristics", expanded=False):
        aa, bb = st.columns(2)
        air = category_input(aa, "Central air", "central_air_code")
        garage = category_input(bb, "Garage capacity", "garage_size_code", default="2 cars")
        basement = category_input(aa, "Basement type", "basement_type_code")
        finish = category_input(bb, "Basement finish", "basement_finish_code")
        wall = category_input(aa, "Exterior wall", "exterior_wall_code")
        condition = category_input(bb, "Repair condition", "repair_condition_code")
        residence = category_input(aa, "Residence type", "residence_type_code")
        local_walk = bundle["township_benchmarks"][township]["median_walk"]
        if local_walk is None:
            local_walk = bundle["numeric_medians"]["access_cmap_walk_total_score"]
        customize_walk = bb.checkbox("I know the area walk-access score", value=False)
        if customize_walk:
            walk_score = bb.number_input(
                "CMAP walk-access score (dataset units)",
                min_value=-10.0,
                max_value=150.0,
                value=float(local_walk),
                step=1.0,
            )
        else:
            walk_score = float(local_walk)
            bb.caption(f"Township median proxy used: {walk_score:.1f} (dataset score units)")

    submitted = st.form_submit_button("Estimate home value", type="primary", use_container_width=True)

if submitted:
    if rooms < bedrooms:
        st.error("Total rooms cannot be fewer than bedrooms. Please adjust the inputs.")
    else:
        record = {
            "building_sqft": building_sqft,
            "lot_sqft": lot_sqft,
            "bedrooms": bedrooms,
            "rooms": rooms,
            "full_baths": full_baths,
            "half_baths": half_baths,
            "year_built": year_built,
            "fireplaces": fireplaces,
            "sale_year": sale_year,
            "access_cmap_walk_total_score": walk_score,
            "township_name": township,
            "central_air_code": air,
            "garage_size_code": garage,
            "basement_type_code": basement,
            "basement_finish_code": finish,
            "exterior_wall_code": wall,
            "repair_condition_code": condition,
            "residence_type_code": residence,
        }
        row = pd.DataFrame([record], columns=bundle["feature_columns"])
        x = bundle["preprocessor"].transform(row)
        log_prediction = float(bundle["regressor"].predict(x)[0])
        estimate = float(np.exp(log_prediction) * bundle["smearing_factor"])

        if not np.isfinite(estimate) or estimate <= 0:
            st.error("A valid estimate could not be generated for these inputs.")
        else:
            interval_low = estimate * float(np.exp(LOG_RATIO_Q025))
            interval_high = estimate * float(np.exp(LOG_RATIO_Q975))
            st.divider()
            st.subheader("Your estimated home value")
            st.markdown('<div class="result-label">Estimated sale price</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-price">${estimate:,.0f}</div>',
                unsafe_allow_html=True,
            )

            how, confidence = st.columns(2, gap="large")
            with how:
                st.markdown("**How we estimate it**")
                st.write(
                    "We use your home's size, rooms, age, condition, and township, "
                    "along with the other details you entered, to estimate value "
                    "from historical Cook County residential sales (2022–2025)."
                )
            with confidence:
                st.markdown("**95% estimated price range**")
                st.markdown(f"**${interval_low:,.0f} – ${interval_high:,.0f}**")
                st.caption(
                    "Approximate 95% prediction interval based on historical "
                    "prediction errors from 1,995 held-out Cook County sales. "
                    "It describes uncertainty in an individual sale price; "
                    "it is not a confidence interval for the average home value "
                    "and 95% coverage is not guaranteed for a new property."
                )
                if estimate > 2_000_000 or estimate < 60_000:
                    st.warning(
                        "This estimate is outside the typical mid-market range; "
                        "use additional caution."
                    )

            st.caption(
                "For information only. Not a professional appraisal or live listing-price recommendation."
            )
else:
    st.caption("No price is calculated until you click **Estimate home value**.")

st.divider()
st.caption("BUSA 310 · Group Project 1 · Cook County Residential Price Prediction Tool")
