"""Cook County HomeValue: a course-project housing price estimator."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='Cook County HomeValue', page_icon='🏠',
    layout='wide', initial_sidebar_state='expanded'
)

st.markdown('''
<style>
.block-container {padding-top:2.1rem; max-width:1130px;}
.hero {background:linear-gradient(110deg,#0c2235,#174462);color:white;
       border-radius:19px;padding:26px 30px;margin-bottom:18px}
.hero h1 {color:white;margin:0;font-size:2.2rem;letter-spacing:-0.6px}
.hero p {color:#d5e3eb;margin:9px 0 0 0;font-size:1rem}
.mini {font-size:.87rem;color:#637485}
</style>
''', unsafe_allow_html=True)

MODEL_PATH = Path(__file__).with_name('model.joblib')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error('Missing model.joblib. See README.md to export and deploy the model.')
    st.stop()

bundle = load_model()
metrics = bundle['metrics']

st.markdown('''<div class="hero"><h1>Cook County HomeValue</h1>
<p>Data-driven residential price estimates · Cook County, Illinois</p></div>''',
unsafe_allow_html=True)

estimator, methodology = st.tabs(['🏡 Home price estimator', '📊 Model & methodology'])

def category_input(column, label, field, display=None, default=None):
    opts = bundle['category_options'][field]
    preferred = default or bundle['category_modes'][field]
    initial = opts.index(preferred) if preferred in opts else 0
    return column.selectbox(label, opts, index=initial,
                            format_func=(display if display else str))

with estimator:
    st.write('Enter property characteristics to estimate its sale price using the trained model. Fields marked * are required.')
    with st.form('estimate'):
        st.subheader('1 · Location & property size')
        a, b, c = st.columns(3)
        township = category_input(a, 'Township *', 'township_name',
                                  default='Lake' if 'Lake' in bundle['category_options']['township_name'] else None)
        building_sqft = b.number_input('Building area (sq ft) *', min_value=200,
                                       max_value=30000, value=1500, step=50)
        lot_sqft = c.number_input('Lot size (sq ft) *', min_value=0,
                                  max_value=250000, value=5500, step=100)

        st.subheader('2 · Interior details')
        c1, c2, c3, c4 = st.columns(4)
        bedrooms = c1.number_input('Bedrooms *', min_value=0, max_value=15, value=3, step=1)
        full_baths = c2.number_input('Full bathrooms *', min_value=0, max_value=12, value=2, step=1)
        half_baths = c3.number_input('Half bathrooms', min_value=0, max_value=8, value=0, step=1)
        rooms = c4.number_input('Total rooms *', min_value=1, max_value=35, value=6, step=1)
        c5, c6, c7 = st.columns(3)
        year_built = c5.number_input('Year built *', min_value=1800,
                                     max_value=int(metrics['max_year']), value=1960, step=1)
        fireplaces = c6.number_input('Fireplaces', min_value=0, max_value=10, value=0, step=1)
        sale_year = int(metrics['max_year'])
        c7.text_input('Reference sale year', value=str(sale_year), disabled=True,
                      help='Uses the latest year contained in the historical dataset, not a live market forecast.')

        with st.expander('3 · More property characteristics', expanded=False):
            aa, bb = st.columns(2)
            air = category_input(aa, 'Central air', 'central_air_code')
            garage = category_input(bb, 'Garage capacity', 'garage_size_code', default='2 cars')
            basement = category_input(aa, 'Basement type', 'basement_type_code')
            finish = category_input(bb, 'Basement finish', 'basement_finish_code')
            wall = category_input(aa, 'Exterior wall', 'exterior_wall_code')
            condition = category_input(bb, 'Repair condition', 'repair_condition_code')
            residence = category_input(aa, 'Residence type', 'residence_type_code')
            local_walk = bundle['township_benchmarks'][township]['median_walk']
            if local_walk is None:
                local_walk = bundle['numeric_medians']['access_cmap_walk_total_score']
            customize_walk = bb.checkbox('I know the area walk-access score', value=False)
            if customize_walk:
                walk_score = bb.number_input('CMAP walk-access score (dataset units)',
                                             min_value=-10.0, max_value=150.0,
                                             value=float(local_walk), step=1.0)
            else:
                walk_score = float(local_walk)
                bb.caption(f'Township median proxy used: {walk_score:.1f} (dataset score units)')

        submitted = st.form_submit_button('Estimate home value', type='primary', use_container_width=True)

    if submitted:
        if rooms < bedrooms:
            st.error('Total rooms cannot be fewer than bedrooms. Please adjust the inputs.')
        else:
            record = {
                'building_sqft': building_sqft, 'lot_sqft': lot_sqft,
                'bedrooms': bedrooms, 'rooms': rooms,
                'full_baths': full_baths, 'half_baths': half_baths,
                'year_built': year_built, 'fireplaces': fireplaces,
                'sale_year': sale_year, 'access_cmap_walk_total_score': walk_score,
                'township_name': township, 'central_air_code': air,
                'garage_size_code': garage, 'basement_type_code': basement,
                'basement_finish_code': finish, 'exterior_wall_code': wall,
                'repair_condition_code': condition, 'residence_type_code': residence,
            }
            row = pd.DataFrame([record], columns=bundle['feature_columns'])
            x = bundle['preprocessor'].transform(row)
            log_prediction = float(bundle['regressor'].predict(x)[0])
            estimate = float(np.exp(log_prediction) * bundle['smearing_factor'])
            if not np.isfinite(estimate) or estimate <= 0:
                st.error('The model could not generate a valid estimate for these inputs.')
            else:
                st.divider()
                st.subheader('Your estimated home value')
                col1, col2 = st.columns([1, 1])
                col1.metric('Estimated sale price', f'${estimate:,.0f}')
                col1.caption('Model estimate based on historical 2022–2025 transactions — not an appraisal or live listing recommendation.')
                local = bundle['township_benchmarks'][township]
                col2.metric(f'Median training sale · {township}', f'${local["median_sale"]:,.0f}')
                col2.caption(f'Historical township sales in training set: {int(local["count"]):,}')
                st.caption(
                    f'Historical township sale-price middle 50% (not a model prediction interval): '
                    f'${local["lower_quartile"]:,.0f} – ${local["upper_quartile"]:,.0f}'
                )
                st.info(
                    'For context, the model’s held-out test MAE was '
                    f'${metrics["MAE"]:,.0f}. This is an average error across test homes, '
                    'NOT a prediction interval for this particular property.'
                )
                if estimate > 2_000_000 or estimate < 60_000:
                    st.warning('This estimate is far from the typical mid-market range. Interpret with extra caution.')
                st.caption('Local context includes transaction mix and is not an apples-to-apples comparable-sales appraisal.')
                with st.expander('See the property inputs used'):
                    st.dataframe(row.T.rename(columns={0:'Entered value'}), use_container_width=True)
    else:
        st.caption('No price is calculated until you click **Estimate home value**.')

with methodology:
    st.subheader('About this model')
    st.write('This educational prototype estimates Cook County residential sale prices from property characteristics and township. It is built from a compiled, real-world housing workbook with multiple underlying source datasets.')
    u, v, w, z = st.columns(4)
    u.metric('Sale records', f'{metrics["all_n"]:,}')
    v.metric('Test RMSE', f'${metrics["RMSE"]:,.0f}')
    w.metric('Test MAE', f'${metrics["MAE"]:,.0f}')
    z.metric('Test R²', f'{metrics["R2"]:.3f}')
    st.markdown('''**Model:** Ordinary least squares on `ln(sale_price)`, transformed to USD using a training-residual smearing factor. Median imputation is applied to numeric features; categorical missing values use the training-mode category and one-hot encoding with one reference category omitted.

**Evaluation:** A fixed random 80/20 train/test split (`random_state=42`), the same split used in the notebook. Performance is held-out on this split but not verified on later-year transactions or repeated sales of the same property.

**Included:** Building area, lot area, bedrooms, rooms, bathrooms, construction year, fireplaces, sale year, township, walk-access score, A/C, garage, basement, exterior material, repair condition and residence type.

**Excluded:** ZIP code, school factors, and crime data are **not** predictors in the deployed model. The property’s street address is not captured, so the estimator cannot account for street-level variation.

**Limitations:** Historical sales (`2022–2025`), not live listings; nonrandom sample; incomplete information about renovations and property condition; unusually priced properties in data; and broad location granularity. The township market summary comes from the training data, not a current MLS feed. The 2025 reference year is a fixed modeling choice, not a forecast of 2026 conditions.

**Responsible use:** This demonstration is not a professional appraisal, mortgage lending decision, investment recommendation, or final asking-price recommendation.''')

st.divider()
st.caption('BUSA 310 · Group Project 1 · Cook County Residential Price Prediction Tool')
