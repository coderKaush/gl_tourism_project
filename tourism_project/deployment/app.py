"""
Streamlit app — Wellness Tourism Package Purchase Predictor
Loads the best-trained model from Hugging Face Model Hub and
serves an interactive prediction interface.
"""
import os, json, pickle
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title='Wellness Tourism Predictor',
    page_icon='🌿',
    layout='centered',
)

# Load model repo from HF Space secret (set in Space → Settings → Variables)
MODEL_REPO = os.getenv('MODEL_REPO', 'k0t0320d/tourism-wellness-model')

@st.cache_resource(show_spinner='Loading model from Hugging Face ...')
def load_artifacts():
    m_path = hf_hub_download(repo_id=MODEL_REPO, filename='best_model.pkl')
    f_path = hf_hub_download(repo_id=MODEL_REPO, filename='feature_columns.json')
    with open(m_path, 'rb') as f: model = pickle.load(f)
    with open(f_path, 'r') as f: feat_cols = json.load(f)
    return model, feat_cols

model, feat_cols = load_artifacts()

# ── Header ──────────────────────────────────────────────────────────
st.title('🌿 Wellness Tourism Package — Purchase Predictor')
st.caption('Powered by an ensemble ML model trained on customer interaction data.')
st.divider()

# ── Input Form ───────────────────────────────────────────────────────
with st.form('input_form'):
    st.subheader('Customer Profile')
    c1, c2, c3 = st.columns(3)
    with c1:
        age           = st.number_input('Age', 18, 70, 35)
        city_tier     = st.selectbox('City Tier', [1, 2, 3])
        passport      = st.radio('Has Passport', ['No', 'Yes'])
    with c2:
        monthly_income = st.number_input('Monthly Income (₹)', 0, 200000, 25000, step=1000)
        num_trips      = st.number_input('Annual Trips', 0, 25, 2)
        own_car        = st.radio('Owns a Car', ['No', 'Yes'])
    with c3:
        num_persons   = st.number_input('Persons Visiting', 1, 10, 2)
        num_children  = st.number_input('Children (< 5 yrs)', 0, 5, 0)
        hotel_stars   = st.selectbox('Preferred Hotel Stars', [3, 4, 5])

    st.subheader('Sales Interaction')
    i1, i2, i3 = st.columns(3)
    with i1:
        pitch_score   = st.slider('Pitch Satisfaction (1–5)', 1, 5, 3)
    with i2:
        followups     = st.number_input('Follow-up Calls', 0, 10, 3)
    with i3:
        pitch_duration = st.number_input('Pitch Duration (min)', 1, 60, 15)

    predict_btn = st.form_submit_button('🔍 Predict Purchase Likelihood', use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────────
if predict_btn:
    input_dict = {
        'Age': age, 'CityTier': city_tier,
        'Passport': 1 if passport == 'Yes' else 0,
        'OwnCar': 1 if own_car == 'Yes' else 0,
        'MonthlyIncome': monthly_income,
        'NumberOfTrips': num_trips,
        'NumberOfPersonVisiting': num_persons,
        'NumberOfChildrenVisiting': num_children,
        'PreferredPropertyStar': hotel_stars,
        'PitchSatisfactionScore': pitch_score,
        'NumberOfFollowups': followups,
        'DurationOfPitch': pitch_duration,
    }
    input_df = pd.DataFrame([input_dict]).reindex(columns=feat_cols, fill_value=0)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f'**LIKELY to purchase** — Estimated probability: **{probability:.1%}**')
    else:
        st.warning(f'**UNLIKELY to purchase** — Estimated probability: **{probability:.1%}**')

    st.progress(float(probability), text=f'Purchase probability: {probability:.1%}')

st.divider()
st.caption('Visit with Us · Wellness Tourism Package Prediction System')
