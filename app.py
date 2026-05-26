import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & DARK THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Breast Cancer Diagnostics Advisor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0F172A !important; color: #F8FAFC !important; }
    section[data-testid="stSidebar"] { background-color: #1E293B !important; border-right: 1px solid #334155 !important; }
    section[data-testid="stSidebar"] * { color: #F8FAFC !important; }
    .main-title { color: #F1F5F9 !important; font-family: 'Inter', sans-serif; font-weight: 700; font-size: 2.6rem; margin-bottom: 0.2rem; }
    .subtitle-text { color: #94A3B8 !important; font-size: 1.15rem; margin-bottom: 2rem; }
    .mode-container { background-color: #1E293B !important; padding: 1.5rem; border-radius: 8px; border: 1px solid #334155 !important; margin-bottom: 1.5rem; color: #F8FAFC !important; }
    .prediction-card { padding: 1.8rem; border-radius: 12px; margin-top: 1.5rem; margin-bottom: 1.5rem; border-left: 6px solid; }
    .malignant-card { background-color: #451A03 !important; color: #FDBA74 !important; border-left-color: #F97316 !important; border: 1px solid #7C2D12 !important; }
    .benign-card { background-color: #064E3B !important; color: #A7F3D0 !important; border-left-color: #10B981 !important; border: 1px solid #065F46 !important; }
    p, span, label, div[data-testid="stWidgetLabel"] p { color: #F8FAFC !important; font-weight: 500 !important; }
    button[data-baseweb="tab"] p { color: #94A3B8 !important; }
    button[aria-selected="true"] p { color: #38BDF8 !important; font-weight: 700 !important; }
    div[data-testid="stMarkdownContainer"] p { color: #E2E8F0 !important; }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. CACHED PIPELINE & DATA UTILITIES
# ==========================================
@st.cache_resource
def load_ml_pipeline():
    paths = {
        "model": ["model.pkl", "Model/model.pkl"],
        "scaler": ["scaler.pkl", "Model/scaler.pkl"],
    }
    loaded = {}
    for key, choices in paths.items():
        found = next((p for p in choices if os.path.exists(p)), None)
        if not found:
            st.error(
                f"❌ Structural Failure: Could not locate required `{key}.pkl` artifact."
            )
            st.stop()
        with open(found, "rb") as f:
            loaded[key] = pickle.load(f)
    return loaded["model"], loaded["scaler"]


try:
    model, scaler = load_ml_pipeline()
except Exception as e:
    st.error(f"Critical error: {str(e)}")
    st.stop()

FEATURE_KEYS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


@st.cache_data
def load_dataset_matrix():
    paths = ["data.csv", "/data.csv"]
    found_path = next((p for p in paths if os.path.exists(p)), None)
    if found_path:
        df = pd.read_csv(found_path)
        clean_df = df[
            [c for c in df.columns if c not in ["id", "diagnosis", "Unnamed: 32"]]
        ]
        return clean_df.mean().to_dict(), df
    return {k: 0.0 for k in FEATURE_KEYS}, None


feature_means, raw_dataset = load_dataset_matrix()

# Initialize session state tracking block to clear stale widget memory caches
if "active_record" not in st.session_state:
    st.session_state.active_record = feature_means.copy()
if "preset_version" not in st.session_state:
    st.session_state.preset_version = 0

# ==========================================
# 3. INTERACTIVE CONTROL SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 🧬 Control Panel")
    st.markdown("---")
    st.markdown(
        "This build predicts breast cancer outcomes based on clinical features."
    )
    st.markdown("---")
    st.success("✅ ML Pipeline: Operational")

# ==========================================
# 4. MAIN WORKFLOW & DATA INTAKE
# ==========================================
st.markdown(
    '<h1 class="main-title">Breast Cancer Prediction Platform</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle-text">Accelerated Diagnostic Intake Hub — select a dataset profile preset or drop an asset record below.</p>',
    unsafe_allow_html=True,
)

st.markdown("### 📥 1. Select Patient Data Intake Mode")
intake_mode = st.radio(
    "Choose your data source selection strategy:",
    [
        "Option A: Load 1-Click Benchmark Profile",
        "Option B: Upload Clinical Record (.CSV)",
        "Option C: Manual Input Form Overrides",
    ],
    horizontal=True,
)

# --- MODE A: LIVE DATASET PRESETS ---
if intake_mode == "Option A: Load 1-Click Benchmark Profile":
    st.markdown('<div class="mode-container">', unsafe_allow_html=True)
    if raw_dataset is not None:
        st.markdown("##### 🎯 Select a Preset Benchmark Case from the Training Dataset")

        sample_malignant = raw_dataset[raw_dataset["diagnosis"] == "M"].iloc[0]
        sample_benign = raw_dataset[raw_dataset["diagnosis"] == "B"].iloc[1]

        preset_selection = st.selectbox(
            "Select Diagnostic Target Case Profile:",
            [
                "Global Dataset Averages (Baseline Control)",
                "Patient Sample Alpha: Confirmed Malignant Core Tumor",
                "Patient Sample Beta: Confirmed Benign Mass Tumor",
            ],
        )

        # FIX: Explicitly update target profiles and advance tracking version
        old_record = st.session_state.active_record.copy()

        if preset_selection == "Patient Sample Alpha: Confirmed Malignant Core Tumor":
            new_rec = sample_malignant[FEATURE_KEYS].to_dict()
        elif preset_selection == "Patient Sample Beta: Confirmed Benign Mass Tumor":
            new_rec = sample_benign[FEATURE_KEYS].to_dict()
        else:
            new_rec = feature_means.copy()

        if new_rec != old_record:
            st.session_state.active_record = new_rec
            st.session_state.preset_version += (
                1  # Forces the widgets below to redraw instantly
            )
            st.rerun()

    else:
        st.warning(
            "Could not find local `data.csv` references to seed diagnostic presets."
        )
    st.markdown("</div>", unsafe_allow_html=True)

# --- MODE B: DATASET DROPPED ASSETS ---
elif intake_mode == "Option B: Upload Clinical Record (.CSV)":
    st.markdown('<div class="mode-container">', unsafe_allow_html=True)
    st.markdown("##### 📂 Upload Single-Row Patient Feature Profile Extracted Data")
    uploaded_file = st.file_uploader(
        "Upload continuous cytological CSV file row matches mapping specifications",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            uploaded_df = uploaded_df.drop(
                columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore"
            )
            missing_cols = [
                col for col in FEATURE_KEYS if col not in uploaded_df.columns
            ]

            if not missing_cols:
                st.session_state.active_record = uploaded_df.iloc[0][
                    FEATURE_KEYS
                ].to_dict()
                st.session_state.preset_version += 1
                st.success(
                    "🎉 Successfully matched and loaded incoming record layout schema!"
                )
                st.rerun()
            else:
                st.error(
                    f"Schema mismatch anomaly discovered. Asset is missing columns: {missing_cols}"
                )
        except Exception as e:
            st.error(f"Error parsing uploaded file format: {str(e)}")

    with st.expander("📝 View Required CSV Column Schema Layout Template"):
        st.dataframe(pd.DataFrame(columns=FEATURE_KEYS), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info(
        "💡 Mode C Active: Update any specific metric coordinates manually within the segmented tabs below."
    )

# ==========================================
# 5. RENDER ACCESSIBLE NUMERIC FIELDS (WITH FIX)
# ==========================================
st.markdown("### 🎛️ 2. Verify or Fine-Tune Parameters")
st.caption("Values are dynamically pre-loaded based on your selection above.")

final_inputs = {}
tab_mean, tab_se, tab_worst = st.tabs(
    [
        "🧬 1. Nuclear Mean Values",
        "📉 2. Standard Error Metrics",
        "🔺 3. Extreme (Worst) Metrics",
    ]
)
tab_groups = [("mean", tab_mean), ("se", tab_se), ("worst", tab_worst)]

for suffix, active_tab in tab_groups:
    with active_tab:
        subset = [f for f in FEATURE_KEYS if f.endswith(f"_{suffix}")]
        cols = st.columns(3)
        for idx, feature in enumerate(subset):
            col_target = cols[idx % 3]
            with col_target:
                clean_label = (
                    feature.replace(f"_{suffix}", "").replace("_", " ").title()
                )
                loaded_val = float(st.session_state.active_record.get(feature, 0.0))

                # FIX: Tying the key name to preset_version forces a clean redraw
                # whenever a user switches choices in the dropdown above
                final_inputs[feature] = col_target.number_input(
                    label=f"{clean_label} ({suffix})",
                    min_value=0.0,
                    value=loaded_val,
                    format="%.5f",
                    key=f"num_{feature}_v{st.session_state.preset_version}",
                )

# ==========================================
# 6. PIPELINE PROCESSING ENGINE & OUTPUTS
# ==========================================
st.markdown("---")
col_action, _ = st.columns([1, 2])
with col_action:
    execute_prediction = st.button(
        "🔮 Run Diagnostic Prediction Matrix", type="primary", use_container_width=True
    )

if execute_prediction:
    eval_df = pd.DataFrame([final_inputs])[FEATURE_KEYS]
    scaled_features = scaler.transform(eval_df)

    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    classes = list(model.classes_)
    prob_malignant = probabilities[classes.index("M")] * 100
    prob_benign = probabilities[classes.index("B")] * 100

    st.markdown("### 📋 Automated Machine Learning Output")

    if prediction == "M":
        st.markdown(
            f"""
            <div class="prediction-card malignant-card">
                <h3 style="color: #FDBA74 !important; margin-top: 0;">🚨 Diagnosis Alert: Malignant Characteristics Detected</h3>
                <p style="color: #FFEDD5 !important;">The pipeline has flagged this configuration as highly indicative of <b>Malignant Tumor Tissue (M)</b>.</p>
                <h4 style="color: #FDBA74 !important;">Statistical Breakdown:</h4>
                <ul style="color: #FFEDD5 !important;">
                    <li>Malignant Confidence Index: <b>{prob_malignant:.2f}%</b></li>
                    <li>Benign Confidence Index: {prob_benign:.2f}%</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="prediction-card benign-card">
                <h3 style="color: #A7F3D0 !important; margin-top: 0;">✅ Diagnosis Summary: Benign Profile Confirmed</h3>
                <p style="color: #ECFDF5 !important;">The pipeline has flagged this configuration as indicative of <b>Benign (Non-Cancerous) Tissue (B)</b>.</p>
                <h4 style="color: #A7F3D0 !important;">Statistical Breakdown:</h4>
                <ul style="color: #ECFDF5 !important;">
                    <li>Benign Confidence Index: <b>{prob_benign:.2f}%</b></li>
                    <li>Malignant Confidence Index: {prob_malignant:.2f}%</li>
                </ul>
            </div>
        """,
            unsafe_allow_html=True,
        )

    chart_data = pd.DataFrame(
        {
            "Diagnostic Outcome": ["Benign (B)", "Malignant (M)"],
            "Confidence Score (%)": [prob_benign, prob_malignant],
        }
    )
    st.bar_chart(
        data=chart_data,
        x="Diagnostic Outcome",
        y="Confidence Score (%)",
        color=["#F97316" if prediction == "M" else "#10B981"],
    )
