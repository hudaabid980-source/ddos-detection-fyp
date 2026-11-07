import os
import pandas as pd
import joblib
import streamlit as st

from inference import prepare_features, predict

# Relative paths: work locally and on the hosting server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_ddos.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "ddos_label_encoder.pkl")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_traffic.csv")

st.set_page_config(page_title="DDoS Detection System", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading model (first start can take a minute)...")
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


@st.cache_data
def load_sample():
    if os.path.exists(SAMPLE_PATH):
        return pd.read_csv(SAMPLE_PATH, low_memory=False)
    return None


model, encoder = load_artifacts()
required_features = list(model.feature_names_in_)
sample_df = load_sample()


def show_analysis(data, demo=False):
    st.write(f"{'Sample' if demo else 'Uploaded'} dataset: {data.shape[0]:,} rows × {data.shape[1]} columns")
    st.dataframe(data.head())

    X, missing = prepare_features(data, required_features)

    if missing:
        st.error(f"CSV is missing {len(missing)} of the {len(required_features)} required features.")
        with st.expander("Show missing columns"):
            st.write(missing)
        return
    if X.empty:
        st.error("The uploaded file has no rows.")
        return

    with st.spinner("Analysing traffic..."):
        result = predict(model, encoder, X)

    output = pd.concat([data.reset_index(drop=True), result.reset_index(drop=True)], axis=1)

    n_attack = int((result["Traffic Type"] == "Attack").sum())
    n_total = len(result)
    c1, c2, c3 = st.columns(3)
    c1.metric("Flows analysed", f"{n_total:,}")
    c2.metric("Flagged as attack", f"{n_attack:,}", f"{n_attack / n_total:.1%}", delta_color="off")
    c3.metric("Benign", f"{n_total - n_attack:,}")

    st.success("Traffic Analysis Completed!")

    st.subheader("Detection Results")
    st.dataframe(result.head(1000))
    if n_total > 1000:
        st.caption("Showing first 1,000 rows. The download contains all rows.")

    st.subheader("Prediction Summary")
    left, right = st.columns(2)
    left.write("Attack vs Benign")
    left.bar_chart(result["Traffic Type"].value_counts())
    right.write("Predicted class")
    right.bar_chart(result["Prediction"].value_counts())

    st.info(
        "Note: the model separates attack from benign traffic very reliably, but some reflection-attack "
        "types overlap in CICDDoS2019 (e.g. DrDoS_SSDP is often labelled DrDoS_UDP, DrDoS_DNS/LDAP are "
        "confused with each other). Treat the specific attack type as indicative."
    )

    st.download_button("Download Results", output.to_csv(index=False),
                       "DDoS_Detection_Results.csv", "text/csv")


st.title("🛡️ DDoS Attack Detection System")
st.write("Machine Learning Based Network Traffic Detection (Random Forest, CICDDoS2019)")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        f"1. Prepare a CSV with the **{len(required_features)} flow features** used in training.\n"
        "2. Upload it on the main page.\n"
        "3. Review predictions and download the results.\n\n"
        "No CSV? A demo with sample data is shown automatically."
    )
    if sample_df is not None:
        st.download_button("Download sample CSV (325 flows)", sample_df.to_csv(index=False),
                           "sample_traffic.csv", "text/csv")
    template = pd.DataFrame(columns=required_features).to_csv(index=False)
    st.download_button("Download CSV template (headers only)", template,
                       "ddos_input_template.csv", "text/csv")
    st.caption("Extra columns (e.g. Label) are ignored. Missing/invalid values are treated as 0.")

uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()
    show_analysis(data)
elif sample_df is not None:
    st.info(
        "👆 No file uploaded yet, so this is a **demo on sample data** (325 network flows, 25 per traffic type "
        "from CICDDoS2019). Upload your own CSV above to analyse your traffic. The sample may overlap with the "
        "training data, so this demo shows how the app works and is not a measure of accuracy."
    )
    show_analysis(sample_df, demo=True)
else:
    st.info("Upload a CSV file to start the analysis.")
