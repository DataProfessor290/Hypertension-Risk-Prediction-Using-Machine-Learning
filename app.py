import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# =============================
# 🚀 Load Model and Dataset
# =============================
model = joblib.load("hypertension_model_v2.pkl")  # Ensure this file is in your working directory

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.columns = df.columns.str.lower().str.strip()
    df["medication"] = df["medication"].fillna("None")
    df["has_hypertension"] = df["has_hypertension"].replace({"Yes": 1, "No": 0})
    return df

df = load_data()

# =============================
# 🌑 Dark Mode Styling
# =============================
st.set_page_config(page_title="Hypertension Predictor", layout="centered")

st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #111111;
        color: #F5F5F5;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.4em 1em;
    }
    .stDownloadButton>button {
        background-color: #117A65;
    }
    .stMetricLabel, .stMetricValue {
        color: #F5F5F5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# ⚖️ BMI Sidebar
# =============================
st.sidebar.header("⚖️ BMI Calculator")
st.sidebar.markdown("_Don't know your BMI? Calculate it here:_")

weight = st.sidebar.number_input(
    "Weight (kg)", min_value=10.0, step=0.5, format="%.1f",
    help="Example: 70.5"
)
height = st.sidebar.number_input(
    "Height (m)", min_value=0.5, step=0.01, format="%.2f",
    help="Example: 1.75"
)

if height > 0:
    calculated_bmi = round(weight / (height ** 2), 2)
    st.sidebar.success(f"Your BMI is: **{calculated_bmi}**")
else:
    calculated_bmi = 25.0  # fallback

st.sidebar.caption("💡 BMI = weight (kg) ÷ height² (m²)")

with st.sidebar.expander("📚 What is BMI?"):
    st.markdown("""
    _Body Mass Index (BMI)_ uses your height and weight to categorize your health:
    - ⚠️ **Underweight**: BMI < 18.5  
    - ✅ **Normal**: 18.5–24.9  
    - ⚠️ **Overweight**: 25–29.9  
    - 🔶 **Obese I**: 30–34.9  
    - 🔴 **Obese II**: 35–39.9  
    - 🔴 **Obese III**: BMI ≥ 40  
    """)

# =============================
# 💡 Header
# =============================
st.markdown("<h2 style='text-align: center;'>🩺 Hypertension Risk Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B0BEC5;'>Powered by XGBoost | Built with Streamlit</p>", unsafe_allow_html=True)

# =============================
# 🔢 Input Section
# =============================
st.markdown("### 🔍 Patient Info")

col1, col2 = st.columns(2)
with col1:
    bmi = st.number_input("💪 BMI", min_value=10.0, max_value=60.0, value=calculated_bmi, step=0.1)
    stress_score = st.slider("😖 Stress Level", 0, 10, 5)

with col2:
    family_history = st.selectbox("👪 Family History", ["yes", "no"])
    smoking_status = st.selectbox("🚬 Smoking Status", ["Never", "Former", "Current"])

bp_history = st.selectbox("💓 BP History", ["Normal", "Elevated", "Stage 1", "Stage 2"])

# =============================
# 🔮 Prediction
# =============================
input_df = pd.DataFrame({
    "bmi": [bmi],
    "family_history": [family_history.lower()],
    "smoking_status": [smoking_status],
    "stress_score": [stress_score],
    "bp_history": [bp_history]
})

if st.button("🔍 Predict Hypertension Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    st.markdown("### 📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Hypertension\n\n**Probability: {probability:.2f}%**")
        st.markdown("*Please consult a healthcare professional.*")
    else:
        st.success(f"✅ Low Risk of Hypertension\n\n**Probability: {probability:.2f}%**")
        st.markdown("*You're in good shape! Keep it up.*")

    # =============================
    # 📊 Accuracy
    # =============================
    st.markdown("---")
    st.subheader("📈 Model Performance")
    feature_cols = ["bmi", "family_history", "smoking_status", "stress_score", "bp_history"]
    X = df[feature_cols]
    y = df["has_hypertension"]
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%")

    # =============================
    # 📂 Download Predictions
    # =============================
    with st.expander("📥 Show & Download Predictions"):
        results_df = X_test.copy()
        results_df["Actual"] = y_test.values
        results_df["Predicted"] = y_pred
        st.dataframe(results_df.head(10))

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📩 Download CSV", data=csv, file_name="hypertension_predictions.csv", mime="text/csv")

# =============================
# 🧾 Footer
# =============================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; font-size: 13px; color: #999;">
        Built with ❤️ by <strong>Tolulope Emuleomo</strong> aka <strong>Data Professor</strong> 🧠<br>
        🔗 <a href="https://twitter.com/dataprofessor_" style="color:#1DA1F2;">Twitter</a> |
        <a href="https://github.com/dataprofessor290" style="color:#6e5494;">GitHub</a> |
        <a href="https://linkedin.com/in/tolulope-emuleomo" style="color:#0A66C2;">LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
