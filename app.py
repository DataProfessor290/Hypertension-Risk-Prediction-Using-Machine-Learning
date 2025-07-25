import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# =============================
# 🚀 Load Model and Dataset
# =============================
model = joblib.load("hypertension_model_v2.pkl") 

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.columns = df.columns.str.lower().str.strip()
    df["medication"] = df["medication"].fillna("None")
    df["has_hypertension"] = df["has_hypertension"].replace({"Yes": 1, "No": 0})
    return df

df = load_data()

# =============================
# 🎨 Global Styling - Dark Mode
# =============================
st.set_page_config(page_title="Hypertension Risk Predictor", layout="centered")

st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #111111;
        color: #F5F5F5;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, label, .stText, .markdown-text-container {
        color: #F5F5F5 !important;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 6px;
        padding: 0.6em 1.2em;
    }
    .stDownloadButton>button {
        background-color: #117A65;
        color: white;
        border-radius: 6px;
    }
    .stMetricLabel, .stMetricValue {
        color: #F5F5F5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# 💡 Header
# =============================
st.markdown("<h1 style='text-align: center;'>🩺 Hypertension Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #B0BEC5;'>Powered by XGBoost | Built with Streamlit</h5>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 1px solid #555;'>", unsafe_allow_html=True)

# =============================
# 🔢 Input Features
# =============================
st.markdown("### 🔍 Enter Patient Information:")

col1, col2 = st.columns(2)

with col1:
    bmi = st.number_input("💪 BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    stress_score = st.slider("😖 Stress Score", 0, 10, 5)

with col2:
    family_history = st.selectbox("👪 Family History of Hypertension?", ["yes", "no"])
    smoking_status = st.selectbox("🚬 Smoking Status", ["Never", "Former", "Current"])

bp_history = st.selectbox("💓 Blood Pressure History", ["Normal", "Elevated", "Stage 1", "Stage 2"])

# =============================
# 🎯 Prediction Logic
# =============================
input_df = pd.DataFrame({
    "bmi": [bmi],
    "family_history": [family_history.lower()],
    "smoking_status": [smoking_status],
    "stress_score": [stress_score],
    "bp_history": [bp_history]
})

st.markdown("---")
st.markdown("### 📊 Prediction Result")

if st.button("🔍 Predict Risk Level"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk of Hypertension\n\n**Probability: {probability:.2f}%**")
        st.markdown("*Please consult a healthcare professional for further screening.*")
    else:
        st.success(f"✅ Low Risk of Hypertension\n\n**Probability: {probability:.2f}%**")
        st.markdown("*Keep up your healthy lifestyle!*")

# =============================
# 📈 Model Performance
# =============================
st.markdown("---")
st.subheader("📈 Model Accuracy (on Test Set)")

feature_cols = ["bmi", "family_history", "smoking_status", "stress_score", "bp_history"]

X = df[feature_cols]
y = df["has_hypertension"]
X_clean = X.dropna()
y_clean = y.loc[X_clean.index]

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")

# =============================
# 📥 Download Section
# =============================
if st.checkbox("📂 Show & Download Predictions"):
    results_df = X_test.copy()
    results_df["Actual"] = y_test.values
    results_df["Predicted"] = y_pred
    st.dataframe(results_df.head(10))

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name='hypertension_predictions.csv',
        mime='text/csv'
    )

# =============================
# 🧾 Footer
# =============================
st.markdown("""<hr style='border-top: 1px solid #555;'>""", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-size: 0.9em; color: #95A5A6; padding: 10px;'>
        Made with ❤️ by <b style='color: #EC7063;'>Tolulope Emuleomo</b><br>
        <span>Data Scientist | Healthcare Analyst</span><br><br>
        <i>“Empowering healthcare through data-driven insights.”</i>
    </div>
""", unsafe_allow_html=True)
