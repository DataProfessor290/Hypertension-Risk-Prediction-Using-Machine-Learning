# hypertension_bmi_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# =============================
# 🚀 Load Model and Dataset
# =============================
model = joblib.load("hypertension_model_v2.pkl")  # Ensure model is in directory

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
        background-color: #0e1117;
        color: #F5F5F5;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, label, .stText, .markdown-text-container {
        color: #F5F5F5 !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 6px;
        padding: 0.6em 1.2em;
    }
    .stMetricLabel, .stMetricValue {
        color: #F5F5F5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# ⚖️ BMI Calculator in Sidebar
# =============================
st.sidebar.header("⚖️ BMI Calculator")
st.sidebar.markdown("Don't know your BMI? Calculate it here:")

weight = st.sidebar.number_input("Enter your weight (kg)", min_value=10.0, step=0.5, format="%.1f")
height = st.sidebar.number_input("Enter your height (meters)", min_value=0.5, step=0.01, format="%.2f")

if height > 0:
    calculated_bmi = round(weight / (height ** 2), 2)
    st.sidebar.success(f"Your BMI is: **{calculated_bmi}**")
else:
    calculated_bmi = 25.0

st.sidebar.caption("\ud83d\udca1 BMI = weight (kg) \u00f7 height\u00b2 (m\u00b2)")

# =============================
# 💡 Header & BMI Explanation
# =============================
st.markdown("<h1 style='text-align: center;'>\ud83e\udda0 Hypertension Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #B0BEC5;'>Powered by XGBoost | Built with Streamlit</h5>", unsafe_allow_html=True)

st.markdown("""
#### 📚 What is BMI?
Body Mass Index (BMI) is a simple calculation using your height and weight. It's used to categorize your weight as:

- ⚠️ *Underweight*: BMI < 18.5
- ✅ *Normal weight*: 18.5 ≤ BMI < 25
- ⚠️ *Overweight*: 25 ≤ BMI < 30
- 🔶 *Obese I*: 30 ≤ BMI < 35
- 🔴 *Obese II*: 35 ≤ BMI < 40
- 🔴 *Obese III*: BMI ≥ 40
""", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid #555;'>", unsafe_allow_html=True)

# =============================
# 🔢 Input Features
# =============================
st.markdown("### \ud83d\udd0d Enter Patient Information:")

col1, col2 = st.columns(2)

with col1:
    bmi = st.number_input("\ud83d\udcaa BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=calculated_bmi, step=0.1)
    st.caption("\u2139\ufe0f *BMI is calculated as weight in kg divided by square of height in meters.*")
    stress_score = st.slider("\ud83d\ude16 Stress Score", 0, 10, 5)
    st.caption("\u2139\ufe0f *Rate your average stress level from 0 (none) to 10 (very high).*")

with col2:
    family_history = st.selectbox("\ud83d\udec6 Family History of Hypertension?", ["yes", "no"])
    st.caption("\u2139\ufe0f *Do your parents or siblings have hypertension?*")
    smoking_status = st.selectbox("\ud83d\udeac Smoking Status", ["Never", "Former", "Current"])
    st.caption("\u2139\ufe0f *Current or past smoking habits?*")

bp_history = st.selectbox("\ud83d\udc93 Blood Pressure History", ["Normal", "Elevated", "Stage 1", "Stage 2"])
st.caption("\u2139\ufe0f *Based on your last clinical BP check.*")

# =============================
# \ud83c\udfaf Prediction Logic
# =============================
input_df = pd.DataFrame({
    "bmi": [bmi],
    "family_history": [family_history.lower()],
    "smoking_status": [smoking_status],
    "stress_score": [stress_score],
    "bp_history": [bp_history]
})

st.markdown("---")
st.markdown("### \ud83d\udcca Prediction Result")

if st.button("\ud83d\udd0d Predict Risk Level"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"\u26a0\ufe0f High Risk of Hypertension\n\n**Probability: {probability:.2f}%**")
        st.markdown("*Please consult a healthcare professional for further screening.*")
    else:
        st.success(f"\u2705 Low Risk of Hypertension\n\n**Probability: {probability:.2f}%**")
        st.markdown("*Keep up your healthy lifestyle!*")

# =============================
# \ud83d\udcc8 Model Performance
# =============================
st.markdown("---")
st.subheader("\ud83d\udcc8 Model Accuracy (on Test Set)")

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
# \ud83d\udcc5 Download Section
# =============================
if st.checkbox("\ud83d\udcc2 Show & Download Predictions"):
    results_df = X_test.copy()
    results_df["Actual"] = y_test.values
    results_df["Predicted"] = y_pred
    st.dataframe(results_df.head(10))

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="\ud83d\udcc5 Download Predictions CSV",
        data=csv,
        file_name='hypertension_predictions.csv',
        mime='text/csv'
    )

# =============================
# \ud83d\udcdd Footer
# =============================
st.markdown("---")
st.markdown(
    """
    <div style="font-size: 13px; color: #999999; text-align: center;">
        Built with ❤️ using <a href="https://streamlit.io" target="_blank" style="color: #1f77b4;">Streamlit</a><br>
        Based on WHO BMI classification<br><br>
        Created by <strong>Tolulope Emuleomo</strong> aka <strong>Data Professor</strong> 🧠<br>
        🔗 <a href="https://twitter.com/dataprofessor_" target="_blank" style="color: #1DA1F2;">@dataprofessor_</a> |
        <a href="https://github.com/dataprofessor290" target="_blank" style="color: #6e5494;">GitHub</a> |
        <a href="https://linkedin.com/in/tolulope-emuleomo" target="_blank" style="color: #0e76a8;">LinkedIn</a><br>
        💼 <span style="color: #cccccc;">Data Scientist</span>
    </div>
    """,
    unsafe_allow_html=True
)
