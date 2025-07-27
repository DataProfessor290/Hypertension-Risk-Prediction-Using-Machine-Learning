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
st.sidebar.markdown("_Don't know your BMI? Calculate it here:_")

weight = st.sidebar.number_input("Enter your weight (kg)", min_value=10.0, step=0.5, format="%.1f")
height = st.sidebar.number_input("Enter your height (meters)", min_value=0.5, step=0.01, format="%.2f")

if height > 0:
    calculated_bmi = round(weight / (height ** 2), 2)
    st.sidebar.success(f"Your BMI is: **{calculated_bmi}**")
else:
    calculated_bmi = 25.0  # fallback value

st.sidebar.markdown("💡 _BMI = weight (kg) ÷ height² (m²)_")

# =============================
# 💡 Header & Explanation
# =============================
st.markdown("<h1 style='text-align: center;'>🧠 Hypertension Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #B0BEC5;'>Powered by XGBoost | Built with Streamlit</h5>", unsafe_allow_html=True)

st.markdown("""
#### 📚 What is BMI?
Body Mass Index (BMI) is a simple calculation using your height and weight. It's used to categorize your weight as:

- ⚠️ *Underweight*: BMI < 18.5  
- ✅ *Normal*: 18.5 ≤ BMI < 25  
- ⚠️ *Overweight*: 25 ≤ BMI < 30  
- 🔶 *Obese I*: 30 ≤ BMI < 35  
- 🔴 *Obese II*: 35 ≤ BMI < 40  
- 🔴 *Obese III*: BMI ≥ 40  
""", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid #555;'>", unsafe_allow_html=True)

# =============================
# 🔢 Input Features
# =============================
st.markdown("### 🔎 Enter Patient Information:")

col1, col2 = st.columns(2)

with col1:
    bmi = st.number_input("💪 BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=calculated_bmi, step=0.1)
    st.caption("_💡 BMI is calculated as weight in kg divided by height² in meters._")
    stress_score = st.slider("😰 Stress Score", 0, 10, 5)
    st.caption("_💡 Rate your average stress level from 0 (none) to 10 (very high)._")

with col2:
    family_history = st.selectbox("👪 Family History of Hypertension?", ["yes", "no"])
    st.caption("_💡 Do your parents or siblings have hypertension?_")
    smoking_status = st.selectbox("🚬 Smoking Status", ["Never", "Former", "Current"])
    st.caption("_💡 What's your smoking history?_")

bp_history = st.selectbox("💓 Blood Pressure History", ["Normal", "Elevated", "Stage 1", "Stage 2"])
st.caption("_💡 Based on your last clinical BP check._")

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
        st.error(f"⚠️ **High Risk of Hypertension**\n\n🧮 Probability: **{probability:.2f}%**")
        st.markdown("*Please consult a healthcare professional for further screening.*")
    else:
        st.success(f"✅ **Low Risk of Hypertension**\n\n🧮 Probability: **{probability:.2f}%**")
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
# 📥 Download Predictions
# =============================
if st.checkbox("📂 Show & Download Predictions"):
    results_df = X_test.copy()
    results_df["Actual"] = y_test.values
    results_df["Predicted"] = y_pred
    st.dataframe(results_df.head(10))

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Predictions CSV",
        data=csv,
        file_name='hypertension_predictions.csv',
        mime='text/csv'
    )

# =============================
# 📝 Footer
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
