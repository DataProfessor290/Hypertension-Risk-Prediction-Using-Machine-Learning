import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# =============================
# 🚀 Load Model and Dataset
# =============================
model = joblib.load("hypertension_model_v2.pkl")  # Your saved model
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.columns = df.columns.str.lower().str.strip()
    df["medication"] = df["medication"].fillna("None")
    df["has_hypertension"] = df["has_hypertension"].replace({"Yes": 1, "No": 0})
    return df

df = load_data()

# =============================
# 🎨 Page Config & Styling
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
        padding: 0.4em 1em;
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
# ⚖️ BMI Calculator in Sidebar
# =============================
st.sidebar.header("⚖️ BMI Calculator")

with st.sidebar:
    with st.expander("📘 Don't know your BMI? Calculate it here!", expanded=True):
        weight = st.number_input("Weight (kg)", min_value=10.0, step=0.5, format="%.1f", help="E.g. 65.0")
        height = st.number_input("Height (m)", min_value=0.5, step=0.01, format="%.2f", help="E.g. 1.70")
        calc_bmi = st.button("📏 Calculate BMI")
        
        if calc_bmi and height > 0:
            calculated_bmi = round(weight / (height ** 2), 2)
            st.success(f"✅ Your BMI is: **{calculated_bmi}**")
        elif not calc_bmi:
            calculated_bmi = 25.0  # Default value before user hits button
        else:
            calculated_bmi = 25.0
            st.warning("Please enter a valid height.")
        
        st.markdown("💡 _Formula: BMI = weight (kg) ÷ height² (m²)_")

        with st.expander("📚 BMI Classification", expanded=False):
            st.markdown("""
            - ⚠️ **Underweight**: BMI < 18.5  
            - ✅ **Normal**: 18.5 ≤ BMI < 25  
            - ⚠️ **Overweight**: 25 ≤ BMI < 30  
            - 🔶 **Obese I**: 30 ≤ BMI < 35  
            - 🔴 **Obese II**: 35 ≤ BMI < 40  
            - 🔴 **Obese III**: BMI ≥ 40
            """)

# =============================
# 🩺 Hypertension Risk Predictor
# =============================
st.markdown("<h1 style='text-align: center;'>🯪 Hypertension Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #B0BEC5;'>Powered by XGBoost | Built with Streamlit</h5>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 1px solid #555;'>", unsafe_allow_html=True)

st.markdown("### 📝 Patient Information")

col1, col2 = st.columns(2)

with col1:
    bmi = st.number_input("💪 BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=calculated_bmi, step=0.1)
    stress_score = st.slider("😖 Stress Score", 0, 10, 5)

with col2:
    family_history = st.selectbox("👪 Family History of Hypertension?", ["yes", "no"])
    smoking_status = st.selectbox("🚬 Smoking Status", ["Never", "Former", "Current"])

bp_history = st.selectbox("💓 Blood Pressure History", ["Normal", "Elevated", "Stage 1", "Stage 2"])

# =============================
# 🎯 Prediction
# =============================
input_df = pd.DataFrame({
    "bmi": [bmi],
    "family_history": [family_history.lower()],
    "smoking_status": [smoking_status],
    "stress_score": [stress_score],
    "bp_history": [bp_history]
})

st.markdown("### 📊 Prediction Result")
if st.button("🔍 Predict Risk Level"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk of Hypertension\n\n**Probability: {probability:.2f}%**")
        st.markdown("*Please consult a healthcare professional for further advice.*")
    else:
        st.success(f"✅ Low Risk of Hypertension\n\n**Probability: {probability:.2f}%**")
        st.markdown("*Great job! Keep maintaining a healthy lifestyle.*")

# =============================
# 🧠 Model Accuracy
# =============================
with st.expander("📈 View Model Accuracy", expanded=False):
    st.subheader("📊 Model Performance on Test Data")

    features = ["bmi", "family_history", "smoking_status", "stress_score", "bp_history"]
    X = df[features]
    y = df["has_hypertension"]
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{acc * 100:.2f}%")

# =============================
# 📥 Download Section
# =============================
with st.expander("📂 Download Sample Predictions", expanded=False):
    result_df = X_test.copy()
    result_df["Actual"] = y_test.values
    result_df["Predicted"] = y_pred
    st.dataframe(result_df.head(10))

    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv_data, file_name="hypertension_predictions.csv", mime="text/csv")

# =============================
# 🦶 Footer
# =============================
st.markdown("---")
st.markdown("""
<div style="font-size: 13px; color: #888888; text-align: center;">
    Built with ❤️ using <a href="https://streamlit.io" target="_blank" style="color:#1f77b4;">Streamlit</a><br>
    Based on WHO BMI standards<br><br>
    Created by <strong>Tolulope Emuleomo</strong> (aka <strong>Data Professor</strong>)<br>
    🔗 <a href="https://twitter.com/dataprofessor_" style="color:#1DA1F2;" target="_blank">@dataprofessor_</a> |
    <a href="https://github.com/dataprofessor290" style="color:#6e5494;" target="_blank">GitHub</a> |
    <a href="https://www.linkedin.com/in/tolulope-emuleomo" style="color:#0A66C2;" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
