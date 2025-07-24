# 🩺 Hypertension Risk Prediction using Machine Learning

This project aims to **predict the risk of hypertension** based on an individual's lifestyle and health-related factors using various **supervised machine learning models**. Hypertension is a critical public health issue, and early detection can play a vital role in preventing complications such as heart disease, stroke, and kidney failure.


---

## 🧠 Objective

To build and compare multiple machine learning models that can **accurately classify individuals** as hypertensive or non-hypertensive based on features such as:
- Age
- Salt Intake
- Stress Level
- Blood Pressure History
- Sleep Duration
- Body Mass Index (BMI)
- Medication
- Family History
- Exercise Level
- Smoking Status

---

## 📊 Dataset Overview

- **Total Records**: 1985
- **Target Variable**: `Has_Hypertension` (Yes/No)
- **Missing Values**: 799 missing values in the `Medication` column
- **Duplicates**: None
- **Class Imbalance**: Originally imbalanced (1032 Yes / 953 No)

---

## 🧹 Data Cleaning and Preparation

- **Standardized column names** to lowercase and stripped whitespaces.
- **Missing Values**: Filled `Medication` missing values with `"None"`.
- **Target Encoding**: Converted `Has_Hypertension` from `Yes/No` to `1/0`.
- **Class Balancing**: Applied **Random UnderSampling** to balance the classes (953 each).
- **Feature Engineering**: Separated features into numeric and categorical for proper preprocessing.

---

## **🧪 Machine Learning Models**


| Model                | Accuracy | Precision | Recall  | F1-Score |
| -------------------- | -------- | --------- | ------- | -------- |
| Logistic Regression  | 82%      | 84%       | 81%     | 82%      |
| Decision Tree        | 95%      | 93%       | 96%     | 95%      |
| Random Forest        | 96%      | 98%       | 94%     | 96%      |
| Support Vector (SVC) | 88%      | 89%       | 89%     | 88%      |
| K-Nearest Neighbors  | 80%      | 89%       | 70%     | 78%      |
| **XGBoost**          | **99%**  | **99%**   | **99%** | **99%**  |

**✅ Best Model:** XGBoost, which achieved near-perfect performance with minimal false positives and negatives.


## **🧬 Feature Importance (from XGBoost)**

### **Top 5 contributing features:**
Rank	Feature	Importance Score
1. bmi	0.37
2. family_history	0.13
3. smoking_status	0.12
4. stress_score	0.09
5. bp_history	0.07

## **🔍 Insight:**
- High **BMI** is the strongest predictor of hypertension in the dataset.

- **Family history** and smoking are also significant contributors.

- **Stress** and **Blood Pressure History** play meaningful but smaller roles.

## **📈 Evaluation Metrics (for XGBoost)**
**Classification Report:**
              precision    recall  f1-score   support
         0       0.99      0.99      0.99       188
         1       0.99      0.99      0.99       194
    accuracy                           0.99       382

**Confusion Matrix:**
[[187   1]
 [  1 193]]

## **📌 Key Takeaways**
**BMI** is a critical variable—encouraging weight management may lower hypertension risk.

Individuals with a **family history** of hypertension or who **smoke** are more likely to be at risk.

Models like **Random Forest** and **XGBoost** are highly effective for this classification problem.

**Data preprocessing** and **balancing** significantly improved model performance.

## **🧭 Recommendations**
**Preventive Screening:**

Use these predictive insights to design early intervention programs, especially for individuals with high BMI or family history.

**Public Health Campaigns:**

Encourage healthier lifestyles (diet, exercise, sleep) to mitigate modifiable risks.

**Model Deployment:**

The trained XGBoost model can be deployed into a health diagnostic app or dashboard for real-time predictions.

## **🚀 Future Work**
- Implement SHAP values for more interpretable model explanations.

- Try ensemble stacking for potentially higher accuracy.

- Explore deep learning models if more data becomes available.

- Build a web app using Streamlit for real-time user predictions.

## **🧰 Technologies Used**
- Python

- Pandas, NumPy

- Scikit-learn

- Imbalanced-learn (RandomUnderSampler)

- XGBoost

- Matplotlib, Seaborn

**📝 Author**
**Tolulope Emuleomo**

🔗 [LinkedIn](https://www.linkedin.com/in/tolulope-emuleomo-77a231270/)   
🔗 [Medium](https://medium.com/@dataprofessor_)

