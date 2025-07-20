import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Utility to load the model only once
@st.cache_resource
def load_model():
    return joblib.load("salary_pipeline.pkl")

@st.cache_data
def load_reference_data():
    df = pd.read_csv("adult 3.csv").replace("?", pd.NA).dropna()
    return df

pipe = load_model()
df_ref = load_reference_data()

# UI Styles
st.markdown("""
    <style>
    .main .block-container{padding-top:1.5rem;}
    .stButton>button {background-color:#0284c7;color:white;}
    .sidebar-content {font-size: 0.95em;}
    </style>
""", unsafe_allow_html=True)

st.title("💸 Employee Salary Prediction")
st.subheader("Predict whether an employee's salary is >50K or <=50K")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("Enter employee details to predict whether the annual salary is **greater than $50K or not**. Powered by Random Forest and real census data.")
    st.write("By [Your Name] — July 2025")
    st.success(
        'Tip: Select realistic values based on education, occupation, and hours worked to see how they affect salary predictions!'
    )
    st.markdown("---")
    st.info("Model trained on U.S. adult census and workforce data.")

# Pre-fill selects from reference data to avoid typos and keep categories current
def get_options(col):
    return sorted(df_ref[col].dropna().unique())

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", int(df_ref.age.min()), int(df_ref.age.max()), 30)
    workclass = st.selectbox("Workclass", get_options("workclass"), index=0)
    education = st.selectbox("Education", get_options("education"), index=0)
    marital = st.selectbox("Marital Status", get_options("marital-status"), index=0)
    occupation = st.selectbox("Occupation", get_options("occupation"), index=0)
    relationship = st.selectbox("Relationship", get_options("relationship"), index=0)
    race = st.selectbox("Race", get_options("race"), index=0)
    gender = st.radio("Gender", get_options("gender"), index=0)

with col2:
    fnlwgt = st.number_input("fnlwgt", min_value=0, value=int(df_ref.fnlwgt.median()))
    ednum = st.slider("Educational-num", int(df_ref["educational-num"].min()), int(df_ref["educational-num"].max()), 9)
    gain = st.number_input("Capital-gain", min_value=0, value=0)
    loss = st.number_input("Capital-loss", min_value=0, value=0)
    hours = st.slider("Hours per week", int(df_ref["hours-per-week"].min()), int(df_ref["hours-per-week"].max()), 40)
    country = st.selectbox("Native Country", get_options("native-country"), index=0)

input_data = pd.DataFrame({
    "age": [age],
    "workclass": [workclass],
    "fnlwgt": [fnlwgt],
    "education": [education],
    "educational-num": [ednum],
    "marital-status": [marital],
    "occupation": [occupation],
    "relationship": [relationship],
    "race": [race],
    "gender": [gender],
    "capital-gain": [gain],
    "capital-loss": [loss],
    "hours-per-week": [hours],
    "native-country": [country],
})

st.markdown("---")
if st.button("🔎 Predict Salary Class", type="primary"):
    pred = pipe.predict(input_data)[0]
    proba = pipe.predict_proba(input_data)[0, 1]  # Probability of >50K
    if pred == 1:
        st.success(f"Likely Salary: **> $50K/year**  \n(Confidence: {proba:.2%})")
        st.balloons()
    else:
        st.info(f"Likely Salary: **<= $50K/year**  \n(Confidence: {1-proba:.2%})")
        st.markdown(
            "<span style='color:#c026d3;font-size:1.1em'>Try increasing education, occupation, or hours worked to see if it changes the result!</span>",
            unsafe_allow_html=True,
        )
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Get reference labels and convert to int
y_true = df_ref["income"]    # Your true labels
y_pred = pipe.predict(df_ref.drop(columns=["income"]))  # Your model's predictions
if y_true.dtype == object or y_true.dtype.name == "category":
    y_true_int = (y_true == '>50K').astype(int)
else:
    y_true_int = y_true.astype(int)

X_ref = df_ref.drop(columns=["income"])
y_true = df_ref["income"]
y_true_int = (y_true == '>50K').astype(int)
y_pred = pipe.predict(X_ref)
y_prob = pipe.predict_proba(X_ref)[:, 1]


from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
)

st.header("📊 Model Evaluation – Five Key Plots")

# 1. Confusion Matrix
st.subheader("1️⃣ Confusion Matrix")
cm = confusion_matrix(y_true_int, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'], ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# 2. ROC Curve
st.subheader("2️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_true_int, y_prob)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# 3. Precision-Recall Curve
st.subheader("3️⃣ Precision-Recall Curve")
prec, rec, _ = precision_recall_curve(y_true_int, y_prob)
fig_pr, ax_pr = plt.subplots()
ax_pr.plot(rec, prec, color="purple", lw=2)
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve")
st.pyplot(fig_pr)

# 4. Cumulative Gains / Lift Curve
st.subheader("4️⃣ Cumulative Gains / Lift Chart")
# For Cumulative Gains: sort by predicted proba, plot cumulative positive rate
order = np.argsort(-y_prob)
y_true_sorted = np.array(y_true_int)[order]
cumulative_true_positives = np.cumsum(y_true_sorted)
percent_population = np.arange(1, len(y_true_sorted)+1) / len(y_true_sorted)
percent_positives = cumulative_true_positives / cumulative_true_positives[-1]

fig_lift, ax_lift = plt.subplots()
ax_lift.plot(percent_population, percent_positives, label='Model', color="green")
ax_lift.plot([0,1], [0,1], 'k--', label='Random')
ax_lift.set_xlabel('Proportion of data (sorted by score)')
ax_lift.set_ylabel('Cumulative proportion of >50K')
ax_lift.set_title('Cumulative Gains / Lift Chart')
ax_lift.legend()
st.pyplot(fig_lift)

# 5. Probability Histogram
st.subheader("5️⃣ Probability Score Histogram")
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(y_prob[y_true_int==1], bins=25, alpha=0.7, label='>50K', color='gold')
ax_hist.hist(y_prob[y_true_int==0], bins=25, alpha=0.7, label='<=50K', color='cornflowerblue')
ax_hist.set_xlabel('Predicted Probability of >50K')
ax_hist.set_ylabel('Number of Samples')
ax_hist.set_title('Probability Score Histogram')
ax_hist.legend()
st.pyplot(fig_hist)
