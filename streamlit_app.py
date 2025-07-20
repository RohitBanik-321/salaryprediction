import streamlit as st
import pandas as pd
import joblib
import gzip
import joblib


# Load the trained model
with gzip.open("salary_prediction_model_compressed.pkl.gz", "rb") as f:
    model = joblib.load(f)

# Title
st.title("Employee Salary Prediction App")
st.write("Predict whether an employee earns more than 50K per year.")

# Input fields
age = st.number_input("Age", min_value=17, max_value=90, value=30)
fnlwgt = st.number_input("FNLWGT (final weight)", min_value=10000, max_value=1500000, value=100000)
education_num = st.slider("Education Number (1-16)", 1, 16, 10)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.slider("Hours per Week", 1, 99, 40)

# Categorical fields
workclass = st.selectbox("Workclass", [
    "Federal-gov", "Local-gov", "Private", "Self-emp-inc", "Self-emp-not-inc",
    "State-gov", "Without-pay", "Never-worked"
])
education = st.selectbox("Education", [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
    "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th",
    "10th", "Doctorate", "5th-6th", "Preschool"
])
marital_status = st.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
])
relationship = st.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
gender = st.selectbox("Gender", ["Male", "Female"])
native_country = st.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "England",
    "Cuba", "Jamaica", "China", "South", "Puerto-Rico", "Honduras"
])

# Create input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'fnlwgt': [fnlwgt],
    'educational-num': [education_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'workclass_' + workclass: [1],
    'education_' + education: [1],
    'marital-status_' + marital_status: [1],
    'occupation_' + occupation: [1],
    'relationship_' + relationship: [1],
    'race_' + race: [1],
    'gender_' + gender: [1],
    'native-country_' + native_country: [1]
})

# Fill missing model columns with 0
model_input_columns = model.feature_names_in_
for col in model_input_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_input_columns]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Salary: {result}")
