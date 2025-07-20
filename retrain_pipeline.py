import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("adult 3.csv").replace("?", np.nan).dropna()
X = df.drop("income", axis=1)
y = df["income"].map({"<=50K": 0, ">50K": 1})

categorical = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]
numerical = [
    'age', 'fnlwgt', 'educational-num', 'capital-gain',
    'capital-loss', 'hours-per-week'
]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numerical)
])
pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=200, solver='liblinear', random_state=42))
])
pipe.fit(X, y)
joblib.dump(pipe, "salary_pipeline.pkl")
print("Lightweight model saved as salary_pipeline.pkl")
