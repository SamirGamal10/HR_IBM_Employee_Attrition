import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

st.set_page_config(page_title="HR IBM Employee Attrition", layout="wide")

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE, "..", "data", "cleaned", "EDA_HR_Employee_Attrition.csv")
RAW_PATH = os.path.join(_BASE, "..", "data", "row", "HR-Employee-Attrition.csv")
MODEL_PATH = os.path.join(_BASE, "..", "src", "models", "logistic_regression.pkl")

DROP_COLS = ["Unnamed: 0", "EmployeeCount", "StandardHours", "EmployeeNumber", "Attrition"]

LABEL_MAP = {
    "BusinessTravel": {"Non-Travel": 0, "Travel_Frequently": 1, "Travel_Rarely": 2},
    "Department": {"Human Resources": 0, "Research & Development": 1, "Sales": 2},
    "EducationField": {"Human Resources": 0, "Life Sciences": 1, "Marketing": 2, "Medical": 3, "Other": 4, "Technical Degree": 5},
    "Gender": {"Female": 0, "Male": 1},
    "JobRole": {"Healthcare Representative": 0, "Human Resources": 1, "Laboratory Technician": 2, "Manager": 3, "Manufacturing Director": 4, "Research Director": 5, "Research Scientist": 6, "Sales Executive": 7, "Sales Representative": 8},
    "MaritalStatus": {"Divorced": 0, "Married": 1, "Single": 2},
    "OverTime": {"No": 0, "Yes": 1},
}

REVERSE_MAP = {col: {v: k for k, v in m.items()} for col, m in LABEL_MAP.items()}

NUM_FEATURES = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
CAT_FEATURES = list(LABEL_MAP.keys())

FEATURE_ORDER = ["Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_raw():
    return pd.read_csv(RAW_PATH)

st.sidebar.markdown("# \U0001f3e2 HR IBM Employee Attrition")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["\U0001f4ca Analysis", "\U0001f52e ML Prediction"])

if page == "\U0001f4ca Analysis":
    st.title("\U0001f4ca HR IBM Employee Attrition Analysis")
    df = load_data()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", f"{len(df):,}")
    attrition_rate = df["Attrition"].value_counts(normalize=True).get("Yes", 0) * 100
    col2.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    col3.metric("Features", f"{df.shape[1]}")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        df["Attrition"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
        ax.set_xticklabels(["Stayed", "Left"], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with c2:
        st.subheader("Age Distribution by Attrition")
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, color in [("No", "green"), ("Yes", "red")]:
            subset = df[df["Attrition"] == label]["Age"]
            sns.kdeplot(subset, label=f"{'Stayed' if label=='No' else 'Left'}", color=color, fill=True, alpha=0.4, ax=ax)
        ax.set_xlabel("Age")
        st.pyplot(fig)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Attrition by Department")
        dept_att = df.groupby("Department")["Attrition"].value_counts(normalize=True).unstack() * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        if "Yes" in dept_att.columns:
            dept_att["Yes"].sort_values().plot(kind="barh", color="red", ax=ax)
            ax.set_xlabel("Attrition Rate (%)")
        st.pyplot(fig)
    with c4:
        st.subheader("Overtime vs Attrition")
        ot_att = df.groupby("OverTime")["Attrition"].value_counts(normalize=True).unstack() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        if "Yes" in ot_att.columns:
            ot_att["Yes"].plot(kind="bar", color=["green", "red"], ax=ax)
            ax.set_xticklabels(["No OT", "OT"], rotation=0)
            ax.set_ylabel("Attrition Rate (%)")
        st.pyplot(fig)
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(num_df.corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

else:
    st.title("\U0001f52e ML Prediction - Employee Attrition")
    model = load_model()
    st.markdown("### Enter Employee Details")
    with st.form("prediction_form"):
        cols = st.columns(3)
        inputs = {}
        cat_idx = 0
        num_idx = 0
        cat_items = list(LABEL_MAP.items())
        for ci, col in enumerate(FEATURE_ORDER):
            with cols[ci % 3]:
                if col in LABEL_MAP:
                    options = list(LABEL_MAP[col].keys())
                    inputs[col] = st.selectbox(col.replace("_", " "), options, key=f"cat_{col}")
                else:
                    r = {"Age": (18, 65, 36), "DailyRate": (100, 1500, 800), "DistanceFromHome": (1, 30, 7), "Education": (1, 5, 3), "EnvironmentSatisfaction": (1, 4, 3), "HourlyRate": (30, 100, 65), "JobInvolvement": (1, 4, 3), "JobLevel": (1, 5, 2), "JobSatisfaction": (1, 4, 3), "MonthlyIncome": (2000, 20000, 6500), "MonthlyRate": (2000, 27000, 14000), "NumCompaniesWorked": (0, 9, 2), "PercentSalaryHike": (10, 25, 15), "PerformanceRating": (1, 4, 3), "RelationshipSatisfaction": (1, 4, 3), "StockOptionLevel": (0, 3, 1), "TotalWorkingYears": (0, 40, 10), "TrainingTimesLastYear": (0, 6, 2), "WorkLifeBalance": (1, 4, 3), "YearsAtCompany": (0, 40, 7), "YearsInCurrentRole": (0, 18, 4), "YearsSinceLastPromotion": (0, 15, 2), "YearsWithCurrManager": (0, 17, 3)}
                    lo, hi, default = r.get(col, (0, 100, 50))
                    inputs[col] = st.slider(col.replace("_", " "), lo, hi, default, key=f"num_{col}")
        submitted = st.form_submit_button("Predict Attrition Risk", type="primary", use_container_width=True)
    if submitted:
        row = {}
        for col in FEATURE_ORDER:
            if col in LABEL_MAP:
                row[col] = LABEL_MAP[col][inputs[col]]
            else:
                row[col] = inputs[col]
        X = pd.DataFrame([row])[FEATURE_ORDER]
        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]
        st.markdown("---")
        res1, res2, res3 = st.columns([1, 2, 1])
        with res2:
            if pred == 1:
                st.error(f"\U000026a0 High Attrition Risk")
                st.metric("Probability of Leaving", f"{prob:.1%}")
            else:
                st.success(f"\U00002705 Low Attrition Risk")
                st.metric("Probability of Leaving", f"{prob:.1%}")
            st.markdown("#### Input Summary")
            summary = {}
            for col in FEATURE_ORDER:
                val = inputs[col]
                if col in REVERSE_MAP:
                    summary[col] = REVERSE_MAP[col].get(LABEL_MAP[col].get(val, 0), val)
                else:
                    summary[col] = val
            st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "Value"}), use_container_width=True)
