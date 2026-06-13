# HR Employee Attrition Analysis & Prediction 📊

This project analyzes IBM's HR dataset to predict employee attrition (likelihood of an employee leaving the company) based on factors such as age, department, job satisfaction, and more.

## 🚀 Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://hribmemployeeattrition-satcewfnhmjaycdqpalsum.streamlit.app)


Run the interactive Streamlit app:

```bash
cd app
streamlit run app.py
```

The app provides two pages:
- **Analysis** — Visual exploration of the dataset with attrition distribution, age patterns, department trends, and correlation heatmaps
- **ML Prediction** — Input employee details via an interactive form to predict attrition risk using a trained Logistic Regression model


## 💡 Project Overview

- **Data Cleaning:** Removed redundant features (EmployeeCount, StandardHours, EmployeeNumber) and handled outliers using IQR clipping
- **Exploratory Data Analysis (EDA):** Deep-dive visualizations to understand drivers of turnover (age, department, overtime, gender)
- **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance
- **Machine Learning:** Built and evaluated a **Logistic Regression** model (L2 regularization) to classify potential leavers
- **Model Saving:** Trained model serialized to `src/models/logistic_regression.pkl` for reuse in the Streamlit app

## 🛠 Tech Stack

- **Python 3.x**
- **Pandas & NumPy** — Data manipulation and processing
- **Matplotlib & Seaborn** — Statistical visualizations
- **Plotly** — Interactive charts in the EDA notebook
- **Scikit-learn** — Model building, preprocessing, and evaluation metrics
- **Imbalanced-learn** — SMOTE implementation for class balancing
- **Streamlit** — Interactive web app for analysis and predictions

## 📂 Project Structure

```
HR_IBM Data/
├── app/                      # Streamlit web application
│   └── app.py
├── data/
│   ├── row/                  # Raw dataset (HR-Employee-Attrition.csv)
│   └── cleaned/              # Cleaned dataset (EDA_HR_Employee_Attrition.csv)
├── notebook/                 # Jupyter notebooks
│   ├── HR_IBM Analyis.ipynb         # EDA notebook
│   └── HR_IBM ML prediction.ipynb   # ML modeling notebook
├── src/
│   └── models/               # Saved model files
│       └── logistic_regression.pkl
├── reports/                  # Generated reports
├── requirements.txt
└── README.md
```

## 📈 Key Insights

1. **Age Factor:** Younger employees (in their 20s) show significantly higher attrition rates compared to older age groups.
2. **Departmental Trends:** R&D/Sales departments show the highest volume of employees leaving.
3. **Overtime Impact:** A significant portion of leavers worked overtime, suggesting work-life balance is a key driver.
4. **Distance from Home:** Employees living farther from the office are slightly more likely to leave.
5. **Gender Impact:** Males have a slightly higher turnover rate (~17% vs ~14.8%), but gender is not a primary driver.

## 🧠 Model Performance

The Logistic Regression model achieves competitive accuracy on the held-out test set, with evaluation metrics including accuracy, confusion matrix, and ROC-AUC score (see the ML prediction notebook for details).
