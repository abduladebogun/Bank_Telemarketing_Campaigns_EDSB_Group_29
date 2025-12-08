
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
import csv

# Load artifacts
model = joblib.load("./artifacts/final_xgb_model.pkl")
preprocessor = joblib.load("./artifacts/preprocessor.pkl")

FINAL_THRESHOLD = 0.30

st.title("Telemarketing Response Prediction App")
st.write("Upload a CSV of customer records to score likelihood of saying YES.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])


# Function to add engineered features to input dataframe

def engineer_features(df):
    df = df.copy()

    # 1. age_group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 45, 65, 120],
        labels=["Youth", "Adult", "Middle_Age", "Senior"]
    )

    # 2. total_contacts
    df["total_contacts"] = df["campaign"] + df["previous"]

    # 3. prev_success
    df["prev_success"] = np.where(df["poutcome"] == "success", 1, 0)

    # 4. prev_contacted
    df["prev_contacted"] = np.where(df["previous"] > 0, 1, 0)

    # 6. pdays_adjusted
    df["pdays_adjusted"] = df["pdays"].replace(999, -1)

    return df


# Streamlit main logic

if uploaded:

    # Read small sample for delimiter detection
    sample = uploaded.read(2048).decode("utf-8")
    uploaded.seek(0)  # reset pointer

    dialect = csv.Sniffer().sniff(sample)

    df = pd.read_csv(uploaded, sep=dialect.delimiter)

    # Apply feature engineering
    df_eng = engineer_features(df)

    # Preprocess + Score
    # X_processed = preprocessor.transform(df_eng)
    proba = model.predict_proba(df_eng)[:, 1]
    preds = (proba >= FINAL_THRESHOLD).astype(int)

    output = df_eng.copy()
    output["probability_of_yes"] = proba
    output["prediction"] = preds

    st.subheader("Prediction Results")
    st.write(output)

    st.download_button(
        label="Download predictions",
        data=output.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
