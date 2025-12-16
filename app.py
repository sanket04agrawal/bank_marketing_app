import streamlit as st
import pandas as pd
import joblib

# Load model bundle
bundle = joblib.load("bank_marketing_model.pkl")
preprocessor = bundle["preprocessor"]
model = bundle["model"]
le = bundle["label_encoder"]

st.set_page_config(page_title="Bank Marketing Predictor", layout="centered")
st.title("📞 Bank Term Deposit Prediction")

st.markdown("Predict if a customer will subscribe to a **term deposit**")

# ---- User Inputs ----
age = st.number_input("Age", 18, 100)
job = st.selectbox("Job", ["admin.", "technician", "services", "management", "retired",
                           "blue-collar", "unemployed", "entrepreneur", "housemaid",
                           "self-employed", "student", "unknown"])

marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Default Credit", ["yes", "no"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])

balance = st.number_input("Account Balance")
duration = st.number_input("Last Call Duration (seconds)")
campaign = st.number_input("Number of Contacts")
pdays = st.number_input("Days Since Last Contact")
previous = st.number_input("Previous Contacts")

last_contact_month = st.number_input("Last Contact Month", 1, 12)
last_contact_day = st.number_input("Last Contact Day", 1, 31)

# ---- Prediction ----
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "balance": balance,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "last_contact_month": last_contact_month,
        "last_contact_day": last_contact_day
    }])

    X_processed = preprocessor.transform(input_df)
    pred = model.predict(X_processed)
    result = le.inverse_transform(pred)[0]

    if result == "yes":
        st.success("✅ Customer WILL subscribe to term deposit")
    else:
        st.error("❌ Customer will NOT subscribe")
