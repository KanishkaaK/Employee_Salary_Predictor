import streamlit as st
import pandas as pd
import joblib

# Load model with preprocessing pipeline
model = joblib.load("salary_predictor_model_clean.joblib")

# -----------------------------
# 🎨 Page Setup
# -----------------------------
st.set_page_config(page_title="Income Class Predictor", layout="centered")
st.title("💼 Income Class Predictor")
st.markdown("Predict if a person earns more than **$50K/year** based on demographics.")

# -----------------------------
# 📥 Input Form
# -----------------------------
with st.form("income_form"):
    st.markdown("### Enter Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 90, 30)
        fnlwgt = st.number_input("FNLWGT", 10000, 1000000, 200000)
        education = st.selectbox("Education", [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school'
        ])
        marital_status = st.selectbox("Marital Status", [
            'Married-civ-spouse', 'Divorced', 'Never-married',
            'Separated', 'Widowed', 'Married-spouse-absent'
        ])
        occupation = st.selectbox("Occupation", [
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
            'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
            'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
            'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
        ])
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 5000, 0)

    with col2:
        workclass = st.selectbox("Workclass", [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
            'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
        ])
        educational_num = st.slider("Education Num", 1, 16, 10)
        relationship = st.selectbox("Relationship", [
            'Wife', 'Own-child', 'Husband', 'Not-in-family',
            'Other-relative', 'Unmarried'
        ])
        race = st.selectbox("Race", [
            'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
        ])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        hours_per_week = st.slider("Hours/Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", [
            'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India'
        ])

    submitted = st.form_submit_button("🔍 Predict")

# -----------------------------
# 🧠 Prediction Logic
# -----------------------------
if submitted:
    # Format user input into DataFrame
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("💰 This person is likely to earn **more than $50K/year**.")
        else:
            st.info("💼 This person is likely to earn **$50K/year or less**.")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

