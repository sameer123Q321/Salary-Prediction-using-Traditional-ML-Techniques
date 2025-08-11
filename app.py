import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("salary_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("💼 Salary Prediction system")
st.write("Predict salary based on Age, Education Level, and Years of Experience.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=70, step=1)
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)

# Education encoding
edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
education_encoded = edu_map[education_level]

# # Predict button
# if st.button("Predict Salary"):
#     input_data = np.array([[age, education_encoded, years_exp]])
#     input_scaled = scaler.transform(input_data)
#     prediction = model.predict(input_scaled)[0]
#     st.success(f"Estimated Salary: ${prediction:,.2f}")
usd_to_pkr = 278  # Conversion rate

if st.button("Predict Salary"):
    input_data = np.array([[age, education_encoded, years_exp]])
    input_scaled = scaler.transform(input_data)
    prediction_usd = model.predict(input_scaled)[0]
    prediction_pkr = prediction_usd * usd_to_pkr
    st.success(f"Estimated Salary: PKR {prediction_pkr:,.0f}")