# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')

# Streamlit UI
st.title("Employee Salary Prediction App")

st.write("Enter the employee details below to predict their salary.")

# Input form
annual_rating = st.slider("Annual Performance Rating (0.0 to 5.0)", 0.0, 5.0, step=0.1)
experience = st.slider("Total Work Experience (in years)", 0, 30, step=1)

# Predict button
if st.button("Predict Salary"):
    input_data = pd.DataFrame([[annual_rating, experience]], columns=["AnnualRating", "TotalWorkExperience"])
    predicted_salary = model.predict(input_data)[0]
    st.success(f"Predicted Salary: ₹ {predicted_salary:,.2f}")
