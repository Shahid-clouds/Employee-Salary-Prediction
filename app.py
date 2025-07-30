import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("salary_prediction_model.pkl")

# Set the app title
st.title("Employee Salary Prediction App")

st.markdown("""
Enter the employee's annual rating and total work experience to predict their salary.
""")

# Input fields
annually_rating = st.number_input("Annually Rating (e.g., 4.2)", min_value=0.0, max_value=5.0, step=0.1)
total_experience = st.number_input("Total Working Experience (in years)", min_value=0, step=1)

# Predict button
if st.button("Predict Salary"):
    # Create a DataFrame with correct column names (MUST match training data)
    input_df = pd.DataFrame({
        "Annually Rating": [annually_rating],
        "Total Working Experience": [total_experience]
    })

    # Predict salary
    predicted_salary = model.predict(input_df)[0]

    # Show result
    st.success(f"Predicted Salary: ₹{predicted_salary:,.2f}")
