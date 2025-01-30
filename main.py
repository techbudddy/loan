import pickle
import streamlit as st
import numpy as np

# Load the trained model
try:
    loan_approv_loaded_model = pickle.load(open("loan_prediction.sav", "rb"))
except FileNotFoundError:
    st.error("Model file 'loan_prediction.sav' not found. Please check the file path.")
    st.stop()

# Prediction Function
def prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    pred = loan_approv_loaded_model.predict(input_array)
    return "✅ Loan Approved" if pred[0] == 1 else "❌ Loan Not Approved"

# Streamlit UI
def main():
    st.title("🏦 Loan Prediction Model")
    st.markdown("Enter the required details to check **loan approval**.")

    col1, col2 = st.columns(2)

    with col1:
        Gender = st.number_input("Gender (0: Male, 1: Female)", min_value=0, max_value=1, value=0, step=1)
        Married = st.number_input("Married (0: No, 1: Yes)", min_value=0, max_value=1, value=1, step=1)
        Dependents = st.number_input("Dependents (0, 1, 2, or 3+)", min_value=0, max_value=3, value=1, step=1)
        Education = st.number_input("Education (0: Graduate, 1: Not Graduate)", min_value=0, max_value=1, value=1, step=1)
        Self_Employed = st.number_input("Self Employed (0: No, 1: Yes)", min_value=0, max_value=1, value=0, step=1)
        ApplicantIncome = st.number_input("Applicant Income", min_value=0, value=4583)

    with col2:
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, value=1500.0)
        LoanAmount = st.number_input("Loan Amount", min_value=0.0, value=128.0)
        Loan_Amount_Term = st.number_input("Loan Term (in days)", min_value=0.0, value=360.0)
        Credit_History = st.number_input("Credit History (0 or 1)", min_value=0.0, max_value=1.0, value=1.0, step=1.0)
        Property_Area = st.number_input("Property Area (0: Rural, 1: Semiurban, 2: Urban)", min_value=0, max_value=2, value=0, step=1)

    # Prediction Button
    if st.button("🔍 Predict"):
        input_data = [
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Property_Area
        ]
        
        result = prediction(input_data)
        st.success(f"### {result}")  # Display result with emphasis

if __name__ == '__main__':
    main()
