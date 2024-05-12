import streamlit as st
import pandas as pd
import pickle as pk

# Function to load model and scaler
def load_model_and_scaler(model_path, scaler_path):
    with open(model_path, 'rb') as model_file:
        model = pk.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pk.load(scaler_file)
    return model, scaler

# Load model and scaler
model, scaler = load_model_and_scaler('C:/Users/Noman Javed/loan_prediction/model.pkl', 'C:/Users/Noman Javed/loan_prediction/scaler.pkl')

# Streamlit app
st.header('Loan Prediction App')

no_of_dependents = st.slider('Choose Number of Dependents', 0, 5)
education = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
self_employed = st.selectbox('Self Employed?', ['Yes', 'No'])
annual_income = st.slider('Choose Annual Income', 0, 10000000)
loan_amount = st.slider('Choose Loan Amount', 0, 10000000)
loan_duration = st.slider('Choose Loan Duration', 0, 20)
cibil_score = st.slider('Choose Cibil Score', 0, 1000)
assets = st.slider('Choose Assets', 0, 10000000)

# Preprocess user inputs
if education == 'Graduated':
    education_encoded = 0
else:
    education_encoded = 1

if self_employed == 'No':
    self_employed_encoded = 0
else:
    self_employed_encoded = 1

# Predict loan approval
if st.button("Predict"):
    user_data = pd.DataFrame([[no_of_dependents, education_encoded, self_employed_encoded,
                               annual_income, loan_amount, loan_duration, cibil_score, assets]],
                             columns=['no_of_dependents', 'education', 'self_employed',
                                      'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    if prediction[0] == 1:
        st.markdown('Loan Is Approved')
    else:
        st.markdown('Loan Is Rejected')
