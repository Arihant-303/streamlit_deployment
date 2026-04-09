import streamlit as st
import joblib

# Load the trained model and scaler
model = joblib.load('heart_failure_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Failure Prediction App")
st.image(
    "https://png.pngtree.com/png-clipart/20250227/original/pngtree-realistic-human-heart-png-image_20524883.png",
    caption="Realistic Human Heart",
    width=150  # Small size
)

# Create input fields for user to enter data

age = st.slider("Age", 10, 95, 60)
anaemia = st.selectbox("Anaemia", [0,1])
creatinine = st.number_input("Creatinine Phosphokinase (CPK)", min_value=0, max_value=10000, value=500)
diabetes = st.radio("Diabetes", [0,1], format_func=lambda x: "Yes" if x else "No")
ef = st.number_input("Ejection Fraction", min_value=0, max_value=100, value=30)
hbp = st.radio("High Blood Pressure", [0,1], format_func=lambda x: "Yes" if x else "No")
platelets = st.number_input("Platelets", min_value=0, max_value=1000000, value=250000)
serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium", min_value=0, max_value=200, value=135)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0,1])
smoking = st.selectbox("Smoking", [0,1])
time = st.number_input("Time (days)", min_value=0, max_value=300, value=100)

# Create a button to make prediction
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = [[age, anaemia, creatinine, diabetes, ef, hbp, platelets, serum_creatinine, serum_sodium,sex, smoking, time]]
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(input_data_scaled)
    # Display the prediction result
    if prediction[0] == 1:
        st.error("Patient is at risk of heart failure.")
    else:
        st.success("Patient is not at risk of heart failure.")
