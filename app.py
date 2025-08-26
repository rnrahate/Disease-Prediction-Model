
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('heart_rf_model.pkl')
    scaler = joblib.load('heart_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Define input fields (based on original dataset before one-hot encoding)
sex_options = ['Male', 'Female']
dataset_options = ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach']
cp_options = ['asymptomatic', 'atypical angina', 'non-anginal', 'typical angina']
fbs_options = ['TRUE', 'FALSE']
restecg_options = ['lv hypertrophy', 'normal', 'st-t abnormality']
exang_options = ['TRUE', 'FALSE']
slope_options = ['downsloping', 'flat', 'upsloping']
thal_options = ['fixed defect', 'normal', 'reversable defect']

st.title('Heart Disease Prediction App')
st.write('Enter patient parameters below:')

with st.form('input_form'):
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', sex_options)
    dataset = st.selectbox('Dataset', dataset_options)
    cp = st.selectbox('Chest Pain Type', cp_options)
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=120)
    chol = st.number_input('Serum Cholesterol (chol)', min_value=50, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', fbs_options)
    restecg = st.selectbox('Resting ECG', restecg_options)
    thalch = st.number_input('Maximum Heart Rate Achieved (thalch)', min_value=50, max_value=250, value=150)
    exang = st.selectbox('Exercise Induced Angina (exang)', exang_options)
    oldpeak = st.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox('Slope of Peak Exercise ST Segment', slope_options)
    ca = st.number_input('Number of Major Vessels (ca)', min_value=0, max_value=4, value=0)
    thal = st.selectbox('Thalassemia', thal_options)
    submitted = st.form_submit_button('Predict')

if submitted:
    # Build a single-row DataFrame with the original columns
    input_dict = {
        'id': [0],
        'age': [age],
        'trestbps': [trestbps],
        'chol': [chol],
        'thalch': [thalch],
        'oldpeak': [oldpeak],
        'ca': [ca],
        'sex': [sex],
        'dataset': [dataset],
        'cp': [cp],
        'fbs': [fbs],
        'restecg': [restecg],
        'exang': [exang],
        'slope': [slope],
        'thal': [thal],
    }
    df_input = pd.DataFrame(input_dict)
    # One-hot encode to match model features
    # Drop 'num' if present, and use the same columns as in training
    cat_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    df_input = pd.get_dummies(df_input, columns=cat_cols)
    # Load template columns to ensure all dummies are present
    template = pd.read_csv('heart-disease/Heart_user_template.csv')
    model_cols = template.columns
    # Add missing columns as 0
    for col in model_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    # Ensure order
    df_input = df_input[model_cols]
    # Scale
    X_scaled = scaler.transform(df_input)
    pred = model.predict(X_scaled)[0]
    st.subheader('Prediction Result:')
    if pred == 1:
        st.error('The model predicts: Heart Disease')
    else:
        st.success('The model predicts: No Heart Disease')
