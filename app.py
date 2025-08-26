import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
rf_model = joblib.load('heart_rf_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

# Define important features (original, before one-hot encoding)
important_features = [
    'cp',        # chest pain type (categorical)
    'oldpeak',   # numeric
    'chol',      # numeric
    'thalach',   # numeric
    'ca',        # numeric (number of major vessels)
    'age',       # numeric
    'trestbps',  # numeric
    'exang',     # exercise induced angina (categorical: yes/no)
    'thal',      # thalassemia (categorical)
]

# Load training columns for alignment
X_train = pd.read_csv('heart-disease/Heart_user_template.csv')
train_columns = X_train.columns.tolist()

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Prediction App")
st.markdown("""
This app predicts the likelihood of heart disease using a RandomForest model.\
You can either upload a batch of patient data or enter information manually for a single prediction.\
**All preprocessing is handled automatically.**
""")

tabs = st.tabs(["Batch Prediction (Upload CSV)", "Single Prediction (Manual Input)"])

# --- Batch Prediction Tab ---
with tabs[0]:
    st.header("Batch Prediction: Upload Patient Dataset")
    uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        # Preprocessing: get columns from training
        numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
        cat_cols = X_train.select_dtypes(include='object').columns.tolist()
        bool_cols = X_train.select_dtypes(include='bool').columns.tolist()
        # Drop extra columns
        numeric_cols = [col for col in numeric_cols if col in user_df.columns]
        cat_cols = [col for col in cat_cols if col in user_df.columns]
        bool_cols = [col for col in bool_cols if col in user_df.columns]
        # Fill missing
        user_df[numeric_cols] = user_df[numeric_cols].fillna(user_df[numeric_cols].mean())
        for col in cat_cols:
            user_df[col] = user_df[col].fillna('na')
        for col in bool_cols:
            user_df[col] = user_df[col].astype(int)
        # One-hot encoding
        user_df_encoded = pd.get_dummies(user_df, columns=cat_cols)
        # Align columns
        user_df_encoded = user_df_encoded.reindex(columns=train_columns, fill_value=0)
        # Scale
        user_df_scaled = scaler.transform(user_df_encoded)
        # Predict
        preds = rf_model.predict(user_df_scaled)
        user_df['Heart_Disease_Prediction'] = preds
        st.success(f"Predictions completed! Showing first 10 results:")
        st.dataframe(user_df.head(10))
        st.download_button("Download Results as CSV", user_df.to_csv(index=False), file_name="predictions.csv")

# --- Single Prediction Tab ---
with tabs[1]:
    st.header("Single Prediction: Manual Input")
    st.markdown("Fill in the most important features for prediction:")
    form = st.form("manual_input_form")
    input_data = {}
    # Categorical options from training data
    cp_options = X_train[[col for col in X_train.columns if col.startswith('cp_')]].columns.str.replace('cp_', '').tolist()
    thal_options = X_train[[col for col in X_train.columns if col.startswith('thal_')]].columns.str.replace('thal_', '').tolist()
    exang_options = ['Yes', 'No']
    # Numeric ranges
    age_min, age_max, age_def = 18, 100, 50
    chol_min, chol_max, chol_def = 100, 600, 200
    oldpeak_min, oldpeak_max, oldpeak_def = 0, 10, 1
    thalach_min, thalach_max, thalach_def = 60, 220, 150
    ca_min, ca_max, ca_def = 0, 4, 0
    trestbps_min, trestbps_max, trestbps_def = 80, 200, 120

    # Collect user input
    input_data['cp'] = form.selectbox('Chest Pain Type (cp)', cp_options)
    input_data['oldpeak'] = form.number_input('Oldpeak', min_value=oldpeak_min, max_value=oldpeak_max, value=oldpeak_def, step=1)
    input_data['chol'] = form.number_input('Cholesterol (chol)', min_value=chol_min, max_value=chol_max, value=chol_def, step=1)
    input_data['thalach'] = form.number_input('Max Heart Rate (thalach)', min_value=thalach_min, max_value=thalach_max, value=thalach_def, step=1)
    input_data['ca'] = form.number_input('Number of Major Vessels (ca)', min_value=ca_min, max_value=ca_max, value=ca_def, step=1)
    input_data['age'] = form.number_input('Age', min_value=age_min, max_value=age_max, value=age_def, step=1)
    input_data['trestbps'] = form.number_input('Resting Blood Pressure (trestbps)', min_value=trestbps_min, max_value=trestbps_max, value=trestbps_def, step=1)
    input_data['exang'] = form.selectbox('Exercise Induced Angina (exang)', exang_options)
    input_data['thal'] = form.selectbox('Thalassemia (thal)', thal_options)

    submitted = form.form_submit_button("Predict")
    if submitted:
        # Build a single-row DataFrame with all columns
        single_df = pd.DataFrame([0]*len(train_columns), index=train_columns).T
        # Set numeric features
        for feat in ['oldpeak', 'chol', 'thalach', 'ca', 'age', 'trestbps']:
            if feat in single_df.columns:
                single_df[feat] = input_data[feat]
        # One-hot encode categorical features
        cp_col = f"cp_{input_data['cp']}"
        if cp_col in single_df.columns:
            single_df[cp_col] = 1
        thal_col = f"thal_{input_data['thal']}"
        if thal_col in single_df.columns:
            single_df[thal_col] = 1
        exang_col = f"exang_{'True' if input_data['exang']=='Yes' else 'False'}"
        if exang_col in single_df.columns:
            single_df[exang_col] = 1
        # Scale
        single_scaled = scaler.transform(single_df)
        pred = rf_model.predict(single_scaled)[0]
        if pred == 0:
            result_html = "<span style='color:green;font-weight:bold'>No Heart Disease</span>"
        else:
            result_html = "<span style='color:red;font-weight:bold'>Heart Disease Detected</span>"
        st.markdown(f"### Prediction: {result_html}", unsafe_allow_html=True)
