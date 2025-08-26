# Heart Disease Prediction Model

## Overview

This repository provides a complete workflow for predicting heart disease using machine learning. It includes data acquisition, preprocessing, exploratory data analysis, model training, evaluation, and a Streamlit web application for batch and single predictions.

---

## 📂 Project Structure

```
.
├── app.py                      # Streamlit web application
├── main.ipynb                  # Jupyter notebook with data analysis & model training
├── requirements.txt            # Python dependencies
├── heart_rf_model.pkl          # Trained Random Forest model
├── heart_scaler.pkl            # Scaler used for feature normalization
├── kaggle.json                 # Kaggle API credentials
├── heart-disease/
│   ├── heart_disease_uci.csv   # Main dataset
│   ├── heart_dataset_for_testing.csv
│   └── Heart_user_template.csv # Template for user uploads
└── ...
```

---

## 🚀 Technologies Used

- **Python 3.10+**
- **Jupyter Notebook** (`main.ipynb`)
- **Streamlit** (`app.py`)
- **Pandas** / **NumPy** (data manipulation)
- **Scikit-learn** (ML algorithms, preprocessing, metrics)
- **Matplotlib** / **Seaborn** (visualization)
- **Joblib** (model serialization)
- **Kaggle API** (dataset download)

---

## 🏷️ Key Features

- **Data Cleaning & Preprocessing:** Handles missing values, encodes categorical variables, feature scaling.
- **Exploratory Data Analysis:** Correlation heatmaps, feature distributions.
- **Model Training:** Logistic Regression, Random Forest Classifier.
- **Model Evaluation:** Accuracy, classification report, confusion matrix, feature importance.
- **Web App:** Batch and single prediction with CSV upload/download.

---

## ⚙️ How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   - Place your `kaggle.json` in the root directory.
   - Run the notebook or use the Kaggle API to download the dataset to `heart-disease/`.

3. **Train models:**
   - Run all cells in `main.ipynb` to preprocess data, train, and save models.

4. **Start the web app:**
   ```sh
   streamlit run app.py
   ```

---

## 📊 Algorithms & Statistical Techniques

### 1. **Data Preprocessing**
- **Missing Value Imputation:**  
  - Numeric columns: filled with column mean.
  - Categorical columns: filled with `'na'`.
- **One-Hot Encoding:**  
  - Converts categorical variables into binary columns.
- **Feature Scaling:**  
  - Standardization using `StandardScaler` (z-score normalization).

### 2. **Algorithms**
- **Logistic Regression**
  - Used for binary classification (`num > 0` as heart disease).
  - Evaluated using accuracy, precision, recall, F1-score.
- **Random Forest Classifier**
  - Ensemble of decision trees for improved accuracy and robustness.
  - Feature importance analysis to interpret model decisions.

### 3. **Model Evaluation**
- **Train/Test Split:**  
  - 80/20 split for model validation.
- **Metrics:**  
  - Accuracy, confusion matrix, classification report.
- **Visualization:**  
  - Feature distributions, correlation heatmaps, feature importance bar plots.

---

## 📝 Usage

- **Batch Prediction:**  
  Upload a CSV file matching the template to get predictions for multiple users.
- **Single Prediction:**  
  Enter patient details in the web form for instant prediction.

---

## 📄 References

- [Heart Disease UCI Dataset on Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 🏷️ Flags & Parameters

- **Model Hyperparameters:**  
  - `RandomForestClassifier(n_estimators=100, random_state=42)`
  - `LogisticRegression()` (default settings)
- **Feature Columns:**  
  - Includes age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, ECG, max heart rate, exercise-induced angina, oldpeak, slope, number of vessels, thalassemia, etc.

---

## 📬 Contact

For questions or contributions, please open an issue or pull request

HAPPY CODING!