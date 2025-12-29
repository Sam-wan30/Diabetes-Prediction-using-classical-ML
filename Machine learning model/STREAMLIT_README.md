# Diabetes Risk Prediction - Streamlit App

## Overview

This Streamlit web application provides an interactive interface for predicting diabetes risk using a trained Logistic Regression model. The app allows users to input patient health metrics and receive real-time diabetes risk predictions with personalized recommendations.

## Features

- **Interactive Input Form**: Easy-to-use form for entering patient health metrics
- **Real-time Predictions**: Instant diabetes risk assessment with probability scores
- **Risk Level Classification**: Categorizes risk as Low, Moderate, or High
- **Personalized Recommendations**: Provides actionable health advice based on input values
- **Feature Analysis**: Shows which features contribute most to the prediction
- **Responsive Design**: Clean, professional interface suitable for healthcare settings

## Installation

1. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Data Files Exist**:
   - The app requires either `data/diabetes_cleaned.csv` or `data/diabetes.csv`
   - If neither exists, the app will show an error

## Running the App

1. **Start Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the App**:
   - The app will automatically open in your default web browser
   - Default URL: `http://localhost:8501`

## Usage

1. **Enter Patient Information**:
   - Fill in all 8 health metrics in the left column:
     - Number of Pregnancies
     - Glucose (mg/dL)
     - Blood Pressure (mmHg)
     - Skin Thickness (mm)
     - Insulin (μU/mL)
     - BMI (kg/m²)
     - Diabetes Pedigree Function
     - Age (years)

2. **Get Prediction**:
   - Click the "🔍 Predict Diabetes Risk" button
   - View the risk level and probability score
   - Read personalized recommendations
   - Check feature contribution analysis

3. **Interpret Results**:
   - **Low Risk** (< 40%): Continue healthy lifestyle
   - **Moderate Risk** (40-70%): Consider screening and lifestyle changes
   - **High Risk** (≥ 70%): Consult healthcare provider immediately

## Input Guidelines

### Normal Ranges (for reference):
- **Glucose**: 70-100 mg/dL (fasting)
- **Blood Pressure**: 60-80 mmHg (diastolic)
- **BMI**: 18.5-24.9 kg/m²
- **Insulin**: 2-25 μU/mL (fasting)

### Important Notes:
- Zero values in Glucose, Blood Pressure, Skin Thickness, Insulin, or BMI will be automatically replaced with median values (as these are biologically impossible)
- All inputs are validated and have appropriate min/max ranges
- The model uses standardized features, so coefficients are directly comparable

## Model Information

- **Algorithm**: Logistic Regression
- **Training Data**: Pima Indians Diabetes Dataset (768 samples)
- **Performance Metrics**:
  - Accuracy: ~73%
  - AUC-ROC: ~81%
  - Precision: ~60%
  - Recall: ~70%

- **Top Features** (by importance):
  1. Glucose (coefficient: 1.18)
  2. BMI (coefficient: 0.71)
  3. Pregnancies (coefficient: 0.37)
  4. DiabetesPedigreeFunction (coefficient: 0.29)

## Technical Details

### Preprocessing:
- Invalid zero values are replaced with median values
- Features are standardized using StandardScaler
- Model uses balanced class weights to handle class imbalance

### Model Training:
- Trained on 80% of data (614 samples)
- Tested on 20% of data (154 samples)
- Uses stratified splitting to maintain class distribution
- Random state: 42 (for reproducibility)

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**:
   - Solution: Install all requirements: `pip install -r requirements.txt`

2. **FileNotFoundError for data**:
   - Solution: Ensure `data/diabetes.csv` or `data/diabetes_cleaned.csv` exists
   - Run `preprocessing.py` to generate cleaned dataset

3. **Port Already in Use**:
   - Solution: Use a different port: `streamlit run streamlit_app.py --server.port 8502`

4. **Slow Loading**:
   - First run may be slow as model is trained
   - Subsequent runs use cached model (faster)

## Customization

### Modify Risk Thresholds:
Edit the `get_risk_level()` function in `streamlit_app.py`:
```python
def get_risk_level(probability):
    if probability >= 0.7:  # Change threshold here
        return "High Risk", "high-risk"
    elif probability >= 0.4:  # Change threshold here
        return "Moderate Risk", "moderate-risk"
    else:
        return "Low Risk", "low-risk"
```

### Add More Recommendations:
Edit the `get_recommendations()` function to add custom advice based on specific conditions.

## Disclaimer

⚠️ **IMPORTANT**: This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Project Structure

```
IITG project/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   ├── diabetes.csv          # Original dataset
│   └── diabetes_cleaned.csv  # Preprocessed dataset
├── model_training.py         # Model training script
└── preprocessing.py          # Data preprocessing script
```

## Author

IITG Project - Diabetes Prediction System

## License

Educational use only

