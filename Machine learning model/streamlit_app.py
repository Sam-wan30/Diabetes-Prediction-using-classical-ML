"""
Diabetes Risk Prediction App
============================
Streamlit web application for predicting diabetes risk using Logistic Regression.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical aesthetic
st.markdown("""
    <style>
    /* Import Google Fonts for medical look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main styling */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    /* Header styling */
    .medical-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(30, 60, 114, 0.2);
    }
    
    .medical-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .medical-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Medical card styling */
    .medical-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .medical-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Risk level boxes */
    .risk-box {
        padding: 2.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        border: 2px solid;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        border-color: #d32f2f;
        color: white;
    }
    
    .moderate-risk {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        border-color: #f57c00;
        color: white;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        border-color: #388e3c;
        color: white;
    }
    
    .risk-box h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .risk-box h3 {
        font-size: 1.5rem;
        font-weight: 400;
        margin: 0.5rem 0;
        opacity: 0.95;
    }
    
    /* Input section styling */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #2a5298;
    }
    
    .section-title {
        color: #1e3c72;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e3f2fd;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    /* Recommendation cards */
    .recommendation-item {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        border-left: 4px solid #2a5298;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Feature analysis table */
    .feature-table {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(30, 60, 114, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(30, 60, 114, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Footer */
    .medical-footer {
        background: #1e3c72;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 3rem;
        font-size: 0.9rem;
        line-height: 1.8;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #2a5298;
        box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv('data/diabetes_cleaned.csv')
    except FileNotFoundError:
        df = pd.read_csv('data/diabetes.csv')
        # Preprocess: replace invalid zeros with NaN and impute with median
        invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in invalid_zero_cols:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    return df

@st.cache_resource
def train_model():
    """Train and return the model and scaler."""
    # Load data
    df = load_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

def get_risk_level(probability):
    """Determine risk level based on probability."""
    if probability >= 0.7:
        return "High Risk", "high-risk"
    elif probability >= 0.4:
        return "Moderate Risk", "moderate-risk"
    else:
        return "Low Risk", "low-risk"

def get_recommendations(probability, user_input):
    """Generate personalized recommendations based on risk and input values."""
    recommendations = []
    
    if probability >= 0.7:
        recommendations.append("⚠️ **Immediate Action Required**: Consult a healthcare provider for comprehensive diabetes screening and management.")
    elif probability >= 0.4:
        recommendations.append("📋 **Preventive Care**: Schedule a diabetes screening test and consider lifestyle modifications.")
    else:
        recommendations.append("✅ **Maintain Healthy Lifestyle**: Continue regular check-ups and maintain healthy habits.")
    
    # Glucose-specific recommendations
    if user_input['Glucose'] >= 126:
        recommendations.append("🔴 **Critical**: Your glucose level indicates diabetes. Please consult a doctor immediately.")
    elif user_input['Glucose'] >= 100:
        recommendations.append("🟡 **Warning**: Your glucose level suggests prediabetes. Lifestyle changes are recommended.")
    
    # BMI-specific recommendations
    if user_input['BMI'] >= 30:
        recommendations.append("⚖️ **Weight Management**: Your BMI indicates obesity. Weight loss of 5-10% can significantly reduce diabetes risk.")
    elif user_input['BMI'] >= 25:
        recommendations.append("⚖️ **Weight Management**: Your BMI indicates overweight. Consider weight loss through diet and exercise.")
    
    # Age-specific recommendations
    if user_input['Age'] >= 45:
        recommendations.append("👴 **Age Factor**: Regular diabetes screening is recommended for your age group.")
    
    # Pregnancy-specific recommendations
    if user_input['Pregnancies'] > 0:
        recommendations.append("🤰 **Pregnancy History**: If you had gestational diabetes, regular follow-up screening is important.")
    
    return recommendations

def main():
    # Medical Header
    st.markdown("""
    <div class="medical-header">
        <h1>🏥 Diabetes Risk Assessment System</h1>
        <p>Clinical Decision Support Tool | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Medical Information
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h3 style="color: white; margin-top: 0;">📋 Clinical Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                    border-left: 4px solid #2a5298; margin-bottom: 1.5rem;">
            <h4 style="color: #1e3c72; margin-top: 0;">Model Performance</h4>
            <p style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">
                <strong>Accuracy:</strong> 73.4%<br>
                <strong>AUC-ROC:</strong> 81.3%<br>
                <strong>Precision:</strong> 60.3%<br>
                <strong>Recall:</strong> 70.4%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                    border-left: 4px solid #2a5298;">
            <h4 style="color: #1e3c72; margin-top: 0;">🔬 Feature Importance</h4>
            <p style="font-size: 0.9rem; color: #666;">
                <strong>1. Glucose</strong> (1.18)<br>
                <strong>2. BMI</strong> (0.71)<br>
                <strong>3. Pregnancies</strong> (0.37)<br>
                <strong>4. Diabetes Pedigree</strong> (0.29)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #ffc107;">
            <p style="font-size: 0.85rem; color: #856404; margin: 0;">
                ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational 
                purposes only and should not replace professional medical advice.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    try:
        model, scaler, feature_names = train_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="input-section">
            <h2 class="section-title">📋 Patient Health Metrics</h2>
            <p style="color: #666; margin-bottom: 1.5rem;">Enter clinical measurements below</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input fields with medical styling
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of times the patient has been pregnant"
        )
        
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=1.0,
            help="Plasma glucose concentration (normal: 70-100 mg/dL)"
        )
        
        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
            help="Diastolic blood pressure (normal: 60-80 mmHg)"
        )
        
        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            help="Triceps skin fold thickness"
        )
        
        insulin = st.number_input(
            "Insulin (μU/mL)",
            min_value=0.0,
            max_value=1000.0,
            value=80.0,
            step=1.0,
            help="2-Hour serum insulin (normal: 2-25 μU/mL fasting)"
        )
        
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index (normal: 18.5-24.9)"
        )
        
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="Function representing genetic predisposition to diabetes"
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=30,
            help="Age in years"
        )
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 12px; 
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); border-left: 4px solid #2a5298;">
            <h2 class="section-title">🔬 Clinical Assessment</h2>
            <p style="color: #666; margin-bottom: 1.5rem;">Risk prediction and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create input dictionary
        user_input = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age
        }
        
        # Prepare input for prediction
        input_df = pd.DataFrame([user_input])
        input_df = input_df[feature_names]  # Ensure correct column order
        
        # Handle invalid zeros (same preprocessing as training)
        invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in invalid_zero_cols:
            if input_df[col].iloc[0] == 0:
                # Replace with median from training data
                df = load_data()
                input_df[col] = df[col].median()
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 Generate Risk Assessment", type="primary", use_container_width=True):
            # Get prediction probability
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = model.predict(input_scaled)[0]
            
            # Determine risk level
            risk_level, risk_class = get_risk_level(probability)
            
            # Display results with medical styling
            st.markdown(f"""
            <div class="risk-box {risk_class}">
                <h2>{risk_level}</h2>
                <h3>Diabetes Risk Probability: {probability*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar with medical styling
            st.markdown(f"<p style='text-align: center; color: #666; margin-top: 1rem;'><strong>Risk Score:</strong> {probability:.3f}</p>", unsafe_allow_html=True)
            st.progress(probability)
            
            # Detailed metrics in medical card style
            st.markdown("""
            <div class="medical-card">
                <h3 style="color: #1e3c72; margin-top: 0; border-bottom: 2px solid #e3f2fd; padding-bottom: 0.5rem;">
                    📊 Clinical Metrics
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Prediction</div>
                    <div class="metric-value">{"Diabetes" if prediction == 1 else "No Diabetes"}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{max(probability, 1-probability)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value">{risk_level.split()[0]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations in medical card style
            st.markdown("""
            <div class="medical-card">
                <h3 style="color: #1e3c72; margin-top: 0; border-bottom: 2px solid #e3f2fd; padding-bottom: 0.5rem;">
                    💊 Clinical Recommendations
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            recommendations = get_recommendations(probability, user_input)
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-item">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
            
            # Feature contribution analysis
            st.markdown("""
            <div class="medical-card">
                <h3 style="color: #1e3c72; margin-top: 0; border-bottom: 2px solid #e3f2fd; padding-bottom: 0.5rem;">
                    🔬 Contributing Factors Analysis
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            coefficients = model.coef_[0]
            feature_contributions = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Value': input_df.iloc[0].values,
                'Contribution': coefficients * scaler.transform(input_df)[0]
            }).sort_values('Contribution', key=abs, ascending=False)
            
            st.markdown('<div class="feature-table">', unsafe_allow_html=True)
            st.dataframe(
                feature_contributions[['Feature', 'Contribution']].head(4),
                use_container_width=True,
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Top 4 clinical factors contributing to this assessment")
    
    # Medical Footer
    st.markdown("""
    <div class="medical-footer">
        <p style="margin: 0.5rem 0;"><strong>⚠️ Medical Disclaimer</strong></p>
        <p style="margin: 0.5rem 0; opacity: 0.9;">
            This clinical decision support tool is for educational and research purposes only. 
            It should not be used as a substitute for professional medical advice, diagnosis, 
            or treatment. Always consult with a qualified healthcare provider for any 
            health-related questions or concerns.
        </p>
        <p style="margin: 1rem 0 0 0; font-size: 0.85rem; opacity: 0.8;">
            Model trained on Pima Indians Diabetes Dataset | IITG Project | 
            Powered by Logistic Regression ML Model
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

