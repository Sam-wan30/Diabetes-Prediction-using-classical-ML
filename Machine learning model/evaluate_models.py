"""
Demonstration: Evaluate Trained Models
======================================
This script loads trained models and evaluates them using the comprehensive
evaluation functions from model_evaluation.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from model_evaluation import evaluate_classification_model, compare_models

print("=" * 80)
print("EVALUATING TRAINED MODELS")
print("=" * 80)
print()

# ============================================================================
# Load and Prepare Data
# ============================================================================

print("Loading data...")
try:
    df = pd.read_csv('data/diabetes_cleaned.csv')
    print("✓ Loaded cleaned dataset")
except FileNotFoundError:
    print("⚠ Cleaned dataset not found. Loading original...")
    df = pd.read_csv('data/diabetes.csv')
    # Quick preprocessing
    invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in invalid_zero_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

# Prepare features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape[0]} samples")
print(f"Test set: {X_test_scaled.shape[0]} samples")
print()

# ============================================================================
# Train Models
# ============================================================================

print("Training models...")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42, 
                             class_weight='balanced', solver='lbfgs')
lr_model.fit(X_train_scaled, y_train)
print("✓ Logistic Regression trained")

# LDA
lda_model = LinearDiscriminantAnalysis(solver='svd')
lda_model.fit(X_train_scaled, y_train)
print("✓ LDA trained")
print()

# ============================================================================
# Make Predictions
# ============================================================================

print("Making predictions...")
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lda_pred = lda_model.predict(X_test_scaled)
lda_pred_proba = lda_model.predict_proba(X_test_scaled)[:, 1]
print("✓ Predictions made")
print()

# ============================================================================
# Evaluate Individual Models
# ============================================================================

print("\n" + "=" * 80)
print("INDIVIDUAL MODEL EVALUATIONS")
print("=" * 80)
print()

# Evaluate Logistic Regression
lr_metrics = evaluate_classification_model(
    y_test, lr_pred, lr_pred_proba,
    model_name="Logistic Regression",
    class_names=['No Diabetes', 'Diabetes'],
    save_plots=True
)

print("\n" + "=" * 80 + "\n")

# Evaluate LDA
lda_metrics = evaluate_classification_model(
    y_test, lda_pred, lda_pred_proba,
    model_name="LDA",
    class_names=['No Diabetes', 'Diabetes'],
    save_plots=True
)

# ============================================================================
# Compare Models
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print()

y_pred_dict = {
    'Logistic Regression': lr_pred,
    'LDA': lda_pred
}

y_pred_proba_dict = {
    'Logistic Regression': lr_pred_proba,
    'LDA': lda_pred_proba
}

all_metrics, comparison_df = compare_models(
    y_test, y_pred_dict, y_pred_proba_dict,
    model_names=['Logistic Regression', 'LDA'],
    class_names=['No Diabetes', 'Diabetes'],
    save_plots=True
)

print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print()
print("Generated files:")
print("  • logistic_regression_confusion_matrix.png")
print("  • logistic_regression_roc_curve.png")
print("  • lda_confusion_matrix.png")
print("  • lda_roc_curve.png")
print("  • model_comparison.png")
print("  • roc_curves_comparison.png")
print()

