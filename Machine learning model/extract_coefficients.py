"""
Extract Logistic Regression Coefficients
========================================
This script extracts Logistic Regression coefficients, creates a DataFrame
with feature names and coefficients, and sorts them by absolute value.

UNDERSTANDING LOGISTIC REGRESSION COEFFICIENTS:
-----------------------------------------------
- Coefficients represent the log-odds change for a one-unit increase in the feature
- Positive coefficients: Increase the probability of the positive class (diabetes)
- Negative coefficients: Decrease the probability of the positive class
- Larger absolute values: Stronger influence on the prediction
- Since features are standardized, coefficients are directly comparable
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print("=" * 80)
print("EXTRACTING LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

print("STEP 1: Loading and Preparing Data")
print("-" * 80)

# Load the cleaned dataset (or original if cleaned doesn't exist)
try:
    df = pd.read_csv('data/diabetes_cleaned.csv')
    print("✓ Loaded cleaned dataset")
except FileNotFoundError:
    print("⚠ Cleaned dataset not found. Loading original and applying preprocessing...")
    df = pd.read_csv('data/diabetes.csv')
    # Quick preprocessing: replace invalid zeros with NaN and impute with median
    invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in invalid_zero_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    print("✓ Applied preprocessing")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Features: {list(X.columns)}")
print(f"Number of features: {len(X.columns)}")
print()

# ============================================================================
# STEP 2: Split and Scale Data
# ============================================================================

print("STEP 2: Splitting and Scaling Data")
print("-" * 80)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (required for Logistic Regression)
# StandardScaler ensures all features have mean=0 and std=1
# This makes coefficients directly comparable
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Data split and scaled")
print(f"Training samples: {X_train_scaled.shape[0]}")
print(f"Test samples: {X_test_scaled.shape[0]}")
print()

# ============================================================================
# STEP 3: Train Logistic Regression Model
# ============================================================================

print("STEP 3: Training Logistic Regression Model")
print("-" * 80)

# Initialize and train Logistic Regression
# class_weight='balanced': Handles class imbalance
# solver='lbfgs': Good optimizer for scaled data
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    solver='lbfgs'
)

lr_model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully")
print()

# ============================================================================
# STEP 4: Extract Coefficients
# ============================================================================

print("STEP 4: Extracting Coefficients")
print("-" * 80)

# Extract coefficients from the trained model
# lr_model.coef_ returns a 2D array: [[coef1, coef2, ..., coefN]]
# For binary classification, we use [0] to get the first (and only) row
coefficients = lr_model.coef_[0]

# Extract intercept
# lr_model.intercept_ returns an array: [intercept]
intercept = lr_model.intercept_[0]

print(f"Number of coefficients: {len(coefficients)}")
print(f"Intercept: {intercept:.4f}")
print()

# ============================================================================
# STEP 5: Create DataFrame with Feature Names and Coefficients
# ============================================================================

print("STEP 5: Creating DataFrame")
print("-" * 80)

# Create DataFrame with feature names and coefficients
# This makes it easy to analyze and visualize feature importance
coefficients_df = pd.DataFrame({
    'Feature': X.columns,           # Feature names
    'Coefficient': coefficients,     # Coefficient values
    'Abs_Coefficient': np.abs(coefficients)  # Absolute values for sorting
})

print("✓ DataFrame created")
print(f"Shape: {coefficients_df.shape}")
print()

# Display the DataFrame before sorting
print("Coefficients DataFrame (before sorting):")
print("-" * 50)
print(coefficients_df.to_string(index=False))
print()

# ============================================================================
# STEP 6: Sort by Absolute Value
# ============================================================================

print("STEP 6: Sorting by Absolute Value")
print("-" * 80)

# Sort DataFrame by absolute coefficient value (descending)
# This shows the most important features first
# ascending=False means largest absolute values first
coefficients_df_sorted = coefficients_df.sort_values('Abs_Coefficient', ascending=False)

print("✓ DataFrame sorted by absolute coefficient value")
print()

# Display sorted DataFrame
print("Coefficients DataFrame (sorted by absolute value):")
print("=" * 80)
print(coefficients_df_sorted.to_string(index=False))
print()

# ============================================================================
# STEP 7: Detailed Analysis
# ============================================================================

print("STEP 7: Detailed Analysis")
print("-" * 80)

print("\nFeature Importance Ranking:")
print("-" * 80)
print(f"{'Rank':<6} {'Feature':<25} {'Coefficient':<15} {'Abs Value':<15} {'Effect':<20}")
print("-" * 80)

for idx, (_, row) in enumerate(coefficients_df_sorted.iterrows(), 1):
    coefficient = row['Coefficient']
    abs_value = row['Abs_Coefficient']
    
    # Determine the effect direction
    if coefficient > 0:
        effect = "↑ Increases risk"
        symbol = "+"
    else:
        effect = "↓ Decreases risk"
        symbol = "-"
    
    print(f"{idx:<6} {row['Feature']:<25} {coefficient:>14.4f} {abs_value:>14.4f} {effect:<20}")

print()

# ============================================================================
# STEP 8: Interpretation
# ============================================================================

print("STEP 8: Interpretation")
print("-" * 80)

print("\nTop 3 Most Important Features (by absolute coefficient):")
print("-" * 50)
for idx, (_, row) in enumerate(coefficients_df_sorted.head(3).iterrows(), 1):
    coef = row['Coefficient']
    direction = "increases" if coef > 0 else "decreases"
    print(f"{idx}. {row['Feature']}")
    print(f"   Coefficient: {coef:.4f}")
    print(f"   Effect: {direction} diabetes risk")
    print(f"   Strength: {row['Abs_Coefficient']:.4f}")
    print()

print("Bottom 3 Least Important Features (by absolute coefficient):")
print("-" * 50)
for idx, (_, row) in enumerate(coefficients_df_sorted.tail(3).iterrows(), 1):
    coef = row['Coefficient']
    direction = "increases" if coef > 0 else "decreases"
    print(f"{idx}. {row['Feature']}")
    print(f"   Coefficient: {coef:.4f}")
    print(f"   Effect: {direction} diabetes risk")
    print(f"   Strength: {row['Abs_Coefficient']:.4f}")
    print()

# ============================================================================
# STEP 9: Save Results
# ============================================================================

print("STEP 9: Saving Results")
print("-" * 80)

# Save the sorted DataFrame to CSV
output_file = 'logistic_regression_coefficients.csv'
coefficients_df_sorted.to_csv(output_file, index=False)
print(f"✓ Saved coefficients to: {output_file}")
print()

# Also save with additional information
detailed_df = coefficients_df_sorted.copy()
detailed_df['Effect'] = detailed_df['Coefficient'].apply(
    lambda x: 'Increases risk' if x > 0 else 'Decreases risk'
)
detailed_df['Rank'] = range(1, len(detailed_df) + 1)
detailed_df = detailed_df[['Rank', 'Feature', 'Coefficient', 'Abs_Coefficient', 'Effect']]

detailed_output_file = 'logistic_regression_coefficients_detailed.csv'
detailed_df.to_csv(detailed_output_file, index=False)
print(f"✓ Saved detailed coefficients to: {detailed_output_file}")
print()

# ============================================================================
# STEP 10: Summary Statistics
# ============================================================================

print("STEP 10: Summary Statistics")
print("-" * 80)

print("\nCoefficient Statistics:")
print("-" * 50)
print(f"Total features: {len(coefficients_df_sorted)}")
print(f"Positive coefficients: {(coefficients_df_sorted['Coefficient'] > 0).sum()}")
print(f"Negative coefficients: {(coefficients_df_sorted['Coefficient'] < 0).sum()}")
print(f"Largest coefficient: {coefficients_df_sorted['Coefficient'].max():.4f}")
print(f"Smallest coefficient: {coefficients_df_sorted['Coefficient'].min():.4f}")
print(f"Largest absolute value: {coefficients_df_sorted['Abs_Coefficient'].max():.4f}")
print(f"Smallest absolute value: {coefficients_df_sorted['Abs_Coefficient'].min():.4f}")
print(f"Mean absolute value: {coefficients_df_sorted['Abs_Coefficient'].mean():.4f}")
print(f"Intercept: {intercept:.4f}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("EXTRACTION COMPLETE!")
print("=" * 80)
print()

print("Summary:")
print(f"  • Extracted {len(coefficients_df_sorted)} coefficients")
print(f"  • Sorted by absolute value (descending)")
print(f"  • Saved to CSV files")
print()

print("Files Generated:")
print(f"  • {output_file}")
print(f"  • {detailed_output_file}")
print()

print("Key Insights:")
print(f"  • Most important feature: {coefficients_df_sorted.iloc[0]['Feature']} "
      f"(coefficient: {coefficients_df_sorted.iloc[0]['Coefficient']:.4f})")
print(f"  • Least important feature: {coefficients_df_sorted.iloc[-1]['Feature']} "
      f"(coefficient: {coefficients_df_sorted.iloc[-1]['Coefficient']:.4f})")
print()

# Display the final sorted DataFrame one more time
print("=" * 80)
print("FINAL SORTED COEFFICIENTS DATAFRAME")
print("=" * 80)
print(coefficients_df_sorted[['Feature', 'Coefficient', 'Abs_Coefficient']].to_string(index=False))
print()

