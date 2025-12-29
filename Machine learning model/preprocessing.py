"""
Data Preprocessing: Handling Invalid Zero Values in Medical Data
================================================================
This script identifies and handles invalid zero values in the Pima Indians
Diabetes dataset by replacing them with NaN and imputing using median values.

WHY THIS IS CRITICAL FOR MEDICAL DATA:
--------------------------------------
In medical datasets, zero values often represent missing data rather than
actual zero measurements. This is because:

1. BIOLOGICAL IMPOSSIBILITY: Many medical measurements cannot be zero
   - Glucose: Cannot be 0 mg/dL (would indicate death)
   - Blood Pressure: Cannot be 0 mmHg (would indicate death)
   - BMI: Cannot be 0 (would indicate no body mass)
   - Skin Thickness: Cannot be 0 mm (everyone has some skin thickness)

2. DATA COLLECTION PRACTICES: Historical datasets often used 0 as a
   placeholder for missing values due to:
   - Limited storage capabilities
   - Simpler data entry systems
   - Lack of standardized missing value indicators

3. IMPACT ON MODELS: Treating missing values as zeros can:
   - Introduce severe bias in model training
   - Lead to incorrect predictions
   - Skew statistical distributions
   - Reduce model accuracy and reliability

4. MEDICAL ETHICS: Incorrect data handling can lead to:
   - Misdiagnosis risk
   - Inappropriate treatment recommendations
   - Unreliable research findings
   - Potential harm to patients

WHY MEDIAN IMPUTATION:
---------------------
- Median is robust to outliers (unlike mean)
- Preserves the distribution shape better
- Less sensitive to extreme values
- More appropriate for skewed medical data distributions
"""

import pandas as pd
import numpy as np

# Load the dataset
print("=" * 80)
print("DATA PREPROCESSING: Handling Invalid Zero Values")
print("=" * 80)
print()

df = pd.read_csv('data/diabetes.csv')
print(f"Original dataset shape: {df.shape}")
print()

# ============================================================================
# STEP 1: Identify Invalid Zero Values
# ============================================================================

print("STEP 1: Identifying Invalid Zero Values")
print("-" * 80)

# Define columns where zero values are medically invalid
# These represent physiological measurements that cannot be zero in living patients
invalid_zero_columns = {
    'Glucose': {
        'reason': 'Glucose cannot be 0 mg/dL in a living person. Normal fasting glucose is 70-100 mg/dL.',
        'normal_range': '70-100 mg/dL (fasting)',
        'critical': True  # Most important feature
    },
    'BloodPressure': {
        'reason': 'Blood pressure cannot be 0 mmHg. Normal range is 90-120/60-80 mmHg.',
        'normal_range': '90-120/60-80 mmHg',
        'critical': True
    },
    'SkinThickness': {
        'reason': 'Skin thickness cannot be 0 mm. Everyone has measurable skin thickness.',
        'normal_range': 'Varies by body location (typically 0.5-4 mm)',
        'critical': False
    },
    'Insulin': {
        'reason': 'While insulin can be low, 0 is extremely rare and likely indicates missing data.',
        'normal_range': '2-25 μU/mL (fasting)',
        'critical': False
    },
    'BMI': {
        'reason': 'BMI cannot be 0. Even severely underweight individuals have BMI > 10.',
        'normal_range': '18.5-24.9 (normal), <18.5 (underweight)',
        'critical': True
    }
}

# Count zero values before processing
print("\nZero values found in each column:")
print("-" * 50)
zero_counts_before = {}
for col in invalid_zero_columns.keys():
    zero_count = (df[col] == 0).sum()
    zero_counts_before[col] = zero_count
    status = "⚠️  CRITICAL" if invalid_zero_columns[col]['critical'] else "⚠️  WARNING"
    print(f"{col:20s}: {zero_count:4d} zeros - {status}")
    print(f"  Reason: {invalid_zero_columns[col]['reason']}")
    print(f"  Normal range: {invalid_zero_columns[col]['normal_range']}")
    print()

total_zeros = sum(zero_counts_before.values())
print(f"Total invalid zero values: {total_zeros}")
print()

# ============================================================================
# STEP 2: Replace Invalid Zeros with NaN
# ============================================================================

print("STEP 2: Replacing Invalid Zeros with NaN")
print("-" * 80)

# Create a copy to preserve original data
df_cleaned = df.copy()

# Replace zeros with NaN for each column
# Using .replace() method which is efficient for this operation
for col in invalid_zero_columns.keys():
    zeros_before = (df_cleaned[col] == 0).sum()
    df_cleaned[col] = df_cleaned[col].replace(0, np.nan)
    nans_after = df_cleaned[col].isna().sum()
    print(f"{col:20s}: Replaced {zeros_before:4d} zeros with NaN")

print()
print("✓ All invalid zeros replaced with NaN")
print()

# ============================================================================
# STEP 3: Calculate Median Values for Imputation
# ============================================================================

print("STEP 3: Calculating Median Values for Imputation")
print("-" * 80)

# Calculate median for each column (excluding NaN values)
# Median is preferred over mean because:
# 1. It's robust to outliers
# 2. It preserves the distribution better
# 3. It's less affected by extreme values
# 4. Medical data often has skewed distributions

median_values = {}
for col in invalid_zero_columns.keys():
    median_val = df_cleaned[col].median()
    mean_val = df_cleaned[col].mean()
    median_values[col] = median_val
    print(f"{col:20s}: Median = {median_val:8.2f}, Mean = {mean_val:8.2f}")

print()
print("Note: Median is used for imputation as it's more robust to outliers")
print()

# ============================================================================
# STEP 4: Impute Missing Values with Median
# ============================================================================

print("STEP 4: Imputing Missing Values with Median")
print("-" * 80)

# Impute NaN values with median
# .fillna() replaces NaN values with the specified value
# We use median_values dictionary to impute each column with its respective median
for col in invalid_zero_columns.keys():
    missing_before = df_cleaned[col].isna().sum()
    df_cleaned[col] = df_cleaned[col].fillna(median_values[col])
    missing_after = df_cleaned[col].isna().sum()
    print(f"{col:20s}: Imputed {missing_before:4d} missing values with median {median_values[col]:.2f}")

print()
print("✓ All missing values imputed")
print()

# ============================================================================
# STEP 5: Verify the Cleaning Process
# ============================================================================

print("STEP 5: Verification")
print("-" * 80)

# Check for remaining zeros in cleaned columns
print("\nRemaining zero values in cleaned columns:")
print("-" * 50)
remaining_zeros = {}
for col in invalid_zero_columns.keys():
    zeros_remaining = (df_cleaned[col] == 0).sum()
    remaining_zeros[col] = zeros_remaining
    if zeros_remaining == 0:
        print(f"{col:20s}: ✓ No zeros remaining")
    else:
        print(f"{col:20s}: ⚠️  {zeros_remaining} zeros still present")

print()

# Check for remaining NaN values
print("Remaining NaN values:")
print("-" * 50)
remaining_nans = df_cleaned[invalid_zero_columns.keys()].isna().sum()
if remaining_nans.sum() == 0:
    print("✓ No NaN values remaining - all imputed successfully")
else:
    print(remaining_nans[remaining_nans > 0])
print()

# ============================================================================
# STEP 6: Compare Statistics Before and After
# ============================================================================

print("STEP 6: Statistical Comparison (Before vs After)")
print("-" * 80)

print("\nComparison of key statistics:")
print("=" * 80)

for col in invalid_zero_columns.keys():
    print(f"\n{col}:")
    print("-" * 50)
    
    # Before cleaning (original data)
    original_mean = df[col].mean()
    original_median = df[col].median()
    original_std = df[col].std()
    original_min = df[col].min()
    original_max = df[col].max()
    
    # After cleaning
    cleaned_mean = df_cleaned[col].mean()
    cleaned_median = df_cleaned[col].median()
    cleaned_std = df_cleaned[col].std()
    cleaned_min = df_cleaned[col].min()
    cleaned_max = df_cleaned[col].max()
    
    print(f"{'Statistic':<15} {'Before':>12} {'After':>12} {'Change':>12}")
    print("-" * 50)
    print(f"{'Mean':<15} {original_mean:>12.2f} {cleaned_mean:>12.2f} {cleaned_mean - original_mean:>+12.2f}")
    print(f"{'Median':<15} {original_median:>12.2f} {cleaned_median:>12.2f} {cleaned_median - original_median:>+12.2f}")
    print(f"{'Std Dev':<15} {original_std:>12.2f} {cleaned_std:>12.2f} {cleaned_std - original_std:>+12.2f}")
    print(f"{'Min':<15} {original_min:>12.2f} {cleaned_min:>12.2f} {cleaned_min - original_min:>+12.2f}")
    print(f"{'Max':<15} {original_max:>12.2f} {cleaned_max:>12.2f} {cleaned_max - original_max:>+12.2f}")

print()

# ============================================================================
# STEP 7: Save the Cleaned Dataset
# ============================================================================

print("STEP 7: Saving Cleaned Dataset")
print("-" * 80)

# Save the cleaned dataset
output_path = 'data/diabetes_cleaned.csv'
df_cleaned.to_csv(output_path, index=False)
print(f"✓ Cleaned dataset saved to: {output_path}")
print(f"  Original shape: {df.shape}")
print(f"  Cleaned shape: {df_cleaned.shape}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("PREPROCESSING SUMMARY")
print("=" * 80)
print()

print("Actions Performed:")
print("  1. ✓ Identified invalid zero values in medical features")
print("  2. ✓ Replaced invalid zeros with NaN")
print("  3. ✓ Calculated median values for imputation")
print("  4. ✓ Imputed missing values using median")
print("  5. ✓ Verified cleaning process")
print("  6. ✓ Compared statistics before and after")
print("  7. ✓ Saved cleaned dataset")
print()

print("Key Improvements:")
print("  • Removed biologically impossible zero values")
print("  • Preserved data distribution using median imputation")
print("  • Improved data quality for model training")
print("  • Enhanced reliability of medical predictions")
print()

print("Files Generated:")
print(f"  • {output_path} - Cleaned dataset ready for modeling")
print()

print("=" * 80)
print("PREPROCESSING COMPLETE!")
print("=" * 80)

# ============================================================================
# ADDITIONAL EXPLANATION: Why This Matters for Medical Data
# ============================================================================

explanation = """
================================================================================
WHY HANDLING INVALID ZEROS IS CRITICAL FOR MEDICAL DATA
================================================================================

1. BIOLOGICAL ACCURACY
   -------------------
   Medical measurements represent real physiological parameters. Zero values in
   features like glucose, blood pressure, or BMI are biologically impossible
   for living patients. Treating these as real values introduces severe errors
   into the dataset.

2. MODEL BIAS AND ACCURACY
   -----------------------
   - Zero values skew distributions toward lower values
   - Models learn incorrect patterns from invalid data
   - Predictions become unreliable and potentially dangerous
   - Model performance metrics become misleading

3. CLINICAL DECISION SUPPORT
   -------------------------
   Medical AI models are used for:
   - Diagnosis assistance
   - Treatment recommendations
   - Risk assessment
   - Screening decisions
   
   Incorrect data leads to incorrect decisions, which can harm patients.

4. RESEARCH INTEGRITY
   ------------------
   - Published research must use clean, accurate data
   - Invalid values can invalidate study conclusions
   - Reproducibility requires proper data handling
   - Scientific validity depends on data quality

5. REGULATORY COMPLIANCE
   ---------------------
   Medical AI systems may need to comply with:
   - FDA regulations (for medical devices)
   - HIPAA (for patient data)
   - Clinical validation requirements
   - Quality assurance standards

6. ETHICAL RESPONSIBILITY
   ----------------------
   As data scientists working with medical data, we have an ethical obligation
   to:
   - Ensure data accuracy
   - Validate data quality
   - Use appropriate preprocessing methods
   - Document all transformations
   - Report limitations transparently

WHY MEDIAN IMPUTATION?
---------------------
1. Robustness: Median is not affected by extreme outliers
2. Distribution Preservation: Maintains the shape of the data distribution
3. Skewed Data: Medical data often has skewed distributions (median is better)
4. Simplicity: Easy to understand and explain to medical professionals
5. Performance: Works well with most machine learning algorithms

ALTERNATIVE APPROACHES (for future consideration):
-------------------------------------------------
1. K-Nearest Neighbors (KNN) Imputation: Uses similar patients' values
2. Multiple Imputation: Creates multiple datasets with different imputations
3. Domain Knowledge: Use medical guidelines to impute (e.g., normal ranges)
4. Advanced Methods: MICE (Multiple Imputation by Chained Equations)
5. Model-Based: Train a model to predict missing values

For this dataset, median imputation is appropriate because:
- It's simple and interpretable
- The dataset is relatively small (768 samples)
- Missing values are not excessive
- Median preserves the distribution well
"""

print(explanation)

