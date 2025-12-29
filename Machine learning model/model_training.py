"""
Model Training: Logistic Regression and Linear Discriminant Analysis (LDA)
==========================================================================
This script trains Logistic Regression and LDA models for diabetes prediction
with proper feature scaling and comprehensive evaluation.

WHY SCALING IS NECESSARY:
-------------------------
Feature scaling (standardization/normalization) is critical for many machine
learning algorithms, especially for Logistic Regression and LDA. Here's why:

1. ALGORITHM REQUIREMENTS:
   -----------------------
   - Logistic Regression: Uses gradient descent optimization. Features with
     different scales cause the algorithm to converge slowly or get stuck.
     Large-scale features dominate the optimization process.
   
   - LDA (Linear Discriminant Analysis): Assumes features are normally
     distributed and have similar variances. Without scaling, features with
     larger variances dominate the discriminant function.

2. GRADIENT DESCENT OPTIMIZATION:
   ------------------------------
   - Features on different scales create elongated loss function contours
   - Gradient descent takes many more iterations to converge
   - Can lead to numerical instability and poor convergence
   - Scaling creates circular contours, allowing faster convergence

3. FEATURE DOMINANCE:
   ------------------
   - Without scaling, features with larger ranges (e.g., Insulin: 0-846)
     dominate those with smaller ranges (e.g., BMI: 0-67)
   - Model coefficients become biased toward larger-scale features
   - Important features with smaller scales may be ignored

4. DISTANCE-BASED CALCULATIONS:
   ----------------------------
   - LDA uses distance calculations in its discriminant function
   - Euclidean distances are sensitive to feature scales
   - Features with larger scales contribute more to distance calculations
   - Scaling ensures all features contribute equally

5. REGULARIZATION EFFECTIVENESS:
   -----------------------------
   - Regularization (L1/L2) treats all features equally
   - Without scaling, regularization penalizes large-scale features more
   - Scaling ensures fair regularization across all features

6. INTERPRETABILITY:
   -----------------
   - Scaled features allow fair comparison of coefficients
   - Feature importance becomes more meaningful
   - Model interpretation is clearer

STANDARDIZATION vs NORMALIZATION:
--------------------------------
- Standardization (Z-score): (x - mean) / std
  - Centers data at 0, scales to unit variance
  - Preserves outliers
  - Best for normally distributed data
  - Used in this script (StandardScaler)

- Normalization (Min-Max): (x - min) / (max - min)
  - Scales to [0, 1] range
  - Sensitive to outliers
  - Best for bounded ranges

For Logistic Regression and LDA, standardization is preferred.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("MODEL TRAINING: Logistic Regression and LDA")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

print("STEP 1: Loading and Preparing Data")
print("-" * 80)

# Try to load cleaned dataset, otherwise load original and clean it
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

print(f"Dataset shape: {df.shape}")
print()

# Separate features (X) and target (y)
# X contains all columns except 'Outcome'
# y contains only the 'Outcome' column (0 = no diabetes, 1 = diabetes)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"Feature names: {list(X.columns)}")
print()

# Display class distribution
print("Class distribution:")
print(y.value_counts())
print(f"Class percentages: {y.value_counts(normalize=True) * 100}")
print()

# ============================================================================
# STEP 2: Train-Test Split
# ============================================================================

print("STEP 2: Splitting Data into Train and Test Sets")
print("-" * 80)

# Split data into training and testing sets
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 ensures reproducibility (same split every time)
# stratify=y ensures both sets have similar class distribution (important for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintains class distribution in both sets
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print()

print("Training set class distribution:")
print(y_train.value_counts())
print(f"Percentages: {y_train.value_counts(normalize=True) * 100}")
print()

print("Test set class distribution:")
print(y_test.value_counts())
print(f"Percentages: {y_test.value_counts(normalize=True) * 100}")
print()

# ============================================================================
# STEP 3: Feature Scaling - WHY THIS IS CRITICAL
# ============================================================================

print("STEP 3: Feature Scaling (Standardization)")
print("-" * 80)

# Display feature ranges before scaling
# This shows why scaling is necessary - features have very different scales
print("\nFeature ranges BEFORE scaling:")
print("-" * 50)
for col in X.columns:
    min_val = X_train[col].min()
    max_val = X_train[col].max()
    mean_val = X_train[col].mean()
    std_val = X_train[col].std()
    print(f"{col:25s}: Range [{min_val:8.2f}, {max_val:8.2f}], "
          f"Mean={mean_val:8.2f}, Std={std_val:8.2f}")

print()
print("⚠️  PROBLEM: Features have vastly different scales!")
print("   - Insulin ranges from 0 to 846 (huge range)")
print("   - DiabetesPedigreeFunction ranges from 0.08 to 2.42 (tiny range)")
print("   - Without scaling, Insulin will dominate the model")
print()

# Initialize StandardScaler
# StandardScaler standardizes features by removing the mean and scaling to unit variance
# Formula: z = (x - mean) / std
# Result: Features have mean=0 and std=1
scaler = StandardScaler()

# CRITICAL: Fit scaler ONLY on training data
# This prevents data leakage - we must not use test set information during training
# .fit() calculates mean and std from training data
scaler.fit(X_train)

# Transform both training and test sets
# .transform() applies the learned scaling (using training set statistics)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better readability (optional but helpful)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("✓ Scaling applied to training and test sets")
print()

# Display feature statistics after scaling
print("Feature statistics AFTER scaling (should be ~0 mean, ~1 std):")
print("-" * 50)
for col in X_train_scaled.columns:
    mean_val = X_train_scaled[col].mean()
    std_val = X_train_scaled[col].std()
    print(f"{col:25s}: Mean={mean_val:8.4f}, Std={std_val:8.4f}")

print()
print("✓ All features now have similar scales (mean≈0, std≈1)")
print("  This ensures fair treatment of all features in the model")
print()

# ============================================================================
# STEP 4: Train Logistic Regression Model
# ============================================================================

print("STEP 4: Training Logistic Regression Model")
print("-" * 80)

# Initialize Logistic Regression
# max_iter: Maximum iterations for convergence (increased for complex problems)
# random_state: For reproducibility
# class_weight='balanced': Automatically adjusts weights to handle class imbalance
#   - Gives more weight to minority class (diabetes)
#   - Helps improve recall for the positive class
# solver='lbfgs': Limited-memory BFGS optimizer (good for small datasets)
#   - Works well with scaled data
#   - Faster than 'liblinear' for this dataset size
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs'  # Good optimizer for scaled data
)

# Train the model on SCALED training data
# .fit() learns the relationship between features and target
print("Training Logistic Regression on scaled data...")
lr_model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully")
print()

# Get feature coefficients (importance)
# Positive coefficients increase probability of diabetes
# Negative coefficients decrease probability of diabetes
# Larger absolute values indicate stronger influence
print("Logistic Regression Coefficients (Feature Importance):")
print("-" * 50)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

for _, row in feature_importance.iterrows():
    direction = "↑ Increases" if row['Coefficient'] > 0 else "↓ Decreases"
    print(f"{row['Feature']:25s}: {row['Coefficient']:8.4f} ({direction} diabetes risk)")

print()
print(f"Intercept: {lr_model.intercept_[0]:.4f}")
print()

# ============================================================================
# STEP 5: Train LDA Model
# ============================================================================

print("STEP 5: Training Linear Discriminant Analysis (LDA) Model")
print("-" * 80)

# Initialize LDA
# solver='svd': Singular Value Decomposition (works well with scaled data)
#   - More stable than 'lsqr' or 'eigen'
#   - No need to compute covariance matrix
# shrinkage=None: No shrinkage (we have enough data)
# priors=None: Let LDA estimate class priors from data
lda_model = LinearDiscriminantAnalysis(
    solver='svd',
    shrinkage=None
)

# Train the model on SCALED training data
# LDA requires scaling because it assumes features have similar variances
print("Training LDA on scaled data...")
lda_model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully")
print()

# Display LDA coefficients (discriminant function coefficients)
print("LDA Coefficients (Discriminant Function):")
print("-" * 50)
lda_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lda_model.coef_[0],
    'Abs_Coefficient': np.abs(lda_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

for _, row in lda_coef.iterrows():
    direction = "↑ Increases" if row['Coefficient'] > 0 else "↓ Decreases"
    print(f"{row['Feature']:25s}: {row['Coefficient']:8.4f} ({direction} diabetes risk)")

print()
print(f"Intercept: {lda_model.intercept_[0]:.4f}")
print()

# ============================================================================
# STEP 6: Make Predictions
# ============================================================================

print("STEP 6: Making Predictions")
print("-" * 80)

# Predictions on test set (using SCALED test data)
# Important: Test data must be scaled using the SAME scaler fitted on training data
lr_pred = lr_model.predict(X_test_scaled)
lda_pred = lda_model.predict(X_test_scaled)

# Probability predictions (for ROC curve and threshold tuning)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
lda_pred_proba = lda_model.predict_proba(X_test_scaled)[:, 1]

print("✓ Predictions made on test set")
print(f"  Logistic Regression predictions: {len(lr_pred)} samples")
print(f"  LDA predictions: {len(lda_pred)} samples")
print()

# ============================================================================
# STEP 7: Model Evaluation
# ============================================================================

print("STEP 7: Model Evaluation")
print("-" * 80)

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Comprehensive model evaluation"""
    print(f"\n{model_name} Performance:")
    print("=" * 50)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    # Display metrics
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"AUC-ROC:   {auc_roc:.4f} ({auc_roc*100:.2f}%)")
    print()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              No Diabetes  Diabetes")
    print(f"Actual No Diab    {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"      Diabetes    {cm[1,0]:4d}        {cm[1,1]:4d}")
    print()
    
    # Interpretation
    tn, fp, fn, tp = cm.ravel()
    print("Interpretation:")
    print(f"  True Negatives (TN):  {tn:3d} - Correctly predicted no diabetes")
    print(f"  False Positives (FP): {fp:3d} - Incorrectly predicted diabetes (Type I error)")
    print(f"  False Negatives (FN): {fn:3d} - Missed diabetes cases (Type II error)")
    print(f"  True Positives (TP):  {tp:3d} - Correctly predicted diabetes")
    print()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm
    }

# Evaluate both models
lr_metrics = evaluate_model(y_test, lr_pred, lr_pred_proba, "Logistic Regression")
lda_metrics = evaluate_model(y_test, lda_pred, lda_pred_proba, "Linear Discriminant Analysis")

# ============================================================================
# STEP 8: Model Comparison
# ============================================================================

print("STEP 8: Model Comparison")
print("-" * 80)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Logistic Regression': [
        lr_metrics['accuracy'],
        lr_metrics['precision'],
        lr_metrics['recall'],
        lr_metrics['f1'],
        lr_metrics['auc_roc']
    ],
    'LDA': [
        lda_metrics['accuracy'],
        lda_metrics['precision'],
        lda_metrics['recall'],
        lda_metrics['f1'],
        lda_metrics['auc_roc']
    ]
})

comparison['Difference'] = comparison['Logistic Regression'] - comparison['LDA']
comparison['Better Model'] = comparison.apply(
    lambda row: 'Logistic Regression' if row['Difference'] > 0 else 'LDA',
    axis=1
)

print("\nSide-by-Side Comparison:")
print("=" * 80)
print(comparison.to_string(index=False))
print()

# ============================================================================
# STEP 9: Cross-Validation
# ============================================================================

print("STEP 9: Cross-Validation (More Robust Evaluation)")
print("-" * 80)

# Stratified K-Fold Cross-Validation
# k=5: Split data into 5 folds
# stratified: Maintains class distribution in each fold
# This gives more reliable performance estimates
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Performing 5-fold stratified cross-validation...")
print()

# Cross-validation for Logistic Regression
lr_cv_scores = cross_val_score(
    lr_model, X_train_scaled, y_train,
    cv=cv, scoring='roc_auc', n_jobs=-1
)

# Cross-validation for LDA
lda_cv_scores = cross_val_score(
    lda_model, X_train_scaled, y_train,
    cv=cv, scoring='roc_auc', n_jobs=-1
)

print("Cross-Validation Results (AUC-ROC):")
print("-" * 50)
print(f"Logistic Regression:")
print(f"  Fold scores: {lr_cv_scores}")
print(f"  Mean: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")
print()
print(f"LDA:")
print(f"  Fold scores: {lda_cv_scores}")
print(f"  Mean: {lda_cv_scores.mean():.4f} ± {lda_cv_scores.std():.4f}")
print()

# ============================================================================
# STEP 10: ROC Curves
# ============================================================================

print("STEP 10: Generating ROC Curves")
print("-" * 80)

# Calculate ROC curve points
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred_proba)
lda_fpr, lda_tpr, _ = roc_curve(y_test, lda_pred_proba)

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_metrics["auc_roc"]:.3f})', linewidth=2)
plt.plot(lda_fpr, lda_tpr, label=f'LDA (AUC = {lda_metrics["auc_roc"]:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves: Logistic Regression vs LDA', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ ROC curves saved to 'model_roc_curves.png'")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print()

print("Key Takeaways:")
print("  1. ✓ Feature scaling is ESSENTIAL for Logistic Regression and LDA")
print("  2. ✓ Both models trained successfully on scaled data")
print("  3. ✓ Models evaluated using multiple metrics (accuracy, precision, recall, F1, AUC-ROC)")
print("  4. ✓ Cross-validation provides robust performance estimates")
print("  5. ✓ Class imbalance handled using class_weight='balanced' in Logistic Regression")
print()

print("Why Scaling Was Necessary:")
print("  • Features had vastly different scales (e.g., Insulin: 0-846 vs DiabetesPedigreeFunction: 0.08-2.42)")
print("  • Without scaling, large-scale features would dominate the models")
print("  • Scaling ensures fair treatment of all features")
print("  • Enables faster convergence and better model performance")
print("  • Makes coefficients comparable and interpretable")
print()

print("Model Performance:")
print(f"  • Logistic Regression AUC-ROC: {lr_metrics['auc_roc']:.4f}")
print(f"  • LDA AUC-ROC: {lda_metrics['auc_roc']:.4f}")
print()

print("Files Generated:")
print("  • model_roc_curves.png - ROC curve comparison")
print()

print("=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)

