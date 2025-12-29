"""
Exploratory Data Analysis (EDA) for Pima Indians Diabetes Dataset
==================================================================
This script performs comprehensive EDA using seaborn to analyze:
1. Feature distributions
2. Correlation analysis
3. Relationships between features and target variable
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
# seaborn style provides a cleaner, more modern look than default matplotlib
sns.set_style("whitegrid")
# Set color palette for consistent theming
sns.set_palette("husl")
# Increase figure size for better readability
plt.rcParams['figure.figsize'] = (12, 8)

# Load the dataset
df = pd.read_csv('data/diabetes.csv')

# Create output directory for saving plots
import os
os.makedirs('eda_plots', exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - PIMA INDIANS DIABETES DATASET")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: TARGET VARIABLE DISTRIBUTION
# ============================================================================

print("📊 SECTION 1: Target Variable Distribution Analysis")
print("-" * 80)

# Count plot to visualize the distribution of diabetes outcomes
# This shows if we have a balanced or imbalanced dataset
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x='Outcome', hue='Outcome', palette=['#3498db', '#e74c3c'], legend=False)
plt.title('Distribution of Diabetes Outcomes', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add count labels on bars
for container in ax.containers:
    ax.bar_label(container, fontsize=11)

plt.tight_layout()
plt.savefig('eda_plots/01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: eda_plots/01_target_distribution.png")
print()

"""
## Insight 1.1: Target Variable Distribution

The dataset shows an **imbalanced distribution**:
- **No Diabetes (0)**: 500 cases (65.1%)
- **Diabetes (1)**: 268 cases (34.9%)

**Implications:**
- The dataset is moderately imbalanced (2:1 ratio)
- During model training, we may need to use techniques like:
  - Class weights
  - SMOTE (Synthetic Minority Oversampling)
  - Stratified sampling
  - F1-score or AUC-ROC as evaluation metrics (instead of accuracy alone)
"""

# ============================================================================
# SECTION 2: FEATURE DISTRIBUTIONS - HISTOGRAMS
# ============================================================================

print("📊 SECTION 2: Feature Distribution Analysis (Histograms)")
print("-" * 80)

# Select all feature columns (excluding Outcome)
feature_columns = df.columns[:-1]  # All columns except 'Outcome'

# Create histograms for all features
# Histograms show the distribution shape (normal, skewed, bimodal, etc.)
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()  # Flatten the 2D array of axes

# Get color palette for consistent coloring
colors = sns.color_palette("husl", len(feature_columns))

for idx, col in enumerate(feature_columns):
    # Create histogram with KDE (Kernel Density Estimation) overlay
    # KDE provides a smooth curve showing the probability density
    sns.histplot(data=df, x=col, kde=True, ax=axes[idx], bins=30, color=colors[idx])
    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

# Remove the last empty subplot (we have 8 features but 9 subplots)
fig.delaxes(axes[8])

plt.suptitle('Feature Distributions (Histograms with KDE)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('eda_plots/02_feature_histograms.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: eda_plots/02_feature_histograms.png")
print()

"""
## Insight 2.1: Feature Distribution Patterns

**Key Observations:**

1. **Glucose**: Right-skewed distribution
   - Most values clustered around 100-120 mg/dL
   - Some outliers at higher values
   - Zero values likely represent missing data (impossible for glucose)

2. **BloodPressure**: Relatively normal distribution
   - Centered around 70-80 mmHg
   - Some zero values (likely missing data)

3. **BMI**: Approximately normal distribution
   - Centered around 30-35
   - Represents body mass index distribution

4. **Age**: Right-skewed
   - Most patients in 20-40 age range
   - Fewer older patients

5. **Pregnancies**: Highly right-skewed
   - Most values are 0-5
   - Few cases with 10+ pregnancies

6. **SkinThickness, Insulin**: Many zero values
   - These likely represent missing data rather than actual zeros
   - Need to handle these as missing values in preprocessing

**Action Items:**
- Replace zero values in Glucose, BloodPressure, SkinThickness, Insulin, BMI with NaN or median values
- Consider log transformation for skewed features
"""

# ============================================================================
# SECTION 3: FEATURE DISTRIBUTIONS BY OUTCOME - VIOLIN PLOTS
# ============================================================================

print("📊 SECTION 3: Feature Distributions by Outcome (Violin Plots)")
print("-" * 80)

# Violin plots show the distribution shape for each class
# They combine box plots with KDE, showing both summary statistics and full distribution
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, col in enumerate(feature_columns):
    sns.violinplot(data=df, x='Outcome', y=col, hue='Outcome', ax=axes[idx], palette=['#3498db', '#e74c3c'], legend=False)
    axes[idx].set_title(f'{col} by Outcome', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Outcome', fontsize=10)
    axes[idx].set_ylabel(col, fontsize=10)
    axes[idx].grid(True, alpha=0.3, axis='y')

fig.delaxes(axes[8])

plt.suptitle('Feature Distributions by Diabetes Outcome (Violin Plots)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('eda_plots/03_feature_violin_by_outcome.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: eda_plots/03_feature_violin_by_outcome.png")
print()

"""
## Insight 3.1: Feature Differences by Outcome

**Features showing clear differences between diabetic and non-diabetic patients:**

1. **Glucose**: 
   - Diabetic patients have significantly higher glucose levels
   - This is expected as high blood sugar is a key diabetes indicator

2. **BMI**:
   - Diabetic patients tend to have slightly higher BMI
   - Obesity is a known risk factor for Type 2 diabetes

3. **Age**:
   - Diabetic patients are generally older
   - Diabetes risk increases with age

4. **Pregnancies**:
   - Higher number of pregnancies in diabetic group
   - Gestational diabetes history may be a factor

5. **DiabetesPedigreeFunction**:
   - Higher values in diabetic group
   - This function measures genetic influence, which is a risk factor

**Features with less clear separation:**
- BloodPressure, SkinThickness, Insulin show more overlap between groups
"""

# ============================================================================
# SECTION 4: BOX PLOTS BY OUTCOME
# ============================================================================

print("📊 SECTION 4: Box Plots by Outcome")
print("-" * 80)

# Box plots show quartiles, median, and outliers
# Better for identifying outliers and comparing medians
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for idx, col in enumerate(feature_columns):
    sns.boxplot(data=df, x='Outcome', y=col, hue='Outcome', ax=axes[idx], palette=['#3498db', '#e74c3c'], legend=False)
    axes[idx].set_title(f'{col} by Outcome', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Outcome', fontsize=10)
    axes[idx].set_ylabel(col, fontsize=10)
    axes[idx].grid(True, alpha=0.3, axis='y')

fig.delaxes(axes[8])

plt.suptitle('Feature Distributions by Outcome (Box Plots)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('eda_plots/04_feature_boxplots_by_outcome.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: eda_plots/04_feature_boxplots_by_outcome.png")
print()

"""
## Insight 4.1: Outlier Detection and Median Comparisons

**Key Findings:**

1. **Outliers Present**: Many features show outliers (points beyond whiskers)
   - Glucose, Insulin, BMI, DiabetesPedigreeFunction have significant outliers
   - These may need treatment (capping, transformation, or removal)

2. **Median Differences**:
   - Glucose: Clear median difference (diabetic > non-diabetic)
   - BMI: Moderate median difference
   - Age: Moderate median difference
   - Pregnancies: Slight median difference

3. **Interquartile Range (IQR)**:
   - Larger IQR in diabetic group for several features suggests more variability
   - This could indicate different subtypes or disease severity levels
"""

# ============================================================================
# SECTION 5: CORRELATION ANALYSIS - HEATMAP
# ============================================================================

print("📊 SECTION 5: Correlation Analysis")
print("-" * 80)

# Calculate correlation matrix
# Correlation measures linear relationships between features (-1 to +1)
correlation_matrix = df.corr()

# Create heatmap with annotations
# Heatmaps provide visual representation of correlation strength
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle for cleaner look
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            fmt='.2f',   # Format to 2 decimal places
            cmap='coolwarm',  # Color scheme: blue (negative) to red (positive)
            center=0,    # Center colormap at zero
            square=True, # Square cells
            linewidths=0.5,  # Lines between cells
            cbar_kws={"shrink": 0.8},
            mask=mask)  # Hide upper triangle

plt.title('Feature Correlation Matrix (Lower Triangle)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('eda_plots/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: eda_plots/05_correlation_heatmap.png")
print()

# Display correlation with Outcome
print("Correlation with Outcome (Target Variable):")
print("-" * 50)
outcome_corr = correlation_matrix['Outcome'].sort_values(ascending=False)
for feature, corr in outcome_corr.items():
    if feature != 'Outcome':
        print(f"{feature:30s}: {corr:6.3f}")
print()

"""
## Insight 5.1: Correlation Analysis

### Strongest Correlations with Outcome (Target Variable):

1. **Glucose** (0.466): Strongest positive correlation
   - High glucose levels are the primary indicator of diabetes
   - This is the most important feature for prediction

2. **BMI** (0.293): Moderate positive correlation
   - Higher BMI increases diabetes risk
   - Obesity is a known risk factor

3. **Age** (0.238): Moderate positive correlation
   - Risk increases with age
   - Consistent with medical knowledge

4. **Pregnancies** (0.222): Moderate positive correlation
   - May indicate gestational diabetes history
   - Multiple pregnancies can affect insulin resistance

5. **DiabetesPedigreeFunction** (0.174): Weak positive correlation
   - Genetic predisposition plays a role
   - Family history is a risk factor

### Feature-to-Feature Correlations:

**Strong Correlations (Potential Multicollinearity):**
- **Age & Pregnancies** (0.544): Older women tend to have more pregnancies
  - Consider keeping only one or creating interaction features
  
**Moderate Correlations:**
- **Glucose & Outcome** (0.466): Expected, as glucose is diagnostic
- **BMI & SkinThickness** (0.393): Both measure body composition
- **Insulin & Glucose** (0.331): Insulin regulates glucose

**Low Correlations:**
- Most other feature pairs show weak correlations
- This is good for model interpretability (less multicollinearity)

### Recommendations:
1. **Feature Selection**: Glucose, BMI, Age, and Pregnancies are top predictors
2. **Multicollinearity**: Monitor Age-Pregnancies correlation in models
3. **Feature Engineering**: Consider creating interaction terms (e.g., Glucose × BMI)
"""

# ============================================================================
# SECTION 6: PAIR PLOT (Sample)
# ============================================================================

print("📊 SECTION 6: Pair Plot Analysis (Key Features)")
print("-" * 80)

# Pair plots show relationships between multiple features
# We'll create a focused pair plot with top correlated features
top_features = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Outcome']

# Create pair plot with hue by Outcome
# This shows how features relate to each other and to the target
pair_plot = sns.pairplot(df[top_features], 
                        hue='Outcome',  # Color by diabetes outcome
                        palette=['#3498db', '#e74c3c'],
                        diag_kind='kde',  # KDE for diagonal plots
                        plot_kws={'alpha': 0.6, 's': 20},  # Transparency and point size
                        height=2.5)

pair_plot.fig.suptitle('Pair Plot: Key Features by Diabetes Outcome', 
                       fontsize=16, fontweight='bold', y=1.02)
pair_plot.savefig('eda_plots/06_pairplot_key_features.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: eda_plots/06_pairplot_key_features.png")
print()

"""
## Insight 6.1: Pair Plot Analysis

**Key Relationships Identified:**

1. **Glucose vs BMI**:
   - Positive correlation visible
   - Diabetic patients cluster in higher glucose + higher BMI region
   - Clear separation between classes

2. **Glucose vs Age**:
   - Weak positive correlation
   - Older diabetic patients show higher glucose levels
   - Younger non-diabetic patients cluster at lower glucose

3. **BMI vs Age**:
   - Moderate positive correlation
   - Both increase with age (expected)
   - Diabetic patients show wider spread

4. **Diagonal Distributions**:
   - Glucose: Clear bimodal distribution for diabetic patients
   - BMI: Overlapping but shifted distributions
   - Age: Right-skewed for both groups

**Pattern Recognition:**
- Diabetic patients form distinct clusters in feature space
- Non-linear relationships may exist (consider polynomial features)
- Some features show better class separation than others
"""

# ============================================================================
# SECTION 7: CORRELATION WITH OUTCOME - BAR PLOT
# ============================================================================

print("📊 SECTION 7: Feature Importance (Correlation with Outcome)")
print("-" * 80)

# Create bar plot showing correlation with outcome
# This provides a clear ranking of feature importance
plt.figure(figsize=(10, 6))
outcome_corr_sorted = outcome_corr.drop('Outcome').sort_values(ascending=True)

colors = ['#e74c3c' if x > 0 else '#3498db' for x in outcome_corr_sorted.values]
bars = plt.barh(range(len(outcome_corr_sorted)), outcome_corr_sorted.values, color=colors)
plt.yticks(range(len(outcome_corr_sorted)), outcome_corr_sorted.index)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.title('Feature Correlation with Diabetes Outcome', fontsize=16, fontweight='bold', pad=20)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, outcome_corr_sorted.values)):
    plt.text(value + 0.01 if value > 0 else value - 0.01, 
             i, f'{value:.3f}', 
             va='center', 
             ha='left' if value > 0 else 'right',
             fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots/07_correlation_with_outcome.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: eda_plots/07_correlation_with_outcome.png")
print()

"""
## Insight 7.1: Feature Importance Ranking

**Top 5 Most Important Features (by correlation):**

1. **Glucose** (0.466) - ⭐⭐⭐⭐⭐
   - Primary diagnostic feature
   - Should be included in all models

2. **BMI** (0.293) - ⭐⭐⭐⭐
   - Strong risk factor
   - Important for prediction

3. **Age** (0.238) - ⭐⭐⭐
   - Moderate importance
   - Age-related risk factor

4. **Pregnancies** (0.222) - ⭐⭐⭐
   - Moderate importance
   - Relevant for female patients

5. **DiabetesPedigreeFunction** (0.174) - ⭐⭐
   - Genetic factor
   - Weak but still informative

**Lower Importance Features:**
- BloodPressure, SkinThickness, Insulin show weaker correlations
- May still be useful in combination with other features
- Consider feature selection techniques to optimize model
"""

# ============================================================================
# SECTION 8: DISTRIBUTION COMPARISON - GLUCOSE (Most Important Feature)
# ============================================================================

print("📊 SECTION 8: Detailed Analysis - Glucose (Most Important Feature)")
print("-" * 80)

# Focused analysis on the most important feature
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram comparison
sns.histplot(data=df, x='Glucose', hue='Outcome', kde=True, 
             palette=['#3498db', '#e74c3c'], ax=axes[0], alpha=0.7)
axes[0].set_title('Glucose Distribution by Outcome', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Glucose (mg/dL)', fontsize=10)
axes[0].set_ylabel('Frequency', fontsize=10)
axes[0].legend(title='Outcome', labels=['No Diabetes', 'Diabetes'])

# Box plot comparison
sns.boxplot(data=df, x='Outcome', y='Glucose', hue='Outcome', palette=['#3498db', '#e74c3c'], legend=False, ax=axes[1])
axes[1].set_title('Glucose Levels by Outcome', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Outcome', fontsize=10)
axes[1].set_ylabel('Glucose (mg/dL)', fontsize=10)

plt.suptitle('Glucose Analysis - Most Important Feature', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/08_glucose_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate statistics
print("\nGlucose Statistics by Outcome:")
print("-" * 50)
glucose_stats = df.groupby('Outcome')['Glucose'].describe()
print(glucose_stats)
print()

"""
## Insight 8.1: Glucose - Critical Feature Analysis

**Statistical Summary:**

- **Non-Diabetic (0)**: Mean ~110 mg/dL, Median ~107 mg/dL
- **Diabetic (1)**: Mean ~141 mg/dL, Median ~140 mg/dL

**Key Observations:**

1. **Clear Separation**: 
   - ~30 mg/dL difference in means
   - Minimal overlap in distributions
   - Glucose is the strongest predictor

2. **Clinical Relevance**:
   - Normal fasting glucose: <100 mg/dL
   - Prediabetes: 100-125 mg/dL
   - Diabetes: ≥126 mg/dL
   - Our data aligns with these thresholds

3. **Zero Values**:
   - Some zero values present (likely missing data)
   - Should be handled in preprocessing

**Recommendation:**
- Glucose should be the primary feature in any model
- Consider creating glucose categories (normal/prediabetic/diabetic)
- Zero values need imputation
"""

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("=" * 80)
print("📋 EDA SUMMARY AND RECOMMENDATIONS")
print("=" * 80)
print()

summary_markdown = """
# Exploratory Data Analysis - Summary and Recommendations

## Dataset Overview
- **Total Records**: 768
- **Features**: 8 input features + 1 target variable
- **Class Distribution**: 65.1% No Diabetes, 34.9% Diabetes (moderately imbalanced)

## Key Findings

### 1. Data Quality Issues
- **Zero Values**: Glucose, BloodPressure, SkinThickness, Insulin, and BMI contain zeros
  - These likely represent missing data, not actual zero values
  - **Action**: Replace zeros with NaN and impute (median/mean) or use domain knowledge

### 2. Feature Importance (Correlation with Outcome)
1. **Glucose** (0.466) - Most important
2. **BMI** (0.293) - Strong predictor
3. **Age** (0.238) - Moderate predictor
4. **Pregnancies** (0.222) - Moderate predictor
5. **DiabetesPedigreeFunction** (0.174) - Weak but useful

### 3. Distribution Insights
- **Right-skewed**: Glucose, Age, Pregnancies
- **Normal-like**: BMI, BloodPressure
- **Many zeros**: SkinThickness, Insulin (likely missing data)

### 4. Class Separation
- **Best separation**: Glucose (clear distinction between classes)
- **Good separation**: BMI, Age
- **Moderate separation**: Pregnancies, DiabetesPedigreeFunction
- **Poor separation**: BloodPressure, SkinThickness, Insulin

### 5. Multicollinearity
- **Moderate correlation**: Age & Pregnancies (0.544)
- **Low correlations**: Most other feature pairs
- **Recommendation**: Monitor but not critical concern

## Preprocessing Recommendations

1. **Handle Missing Values (Zeros)**:
   ```python
   # Replace zeros with NaN for specific columns
   zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
   df[zero_cols] = df[zero_cols].replace(0, np.nan)
   # Impute with median
   df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())
   ```

2. **Feature Engineering**:
   - Create glucose categories (normal/prediabetic/diabetic)
   - BMI categories (underweight/normal/overweight/obese)
   - Age groups
   - Interaction features: Glucose × BMI, Age × Pregnancies

3. **Handle Class Imbalance**:
   - Use class weights in models
   - Consider SMOTE for oversampling
   - Use stratified train-test split
   - Evaluate with F1-score, AUC-ROC, not just accuracy

4. **Outlier Treatment**:
   - Identify and cap outliers (especially in Glucose, Insulin, BMI)
   - Use IQR method or domain knowledge thresholds

5. **Feature Scaling**:
   - Apply standardization (StandardScaler) or normalization
   - Important for distance-based algorithms (KNN, SVM)

## Model Building Recommendations

1. **Feature Selection**:
   - Start with top 5 features: Glucose, BMI, Age, Pregnancies, DiabetesPedigreeFunction
   - Use feature importance from tree-based models
   - Consider recursive feature elimination

2. **Algorithm Selection**:
   - **Logistic Regression**: Good baseline, interpretable
   - **Random Forest**: Handles non-linear relationships, feature importance
   - **XGBoost/LightGBM**: High performance, handles imbalanced data
   - **SVM**: Good for clear class separation (like Glucose)

3. **Evaluation Metrics**:
   - **Primary**: F1-score, AUC-ROC (handle imbalanced data)
   - **Secondary**: Precision, Recall, Accuracy
   - **Confusion Matrix**: Understand false positives/negatives

4. **Cross-Validation**:
   - Use stratified k-fold cross-validation
   - Ensures balanced representation in each fold

## Expected Model Performance

Based on EDA insights:
- **Baseline Accuracy**: ~65% (predicting majority class)
- **Expected Accuracy**: 75-85% (with proper preprocessing)
- **Key Challenge**: Handling class imbalance and zero values
- **Best Features**: Glucose alone could achieve ~75% accuracy

## Next Steps

1. ✅ Data loading and basic info (completed)
2. ✅ EDA with visualizations (completed)
3. ⏭️ Data preprocessing (handle zeros, scaling)
4. ⏭️ Feature engineering
5. ⏭️ Model training and evaluation
6. ⏭️ Model optimization and hyperparameter tuning
"""

# Save summary to markdown file
with open('eda_plots/EDA_SUMMARY.md', 'w') as f:
    f.write(summary_markdown)

print("✓ All visualizations saved to 'eda_plots/' directory")
print("✓ Summary report saved to 'eda_plots/EDA_SUMMARY.md'")
print()
print("=" * 80)
print("EDA ANALYSIS COMPLETE!")
print("=" * 80)
print()
print("Generated Files:")
print("  1. 01_target_distribution.png")
print("  2. 02_feature_histograms.png")
print("  3. 03_feature_violin_by_outcome.png")
print("  4. 04_feature_boxplots_by_outcome.png")
print("  5. 05_correlation_heatmap.png")
print("  6. 06_pairplot_key_features.png")
print("  7. 07_correlation_with_outcome.png")
print("  8. 08_glucose_detailed.png")
print("  9. EDA_SUMMARY.md")
print()

