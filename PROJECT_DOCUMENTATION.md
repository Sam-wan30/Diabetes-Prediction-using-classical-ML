# Diabetes Prediction using Classical Machine Learning
## IIT Guwahati - Data Science Project

**Author:** Samiksha Vishwanath Wanjari  
**Institution:** IIT Guwahati – Certified Program  
**Date:** 2024

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Approach](#2-approach)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Insights and Conclusions](#5-insights-and-conclusions)
6. [Future Scope](#6-future-scope)

---

## 1. Problem Statement

### 1.1 Background

Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels, affecting millions of people worldwide. According to the World Health Organization, diabetes is a leading cause of blindness, kidney failure, heart attacks, stroke, and lower limb amputation. Early detection and risk assessment are crucial for effective prevention and management.

### 1.2 Problem Definition

Traditional diabetes diagnosis relies on clinical assessment and laboratory tests, which may not always be readily available or cost-effective for screening large populations. There is a critical need for automated risk prediction systems that can:

- Identify individuals at high risk of developing diabetes based on readily available health metrics
- Provide early warning before glucose levels reach diagnostic thresholds
- Support healthcare providers in prioritizing screening and intervention
- Enable preventive care through early risk identification

### 1.3 Research Questions

1. Can machine learning models accurately predict diabetes risk using basic health metrics?
2. Which health factors are most predictive of diabetes risk?
3. How do different machine learning algorithms compare in diabetes prediction?
4. Can the models identify risk before glucose levels reach diagnostic thresholds?

### 1.4 Objectives

The primary objectives of this project are:

1. To perform comprehensive exploratory data analysis on the Pima Indians Diabetes dataset
2. To preprocess medical data appropriately, handling missing values and ensuring data quality
3. To train and evaluate classification models using Logistic Regression and Linear Discriminant Analysis (LDA)
4. To interpret model coefficients to understand feature importance and clinical significance
5. To assess model performance using multiple evaluation metrics
6. To develop an interactive web application for real-time diabetes risk assessment

---

## 2. Approach

### 2.1 Dataset Selection

**Dataset:** Pima Indians Diabetes Dataset  
**Source:** UCI Machine Learning Repository / National Institute of Diabetes and Digestive and Kidney Diseases  
**Characteristics:**
- **Total Samples:** 768 female patients
- **Features:** 8 medical attributes
- **Target:** Binary outcome (0 = No Diabetes, 1 = Diabetes)
- **Class Distribution:** 500 (65.1%) No Diabetes, 268 (34.9%) Diabetes

**Rationale for Selection:**
- Well-established benchmark dataset in medical machine learning
- Contains real clinical measurements
- Appropriate size for classical ML approaches
- Represents a specific population (Pima Indians) with documented high diabetes prevalence

### 2.2 Feature Description

The dataset includes the following 8 features:

1. **Pregnancies:** Number of times pregnant
2. **Glucose:** Plasma glucose concentration (mg/dL) - Primary diagnostic marker
3. **BloodPressure:** Diastolic blood pressure (mmHg)
4. **SkinThickness:** Triceps skin fold thickness (mm) - Indicator of body fat
5. **Insulin:** 2-Hour serum insulin (μU/mL) - Insulin resistance marker
6. **BMI:** Body Mass Index (kg/m²) - Obesity indicator
7. **DiabetesPedigreeFunction:** Function representing genetic predisposition to diabetes
8. **Age:** Age in years

### 2.3 Algorithm Selection

**Selected Algorithms:**

1. **Logistic Regression**
   - **Rationale:** Interpretable, provides probability outputs, handles binary classification well
   - **Advantages:** Coefficients provide clinical insights, probabilistic predictions, widely used in medical applications
   - **Configuration:** Balanced class weights, LBFGS solver, max_iter=1000

2. **Linear Discriminant Analysis (LDA)**
   - **Rationale:** Classical statistical method, assumes normal distributions, provides discriminant functions
   - **Advantages:** Simple assumptions, good for comparison, provides class separation insights
   - **Configuration:** SVD solver, no shrinkage

**Why These Algorithms:**
- Both are classical, interpretable methods suitable for medical applications
- Provide coefficients/weights that can be clinically interpreted
- Well-established in medical literature
- Good baseline for comparison with more complex models

### 2.4 Evaluation Strategy

**Metrics Selected:**
- **Accuracy:** Overall correctness
- **Precision:** Correctness of positive predictions
- **Recall (Sensitivity):** Ability to catch all diabetes cases (critical for medical diagnosis)
- **F1-Score:** Balance between precision and recall
- **ROC-AUC:** Threshold-independent discriminative ability

**Why These Metrics:**
- Medical applications require high recall (minimize false negatives)
- Multiple metrics provide comprehensive assessment
- ROC-AUC allows threshold-independent evaluation
- Confusion matrices provide detailed error analysis

---

## 3. Methodology

### 3.1 Data Preprocessing Pipeline

#### Step 1: Data Loading and Initial Inspection
- Loaded dataset from UCI repository
- Examined data structure, missing values, and data types
- Identified 652 invalid zero values across 5 medical features

#### Step 2: Handling Invalid Zero Values
**Problem:** Zero values in medical measurements are biologically impossible:
- Glucose: 5 zeros (cannot be 0 mg/dL in living patients)
- BloodPressure: 35 zeros (cannot be 0 mmHg)
- SkinThickness: 227 zeros (cannot be 0 mm)
- Insulin: 374 zeros (extremely rare, likely missing data)
- BMI: 11 zeros (cannot be 0)

**Solution:**
1. Replaced zeros with NaN to properly mark missing values
2. Imputed missing values using median (robust to outliers)
3. Verified all zeros and NaN values were handled

**Imputation Values:**
- Glucose: 117.00 mg/dL
- BloodPressure: 72.00 mmHg
- SkinThickness: 29.00 mm
- Insulin: 125.00 μU/mL
- BMI: 32.30 kg/m²

**Why Median Imputation:**
- Robust to outliers (unlike mean)
- Preserves distribution shape better
- More appropriate for skewed medical data
- Represents typical values in medical contexts

#### Step 3: Feature Scaling
**Method:** StandardScaler (mean=0, std=1)

**Why Scaling is Critical:**
- **Logistic Regression:** Uses gradient descent; features with different scales cause slow convergence
- **LDA:** Assumes features have similar variances; distance calculations are sensitive to scales
- **Fair Comparison:** Ensures all features contribute equally to the model

### 3.2 Exploratory Data Analysis (EDA)

#### 3.2.1 Class Distribution Analysis
- **No Diabetes (0):** 500 cases (65.1%)
- **Diabetes (1):** 268 cases (34.9%)
- **Imbalance Ratio:** ~2:1 (moderate imbalance)
- **Handling:** Used stratified splitting and balanced class weights

#### 3.2.2 Feature Distribution Analysis
- **Right-skewed:** Glucose, Age, Pregnancies
- **Approximately normal:** BMI, BloodPressure
- **Many zeros:** SkinThickness, Insulin (indicating missing data)

#### 3.2.3 Correlation Analysis
**Correlation with Outcome (Target Variable):**

| Rank | Feature | Correlation | Strength |
|------|---------|-------------|----------|
| 1 | Glucose | 0.467 | Strong |
| 2 | BMI | 0.293 | Strong |
| 3 | Age | 0.238 | Moderate |
| 4 | Pregnancies | 0.222 | Moderate |
| 5 | DiabetesPedigreeFunction | 0.174 | Weak |
| 6 | Insulin | 0.131 | Weak |
| 7 | SkinThickness | 0.075 | Very Weak |
| 8 | BloodPressure | 0.065 | Very Weak |

**Key Findings:**
- Glucose shows the strongest correlation (0.467), validating its role as primary diagnostic marker
- BMI shows strong association (0.293), confirming obesity-diabetes link
- Age and Pregnancies show moderate correlations
- Features have vastly different scales, confirming need for scaling

### 3.3 Model Training Process

#### 3.3.1 Data Splitting
- **Training Set:** 614 samples (80%)
- **Test Set:** 154 samples (20%)
- **Method:** Stratified splitting (maintains class distribution)
- **Random State:** 42 (for reproducibility)

#### 3.3.2 Logistic Regression Training
**Configuration:**
- Solver: LBFGS (Limited-memory BFGS)
- Max iterations: 1000
- Class weights: Balanced (handles class imbalance)
- Random state: 42

**Training Process:**
1. Features standardized using StandardScaler
2. Model trained on scaled training data
3. Balanced class weights ensure minority class (diabetes) receives appropriate attention
4. Coefficients extracted for interpretation

#### 3.3.3 LDA Training
**Configuration:**
- Solver: SVD (Singular Value Decomposition)
- Shrinkage: None
- Priors: Estimated from data

**Training Process:**
1. Features standardized (required for LDA)
2. Model trained on scaled training data
3. Discriminant function coefficients extracted
4. Class separation maximized

### 3.4 Model Evaluation Process

#### 3.4.1 Prediction Generation
- Generated binary predictions (0/1) for both models
- Generated probability predictions for ROC curve analysis
- Applied to test set only (no data leakage)

#### 3.4.2 Metric Calculation
- Calculated accuracy, precision, recall, F1-score for both models
- Generated confusion matrices for detailed error analysis
- Calculated ROC-AUC scores for threshold-independent evaluation
- Created ROC curves for visual comparison

#### 3.4.3 Feature Importance Analysis
- Extracted Logistic Regression coefficients
- Sorted by absolute value to identify most important features
- Interpreted coefficients in clinical context
- Validated findings against medical literature

---

## 4. Results

### 4.1 Model Performance Comparison

#### 4.1.1 Logistic Regression Results

**Performance Metrics:**
- **Accuracy:** 73.38%
- **Precision:** 60.32%
- **Recall:** 70.37%
- **F1-Score:** 64.96%
- **ROC-AUC:** 81.26%

**Confusion Matrix:**
- True Negatives (TN): 75
- False Positives (FP): 25
- False Negatives (FN): 16
- True Positives (TP): 38

**Interpretation:**
- Correctly identifies 70.37% of all diabetes cases (good recall)
- When predicting diabetes, 60.32% are correct (moderate precision)
- Overall accuracy of 73.38% indicates good performance
- ROC-AUC of 81.26% shows strong discriminative ability

#### 4.1.2 LDA Results

**Performance Metrics:**
- **Accuracy:** 70.13%
- **Precision:** 59.09%
- **Recall:** 48.15%
- **F1-Score:** 53.06%
- **ROC-AUC:** 81.26%

**Confusion Matrix:**
- True Negatives (TN): 82
- False Positives (FP): 18
- False Negatives (FN): 28
- True Positives (TP): 26

**Interpretation:**
- Lower recall (48.15%) means it misses more diabetes cases
- Better precision (59.09%) with fewer false positives
- Similar ROC-AUC (81.26%) indicates comparable discriminative ability
- Lower overall accuracy (70.13%) compared to Logistic Regression

#### 4.1.3 Side-by-Side Comparison

| Metric | Logistic Regression | LDA | Winner |
|--------|---------------------|-----|--------|
| **Accuracy** | 73.38% | 70.13% | Logistic Regression |
| **Precision** | 60.32% | 59.09% | Logistic Regression |
| **Recall** | 70.37% | 48.15% | **Logistic Regression** |
| **F1-Score** | 64.96% | 53.06% | Logistic Regression |
| **ROC-AUC** | 81.26% | 81.26% | Tie |

**Key Observations:**
1. **Similar Discriminative Ability:** Both models achieve identical ROC-AUC (81.26%), indicating comparable ability to distinguish between classes
2. **Recall Advantage:** Logistic Regression has significantly better recall (70.37% vs. 48.15%), making it superior for medical applications where missing diabetes cases is critical
3. **Overall Performance:** Logistic Regression outperforms LDA across all metrics except having slightly more false positives
4. **Clinical Preference:** For medical diagnosis, Logistic Regression is preferred due to higher recall (catches more diabetes cases)

### 4.2 Feature Importance Results

#### 4.2.1 Logistic Regression Coefficients

**Top 4 Most Important Features (by absolute coefficient value):**

| Rank | Feature | Coefficient | Absolute Value | Clinical Significance |
|------|---------|------------|---------------|---------------------|
| 1 | **Glucose** | 1.1834 | 1.1834 | Primary diagnostic marker |
| 2 | **BMI** | 0.7097 | 0.7097 | Obesity-diabetes link |
| 3 | **Pregnancies** | 0.3730 | 0.3730 | Gestational diabetes history |
| 4 | **DiabetesPedigreeFunction** | 0.2877 | 0.2877 | Genetic predisposition |

**Complete Coefficient Table:**

| Feature | Coefficient | Effect on Risk |
|---------|------------|---------------|
| Glucose | 1.1834 | ↑ Strongly increases |
| BMI | 0.7097 | ↑ Increases |
| Pregnancies | 0.3730 | ↑ Increases |
| DiabetesPedigreeFunction | 0.2877 | ↑ Increases |
| Age | 0.1864 | ↑ Increases |
| SkinThickness | 0.0139 | ↑ Slightly increases |
| Insulin | -0.0447 | ↓ Slightly decreases |
| BloodPressure | -0.0145 | ↓ Slightly decreases |

**Intercept:** -0.2570

#### 4.2.2 Clinical Validation of Feature Importance

**1. Glucose (Coefficient: 1.18) - Most Important**
- **Medical Validation:** Blood glucose is the primary diagnostic criterion for diabetes (ADA guidelines: ≥126 mg/dL fasting)
- **Correlation:** 0.467 (strongest correlation with outcome)
- **Clinical Relevance:** Directly reflects insulin function and glucose metabolism
- **Interpretation:** A one standard deviation increase in glucose increases log-odds of diabetes by 1.18

**2. BMI (Coefficient: 0.71) - Second Most Important**
- **Medical Validation:** Obesity is a well-established major risk factor for Type 2 diabetes
- **Correlation:** 0.293 (strong correlation)
- **Clinical Relevance:** Excess adipose tissue promotes insulin resistance through inflammatory cytokines
- **Interpretation:** Higher BMI significantly increases diabetes risk

**3. Pregnancies (Coefficient: 0.37) - Third Most Important**
- **Medical Validation:** Gestational diabetes history is a recognized risk factor for future Type 2 diabetes
- **Correlation:** 0.222 (moderate correlation)
- **Clinical Relevance:** Pregnancy-induced insulin resistance can unmask underlying β-cell dysfunction
- **Interpretation:** More pregnancies associated with higher diabetes risk

**4. DiabetesPedigreeFunction (Coefficient: 0.29) - Fourth Most Important**
- **Medical Validation:** Family history is a well-documented non-modifiable risk factor
- **Correlation:** 0.174 (weak but significant)
- **Clinical Relevance:** Type 2 diabetes has strong genetic component (heritability ~30-70%)
- **Interpretation:** Stronger family history increases diabetes risk

### 4.3 ROC Curve Analysis

**Results:**
- **Logistic Regression AUC:** 0.8126
- **LDA AUC:** 0.8126
- **Random Classifier AUC:** 0.5000

**Interpretation:**
- Both models perform significantly better than random chance
- Identical AUC scores indicate comparable discriminative ability
- ROC curves show good separation between classes
- Models can effectively distinguish between diabetes and no-diabetes cases

### 4.4 Model Selection Decision

**Selected Model: Logistic Regression**

**Rationale:**
1. **Higher Recall (70.37% vs. 48.15%):** Critical for medical applications - catches more diabetes cases
2. **Better Overall Accuracy (73.38% vs. 70.13%):** More correct predictions overall
3. **Higher F1-Score (64.96% vs. 53.06%):** Better balance of precision and recall
4. **Interpretable Coefficients:** Provides clinical insights through feature importance
5. **Probabilistic Outputs:** Enables risk stratification (low/moderate/high risk)

**Trade-offs:**
- Slightly more false positives than LDA (25 vs. 18)
- But significantly fewer false negatives (16 vs. 28)
- In medical context, false negatives (missing diabetes) are more costly than false positives

---

## 5. Insights and Conclusions

### 5.1 Key Insights

#### 5.1.1 Data Quality Insights
- **Invalid Zero Values:** Medical datasets require careful preprocessing - 652 zeros represented missing data, not actual measurements
- **Median Imputation:** More appropriate than mean for skewed medical data
- **Feature Scaling:** Critical for algorithm performance - features had vastly different scales (Insulin: 0-846 vs. DiabetesPedigreeFunction: 0.08-2.42)

#### 5.1.2 Feature Importance Insights
- **Glucose Dominance:** Glucose is 1.67× more important than BMI (1.18 vs. 0.71), validating its role as primary diagnostic marker
- **Multi-Factor Risk:** While glucose is most important, BMI, pregnancies, and family history also significantly contribute
- **Clinical Alignment:** Model coefficients perfectly align with established medical knowledge, providing confidence in model validity

#### 5.1.3 Model Performance Insights
- **Classical ML Effectiveness:** Simple linear models achieve 81.3% AUC-ROC, demonstrating effectiveness of classical approaches
- **Recall vs. Precision Trade-off:** Logistic Regression prioritizes recall (catches more cases), while LDA prioritizes precision (fewer false positives)
- **Medical Context Matters:** For diabetes prediction, higher recall is preferred to minimize missed cases

#### 5.1.4 Clinical Application Insights
- **Early Detection Capability:** Model can identify risk before glucose reaches diagnostic threshold (130 mg/dL)
- **Risk Stratification:** Probability outputs enable categorization into low/moderate/high risk groups
- **Multi-Factor Assessment:** Considers 8 factors simultaneously, not just glucose levels
- **Preventive Potential:** Enables intervention before diabetes develops

### 5.2 Why This Model is Useful Beyond Glucose Levels

**Common Misconception:** "If glucose > 130 indicates diabetes, why do we need a model?"

**Answer:** The model provides value in multiple ways:

1. **Early Detection:**
   - Glucose ≥130 mg/dL = **Diabetes diagnosis** (after disease develops)
   - Model = **Risk prediction** (before disease develops)
   - Can identify risk when glucose is 95-125 mg/dL (prediabetes/normal range)

2. **Multi-Factor Risk Assessment:**
   - Example: Person with glucose 95 (normal) but BMI 35, age 50, family history → High risk
   - Glucose alone would miss this risk
   - Model combines 8 factors for comprehensive assessment

3. **Risk Stratification:**
   - Identifies who needs immediate screening vs. routine monitoring
   - Helps prioritize healthcare resources
   - Enables personalized prevention plans

4. **Preventive Intervention:**
   - Identifies at-risk individuals before diabetes develops
   - Enables lifestyle interventions (diet, exercise, weight management)
   - More effective when started early

5. **Complex Interactions:**
   - Captures interactions between factors (e.g., high BMI + family history + age)
   - Simple glucose threshold doesn't capture these interactions

6. **Population Health:**
   - Useful for large-scale screening where full lab work isn't feasible
   - Can use basic measurements (BMI, age, family history) to flag high-risk individuals
   - Cost-effective for resource-limited settings

### 5.3 Clinical Validation

**Model Alignment with Medical Knowledge:**
- ✅ Glucose identified as most important (validates primary diagnostic marker)
- ✅ BMI identified as second most important (validates obesity-diabetes link)
- ✅ Pregnancies identified as important (validates gestational diabetes risk)
- ✅ Family history identified as important (validates genetic predisposition)
- ✅ Coefficients align with established clinical risk factors

**This alignment provides confidence in:**
- Model's clinical relevance
- Predictive validity
- Potential for real-world application

### 5.4 Practical Applications

**Use Cases:**
1. **Primary Care Screening:** Identify high-risk patients for diabetes screening
2. **Preventive Care Programs:** Target lifestyle interventions to at-risk individuals
3. **Population Health:** Screen large populations cost-effectively
4. **Clinical Decision Support:** Assist healthcare providers in risk assessment
5. **Patient Education:** Help patients understand their risk factors

**Implementation Considerations:**
- Model should be used as decision support, not replacement for clinical judgment
- Requires validation on diverse populations before widespread use
- Should be integrated with clinical workflows
- Needs regular monitoring and retraining

### 5.5 Limitations

1. **Dataset Size:** 768 samples may limit generalization
2. **Population Specificity:** Pima Indian population may not generalize to other ethnic groups
3. **Feature Engineering:** Limited to original features; could benefit from interaction terms
4. **Model Complexity:** Linear models may miss non-linear relationships
5. **Temporal Aspects:** Cannot capture disease progression over time
6. **External Validation:** Tested only on internal test set

---

## 6. Future Scope

### 6.1 Model Improvements

1. **Advanced Algorithms:**
   - Random Forest for non-linear relationships
   - XGBoost for improved performance
   - Neural Networks for complex patterns

2. **Feature Engineering:**
   - Interaction features (Glucose × BMI, Age × Pregnancies)
   - Polynomial features for non-linear relationships
   - Domain-specific feature creation

3. **Hyperparameter Tuning:**
   - Grid search or Bayesian optimization
   - Cross-validation for robust evaluation
   - Ensemble methods

### 6.2 Data Improvements

1. **Larger Datasets:** Train on larger, more diverse populations
2. **External Validation:** Test on independent datasets
3. **Temporal Data:** Incorporate longitudinal studies
4. **Additional Features:** Include lifestyle factors, medication history

### 6.3 Clinical Integration

1. **User Interface:** Develop user-friendly web/mobile applications
2. **Clinical Workflows:** Integrate with electronic health records
3. **Risk Stratification Guidelines:** Develop evidence-based guidelines
4. **Performance Monitoring:** Implement continuous monitoring and retraining

### 6.4 Research Directions

1. **Explainable AI:** Implement SHAP values or LIME for enhanced interpretability
2. **Cost-Sensitive Learning:** Incorporate misclassification costs
3. **Multi-Class Classification:** Predict prediabetes, diabetes, and complications
4. **Personalized Medicine:** Develop patient-specific risk models

---

## 7. Technical Implementation

### 7.1 Technologies Used

- **Programming Language:** Python 3.x
- **Libraries:**
  - pandas, numpy (data manipulation)
  - scikit-learn (machine learning)
  - matplotlib, seaborn (visualization)
  - streamlit (web application)
- **Development Environment:** Jupyter Notebook, Google Colab

### 7.2 Project Deliverables

1. **Complete Jupyter Notebook:** End-to-end analysis with all code
2. **Streamlit Web Application:** Interactive diabetes risk assessment tool
3. **Model Evaluation Visualizations:** Confusion matrices, ROC curves
4. **Feature Importance Analysis:** Coefficient interpretation
5. **Comprehensive Documentation:** This document

### 7.3 Code Repository

All code, datasets (instructions), and documentation are available in the project repository, including:
- Data preprocessing scripts
- Model training scripts
- Evaluation modules
- Streamlit web application
- Complete Jupyter notebook

---

## 8. Conclusion

This project successfully demonstrates the application of classical machine learning approaches to diabetes risk prediction. The work emphasizes:

1. **Proper Data Preprocessing:** Critical for medical data quality
2. **Comprehensive Evaluation:** Multiple metrics provide thorough assessment
3. **Clinical Interpretation:** Model insights align with medical knowledge
4. **Practical Application:** Interactive web app enables real-world use

**Key Achievement:** Developed a model that achieves 81.3% AUC-ROC with interpretable insights, validated against clinical knowledge, and implemented as an interactive web application.

**Impact:** The model can assist healthcare providers in early diabetes risk assessment, enabling preventive intervention and supporting clinical decision-making.

---

## References

1. American Diabetes Association. (2023). Standards of Medical Care in Diabetes. *Diabetes Care*, 46(Supplement 1).

2. Smith, J. W., et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261-265.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.

5. James, G., et al. (2013). *An Introduction to Statistical Learning: with Applications in R*. Springer.

---

## Appendix

### A. Dataset Information
- **Name:** Pima Indians Diabetes Dataset
- **Source:** UCI Machine Learning Repository
- **License:** Public domain
- **URL:** https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes

### B. Model Performance Summary

**Logistic Regression:**
- Accuracy: 73.38%
- Precision: 60.32%
- Recall: 70.37%
- F1-Score: 64.96%
- ROC-AUC: 81.26%

**LDA:**
- Accuracy: 70.13%
- Precision: 59.09%
- Recall: 48.15%
- F1-Score: 53.06%
- ROC-AUC: 81.26%

### C. Feature Importance Ranking

1. Glucose (1.18)
2. BMI (0.71)
3. Pregnancies (0.37)
4. DiabetesPedigreeFunction (0.29)
5. Age (0.19)
6. Insulin (-0.04)
7. BloodPressure (-0.01)
8. SkinThickness (0.01)

---

**Project completed for IIT Guwahati Data Science Course**

*This documentation demonstrates the application of classical machine learning approaches to medical data analysis, emphasizing proper methodology, comprehensive evaluation, and clinical interpretation.*
