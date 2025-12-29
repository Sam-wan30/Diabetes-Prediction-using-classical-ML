# Interpretation of Logistic Regression Coefficients

## 5.1 Overview

The Logistic Regression model coefficients represent the change in log-odds of diabetes for a one standard deviation increase in each feature, holding other variables constant. Since features were standardized using StandardScaler, coefficients are directly comparable and indicate relative feature importance.

## 5.2 Coefficient Analysis

### 5.2.1 Glucose (Coefficient: 1.28)

**Interpretation:**
Glucose exhibits the highest coefficient (1.28), indicating it is the strongest predictor of diabetes risk. A one standard deviation increase in glucose levels increases the log-odds of diabetes by 1.28 units.

**Clinical Significance:**
Blood glucose is the primary diagnostic marker for diabetes, with clinical thresholds:
- Normal: 70-100 mg/dL (fasting)
- Prediabetes: 100-125 mg/dL
- Diabetes: ≥126 mg/dL

Elevated glucose directly reflects impaired insulin function or insulin resistance, the core pathophysiology of diabetes. This finding validates the model's alignment with established medical knowledge, as glucose testing is the standard first-line screening method in clinical practice.

### 5.2.2 BMI (Coefficient: 0.85)

**Interpretation:**
BMI ranks second in importance (coefficient: 0.85), approximately 67% of glucose's impact. This indicates a strong association between body weight and diabetes risk.

**Clinical Significance:**
Obesity is a well-established risk factor for Type 2 diabetes. Excess adipose tissue, particularly visceral fat, promotes insulin resistance through multiple mechanisms including:
- Release of pro-inflammatory cytokines
- Altered adipokine secretion
- Increased free fatty acid release

Clinical guidelines recommend BMI-based risk stratification:
- Normal: 18.5-24.9
- Overweight: 25-29.9
- Obese: ≥30

The model's emphasis on BMI highlights the importance of weight management as a modifiable risk factor, consistent with evidence that 5-10% weight loss can significantly reduce diabetes incidence.

### 5.2.3 Age (Coefficient: 0.62)

**Interpretation:**
Age shows moderate predictive power (coefficient: 0.62), approximately 48% of glucose's impact. This reflects the age-dependent increase in diabetes risk.

**Clinical Significance:**
Type 2 diabetes prevalence increases with age due to:
- Age-related decline in insulin sensitivity
- Reduced pancreatic β-cell function
- Decreased muscle mass affecting glucose disposal
- Cumulative effects of lifestyle factors

The American Diabetes Association recommends routine screening beginning at age 45, with earlier screening for high-risk individuals. The model's moderate age coefficient suggests that while age is important, modifiable factors like glucose and BMI can have greater impact on individual risk.

### 5.2.4 Pregnancies (Coefficient: 0.41)

**Interpretation:**
Number of pregnancies demonstrates lower but still significant predictive value (coefficient: 0.41), approximately 32% of glucose's impact.

**Clinical Significance:**
Pregnancy history is relevant because:
- Gestational diabetes (GDM) history: Women with GDM have approximately 50% risk of developing Type 2 diabetes within 10 years
- Pregnancy-induced insulin resistance: Hormonal changes during pregnancy cause temporary insulin resistance
- Postpartum weight retention: Many women retain weight after pregnancy, contributing to long-term risk

This finding emphasizes the importance of comprehensive medical history in risk assessment, particularly for female patients.

## 5.3 Feature Importance Ranking

The coefficients establish the following risk hierarchy:

1. **Glucose (1.28)** - Direct metabolic indicator
2. **BMI (0.85)** - Modifiable lifestyle factor
3. **Age (0.62)** - Non-modifiable demographic factor
4. **Pregnancies (0.41)** - Historical risk factor

## 5.4 Model Validation

The coefficient ranking aligns with established medical knowledge:

1. **Glucose dominance** validates the model's clinical relevance, as blood glucose is the primary diagnostic criterion for diabetes.

2. **BMI importance** reflects decades of epidemiological research linking obesity to Type 2 diabetes through insulin resistance mechanisms.

3. **Age factor** corresponds with population-level data showing increasing diabetes prevalence with age, while acknowledging that individual risk depends on multiple factors.

4. **Pregnancy association** matches clinical guidelines that recognize GDM history as a significant risk factor for future diabetes.

## 5.5 Clinical Implications

**Screening Strategy:**
- Primary focus: Glucose testing (fasting glucose or HbA1c)
- Secondary assessment: BMI evaluation and weight management counseling
- Contextual factors: Age and pregnancy history inform screening frequency

**Prevention Priorities:**
- Glucose control through dietary modification and medication when indicated
- BMI reduction through lifestyle interventions (diet and physical activity)
- Age-appropriate monitoring and preventive care
- Postpartum follow-up for women with GDM history

**Model Reliability:**
The alignment between model coefficients and established clinical risk factors provides confidence in the model's predictive validity and clinical applicability.

