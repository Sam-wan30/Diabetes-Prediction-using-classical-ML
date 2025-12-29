"""
Comprehensive Classification Model Evaluation
============================================
This script provides comprehensive evaluation of classification models using:
1. Confusion Matrix
2. Precision
3. Recall (Sensitivity)
4. F1-Score
5. ROC-AUC (Area Under the ROC Curve)

UNDERSTANDING CLASSIFICATION METRICS:
------------------------------------

1. CONFUSION MATRIX:
   -----------------
   A table showing the four possible outcomes of predictions:
   
   ┌─────────────┬──────────────┬──────────────┐
   │             │ Predicted    │ Predicted    │
   │             │ Negative (0) │ Positive (1) │
   ├─────────────┼──────────────┼──────────────┤
   │ Actual Neg  │ True Negative│ False Positive│
   │             │     (TN)     │     (FP)      │
   ├─────────────┼──────────────┼──────────────┤
   │ Actual Pos  │ False Negative│ True Positive│
   │             │     (FN)     │     (TP)      │
   └─────────────┴──────────────┴──────────────┘
   
   - TN: Correctly predicted negative (no diabetes)
   - FP: Incorrectly predicted positive (Type I error - false alarm)
   - FN: Missed positive case (Type II error - missed diagnosis)
   - TP: Correctly predicted positive (diabetes detected)

2. PRECISION:
   ----------
   Precision = TP / (TP + FP)
   
   - Measures: Of all positive predictions, how many were correct?
   - Interpretation: "When the model says diabetes, how often is it right?"
   - High precision = Few false positives
   - Important when: False positives are costly (e.g., unnecessary treatment)
   - Range: 0 to 1 (higher is better)

3. RECALL (Sensitivity):
   ---------------------
   Recall = TP / (TP + FN)
   
   - Measures: Of all actual positives, how many did we catch?
   - Interpretation: "Of all diabetes cases, how many did we detect?"
   - High recall = Few false negatives (missed cases)
   - Important when: Missing a case is costly (e.g., undiagnosed diabetes)
   - Range: 0 to 1 (higher is better)
   - Also called: Sensitivity, True Positive Rate (TPR)

4. F1-SCORE:
   ----------
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   
   - Harmonic mean of precision and recall
   - Balances both metrics
   - Useful when: You need a single metric balancing precision and recall
   - Best when: Both false positives and false negatives matter
   - Range: 0 to 1 (higher is better)
   - F1 = 1 when precision = recall = 1

5. ROC-AUC (Area Under ROC Curve):
   --------------------------------
   - ROC Curve: Plots True Positive Rate (TPR) vs False Positive Rate (FPR)
   - AUC: Area under the ROC curve
   - Measures: Model's ability to distinguish between classes
   - Interpretation:
     * AUC = 1.0: Perfect classifier
     * AUC = 0.5: Random classifier (no better than guessing)
     * AUC > 0.7: Good classifier
     * AUC > 0.8: Excellent classifier
   - Advantages:
     * Works well with imbalanced datasets
     * Threshold-independent (considers all thresholds)
     * Measures ranking quality, not just classification

TRADE-OFFS:
----------
- Precision vs Recall: Often inversely related
  * High precision → Lower recall (conservative model)
  * High recall → Lower precision (aggressive model)
- Choose based on use case:
  * Medical diagnosis: Prioritize recall (don't miss cases)
  * Spam detection: Prioritize precision (don't block legitimate emails)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, accuracy_score,
    classification_report, precision_recall_curve, auc
)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def evaluate_classification_model(y_true, y_pred, y_pred_proba, model_name="Model", 
                                   class_names=None, save_plots=True):
    """
    Comprehensive evaluation of a classification model.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    y_pred_proba : array-like
        Predicted probabilities for positive class (class 1)
    model_name : str
        Name of the model (for display purposes)
    class_names : list
        Names of classes (e.g., ['No Diabetes', 'Diabetes'])
    save_plots : bool
        Whether to save visualization plots
    """
    
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    print("=" * 80)
    print(f"COMPREHENSIVE MODEL EVALUATION: {model_name}")
    print("=" * 80)
    print()
    
    # ========================================================================
    # METRIC 1: CONFUSION MATRIX
    # ========================================================================
    
    print("METRIC 1: CONFUSION MATRIX")
    print("-" * 80)
    
    # Calculate confusion matrix
    # Returns: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print("=" * 50)
    print(f"{'':20s} {'Predicted':^30s}")
    print(f"{'':20s} {class_names[0]:^15s} {class_names[1]:^15s}")
    print("-" * 50)
    print(f"{'Actual ' + class_names[0]:20s} {tn:^15d} {fp:^15d}")
    print(f"{'Actual ' + class_names[1]:20s} {fn:^15d} {tp:^15d}")
    print()
    
    # Detailed interpretation
    print("Interpretation:")
    print("-" * 50)
    print(f"True Negatives (TN):  {tn:4d} - Correctly predicted {class_names[0]}")
    print(f"False Positives (FP): {fp:4d} - Incorrectly predicted {class_names[1]} (Type I Error)")
    print(f"False Negatives (FN): {fn:4d} - Missed {class_names[1]} cases (Type II Error)")
    print(f"True Positives (TP):  {tp:4d} - Correctly predicted {class_names[1]}")
    print()
    
    # Calculate percentages
    total = tn + fp + fn + tp
    print("Percentages:")
    print("-" * 50)
    print(f"True Negatives:  {tn/total*100:5.2f}%")
    print(f"False Positives: {fp/total*100:5.2f}%")
    print(f"False Negatives: {fn/total*100:5.2f}%")
    print(f"True Positives:  {tp/total*100:5.2f}%")
    print()
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix plot")
    plt.close()
    
    # ========================================================================
    # METRIC 2: PRECISION
    # ========================================================================
    
    print("METRIC 2: PRECISION")
    print("-" * 80)
    
    # Calculate precision
    # precision_score calculates: TP / (TP + FP)
    precision = precision_score(y_true, y_pred)
    
    print(f"\nPrecision = TP / (TP + FP)")
    print(f"          = {tp} / ({tp} + {fp})")
    print(f"          = {tp} / {tp + fp}")
    print(f"          = {precision:.4f} ({precision*100:.2f}%)")
    print()
    print("Interpretation:")
    print(f"  When the model predicts '{class_names[1]}', it is correct {precision*100:.2f}% of the time.")
    print(f"  {fp} out of {tp + fp} positive predictions were incorrect (false positives).")
    print()
    
    # ========================================================================
    # METRIC 3: RECALL (SENSITIVITY)
    # ========================================================================
    
    print("METRIC 3: RECALL (SENSITIVITY)")
    print("-" * 80)
    
    # Calculate recall
    # recall_score calculates: TP / (TP + FN)
    recall = recall_score(y_true, y_pred)
    
    print(f"\nRecall = TP / (TP + FN)")
    print(f"      = {tp} / ({tp} + {fn})")
    print(f"      = {tp} / {tp + fn}")
    print(f"      = {recall:.4f} ({recall*100:.2f}%)")
    print()
    print("Interpretation:")
    print(f"  The model correctly identifies {recall*100:.2f}% of all actual '{class_names[1]}' cases.")
    print(f"  {fn} out of {tp + fn} actual positive cases were missed (false negatives).")
    print()
    
    # ========================================================================
    # METRIC 4: F1-SCORE
    # ========================================================================
    
    print("METRIC 4: F1-SCORE")
    print("-" * 80)
    
    # Calculate F1-score
    # f1_score calculates: 2 × (Precision × Recall) / (Precision + Recall)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\nF1-Score = 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"        = 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
    print(f"        = 2 × {precision * recall:.4f} / {precision + recall:.4f}")
    print(f"        = {f1:.4f} ({f1*100:.2f}%)")
    print()
    print("Interpretation:")
    print(f"  F1-score balances precision and recall: {f1*100:.2f}%")
    if f1 > 0.7:
        print("  ✓ Good balance between precision and recall")
    elif f1 > 0.5:
        print("  ⚠ Moderate balance - consider optimizing for specific metric")
    else:
        print("  ✗ Poor balance - model needs improvement")
    print()
    
    # ========================================================================
    # METRIC 5: ROC-AUC
    # ========================================================================
    
    print("METRIC 5: ROC-AUC (Area Under ROC Curve)")
    print("-" * 80)
    
    # Calculate ROC-AUC
    # roc_auc_score calculates area under ROC curve
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    print(f"\nROC-AUC Score: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print()
    print("Interpretation:")
    if roc_auc >= 0.9:
        print("  ⭐⭐⭐ Excellent classifier (AUC ≥ 0.9)")
    elif roc_auc >= 0.8:
        print("  ⭐⭐ Very good classifier (AUC ≥ 0.8)")
    elif roc_auc >= 0.7:
        print("  ⭐ Good classifier (AUC ≥ 0.7)")
    elif roc_auc >= 0.6:
        print("  ⚠ Fair classifier (AUC ≥ 0.6)")
    else:
        print("  ✗ Poor classifier (AUC < 0.6)")
    print()
    print(f"  The model can distinguish between classes with {roc_auc*100:.1f}% accuracy.")
    print(f"  Random guessing would achieve 50% (AUC = 0.5).")
    print()
    
    # Visualize ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_roc_curve.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve plot")
    plt.close()
    
    # ========================================================================
    # ADDITIONAL METRICS
    # ========================================================================
    
    print("ADDITIONAL METRICS")
    print("-" * 80)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Formula: (TP + TN) / Total = ({tp} + {tn}) / {total}")
    print()
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  Formula: TN / (TN + FP) = {tn} / ({tn} + {fp})")
    print(f"  Interpretation: Correctly identifies {specificity*100:.2f}% of negative cases")
    print()
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()
    
    summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity'],
        'Value': [accuracy, precision, recall, f1, roc_auc, specificity],
        'Percentage': [f'{accuracy*100:.2f}%', f'{precision*100:.2f}%', 
                      f'{recall*100:.2f}%', f'{f1*100:.2f}%', 
                      f'{roc_auc*100:.2f}%', f'{specificity*100:.2f}%']
    })
    
    print(summary.to_string(index=False))
    print()
    
    # Return all metrics as dictionary
    metrics_dict = {
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }
    
    return metrics_dict


def compare_models(y_true, y_pred_dict, y_pred_proba_dict, model_names, 
                   class_names=None, save_plots=True):
    """
    Compare multiple classification models side-by-side.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred_dict : dict
        Dictionary of {model_name: y_pred} for each model
    y_pred_proba_dict : dict
        Dictionary of {model_name: y_pred_proba} for each model
    model_names : list
        List of model names
    class_names : list
        Names of classes
    save_plots : bool
        Whether to save plots
    """
    
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print()
    
    # Evaluate each model
    all_metrics = {}
    for model_name in model_names:
        metrics = evaluate_classification_model(
            y_true, y_pred_dict[model_name], y_pred_proba_dict[model_name],
            model_name, class_names, save_plots
        )
        all_metrics[model_name] = metrics
        print()
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name in model_names:
        m = all_metrics[model_name]
        comparison_data.append({
            'Model': model_name,
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1-Score': m['f1_score'],
            'ROC-AUC': m['roc_auc'],
            'Specificity': m['specificity']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    print()
    print(comparison_df.to_string(index=False))
    print()
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = comparison_df[metric].values
        bars = ax.bar(range(len(model_names)), values, color=sns.color_palette("husl", len(model_names)))
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Comparison - All Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_plots:
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved model comparison plot")
    plt.close()
    
    # ROC curves comparison
    plt.figure(figsize=(10, 8))
    for model_name in model_names:
        m = all_metrics[model_name]
        plt.plot(m['fpr'], m['tpr'], lw=2, 
                label=f'{model_name} (AUC = {m["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved ROC curves comparison plot")
    plt.close()
    
    return all_metrics, comparison_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Evaluate models from model_training.py
    This demonstrates how to use the evaluation functions.
    """
    
    print("=" * 80)
    print("CLASSIFICATION MODEL EVALUATION - EXAMPLE USAGE")
    print("=" * 80)
    print()
    print("This script provides comprehensive evaluation functions.")
    print("To use with your models, import and call the functions:")
    print()
    print("  from model_evaluation import evaluate_classification_model, compare_models")
    print()
    print("  # For a single model:")
    print("  metrics = evaluate_classification_model(y_true, y_pred, y_pred_proba, 'Model Name')")
    print()
    print("  # For multiple models:")
    print("  all_metrics, comparison = compare_models(")
    print("      y_true, y_pred_dict, y_pred_proba_dict, model_names")
    print("  )")
    print()
    print("=" * 80)

