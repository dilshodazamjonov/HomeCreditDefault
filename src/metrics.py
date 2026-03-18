# metrics.py
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    roc_curve, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score
)
import numpy as np
import matplotlib.pyplot as plt

def ks_score(y_true, y_prob):
    """
    Calculates the Kolmogorov-Smirnov (KS) statistic for a binary classification model.
    
    KS measures the maximum separation between cumulative TPR and FPR.
    Calculates the KS statistic and the threshold that achieves it.
    Higher KS => better separation.
    
    Returns:
        tuple: (KS statistic, optimal threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    diffs = tpr - fpr

    ks = np.max(diffs)

    ks_idx = np.argmax(diffs)
    return ks, thresholds[ks_idx]

def gini_score(y_true, y_pred_proba):
    """
    Calculates the Gini coefficient based on ROC AUC.
    Gini = 2*AUC - 1
    """
    return 2 * roc_auc_score(y_true, y_pred_proba) - 1

def evaluate_model(y_true, y_pred_proba, threshold=0.5):# threshold needs to be tuned based KS-maximizing threshold
    """
    Evaluates a binary classification model across metrics and business KPIs.
    
    Returns:
        dict: {
            'gini', 'auc', 'ks', 'ks_threshold', 'confusion_matrix',
            'precision', 'recall', 'f1_score', 'accuracy',
            'approval_rate', 'bad_rate_approved'
        }
    """
    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # KS and threshold
    ks, ks_thresh = ks_score(y_true, y_pred_proba)
    
    # Business metrics
    approval_rate = float(np.mean(y_pred == 0))  # % approved
    approved_mask = (y_pred == 0)
    bad_rate_approved = float(np.mean(y_true[approved_mask])) if np.sum(approved_mask) > 0 else 0
    
    return {
        # ML metrics
        "gini": gini_score(y_true, y_pred_proba),
        "auc": roc_auc_score(y_true, y_pred_proba),
        "ks": ks,
        "ks_threshold": ks_thresh,
        
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        
        # Business metrics
        "approval_rate": approval_rate,
        "bad_rate_approved": bad_rate_approved
    }


# -----------------------------
# Threshold analysis plot
# -----------------------------
def plot_threshold_analysis(y_true, y_prob, train_metrics):
    thresholds = [0.01 * i for i in range(1, 100)]
    approval_rates = []
    bad_rates = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        approval_rates.append(float(np.mean(y_pred == 0)))
        approved_mask = (y_pred == 0)
        bad_rate = float(np.mean(y_true[approved_mask])) if np.sum(approved_mask) > 0 else 0
        bad_rates.append(bad_rate)

    plt.figure(figsize=(8,5))
    plt.plot(thresholds, approval_rates, label="Approval Rate")
    plt.plot(thresholds, bad_rates, label="Bad Rate among Approved")
    plt.axvline(x=train_metrics['ks_threshold'], color='red', linestyle='--', label='KS Threshold')
    plt.xlabel("Probability Threshold")
    plt.ylabel("Rate")
    plt.title("Threshold Analysis")
    plt.legend()
    plt.savefig("data/outputs/threshold_analysis.png")