# metrics.py
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    roc_curve,
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
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

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


def precision_score(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    if tp + fn == 0:
        return 0
    
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """
    Calculates the F1 score based on precision and recall.
    F1 = 2 * (precision * recall) / (precision + recall)
    Handles zero division by returning 0 if precision + recall is 0.    
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    
    if p + r == 0:
        return 0
    
    return 2 * (p * r) / (p + r)


def evaluate_model(y_true, y_pred_proba, threshold=None):
    """
    Evaluates a binary classification model across metrics and business KPIs.
    
    If threshold is not provided, it defaults to KS-based threshold (for analysis only).
    In production, threshold must be precomputed on train data and passed explicitly.
    
    Returns:
        dict: {
            'gini', 'auc', 'ks', 'ks_threshold', 'confusion_matrix',
            'precision', 'recall', 'f1_score', 'accuracy',
            'approval_rate', 'bad_rate_approved'
        }
    """

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    # KS (always calculated for monitoring)
    ks, ks_thresh = ks_score(y_true, y_pred_proba)

    # NOTE:
    # If threshold is None → fallback to KS threshold (NOT production-safe)
    # Proper usage: pass threshold computed on train data
    if threshold is None:
        threshold = ks_thresh

    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

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
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),

        # Business metrics
        "approval_rate": approval_rate,
        "bad_rate_approved": bad_rate_approved
    }

# -----------------------------
# Threshold analysis plot
# -----------------------------
def plot_threshold_analysis(y_true, y_prob, train_metrics):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    thresholds = np.linspace(0, 1, 100)
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


    