# run_pipeline_rf.py
import pandas as pd
import time
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.metrics import evaluate_model, ks_score, plot_threshold_analysis
from src.data import load_data, merge_left


def get_model():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=150,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )


def run():
    """
    End-to-end Random Forest pipeline WITH Cross-Validation:
    - Data loading
    - Feature aggregation
    - CV training
    - Proper threshold handling (NO leakage)
    - Metrics averaging
    - Final model training
    - Submission creation
    """

    # ===================== LOAD DATA =====================
    bureau_df, raw_train_df, raw_test_df = load_data(
        data_dir='data/inputs/home-credit-default-risk'
    )

    # ===================== FEATURE ENGINEERING =====================
    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        avg_credit_sum=("AMT_CREDIT_SUM", "mean"),
        total_credit_sum=("AMT_CREDIT_SUM", "sum"),
        total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        max_overdue=("CREDIT_DAY_OVERDUE", "max"),
        avg_credit_days=("DAYS_CREDIT", "mean")
    )

    print("\nMerging & preprocessing...")
    train_combined = merge_left(raw_train_df, bureau_agg, on="SK_ID_CURR")
    test_combined = merge_left(raw_test_df, bureau_agg, on="SK_ID_CURR")

    X = train_combined.drop(columns=["TARGET", "SK_ID_CURR"]) \
        .select_dtypes(include=['number']) \
        .fillna(0)

    y = raw_train_df["TARGET"]

    X_test = test_combined.drop(columns=["SK_ID_CURR"]) \
        .select_dtypes(include=['number']) \
        .fillna(0)

    test_ids = raw_test_df["SK_ID_CURR"]

    print(f"Dataset shape: {X.shape}")

    # ===================== CROSS VALIDATION =====================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []

    print("\nStarting Cross-Validation...\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = get_model()

        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"Training time: {time.time() - start_time:.2f}s")

        # ===================== THRESHOLD (TRAIN ONLY) =====================
        y_train_proba = model.predict_proba(X_train)[:, 1]
        _, threshold = ks_score(y_train, y_train_proba)

        # ===================== VALIDATION =====================
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = evaluate_model(y_val, y_val_pred_proba, threshold=threshold)
        cv_results.append(metrics)

        print(f"AUC: {metrics['auc']:.4f}, Gini: {metrics['gini']:.4f}, KS: {metrics['ks']:.4f}")

    # ===================== CV SUMMARY =====================
    print("\n===== CROSS-VALIDATION RESULTS =====")

    cv_df = pd.DataFrame(cv_results)

    for col in ["auc", "gini", "ks", "precision", "recall", "f1"]:
        print(f"{col.upper()} mean: {cv_df[col].mean():.4f}")

    # ===================== FINAL MODEL =====================
    print("\nTraining final model on full dataset...")

    model = get_model()
    model.fit(X, y)

    y_full_proba = model.predict_proba(X)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # threshold from FULL TRAIN (acceptable for demo)
    _, final_threshold = ks_score(y, y_full_proba)

    final_metrics = evaluate_model(y, y_full_proba, threshold=final_threshold)

    print("\n===== FINAL MODEL METRICS =====")
    for metric, value in final_metrics.items():
        if metric == "confusion_matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")

    # ===================== FEATURE IMPORTANCE =====================
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop Features:")
    print(feature_importance.head(10))

    # ===================== SAVE SUBMISSION =====================
    submission_df = pd.DataFrame({
        "SK_ID_CURR": test_ids,
        "TARGET": y_test_pred_proba
    })

    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    submission_path = os.path.join(output_dir, "submission_rf.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to: {submission_path}")

    # ===================== PLOT =====================
    plot_threshold_analysis(y.values, y_full_proba, final_metrics)


if __name__ == "__main__":
    run()