# run_pipeline_rf_with_submission.py
import pandas as pd
from tqdm import tqdm
import time
from sklearn.ensemble import RandomForestClassifier
from src.metrics import evaluate_model, plot_threshold_analysis
from src.data import load_data, merge_left
import os

def run():
    """
    Executes the end-to-end credit risk modeling pipeline using Random Forest:
    1. Loads Home Credit datasets.
    2. Aggregates bureau data per applicant.
    3. Merges aggregated bureau features with training and test data.
    4. Trains a Random Forest classifier on training data.
    5. Evaluates model metrics on train set (test metrics if labels exist).
    6. Saves test predictions to `data/output/submission.csv`.
    7. Generates threshold analysis plot for train.
    """
    
    # Load data
    bureau_df, raw_train_df, raw_test_df = load_data(data_dir='data/inputs/home-credit-default-risk')

    # Aggregate bureau data
    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        avg_credit_sum=("AMT_CREDIT_SUM", "mean"),
        total_credit_sum=("AMT_CREDIT_SUM", "sum"),
        total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        max_overdue=("CREDIT_DAY_OVERDUE", "max"),
        avg_credit_days=("DAYS_CREDIT", "mean")
    )

    # Merge train and test with bureau aggregates
    with tqdm(total=2, desc="Merging & Preprocessing") as pbar:
        train_combined = merge_left(raw_train_df, bureau_agg, on="SK_ID_CURR")
        test_combined = merge_left(raw_test_df, bureau_agg, on="SK_ID_CURR")
        X_train = train_combined.drop(columns=["TARGET", "SK_ID_CURR"]).select_dtypes(include=['number']).fillna(0)
        y_train = raw_train_df["TARGET"]
        X_test = test_combined.drop(columns=["SK_ID_CURR"]).select_dtypes(include=['number']).fillna(0)
        test_ids = raw_test_df["SK_ID_CURR"]
        pbar.update(2)

    # Train Random Forest
    print(f"\nStarting Random Forest Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    start_time = time.time()
    with tqdm(total=1, desc="Fitting Random Forest", bar_format="{desc}: {elapsed}") as pbar:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        pbar.update(1)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Predictions
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate train metrics & KS threshold
    train_metrics = evaluate_model(y_train, y_train_pred_proba)
    optimal_threshold = train_metrics['ks_threshold']
    train_metrics = evaluate_model(y_train, y_train_pred_proba, threshold=optimal_threshold)

    # Print train metrics
    print("\n--- Train Metrics ---")
    for metric, value in train_metrics.items():
        if metric == "confusion_matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")

    # Save test predictions
    submission_df = pd.DataFrame({
        "SK_ID_CURR": test_ids,
        "TARGET": y_test_pred_proba
    })
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"\nTest predictions saved to: {submission_path}")

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    print("\nTop Features by Importance:")
    print(feature_importance.head(10))

    # Threshold analysis plot for train
    plot_threshold_analysis(y_train.values, y_train_pred_proba, train_metrics)

if __name__ == "__main__":
    run()