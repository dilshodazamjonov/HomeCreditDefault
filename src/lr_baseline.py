# run_pipeline.py
import pandas as pd
from tqdm import tqdm
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.metrics import evaluate_model, plot_threshold_analysis
from src.data import load_data, merge_left

def run():
    """
    Executes the end-to-end credit risk modeling pipeline:
    1. Loads Home Credit datasets.
    2. Aggregates bureau data per applicant.
    3. Merges aggregated bureau features with training data.
    4. Scales numeric features using StandardScaler.
    5. Trains a Logistic Regression model.
    6. Evaluates model metrics including Gini, KS, AUC, Precision, Recall, F1.
    7. Computes the KS-optimal threshold and re-evaluates metrics at that threshold.
    8. Prints top features by absolute coefficient.
    9. Generates a threshold analysis plot showing Approval Rate, Bad Rate, and KS threshold.
    """
    
    bureau_df, raw_train_df, raw_test_df = load_data(data_dir='data/inputs/home-credit-default-risk')

    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        avg_credit_sum=("AMT_CREDIT_SUM", "mean"),
        total_credit_sum=("AMT_CREDIT_SUM", "sum"),
        total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        max_overdue=("CREDIT_DAY_OVERDUE", "max"),
        avg_credit_days=("DAYS_CREDIT", "mean")
    )

    with tqdm(total=1, desc="Merging & Preprocessing") as pbar:
        train_combined = merge_left(raw_train_df, bureau_agg, on="SK_ID_CURR")
        X_train = train_combined.drop(columns=["TARGET", "SK_ID_CURR"]).select_dtypes(include=['number']).fillna(0)
        y_train = raw_train_df["TARGET"]
        pbar.update(1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"\nStarting Model Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    start_time = time.time()
    
    with tqdm(total=1, desc="Fitting Logistic Regression", bar_format="{desc}: {elapsed}") as pbar:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        pbar.update(1)
        
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    with tqdm(total=1, desc="Calculating Metrics") as pbar:
        y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        train_metrics = evaluate_model(y_train, y_train_pred_proba)
        optimal_threshold = train_metrics['ks_threshold']
        train_metrics = evaluate_model(y_train, y_train_pred_proba, threshold=optimal_threshold)
        pbar.update(1)

    print("\n--- Train Metrics ---")
    for metric, value in train_metrics.items():
        if metric == "confusion_matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.4f}")

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_[0]
    }).sort_values(by="coefficient", key=abs, ascending=False)
    print("\nTop Features by Absolute Coefficient:")
    print(feature_importance.head(10))

    plot_threshold_analysis(y_train.values, y_train_pred_proba, train_metrics)

if __name__ == "__main__":
    run()