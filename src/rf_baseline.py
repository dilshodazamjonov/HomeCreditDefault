# run_pipeline_rf.py
import pandas as pd
import os

from scipy.sparse import hstack, csr_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from src.metrics import evaluate_model, ks_score
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

    output_dir = "data/outputs/rf"
    os.makedirs(output_dir, exist_ok=True)

    bureau_df, raw_train_df, raw_test_df = load_data(
        data_dir='data/inputs/home-credit-default-risk'
    )

    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        avg_credit_sum=("AMT_CREDIT_SUM", "mean"),
        total_credit_sum=("AMT_CREDIT_SUM", "sum"),
        total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        max_overdue=("CREDIT_DAY_OVERDUE", "max"),
        avg_credit_days=("DAYS_CREDIT", "mean")
    )

    train_combined = merge_left(raw_train_df, bureau_agg, on="SK_ID_CURR")

    y = train_combined["TARGET"]
    X_full = train_combined.drop(columns=["TARGET", "SK_ID_CURR"])

    cat_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_full.select_dtypes(include=["number"]).columns.tolist()

    low_card_cat_cols = [c for c in cat_cols if X_full[c].nunique() <= 7]

    print(f"Categorical cols after filtering: {len(low_card_cat_cols)}")

    X_num = csr_matrix(X_full[num_cols].fillna(0).values)
    X_cat = X_full[low_card_cat_cols].fillna("Missing")

    ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    X_cat_sparse = ohe.fit_transform(X_cat)

    X = hstack([X_num, X_cat_sparse])

    # ===================== CV =====================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):

        X_train, X_val = X[tr], X[va]
        y_train, y_val = y.iloc[tr], y.iloc[va]

        model = get_model()
        model.fit(X_train, y_train)

        y_tr_p = model.predict_proba(X_train)[:, 1]
        _, thr = ks_score(y_train, y_tr_p)

        y_val_p = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model(y_val, y_val_p, threshold=thr)
        metrics["fold"] = fold

        cv_results.append(metrics)
        

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(f"{output_dir}/cv_results_rf.csv", index=False)

    summary = cv_df.mean(numeric_only=True)
    summary.to_csv(f"{output_dir}/cv_summary_rf.csv")

    # ===================== FINAL =====================
    model = get_model()
    model.fit(X, y)

    y_full = model.predict_proba(X)[:, 1]
    _, thr = ks_score(y, y_full)

    final_metrics = evaluate_model(y, y_full, threshold=thr)
    pd.DataFrame([final_metrics]).to_csv(f"{output_dir}/final_metrics_rf.csv", index=False)

    # ===================== FEATURE IMPORTANCE =====================
    feature_names = num_cols + list(ohe.get_feature_names_out(low_card_cat_cols))

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    fi.to_csv(f"{output_dir}/feature_importance_rf.csv", index=False)

    print("RF results saved.")


if __name__ == "__main__":
    run()