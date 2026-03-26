# run_pipeline_lr.py
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold

from src.metrics import evaluate_model, ks_score
from src.data import load_data, merge_left


def run():

    output_dir = "data/outputs/lr"
    os.makedirs(output_dir, exist_ok=True)

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

    train_combined = merge_left(raw_train_df, bureau_agg, on="SK_ID_CURR")

    y = train_combined["TARGET"]
    X = train_combined.drop(columns=["TARGET", "SK_ID_CURR"])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    low_card_cat_cols = [c for c in cat_cols if X[c].nunique() <= 7]

    print(f"Categorical cols before {cat_cols} and after filtering: {len(low_card_cat_cols)}")

    X_cat = X[low_card_cat_cols].fillna("Missing")
    X_num = X[num_cols].fillna(0)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = ohe.fit_transform(X_cat)

    X_final = pd.concat([
        X_num.reset_index(drop=True),
        pd.DataFrame(X_cat_enc, columns=ohe.get_feature_names_out(low_card_cat_cols))
    ], axis=1)

    # ===================== CV =====================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []

    for fold, (tr, va) in enumerate(skf.split(X_final, y), 1):

        X_train, X_val = X_final.iloc[tr], X_final.iloc[va]
        y_train, y_val = y.iloc[tr], y.iloc[va]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
        model.fit(X_train, y_train)

        y_tr_p = model.predict_proba(X_train)[:, 1]
        _, thr = ks_score(y_train, y_tr_p)

        y_val_p = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_model(y_val, y_val_p, threshold=thr)
        metrics["fold"] = fold

        cv_results.append(metrics)
        print(f"Fold {fold}: {metrics}")

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(f"{output_dir}/cv_results_lr.csv", index=False)

    summary = cv_df.mean(numeric_only=True)
    summary.to_csv(f"{output_dir}/cv_summary_lr.csv")

    # ===================== FINAL =====================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    model = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
    model.fit(X_scaled, y)

    y_full = model.predict_proba(X_scaled)[:, 1]
    _, thr = ks_score(y, y_full)

    final_metrics = evaluate_model(y, y_full, threshold=thr)

    pd.DataFrame([final_metrics]).to_csv(f"{output_dir}/final_metrics_lr.csv", index=False)

    # ===================== FEATURE IMPORTANCE =====================
    fi = pd.DataFrame({
        "feature": X_final.columns,
        "coefficient": model.coef_[0]
    }).sort_values(by="coefficient", key=abs, ascending=False)

    fi.to_csv(f"{output_dir}/feature_importance_lr.csv", index=False)

    print("LR results saved.")


if __name__ == "__main__":
    run()