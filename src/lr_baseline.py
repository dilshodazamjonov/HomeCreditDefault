import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.metrics import evaluate_model, ks_score
from src.data import load_data, merge_left
from src.feature_selection import FeatureSelector


def run():
    output_dir = "data/outputs/lr"
    os.makedirs(output_dir, exist_ok=True)

    # ===================== LOAD DATA =====================
    print("Loading data...")
    bureau_df, raw_train_df, raw_test_df = load_data(
        data_dir='data/inputs/home-credit-default-risk'
    )

    # ===================== FEATURE ENGINEERING =====================
    print("Aggregating bureau data...")
    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        avg_credit_sum=("AMT_CREDIT_SUM", "mean"),
        total_credit_sum=("AMT_CREDIT_SUM", "sum"),
        total_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        max_overdue=("CREDIT_DAY_OVERDUE", "max"),
        avg_credit_days=("DAYS_CREDIT", "mean")
    )

    print("Merging data...")
    train_combined = merge_left(raw_train_df, bureau_agg, on="SK_ID_CURR")

    y = train_combined["TARGET"]
    X = train_combined.drop(columns=["TARGET", "SK_ID_CURR"])

    # ===================== CATEGORICAL ENCODING =====================
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    low_card_cat_cols = [c for c in cat_cols if X[c].nunique() <= 7]

    print(f"Categorical cols before {len(cat_cols)} and after filtering: {len(low_card_cat_cols)}")

    X_cat = X[low_card_cat_cols].fillna("Missing")
    X_num = X[num_cols].fillna(0)

    print("Encoding categorical features...")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = ohe.fit_transform(X_cat)
    X_cat_enc = pd.DataFrame(X_cat_enc, columns=ohe.get_feature_names_out(low_card_cat_cols))

    X_final = pd.concat([X_num.reset_index(drop=True), X_cat_enc.reset_index(drop=True)], axis=1)

    # ===================== CROSS-VALIDATION =====================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    print("Starting cross-validation...")
    for fold, (tr, va) in enumerate(tqdm(skf.split(X_final, y), total=5, desc="CV folds"), 1):
        print(f"\nTraining fold {fold}...")
        X_train, X_val = X_final.iloc[tr], X_final.iloc[va]
        y_train, y_val = y.iloc[tr], y.iloc[va]

        # -------- MRMR feature selection ----------
        selector = FeatureSelector(k=30, method="mrmr", n_iter=5, random_state=42)
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_val = selector.transform(X_val)

        # -------- SCALING --------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # -------- TRAIN MODEL --------
        model = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
        model.fit(X_train_scaled, y_train)

        # -------- PREDICTIONS AND METRICS --------
        y_tr_p = model.predict_proba(X_train_scaled)[:, 1]
        _, thr = ks_score(y_train, y_tr_p)

        y_val_p = model.predict_proba(X_val_scaled)[:, 1]
        metrics = evaluate_model(y_val, y_val_p, threshold=thr)
        metrics["fold"] = fold

        cv_results.append(metrics)
        print(f"Fold {fold} metrics: {metrics}")

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(f"{output_dir}/cv_results_lr.csv", index=False)
    cv_df.mean(numeric_only=True).to_csv(f"{output_dir}/cv_summary_lr.csv")
    print("Cross-validation complete.")

    # ===================== FINAL MODEL =====================
    print("Training final model on full data...")
    selector = FeatureSelector(k=30, method="mrmr", n_iter=5, random_state=42)
    selector.fit(X_final, y)
    X_final_sel = selector.transform(X_final)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final_sel)

    model = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
    model.fit(X_scaled, y)

    y_full = model.predict_proba(X_scaled)[:, 1]
    _, thr = ks_score(y, y_full)

    final_metrics = evaluate_model(y, y_full, threshold=thr)
    pd.DataFrame([final_metrics]).to_csv(f"{output_dir}/final_metrics_lr.csv", index=False)
    print("Final model trained.")

    # ===================== FEATURE IMPORTANCE =====================
    print("Saving feature importance...")
    fi = pd.DataFrame({
        "feature": X_final_sel.columns,
        "coefficient": model.coef_[0]
    }).sort_values(by="coefficient", key=abs, ascending=False)
    fi.to_csv(f"{output_dir}/feature_importance_lr.csv", index=False)
    print("LR results saved.")


if __name__ == "__main__":
    run()