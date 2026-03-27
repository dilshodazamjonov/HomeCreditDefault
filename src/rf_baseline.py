# run_pipeline_rf.py
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from src.metrics import evaluate_model, ks_score
from src.data import load_data, merge_left
from src.feature_selection import FeatureSelector


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
    X_full = train_combined.drop(columns=["TARGET", "SK_ID_CURR"])

    # ===================== FEATURE TYPES =====================
    cat_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_full.select_dtypes(include=["number"]).columns.tolist()

    # keep only low-cardinality categorical
    low_card_cat_cols = [c for c in cat_cols if X_full[c].nunique() <= 7]

    print(f"Categorical columns after filtering: {len(low_card_cat_cols)}")

    # ===================== PREPROCESS =====================
    # numeric
    X_num = X_full[num_cols].fillna(0)

    # categorical
    X_cat = X_full[low_card_cat_cols].fillna("Missing")

    print("One-hot encoding categorical features...")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = ohe.fit_transform(X_cat)

    X_cat_enc = pd.DataFrame(
        X_cat_enc,
        columns=ohe.get_feature_names_out(low_card_cat_cols),
        index=X_full.index
    )

    # final dataset
    X_final = pd.concat([X_num, X_cat_enc], axis=1)

    print(f"Final feature count before selection: {X_final.shape[1]}")

    # ===================== CV =====================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    print("Starting cross-validation...")
    for fold, (tr, va) in enumerate(tqdm(skf.split(X_final, y), total=5, desc="CV folds"), 1):

        print(f"\nTraining fold {fold}...")

        X_train, X_val = X_final.iloc[tr], X_final.iloc[va]
        y_train, y_val = y.iloc[tr], y.iloc[va]

        # ===================== FEATURE SELECTION =====================
        selector = FeatureSelector(k=20, method="mrmr", n_iter=5, random_state=42)
        selector.fit(X_train, y_train)

        X_train_sel = selector.transform(X_train)
        X_val_sel = selector.transform(X_val)

        selected_features = selector.selected_features_

        # ===================== MODEL =====================
        model = get_model()
        model.fit(X_train_sel, y_train)

        # threshold via KS
        y_tr_p = model.predict_proba(X_train_sel)[:, 1]
        _, thr = ks_score(y_train, y_tr_p)

        # validation
        y_val_p = model.predict_proba(X_val_sel)[:, 1]
        metrics = evaluate_model(y_val, y_val_p, threshold=thr)
        metrics["fold"] = fold

        cv_results.append(metrics)

    # save CV
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(f"{output_dir}/cv_results_rf.csv", index=False)

    summary = cv_df.mean(numeric_only=True)
    summary.to_csv(f"{output_dir}/cv_summary_rf.csv")

    print("Cross-validation complete.")

    # ===================== FINAL MODEL =====================
    print("Training final model on full data...")

    selector = FeatureSelector(k=30, method="mrmr", n_iter=5, random_state=42)
    selector.fit(X_final, y)

    X_final_sel = selector.transform(X_final)
    selected_features = selector.selected_features_

    model = get_model()
    model.fit(X_final_sel, y)

    # predictions
    y_full = model.predict_proba(X_final_sel)[:, 1]
    _, thr = ks_score(y, y_full)

    final_metrics = evaluate_model(y, y_full, threshold=thr)
    pd.DataFrame([final_metrics]).to_csv(
        f"{output_dir}/final_metrics_rf.csv", index=False
    )

    print("Final model trained.")

    # ===================== FEATURE IMPORTANCE =====================
    print("Saving feature importance...")

    fi = pd.DataFrame({
        "feature": selected_features,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    fi.to_csv(f"{output_dir}/feature_importance_rf.csv", index=False)

    print("RF results saved.")


if __name__ == "__main__":
    run()