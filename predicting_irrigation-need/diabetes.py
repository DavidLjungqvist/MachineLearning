# next: submission based on a meta model (use all training data!)

from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
import numpy as np
import os
import optuna
import pandas as pd
import seaborn as sns
import sys
import time
import winsound

# from sklearn.model_selection import
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split  # , cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
import lightgbm
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Options as global constants
MODEL = "lgbm"  # use model lgbm for fitting
# MODEL = "xgb"                            # use model xgb for fitting
SOUND = True  # beep after each validation, beep extra after all validations
MODEL_STACKING = False  # Stack LGBM, XGB and catbbost models into a meta LGBM model
OPTUNA = False  # Use Optuna to find optimal model parameters while validating
INCLUDE_ORIGINAL = False  # use the original database as additional training data?
VALIDATE_MODELS = False  # compare a lot of models with different model parameters?
SUBMIT_PREDICTION = not VALIDATE_MODELS  # create a submission?
VERBOSE = False  # print a lot of variables for debugging purposes?


def suppress_python_output():
    """
    Redirect Python-level stdout/stderr to os.devnull.
    Idempotent: repeated calls do nothing.
    """
    if getattr(suppress_python_output, "_active", False):
        return
    suppress_python_output._old_stdout = sys.stdout
    suppress_python_output._old_stderr = sys.stderr
    suppress_python_output._devnull = open(os.devnull, "w")
    sys.stdout = suppress_python_output._devnull
    sys.stderr = suppress_python_output._devnull
    suppress_python_output._active = True


def restore_python_output():
    """
    Restore Python-level stdout/stderr previously redirected by suppress_python_output.
    Safe to call even if not active.
    """
    if not getattr(suppress_python_output, "_active", False):
        return
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    # close devnull and restore originals
    try:
        suppress_python_output._devnull.close()
    except Exception:
        pass
    sys.stdout = suppress_python_output._old_stdout
    sys.stderr = suppress_python_output._old_stderr
    suppress_python_output._active = False


def read_data(train="train.csv", test="test.csv", original="diabetes dataset.csv"):
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)
    df_original = pd.read_csv(original)
    return df_train, df_test, df_original


def add_original_data(df_train, df_original, originals=1):
    # concat df_train and df_original to df_train
    # ignore columns that exist in df_original only
    # fill the id column for rows from df_original with negative values up to -1
    # add the same original data <originals> times
    for _ in range(originals):
        if INCLUDE_ORIGINAL:
            df_orig_aligned = df_original.reindex(columns=df_train.columns)  # drop columns that exist only in df_original
            start = -len(df_original)
            ids = np.arange(start, 0)
            try:  # try to cast ids to the same dtype as df_train["id"], fallback to object
                ids = ids.astype(df_train["id"].dtype)
            except Exception:
                ids = ids.astype(object)

            df_orig_aligned.loc[:, "id"] = ids
            df_train = pd.concat([df_train, df_orig_aligned], ignore_index=True, sort=False)  # concat keeping df_train's column set
    return df_train


def report_missing_values(df_train):
    missing_values = df_train.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Missing Values")
        plt.title("Missing Values per Feature")
        plt.tight_layout()
        plt.show()
    else:
        print("✅ No missing values found in the dataset.")


def print_categories(df, cols):
    # print all existing categories and their percentages.
    for col in cols:
        print(f"\nColumn: `{col}`")
        number_of = df[col].value_counts(dropna=False, normalize=True) * 100
        for name, percentage in number_of.items():
            print(f"  {name:13}: {percentage:6.2f}%")


def map_column_values(col, mapping, dataframes):
    # map categories, e.g. male->0, female->1
    # Case-insensitive mapping that preserves missing values
    map_lower = {k.lower(): v for k, v in mapping.items()}

    def map_series_ci(series, mapping_lower):
        mask_na = series.isna()
        s = series.astype(str).str.strip().str.lower()
        s = s.where(~mask_na, pd.NA)  # keep original missing as <NA>
        return s.map(mapping_lower).astype("Int64")

    for df in dataframes:
        df[col] = map_series_ci(df[col], map_lower)
        values = df[col].value_counts(dropna=False)
        if VERBOSE:
            print(values)
    return dataframes


def map_diabetes_features(dataframes):
    mapping = {"Female": 0, "Other": 1, "Male": 2}
    map_column_values("gender", mapping, dataframes)

    mapping = {"No formal": 0, "Highschool": 1, "Graduate": 2, "Postgraduate": 3}
    map_column_values("education_level", mapping, dataframes)

    mapping = {"Low": 0, "Lower-Middle": 1, "Middle": 2, "Upper-Middle": 3, "High": 4}
    map_column_values("income_level", mapping, dataframes)

    mapping = {"Unemployed": 0, "Retired": 1, "Employed": 2, "Student": 3}
    map_column_values("employment_status", mapping, dataframes)

    mapping = {"Never": 0, "Former": 1, "Current": 2}
    map_column_values("smoking_status", mapping, dataframes)

    return dataframes


def add_derived_diabetes_features(dataframes):
    for df in dataframes:
        # df["TG_HDL_Ratio"] = df["triglycerides"] / df["hdl_cholesterol"]
        # df["LDL_HDL_Ratio"] = df["ldl_cholesterol"] / df["hdl_cholesterol"]
        # df["non_HDL"] = df["cholesterol_total"] - df["hdl_cholesterol"]
        # df["total_div_HDL"] = df["cholesterol_total"] / df["hdl_cholesterol"]
        # df["triglycerides_div_HDL"] = df["triglycerides"] / df["hdl_cholesterol"]
        # df["MAP"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
        # df["Pulse_Pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        # df["Pulse_Pressure_Ratio"] = (df["systolic_bp"] - df["diastolic_bp"]) / df["systolic_bp"]
        # df["Central_Obesity_Index"] = df["bmi"] * df["waist_to_hip_ratio"]
        # df["Sleep_Deviation_squared"] = (df["sleep_hours_per_day"] - 7.5) * (df["sleep_hours_per_day"] - 7.5)

        # df["log_triglycerides"] = np.log1p(df["triglycerides"])
        # df["log_bmi"] = np.log1p(df["bmi"])
        # df["BMI_Waist_Interaction"] = df["bmi"] * df["waist_to_hip_ratio"]

        # case-insensitive female mask (works if gender is string or already encoded)
        # gender_female = df["gender"].astype(str).str.strip().str.lower() == "female"

        # Lipid Accumulation Product (gender-specific constants)
        # df["Lipid_Accumulation_Product"] = ((df["waist_to_hip_ratio"] * 100) - np.where(gender_female, 65, 58)) * df["triglycerides"]

        # Visceral Adiposity Index (gender-specific formula from comments)
        # waist_term = (df["waist_to_hip_ratio"] * 100) / np.where(gender_female, (36.58 + 1.89 * df["bmi"]), (39.68 + 1.88 * df["bmi"]))
        # tri_term = df["triglycerides"] / 88.5 / np.where(gender_female, 0.81, 1.03)
        # gender_multiplier = np.where(gender_female, (1.52 * 38.6) / df["hdl_cholesterol"], (1.31 * 38.6) / df["hdl_cholesterol"])
        # df["Visceral_Adiposity_Index"] = waist_term * tri_term * gender_multiplier

        # df["age_history_multiplier"] = df["age"] * df["family_history_diabetes"]
        # df["age_inactivity_multiplier"] = df["age"] / df["physical_activity_minutes_per_week"]

        # df["gendered_bmi"] = df["bmi"] * np.where(gender_female, 1.1, 1.0)

        df["physical_inactivity"] = 1.0 / (df["physical_activity_minutes_per_week"] + 0.0000001)
        # df["physical_inactivity_squared"] = 1.0 / (df["physical_activity_minutes_per_week"] * df["physical_activity_minutes_per_week"] + 0.0000001)
        # df["log_physical_activity"] = np.log1p(df["physical_activity_minutes_per_week"] + 0.0000001)

        df["age_squared"] = df["age"] * df["age"]

        df["triglycerides_squared"] = df["triglycerides"] * df["triglycerides"]

        # df["inverse_diet_score"] = 1.0 / (df["diet_score"] + 0.0000001)

        # df["bmi_squared"] = df["bmi"] * df["bmi"]
        # df["bmi_root"] = np.sqrt(df["bmi"])

    return dataframes


def remove_base_diabetes_features(dataframes):
    for df in dataframes:
        # df.drop(columns=["id"], inplace=True)
        df.drop(columns=["age"], inplace=True)
        # df.drop(columns=["alcohol_consumption_per_week"], inplace=True)
        # df.drop(columns=["physical_activity_minutes_per_week"], inplace=True)
        # df.drop(columns=["diet_score"], inplace=True)
        # df.drop(columns=["sleep_hours_per_day"], inplace=True)
        # df.drop(columns=["screen_time_hours_per_day"], inplace=True)
        # df.drop(columns=["bmi"], inplace=True)
        # df.drop(columns=["waist_to_hip_ratio"], inplace=True)
        # df.drop(columns=["systolic_bp"], inplace=True)
        # df.drop(columns=["diastolic_bp"], inplace=True)
        # df.drop(columns=["heart_rate"], inplace=True)
        # df.drop(columns=["cholesterol_total"], inplace=True)
        # df.drop(columns=["hdl_cholesterol"], inplace=True)
        # df.drop(columns=["ldl_cholesterol"], inplace=True)
        # df.drop(columns=["triglycerides"], inplace=True)
        # df.drop(columns=["gender"], inplace=True)
        # df.drop(columns=["ethnicity"], inplace=True)
        # df.drop(columns=["education_level"], inplace=True)
        # df.drop(columns=["income_level"], inplace=True)
        # df.drop(columns=["smoking_status"], inplace=True)
        # df.drop(columns=["employment_status"], inplace=True)
        # df.drop(columns=["family_history_diabetes"], inplace=True)
        # df.drop(columns=["hypertension_history"], inplace=True)
        # df.drop(columns=["cardiovascular_history"], inplace=True)
    return dataframes


def one_hot_encode(dataframes, features):
    new_dataframes = []
    for df in dataframes:
        df_out = df.copy()
        dummies = pd.get_dummies(df[features], prefix=features, dummy_na=False)  # new columns
        df_out = df_out.drop(columns=features)  # remove original columns from df
        df_out = pd.concat([df_out, dummies], axis=1)  # add one hot encoded columns to df
        new_dataframes.append(df_out)
    return new_dataframes


def target_to_array(y):
    # transform y into an array
    y_str = y.astype(str)
    le = LabelEncoder().fit(y_str)
    y = le.transform(y_str)
    return y


def target_classes_count(y):
    # how many different classes does the target have?
    n_classes = len(np.unique(y))
    if VERBOSE:
        print("Target shape:", getattr(y, "shape", None), "unique classes:", np.unique(y))
        print(f"{n_classes=}")
    return n_classes


def validate_model(x_training, y, n_classes=None, n_estimators=None, learning_rate=None, num_leaves=None, max_depth=None, min_child_samples=None, subsample=None, colsample_bytree=None, reg_alpha=None, reg_lambda=None, early_stopping_rounds=None, model="lgbm"):
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if n_classes == 2:
        oof_preds = np.zeros(len(y), dtype=float)
    else:
        oof_preds = np.zeros((len(y), n_classes), dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_training, y), start=1):
        x_tr, x_val = x_training.iloc[tr_idx], x_training.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if model == "xgb":  # XGBoost
            if n_classes == 2:
                kwargs = {
                    "objective": "binary:logistic",  # choose objective and eval_metric based on number of classes
                    "eval_metric": "logloss",
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "tree_method": "hist",
                    "early_stopping_rounds": early_stopping_rounds,
                }
            elif n_classes > 2:
                kwargs = {
                    "objective": "multi:softprob",
                    "num_class": n_classes,
                    "eval_metric": "mlogloss",
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "tree_method": "hist",
                    "early_stopping_rounds": early_stopping_rounds,
                }
            else:
                raise ValueError(f"Function validate_model(): Target has no or only 1 class. {n_classes=}")
            m = XGBClassifier(**kwargs, random_state=42 + fold)
            m.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=0)

        elif model == "lgbm":  # LightGBM
            kwargs = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "min_child_samples": min_child_samples,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "metric": "auc",
                "random_state": 42 + fold,
                "n_jobs": -1,
            }
            suppress_python_output()
            m = lightgbm.LGBMClassifier(**kwargs)
            m.fit(x_tr, y_tr, eval_set=[(x_val, y_val)])
            restore_python_output()

        elif model == "catboost":
            kwargs = {
                "iterations": n_estimators,
                "learning_rate": learning_rate,
                "depth": max_depth,
                "random_state": 42 + fold,
                "verbose": False,
            }
            m = CatBoostClassifier(**kwargs)
            m.fit(x_tr, y_tr, eval_set=(x_val, y_val), early_stopping_rounds=early_stopping_rounds, verbose=False)

        elif model == "randomforest":  # very slow, usually bad model
            kwargs = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": colsample_bytree,  # sklearn accepts float for max_features as fraction of features
                "random_state": 42 + fold,
                "n_jobs": -1,
            }
            m = RandomForestClassifier(**kwargs)
            m.fit(x_tr, y_tr)

        else:
            raise ValueError("Unknown model. Choose from xgb, lgbm, catboost, randomforest.")

        if n_classes == 2:
            oof_preds[val_idx] = m.predict_proba(x_val)[:, 1]
        else:
            oof_preds[val_idx] = m.predict_proba(x_val)
    if n_classes == 2:
        auc = roc_auc_score(y, oof_preds)
        logloss = log_loss(y, oof_preds)
        brier = brier_score_loss(y, oof_preds)
    else:
        auc = roc_auc_score(y, oof_preds, multi_class="ovr")
        logloss = log_loss(y, oof_preds)
        brier = np.mean([brier_score_loss((y == c).astype(int), oof_preds[:, c]) for c in range(n_classes)])

    return auc, logloss, brier, oof_preds


def validate_many_models(x_training, y, n_classes):
    # n_estimators =      [6800]
    # learning_rate =     [0.00806338]
    # num_leaves =        [107]
    # max_depth =         [12]
    # min_child_samples = [96]
    # subsample =         [0.8]
    # colsample_bytree =  [0.158]
    # reg_alpha =         [4.7]
    # reg_lambda =        [0.66]
    # models =            ["lgbm"]

    n_estimators = [6800]
    learning_rate = [0.0089]
    num_leaves = [108]
    max_depth = [12]
    min_child_samples = [80]
    subsample = [0.8]
    colsample_bytree = [0.15]
    reg_alpha = [8.14]
    reg_lambda = [0.00428]
    models = ["lgbm"]

    # n_estimators = [6800]
    # learning_rate = [0.008]
    # num_leaves = [107]
    # max_depth = [12]
    # min_child_samples = [96]
    # subsample = [0.8]
    # colsample_bytree = [0.16]
    # reg_alpha = [4.70e+00]
    # reg_lambda = [6.60e-01]
    # models = ["lgbm"]

    # models = ["xgb", "lgbm", "catboost", "randomforest"]
    print(f"Comparing {len(n_estimators) * len(learning_rate) * len(num_leaves) * len(max_depth) * len(min_child_samples) * len(subsample) * len(colsample_bytree) * len(reg_alpha) * len(reg_lambda) * len(models)} different models ")
    counter = 0
    tic = time.perf_counter()
    for ne in n_estimators:
        for lr in learning_rate:
            for nl in num_leaves:
                for md in max_depth:
                    for mc in min_child_samples:
                        for ss in subsample:
                            for cb in colsample_bytree:
                                for ra in reg_alpha:
                                    for rl in reg_lambda:
                                        for model in models:
                                            kwargs = {"n_estimators": ne,
                                                      "learning_rate": lr,
                                                      "num_leaves": nl,
                                                      "max_depth": md,
                                                      "min_child_samples": mc,
                                                      "subsample": ss,
                                                      "colsample_bytree": cb,
                                                      "reg_alpha": ra,
                                                      "reg_lambda": rl,
                                                      "model": model}
                                            auc, logloss, brier, _ = validate_model(x_training, y, n_classes, **kwargs)
                                            new_tic = time.perf_counter()
                                            minutes = (new_tic - tic) / 60
                                            tic = new_tic
                                            counter += 1
                                            # line = f"{counter};{ne:3};{lr:.3f};xx;{md};xx;{ss:.2f};{cb:.2f};xxxxxxxx;xxxxxxxx;{auc:.6f};{logloss:.6f};{model};{minutes:.2f}"
                                            line = f"{counter};{ne:3};{lr:.3f};{nl};{md};{mc};{ss:.2f};{cb:.2f};{ra:.2e};{rl:.2e};{auc:.6f};{logloss:.6f};{model};{minutes:.2f}"
                                            print(line)
                                            if SOUND:
                                                winsound.Beep(frequency=500, duration=300)
                                            with open("model_comparisons.csv", "a", encoding="utf8") as f:
                                                f.write(line + "\n")
    if SOUND:
        winsound.Beep(frequency=1000, duration=1000)


def stack_and_validate_some_models(x_training, y, n_classes):
    # Tvilsom kode!!! Skal nok ignoreres.
    # Hold out 20% for meta validation (honest evaluation)
    x_meta_tr, x_meta_hold, y_meta_tr, y_meta_hold = train_test_split(
        x_training, y, test_size=0.2, random_state=42, stratify=y
    )

    params_list = [
        (2000, 0.07, None, 4, None, 0.8, 0.2, None, None, "lgbm"),         # alt: 0.720773;0.588147 hier: 0.720099;0.588634
        (6000, 0.02, 50, 6, 70, 0.80, 0.20, 9.50e+00, 2.90e-08, "lgbm"),   # alt: 0.722024;0.587087 hier: 0.721279;0.587634
        (6000, 0.02, None, 6, None, 0.8, 0.2, None, None, "lgbm"),         # alt: 0.721163;0.587812 hier: 0.720507;0.588274
    ]                                                                    # stack:                         0.721561;0.587539

    meta_train_cols = []
    meta_hold_cols = []
    col_names = []

    for i, params in enumerate(params_list):
        ne, lr, nl, md, mc, ss, cb, ra, rl, mo = params
        kwargs = {
            "n_estimators": ne, "learning_rate": lr, "num_leaves": nl,
            "max_depth": md, "min_child_samples": mc, "subsample": ss,
            "colsample_bytree": cb, "reg_alpha": ra, "reg_lambda": rl, "model": mo
        }

        # 1) Obtain OOF predictions on the 80% (for meta training)
        auc, logloss, brier, oof_preds = validate_model(x_meta_tr, y_meta_tr, n_classes, **kwargs)

        if n_classes == 2:
            oof_arr = np.asarray(oof_preds).reshape(-1, 1)
            names = [f"model{i}_prob"]
        else:
            oof_arr = np.asarray(oof_preds)
            names = [f"model{i}_class{c}" for c in range(oof_arr.shape[1])]

        meta_train_cols.append(oof_arr)
        col_names.extend(names)

        # 2) Fit base model on full 80% and predict the 20% holdout -> meta validation features
        model_type = mo
        # prepare safe params (remove "model" and None values)
        safe = {k: v for k, v in kwargs.items() if k != "model" and v is not None}

        # train & predict depending on model type
        if model_type == "lgbm":
            safe.setdefault("random_state", 42)
            safe.setdefault("n_jobs", -1)
            # lgbm expects num_leaves/min_child_samples naming already matched
            suppress_python_output()
            m = lightgbm.LGBMClassifier(**safe)
            m.fit(x_meta_tr, y_meta_tr, eval_set=[(x_meta_hold, y_meta_hold)])
            restore_python_output()
            preds_hold = m.predict_proba(x_meta_hold)
        elif model_type == "xgb":
            if n_classes == 2:
                xgb_kwargs = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "n_estimators": safe.get("n_estimators"),
                    "learning_rate": safe.get("learning_rate"),
                    "max_depth": safe.get("max_depth"),
                    "subsample": safe.get("subsample"),
                    "colsample_bytree": safe.get("colsample_bytree"),
                    "random_state": 42,
                }
            else:
                xgb_kwargs = {
                    "objective": "multi:softprob",
                    "num_class": n_classes,
                    "eval_metric": "mlogloss",
                    "n_estimators": safe.get("n_estimators"),
                    "learning_rate": safe.get("learning_rate"),
                    "max_depth": safe.get("max_depth"),
                    "subsample": safe.get("subsample"),
                    "colsample_bytree": safe.get("colsample_bytree"),
                    "random_state": 42,
                }
            suppress_python_output()
            m = XGBClassifier(**xgb_kwargs, use_label_encoder=False)
            m.fit(x_meta_tr, y_meta_tr, eval_set=[(x_meta_hold, y_meta_hold)], verbose=0)
            restore_python_output()
            preds_hold = m.predict_proba(x_meta_hold)
        elif model_type == "catboost":
            cb_kwargs = {
                "iterations": safe.get("n_estimators"),
                "learning_rate": safe.get("learning_rate"),
                "depth": safe.get("max_depth"),
                "random_state": 42,
                "verbose": False,
            }
            m = CatBoostClassifier(**cb_kwargs)
            m.fit(x_meta_tr, y_meta_tr, eval_set=(x_meta_hold, y_meta_hold), verbose=False)
            preds_hold = m.predict_proba(x_meta_hold)
        elif model_type == "randomforest":
            rf_kwargs = {
                "n_estimators": safe.get("n_estimators"),
                "max_depth": safe.get("max_depth"),
                "max_features": safe.get("colsample_bytree"),
                "random_state": 42,
                "n_jobs": -1,
            }
            m = RandomForestClassifier(**rf_kwargs)
            m.fit(x_meta_tr, y_meta_tr)
            preds_hold = m.predict_proba(x_meta_hold)
        else:
            raise ValueError("Unknown model. Choose from xgb, lgbm, catboost, randomforest.")

        # select proper holdout columns and append
        if n_classes == 2:
            meta_hold_cols.append(preds_hold[:, 1].reshape(-1, 1))
        else:
            meta_hold_cols.append(preds_hold)

        # print base model summary (metrics are from OOF on 80%)
        print(f"Base {i}: {mo} -> AUC: {auc:.6f}, LogLoss: {logloss:.6f}, Brier: {brier:.6f}")

    # assemble meta train and holdout matrices
    meta_X_tr = np.hstack(meta_train_cols)
    meta_X_hold = np.hstack(meta_hold_cols)
    meta_df_tr = pd.DataFrame(meta_X_tr, columns=col_names)
    meta_df_hold = pd.DataFrame(meta_X_hold, columns=col_names)

    # train meta model on meta_df_tr and evaluate on meta_df_hold
    meta_params = {
        "n_estimators": 500, "learning_rate": 0.01, "num_leaves": 31,
        "max_depth": 3, "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.0,
        "random_state": 42, "n_jobs": -1
    }
    suppress_python_output()
    meta_m = lightgbm.LGBMClassifier(**meta_params)
    meta_m.fit(meta_df_tr, y_meta_tr, eval_set=[(meta_df_hold, y_meta_hold)])
    restore_python_output()

    if n_classes == 2:
        preds = meta_m.predict_proba(meta_df_hold)[:, 1]
        meta_auc = roc_auc_score(y_meta_hold, preds)
        meta_logloss = log_loss(y_meta_hold, preds)
        meta_brier = brier_score_loss(y_meta_hold, preds)
    else:
        preds = meta_m.predict_proba(meta_df_hold)
        meta_auc = roc_auc_score(y_meta_hold, preds, multi_class="ovr")
        meta_logloss = log_loss(y_meta_hold, preds)
        meta_brier = np.mean([brier_score_loss((y_meta_hold == c).astype(int), preds[:, c]) for c in range(n_classes)])

    print(f"Meta model (holdout 20%) -> AUC: {meta_auc:.6f}, LogLoss: {meta_logloss:.6f}, Brier: {meta_brier:.6f}")
    return meta_auc, meta_logloss, meta_brier


def optuna_optimization(x, y, n_trials=50):
    """
    Optimizes LightGBM hyperparameters using Optuna.
    Replaces the manual loops in validate_many_models.
    All printing during the optimization is appended to `optuna.csv`.
    """
    header = f"Starting Optuna Optimization with {n_trials} trials...\n"
    with open("optuna.csv", "a", encoding="utf8") as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
        print(header, end="")

        def objective(trial):
            # 1. Define the search space
            param = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_jobs": -1,
                "random_state": 42,
                "n_estimators": trial.suggest_int("n_estimators", 6000, 8000, step=400),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.010),
                "num_leaves": trial.suggest_int("num_leaves", 60, 200),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 40, 150),
                "subsample": trial.suggest_float("subsample", 0.8, 0.8),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.12, 0.24),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            log_losses = []

            for tr_idx, val_idx in skf.split(x, y):
                x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                suppress_python_output()
                model = lightgbm.LGBMClassifier(**param)
                model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)])
                restore_python_output()

                preds = model.predict_proba(x_val)[:, 1]
                log_losses.append(log_loss(y_val, preds))

            return np.mean(log_losses)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("\n" + "=" * 40)
        print("✅ OPTIMIZATION FINISHED")
        print(f"Best LogLoss: {study.best_value:.6f}")
        print("Best Params: ")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        print("=" * 40 + "\n")

    return study.best_params


# def model_for_submission_xgb(x_training, y, seeds, n_classes):
#     # train models and keep them in a list
#     models = []
#     if n_classes == 2:
#         kwargs = {
#             "objective": "binary:logistic",  # choose objective and eval_metric based on number of classes
#             "eval_metric": "logloss",
#             "n_estimators": "do not use LGBM is better",
#             "learning_rate": 1,
#             "subsample": 0.8,
#             "colsample_bytree": 0.2,
#             "tree_method": "hist",
#             "early_stopping_rounds": 50,
#         }
#     # elif n_classes > 2:
#     #     kwargs = {
#     #         "objective": "multi:softprob",
#     #         "num_class": n_classes,
#     #         "eval_metric": "mlogloss",
#     #         "n_estimators": n_estimators,
#     #         "learning_rate": learning_rate,
#     #         "max_depth": max_depth,
#     #         "subsample": subsample,
#     #         "colsample_bytree": colsample_bytree,
#     #         "tree_method": "hist",
#     #         "early_stopping_rounds": early_stopping_rounds,
#     #     }
#     else:
#         raise ValueError(f"Function validate_model(): Target has no or only 1 class. {n_classes=}")
#
#     for s in seeds:
#         # take 80% of the trainingsdata, then split these 80% again to obtain trainings and validation data
#         x_train, _, y_train, _ = train_test_split(x_training, y, test_size=0.2, random_state=s, stratify=y)
#         x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=s, stratify=y_train)
#
#         model = XGBClassifier(**kwargs, early_stopping_rounds=50, random_state=s)
#         model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=0)
#         models.append(model)
#     return models


def model_for_submission_lgbm(x_training, y, seeds):
    # train models and keep them in a list
    models = []
    kwargs = {
        # "objective": "binary",
        # "metric": "binary_logloss",
        # "boosting_type": "gbdt",
        "n_estimators": 6800,
        "learning_rate": 0.0089,
        "num_leaves": 108,
        "max_depth": 12,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.15,
        "reg_alpha": 8.14,
        "reg_lambda": 0.00428,
        "random_state": 44,
        "metric": "auc",
        "n_jobs": -1,
        "verbosity": -1,
    }

    print("Fitting    ", end="")
    for s in seeds:
        # take 80% of the trainingsdata, then split these 80% again to obtain trainings and validation data
        x_train, _, y_train, _ = train_test_split(x_training, y, test_size=0.2, random_state=s, stratify=y)
        x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=s, stratify=y_train)
        suppress_python_output()
        model = lightgbm.LGBMClassifier(**kwargs)
        model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)])
        restore_python_output()
        models.append(model)
        print(".", end="")
    print(f"{len(seeds)} fits done.")
    return models


# def smart_ensemble_submission(x_training, y, x_test_final, x_test):
#     """3 models total: LGBM + XGB + CatBoost → 5 min training, no warnings"""
#
#     # Your BEST params from validation
#     best_params = {
#         "n_estimators": 6800, "learning_rate": 0.008, "max_depth": 12,
#         "subsample": 0.8, "colsample_bytree": 0.158
#     }
#
#     print("Training FAST ensemble...")
#
#     # 1. Best LGBM (your winner)
#     suppress_python_output()
#     lgbm_model = lightgbm.LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)
#     lgbm_model.fit(x_training, y)
#     lgbm_probs = lgbm_model.predict_proba(x_test_final.values)[:, 1]
#     restore_python_output()
#
#     # 2. XGBoost (diversity)
#     xgb_model = XGBClassifier(**best_params, tree_method="hist", random_state=42, n_jobs=-1)
#     xgb_model.fit(x_training.values, y)
#     xgb_probs = xgb_model.predict_proba(x_test_final.values)[:, 1]
#
#     # 3. CatBoost (diversity)
#     cat_model = CatBoostClassifier(iterations=best_params["n_estimators"],
#                                   learning_rate=best_params["learning_rate"],
#                                   depth=best_params["max_depth"],
#                                   random_seed=42, verbose=False, thread_count=-1)
#     cat_model.fit(x_training.values, y)
#     cat_probs = cat_model.predict_proba(x_test_final.values)[:, 1]
#
#     # Weighted average (LGBM gets 50% weight as it's your best)
#     final_probs = 0.5 * lgbm_probs + 0.25 * xgb_probs + 0.25 * cat_probs
#
#     print("✅ Smart ensemble complete (5 mins)!")
#     save_submission(x_test, final_probs)
#     return final_probs


def predict_and_average_probabilities(models, x_test):
    probs = []
    print("Predicting ", end="")
    for model in models:
        bi = getattr(model, "best_iteration", None)
        if bi is not None:
            probs.append(model.predict_proba(x_test, iteration_range=(0, bi + 1))[:, 1])
        else:
            probs.append(model.predict_proba(x_test)[:, 1])
        # probs shape: seeds x samples x classes  , classes is = 1 here due to indexing
        print(".", end="")
    print(f"{len(models)} predictions done.")
    probs_arr = np.vstack(probs)  # shape: (n_models, n_samples)
    mean_prob = probs_arr.mean(axis=0)
    return mean_prob


def save_submission(x_test, mean_prob):
    # Build exact submission format CSV
    submission_df = pd.DataFrame({"id": x_test["id"], "diagnosed_diabetes": mean_prob})

    submission_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    if SOUND:
        winsound.Beep(frequency=250, duration=600)


def main():
    df_train, df_test, df_original = read_data(train="train.csv", test="test.csv", original="diabetes dataset.csv")
    df_train = add_original_data(df_train, df_original, originals=1)
    # report_missing_values(df_train)
    # df_test.replace([np.inf, -np.inf], 0, inplace=True)  # replace all inf/-inf with 0
    # df_test.fillna(0, inplace=True)  # replace all NaN with 0
    # df_train.replace([np.inf, -np.inf], 0, inplace=True)
    # df_train.fillna(0, inplace=True)
    cat_cols = ["gender", "ethnicity", "education_level", "income_level", "employment_status", "smoking_status"]
    if VERBOSE:
        print_categories(df_train, cols=cat_cols)
    # df_train, df_test = map_diabetes_features([df_train, df_test])
    add_derived_diabetes_features([df_train, df_test])
    remove_base_diabetes_features([df_train, df_test])
    one_hot_features = ["gender", "ethnicity", "education_level", "income_level", "employment_status", "smoking_status"]
    x_training, x_test = one_hot_encode([df_train, df_test], one_hot_features)
    y_init = x_training["diagnosed_diabetes"]  # the target
    x_training = x_training.drop(columns=["diagnosed_diabetes"])  # remove target from traings data
    y = target_to_array(y_init)  # transform target into an array
    n_classes = target_classes_count(y)
    if VALIDATE_MODELS:  # execute only if validating (many) models
        if MODEL_STACKING:
            stack_and_validate_some_models(x_training, y, n_classes)
        else:
            validate_many_models(x_training, y, n_classes)
        if OPTUNA:
            best_params = optuna_optimization(x_training, y, n_trials=150)
            print(best_params)
    if SUBMIT_PREDICTION:  # execute only if generating a file for submission
        x_test_final = x_test[x_training.columns]  # Make sure column order in test data matches training data
        seeds = list(range(40))
        # model_for_submission_xgb(x_training, y, seeds, n_classes)
        models = model_for_submission_lgbm(x_training, y, seeds)
        mean_prob = predict_and_average_probabilities(models, x_test_final)
        save_submission(x_test, mean_prob)


main()
