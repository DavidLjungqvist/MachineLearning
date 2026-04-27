import pandas as pd
import numpy as np
import time
import sys
import os
from math import floor

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
import lightgbm
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

INCLUDE_ORIGINAL = True
VERBOSE = True
VALIDATE_MODELS = False
GENERATE_PREDICTION = True

def read_data(train="train.csv", test="test.csv", original="original.csv"):
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)
    df_original = pd.read_csv(original)
    return df_train, df_test, df_original

def add_original(df_train, df_original, originals=1):
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

### Fra Diabetes ###
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
                "metric": "multi_logloss",
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

    n_estimators = [200, 300]
    learning_rate = [0.08]
    num_leaves = [50]
    max_depth = [5]
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
                                            # if SOUND:
                                            #     winsound.Beep(frequency=500, duration=300)
                                            with open("model_comparisons.csv", "a", encoding="utf8") as f:
                                                f.write(line + "\n")
    # if SOUND:
    #     winsound.Beep(frequency=1000, duration=1000)

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
    # y = le.transform(y_str)
    y_encoded = le.transform(y_str)
    return y_encoded, le

def target_classes_count(y):
    # how many different classes does the target have?
    n_classes = len(np.unique(y))
    if VERBOSE:
        print("Target shape:", getattr(y, "shape", None), "unique classes:", np.unique(y))
        print(f"{n_classes=}")
    return n_classes

def model_for_submission_lgbm(x_training, y, seeds):
    # train models and keep them in a list
    models = []
    kwargs = {
        # "objective": "binary",
        # "metric": "binary_logloss",
        # "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.08,
        "num_leaves": 50,
        "max_depth": 5,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.15,
        "reg_alpha": 8.14,
        "reg_lambda": 0.00428,
        "random_state": 44,
        "metric": "multi_logloss",
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


def predict_and_average_probabilities(models, x_test):
    probs = []
    print("Predicting ", end="")

    for model in models:
        bi = getattr(model, "best_iteration", None)
        if bi is not None:
            p = model.predict_proba(x_test, iteration_range=(0, bi + 1))
        else:
            p = model.predict_proba(x_test)

        probs.append(p)  # keep ALL classes
        print(".", end="")

    print(f"{len(models)} predictions done.")

    probs_arr = np.stack(probs)  # (n_models, n_samples, n_classes)
    mean_prob = probs_arr.mean(axis=0)  # (n_samples, n_classes)

    return mean_prob

# def save_submission(x_test, mean_prob):
    # submission_df = pd.DataFrame(mean_prob, columns=["class_0", "class_1", "class_2"])
    # submission_df.insert(0, "id", x_test["id"])
    #
    # submission_df.to_csv("submission.csv", index=False)
    # print("Saved submission.csv")
def save_submission(x_test, mean_prob, le):
    # convert probabilities → class index
    labels_encoded = np.argmax(mean_prob, axis=1)

    # convert index → original labels (Low/Medium/High)
    labels = le.inverse_transform(labels_encoded)

    submission_df = pd.DataFrame({
        "id": x_test["id"],
        "Irrigation_Need": labels
    })

    submission_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

def add_derived_features(dataframes):
    enhanced_dataframes = []
    for df in dataframes:
        df["soil_lt25"] = (df["Soil_Moisture"] < 25).astype(int)
        df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
        df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
        df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)
        enhanced_dataframes.append(df)
    return enhanced_dataframes

def main():
    df_train, df_test, df_original = read_data("train.csv", "test.csv", "original.csv")
    df_train = add_original(df_train, df_original, originals=1)

    df_train, df_test = add_derived_features([df_train, df_test])

    one_hot_features = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

    x_training, x_test = one_hot_encode([df_train, df_test], one_hot_features)
    y_init = x_training["Irrigation_Need"]  # the target
    x_training = x_training.drop(columns=["Irrigation_Need"])
    y, le = target_to_array(y_init)  # transform target into an array
    n_classes = target_classes_count(y)
    if VALIDATE_MODELS:  # execute only if validating (many) models
        # if MODEL_STACKING:
        #     stack_and_validate_some_models(x_training, y, n_classes)
        # else:
        validate_many_models(x_training, y, n_classes)
    if GENERATE_PREDICTION:
        start = time.perf_counter()
        x_test_final = x_test[x_training.columns]
        seeds = list(range(40))
        models = model_for_submission_lgbm(x_training, y, seeds)
        mean_prob = predict_and_average_probabilities(models, x_test_final)
        save_submission(x_test, mean_prob, le)
        end = time.perf_counter()
        completion_time = end - start
        mins = floor(completion_time / 60)
        secs = completion_time - (mins * 60)
        print(f"Generated predition in {mins} minutes and {secs} seconds")


main()
