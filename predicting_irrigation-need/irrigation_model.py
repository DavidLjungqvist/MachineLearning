import pandas as pd
import numpy as np

INCLUDE_ORIGINAL = True

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

def main():
    df_train, df_test, df_original = read_data("train.csv", "test.csv", "original.csv")
    print(len(df_train))
    df_train = add_original(df_train, df_original, originals=1)
    print(len(df_train))


main()
