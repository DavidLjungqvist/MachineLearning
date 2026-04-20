import pandas as pd


def read_data(train="train.csv", test="test.csv", original="original.csv"):
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)
    df_original = pd.read_csv(original)
    return df_train, df_test, df_original

def add_original(df_train, df_original, originals=1):


def main():
    df_train, df_test, df_original = read_data("train.csv", "test.csv", "original.csv")


main()