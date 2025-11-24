import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def train_test_split_data(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
