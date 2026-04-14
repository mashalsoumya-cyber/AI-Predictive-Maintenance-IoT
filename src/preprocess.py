import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Basic preprocessing:
    - remove missing values
    - separate features and target
    """
    df = df.dropna()

    X = df[['temperature', 'vibration', 'current']]
    y = df['failure']

    return X, y