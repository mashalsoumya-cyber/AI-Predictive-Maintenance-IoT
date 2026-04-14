import joblib
import pandas as pd


def load_model(model_path: str):
    """
    Load saved model.
    """
    return joblib.load(model_path)


def predict_failure(model, temperature, vibration, current):
    """
    Predict machine failure from new sensor values.
    """
    sample = pd.DataFrame([{
        "temperature": temperature,
        "vibration": vibration,
        "current": current
    }])

    prediction = model.predict(sample)[0]
    return prediction