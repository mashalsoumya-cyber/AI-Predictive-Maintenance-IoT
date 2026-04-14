import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.preprocess import load_data, preprocess_data
from src.train import train_model, save_model
from src.predict import load_model, predict_failure


def main():
    data_path = "data/iot_sensor_data.csv"
    model_path = "models/predictive_maintenance_model.pkl"

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Loading dataset...")
    df = load_data(data_path)

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Training model...")
    model, results = train_model(X, y)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy : {results['accuracy']:.2f}")
    print(f"Precision: {results['precision']:.2f}")
    print(f"Recall   : {results['recall']:.2f}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    print("\nSaving model...")
    save_model(model, model_path)

    print("\nTesting a sample prediction...")
    loaded_model = load_model(model_path)

    sample_prediction = predict_failure(
        loaded_model,
        temperature=75,
        vibration=1.8,
        current=12
    )

    if sample_prediction == 1:
        print("Prediction Result: Failure Predicted")
    else:
        print("Prediction Result: Machine Normal")

    print("\nGenerating Temperature graph...")
    plt.figure(figsize=(8, 5))
    plt.scatter(df["temperature"], df["failure"])
    plt.xlabel("Temperature")
    plt.ylabel("Failure")
    plt.title("Failure vs Temperature")
    plt.savefig("outputs/failure_vs_temperature.png")
    plt.show()

    print("\nGenerating Vibration graph...")
    plt.figure(figsize=(8, 5))
    plt.scatter(df["vibration"], df["failure"])
    plt.xlabel("Vibration")
    plt.ylabel("Failure")
    plt.title("Failure vs Vibration")
    plt.savefig("outputs/failure_vs_vibration.png")
    plt.show()

    print("\nGenerating Confusion Matrix...")
    ConfusionMatrixDisplay.from_predictions(results["y_test"], results["y_pred"])
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

    print("\nDone successfully.")


if __name__ == "__main__":
    main()