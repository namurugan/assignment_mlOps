import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from urllib.parse import urlparse

def preprocess_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Change if needed
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    with mlflow.start_run(run_name="data_preprocessing"):
        # Log parameters
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Load raw data
        df = pd.read_csv(input_path)
        df.dropna(inplace=True)

        # Split features and target
        X = df.drop(["median_house_value", "ocean_proximity"], axis=1)
        y = df["median_house_value"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Save processed data
        os.makedirs(output_dir, exist_ok=True)

        X_train_path = os.path.join(output_dir, "X_train.csv")
        X_test_path = os.path.join(output_dir, "X_test.csv")
        y_train_path = os.path.join(output_dir, "y_train.csv")
        y_test_path = os.path.join(output_dir, "y_test.csv")

        X_train.to_csv(X_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)

        # Log artifacts (split files)
        mlflow.log_artifact(X_train_path, artifact_path="splits")
        mlflow.log_artifact(X_test_path, artifact_path="splits")
        mlflow.log_artifact(y_train_path, artifact_path="splits")
        mlflow.log_artifact(y_test_path, artifact_path="splits")

        # Log metrics
        mlflow.log_metric("X_train_size", len(X_train))
        mlflow.log_metric("X_test_size", len(X_test))
        mlflow.log_metric("y_train_size", len(y_train))
        mlflow.log_metric("y_test_size", len(y_test))

        print("✅ Preprocessing complete and artifacts logged to MLflow.")

# Allow CLI execution
if __name__ == "__main__":
    preprocess_data("data/raw/housing.csv", "data/processed")
