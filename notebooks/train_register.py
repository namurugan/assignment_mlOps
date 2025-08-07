import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

# ✅ Set the MLflow tracking URI to your running tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Adjust if needed

# ✅ Optionally confirm URI
print("Tracking URI:", mlflow.get_tracking_uri())


def load_data():
    """Load preprocessed training and test datasets."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2


def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    """Train the model, evaluate, and log to MLflow."""
    with mlflow.start_run(run_name=model_name) as run:
        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        rmse, mae, r2 = evaluate_model(model, X_test, y_test)

        # Log parameters and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        return rmse, run.info.run_id, model


def register_best_model(model, run_id, model_name="CaliforniaHousingModel"):
    """Register and promote the best model to Staging in MLflow Model Registry."""
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, f"model/{model_name}.pkl")

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    # Register model (create only if not already exists)
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException:
        pass  # Already exists

    # Create new model version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )

    # Transition to staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )

    print(f"✅ Registered model '{model_name}' version {model_version.version} and moved to Staging.")


if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train and evaluate models
    rmse_lr, run_id_lr, model_lr = train_and_log_model("LinearRegression", LinearRegression(), X_train, y_train, X_test, y_test)
    rmse_dt, run_id_dt, model_dt = train_and_log_model("DecisionTree", DecisionTreeRegressor(), X_train, y_train, X_test, y_test)

    # Choose best model based on RMSE
    if rmse_lr < rmse_dt:
        best_run_id = run_id_lr
        best_model = model_lr
        best_model_name = "LinearRegression"
    else:
        best_run_id = run_id_dt
        best_model = model_dt
        best_model_name = "DecisionTree"

    print(f"🏆 Best model: {best_model_name}")
    
    # Register best model
    register_best_model(best_model, best_run_id)
