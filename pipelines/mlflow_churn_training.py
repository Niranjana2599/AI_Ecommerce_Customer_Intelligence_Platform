import mlflow
import mlflow.lightgbm

import lightgbm as lgb

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start MLflow run
with mlflow.start_run(run_name="LightGBM_Churn_Model"):
    
    # LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 5)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.lightgbm.log_model(model, "lightgbm_model")

    print(f"Model Accuracy: {accuracy}")
    print("MLflow LightGBM experiment logged successfully")