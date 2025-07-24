import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Load and preprocess data
df = pd.read_csv("titanic.csv")
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# MLflow experiment + registration
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Change this if using remote MLflow
mlflow.set_experiment("Titanic Survival Prediction")

with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

    # Log the model and register it
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="TitanicModel"
    )

    print("Model registered and logged to MLflow!")
