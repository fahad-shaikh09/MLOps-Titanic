# model_utils.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Paths to artifacts
MODEL_PATH = "model.pkl"
IMPUTER_PATH = "imputer.pkl"

# Globals for prediction
model = None
imputer = None


def train_model():
    # Load dataset
    df = pd.read_csv("titanic.csv")
    df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = df.dropna()

    # Features and label
    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    imputer = SimpleImputer()
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
# Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_imputed, y_train)

    # Predict and evaluate
    preds = model.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, preds)

    # Save artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(IMPUTER_PATH, "wb") as f:
        pickle.dump(imputer, f)

    # Log and register model with MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)

        # Logs and registers the model under the name "TitanicModel"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="TitanicModel"  # ✅ This line registers it
        )

    # Save model locally as well (optional, for fallback)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


def predict(features):
    global model, imputer

    if model is None or imputer is None:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(IMPUTER_PATH, "rb") as f:
            imputer = pickle.load(f)

    X = imputer.transform([features])
    return model.predict(X)[0]
