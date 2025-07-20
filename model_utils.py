# model_utils.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

model = None
imputer = None

def train_model():
    global model, imputer

    df = pd.read_csv("titanic.csv")

    # Keep only useful features
    df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]

    # Convert categorical to numeric
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Separate features/labels
    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]

    # Handle missing values
    imputer = SimpleImputer()
    X = imputer.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

def predict(features):
    global model, imputer
    X = imputer.transform([features])
    return model.predict(X)[0]
