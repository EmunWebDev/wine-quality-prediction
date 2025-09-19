import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("data/winequality-red-selected-missing.csv")

# 2. Create binary target: good (>=7) / not good (<7)
df['good_quality'] = [1 if x >= 7 else 0 for x in df['quality']]

# Features and target
X = df.drop(['quality', 'good_quality'], axis=1)
y = df['good_quality']

# 3. Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Save model, imputer, and scaler
joblib.dump(model, "models/wine_quality_model.pkl")
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model, imputer, and scaler saved successfully.")
