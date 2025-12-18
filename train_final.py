import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


print("Loading dataset...")

df = pd.read_csv("AMLNet_August 2025.csv")


drop_cols = [
    "metadata", 
    "fraud_probability", 
    "isMoneyLaundering",
    "nameOrig", 
    "nameDest"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])


target = "isFraud"
X = df.drop(columns=[target])
y = df[target]

categorical_features = ["type", "category", "laundering_typology"]
for col in categorical_features:
    if col in X.columns:
        X[col] = X[col].astype("category")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fraud_count = y_train.sum()
non_fraud_count = len(y_train) - fraud_count
scale_pos_weight = non_fraud_count / fraud_count
print(f"Training with Scale Weight: {scale_pos_weight:.2f}")

print("Training Model...")
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, 
    y_train, 
    categorical_feature=[c for c in categorical_features if c in X.columns]
)

print("Evaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("Saving model files...")
joblib.dump(model, "fraud_model.pkl")
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
print("✅ Done. Ready for API.")