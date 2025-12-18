import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc

# Load data
df = pd.read_csv("AMLNet_August 2025.csv")

# Drop leaky / unused columns
drop_cols = [
    "metadata",
    "fraud_probability",
    "isMoneyLaundering"
]
df = df.drop(columns=drop_cols)

# Target
target = "isFraud"
X = df.drop(columns=[target])
y = df[target]

# Categorical features
categorical_features = [
    "type",
    "category",
    "laundering_typology",
    "nameOrig",
    "nameDest"
]

for col in categorical_features:
    X[col] = X[col].astype("category")

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Class weighting
fraud = y_train.sum()
non_fraud = len(y_train) - fraud
scale_pos_weight = non_fraud / fraud
print("Scale pos weight:", scale_pos_weight)

# LightGBM model
model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",      # ✅ fixed
    n_estimators=300,          # ✅ fixed
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# PR-AUC
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print("PR-AUC:", pr_auc)

# Threshold tuning
threshold = 0.2
y_pred = (y_prob >= threshold).astype(int)

print(classification_report(y_test, y_pred))
