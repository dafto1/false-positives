import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Drop high cardinality columns and identifier columns
    cols_to_drop = ['nameOrig', 'nameDest', 'step', 'isMoneyLaundering', 'laundering_typology', 'fraud_probability']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Parse metadata column
    print("Parsing metadata...")
    def parse_metadata(row):
        try:
            if pd.isna(row):
                return pd.Series({'risk_score': 0, 'customer_risk_score': 0, 'amount_vs_average': 0})
            meta = ast.literal_eval(row)
            return pd.Series({
                'risk_score': meta.get('risk_score', 0),
                'customer_risk_score': meta.get('customer_risk_score', 0),
                'amount_vs_average': meta.get('risk_indicators', {}).get('amount_vs_average', 0) if 'risk_indicators' in meta else meta.get('amount_vs_average', 0)
            })
        except Exception as e:
            return pd.Series({'risk_score': 0, 'customer_risk_score': 0, 'amount_vs_average': 0})

    if 'metadata' in df.columns:
        meta_features = df['metadata'].apply(parse_metadata)
        df = pd.concat([df, meta_features], axis=1)
        df = df.drop(columns=['metadata'])
    
    # Handle categorical variables
    print("Encoding categorical variables...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # Fill missing values
    df = df.fillna(0)
    
    return df

def train_model(df):
    print("Preparing data for training...")
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    return model

if __name__ == "__main__":
    filepath = 'd:\\college\\hackathons\\techfiesta\\prototype-1\\AMLNet_August 2025.csv'
    try:
        # Load raw data first to pick samples
        print("Loading raw dataset to select samples...")
        df_raw = pd.read_csv(filepath)
        
        # Find one fraud and one non-fraud index
        fraud_idx = df_raw[df_raw['isFraud'] == 1].index[0]
        legit_idx = df_raw[df_raw['isFraud'] == 0].index[0]
        
        print(f"\nSelected Sample Indices - Fraud: {fraud_idx}, Legit: {legit_idx}")
        
        print("\n--- Sample Fraud Transaction (Raw) ---")
        print(df_raw.iloc[fraud_idx])
        
        print("\n--- Sample Non-Fraud Transaction (Raw) ---")
        print(df_raw.iloc[legit_idx])
        
        # Process data
        df_processed = load_and_preprocess_data(filepath)
        
        # Train model
        model = train_model(df_processed)
        
        print("\n--- Testing Model with Selected Samples ---")
        
        # Prepare samples for prediction (drop target)
        # Note: df_processed has the same index as df_raw because we only dropped columns, not rows
        sample_fraud = df_processed.iloc[[fraud_idx]].drop(columns=['isFraud'])
        sample_legit = df_processed.iloc[[legit_idx]].drop(columns=['isFraud'])
        
        # Predict
        pred_fraud = model.predict(sample_fraud)[0]
        prob_fraud = model.predict_proba(sample_fraud)[0][1]
        
        pred_legit = model.predict(sample_legit)[0]
        prob_legit = model.predict_proba(sample_legit)[0][1]
        
        print(f"\nFraud Sample Prediction: {'FRAUD' if pred_fraud == 1 else 'NORMAL'} (Probability: {prob_fraud:.4f})")
        print(f"Ground Truth: FRAUD")
        
        print(f"\nLegit Sample Prediction: {'FRAUD' if pred_legit == 1 else 'NORMAL'} (Probability: {prob_legit:.4f})")
        print(f"Ground Truth: NORMAL")
        
        print("\nModel training and testing completed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
