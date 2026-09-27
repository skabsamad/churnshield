import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

print("=" * 60)
print("     🚀 CHURNSHIELD AI — MODEL TRAINING ENGINE 🚀")
print("=" * 60)

# 1. Smart CSV File Finder
possible_paths = [
    os.path.join("data", "Telco_Customer_Churn.csv"),
    os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    "Telco_Customer_Churn.csv",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
]

data_path = None
for p in possible_paths:
    if os.path.exists(p):
        data_path = p
        break

if not data_path:
    # Look inside data/ folder for any CSV file
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.endswith(".csv"):
                data_path = os.path.join("data", f)
                break

if not data_path or not os.path.exists(data_path):
    print("\n[❌ ERROR] CSV File nahi mili! Kripya 'data' folder me CSV file rakhein.")
    exit()

print(f"[*] Found & Loading CSV from: {data_path}")
df = pd.read_csv(data_path)
print(f"[+] Total Customers Loaded: {len(df)}")

# 2. Clean Data
for col in ['customerID', 'CustomerID', 'Customer_ID']:
    if col in df.columns:
        df = df.drop(columns=[col])

if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

target = 'Churn' if 'Churn' in df.columns else 'Exited'
df[target] = df[target].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 1: 1, 0: 0})

X = df.drop(columns=[target])
y = df[target].astype(int)

for c in X.columns:
    if X[c].dtype == 'object' and X[c].nunique() == 2:
        X[c] = X[c].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

X = pd.get_dummies(X, drop_first=True)

# 3. Split Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"[+] Training on {len(X_train)} customers, Testing on {len(X_test)} customers...")

# 4. Train Model
print("\n[*] Training AI Model (XGBoost)...")
model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred) * 100
rec = recall_score(y_test, y_pred) * 100
auc = roc_auc_score(y_test, y_prob) * 100

print("\n" + "=" * 60)
print("               MODEL PERFORMANCE REPORT")
print("=" * 60)
print(f"  • Model Accuracy          : {acc:.2f}%")
print(f"  • Churn Detection (Recall): {rec:.2f}%")
print(f"  • ROC-AUC Score           : {auc:.2f}%")
print("=" * 60)

# 6. Save Model
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/churn_model.pkl")
joblib.dump(X.columns.tolist(), "saved_models/feature_columns.pkl")
print("\n[✔] SUCCESS: Trained Model saved in 'saved_models/churn_model.pkl'")

# 7. Customer Risk Simulation
print("\n" + "=" * 60)
print("      CHURNSHIELD AI — LIVE CUSTOMER RISK PREDICTION")
print("=" * 60)

sample_customers = X_test.head(3)
sample_risks = model.predict_proba(sample_customers)[:, 1]

for i, risk in enumerate(sample_risks):
    risk_pct = risk * 100
    if risk_pct > 70:
        badge = "🔴 HIGH RISK"
        action = "Give 20% discount coupon + call personally."
    elif risk_pct > 40:
        badge = "🟠 MEDIUM RISK"
        action = "Send email checking if they need help."
    else:
        badge = "🟢 SAFE (LOYAL)"
        action = "Send loyalty thank-you message."

    print(f"Customer #{i+1} ➜ Risk: {risk_pct:.1f}% | Status: {badge}")
    print(f"   ↳ Action Plan: {action}\n")

print("=" * 60)
print("🎉 Model Training Complete! Ab Dashboard banane ke liye ready ho!")
print("=" * 60)