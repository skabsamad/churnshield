# =====================================================================
#  ChurnShield AI — Production Ready (Cloud Compatible)
# =====================================================================

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

st.set_page_config(page_title="ChurnShield AI", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------
# SMART MODEL LOADER (Cloud + Local Compatible)
# ---------------------------------------------------------------------
@st.cache_resource
def get_model_and_features():
    possible_model_paths = [
        "saved_models/churn_model.pkl",
        "churn_model.pkl",
        "data/saved_models/churn_model.pkl",
        os.path.join("saved_models", "churn_model.pkl")
    ]
    possible_cols_paths = [
        "saved_models/feature_columns.pkl",
        "feature_columns.pkl",
        "data/saved_models/feature_columns.pkl"
    ]

    model = None
    feature_cols = None

    for mp in possible_model_paths:
        if os.path.exists(mp):
            model = joblib.load(mp)
            break

    for cp in possible_cols_paths:
        if os.path.exists(cp):
            feature_cols = joblib.load(cp)
            break

    # Agar model nahi mila toh on-the-fly train kar lo (Cloud fallback)
    if model is None or feature_cols is None:
        st.warning("⚠️ Model file nahi mili. Cloud pe auto-training shuru ho rahi hai... (30 sec)")
        model, feature_cols = train_model_on_cloud()
    
    return model, feature_cols


def train_model_on_cloud():
    """Agar .pkl missing ho toh sample data se model train karke return karta hai"""
    # Sample data dhoondho
    sample_path = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".csv") and ("Telco" in f or "Gym" in f or "Churn" in f or "Customer" in f):
                sample_path = os.path.join(root, f)
                break
        if sample_path:
            break

    if not sample_path:
        # Last fallback - dummy data
        st.error("Koi CSV nahi mili training ke liye.")
        st.stop()

    df = pd.read_csv(sample_path)

    # Basic cleaning
    for col in ['customerID', 'CustomerID', 'Customer_ID', 'id']:
        if col in df.columns:
            df = df.drop(columns=[col])

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    target = 'Churn' if 'Churn' in df.columns else 'Exited'
    if df[target].dtype == 'object':
        df[target] = df[target].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0})

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    for c in X.columns:
        if X[c].dtype == 'object' and X[c].nunique() == 2:
            X[c] = X[c].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Save for next time
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/churn_model.pkl")
    joblib.dump(X.columns.tolist(), "saved_models/feature_columns.pkl")

    return model, X.columns.tolist()


model, feature_cols = get_model_and_features()

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def preprocess_for_model(raw_df):
    df = raw_df.copy()
    for col in ['customerID', 'CustomerID', 'Customer_ID', 'id', 'RowNumber']:
        if col in df.columns:
            df = df.drop(columns=[col])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    for t in ['Churn', 'Exited', 'churn', 'target']:
        if t in df.columns:
            df = df.drop(columns=[t])
    for c in df.columns:
        if df[c].dtype == 'object' and df[c].nunique() == 2:
            df[c] = df[c].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

def risk_label(p):
    if p >= 0.70: return "🔴 High Risk"
    elif p >= 0.40: return "🟠 Medium Risk"
    return "🟢 Safe"

def get_reasons(row):
    reasons = []
    if row.get('Contract') == 'Month-to-month': reasons.append("Month-to-month contract")
    if row.get('tenure') is not None and row.get('tenure') <= 12: reasons.append("New customer (low tenure)")
    if row.get('MonthlyCharges') is not None and row.get('MonthlyCharges') >= 75: reasons.append("High monthly charges")
    if row.get('TechSupport') == 'No': reasons.append("No tech support")
    if row.get('InternetService') == 'Fiber optic': reasons.append("Fiber optic issues")
    if row.get('PaymentMethod') == 'Electronic check': reasons.append("Manual payment method")
    if not reasons: reasons.append("Multiple small risk factors")
    return reasons[:3]

def get_actions(risk_pct, row):
    actions = []
    if risk_pct >= 70:
        actions.append("📞 Priority retention call")
        actions.append("🎁 20% discount / 1 month free")
    elif risk_pct >= 40:
        actions.append("📧 Feedback + re-engagement email")
        actions.append("🎁 Small loyalty reward")
    else:
        actions.append("⭐ Enroll in loyalty program")
    if row.get('Contract') == 'Month-to-month':
        actions.append("📄 Offer annual plan upgrade")
    return actions

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.title("🛡️ ChurnShield AI")
st.sidebar.markdown("*Predict before they leave. Act before it's too late.*")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("📂 Upload Customer CSV", type=['csv'])
use_sample = st.sidebar.checkbox("📊 Use Sample Data", value=True)

st.sidebar.divider()
st.sidebar.markdown("### 💰 Pricing")
st.sidebar.markdown("🟢 Starter — ₹999/mo\n\n🟡 Growth — ₹2,499/mo\n\n🔴 Pro — ₹4,999/mo")

# ---------------------------------------------------------------------
# DATA LOAD
# ---------------------------------------------------------------------
raw_df = None
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ {len(raw_df)} customers loaded")
elif use_sample:
    sample_path = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".csv"):
                sample_path = os.path.join(root, f)
                break
        if sample_path: break
    if sample_path:
        raw_df = pd.read_csv(sample_path)
        st.sidebar.success(f"✅ Sample: {len(raw_df)} customers")

st.title("🛡️ ChurnShield AI")
st.markdown("#### Affordable AI-Powered Customer Churn Prediction for Indian SMBs")

if raw_df is None:
    st.info("👈 Left se CSV upload karo ya Sample Data tick karo")
    st.stop()

# ---------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------
with st.spinner("🤖 AI analyzing customers..."):
    X = preprocess_for_model(raw_df)
    probs = model.predict_proba(X)[:, 1]

results = raw_df.copy()
results['Churn_Risk_%'] = (probs * 100).round(1)
results['Risk_Category'] = [risk_label(p) for p in probs]
results['Top_Reasons'] = ["; ".join(get_reasons(row)) for _, row in raw_df.iterrows()]
results['Recommended_Actions'] = ["; ".join(get_actions(p*100, row)) for p, (_, row) in zip(probs, raw_df.iterrows())]

# Metrics
total = len(results)
high = int((probs >= 0.7).sum())
medium = int(((probs >= 0.4) & (probs < 0.7)).sum())
safe = int((probs < 0.4).sum())

rev_col = None
for c in ['MonthlyCharges', 'Balance', 'EstimatedSalary', 'Monthly_Spend']:
    if c in results.columns:
        rev_col = c
        break
rev_risk = results.loc[probs >= 0.7, rev_col].sum() if rev_col else high * 1200

# KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👥 Total Customers", f"{total:,}")
c2.metric("🔴 High Risk", f"{high:,}")
c3.metric("🟠 Medium Risk", f"{medium:,}")
c4.metric("🟢 Safe", f"{safe:,}")
c5.metric("💸 Revenue at Risk", f"₹{rev_risk:,.0f}")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Customer Report", "🔍 Why They Leave", "📥 Download"])

COLOR_MAP = {'🔴 High Risk': '#e74c3c', '🟠 Medium Risk': '#f39c12', '🟢 Safe': '#2ecc71'}

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        dist = results['Risk_Category'].value_counts().reset_index()
        dist.columns = ['Category', 'Count']
        fig = px.pie(dist, names='Category', values='Count', hole=0.45, color='Category', color_discrete_map=COLOR_MAP, title="Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(results, x='Churn_Risk_%', nbins=25, title="Risk Score Distribution", color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🚨 Top 5 High Risk Customers")
    top5 = results.sort_values('Churn_Risk_%', ascending=False).head(5)
    for _, r in top5.iterrows():
        with st.expander(f"🔴 Risk {r['Churn_Risk_%']}% — {r['Top_Reasons'][:50]}..."):
            st.write("**Reasons:**", r['Top_Reasons'])
            st.write("**Action:**", r['Recommended_Actions'])

with tab2:
    filt = st.selectbox("Filter:", ["All", "🔴 High Risk", "🟠 Medium Risk", "🟢 Safe"])
    show = results if filt == "All" else results[results['Risk_Category'] == filt]
    st.dataframe(show.sort_values('Churn_Risk_%', ascending=False), use_container_width=True, height=450)

with tab3:
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
        imp = imp.sort_values('Importance', ascending=False).head(12)
        fig = px.bar(imp[::-1], x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Reds', title="Top Churn Drivers")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Report (CSV)", csv, "ChurnShield_Report.csv", "text/csv")
    st.dataframe(results.head(10), use_container_width=True)
