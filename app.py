# =====================================================================
#  Project : ChurnShield AI — Streamlit Dashboard
#  Purpose : CSV Upload → Churn Prediction → Risk Dashboard → Report
# =====================================================================

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnShield AI",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = os.path.join("saved_models", "churn_model.pkl")
COLS_PATH = os.path.join("saved_models", "feature_columns.pkl")

# ---------------------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)
    return model, cols

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file nahi mili! Pehle terminal me run karo: python data/train.py")
    st.stop()

model, feature_cols = load_model()

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def preprocess_for_model(raw_df):
    """Uploaded CSV ko model ke format me convert karta hai"""
    df = raw_df.copy()
    for col in ['customerID', 'CustomerID', 'Customer_ID']:
        if col in df.columns:
            df = df.drop(columns=[col])
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    for t in ['Churn', 'Exited', 'churn']:
        if t in df.columns:
            df = df.drop(columns=[t])
    for c in df.columns:
        if df[c].dtype == 'object' and df[c].nunique() == 2:
            df[c] = df[c].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

def risk_label(p):
    if p >= 0.70:
        return "🔴 High Risk"
    elif p >= 0.40:
        return "🟠 Medium Risk"
    return "🟢 Safe"

def get_reasons(row):
    """Customer kyun jaa sakta hai — top reasons"""
    reasons = []
    if row.get('Contract') == 'Month-to-month':
        reasons.append("Month-to-month contract (no long-term commitment)")
    if row.get('tenure') is not None and row.get('tenure') <= 12:
        reasons.append("Naya customer (kam tenure)")
    if row.get('MonthlyCharges') is not None and row.get('MonthlyCharges') >= 80:
        reasons.append("Monthly charges zyada hain")
    if row.get('TechSupport') == 'No':
        reasons.append("Tech support service nahi li")
    if row.get('InternetService') == 'Fiber optic':
        reasons.append("Fiber optic service issues")
    if row.get('PaymentMethod') == 'Electronic check':
        reasons.append("Manual payment (electronic check)")
    if row.get('OnlineSecurity') == 'No':
        reasons.append("Online security add-on nahi hai")
    if not reasons:
        reasons.append("Kai chhote factors mil kar risk badha rahe hain")
    return reasons[:3]

def get_actions(risk_pct, row):
    """Business ko kya action lena chahiye"""
    actions = []
    if risk_pct >= 70:
        actions.append("📞 Retention team se personal call")
        actions.append("🎁 20% discount / 1 month free offer")
    elif risk_pct >= 40:
        actions.append("📧 Feedback survey / re-engagement email")
        actions.append("🎁 Chhota loyalty reward")
    else:
        actions.append("⭐ Loyalty program me enroll karo")
    if row.get('Contract') == 'Month-to-month':
        actions.append("📄 Annual plan par extra discount")
    if row.get('TechSupport') == 'No':
        actions.append("🛠️ Free tech-support trial")
    return actions

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.title("🛡️ ChurnShield AI")
st.sidebar.markdown("*Predict before they leave. Act before it's too late.*")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("📂 Apna Customer CSV Upload Karo", type=['csv'])
use_sample = st.sidebar.checkbox("📊 Sample Telco Data Use Karo", value=True)

st.sidebar.divider()
st.sidebar.markdown("### 💰 Pricing Plans")
st.sidebar.markdown("🟢 Starter — ₹999/mo\n\n🟡 Growth — ₹2,499/mo\n\n🔴 Pro — ₹4,999/mo")

# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
raw_df = None

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ {len(raw_df)} customers loaded!")
elif use_sample:
    sample_path = None
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.endswith(".csv"):
                sample_path = os.path.join("data", f)
                break
    if sample_path:
        raw_df = pd.read_csv(sample_path)
        st.sidebar.success(f"✅ Sample data: {len(raw_df)} customers")
    else:
        st.sidebar.error("data/ folder me koi CSV nahi mili")

# ---------------------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------------------
st.title("🛡️ ChurnShield AI")
st.markdown("#### Affordable AI-Powered Customer Churn Prediction for Indian SMBs")

if raw_df is None:
    st.info("👈 Left sidebar se CSV upload karo ya **Sample Data** checkbox tick karo")
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Price", "₹999/mo", "vs ₹30K+ competitors")
    c2.metric("⏱️ Setup Time", "2 Minutes", "Sirf CSV upload")
    c3.metric("🧠 Coding Skill", "ZERO", "Koi technical knowledge nahi")
    st.stop()

# ---------------------------------------------------------------------
# RUN PREDICTIONS
# ---------------------------------------------------------------------
with st.spinner("🤖 AI model sabhi customers ka churn risk calculate kar raha hai..."):
    X = preprocess_for_model(raw_df)
    probs = model.predict_proba(X)[:, 1]

results = raw_df.copy()
results['Churn_Risk_%'] = (probs * 100).round(1)
results['Risk_Category'] = [risk_label(p) for p in probs]
results['Top_Reasons'] = ["; ".join(get_reasons(row)) for _, row in raw_df.iterrows()]
results['Recommended_Actions'] = [
    "; ".join(get_actions(p * 100, row)) for p, (_, row) in zip(probs, raw_df.iterrows())
]

# ---------------------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------------------
total = len(results)
high = int((probs >= 0.7).sum())
medium = int(((probs >= 0.4) & (probs < 0.7)).sum())
safe = int((probs < 0.4).sum())
rev_risk = results.loc[probs >= 0.7, 'MonthlyCharges'].sum() if 'MonthlyCharges' in results.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("👥 Total Customers", total)
c2.metric("🔴 High Risk", high)
c3.metric("🟠 Medium Risk", medium)
c4.metric("🟢 Safe", safe)
c5.metric("💸 Revenue at Risk/mo", f"₹{rev_risk:,.0f}")

st.divider()

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "📋 Customer Risk Report", "🔍 Why Customers Leave", "📥 Download Report"
])

COLOR_MAP = {'🔴 High Risk': '#e74c3c', '🟠 Medium Risk': '#f39c12', '🟢 Safe': '#2ecc71'}

with tab1:
    colA, colB = st.columns(2)
    with colA:
        dist = results['Risk_Category'].value_counts().reset_index()
        dist.columns = ['Category', 'Count']
        fig1 = px.pie(dist, names='Category', values='Count', hole=0.45,
                      title="Customer Risk Distribution",
                      color='Category', color_discrete_map=COLOR_MAP)
        st.plotly_chart(fig1, use_container_width=True)
    with colB:
        fig2 = px.histogram(results, x='Churn_Risk_%', nbins=25,
                            title="Churn Risk Score Distribution",
                            color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig2, use_container_width=True)

    if 'tenure' in results.columns and 'MonthlyCharges' in results.columns:
        fig3 = px.scatter(results, x='tenure', y='MonthlyCharges', color='Risk_Category',
                          title="Tenure vs Monthly Charges (Risk ke hisaab se)",
                          color_discrete_map=COLOR_MAP, opacity=0.6)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🚨 Immediate Action Required — Top 5 High Risk Customers")
    top5 = results.sort_values('Churn_Risk_%', ascending=False).head(5)
    for _, r in top5.iterrows():
        with st.expander(f"🔴 Risk: {r['Churn_Risk_%']}%  |  Reasons: {r['Top_Reasons'][:60]}..."):
            st.markdown(f"**Kyun jaa sakta hai:** {r['Top_Reasons']}")
            st.markdown(f"**Kya karo:** {r['Recommended_Actions']}")

with tab2:
    filt = st.selectbox("Risk Category se Filter Karo:",
                        ["All", "🔴 High Risk", "🟠 Medium Risk", "🟢 Safe"])
    display_df = results if filt == "All" else results[results['Risk_Category'] == filt]
    id_cols = [c for c in ['customerID', 'CustomerID'] if c in results.columns]
    show_cols = id_cols + ['Churn_Risk_%', 'Risk_Category', 'Top_Reasons', 'Recommended_Actions']
    st.markdown(f"**{len(display_df)} customers dikh rahe hain**")
    st.dataframe(display_df[show_cols].sort_values('Churn_Risk_%', ascending=False),
                 use_container_width=True, height=500)

with tab3:
    st.subheader("AI ke hisaab se Churn ke Sabse Bade Reasons")
    imp = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
    imp = imp.sort_values('Importance', ascending=False).head(12)
    fig = px.bar(imp[::-1], x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Reds',
                 title="Top 12 Churn Drivers (Feature Importance)")
    st.plotly_chart(fig, use_container_width=True)
    st.success("💡 **Business Tip:** In top factors par focus karo — contract type, tenure, aur monthly charges sabse zyada matter karte hain.")

with tab4:
    st.subheader("📥 Complete Churn Prediction Report Download Karo")
    st.markdown("Ye CSV file business owner apne team ko de sakta hai — har customer ka risk %, reason aur action plan ke saath.")
    csv_data = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv_data,
        file_name="churnshield_report.csv",
        mime="text/csv"
    )
    st.dataframe(results.head(10), use_container_width=True)