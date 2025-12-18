import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
import random

# Page Config
st.set_page_config(page_title="FraudGuard AI", layout="wide", page_icon="🛡️")
st.markdown("""
<style>
    .metric-card { background-color: #0E1117; border: 1px solid #333; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    auto_refresh = st.toggle("Live Stream", value=True)
    refresh_rate = st.slider("Speed (s)", 0.2, 3.0, 1.0)
    st.divider()
    st.info("System Status: ONLINE")

# --- HEADER ---
st.title("🛡️ AI Fraud Detection System")
st.markdown("Real-time anomaly monitoring powered by LightGBM.")

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
kpi_total = col1.empty()
kpi_fraud = col2.empty()
kpi_safe = col3.empty()
kpi_risk = col4.empty()

# --- CHARTS ROW ---
c1, c2 = st.columns([2, 1])
chart_main = c1.empty()
chart_pie = c2.empty()

# --- TABLE ---
st.subheader("Recent Transactions")
table_ph = st.empty()

# --- DATA STATE ---
if "data" not in st.session_state:
    st.session_state.data = []

def generate_tx():
    """Generates fake transactions for the demo"""
    types = ["PAYMENT", "TRANSFER", "CASH_OUT"]
    is_fraud = random.random() < 0.1 # 10% chance
    amount = random.uniform(10, 10000)
    if is_fraud: amount = amount * 20 # Spikes for fraud
    
    return {
        "type": random.choice(types),
        "amount": round(amount, 2),
        "oldbalanceOrg": round(random.uniform(1000, 50000), 2),
        "newbalanceOrig": round(random.uniform(0, 1000), 2),
        "oldbalanceDest": 0.0,
        "newbalanceDest": round(amount, 2),
        "category": "simulated",
        "laundering_typology": "none"
    }

# --- LIVE LOOP ---
if auto_refresh:
    new_tx = generate_tx()
    
    # Call API
    try:
        res = requests.post(API_URL, json=new_tx).json()
        new_tx["Status"] = "FRAUD" if res["is_fraud"] == 1 else "SAFE"
        new_tx["Risk Score"] = res["risk_score"]
        new_tx["Time"] = pd.Timestamp.now().strftime("%H:%M:%S")
        
        # Update Data
        st.session_state.data.insert(0, new_tx)
        if len(st.session_state.data) > 100:
            st.session_state.data.pop()
            
    except:
        st.error("API Error: Make sure uvicorn is running!")

    # Render
    df = pd.DataFrame(st.session_state.data)
    if not df.empty:
        # Metrics
        total = len(df)
        frauds = len(df[df["Status"]=="FRAUD"])
        
        kpi_total.metric("Total Scanned", total)
        kpi_fraud.metric("Fraud Detected", frauds, delta_color="inverse")
        kpi_safe.metric("Safe", total - frauds)
        kpi_risk.metric("Avg Risk Score", f"{df['Risk Score'].mean():.2f}")
        
        # Charts
        with chart_main.container():
            fig = px.scatter(df, x="Time", y="amount", color="Status", 
                           color_discrete_map={"SAFE": "#00FF00", "FRAUD": "#FF0000"},
                           title="Live Transaction Volume")
            st.plotly_chart(fig, use_container_width=True)
            
        with chart_pie.container():
            fig2 = px.pie(df, names="Status", hole=0.5, 
                        color="Status", color_discrete_map={"SAFE": "#00FF00", "FRAUD": "#FF0000"})
            st.plotly_chart(fig2, use_container_width=True)
            
        # Table
        table_ph.dataframe(df[["Time", "Status", "amount", "Risk Score", "type"]], use_container_width=True)

    time.sleep(refresh_rate)