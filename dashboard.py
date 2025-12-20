import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
import random
import ast

# Page Config
st.set_page_config(page_title="FraudGuard AI", layout="wide", page_icon="🛡️")
st.markdown("""
<style>
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: left;
        margin-bottom: 20px;
    }
    .metric-label {
        font-size: 14px;
        font-weight: 600;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .metric-sub {
        font-size: 12px;
        color: #95a5a6;
    }
    .stApp {
        background-color: #FFFFFF;
        color: #2c3e50;
    }
    p, h1, h2, h3, div, span {
        color: #2c3e50 !important;
    }
    /* Hide Streamlit Header */
    .stAppHeader {
        display: none;
    }
    .stDeployButton {
        display: none;
    }
    #MainMenu {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Helper for Metric Cards
def metric_card(label, value, sub_text, color="black"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-sub">{sub_text}</div>
    </div>
    """, unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    auto_refresh = st.toggle("Live Stream", value=True)
    refresh_rate = st.slider("Speed (s)", 1.0, 10.0, 3.0)
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

# --- TABS LAYOUT ---
tab_trans, tab_analytics, tab_trends = st.tabs(["Transactions", "Analytics", "Trends"])

# 1. Transactions Tab
with tab_trans:
    st.subheader("Recent Transactions")
    table_ph = st.empty()

# 2. Analytics Tab
with tab_analytics:
    # Create persistent placeholders to avoid layout shifts/flashing
    c1, c2 = st.columns([2, 1])
    with c1:
        chart_main_ph = st.empty()
    with c2:
        chart_pie_ph = st.empty()

# 3. Trends Tab
with tab_trends:
    st.info("Historical trend analysis module is currently under development.")

@st.cache_data
def load_and_sample_data():
    """Loads dataset and picks a random sample of transactions"""
    try:
        # Load only necessary columns
        cols = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud", "category", "laundering_typology", "nameOrig", "nameDest", "metadata"]
        
        # Read full dataset
        df_full = pd.read_csv("AMLNet_August 2025.csv", usecols=cols)
        
        # Add missing columns expected by API
        df_full["oldbalanceDest"] = 0.0
        df_full["newbalanceDest"] = 0.0
        
        # Split fraud and safe
        fraud_df = df_full[df_full["isFraud"] == 1]
        safe_df = df_full[df_full["isFraud"] == 0]
        
        # Determine sample size (random between 10 and 39)
        total_sample_size = random.randint(10, 39)
        
        # Ensure at least 5 frauds
        n_fraud = min(len(fraud_df), max(5, int(total_sample_size * 0.2)))
        n_safe = total_sample_size - n_fraud
        
        # Sample
        sample_fraud = fraud_df.sample(n=n_fraud)
        sample_safe = safe_df.sample(n=n_safe)
        
        # Combine and shuffle
        final_sample = pd.concat([sample_fraud, sample_safe]).sample(frac=1).reset_index(drop=True)
        
        # Parse Metadata for City/Country
        def parse_meta(row):
            try:
                # Assuming metadata is a string representation of a dict
                meta_dict = ast.literal_eval(row)
                return meta_dict.get("device_info", {}).get("location", "Unknown"), meta_dict.get("device_info", {}).get("city", "Unknown")
            except:
                return "Unknown", "Unknown"

        # Apply robust parsing if structure is unknown, defaulting to a safe get
        # Let's assume keys "location" or "city" might be directly in dict or nested.
        # Based on user hint: "location and city data".
        # I'll try to extract them generically.
        
        parsed_data = []
        for _, row in final_sample.iterrows():
            d = row.to_dict()
            try:
                meta = ast.literal_eval(row["metadata"]) if isinstance(row["metadata"], str) else {}
                # Extracting typical location keys
                # Adjust these keys based on actual data if known. 
                # For now assuming 'city' and 'country' keys exist as per user request.
                d["City"] = meta.get("payment_location", {}).get("city", meta.get("city", "Unknown"))
                d["Country"] = meta.get("payment_location", {}).get("country", meta.get("country", "Unknown"))
            except:
                d["City"] = "Unknown"
                d["Country"] = "Unknown"
            parsed_data.append(d)
            
        return parsed_data
        
    except FileNotFoundError:
        st.error("Dataset 'AMLNet_August 2025.csv' not found. Please put it in the root directory.")
        return []

# --- PERSISTENCE ---
# Use st.cache_resource to persist data across browser refreshes (sessions)
class GlobalState:
    def __init__(self):
        self.data = []
        self.tx_queue = load_and_sample_data()
        self.tx_index = 0

@st.cache_resource
def get_state():
    return GlobalState()

state = get_state()

def get_next_tx():
    """Fetches the next transaction from the sampled queue"""
    if not state.tx_queue:
        return None
        
    # Cycle through if reached end, or stop? Let's cycle for infinite stream effect
    idx = state.tx_index % len(state.tx_queue)
    tx = state.tx_queue[idx]
    
    state.tx_index += 1
    return tx

# --- LIVE LOOP ---
# --- LIVE LOOP ---
# --- LIVE LOOP ---
if auto_refresh:
    # Use a loop for smooth animation without page reload
    while True:
        new_tx = get_next_tx()
        
        if new_tx:
            # Call API
            try:
                res = requests.post(API_URL, json=new_tx).json()
                new_tx["Status"] = "FRAUD" if res["is_fraud"] == 1 else "SAFE"
                new_tx["Risk Score"] = res["risk_score"]
                new_tx["Time"] = pd.Timestamp.now().strftime("%H:%M:%S")
                
                # Update Data
                state.data.insert(0, new_tx)
                if len(state.data) > 27:
                    state.data.pop()
                    
            except:
                st.error("API Error: Make sure uvicorn is running!")

            # Render
            df = pd.DataFrame(state.data)
            if not df.empty:
                # Metrics
                total = len(df)
                frauds_df = df[df["Status"]=="FRAUD"]
                frauds = len(frauds_df)
                safe = total - frauds
                fraud_rate = (frauds / total * 100) if total > 0 else 0
                total_blocked = frauds_df["amount"].sum()
                avg_risk = df['Risk Score'].mean()

                # Render Metrics in nicely styled cards
                with kpi_total:
                    metric_card("Total Transactions", total, f"{fraud_rate:.1f}% fraud rate")
                with kpi_fraud:
                    metric_card("Fraud Detected", frauds, f"${total_blocked:,.2f} blocked", color="#e74c3c")
                with kpi_safe:
                    metric_card("Legitimate", safe, f"{100-fraud_rate:.1f}% of total", color="#27ae60")
                with kpi_risk:
                    metric_card("Avg Risk Score", f"{avg_risk:.2f}", "Model Confidence")
                
                # Charts
                
                # 1. Amount Distribution (Bar Chart)
                # Binning the amounts
                bins = [0, 100, 500, 1000, 5000, float('inf')]
                labels = ['$0-$100', '$100-$500', '$500-$1K', '$1K-$5K', '$5K+']
                df['Amount Range'] = pd.cut(df['amount'], bins=bins, labels=labels, right=False)
                
                amount_counts = df['Amount Range'].value_counts().reindex(labels).reset_index()
                amount_counts.columns = ['Amount Range', 'Count']
                
                fig = px.bar(amount_counts, x='Amount Range', y='Count', 
                             title="Amount Distribution", 
                             text_auto=True,
                             color_discrete_sequence=['#8e44ad'])
                
                # STRICT Layout Control to prevent flashing/black background
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", # Transparent background
                    paper_bgcolor="rgba(0,0,0,0)", # Transparent background
                    font=dict(color="#2c3e50"),
                    transition_duration=0, # Disable transition animation to stop flickering artifact if any
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(range=[0, 60], title="Count"),
                    height=350
                )
                chart_main_ph.plotly_chart(fig, use_container_width=True, key=f"chart_main_{time.time()}")
                
                fig2 = px.pie(df, names="Status", hole=0.5, 
                            color="Status", color_discrete_map={"SAFE": "#00FF00", "FRAUD": "#FF0000"})
                fig2.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#2c3e50"),
                    transition_duration=0,
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=350
                )
                chart_pie_ph.plotly_chart(fig2, use_container_width=True, key=f"chart_pie_{time.time()}")
                    
                # Table
                display_df = df.copy()
                # Create Sr No (1-indexed)
                display_df["Sr No"] = range(1, len(display_df) + 1)
                
                # Select and reorder columns
                cols_to_show = ["Sr No", "nameOrig", "nameDest", "type", "amount", "category", "oldbalanceOrg", "newbalanceOrig", "City", "Country", "Risk Score", "Status"]
                display_df = display_df[cols_to_show]
                
                # Rename columns
                display_df.columns = ["Sr No", "Customer Id", "Recipient Id", "Type", "Amount", "Category", "Old Balance", "New Balance", "City", "Country", "Risk", "Status"]
                
                # Apply Styling
                def highlight_status(val):
                    if val == 'FRAUD':
                        return 'background-color: #e74c3c; color: white; font-weight: bold' # Red
                    elif val == 'SAFE':
                        return 'background-color: #2ecc71; color: white; font-weight: bold' # Green
                    return ''
                
                styled_df = display_df.style.map(highlight_status, subset=['Status'])
                
                table_ph.dataframe(styled_df, use_container_width=True, hide_index=True)

        time.sleep(refresh_rate)
