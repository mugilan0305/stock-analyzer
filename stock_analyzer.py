import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import html
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import quote
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# APP CONFIG & INSTITUTIONAL THEME
# ============================================================
st.set_page_config(
    page_title="Quant Terminal | India Equities",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e17; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .block-container { max-width: 1600px; padding-top: 1rem; }
    h1, h2, h3 { color: #f8fafc; font-weight: 800; tracking: tight; }
    .metric-card { background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .badge-buy { background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 4px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #10b981; }
    .badge-sell { background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 4px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #ef4444; }
    .badge-watch { background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 4px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #f59e0b; }
    .top-pick { background: linear-gradient(145deg, #1e1b4b, #0f172a); border: 1px solid #6366f1; border-radius: 12px; padding: 2rem; }
    .value-up { color: #10b981; font-weight: bold; }
    .value-down { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

IST = ZoneInfo("Asia/Kolkata")

WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ZOMATO.NS", "TATAMOTORS.NS", 
    "INFY.NS", "ITC.NS", "SBIN.NS", "LT.NS", "HAL.NS", "BAJFINANCE.NS", 
    "TRENT.NS", "MARUTI.NS", "KOTAKBANK.NS", "BSE.NS"
]

# ============================================================
# UTILITIES & INDICATORS
# ============================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def money(value):
    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return "N/A"

def get_signal_badge(signal):
    if "BUY" in signal: return f'<span class="badge-buy">BUY</span>'
    if "SELL" in signal: return f'<span class="badge-sell">SELL</span>'
    return f'<span class="badge-watch">{signal}</span>'

# ============================================================
# DATA PIPELINE
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(tickers):
    return yf.download(tickers, period="2y", interval="1d", auto_adjust=True, progress=False)

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
            "de": info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else 0,
            "pe": info.get("trailingPE", 0),
            "eps_g": info.get("earningsGrowth", 0) * 100 if info.get("earningsGrowth") else 0,
            "margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0,
        }
    except Exception:
        return {"roe": 0, "de": 0, "pe": 0, "eps_g": 0, "margin": 0}

# ============================================================
# AI MODELING PIPELINE
# ============================================================
def run_ai_prediction(df):
    if df is None or len(df) < 100: return None
    
    df = df.copy()
    close = df["Close"]
    
    # Advanced Features
    df["RSI_14"] = calculate_rsi(close, 14)
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["DIST_50"] = (close - df["SMA_50"]) / df["SMA_50"]
    df["RET_5"] = close.pct_change(5)
    df["VOL_Z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    
    # Target: 5-Day forward return
    df["TARGET"] = close.shift(-5) / close - 1
    
    features = ["RSI_14", "DIST_50", "RET_5", "VOL_Z"]
    dataset = df.dropna().copy()
    
    if len(dataset) < 50: return None
    
    X = dataset[features]
    y = dataset["TARGET"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dual Ensemble
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    
    gbr.fit(X_scaled, y)
    rf.fit(X_scaled, y)
    
    latest_X = scaler.transform(df[features].iloc[[-1]].fillna(0))
    pred_return = (gbr.predict(latest_X)[0] * 0.6) + (rf.predict(latest_X)[0] * 0.4)
    
    predicted_price = float(close.iloc[-1]) * (1 + pred_return)
    confidence = max(50, 95 - abs(gbr.predict(latest_X)[0] - rf.predict(latest_X)[0]) * 1000)
    
    return {
        "expected_return": pred_return * 100,
        "target_price": predicted_price,
        "confidence": confidence
    }

# ============================================================
# UI INTERFACE
# ============================================================
st.title("🏛️ India Equities: Institutional Quant Terminal")

data = fetch_market_data(WATCHLIST)
if data.empty:
    st.error("Data feed interrupted. Please refresh.")
    st.stop()

tabs = st.tabs(["⚡ AI Screener & Signals", "📈 Advanced Charting"])

with tabs[0]:
    st.markdown("### Algorithmic Target Matrix (5-Day Outlook)")
    
    results = []
    for ticker in WATCHLIST:
        if ticker not in data.columns.levels[1] if isinstance(data.columns, pd.MultiIndex) else ticker not in data:
            continue
            
        stock_df = data.xs(ticker, axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data
        cmp = stock_df["Close"].iloc[-1]
        
        ai_res = run_ai_prediction(stock_df)
        if not ai_res: continue
        
        fund = fetch_fundamentals(ticker)
        
        signal = "BUY" if ai_res["expected_return"] > 1.5 else "SELL" if ai_res["expected_return"] < -1.0 else "WATCH"
        
        results.append({
            "Symbol": ticker.replace(".NS", ""),
            "Signal": signal,
            "CMP": cmp,
            "AI Target": ai_res["target_price"],
            "Exp. Return (%)": ai_res["expected_return"],
            "Conviction (%)": ai_res["confidence"],
            "ROE (%)": fund["roe"],
            "P/E": fund["pe"]
        })
    
    df_results = pd.DataFrame(results).sort_values("Exp. Return (%)", ascending=False)
    
    # Highlight Top Pick
    if not df_results.empty:
        top = df_results.iloc[0]
        st.markdown(f"""
        <div class="top-pick">
            <h4 style='color: #818cf8; margin: 0;'>🔥 Highest Probability Setup</h4>
            <h1 style='margin: 10px 0;'>{top['Symbol']} {get_signal_badge(top['Signal'])}</h1>
            <div style="display: flex; gap: 30px; margin-top: 20px;">
                <div><div style="color: #94a3b8;">Entry Price</div><div style="font-size: 1.5rem; font-weight: bold;">{money(top['CMP'])}</div></div>
                <div><div style="color: #94a3b8;">AI Target (5D)</div><div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">{money(top['AI Target'])}</div></div>
                <div><div style="color: #94a3b8;">Model Conviction</div><div style="font-size: 1.5rem; font-weight: bold;">{top['Conviction (%)']:.1f}%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render Dataframe
    def style_df(val):
        if val == 'BUY': return 'color: #10b981; font-weight: bold;'
        if val == 'SELL': return 'color: #ef4444; font-weight: bold;'
        return 'color: #f59e0b; font-weight: bold;'

    st.dataframe(
        df_results.style.map(style_df, subset=['Signal']),
        column_config={
            "CMP": st.column_config.NumberColumn(format="₹%.2f"),
            "AI Target": st.column_config.NumberColumn(format="₹%.2f"),
            "Exp. Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Conviction (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)
        },
        use_container_width=True, hide_index=True
    )

with tabs[1]:
    st.markdown("### Institutional Charting & Volume Profile")
    selected_ticker = st.selectbox("Select Asset", [t.replace(".NS", "") for t in WATCHLIST])
    
    ticker_ns = f"{selected_ticker}.NS"
    df_chart = data.xs(ticker_ns, axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data
    df_chart = df_chart.tail(200).copy() # Last 200 days
    
    df_chart["SMA20"] = df_chart["Close"].rolling(20).mean()
    df_chart["SMA50"] = df_chart["Close"].rolling(50).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price"), row=1, col=1)
    
    # SMAs
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA20'], line=dict(color='#f59e0b', width=1), name="20 SMA"), row=1, col=1)
    
    # Fixed the missing closing parenthesis here
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA50'], line=dict(color='#3b82f6', width=1), name="50 SMA"), row=1, col=1)
    
    # Volume
    colors = ['#ef4444' if row['Open'] - row['Close'] >= 0 else '#10b981' for index, row in df_chart.iterrows()]
    fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
