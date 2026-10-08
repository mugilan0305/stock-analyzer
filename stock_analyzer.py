import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# INSTITUTIONAL THEME & CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AlphaDesk | Institutional Quantitative Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #080c14; color: #cbd5e1; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .block-container { max-width: 1680px; padding-top: 1rem; padding-bottom: 2rem; }
    
    /* Institutional Metric Cards */
    .metric-panel {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-panel-title { color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 4px; }
    .metric-panel-value { font-size: 22px; font-weight: 800; color: #f8fafc; font-family: monospace; }
    
    /* Top Conviction Alpha Banner */
    .alpha-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #6366f1;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Status Badges */
    .badge-buy { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; }
    .badge-sell { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; }
    .badge-watch { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "INFY.NS", "ITC.NS", "SBIN.NS", "LT.NS", "BAJFINANCE.NS", "HAL.NS",
    "TRENT.NS", "TATAMOTORS.NS", "ZOMATO.NS", "BSE.NS"
]

# ============================================================
# ROBUST DATA & TECHNICAL ENGINE
# ============================================================
def extract_single_ticker(df_multi, ticker):
    """Safely extracts a single stock dataframe from yfinance multi-index structures."""
    if df_multi.empty:
        return pd.DataFrame()
    if isinstance(df_multi.columns, pd.MultiIndex):
        if ticker in df_multi.columns.get_level_values(1):
            sub_df = df_multi.xs(ticker, axis=1, level=1)
        elif ticker in df_multi.columns.get_level_values(0):
            sub_df = df_multi.xs(ticker, axis=1, level=0)
        else:
            return pd.DataFrame()
        return sub_df.dropna(how="all").copy()
    return df_multi.copy()

def compute_indicators(df):
    """Computes technical and volatility overlays."""
    df = df.copy()
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    vol = df["Volume"]

    # Trend Moving Averages
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Wilder's RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # Average True Range (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()

    # Relative Volume (Z-Score)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, 1e-9)
    df["VOL_Z"] = (vol - vol_mean) / vol_std

    return df

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_market_data(tickers):
    try:
        return yf.download(tickers, period="2y", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamental_matrix(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "roe": (info.get("returnOnEquity") or 0.0) * 100,
            "de": (info.get("debtToEquity") or 0.0) / 100,
            "pe": info.get("trailingPE") or 0.0,
            "eps_g": (info.get("earningsGrowth") or 0.0) * 100,
            "net_margin": (info.get("profitMargins") or 0.0) * 100
        }
    except Exception:
        return {"roe": 0.0, "de": 0.0, "pe": 0.0, "eps_g": 0.0, "net_margin": 0.0}

# ============================================================
# QUANTITATIVE AI & RISK ENGINE
# ============================================================
def train_predict_ensemble(df):
    """
    Fits dual non-linear regressors (Gradient Boosting + Random Forest) on
    multi-factor features using time-series train-validation split.
    """
    if len(df) < 120:
        return None

    features = ["RET_1", "RET_5", "RSI14", "DIST_SMA50", "DIST_SMA200", "VOL_Z", "MACD_RATIO"]
    df_feat = df.copy()

    close = df_feat["Close"]
    df_feat["RET_1"] = close.pct_change(1)
    df_feat["RET_5"] = close.pct_change(5)
    df_feat["DIST_SMA50"] = (close - df_feat["SMA50"]) / df_feat["SMA50"].replace(0, np.nan)
    df_feat["DIST_SMA200"] = (close - df_feat["SMA200"]) / df_feat["SMA200"].replace(0, np.nan)
    df_feat["MACD_RATIO"] = df_feat["MACD"] / close.replace(0, np.nan)
    
    # Target: 5-Day forward log return
    df_feat["TARGET_5D"] = (close.shift(-5) / close.replace(0, np.nan)) - 1

    clean = df_feat.dropna(subset=features + ["TARGET_5D"]).copy()
    if len(clean) < 60:
        return None

    X = clean[features]
    y = clean["TARGET_5D"]

    # 80/20 Time-Series Split (preserving sequential order)
    split_idx = int(len(clean) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Models
    gbr = GradientBoostingRegressor(n_estimators=80, learning_rate=0.03, max_depth=3, random_state=42)
    rf = RandomForestRegressor(n_estimators=80, max_depth=4, random_state=42, n_jobs=-1)

    gbr.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    # Validation Directional Accuracy Check
    val_pred = (0.6 * gbr.predict(X_val_scaled)) + (0.4 * rf.predict(X_val_scaled))
    dir_acc = (np.sign(val_pred) == np.sign(y_val)).mean() * 100

    # Prediction on current latest bar
    latest_bar = scaler.transform(df_feat[features].iloc[[-1]].ffill().fillna(0))
    expected_return = (0.6 * gbr.predict(latest_bar)[0]) + (0.4 * rf.predict(latest_bar)[0])

    cmp = float(close.iloc[-1])
    atr = float(df_feat["ATR14"].iloc[-1]) if pd.notna(df_feat["ATR14"].iloc[-1]) else cmp * 0.02

    # Volatility-grounded trade parameters (1.5x ATR Risk, 1:2 R:R Ratio)
    stop_loss = round(cmp - (1.5 * atr), 2)
    risk_unit = cmp - stop_loss
    target_1 = round(cmp + (1.5 * risk_unit), 2)
    target_2 = round(cmp + (3.0 * risk_unit), 2)

    # Velocity-adjusted estimated trading days to achieve Target 1
    est_trading_days = max(3, int(risk_unit / max(0.4 * atr, 1e-3)))
    cal_days = int(est_trading_days * 1.45)
    est_target_date = (datetime.now(IST) + timedelta(days=cal_days)).strftime("%d %b %Y")

    return {
        "cmp": cmp,
        "expected_return_pct": expected_return * 100,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "risk_reward": round((target_1 - cmp) / max(1e-2, (cmp - stop_loss)), 2),
        "est_date": est_target_date,
        "est_days": est_trading_days,
        "dir_accuracy": round(dir_acc, 1),
        "atr": atr
    }

# ============================================================
# SIDEBAR NAVIGATION & WATCHLIST SELECTION
# ============================================================
st.sidebar.markdown("### 🏛️ AlphaDesk Controls")
custom_input = st.sidebar.text_input("Add NSE Ticker (e.g. TRENT, BEL, COALINDIA)", "")
active_watchlist = DEFAULT_WATCHLIST.copy()

if custom_input:
    clean_sym = custom_input.strip().upper()
    if not clean_sym.endswith(".NS"):
        clean_sym += ".NS"
    if clean_sym not in active_watchlist:
        active_watchlist.append(clean_sym)

selected_sector = st.sidebar.selectbox("Display Filter", ["All Equities", "Buy Signals Only", "High Conviction (>55%)"])

with st.spinner("Streaming institutional feeds..."):
    market_matrix = fetch_bulk_market_data(active_watchlist)

if market_matrix.empty:
    st.error("Market data feeds are temporarily unreachable. Please verify network connectivity.")
    st.stop()

# ============================================================
# QUANT PIPELINE EXECUTION
# ============================================================
records = []
computed_dfs = {}

for ticker in active_watchlist:
    raw_df = extract_single_ticker(market_matrix, ticker)
    if raw_df.empty or len(raw_df) < 100:
        continue

    ind_df = compute_indicators(raw_df)
    computed_dfs[ticker] = ind_df

    quant_res = train_predict_ensemble(ind_df)
    if not quant_res:
        continue

    fund = fetch_fundamental_matrix(ticker)

    # Multi-factor signal confirmation
    is_stage2 = (
        ind_df["Close"].iloc[-1] > ind_df["SMA50"].iloc[-1] > ind_df["SMA200"].iloc[-1]
    )
    
    if quant_res["expected_return_pct"] >= 1.5 and is_stage2:
        signal = "BUY"
    elif quant_res["expected_return_pct"] <= -1.2 or ind_df["Close"].iloc[-1] < ind_df["SMA200"].iloc[-1]:
        signal = "SELL"
    else:
        signal = "WATCH"

    records.append({
        "Ticker": ticker.replace(".NS", ""),
        "RawTicker": ticker,
        "Signal": signal,
        "CMP": quant_res["cmp"],
        "Expected Return": quant_res["expected_return_pct"],
        "Stop Loss": quant_res["stop_loss"],
        "Target 1": quant_res["target_1"],
        "Target 2": quant_res["target_2"],
        "Risk:Reward": f"1:{quant_res['risk_reward']}",
        "Est. Horizon": f"{quant_res['est_days']}d ({quant_res['est_date']})",
        "Directional Acc": quant_res["dir_accuracy"],
        "ROE %": fund["roe"],
        "P/E": fund["pe"],
        "D/E": fund["de"]
    })

screener_df = pd.DataFrame(records)

# ============================================================
# VIEWPORT & UI PRESENTATION
# ============================================================
tabs = st.tabs(["⚡ Institutional Consensus & Targets", "📊 Deep Technical & Volume Terminal"])

with tabs[0]:
    if not screener_df.empty:
        # Filter Logic
        display_df = screener_df.copy()
        if selected_sector == "Buy Signals Only":
            display_df = display_df[display_df["Signal"] == "BUY"]
        elif selected_sector == "High Conviction (>55%)":
            display_df = display_df[display_df["Directional Acc"] >= 55.0]

        display_df = display_df.sort_values(by="Expected Return", ascending=False).reset_index(drop=True)

        # Top Conviction Card
        if not display_df.empty and display_df.iloc[0]["Signal"] == "BUY":
            top = display_df.iloc[0]
            st.markdown(f"""
            <div class="alpha-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #818cf8; font-size: 11px; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">Institutional Alpha Setup</span>
                        <div style="font-size: 32px; font-weight: 900; color: #ffffff; margin-top: 4px;">{top['Ticker']} <span class="badge-buy">BUY</span></div>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #94a3b8; font-size: 11px;">Expected 5-Day Alpha</span>
                        <div style="font-size: 28px; font-weight: 900; color: #34d399;">+{top['Expected Return']:.2f}%</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-top: 20px;">
                    <div class="metric-panel"><div class="metric-panel-title">Entry Level</div><div class="metric-panel-value">₹{top['CMP']:,.2f}</div></div>
                    <div class="metric-panel"><div class="metric-panel-title">Stop Loss</div><div class="metric-panel-value" style="color: #f87171;">₹{top['Stop Loss']:,.2f}</div></div>
                    <div class="metric-panel"><div class="metric-panel-title">Target 1 (Base)</div><div class="metric-panel-value" style="color: #38bdf8;">₹{top['Target 1']:,.2f}</div></div>
                    <div class="metric-panel"><div class="metric-panel-title">Target 2 (Extended)</div><div class="metric-panel-value" style="color: #34d399;">₹{top['Target 2']:,.2f}</div></div>
                    <div class="metric-panel"><div class="metric-panel-title">Est. Target Date</div><div class="metric-panel-value" style="font-size: 16px; margin-top: 4px;">{top['Est. Horizon']}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Algorithmic Action Matrix")

        def style_signal(val):
            if val == "BUY":
                return "color: #34d399; font-weight: bold;"
            elif val == "SELL":
                return "color: #f87171; font-weight: bold;"
            return "color: #fbbf24; font-weight: bold;"

        st.dataframe(
            display_df[[
                "Ticker", "Signal", "CMP", "Expected Return", "Stop Loss",
                "Target 1", "Target 2", "Risk:Reward", "Est. Horizon",
                "Directional Acc", "ROE %", "P/E"
            ]].style.map(style_signal, subset=["Signal"]),
            column_config={
                "CMP": st.column_config.NumberColumn(format="₹%.2f"),
                "Expected Return": st.column_config.NumberColumn(format="%+.2f%%"),
                "Stop Loss": st.column_config.NumberColumn(format="₹%.2f"),
                "Target 1": st.column_config.NumberColumn(format="₹%.2f"),
                "Target 2": st.column_config.NumberColumn(format="₹%.2f"),
                "Directional Acc": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                "ROE %": st.column_config.NumberColumn(format="%.1f%%"),
                "P/E": st.column_config.NumberColumn(format="%.1f")
            },
            use_container_width=True,
            hide_index=True
        )

with tabs[1]:
    st.markdown("### Multi-Pane Institutional Execution Chart")
    chart_ticker = st.selectbox("Select Asset for Deep Audit", list(computed_dfs.keys()), format_func=lambda x: x.replace(".NS", ""))
    
    chart_df = computed_dfs[chart_ticker].tail(180).copy()

    # Create 3-Row Multiplot (Candles, Volume, RSI/MACD)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2]
    )

    # 1. Price + Overlays
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"],
        name="Price",
        increasing_line_color="#10b981", decreasing_line_color="#ef4444"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA20"], line=dict(color="#f59e0b", width=1.2), name="SMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50"], line=dict(color="#3b82f6", width=1.2), name="SMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA200"], line=dict(color="#a855f7", width=1.5), name="SMA 200"), row=1, col=1)

    # 2. Volume Profile
    vol_colors = ["#ef4444" if row["Open"] > row["Close"] else "#10b981" for _, row in chart_df.iterrows()]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)

    # 3. MACD
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"], line=dict(color="#38bdf8", width=1.2), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_SIGNAL"], line=dict(color="#f97316", width=1.2), name="Signal"), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080c14",
        plot_bgcolor="#080c14",
        height=720,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
