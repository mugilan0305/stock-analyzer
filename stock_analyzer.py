import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import html
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import quote
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# APP CONFIG & DARK THEME
# ============================================================

st.set_page_config(
    page_title="Institutional Quantitative Terminal",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background: #0b1120; color: #f8fafc; }
.block-container { max-width: 1440px; padding-top: 1.2rem; padding-bottom: 3rem; }
.card { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.badge-buy { background: #064e3b; color: #34d399; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; text-align: center; }
.badge-watch { background: #451a03; color: #fbbf24; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; text-align: center; }
.badge-avoid { background: #27272a; color: #a1a1aa; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; text-align: center; }
.badge-sell { background: #450a0a; color: #f87171; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; text-align: center; }
.metric-val { font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 4px; }
.metric-lbl { font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
.live-badge { display: inline-flex; align-items: center; background: #0f2e1b; border: 1px solid #10b981; color: #34d399; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 12px; }
.pulse-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; display: inline-block; margin-right: 6px; }
.stock-box { background: #111827; border: 1px solid #1e293b; border-radius: 14px; padding: 22px; margin-bottom: 18px; }
.price { font-size: 38px; font-weight: 800; margin-top: 8px; }
.positive { color: #22c55e; font-weight: 700; }
.negative { color: #ef4444; font-weight: 700; }
.top-pick-card { background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%); border: 2px solid #818cf8; border-radius: 14px; padding: 22px; margin-bottom: 20px; }
.news-box { background: #0a0f1d; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; margin-bottom: 10px; transition: 0.2s; }
.news-box:hover { border-color: #38bdf8; background: #111827; }
</style>
""", unsafe_allow_html=True)

IST = ZoneInfo("Asia/Kolkata")

WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS",
    "ITC.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "AXISBANK.NS",
    "MARUTI.NS", "KOTAKBANK.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "TATASTEEL.NS", "NTPC.NS", "M&M.NS", "POWERGRID.NS", "TRENT.NS", "HAL.NS", "BEL.NS"
]

# ============================================================
# UTILITIES & SIGNAL FORMATTING
# ============================================================

def safe_float(value, default=0.0):
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except Exception:
        return default

def money(value):
    val = safe_float(value, np.nan)
    return f"₹{val:,.2f}" if np.isfinite(val) else "N/A"

def get_signal_badge(signal_text):
    if "BUY" in signal_text.upper():
        return f'<span class="badge-buy">{signal_text}</span>'
    elif "SELL" in signal_text.upper():
        return f'<span class="badge-sell">{signal_text}</span>'
    elif "WATCH" in signal_text.upper():
        return f'<span class="badge-watch">{signal_text}</span>'
    else:
        return f'<span class="badge-avoid">{signal_text}</span>'

def estimate_target_date(entry, target, atr):
    try:
        if pd.isna(entry) or pd.isna(target) or pd.isna(atr) or atr <= 0 or entry == target:
            return "N/A"
        trading_days = max(1, int(abs(target - entry) / (atr * 0.4)))
        calendar_days = int(trading_days * 1.4)
        est_date = datetime.now(IST) + timedelta(days=calendar_days)
        return est_date.strftime("%d %b %y")
    except Exception:
        return "N/A"

def get_stock_df(bulk_df, ticker):
    if isinstance(bulk_df.columns, pd.MultiIndex):
        if ticker in bulk_df.columns.get_level_values(0):
            return bulk_df[ticker]
        elif ticker in bulk_df.columns.get_level_values(1):
            return bulk_df.xs(ticker, axis=1, level=1)
    return bulk_df

def get_price_df(bulk_df, price_type):
    if isinstance(bulk_df.columns, pd.MultiIndex):
        if price_type in bulk_df.columns.get_level_values(0):
            return bulk_df[price_type]
        elif price_type in bulk_df.columns.get_level_values(1):
            return bulk_df.xs(price_type, axis=1, level=1)
    return bulk_df

# ============================================================
# DATA FETCHING PIPELINE
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamental_metrics(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        eps_g = info.get("earningsGrowth", None)
        rev_g = info.get("revenueGrowth", None)
        roe = info.get("returnOnEquity", None)
        de = info.get("debtToEquity", None)
        pe = info.get("trailingPE", None)
        pm = info.get("profitMargins", None)
        
        peg = None
        if pe is not None and eps_g is not None and eps_g > 0:
            peg = pe / (eps_g * 100)

        return {
            "eps_growth": (eps_g * 100) if eps_g is not None and np.isfinite(eps_g) else None,
            "rev_growth": (rev_g * 100) if rev_g is not None and np.isfinite(rev_g) else None,
            "roe": (roe * 100) if roe is not None and np.isfinite(roe) else None,
            "debt_to_equity": (de / 100) if de is not None and np.isfinite(de) else None,
            "pe": pe if pe is not None and np.isfinite(pe) else None,
            "peg": peg if peg is not None and np.isfinite(peg) else None,
            "profit_margin": (pm * 100) if pm is not None and np.isfinite(pm) else None
        }
    except Exception:
        return {"eps_growth": None, "rev_growth": None, "roe": None, "debt_to_equity": None, "pe": None, "peg": None, "profit_margin": None}

def get_live_quote(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker)}?interval=1m&range=1d"
        res = requests.get(url, timeout=3, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200:
            meta = res.json()["chart"]["result"][0]["meta"]
            p = safe_float(meta.get("regularMarketPrice"), np.nan)
            prev = safe_float(meta.get("previousClose"), np.nan)
            if np.isfinite(p):
                chg = ((p / prev) - 1) * 100 if np.isfinite(prev) and prev > 0 else 0.0
                return {"price": p, "change": chg, "prev_close": prev, "time": datetime.now(IST)}
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_market_data(tickers):
    try:
        return yf.download(tickers + ["^NSEI"], period="2y", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol):
    try:
        query = quote(f"{symbol} India stock nse market")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return []
        root = ET.fromstring(response.content)
        articles = []
        for item in root.findall(".//item")[:5]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            date = item.findtext("pubDate", "")
            if title:
                articles.append({
                    "title": html.unescape(title),
                    "link": link,
                    "date": date,
                })
        return articles
    except Exception:
        return []

# ============================================================
# ALGORITHMIC & AI MODELS
# ============================================================

def evaluate_institutional_models(fund_data, tech_data):
    roe = fund_data.get("roe") or 0.0
    de = fund_data.get("debt_to_equity") or 0.0
    margin = fund_data.get("profit_margin") or 0.0
    eps_g = fund_data.get("eps_growth") or 0.0
    rev_g = fund_data.get("rev_growth") or 0.0
    pe = fund_data.get("pe") or 0.0
    peg = fund_data.get("peg") or 99.0
    is_stage2 = tech_data.get("is_stage2", False)

    s_growth = 0
    if roe >= 20: s_growth += 35
    elif roe >= 15: s_growth += 20
    if eps_g >= 18: s_growth += 35
    elif eps_g >= 10: s_growth += 20
    if rev_g >= 15: s_growth += 20
    if is_stage2: s_growth += 10

    s_moat = 0
    if de <= 0.2: s_moat += 40
    elif de <= 0.5: s_moat += 25
    elif de > 1.0: s_moat -= 20
    if margin >= 18: s_moat += 35
    elif margin >= 12: s_moat += 20
    if roe >= 18: s_moat += 25

    s_garp = 0
    if roe >= 20: s_garp += 30
    if eps_g >= 15 and rev_g >= 12: s_garp += 30
    if 0 < peg <= 1.5: s_garp += 30
    elif 1.5 < peg <= 2.2: s_garp += 15
    if is_stage2: s_garp += 10

    s_mom = 0
    if eps_g >= 22: s_mom += 40
    elif eps_g >= 12: s_mom += 20
    if de <= 0.6: s_mom += 25
    if is_stage2: s_mom += 35

    s_value = 0
    if 0 < pe <= 25: s_value += 40
    elif 25 < pe <= 35: s_value += 20
    if de <= 0.3: s_value += 30
    if roe >= 16: s_value += 20
    if eps_g > 0: s_value += 10

    s_growth = max(0, min(100, s_growth))
    s_moat = max(0, min(100, s_moat))
    s_garp = max(0, min(100, s_garp))
    s_mom = max(0, min(100, s_mom))
    s_value = max(0, min(100, s_value))

    composite = round((0.25 * s_growth) + (0.25 * s_moat) + (0.20 * s_garp) + (0.15 * s_mom) + (0.15 * s_value), 1)

    return {
        "Composite": composite,
        "Growth_Scale": s_growth,
        "Fortress_Moat": s_moat,
        "GARP": s_garp,
        "Momentum": s_mom,
        "Deep_Value": s_value
    }

@st.cache_data(ttl=900, show_spinner=False)
def run_advanced_ai_model(df_stock):
    try:
        if df_stock is None or df_stock.empty: return None
        df = df_stock.copy()
        if "Close" not in df.columns or "Volume" not in df.columns: return None
        
        df = df.dropna(subset=["Close"]).copy()
        if len(df) < 50: return None
        
        close = df["Close"]
        high = df["High"] if "High" in df.columns else close
        low = df["Low"] if "Low" in df.columns else close
        vol = df["Volume"]

        df["RET_1"] = close.pct_change(1).fillna(0)
        df["RET_5"] = close.pct_change(5).fillna(0)
        df["RET_20"] = close.pct_change(20).fillna(0)
        df["SMA20"] = close.rolling(20).mean()
        df["SMA50"] = close.rolling(50).mean()
        
        df["DIST_SMA20"] = (close - df["SMA20"]) / df["SMA20"].replace(0, np.nan)
        df["DIST_SMA50"] = (close - df["SMA50"]) / df["SMA50"].replace(0, np.nan)
        
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["ATR_RATIO"] = df["ATR"] / close.replace(0, np.nan)

        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std().replace(0, np.nan)
        df["VOL_Z"] = (vol - vol_mean) / vol_std
        
        df["TARGET_5D"] = (close.shift(-5) / close.replace(0, np.nan)) - 1

        features = ["RET_1", "RET_5", "RET_20", "DIST_SMA20", "DIST_SMA50", "ATR_RATIO", "VOL_Z"]
        for f in features:
            df[f] = df[f].ffill().fillna(0)

        dataset = df.dropna(subset=["TARGET_5D"]).copy()
        if len(dataset) < 40: return None

        X = dataset[features]
        y = dataset["TARGET_5D"]

        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, random_state=42)
        gbr.fit(X, y)

        rf = RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        latest_features = df[features].iloc[[-1]]
        pred_gbr = gbr.predict(latest_features)[0]
        pred_rf = rf.predict(latest_features)[0]
        
        expected_5d_ret = (0.65 * pred_gbr) + (0.35 * pred_rf)
        predicted_price = float(close.iloc[-1]) * (1 + expected_5d_ret)
        
        disagreement = abs(pred_gbr - pred_rf)
        confidence = max(40, min(92, int(85 - (disagreement * 400))))

        return {
            "expected_return_pct": expected_5d_ret * 100,
            "predicted_target": predicted_price,
            "confidence": confidence,
            "model_type": "Institutional Dual Ensemble"
        }
    except Exception:
        return None

# ============================================================
# APP UI & NAVIGATION
# ============================================================

st.sidebar.title("🏛️ Terminal Navigation")

# FIX: Use components.html instead of markdown for JS execution in Streamlit
clock_html = """
<div style="background:#0f2e1b; border:1px solid #10b981; border-radius:8px; padding:12px; text-align:center; font-family: sans-serif; color: white;">
    <div style="font-size:11px; color:#34d399; font-weight:700; margin-bottom:4px; letter-spacing:1px;">LIVE MARKET CLOCK (IST)</div>
    <div id="live-clock" style="font-size:18px; font-family:monospace; font-weight:bold;">Loading...</div>
</div>
<script>
    setInterval(() => {
        let options = { timeZone: 'Asia/Kolkata', hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' };
        document.getElementById('live-clock').innerText = new Date().toLocaleTimeString('en-IN', options);
    }, 1000);
</script>
"""
with st.sidebar:
    components.html(clock_html, height=80)

nav_mode = st.sidebar.radio(
    "Select Operating View",
    [
        "🏆 Institutional Consensus Matrix",
        "⚡ 5-Second Real-Time Pulse & AI",
        "🔍 Single Stock Target Diagnosis"
    ]
)

with st.spinner("Downloading market matrices..."):
    bulk_data = fetch_all_market_data(WATCHLIST)

if bulk_data.empty:
    st.error("Market data feeds are temporarily unreachable. Please refresh.")
    st.stop()

closes = get_price_df(bulk_data, "Close")
highs = get_price_df(bulk_data, "High")
lows = get_price_df(bulk_data, "Low")

# ============================================================
# VIEW 1: INSTITUTIONAL CONSENSUS MATRIX
# ============================================================
if nav_mode == "🏆 Institutional Consensus Matrix":
    st.title("🏆 Institutional Consensus Matrix")
    st.markdown('<p style="color:#94a3b8;">Audits Indian large caps across 5 core quantitative methodologies. Includes explicit Buy/Sell/Avoid signals based on technical alignment and institutional scoring.</p>', unsafe_allow_html=True)

    matrix_records = []
    
    for s in WATCHLIST:
        if s not in closes.columns: continue
        c = closes[s].dropna()
        h = highs[s].dropna()
        l = lows[s].dropna()
        if len(c) < 100: continue
        
        cmp = float(c.iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        sma150 = float(c.rolling(150).mean().iloc[-1]) if len(c) > 150 else cmp
        sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) > 200 else cmp
        high52 = float(h.tail(252).max())
        low52 = float(l.tail(252).min())
        
        is_stage2 = (cmp > sma150) and (cmp > sma200) and (sma150 > sma200) and (cmp > sma50) and (cmp >= 1.25 * low52) and (cmp >= 0.75 * high52)
        
        pivot_20d = float(h.iloc[-21:-1].max()) if len(h) >= 21 else cmp
        atr14 = float((h - l).rolling(14).mean().iloc[-1]) if len(h) >= 14 else (cmp * 0.02)
        
        fund = get_fundamental_metrics(s)
        scores = evaluate_institutional_models(fund, {"is_stage2": is_stage2})
        
        entry = cmp if is_stage2 else pivot_20d
        stop_loss = round(max(entry * 0.94, entry - (1.4 * atr14)), 2)
        risk = entry - stop_loss
        
        target_1 = round(entry + (1.0 * risk), 2)
        target_2 = round(entry + (2.0 * risk), 2)
        target_3 = round(entry + (3.0 * risk), 2)
        
        t1_date = estimate_target_date(entry, target_1, atr14)
        t2_date = estimate_target_date(entry, target_2, atr14)
        t3_date = estimate_target_date(entry, target_3, atr14)

        if is_stage2 and scores["Composite"] >= 65:
            signal_out = "BUY"
        elif cmp < sma200 or scores["Composite"] < 40:
            signal_out = "SELL"
        elif not is_stage2:
            signal_out = "AVOID"
        else:
            signal_out = "WATCH"
            
        raw_df = get_stock_df(bulk_data, s)
        ai_res = run_advanced_ai_model(raw_df)
        pred_acc = ai_res["confidence"] if ai_res else 0

        matrix_records.append({
            "Symbol": s.replace(".NS", ""),
            "Signal": signal_out,
            "CMP": cmp,
            "Consensus Score": scores["Composite"],
            "Prediction Accuracy %": pred_acc,
            "Growth & Scale": scores["Growth_Scale"],
            "Fortress Moat": scores["Fortress_Moat"],
            "GARP": scores["GARP"],
            "Momentum": scores["Momentum"],
            "Deep Value": scores["Deep_Value"],
            "ROE %": fund["roe"],
            "D/E": fund["debt_to_equity"],
            "P/E": fund["pe"],
            "Entry": entry,
            "Stop Loss": stop_loss,
            "Target 1": target_1,
            "Target 2": target_2,
            "Target 3": target_3,
            "T1 Date": t1_date,
            "T2 Date": t2_date,
            "T3 Date": t3_date
        })

    consensus_df = pd.DataFrame(matrix_records).sort_values(by="Consensus Score", ascending=False).reset_index(drop=True)

    if not consensus_df.empty:
        top_pick = consensus_df.iloc[0]
        
        st.markdown(f"""
        <div class="top-pick-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:12px; color:#a5b4fc; text-transform:uppercase; font-weight:700; letter-spacing:1px;">🏆 Consensus Top Institutional Candidate</span>
                    <div style="font-size:32px; font-weight:900; color:#ffffff; margin-top:2px;">{top_pick['Symbol']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:28px; font-weight:900; color:#38bdf8;">{top_pick['Consensus Score']} / 100</div>
                    {get_signal_badge(top_pick['Signal'])}
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(6, 1fr); gap:12px; margin-top:18px;">
                <div><div class="metric-lbl">Optimal Entry</div><div class="metric-val">{money(top_pick['Entry'])}</div></div>
                <div><div class="metric-lbl">Stop Loss</div><div class="metric-val" style="color:#f87171;">{money(top_pick['Stop Loss'])}</div></div>
                <div><div class="metric-lbl">Target 1 (Est: {top_pick['T1 Date']})</div><div class="metric-val">{money(top_pick['Target 1'])}</div></div>
                <div><div class="metric-lbl">Target 2 (Est: {top_pick['T2 Date']})</div><div class="metric-val" style="color:#34d399;">{money(top_pick['Target 2'])}</div></div>
                <div><div class="metric-lbl">Target 3 (Est: {top_pick['T3 Date']})</div><div class="metric-val" style="color:#059669;">{money(top_pick['Target 3'])}</div></div>
                <div><div class="metric-lbl">Prediction Accuracy</div><div class="metric-val" style="color:#fbbf24;">{top_pick['Prediction Accuracy %']}%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Cross-Strategy Comparison Grid")
    
    def highlight_signal(val):
        if val == 'BUY': return 'color: #34d399; font-weight: bold'
        elif val == 'SELL': return 'color: #f87171; font-weight: bold'
        elif val == 'WATCH': return 'color: #fbbf24; font-weight: bold'
        return 'color: #a1a1aa'
        
    styled_df = consensus_df[["Symbol", "Signal", "CMP", "Consensus Score", "Prediction Accuracy %", "Growth & Scale", "Fortress Moat", "GARP", "Momentum", "Deep Value", "ROE %", "D/E"]]
    
    st.dataframe(
        styled_df.style.map(highlight_signal, subset=['Signal']),
        column_config={
            "CMP": st.column_config.NumberColumn(format="₹%.2f"),
            "Consensus Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
            "Prediction Accuracy %": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
            "ROE %": st.column_config.NumberColumn(format="%.1f%%"),
            "D/E": st.column_config.NumberColumn(format="%.2f")
        },
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# VIEW 2: 5-SECOND REAL-TIME PULSE & AI
# ============================================================
elif nav_mode == "⚡ 5-Second Real-Time Pulse & AI":
    st.title("⚡ Real-Time Market Pulse & AI Predictor")
    st.markdown('<p style="color:#94a3b8;">Sub-minute quotes via isolated UI fragments combined with quantitative Gradient Boosting models.</p>', unsafe_allow_html=True)
    
    selected_stock = st.selectbox("Select Active Stock to Stream", [s.replace(".NS", "") for s in WATCHLIST])
    ticker_sym = f"{selected_stock}.NS"
    raw_df = get_stock_df(bulk_data, ticker_sym)

    @st.fragment(run_every="5s")
    def live_stream_widget(symbol, t_symbol, hist_data):
        quote_data = get_live_quote(t_symbol)
        
        close_series = hist_data["Close"].dropna()
        last_close = float(close_series.iloc[-1]) if not close_series.empty else 0.0
        
        if quote_data:
            cmp = quote_data["price"]
            day_chg = quote_data["change"]
            refresh_time = quote_data["time"].strftime("%I:%M:%S %p IST")
        else:
            cmp = last_close
            day_chg = 0.0
            refresh_time = datetime.now(IST).strftime("%I:%M:%S %p IST")

        p_class = "positive" if day_chg >= 0 else "negative"
        arrow = "↑" if day_chg >= 0 else "↓"

        st.markdown(f"""
        <div class="stock-box">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="stock-name">{symbol}</span>
                    <span style="color:#94a3b8; font-size:13px; margin-left:8px;">NSE: {t_symbol}</span>
                </div>
                <div class="live-badge"><span class="pulse-dot"></span>LIVE 5s POLLING</div>
            </div>
            <div class="price">{money(cmp)}</div>
            <div class="{p_class}" style="font-size:18px;">{arrow} {day_chg:+.2f}% Today</div>
            <div style="color:#64748b; font-size:11px; margin-top:8px;">Last Synced Data: {refresh_time}</div>
        </div>
        """, unsafe_allow_html=True)

    live_stream_widget(selected_stock, ticker_sym, raw_df)
        
    st.subheader("🤖 Gradient Boosted Expected Return")
    with st.spinner("Executing volatility modeling..."):
        ai_res = run_advanced_ai_model(raw_df)

    if ai_res:
        ret_pct = ai_res['expected_return_pct']
        
        if ret_pct >= 1.5:
            ai_signal = "BUY"
        elif ret_pct <= -1.0:
            ai_signal = "SELL"
        elif ret_pct < 0:
            ai_signal = "AVOID"
        else:
            ai_signal = "WATCH"

        st.markdown(f"### AI Momentum Signal: {get_signal_badge(ai_signal)}", unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted 5-Day Target", money(ai_res["predicted_target"]), f"{ret_pct:+.2f}% Expected")
        m2.metric("Prediction Accuracy", f"{ai_res['confidence']}%")
        m3.metric("Architecture", ai_res["model_type"])
    else:
        st.info("Insufficient bar depth to fit dual ensemble.")
        
    c1, c2 = st.columns([2, 1])
    
    with c1:
        if "Close" in raw_df.columns:
            st.line_chart(raw_df["Close"].dropna().tail(180))
        
    with c2:
        st.subheader("📰 Recent News & Catalysts")
        news_items = get_news(selected_stock)
        if news_items:
            for item in news_items[:4]:
                st.markdown(f"""
                <div class="news-box">
                    <div style="font-weight:700; font-size:13px; color:#f8fafc;">{item['title']}</div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:4px;">{item['date']}</div>
                    <a href="{item['link']}" target="_blank" style="color:#38bdf8; font-size:12px; font-weight:600; text-decoration:none;">Read full article →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No recent news found for this stock.")


# =
