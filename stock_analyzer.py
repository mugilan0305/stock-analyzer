import streamlit as st
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
    """Calculates the expected date to hit a target based on ATR momentum velocity."""
    if pd.isna(entry) or pd.isna(target) or pd.isna(atr) or atr <= 0 or target <= entry:
        return "N/A"
    # Assume the stock moves 40% of its ATR directionally per trading day on average
    trading_days = max(1, int(abs(target - entry) / (atr * 0.4)))
    # Convert trading days to calendar days (multiply by approx 1.4)
    calendar_days = int(trading_days * 1.4)
    est_date = datetime.now(IST) + timedelta(days=calendar_days)
    return est_date.strftime("%d %b %Y")

def get_stock_df(bulk_df, ticker):
    """Safely extracts a single stock's dataframe regardless of yfinance version."""
    if isinstance(bulk_df.columns, pd.MultiIndex):
        if ticker in bulk_df.columns.get_level_values(0):
            return bulk_df[ticker]
        elif ticker in bulk_df.columns.get_level_values(1):
            return bulk_df.xs(ticker, axis=1, level=1)
    return bulk_df

def get_price_df(bulk_df, price_type):
    """Safely extracts all stocks for a specific metric (Close, High, etc)."""
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
    """Institutional Gradient Boosting + Volatility Ensemble with robust NaN and division-by-zero handling."""
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
        df["RET_20"] =
