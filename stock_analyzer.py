
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
import re
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from urllib.parse import quote

# ============================================================
# Indian Stock AI Analyzer v2
# Technical + multi-horizon ML + market regime + news sentiment
# + event-risk flags + Nifty 10/50/100 scanner
# ============================================================

st.set_page_config(
    page_title="Indian Stock AI Analyzer",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Universe
# -----------------------------
NIFTY_50 = {
    "ADANIENT":"Adani Enterprises","ADANIPORTS":"Adani Ports","APOLLOHOSP":"Apollo Hospitals",
    "ASIANPAINT":"Asian Paints","AXISBANK":"Axis Bank","BAJAJ-AUTO":"Bajaj Auto",
    "BAJFINANCE":"Bajaj Finance","BAJAJFINSV":"Bajaj Finserv","BEL":"Bharat Electronics",
    "BHARTIARTL":"Bharti Airtel","CIPLA":"Cipla","COALINDIA":"Coal India",
    "DRREDDY":"Dr Reddy's Laboratories","EICHERMOT":"Eicher Motors","ETERNAL":"Eternal",
    "GRASIM":"Grasim Industries","HCLTECH":"HCL Technologies","HDFCBANK":"HDFC Bank",
    "HDFCLIFE":"HDFC Life","HEROMOTOCO":"Hero MotoCorp","HINDALCO":"Hindalco",
    "HINDUNILVR":"Hindustan Unilever","ICICIBANK":"ICICI Bank","INDUSINDBK":"IndusInd Bank",
    "INFY":"Infosys","ITC":"ITC","JINDALSTEL":"Jindal Steel & Power","JSWSTEEL":"JSW Steel",
    "KOTAKBANK":"Kotak Mahindra Bank","LT":"Larsen & Toubro","M&M":"Mahindra & Mahindra",
    "MARUTI":"Maruti Suzuki","MAXHEALTH":"Max Healthcare","NESTLEIND":"Nestle India",
    "NTPC":"NTPC","ONGC":"ONGC","POWERGRID":"Power Grid","RELIANCE":"Reliance Industries",
    "SBILIFE":"SBI Life","SBIN":"State Bank of India","SHRIRAMFIN":"Shriram Finance",
    "SUNPHARMA":"Sun Pharma","TATACONSUM":"Tata Consumer","TATAMOTORS":"Tata Motors",
    "TATASTEEL":"Tata Steel","TCS":"TCS","TECHM":"Tech Mahindra","TITAN":"Titan",
    "TRENT":"Trent","ULTRACEMCO":"UltraTech Cement"
}
NIFTY_NEXT_50 = {
    "ABB":"ABB India","ADANIENSOL":"Adani Energy Solutions","ADANIGREEN":"Adani Green Energy",
    "AMBUJACEM":"Ambuja Cements","BAJAJHLDNG":"Bajaj Holdings","BANKBARODA":"Bank of Baroda",
    "BERGEPAINT":"Berger Paints","BOSCHLTD":"Bosch","CANBK":"Canara Bank","CHOLAFIN":"Cholamandalam Investment",
    "COLPAL":"Colgate-Palmolive","DABUR":"Dabur India","DIVISLAB":"Divi's Laboratories",
    "DLF":"DLF","DMART":"Avenue Supermarts","GAIL":"GAIL","GODREJCP":"Godrej Consumer",
    "GODREJPROP":"Godrej Properties","HAL":"Hindustan Aeronautics","HAVELLS":"Havells India",
    "ICICIGI":"ICICI Lombard","ICICIPRULI":"ICICI Prudential Life","INDHOTEL":"Indian Hotels",
    "INDIGO":"InterGlobe Aviation","IOC":"Indian Oil","IRCTC":"IRCTC","IRFC":"IRFC",
    "JUBLFOOD":"Jubilant FoodWorks","LICI":"LIC","LODHA":"Macrotech Developers",
    "MARICO":"Marico","MOTHERSON":"Samvardhana Motherson","MUTHOOTFIN":"Muthoot Finance",
    "NAUKRI":"Info Edge","NHPC":"NHPC","PIDILITIND":"Pidilite Industries","PNB":"Punjab National Bank",
    "RECLTD":"REC","SAIL":"Steel Authority of India","SIEMENS":"Siemens",
    "SRF":"SRF","TORNTPHARM":"Torrent Pharmaceuticals","TVSMOTOR":"TVS Motor",
    "UNITDSPR":"United Spirits","VEDL":"Vedanta","VBL":"Varun Beverages",
    "VOLTAS":"Voltas","WIPRO":"Wipro","YESBANK":"Yes Bank"
}
UNIVERSES = {
    "Nifty 10": dict(list(NIFTY_50.items())[:10]),
    "Nifty 50": NIFTY_50,
    "Nifty 100": {**NIFTY_50, **NIFTY_NEXT_50}
}

# -----------------------------
# Lightweight sentiment engine
# -----------------------------
POSITIVE = {
    "beat","beats","strong","growth","surge","surges","gain","gains","profit","profits",
    "positive","bullish","record","recovery","outperform","buy","approval","order","orders",
    "deal","contract","dividend","upgrade","upbeat","improves","improved","optimism",
    "outlook","guidance","expansion","rebound","rally","wins","win"
}
NEGATIVE = {
    "fall","falls","drop","drops","weak","loss","losses","downgrade","negative","bearish",
    "cut","cuts","warning","decline","declines","miss","misses","risk","lawsuit","probe",
    "debt","fraud","lower","slowdown","pressure","disappointing","disappoint","crisis",
    "outflow","selloff","selling","concern","concerns","caution","weakness"
}

def sentiment_score(text):
    words = set(re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower()))
    p = len(words & POSITIVE)
    n = len(words & NEGATIVE)
    if p + n == 0:
        return 0.0
    return float(max(-1, min(1, (p - n) / (p + n))))

def sentiment_label(s):
    if s >= 0.25:
        return "🟢 Positive"
    if s <= -0.25:
        return "🔴 Negative"
    return "🟡 Neutral"

# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def history(ticker, period="3y"):
    try:
        d = yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=False, progress=False, threads=False
        )
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        for c in ["Open","High","Low","Close","Adj Close","Volume"]:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        return d.dropna(subset=["Open","High","Low","Close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def latest_price(ticker):
    d = history(ticker, "10d")
    if d.empty:
        return None, None
    return float(d["Close"].iloc[-1]), d.index[-1]

def add_indicators(d):
    d = d.copy()
    c = d["Close"]
    d["MA20"] = c.rolling(20).mean()
    d["MA50"] = c.rolling(50).mean()
    d["MA200"] = c.rolling(200).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    d["BB_Middle"] = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_Upper"] = d["BB_Middle"] + 2 * std
    d["BB_Lower"] = d["BB_Middle"] - 2 * std

    d["Return1"] = c.pct_change()
    d["Return5"] = c.pct_change(5)
    d["Return20"] = c.pct_change(20)
    d["Volatility20"] = d["Return1"].rolling(20).std()

    prev = c.shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev).abs(),
        (d["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    d["VolumeRatio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    return d

FEATURES = [
    "Close","Volume","MA20","MA50","MA200","RSI","MACD","Signal",
    "BB_Middle","BB_Upper","BB_Lower","Return5","Return20",
    "Volatility20","ATR14","VolumeRatio"
]

@st.cache_data(ttl=900, show_spinner=False)
def ml_predictions(d):
    results = {}
    clean = d.copy()
    for horizon in [1, 5, 20]:
        x = clean.copy()
        x["Target"] = x["Close"].shift(-horizon)
        x = x.dropna(subset=FEATURES + ["Target"])
        if len(x) < 180:
            results[horizon] = {"pred": None, "validation": None, "std": None}
            continue

        split = int(len(x) * 0.80)
        model = RandomForestRegressor(
            n_estimators=220, max_depth=11,
            min_samples_leaf=3, random_state=42,
            n_jobs=-1
        )
        model.fit(x[FEATURES].iloc[:split], x["Target"].iloc[:split])

        latest = clean.dropna(subset=FEATURES).iloc[-1:]
        pred = float(model.predict(latest[FEATURES])[0])

        if split < len(x):
            test_pred = model.predict(x[FEATURES].iloc[split:])
            mae = float(np.mean(np.abs(test_pred - x["Target"].iloc[split:].values)))
            base = float(np.mean(np.abs(x["Target"].iloc[split:].values)))
            validation = max(0, min(100, 100 - (mae / base * 100))) if base else 0
        else:
            validation = 0

        tree_preds = np.array([tree.predict(latest[FEATURES])[0] for tree in model.estimators_])
        results[horizon] = {
            "pred": pred,
            "validation": float(validation),
            "std": float(np.std(tree_preds))
        }
    return results

# -----------------------------
# News
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def google_news(query, limit=10):
    try:
        url = (
            "https://news.google.com/rss/search?q=" +
            quote(query) +
            "&hl=en-IN&gl=IN&ceid=IN:en"
        )
        r = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        rows = []
        for item in root.findall(".//item")[:limit]:
            title = item.findtext("title", "")
            pub = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            source = item.findtext("source", "")
            rows.append({
                "title": title,
                "date": pub,
                "link": link,
                "source": source,
                "sentiment": sentiment_score(title)
            })
        return rows
    except Exception:
        return []

def news_summary(rows):
    if not rows:
        return 0.0, "No recent headlines"
    scores = np.array([x["sentiment"] for x in rows])
    # Recent items receive slightly more weight.
    weights = np.linspace(1.0, 0.5, len(scores))
    score = float(np.average(scores, weights=weights))
    return score, sentiment_label(score)

# -----------------------------
# Market regime
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def market_regime():
    d = history("^NSEI", "1y")
    if d.empty:
        return None
    d = add_indicators(d)
    x = d.iloc[-1]
    current = float(x["Close"])
    ma20 = float(x["MA20"])
    ma50 = float(x["MA50"])
    ma200 = float(x["MA200"]) if pd.notna(x["MA200"]) else ma50
    rsi = float(x["RSI"]) if pd.notna(x["RSI"]) else 50
    score = 0
    score += 1 if current > ma20 else -1
    score += 1 if current > ma50 else -1
    score += 1 if current > ma200 else -1
    score += 1 if rsi >= 50 else -1
    if score >= 3:
        regime = "🟢 Bullish"
    elif score <= -2:
        regime = "🔴 Bearish"
    else:
        regime = "🟡 Mixed"
    return {
        "current": current, "ma20": ma20, "ma50": ma50,
        "ma200": ma200, "rsi": rsi, "score": score, "regime": regime
    }

# -----------------------------
# Events / catalysts
# -----------------------------
def event_risk(news_rows):
    text = " ".join(x["title"].lower() for x in news_rows)
    terms = [
        "earnings","results","quarterly","dividend","board meeting",
        "rbi","budget","election","acquisition","merger","approval",
        "court","order","guidance","rating","downgrade","upgrade"
    ]
    hits = [t for t in terms if t in text]
    if not hits:
        return "Low", []
    level = "High" if len(hits) >= 3 else "Medium"
    return level, hits

# -----------------------------
# Signal
# -----------------------------
def technical_score(d, current):
    x = d.iloc[-1]
    score = 0.0
    reasons = []

    if pd.notna(x["MA20"]) and pd.notna(x["MA50"]):
        if x["MA20"] > x["MA50"]:
            score += 12; reasons.append("MA20 above MA50")
        else:
            score -= 12; reasons.append("MA20 below MA50")

    if pd.notna(x["MA200"]):
        if current > x["MA200"]:
            score += 10; reasons.append("Price above MA200")
        else:
            score -= 10; reasons.append("Price below MA200")

    rsi = x["RSI"]
    if pd.notna(rsi):
        if 50 <= rsi <= 68:
            score += 10; reasons.append("RSI supports momentum")
        elif rsi < 30:
            score += 8; reasons.append("RSI oversold")
        elif rsi > 70:
            score -= 10; reasons.append("RSI overbought")
        else:
            score -= 2

    if x["MACD"] > x["Signal"]:
        score += 10; reasons.append("MACD bullish")
    else:
        score -= 10; reasons.append("MACD bearish")

    if pd.notna(x["VolumeRatio"]) and x["VolumeRatio"] > 1.2:
        score += 5; reasons.append("Volume above average")

    return float(max(-50, min(50, score))), reasons

def final_signal(tech, ml_change_5d, news_score, regime_score, event_level):
    ml = max(-50, min(50, ml_change_5d * 5))
    news = max(-25, min(25, news_score * 25))
    regime = max(-10, min(10, regime_score * 3))
    total = 0.45 * tech + 0.35 * ml + 0.15 * news + 0.05 * regime

    # High event risk does not automatically change direction,
    # but prevents a strong label when the catalyst is uncertain.
    if event_level == "High":
        if total >= 25:
            signal = "🟢 BUY — EVENT RISK"
        elif total <= -25:
            signal = "🔴 SELL — EVENT RISK"
        elif total >= 10:
            signal = "🟡 BUY — EVENT RISK"
        elif total <= -10:
            signal = "🟡 SELL — EVENT RISK"
        else:
            signal = "🟡 HOLD — EVENT RISK"
    else:
        if total >= 25: signal = "🟢 STRONG BUY"
        elif total >= 10: signal = "🟢 BUY"
        elif total <= -25: signal = "🔴 STRONG SELL"
        elif total <= -10: signal = "🔴 SELL"
        else: signal = "🟡 HOLD"

    return float(total), signal

def levels(d, current):
    x = d.tail(60)
    support = float(x["Low"].min())
    resistance = float(x["High"].max())
    atr = float(d["ATR14"].iloc[-1]) if pd.notna(d["ATR14"].iloc[-1]) else current * 0.02
    stop = max(0, current - 1.5 * atr)
    t1 = current + 1.5 * atr
    t2 = current + 3 * atr
    return support, resistance, stop, t1, t2

# -----------------------------
# Single stock analysis
# -----------------------------
def analyze_stock(symbol, period="3y", include_news=True):
    ticker = f"{symbol}.NS"
    d0 = history(ticker, period)
    if d0.empty or len(d0) < 100:
        return None

    current, price_date = latest_price(ticker)
    if current is None:
        current = float(d0["Close"].iloc[-1])
        price_date = d0.index[-1]

    d = add_indicators(d0)
    preds = ml_predictions(d)
    if preds[1]["pred"] is None:
        return None

    news = google_news(f"{symbol} stock India", 10) if include_news else []
    nscore, nlabel = news_summary(news)
    event, event_terms = event_risk(news)
    tech, reasons = technical_score(d, current)

    p1 = preds[1]["pred"]
    p5 = preds[5]["pred"]
    p20 = preds[20]["pred"]
    ml_change = (p5 / current - 1) * 100

    mr = market_regime()
    regime_score = mr["score"] if mr else 0
    total, signal = final_signal(tech, ml_change, nscore, regime_score, event)
    support, resistance, stop, t1, t2 = levels(d, current)

    return {
        "symbol": symbol, "ticker": ticker, "data": d,
        "current": current, "date": price_date, "preds": preds,
        "news": news, "news_score": nscore, "news_label": nlabel,
        "event": event, "event_terms": event_terms,
        "tech": tech, "reasons": reasons,
        "market": mr, "total": total, "signal": signal,
        "support": support, "resistance": resistance,
        "stop": stop, "target1": t1, "target2": t2
    }

@st.cache_data(ttl=900, show_spinner=False)
def scan_one(symbol, name):
    try:
        r = analyze_stock(symbol, "2y", True)
        if not r:
            return None
        p = r["preds"]
        return {
            "Symbol": symbol, "Company": name,
            "CMP": r["current"],
            "1D %": (p[1]["pred"]/r["current"]-1)*100,
            "5D %": (p[5]["pred"]/r["current"]-1)*100,
            "20D %": (p[20]["pred"]/r["current"]-1)*100,
            "News": r["news_score"], "Tech": r["tech"],
            "Score": r["total"], "Signal": r["signal"],
            "Event": r["event"]
        }
    except Exception:
        return None

# ============================================================
# UI
# ============================================================
st.title("📈 Indian Stock AI Analyzer")
st.caption("Multi-horizon ML • technicals • market regime • news sentiment • event risk")

m = market_regime()
if m:
    a,b,c,d,e = st.columns(5)
    a.metric("Nifty 50", f"{m['current']:,.2f}")
    b.metric("Market Regime", m["regime"])
    c.metric("Nifty MA20", f"{m['ma20']:,.2f}")
    d.metric("Nifty MA50", f"{m['ma50']:,.2f}")
    e.metric("Nifty RSI", f"{m['rsi']:.1f}")

tab1, tab2, tab3 = st.tabs(["🔎 Stock Analysis", "🏆 Nifty Scanner", "📰 Market News"])

with tab1:
    c1,c2,c3 = st.columns(3)
    with c1:
        symbol = st.text_input("NSE Symbol", "TCS").strip().upper()
    with c2:
        period = st.selectbox("Training History", ["2y","3y","5y"], index=1)
    with c3:
        run = st.button("🚀 Analyze", type="primary")

    if run:
        with st.spinner(f"Analyzing {symbol}..."):
            result = analyze_stock(symbol, period, True)

        if not result:
            st.error("Insufficient data or symbol not found.")
        else:
            p = result["preds"]
            current = result["current"]

            st.success(f"{symbol} analysis completed")
            st.caption(
                f"CMP data: {pd.Timestamp(result['date']).strftime('%d %b %Y')} • "
                "Yahoo Finance data may be delayed."
            )

            a,b,c,d = st.columns(4)
            a.metric("CMP", f"₹{current:,.2f}")
            d1 = (p[1]["pred"]/current-1)*100
            d5 = (p[5]["pred"]/current-1)*100
            d20 = (p[20]["pred"]/current-1)*100
            b.metric("1 Trading Day", f"₹{p[1]['pred']:,.2f}", f"{d1:+.2f}%")
            c.metric("5 Trading Days", f"₹{p[5]['pred']:,.2f}", f"{d5:+.2f}%")
            d.metric("20 Trading Days", f"₹{p[20]['pred']:,.2f}", f"{d20:+.2f}%")

            st.subheader("🎯 Combined Decision")
            a,b,c,d,e = st.columns(5)
            a.metric("Signal", result["signal"])
            b.metric("Overall Score", f"{result['total']:.1f}/50")
            c.metric("Technical", f"{result['tech']:+.1f}")
            d.metric("News", result["news_label"])
            e.metric("Event Risk", result["event"])

            st.subheader("⏱️ Prediction Timeline")
            timeline = pd.DataFrame({
                "Horizon":["Next trading day","5 trading days","20 trading days"],
                "Prediction":[p[1]["pred"],p[5]["pred"],p[20]["pred"]],
                "Expected Move %":[d1,d5,d20],
                "Validation Score %":[p[1]["validation"],p[5]["validation"],p[20]["validation"]],
                "Model Range":[
                    f"₹{max(0,p[1]['pred']-p[1]['std']):,.0f} – ₹{p[1]['pred']+p[1]['std']:,.0f}",
                    f"₹{max(0,p[5]['pred']-p[5]['std']):,.0f} – ₹{p[5]['pred']+p[5]['std']:,.0f}",
                    f"₹{max(0,p[20]['pred']-p[20]['std']):,.0f} – ₹{p[20]['pred']+p[20]['std']:,.0f}"
                ]
            })
            st.dataframe(
                timeline.style.format({
                    "Prediction":"₹{:,.2f}",
                    "Expected Move %":"{:+.2f}%",
                    "Validation Score %":"{:.1f}%"
                }),
                use_container_width=True
            )
            st.info(
                "The horizon is a trading-session estimate, not an exact date. "
                "A model range is uncertainty, not a guaranteed price interval."
            )

            st.subheader("📰 News & Catalyst Risk")
            st.write(
                f"**News sentiment:** {result['news_label']} "
                f"({result['news_score']:+.2f})"
            )
            if result["event_terms"]:
                st.warning(
                    "Potential catalysts detected: " +
                    ", ".join(result["event_terms"])
                )
            if result["news"]:
                for n in result["news"][:8]:
                    st.write(
                        f"**{sentiment_label(n['sentiment'])}** {n['title']} "
                        f"— {n['source']}"
                    )
            else:
                st.info("No recent Google News RSS headlines were returned.")

            st.subheader("📊 Technical Picture")
            x = result["data"].iloc[-1]
            a,b,c,d,e = st.columns(5)
            a.metric("RSI", f"{x['RSI']:.1f}")
            b.metric("MA20", f"₹{x['MA20']:,.2f}")
            c.metric("MA50", f"₹{x['MA50']:,.2f}")
            d.metric("MA200", f"₹{x['MA200']:,.2f}")
            e.metric("Volume Ratio", f"{x['VolumeRatio']:.2f}x")
            st.write("**Technical reasons:** " + " • ".join(result["reasons"]))

            st.subheader("🎯 Levels")
            a,b,c,d,e = st.columns(5)
            a.metric("Support", f"₹{result['support']:,.2f}")
            b.metric("Resistance", f"₹{result['resistance']:,.2f}")
            c.metric("Stop Loss", f"₹{result['stop']:,.2f}")
            d.metric("Target 1", f"₹{result['target1']:,.2f}")
            e.metric("Target 2", f"₹{result['target2']:,.2f}")

            st.subheader("📈 Price Chart")
            recent = result["data"].tail(300)
            fig, ax = plt.subplots(figsize=(14,5))
            ax.plot(recent.index, recent["Close"], label="Close")
            ax.plot(recent.index, recent["MA20"], label="MA20")
            ax.plot(recent.index, recent["MA50"], label="MA50")
            ax.plot(recent.index, recent["MA200"], label="MA200")
            ax.axhline(current, linestyle=":", label="CMP")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig, clear_figure=True)

            st.download_button(
                "⬇️ Download Stock CSV",
                result["data"].to_csv().encode("utf-8"),
                f"{symbol}_analysis.csv",
                "text/csv"
            )

with tab2:
    st.subheader("🏆 Nifty Stock Scanner")
    universe_name = st.selectbox("Universe", ["Nifty 10","Nifty 50","Nifty 100"])
    universe = UNIVERSES[universe_name]
    limit = st.slider("Number of stocks to scan", 10, len(universe), min(25, len(universe)))
    st.caption("The scanner uses 2 years of daily data plus recent news. Nifty 100 is Nifty 50 + Nifty Next 50 in this app.")

    if st.button("🔍 Run Scanner", type="primary"):
        rows = []
        progress = st.progress(0)
        status = st.empty()
        selected = list(universe.items())[:limit]

        for i,(sym,name) in enumerate(selected):
            status.write(f"Scanning {sym} ({i+1}/{len(selected)})")
            row = scan_one(sym,name)
            if row:
                rows.append(row)
            progress.progress((i+1)/len(selected))

        progress.empty()
        status.empty()

        if rows:
            scan = pd.DataFrame(rows).sort_values("Score", ascending=False)

            st.subheader("🥇 Highest-Ranked Opportunities")
            st.dataframe(
                scan.head(15).style.format({
                    "CMP":"₹{:,.2f}",
                    "1D %":"{:+.2f}%",
                    "5D %":"{:+.2f}%",
                    "20D %":"{:+.2f}%",
                    "News":"{:+.2f}",
                    "Tech":"{:+.1f}",
                    "Score":"{:+.1f}"
                }),
                use_container_width=True
            )

            a,b,c = st.columns(3)
            with a:
                st.subheader("🟢 BUY")
                st.dataframe(scan[scan["Signal"].str.contains("BUY")].head(10), use_container_width=True, hide_index=True)
            with b:
                st.subheader("🟡 HOLD")
                st.dataframe(scan[scan["Signal"].str.contains("HOLD")].head(10), use_container_width=True, hide_index=True)
            with c:
                st.subheader("🔴 SELL")
                st.dataframe(scan[scan["Signal"].str.contains("SELL")].sort_values("Score").head(10), use_container_width=True, hide_index=True)

            st.subheader("🚀 Highest Expected 20-Day Upside")
            st.dataframe(
                scan.sort_values("20D %", ascending=False).head(10),
                use_container_width=True, hide_index=True
            )

            st.subheader("📰 Strongest News Sentiment")
            st.dataframe(
                scan.sort_values("News", ascending=False).head(10),
                use_container_width=True, hide_index=True
            )

            st.download_button(
                "⬇️ Download Scanner CSV",
                scan.to_csv(index=False).encode("utf-8"),
                f"{universe_name.replace(' ','_')}_scanner.csv",
                "text/csv"
            )
        else:
            st.error("No stocks returned data. Try again later.")

with tab3:
    st.subheader("📰 Indian Market News")
    market_rows = google_news("Nifty 50 India stock market RBI NSE", 15)
    if market_rows:
        score, label = news_summary(market_rows)
        a,b = st.columns(2)
        a.metric("Market News Sentiment", label)
        b.metric("News Score", f"{score:+.2f}")
        for n in market_rows:
            st.write(f"**{sentiment_label(n['sentiment'])}** {n['title']} — {n['source']}")
    else:
        st.info("No market headlines returned.")

    st.subheader("⚠️ How to interpret the app")
    st.write("""
    • 1D, 5D and 20D are trading-session horizons, not guaranteed dates.
    • ML predictions are estimates from historical patterns; they are not guaranteed targets.
    • News sentiment is a lightweight text classifier, not a professional analyst rating.
    • Event risk is a warning layer; it does not predict the direction of an event.
    • Yahoo Finance prices may be delayed and are not a guaranteed real-time NSE execution feed.
    • Scanner scores are ranking tools, not investment advice.
    • Nifty constituents can change, so the built-in universe should be refreshed periodically against NSE's official constituent files.
    """)

st.divider()
st.caption("Research/decision-support tool only. Not investment advice.")
