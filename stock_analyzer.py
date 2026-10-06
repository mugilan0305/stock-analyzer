import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Indian Stock Analyzer")
st.caption(
    "Indian stock technical analysis + machine learning"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Analysis Settings")

market = st.sidebar.selectbox(
    "Market",
    ["NSE", "BSE"]
)

symbol = st.sidebar.text_input(
    "Stock Symbol",
    value="TCS"
).strip().upper()

start_date = st.sidebar.date_input(
    "Historical Start Date",
    value=datetime(2020, 1, 1)
)

end_date = st.sidebar.date_input(
    "Historical End Date",
    value=datetime.today()
)

analyze = st.sidebar.button(
    "🔍 Analyze Stock",
    type="primary"
)


# ============================================================
# FUNCTIONS
# ============================================================

def clean_yfinance_columns(data):

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = [
            column[0]
            if isinstance(column, tuple)
            else column
            for column in data.columns
        ]

    return data


def get_stock_data(
    ticker,
    start,
    end
):

    try:

        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False
        )

        data = clean_yfinance_columns(data)

        return data

    except Exception as e:

        st.error(
            f"Yahoo Finance error: {e}"
        )

        return pd.DataFrame()


def get_latest_market_data(ticker):

    try:

        # Fetch the latest 10 calendar days
        # so we capture the latest trading session.

        recent_start = (
            datetime.today() -
            timedelta(days=10)
        ).strftime("%Y-%m-%d")

        recent_end = (
            datetime.today() +
            timedelta(days=1)
        ).strftime("%Y-%m-%d")

        recent = yf.download(
            ticker,
            start=recent_start,
            end=recent_end,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        recent = clean_yfinance_columns(
            recent
        )

        if recent.empty:

            return None, None

        recent = recent.dropna(
            subset=["Close"]
        )

        if recent.empty:

            return None, None

        latest_date = recent.index[-1]

        latest_close = float(
            recent["Close"].iloc[-1]
        )

        return (
            latest_close,
            latest_date
        )

    except Exception:

        return None, None


def calculate_indicators(data):

    data = data.copy()

    close = data["Close"]

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    data["MA20"] = (
        close
        .rolling(20)
        .mean()
    )

    data["MA50"] = (
        close
        .rolling(50)
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    data["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    data["MACD"] = (
        ema12 -
        ema26
    )

    data["Signal"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    data["MACD_Histogram"] = (
        data["MACD"] -
        data["Signal"]
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    data["BB_Middle"] = (
        close
        .rolling(20)
        .mean()
    )

    std20 = (
        close
        .rolling(20)
        .std()
    )

    data["BB_Upper"] = (
        data["BB_Middle"] +
        2 * std20
    )

    data["BB_Lower"] = (
        data["BB_Middle"] -
        2 * std20
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    data["Momentum_5"] = (
        close.pct_change(5)
    )

    data["Momentum_20"] = (
        close.pct_change(20)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    data["Daily_Return"] = (
        close.pct_change()
    )

    data["Volatility"] = (
        data["Daily_Return"]
        .rolling(20)
        .std()
    )

    return data


def train_model(data):

    features = [
        "Close",
        "Volume",
        "MA20",
        "MA50",
        "RSI",
        "MACD",
        "Signal",
        "MACD_Histogram",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "Momentum_5",
        "Momentum_20",
        "Volatility"
    ]

    model_data = data.copy()

    # Next trading day's close
    model_data["Target"] = (
        model_data["Close"]
        .shift(-1)
    )

    model_data = model_data.dropna(
        subset=features + ["Target"]
    )

    if len(model_data) < 150:

        return (
            None,
            None,
            None,
            None
        )

    X = model_data[features]
    y = model_data["Target"]

    split = int(
        len(model_data) * 0.8
    )

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(X_test) > 0:

        predictions = model.predict(
            X_test
        )

        mae = np.mean(
            np.abs(
                predictions -
                y_test.values
            )
        )

        mean_price = np.mean(
            np.abs(
                y_test.values
            )
        )

        if mean_price > 0:

            model_score = (
                100 -
                (
                    mae /
                    mean_price *
                    100
                )
            )

            model_score = max(
                0,
                min(
                    100,
                    model_score
                )
            )

        else:

            model_score = 0

    else:

        model_score = 0

    # --------------------------------------------------------
    # Latest prediction
    # --------------------------------------------------------

    latest = data.dropna(
        subset=features
    ).iloc[-1:]

    X_latest = latest[features]

    prediction = model.predict(
        X_latest
    )[0]

    # Tree-by-tree prediction
    # gives an estimate of uncertainty.

    tree_predictions = np.array(
        [
            tree.predict(
                X_latest
            )[0]
            for tree in model.estimators_
        ]
    )

    prediction_std = float(
        np.std(
            tree_predictions
        )
    )

    return (
        model,
        float(prediction),
        prediction_std,
        model_score
    )


def calculate_signal(
    data,
    current_price,
    predicted_price
):

    latest = data.iloc[-1]

    score = 0

    reasons = []

    ma20 = float(
        latest["MA20"]
    )

    ma50 = float(
        latest["MA50"]
    )

    rsi = float(
        latest["RSI"]
    )

    macd = float(
        latest["MACD"]
    )

    signal = float(
        latest["Signal"]
    )

    upper = float(
        latest["BB_Upper"]
    )

    lower = float(
        latest["BB_Lower"]
    )

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    if ma20 > ma50:

        score += 1

        reasons.append(
            "MA20 is above MA50."
        )

    else:

        score -= 1

        reasons.append(
            "MA20 is below MA50."
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi < 30:

        score += 2

        reasons.append(
            "RSI indicates oversold conditions."
        )

    elif rsi > 70:

        score -= 2

        reasons.append(
            "RSI indicates overbought conditions."
        )

    elif rsi >= 50:

        score += 1

        reasons.append(
            "RSI is above 50."
        )

    else:

        score -= 1

        reasons.append(
            "RSI is below 50."
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if macd > signal:

        score += 1

        reasons.append(
            "MACD is above the signal line."
        )

    else:

        score -= 1

        reasons.append(
            "MACD is below the signal line."
        )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    if current_price <= lower:

        score += 1

        reasons.append(
            "Price is near the lower Bollinger Band."
        )

    elif current_price >= upper:

        score -= 1

        reasons.append(
            "Price is near the upper Bollinger Band."
        )

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    predicted_change = (
        (
            predicted_price -
            current_price
        )
        /
        current_price
    ) * 100

    if predicted_change > 2:

        score += 2

        reasons.append(
            f"ML model predicts +{predicted_change:.2f}%."
        )

    elif predicted_change < -2:

        score -= 2

        reasons.append(
            f"ML model predicts {predicted_change:.2f}%."
        )

    else:

        reasons.append(
            "ML prediction is relatively close to CMP."
        )

    # --------------------------------------------------------
    # Final recommendation
    # --------------------------------------------------------

    if score >= 4:

        recommendation = "🟢 STRONG BUY"

    elif score >= 2:

        recommendation = "🟢 BUY"

    elif score <= -4:

        recommendation = "🔴 STRONG SELL"

    elif score <= -2:

        recommendation = "🔴 SELL"

    else:

        recommendation = "🟡 HOLD"

    return (
        recommendation,
        score,
        reasons
    )


def calculate_levels(
    data,
    current_price
):

    recent = data.tail(60)

    support = float(
        recent["Low"].min()
    )

    resistance = float(
        recent["High"].max()
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        data["Close"]
        .shift(1)
    )

    tr1 = (
        data["High"] -
        data["Low"]
    )

    tr2 = (
        abs(
            data["High"] -
            previous_close
        )
    )

    tr3 = (
        abs(
            data["Low"] -
            previous_close
        )
    )

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = (
        true_range
        .rolling(14)
        .mean()
        .iloc[-1]
    )

    if pd.isna(atr):

        atr = (
            current_price *
            0.02
        )

    # --------------------------------------------------------
    # Stop loss
    # --------------------------------------------------------

    stop_loss = (
        current_price -
        1.5 * atr
    )

    # Don't put stop above CMP.

    stop_loss = min(
        stop_loss,
        current_price * 0.98
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    target1 = (
        current_price +
        1.5 * atr
    )

    target2 = (
        current_price +
        3 * atr
    )

    return (
        support,
        resistance,
        stop_loss,
        target1,
        target2,
        float(atr)
    )


# ============================================================
# APPLICATION
# ============================================================

if analyze or symbol:

    if not symbol:

        st.warning(
            "Enter a stock symbol."
        )

        st.stop()

    if start_date >= end_date:

        st.error(
            "Start date must be before end date."
        )

        st.stop()

    # --------------------------------------------------------
    # Ticker
    # --------------------------------------------------------

    if market == "NSE":

        ticker = (
            f"{symbol}.NS"
        )

    else:

        ticker = (
            f"{symbol}.BO"
        )

    # ========================================================
    # HISTORICAL DATA
    # ========================================================

    st.info(
        f"Downloading historical data for {ticker}..."
    )

    historical = get_stock_data(
        ticker,
        start_date,
        end_date + timedelta(days=1)
    )

    if historical.empty:

        st.error(
            f"No historical data found for {ticker}."
        )

        st.stop()

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        col
        for col in required_columns
        if col not in historical.columns
    ]

    if missing:

        st.error(
            f"Missing data columns: {missing}"
        )

        st.stop()

    for column in required_columns:

        historical[column] = pd.to_numeric(
            historical[column],
            errors="coerce"
        )

    historical = historical.dropna(
        subset=required_columns
    )

    if len(historical) < 100:

        st.error(
            "Not enough historical data. "
            "Choose an earlier start date."
        )

        st.stop()

    # ========================================================
    # INDICATORS
    # ========================================================

    data = calculate_indicators(
        historical
    )

    # ========================================================
    # LATEST MARKET PRICE
    # ========================================================

    latest_price, latest_price_date = (
        get_latest_market_data(
            ticker
        )
    )

    if latest_price is None:

        # Fallback to latest historical price.

        current_price = float(
            data["Close"].iloc[-1]
        )

        price_source = (
            "Historical data fallback"
        )

        price_date = data.index[-1]

    else:

        current_price = latest_price

        price_source = (
            "Yahoo Finance latest daily quote"
        )

        price_date = latest_price_date

    # ========================================================
    # TRAIN ML MODEL
    # ========================================================

    with st.spinner(
        "🤖 Training machine-learning model..."
    ):

        (
            model,
            predicted_price,
            prediction_std,
            model_score
        ) = train_model(
            data
        )

    if model is None:

        st.error(
            "Unable to train the prediction model."
        )

        st.stop()

    # ========================================================
    # SIGNAL
    # ========================================================

    (
        recommendation,
        score,
        reasons
    ) = calculate_signal(
        data,
        current_price,
        predicted_price
    )

    # ========================================================
    # TRADING LEVELS
    # ========================================================

    (
        support,
        resistance,
        stop_loss,
        target1,
        target2,
        atr
    ) = calculate_levels(
        data,
        current_price
    )

    # ========================================================
    # EXPECTED CHANGE
    # ========================================================

    expected_change = (
        (
            predicted_price -
            current_price
        )
        /
        current_price
    ) * 100

    # ========================================================
    # HEADER
    # ========================================================

    st.success(
        f"Analysis completed for {symbol} ({market})"
    )

    st.caption(
        f"📅 CMP data: {price_date.strftime('%d %b %Y')} | "
        f"Source: {price_source}"
    )

    # ========================================================
    # CMP
    # ========================================================

    st.subheader(
        "💰 Current Market Price"
    )

    cmp1, cmp2, cmp3 = st.columns(3)

    with cmp1:

        st.metric(
            "CMP",
            f"₹{current_price:,.2f}"
        )

    with cmp2:

        historical_close = float(
            data["Close"].iloc[-1]
        )

        difference = (
            current_price -
            historical_close
        )

        st.metric(
            "Latest Historical Close",
            f"₹{historical_close:,.2f}",
            f"₹{difference:+,.2f}"
        )

    with cmp3:

        st.metric(
            "ML Predicted Price",
            f"₹{predicted_price:,.2f}",
            f"{expected_change:+.2f}%"
        )

    # ========================================================
    # MAIN SIGNAL
    # ========================================================

    st.subheader(
        "🎯 Trading Signal"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Recommendation",
            recommendation
        )

    with c2:

        st.metric(
            "Signal Score",
            f"{score:+d}"
        )

    with c3:

        st.metric(
            "ML Prediction",
            f"₹{predicted_price:,.2f}"
        )

    with c4:

        st.metric(
            "Model Score",
            f"{model_score:.1f}%"
        )

    # ========================================================
    # SIGNAL REASONS
    # ========================================================

    st.subheader(
        "🧠 Signal Explanation"
    )

    for reason in reasons:

        st.write(
            f"• {reason}"
        )

    # ========================================================
    # TRADING LEVELS
    # ========================================================

    st.subheader(
        "🎯 Trading Levels"
    )

    t1, t2, t3, t4, t5 = st.columns(5)

    with t1:

        st.metric(
            "Support",
            f"₹{support:,.2f}"
        )

    with t2:

        st.metric(
            "Resistance",
            f"₹{resistance:,.2f}"
        )

    with t3:

        st.metric(
            "Stop Loss",
            f"₹{stop_loss:,.2f}"
        )

    with t4:

        st.metric(
            "Target 1",
            f"₹{target1:,.2f}"
        )

    with t5:

        st.metric(
            "Target 2",
            f"₹{target2:,.2f}"
        )

    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

    latest = data.iloc[-1]

    st.subheader(
        "📊 Technical Indicators"
    )

    i1, i2, i3, i4 = st.columns(4)

    with i1:

        st.metric(
            "RSI",
            f"{float(latest['RSI']):.2f}"
        )

    with i2:

        st.metric(
            "MA20",
            f"₹{float(latest['MA20']):,.2f}"
        )

    with i3:

        st.metric(
            "MA50",
            f"₹{float(latest['MA50']):,.2f}"
        )

    with i4:

        macd_status = (
            "Bullish"
            if float(latest["MACD"])
            >
            float(latest["Signal"])
            else
            "Bearish"
        )

        st.metric(
            "MACD",
            macd_status
        )

    # ========================================================
    # PRICE CHART
    # ========================================================

    st.subheader(
        "📈 Price & Moving Averages"
    )

    fig1, ax1 = plt.subplots(
        figsize=(14, 6)
    )

    recent = data.tail(250)

    ax1.plot(
        recent.index,
        recent["Close"],
        label="Historical Close"
    )

    ax1.plot(
        recent.index,
        recent["MA20"],
        label="MA20"
    )

    ax1.plot(
        recent.index,
        recent["MA50"],
        label="MA50"
    )

    ax1.axhline(
        support,
        linestyle="--",
        label="Support"
    )

    ax1.axhline(
        resistance,
        linestyle="--",
        label="Resistance"
    )

    ax1.axhline(
        current_price,
        linestyle=":",
        label="CMP"
    )

    ax1.set_title(
        f"{symbol} Price Analysis"
    )

    ax1.legend()

    ax1.grid(True)

    st.pyplot(
        fig1,
        clear_figure=True
    )

    # ========================================================
    # ML PREDICTION
    # ========================================================

    st.subheader(
        "🤖 Machine Learning Prediction"
    )

    prediction_low = (
        predicted_price -
        prediction_std
    )

    prediction_high = (
        predicted_price +
        prediction_std
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "Predicted Price",
            f"₹{predicted_price:,.2f}"
        )

    with p2:

        st.metric(
            "Prediction Range Low",
            f"₹{prediction_low:,.2f}"
        )

    with p3:

        st.metric(
            "Prediction Range High",
            f"₹{prediction_high:,.2f}"
        )

    # ========================================================
    # RSI
    # ========================================================

    st.subheader(
        "🌀 RSI"
    )

    fig2, ax2 = plt.subplots(
        figsize=(14, 4)
    )

    ax2.plot(
        data.index,
        data["RSI"],
        label="RSI"
    )

    ax2.axhline(
        70,
        linestyle="--",
        label="Overbought"
    )

    ax2.axhline(
        30,
        linestyle="--",
        label="Oversold"
    )

    ax2.axhline(
        50,
        linestyle=":",
        label="Midline"
    )

    ax2.set_ylim(
        0,
        100
    )

    ax2.legend()

    ax2.grid(True)

    st.pyplot(
        fig2,
        clear_figure=True
    )

    # ========================================================
    # MACD
    # ========================================================

    st.subheader(
        "📉 MACD"
    )

    fig3, ax3 = plt.subplots(
        figsize=(14, 4)
    )

    ax3.plot(
        data.index,
        data["MACD"],
        label="MACD"
    )

    ax3.plot(
        data.index,
        data["Signal"],
        label="Signal"
    )

    ax3.bar(
        data.index,
        data["MACD_Histogram"],
        alpha=0.3,
        label="Histogram"
    )

    ax3.axhline(
        0,
        linestyle="--"
    )

    ax3.legend()

    ax3.grid(True)

    st.pyplot(
        fig3,
        clear_figure=True
    )

    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    st.subheader(
        "📌 Bollinger Bands"
    )

    recent_bb = data.tail(200)

    fig4, ax4 = plt.subplots(
        figsize=(14, 5)
    )

    ax4.plot(
        recent_bb.index,
        recent_bb["Close"],
        label="Close"
    )

    ax4.plot(
        recent_bb.index,
        recent_bb["BB_Middle"],
        label="Middle"
    )

    ax4.plot(
        recent_bb.index,
        recent_bb["BB_Upper"],
        linestyle="--",
        label="Upper"
    )

    ax4.plot(
        recent_bb.index,
        recent_bb["BB_Lower"],
        linestyle="--",
        label="Lower"
    )

    ax4.legend()

    ax4.grid(True)

    st.pyplot(
        fig4,
        clear_figure=True
    )

    # ========================================================
    # MODEL DETAILS
    # ========================================================

    st.subheader(
        "🤖 Model Details"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Historical Records",
            f"{len(data):,}"
        )

    with m2:

        st.metric(
            "Random Forest Trees",
            "200"
        )

    with m3:

        st.metric(
            "Prediction Uncertainty",
            f"±₹{prediction_std:,.2f}"
        )

    st.warning(
        "⚠️ The machine-learning prediction and trading signal "
        "are estimates based on historical patterns and technical "
        "indicators. They are not guaranteed future prices or "
        "investment advice."
    )

    # ========================================================
    # RECENT DATA
    # ========================================================

    st.subheader(
        "📋 Recent Market Data"
    )

    display_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA20",
        "MA50",
        "RSI",
        "MACD",
        "Signal",
        "BB_Upper",
        "BB_Lower"
    ]

    st.dataframe(
        data[
            display_columns
        ]
        .tail(20)
        .round(2),
        use_container_width=True
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader(
        "📁 Export Data"
    )

    csv = data.to_csv().encode(
        "utf-8"
    )

    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=(
            f"{symbol}_{market}_analysis.csv"
        ),
        mime="text/csv"
    )

else:

    st.info(
        "Enter a stock symbol and click "
        "**Analyze Stock**."
    )
