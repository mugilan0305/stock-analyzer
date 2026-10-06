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
    "Technical analysis + machine learning stock prediction"
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
    value="RELIANCE"
).strip().upper()

start_date = st.sidebar.date_input(
    "Historical Start Date",
    value=datetime(2020, 1, 1)
)

end_date = st.sidebar.date_input(
    "Historical End Date",
    value=datetime.today()
)

run_analysis = st.sidebar.button(
    "🔍 Analyze Stock",
    type="primary"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def flatten_yfinance_columns(df):
    """
    Handles the newer yfinance MultiIndex format.
    """

    if isinstance(df.columns, pd.MultiIndex):

        # If columns look like:
        # ('Close', 'RELIANCE.NS')
        # ('Open', 'RELIANCE.NS')
        #
        # keep the first level.

        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    return df


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_indicators(data):

    data = data.copy()

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    data["MA20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["MA50"] = (
        data["Close"]
        .rolling(50)
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    data["RSI"] = calculate_rsi(
        data["Close"],
        14
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = (
        data["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        data["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    data["MACD"] = ema12 - ema26

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
        data["Close"]
        .rolling(20)
        .mean()
    )

    bb_std = (
        data["Close"]
        .rolling(20)
        .std()
    )

    data["BB_Upper"] = (
        data["BB_Middle"] +
        2 * bb_std
    )

    data["BB_Lower"] = (
        data["BB_Middle"] -
        2 * bb_std
    )

    # --------------------------------------------------------
    # Daily return
    # --------------------------------------------------------

    data["Daily_Return"] = (
        data["Close"].pct_change()
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    data["Volatility"] = (
        data["Daily_Return"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # Price momentum
    # --------------------------------------------------------

    data["Momentum_5"] = (
        data["Close"].pct_change(5)
    )

    data["Momentum_20"] = (
        data["Close"].pct_change(20)
    )

    return data


def train_prediction_model(data):

    model_data = data.copy()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

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
        "Volatility",
        "Momentum_5",
        "Momentum_20"
    ]

    # Target = next day's closing price

    model_data["Target"] = (
        model_data["Close"].shift(-1)
    )

    model_data = model_data.dropna(
        subset=features + ["Target"]
    )

    if len(model_data) < 150:

        return None, None, None, None

    X = model_data[features]
    y = model_data["Target"]

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    split = int(
        len(model_data) * 0.80
    )

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

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
    # Validation score
    # --------------------------------------------------------

    if len(X_test) > 0:

        predictions = model.predict(X_test)

        mae = np.mean(
            np.abs(
                predictions -
                y_test.values
            )
        )

        actual_mean = np.mean(
            np.abs(y_test.values)
        )

        if actual_mean > 0:

            accuracy = max(
                0,
                100 -
                (mae / actual_mean * 100)
            )

        else:

            accuracy = 0

    else:

        accuracy = 0

    # --------------------------------------------------------
    # Latest prediction
    # --------------------------------------------------------

    latest_row = data.dropna(
        subset=features
    ).iloc[-1:]

    X_latest = latest_row[features]

    tree_predictions = np.array(
        [
            tree.predict(X_latest)[0]
            for tree in model.estimators_
        ]
    )

    predicted_price = float(
        np.mean(tree_predictions)
    )

    prediction_std = float(
        np.std(tree_predictions)
    )

    return (
        model,
        predicted_price,
        prediction_std,
        accuracy
    )


def calculate_signal(
    data,
    predicted_price
):

    latest = data.iloc[-1]

    current_price = float(
        latest["Close"]
    )

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

    bb_upper = float(
        latest["BB_Upper"]
    )

    bb_lower = float(
        latest["BB_Lower"]
    )

    score = 0

    reasons = []

    # ========================================================
    # MOVING AVERAGE
    # ========================================================

    if ma20 > ma50:

        score += 1

        reasons.append(
            "MA20 is above MA50 — bullish trend."
        )

    else:

        score -= 1

        reasons.append(
            "MA20 is below MA50 — bearish trend."
        )

    # ========================================================
    # RSI
    # ========================================================

    if rsi < 30:

        score += 2

        reasons.append(
            "RSI is below 30 — potentially oversold."
        )

    elif rsi > 70:

        score -= 2

        reasons.append(
            "RSI is above 70 — potentially overbought."
        )

    elif rsi >= 50:

        score += 1

        reasons.append(
            "RSI is above 50 — positive momentum."
        )

    else:

        score -= 1

        reasons.append(
            "RSI is below 50 — weaker momentum."
        )

    # ========================================================
    # MACD
    # ========================================================

    if macd > signal:

        score += 1

        reasons.append(
            "MACD is above its signal line."
        )

    else:

        score -= 1

        reasons.append(
            "MACD is below its signal line."
        )

    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    if current_price <= bb_lower:

        score += 1

        reasons.append(
            "Price is near/below the lower Bollinger Band."
        )

    elif current_price >= bb_upper:

        score -= 1

        reasons.append(
            "Price is near/above the upper Bollinger Band."
        )

    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    if predicted_price > current_price:

        score += 2

        reasons.append(
            "ML model predicts a higher next-day price."
        )

    else:

        score -= 2

        reasons.append(
            "ML model predicts a lower next-day price."
        )

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

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


def calculate_levels(data):

    latest = data.iloc[-1]

    current_price = float(
        latest["Close"]
    )

    # Recent support

    recent = data.tail(60)

    support = float(
        recent["Low"].min()
    )

    resistance = float(
        recent["High"].max()
    )

    # ATR-like volatility calculation

    data = data.copy()

    data["TR"] = np.maximum(
        data["High"] - data["Low"],
        np.maximum(
            abs(
                data["High"] -
                data["Close"].shift(1)
            ),
            abs(
                data["Low"] -
                data["Close"].shift(1)
            )
        )
    )

    atr = float(
        data["TR"]
        .rolling(14)
        .mean()
        .iloc[-1]
    )

    if np.isnan(atr) or atr <= 0:

        atr = current_price * 0.02

    # --------------------------------------------------------
    # Stop loss
    # --------------------------------------------------------

    stop_loss = max(
        support,
        current_price - (1.5 * atr)
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    target1 = current_price + (
        1.5 * atr
    )

    target2 = current_price + (
        3 * atr
    )

    return (
        support,
        resistance,
        stop_loss,
        target1,
        target2,
        atr
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

if run_analysis or symbol:

    if not symbol:

        st.warning(
            "Please enter a stock symbol."
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

    ticker = (
        f"{symbol}.NS"
        if market == "NSE"
        else f"{symbol}.BO"
    )

    st.info(
        f"Fetching {ticker} historical data..."
    )

    # --------------------------------------------------------
    # Download data
    # --------------------------------------------------------

    try:

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=False,
            progress=False
        )

    except Exception as e:

        st.error(
            f"Unable to download stock data: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Fix yfinance columns
    # --------------------------------------------------------

    data = flatten_yfinance_columns(
        data
    )

    # --------------------------------------------------------
    # Validate
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
        if col not in data.columns
    ]

    if missing:

        st.error(
            f"Missing columns from Yahoo Finance: {missing}"
        )

        st.stop()

    if data.empty:

        st.error(
            f"No data found for {ticker}. "
            "Check the symbol and market."
        )

        st.stop()

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    for col in required_columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=required_columns
    )

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    data = calculate_indicators(
        data
    )

    # --------------------------------------------------------
    # Need enough history
    # --------------------------------------------------------

    if len(data) < 100:

        st.error(
            "Not enough historical data for analysis. "
            "Please select an earlier start date."
        )

        st.stop()

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    with st.spinner(
        "🤖 Training prediction model..."
    ):

        (
            model,
            predicted_price,
            prediction_std,
            model_accuracy
        ) = train_prediction_model(
            data
        )

    if model is None:

        st.error(
            "Not enough data to train the prediction model."
        )

        st.stop()

    # ========================================================
    # CURRENT VALUES
    # ========================================================

    latest = data.iloc[-1]

    current_price = float(
        latest["Close"]
    )

    rsi = float(
        latest["RSI"]
    )

    ma20 = float(
        latest["MA20"]
    )

    ma50 = float(
        latest["MA50"]
    )

    macd = float(
        latest["MACD"]
    )

    macd_signal = float(
        latest["Signal"]
    )

    # ========================================================
    # SIGNAL
    # ========================================================

    (
        recommendation,
        score,
        reasons
    ) = calculate_signal(
        data,
        predicted_price
    )

    # ========================================================
    # SUPPORT / RESISTANCE / TARGETS
    # ========================================================

    (
        support,
        resistance,
        stop_loss,
        target1,
        target2,
        atr
    ) = calculate_levels(
        data
    )

    # ========================================================
    # UPSIDE
    # ========================================================

    expected_change = (
        (predicted_price - current_price)
        / current_price
    ) * 100

    # ========================================================
    # HEADER
    # ========================================================

    st.success(
        f"Analysis completed for {symbol} ({market})"
    )

    st.subheader(
        f"📊 {symbol} — {market}"
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Current Price",
            f"₹{current_price:,.2f}"
        )

    with c2:

        st.metric(
            "ML Next-Day Prediction",
            f"₹{predicted_price:,.2f}",
            f"{expected_change:+.2f}%"
        )

    with c3:

        st.metric(
            "Technical Signal",
            recommendation
        )

    with c4:

        st.metric(
            "Signal Score",
            f"{score:+d}"
        )

    with c5:

        st.metric(
            "Model Confidence",
            f"{model_accuracy:.1f}%"
        )

    # ========================================================
    # TRADING LEVELS
    # ========================================================

    st.subheader(
        "🎯 Trading Levels"
    )

    l1, l2, l3, l4, l5 = st.columns(5)

    with l1:

        st.metric(
            "Support",
            f"₹{support:,.2f}"
        )

    with l2:

        st.metric(
            "Resistance",
            f"₹{resistance:,.2f}"
        )

    with l3:

        st.metric(
            "Stop Loss",
            f"₹{stop_loss:,.2f}"
        )

    with l4:

        st.metric(
            "Target 1",
            f"₹{target1:,.2f}"
        )

    with l5:

        st.metric(
            "Target 2",
            f"₹{target2:,.2f}"
        )

    # ========================================================
    # SIGNAL EXPLANATION
    # ========================================================

    st.subheader(
        "🧠 Why is the app giving this signal?"
    )

    for reason in reasons:

        st.write(
            f"• {reason}"
        )

    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

    st.subheader(
        "📊 Technical Indicators"
    )

    t1, t2, t3, t4 = st.columns(4)

    with t1:

        st.metric(
            "RSI",
            f"{rsi:.2f}"
        )

    with t2:

        st.metric(
            "MA20",
            f"₹{ma20:,.2f}"
        )

    with t3:

        st.metric(
            "MA50",
            f"₹{ma50:,.2f}"
        )

    with t4:

        macd_status = (
            "Bullish"
            if macd > macd_signal
            else "Bearish"
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

    ax1.plot(
        data.index,
        data["Close"],
        label="Close"
    )

    ax1.plot(
        data.index,
        data["MA20"],
        label="MA20"
    )

    ax1.plot(
        data.index,
        data["MA50"],
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
    # ML PREDICTION VISUALIZATION
    # ========================================================

    st.subheader(
        "🤖 Machine Learning Prediction"
    )

    recent_data = data.tail(120)

    fig2, ax2 = plt.subplots(
        figsize=(14, 5)
    )

    ax2.plot(
        recent_data.index,
        recent_data["Close"],
        label="Historical Price"
    )

    last_date = data.index[-1]

    next_date = (
        last_date +
        pd.Timedelta(days=1)
    )

    ax2.scatter(
        next_date,
        predicted_price,
        s=100,
        label="ML Prediction"
    )

    ax2.axhline(
        predicted_price,
        linestyle="--",
        label="Predicted Price"
    )

    ax2.set_title(
        f"Next-Day ML Prediction: ₹{predicted_price:,.2f}"
    )

    ax2.legend()

    ax2.grid(True)

    st.pyplot(
        fig2,
        clear_figure=True
    )

    # ========================================================
    # RSI
    # ========================================================

    st.subheader(
        "🌀 RSI"
    )

    fig3, ax3 = plt.subplots(
        figsize=(14, 4)
    )

    ax3.plot(
        data.index,
        data["RSI"],
        label="RSI"
    )

    ax3.axhline(
        70,
        linestyle="--",
        label="Overbought"
    )

    ax3.axhline(
        30,
        linestyle="--",
        label="Oversold"
    )

    ax3.axhline(
        50,
        linestyle=":",
        label="Midline"
    )

    ax3.set_ylim(
        0,
        100
    )

    ax3.legend()

    ax3.grid(True)

    st.pyplot(
        fig3,
        clear_figure=True
    )

    # ========================================================
    # MACD
    # ========================================================

    st.subheader(
        "📉 MACD"
    )

    fig4, ax4 = plt.subplots(
        figsize=(14, 4)
    )

    ax4.plot(
        data.index,
        data["MACD"],
        label="MACD"
    )

    ax4.plot(
        data.index,
        data["Signal"],
        label="Signal"
    )

    ax4.bar(
        data.index,
        data["MACD_Histogram"],
        alpha=0.3,
        label="Histogram"
    )

    ax4.axhline(
        0,
        linestyle="--"
    )

    ax4.legend()

    ax4.grid(True)

    st.pyplot(
        fig4,
        clear_figure=True
    )

    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    st.subheader(
        "📌 Bollinger Bands"
    )

    recent_bb = data.tail(150)

    fig5, ax5 = plt.subplots(
        figsize=(14, 5)
    )

    ax5.plot(
        recent_bb.index,
        recent_bb["Close"],
        label="Close"
    )

    ax5.plot(
        recent_bb.index,
        recent_bb["BB_Middle"],
        label="Middle"
    )

    ax5.plot(
        recent_bb.index,
        recent_bb["BB_Upper"],
        linestyle="--",
        label="Upper"
    )

    ax5.plot(
        recent_bb.index,
        recent_bb["BB_Lower"],
        linestyle="--",
        label="Lower"
    )

    ax5.fill_between(
        recent_bb.index,
        recent_bb["BB_Lower"].values,
        recent_bb["BB_Upper"].values,
        alpha=0.10
    )

    ax5.legend()

    ax5.grid(True)

    st.pyplot(
        fig5,
        clear_figure=True
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "🤖 Model Information"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Training Records",
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

    st.info(
        "The ML prediction is an estimate based on historical "
        "price and technical-indicator patterns. It is not a "
        "guarantee of future performance."
    )

    # ========================================================
    # DATA TABLE
    # ========================================================

    st.subheader(
        "📋 Recent Data"
    )

    display_columns = [
        "Close",
        "MA20",
        "MA50",
        "RSI",
        "MACD",
        "Signal",
        "BB_Upper",
        "BB_Lower"
    ]

    st.dataframe(
        data[display_columns]
        .tail(20)
        .round(2),
        use_container_width=True
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader(
        "📁 Export Analysis"
    )

    csv = data.to_csv().encode(
        "utf-8"
    )

    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name=(
            f"{symbol}_{market}_analysis.csv"
        ),
        mime="text/csv"
    )

else:

    st.info(
        "Enter a stock symbol in the sidebar "
        "and click **Analyze Stock**."
    )

    st.markdown(
        """
        ### Example symbols

        **NSE**
        - RELIANCE
        - TCS
        - INFY
        - HDFCBANK
        - ICICIBANK
        - SBIN
        - TATAMOTORS
        - MARUTI

        **BSE**

        Use the same symbol and select **BSE**.
        """
    )
