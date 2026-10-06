import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Indian Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Indian Stock Predictor")
st.caption(
    "Machine-learning based next-day direction prediction "
    "using historical market data."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Analysis Settings")

market = st.sidebar.selectbox(
    "Market",
    ["NSE", "BSE"]
)

symbol = st.sidebar.text_input(
    "Stock Symbol",
    "RELIANCE"
).upper().strip()

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2020-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    datetime.today()
)

analyze = st.sidebar.button(
    "🚀 Analyze Stock",
    use_container_width=True
)


# =========================================================
# FUNCTIONS
# =========================================================

def download_stock_data(ticker, start_date, end_date):

    download_end = end_date + timedelta(days=1)

    data = yf.download(
        ticker,
        start=start_date,
        end=download_end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    if data.empty:
        return None

    # Safety check for yfinance MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    for column in required:
        if column not in data.columns:
            return None

    return data


def calculate_indicators(data):

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    # -----------------------------------------------------
    # Moving averages
    # -----------------------------------------------------

    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # -----------------------------------------------------
    # Bollinger Bands
    # -----------------------------------------------------

    data["BB_Middle"] = close.rolling(20).mean()

    std20 = close.rolling(20).std()

    data["BB_Upper"] = (
        data["BB_Middle"] + 2 * std20
    )

    data["BB_Lower"] = (
        data["BB_Middle"] - 2 * std20
    )

    # -----------------------------------------------------
    # Daily return
    # -----------------------------------------------------

    data["Return_1D"] = close.pct_change()

    data["Return_5D"] = close.pct_change(5)

    data["Return_20D"] = close.pct_change(20)

    # -----------------------------------------------------
    # Volatility
    # -----------------------------------------------------

    data["Volatility_10D"] = (
        data["Return_1D"]
        .rolling(10)
        .std()
    )

    # -----------------------------------------------------
    # Volume change
    # -----------------------------------------------------

    data["Volume_Change"] = (
        data["Volume"].pct_change()
    )

    # -----------------------------------------------------
    # Target
    #
    # 1 = next day goes UP
    # 0 = next day goes DOWN
    # -----------------------------------------------------

    data["Target"] = (
        close.shift(-1) > close
    ).astype(int)

    return data


# =========================================================
# MAIN ANALYSIS
# =========================================================

if analyze:

    if not symbol:
        st.error("Please enter a stock symbol.")
        st.stop()

    if start_date >= end_date:
        st.error(
            "Start date must be before end date."
        )
        st.stop()

    ticker = (
        f"{symbol}.NS"
        if market == "NSE"
        else f"{symbol}.BO"
    )

    with st.spinner(
        f"Downloading {ticker} market data..."
    ):

        data = download_stock_data(
            ticker,
            start_date,
            end_date
        )

    if data is None:
        st.error(
            f"❌ Could not find data for `{ticker}`."
        )
        st.info(
            "Check the symbol and market. "
            "Example: RELIANCE for NSE."
        )
        st.stop()

    # Calculate indicators
    data = calculate_indicators(data)

    # -----------------------------------------------------
    # Remove invalid rows
    # -----------------------------------------------------

    features = [
        "MA20",
        "MA50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "Return_1D",
        "Return_5D",
        "Return_20D",
        "Volatility_10D",
        "Volume_Change"
    ]

    model_data = data.dropna(
        subset=features + ["Target"]
    ).copy()

    if len(model_data) < 250:
        st.error(
            "Not enough historical data to train "
            "the model reliably."
        )
        st.stop()

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    # Time-series split:
    # Never randomly shuffle stock data.

    split_index = int(
        len(model_data) * 0.80
    )

    train = model_data.iloc[:split_index]
    test = model_data.iloc[split_index:]

    X_train = train[features]
    y_train = train["Target"]

    X_test = test[features]
    y_test = test["Target"]

    # =====================================================
    # RANDOM FOREST MODEL
    # =====================================================

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )

    with st.spinner(
        "Training machine-learning model..."
    ):

        model.fit(
            X_train,
            y_train
        )

    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    test_predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    # =====================================================
    # CURRENT PREDICTION
    # =====================================================

    latest = data.dropna(
        subset=features
    ).iloc[-1]

    latest_features = latest[
        features
    ].to_frame().T

    prediction = model.predict(
        latest_features
    )[0]

    probabilities = model.predict_proba(
        latest_features
    )[0]

    probability_down = probabilities[0]
    probability_up = probabilities[1]

    current_price = float(
        latest["Close"]
    )

    # =====================================================
    # SIGNAL
    # =====================================================

    if probability_up >= 0.60:

        signal = "🟢 BUY"
        signal_text = (
            "Model shows a bullish probability."
        )

    elif probability_down >= 0.60:

        signal = "🔴 SELL"
        signal_text = (
            "Model shows a bearish probability."
        )

    else:

        signal = "🟡 HOLD"
        signal_text = (
            "Model does not show a strong directional edge."
        )

    # =====================================================
    # HEADER METRICS
    # =====================================================

    st.subheader(
        f"{symbol} ({market})"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Price",
            f"₹{current_price:,.2f}"
        )

    with col2:

        st.metric(
            "UP Probability",
            f"{probability_up * 100:.1f}%"
        )

    with col3:

        st.metric(
            "DOWN Probability",
            f"{probability_down * 100:.1f}%"
        )

    with col4:

        st.metric(
            "Historical Test Accuracy",
            f"{accuracy * 100:.1f}%"
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    st.divider()

    st.subheader(
        "🤖 AI Prediction"
    )

    prediction_col1, prediction_col2 = st.columns(
        [1, 2]
    )

    with prediction_col1:

        st.markdown(
            f"# {signal}"
        )

    with prediction_col2:

        st.write(signal_text)

        st.progress(
            float(probability_up),
            text=(
                f"Probability of next-day UP move: "
                f"{probability_up * 100:.1f}%"
            )
        )

    # =====================================================
    # PRICE CHART
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Price & Moving Averages"
    )

    chart_data = data[
        ["Close", "MA20", "MA50"]
    ].copy()

    st.line_chart(
        chart_data,
        use_container_width=True
    )

    # =====================================================
    # RSI
    # =====================================================

    st.subheader(
        "🌀 RSI"
    )

    st.line_chart(
        data[["RSI"]],
        use_container_width=True
    )

    # =====================================================
    # MACD
    # =====================================================

    st.subheader(
        "📉 MACD"
    )

    st.line_chart(
        data[
            ["MACD", "MACD_Signal"]
        ],
        use_container_width=True
    )

    # =====================================================
    # BOLLINGER BANDS
    # =====================================================

    st.subheader(
        "📌 Bollinger Bands"
    )

    st.line_chart(
        data[
            [
                "Close",
                "BB_Upper",
                "BB_Middle",
                "BB_Lower"
            ]
        ],
        use_container_width=True
    )

    # =====================================================
    # MODEL FEATURE IMPORTANCE
    # =====================================================

    st.divider()

    st.subheader(
        "🧠 What influenced the model?"
    )

    importance = pd.DataFrame({
        "Indicator": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # LATEST INDICATORS
    # =====================================================

    st.subheader(
        "📋 Current Indicators"
    )

    indicator_col1, indicator_col2, indicator_col3 = st.columns(3)

    with indicator_col1:

        st.metric(
            "RSI",
            f"{latest['RSI']:.2f}"
        )

    with indicator_col2:

        st.metric(
            "MA20",
            f"₹{latest['MA20']:,.2f}"
        )

    with indicator_col3:

        st.metric(
            "MA50",
            f"₹{latest['MA50']:,.2f}"
        )

    # =====================================================
    # DOWNLOAD DATA
    # =====================================================

    st.divider()

    st.subheader(
        "📁 Export Analysis"
    )

    csv = data.to_csv(
        index=True
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=(
            f"{symbol}_{market}_AI_analysis.csv"
        ),
        mime="text/csv"
    )

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning(
        "⚠️ This model is for educational and research "
        "purposes only. Historical accuracy does not "
        "guarantee future performance. Do not make "
        "investment decisions solely from this model."
    )

else:

    st.info(
        "👈 Enter an NSE/BSE stock symbol and click "
        "**Analyze Stock** to start."
    )
