import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Indian Stock Analyzer")
st.caption("Technical analysis using Yahoo Finance market data")

# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    market = st.selectbox(
        "Choose Market",
        ["NSE", "BSE"]
    )

with col2:
    symbol = st.text_input(
        "Enter Stock Symbol",
        placeholder="e.g. RELIANCE, TCS, INFY"
    ).upper().strip()

col3, col4 = st.columns(2)

with col3:
    start_date = st.date_input(
        "Start Date",
        pd.to_datetime("2020-01-01")
    )

with col4:
    end_date = st.date_input(
        "End Date",
        datetime.today()
    )

# ---------------------------------------------------------
# Validate Dates
# ---------------------------------------------------------
if start_date >= end_date:
    st.error("❌ Start Date must be before End Date.")
    st.stop()

# ---------------------------------------------------------
# Run Analysis
# ---------------------------------------------------------
if symbol:

    ticker = f"{symbol}.NS" if market == "NSE" else f"{symbol}.BO"

    st.info(
        f"Fetching data for **{ticker}** from "
        f"**{start_date}** to **{end_date}**..."
    )

    try:
        # yfinance end date is exclusive, so add one day
        download_end = end_date + timedelta(days=1)

        data = yf.download(
            ticker,
            start=start_date,
            end=download_end,
            auto_adjust=True,
            progress=False,
            multi_level_index=False
        )

    except Exception as e:
        st.error(f"❌ Error downloading data: {e}")
        st.stop()

    # -----------------------------------------------------
    # Check Data
    # -----------------------------------------------------
    if data.empty:
        st.error(
            f"❌ No data found for `{ticker}`. "
            "Please check the stock symbol and market."
        )
        st.stop()

    # -----------------------------------------------------
    # Safety Fix for yfinance MultiIndex
    # -----------------------------------------------------
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Make sure Close exists
    if "Close" not in data.columns:
        st.error("❌ Unable to find Close price in downloaded data.")
        st.stop()

    # Force Close to be a single Series
    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")

    # -----------------------------------------------------
    # Technical Indicators
    # -----------------------------------------------------

    # Moving Averages
    data["MA20"] = close.rolling(window=20).mean()
    data["MA50"] = close.rolling(window=50).mean()

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (100 / (1 + rs))

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    exp1 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    exp2 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = exp1 - exp2

    data["Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # -----------------------------------------------------
    # Bollinger Bands
    # -----------------------------------------------------

    data["BB_Middle"] = close.rolling(
        window=20
    ).mean()

    rolling_std = close.rolling(
        window=20
    ).std()

    data["BB_Upper"] = (
        data["BB_Middle"] + 2 * rolling_std
    )

    data["BB_Lower"] = (
        data["BB_Middle"] - 2 * rolling_std
    )

    # -----------------------------------------------------
    # Prepare Data for Regression
    # -----------------------------------------------------

    data = data.reset_index()

    # Handle different possible date column names
    if "Date" not in data.columns:
        if "Datetime" in data.columns:
            data.rename(
                columns={"Datetime": "Date"},
                inplace=True
            )

    data["Date"] = pd.to_datetime(data["Date"])

    data["Date_ordinal"] = data["Date"].map(
        datetime.toordinal
    )

    # -----------------------------------------------------
    # Linear Regression Trend
    # -----------------------------------------------------

    regression_data = data.dropna(
        subset=["Date_ordinal", "Close"]
    )

    if len(regression_data) >= 2:

        model = LinearRegression()

        model.fit(
            regression_data[["Date_ordinal"]],
            regression_data["Close"]
        )

        data["Trend"] = model.predict(
            data[["Date_ordinal"]]
        )

    else:
        data["Trend"] = None

    # -----------------------------------------------------
    # Current Price Information
    # -----------------------------------------------------

    latest_close = float(close.dropna().iloc[-1])

    previous_close = (
        float(close.dropna().iloc[-2])
        if len(close.dropna()) > 1
        else latest_close
    )

    daily_change = latest_close - previous_close

    daily_change_pct = (
        daily_change / previous_close * 100
        if previous_close != 0
        else 0
    )

    # -----------------------------------------------------
    # Display Price Information
    # -----------------------------------------------------

    st.subheader(f"📊 {symbol} ({market})")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Current Price",
            f"₹{latest_close:,.2f}"
        )

    with metric2:
        st.metric(
            "Daily Change",
            f"₹{daily_change:,.2f}",
            f"{daily_change_pct:.2f}%"
        )

    with metric3:
        latest_rsi = data["RSI"].dropna().iloc[-1]

        st.metric(
            "RSI",
            f"{latest_rsi:.2f}"
        )

    with metric4:
        latest_macd = data["MACD"].dropna().iloc[-1]

        st.metric(
            "MACD",
            f"{latest_macd:.2f}"
        )

    # -----------------------------------------------------
    # Price & Moving Averages
    # -----------------------------------------------------

    st.subheader("📊 Stock Price & Moving Averages")

    fig1, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        data["Date"],
        data["Close"],
        label="Close Price"
    )

    ax1.plot(
        data["Date"],
        data["MA20"],
        label="MA20"
    )

    ax1.plot(
        data["Date"],
        data["MA50"],
        label="MA50"
    )

    ax1.set_title(
        f"{symbol} Price & Moving Averages"
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (₹)")

    ax1.legend()
    ax1.grid(True)

    st.pyplot(fig1)

    plt.close(fig1)

    # -----------------------------------------------------
    # Trend Line
    # -----------------------------------------------------

    st.subheader("📈 Trend Line")

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(
        data["Date"],
        data["Close"],
        label="Actual Price"
    )

    ax2.plot(
        data["Date"],
        data["Trend"],
        label="Trend Line",
        linestyle="--"
    )

    ax2.set_title(
        f"{symbol} Price Trend"
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (₹)")

    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)

    plt.close(fig2)

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    st.subheader("🌀 Relative Strength Index (RSI)")

    fig3, ax3 = plt.subplots(figsize=(12, 3))

    ax3.plot(
        data["Date"],
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

    ax3.set_title(
        "RSI (Overbought >70, Oversold <30)"
    )

    ax3.set_ylim(0, 100)

    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)

    plt.close(fig3)

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    st.subheader("📉 MACD")

    fig4, ax4 = plt.subplots(figsize=(12, 3))

    ax4.plot(
        data["Date"],
        data["MACD"],
        label="MACD"
    )

    ax4.plot(
        data["Date"],
        data["Signal"],
        label="Signal Line"
    )

    ax4.axhline(0, linestyle="--")

    ax4.set_title(
        "MACD & Signal Line"
    )

    ax4.legend()
    ax4.grid(True)

    st.pyplot(fig4)

    plt.close(fig4)

    # -----------------------------------------------------
    # Bollinger Bands
    # -----------------------------------------------------

    st.subheader("📌 Bollinger Bands")

    fig5, ax5 = plt.subplots(figsize=(12, 5))

    ax5.plot(
        data["Date"],
        data["Close"],
        label="Close"
    )

    ax5.plot(
        data["Date"],
        data["BB_Middle"],
        label="Middle Band"
    )

    ax5.plot(
        data["Date"],
        data["BB_Upper"],
        label="Upper Band",
        linestyle="--"
    )

    ax5.plot(
        data["Date"],
        data["BB_Lower"],
        label="Lower Band",
        linestyle="--"
    )

    ax5.set_title(
        "Bollinger Bands"
    )

    ax5.legend()
    ax5.grid(True)

    st.pyplot(fig5)

    plt.close(fig5)

    # -----------------------------------------------------
    # Simple Technical Signal
    # -----------------------------------------------------

    st.subheader("🤖 Technical Signal")

    latest_ma20 = data["MA20"].dropna().iloc[-1]
    latest_ma50 = data["MA50"].dropna().iloc[-1]
    latest_rsi = data["RSI"].dropna().iloc[-1]

    if (
        latest_close > latest_ma20
        and latest_ma20 > latest_ma50
        and latest_rsi < 70
    ):
        signal = "🟢 BUY"

    elif (
        latest_close < latest_ma20
        and latest_ma20 < latest_ma50
        and latest_rsi > 30
    ):
        signal = "🔴 SELL"

    else:
        signal = "🟡 HOLD"

    st.success(
        f"Current Technical Signal: **{signal}**"
    )

    st.caption(
        "This is a technical-analysis signal, not financial advice."
    )

    # -----------------------------------------------------
    # Latest Data
    # -----------------------------------------------------

    st.subheader("📋 Latest Market Data")

    display_columns = [
        "Date",
        "Close",
        "MA20",
        "MA50",
        "RSI",
        "MACD",
        "Signal",
        "BB_Upper",
        "BB_Lower"
    ]

    available_columns = [
        col for col in display_columns
        if col in data.columns
    ]

    st.dataframe(
        data[available_columns].tail(20),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Download Button
    # -----------------------------------------------------

    st.subheader("📁 Export Data")

    csv = data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=f"{symbol}_{market}_analysis.csv",
        mime="text/csv"
    )
