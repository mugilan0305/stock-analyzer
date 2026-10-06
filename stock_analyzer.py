```python
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Indian Stock Analyzer")

# =========================================================
# USER INPUT
# =========================================================

market = st.selectbox(
    "Choose Market",
    ["NSE", "BSE"]
)

symbol = st.text_input(
    "Enter Stock Symbol (e.g., RELIANCE, TCS, INFY)"
).upper().strip()

start_date = st.date_input(
    "Start Date",
    pd.to_datetime("2020-01-01")
)

end_date = st.date_input(
    "End Date",
    datetime.today()
)

# =========================================================
# VALIDATE INPUT
# =========================================================

if start_date >= end_date:
    st.error("❌ Start Date must be before End Date.")
    st.stop()

# =========================================================
# DOWNLOAD DATA
# =========================================================

if symbol:

    ticker = (
        f"{symbol}.NS"
        if market == "NSE"
        else f"{symbol}.BO"
    )

    st.info(
        f"Fetching data for `{ticker}` "
        f"from {start_date} to {end_date}..."
    )

    try:

        # Add one day because Yahoo Finance end date
        # is exclusive
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

        st.error(
            f"❌ Error downloading stock data: {e}"
        )

        st.stop()

    # =====================================================
    # CHECK DATA
    # =====================================================

    if data.empty:

        st.error(
            f"❌ No data found for `{ticker}`."
        )

        st.info(
            "Check the stock symbol and selected market."
        )

        st.stop()

    # =====================================================
    # FIX YFINANCE MULTI-LEVEL COLUMNS
    # =====================================================

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    # Make sure required columns exist

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        st.error(
            f"❌ Missing columns: {missing_columns}"
        )

        st.stop()

    # =====================================================
    # MAKE CLOSE A SINGLE SERIES
    # =====================================================

    close = data["Close"]

    if isinstance(close, pd.DataFrame):

        close = close.iloc[:, 0]

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    # =====================================================
    # TECHNICAL INDICATORS
    # =====================================================

    # -----------------------------------------------------
    # Moving Averages
    # -----------------------------------------------------

    data["MA20"] = (
        close
        .rolling(window=20)
        .mean()
    )

    data["MA50"] = (
        close
        .rolling(window=50)
        .mean()
    )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(window=14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(window=14)
        .mean()
    )

    rs = avg_gain / avg_loss

    data["RSI"] = (
        100 - (100 / (1 + rs))
    )

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

    data["Signal"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    # -----------------------------------------------------
    # Bollinger Bands
    # -----------------------------------------------------

    data["BB_Middle"] = (
        close
        .rolling(window=20)
        .mean()
    )

    rolling_std = (
        close
        .rolling(window=20)
        .std()
    )

    data["BB_Upper"] = (
        data["BB_Middle"]
        + (2 * rolling_std)
    )

    data["BB_Lower"] = (
        data["BB_Middle"]
        - (2 * rolling_std)
    )

    # =====================================================
    # CLEAN DATA BEFORE REGRESSION
    # =====================================================

    regression_data = data[
        ["Close"]
    ].copy()

    regression_data = (
        regression_data
        .replace(
            [float("inf"), float("-inf")],
            pd.NA
        )
        .dropna()
    )

    # =====================================================
    # TREND LINE
    # =====================================================

    data = data.reset_index()

    # Handle Date / Datetime differences

    if "Date" not in data.columns:

        if "Datetime" in data.columns:

            data.rename(
                columns={
                    "Datetime": "Date"
                },
                inplace=True
            )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data["Date_ordinal"] = (
        data["Date"]
        .map(datetime.toordinal)
    )

    # Only use valid rows for regression

    valid_regression = (
        data[
            ["Date_ordinal", "Close"]
        ]
        .dropna()
    )

    if len(valid_regression) >= 2:

        model = LinearRegression()

        model.fit(
            valid_regression[
                ["Date_ordinal"]
            ],
            valid_regression["Close"]
        )

        data["Trend"] = model.predict(
            data[
                ["Date_ordinal"]
            ]
        )

    else:

        data["Trend"] = pd.NA

    # =====================================================
    # CURRENT STOCK INFORMATION
    # =====================================================

    valid_close = close.dropna()

    current_price = float(
        valid_close.iloc[-1]
    )

    if len(valid_close) > 1:

        previous_price = float(
            valid_close.iloc[-2]
        )

        price_change = (
            current_price
            - previous_price
        )

        price_change_pct = (
            price_change
            / previous_price
            * 100
        )

    else:

        price_change = 0
        price_change_pct = 0

    # =====================================================
    # PRICE SUMMARY
    # =====================================================

    st.subheader(
        f"📊 {symbol} ({market})"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Price",
            f"₹{current_price:,.2f}"
        )

    with col2:

        st.metric(
            "Daily Change",
            f"₹{price_change:,.2f}",
            f"{price_change_pct:.2f}%"
        )

    with col3:

        latest_rsi = (
            data["RSI"]
            .dropna()
            .iloc[-1]
        )

        st.metric(
            "RSI",
            f"{latest_rsi:.2f}"
        )

    with col4:

        latest_macd = (
            data["MACD"]
            .dropna()
            .iloc[-1]
        )

        st.metric(
            "MACD",
            f"{latest_macd:.2f}"
        )

    # =====================================================
    # PRICE & MOVING AVERAGES
    # =====================================================

    st.subheader(
        "📊 Stock Price & Moving Averages"
    )

    fig1, ax1 = plt.subplots(
        figsize=(12, 5)
    )

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

    # =====================================================
    # TREND LINE
    # =====================================================

    st.subheader(
        "📈 Trend Line (Linear Regression)"
    )

    fig2, ax2 = plt.subplots(
        figsize=(12, 5)
    )

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

    # =====================================================
    # RSI
    # =====================================================

    st.subheader(
        "🌀 Relative Strength Index (RSI)"
    )

    fig3, ax3 = plt.subplots(
        figsize=(12, 3)
    )

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
        "RSI"
    )

    ax3.set_ylim(
        0,
        100
    )

    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)

    plt.close(fig3)

    # =====================================================
    # MACD
    # =====================================================

    st.subheader(
        "📉 MACD"
    )

    fig4, ax4 = plt.subplots(
        figsize=(12, 3)
    )

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

    ax4.axhline(
        0,
        linestyle="--"
    )

    ax4.set_title(
        "MACD & Signal Line"
    )

    ax4.legend()
    ax4.grid(True)

    st.pyplot(fig4)

    plt.close(fig4)

    # =====================================================
    # BOLLINGER BANDS
    # =====================================================

    st.subheader(
        "📌 Bollinger Bands"
    )

    fig5, ax5 = plt.subplots(
        figsize=(12, 5)
    )

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

    # =====================================================
    # LATEST DATA TABLE
    # =====================================================

    st.subheader(
        "📋 Latest Market Data"
    )

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
        column
        for column in display_columns
        if column in data.columns
    ]

    st.dataframe(
        data[
            available_columns
        ].tail(20),
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.subheader(
        "📁 Export Data"
    )

    csv = data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name=(
            f"{symbol}_{market}_analysis.csv"
        ),
        mime="text/csv"
    )

    st.caption(
        "Technical analysis is for informational purposes "
        "only and is not financial advice."
    )
```
