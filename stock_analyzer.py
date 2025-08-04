import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📈 Indian Stock Analyzer")

# --- User Input ---
market = st.selectbox("Choose Market", ["NSE", "BSE"])
symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE, TATAMOTORS)").upper()
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", datetime.today())

if symbol:
    ticker = f"{symbol}.NS" if market == "NSE" else f"{symbol}.BO"
    
    st.info(f"Fetching data for `{ticker}` from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        st.error("❌ No data found. Please check the symbol or date range.")
        st.stop()
    
    # --- Technical Indicators ---
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()

    # Trend Line
    data = data.reset_index()
    data['Date_ordinal'] = pd.to_datetime(data['Date']).map(datetime.toordinal)
    model = LinearRegression()
    model.fit(data[['Date_ordinal']], data['Close'])
    data['Trend'] = model.predict(data[['Date_ordinal']])

    # --- Price & MAs Plot ---
    st.subheader("📊 Stock Price & Moving Averages")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(data['Date'], data['Close'], label='Close Price')
    ax1.plot(data['Date'], data['MA20'], label='MA20', color='red')
    ax1.plot(data['Date'], data['MA50'], label='MA50', color='green')
    ax1.set_title(f"{symbol} Price & Moving Averages")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # --- Trend Line Plot ---
    st.subheader("📈 Trend Line (Linear Regression)")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(data['Date'], data['Close'], label='Actual Price')
    ax2.plot(data['Date'], data['Trend'], label='Trend Line', linestyle='--', color='orange')
    ax2.set_title(f"{symbol} Price Trend")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # --- RSI Plot ---
    st.subheader("🌀 Relative Strength Index (RSI)")
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(data['Date'], data['RSI'], label='RSI', color='purple')
    ax3.axhline(70, color='red', linestyle='--')
    ax3.axhline(30, color='green', linestyle='--')
    ax3.set_title("RSI (Overbought >70, Oversold <30)")
    ax3.grid(True)
    st.pyplot(fig3)

    # --- MACD Plot ---
    st.subheader("📉 MACD")
    fig4, ax4 = plt.subplots(figsize=(12, 3))
    ax4.plot(data['Date'], data['MACD'], label='MACD', color='blue')
    ax4.plot(data['Date'], data['Signal'], label='Signal Line', color='orange')
    ax4.set_title("MACD & Signal Line")
    ax4.grid(True)
    st.pyplot(fig4)

    # --- Bollinger Bands Plot ---
    st.subheader("📌 Bollinger Bands")
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    ax5.plot(data['Date'], data['Close'], label='Close')
    ax5.plot(data['Date'], data['BB_Middle'], label='Middle Band')
    ax5.plot(data['Date'], data['BB_Upper'], label='Upper Band', linestyle='--')
    ax5.plot(data['Date'], data['BB_Lower'], label='Lower Band', linestyle='--')
    ax5.set_title("Bollinger Bands")
    ax5.legend()
    ax5.grid(True)
    st.pyplot(fig5)

    # --- Download Button ---
    st.subheader("📁 Export Data")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name=f"{symbol}_{market}_analysis.csv", mime='text/csv')
