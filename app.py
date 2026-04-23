# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/World_Stock_Prices_Dataset.csv")

    # Fix datetime issues
    df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
    df['Date'] = df['Date'].dt.tz_localize(None)
    df = df.dropna(subset=['Date'])

    # Sort
    df = df.sort_values(by='Date')

    # Feature Engineering
    df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    return df


df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.title("🔍 Filters")

industry = st.sidebar.selectbox("Select Industry", df['Industry_Tag'].unique())

filtered_industry = df[df['Industry_Tag'] == industry]

ticker = st.sidebar.selectbox("Select Ticker", filtered_industry['Ticker'].unique())

filtered = filtered_industry[filtered_industry['Ticker'] == ticker]

filtered = filtered.dropna(subset=['Daily_Return'])

# ------------------ TITLE ------------------
st.title("📊 Global Stock Market Intelligence Dashboard")

# ------------------ KPIs ------------------
col1, col2, col3 = st.columns(3)

col1.metric("Latest Price", f"${filtered['Close'].iloc[-1]:.2f}")
col2.metric("Average Volume", f"{int(filtered['Volume'].mean()):,}")
col3.metric("Max Price", f"${filtered['High'].max():.2f}")

# ------------------ INSIGHT ------------------
st.subheader("🧠 Market Insight")

latest_return = filtered['Daily_Return'].iloc[-1]

if latest_return > 0:
    st.success(f"{ticker} is showing positive momentum 📈")
else:
    st.error(f"{ticker} is under downward pressure 📉")

# ------------------ PRICE TREND ------------------
st.subheader("📈 Price Trend")

fig1 = px.line(filtered, x='Date', y='Close', title="Closing Price")
st.plotly_chart(fig1, use_container_width=True)

# ------------------ MOVING AVERAGE ------------------
st.subheader("📉 Moving Average")

filtered = filtered.sort_values('Date')
filtered['MA_50'] = filtered['Close'].rolling(50).mean()

fig2 = px.line(filtered, x='Date', y=['Close', 'MA_50'],
               title='Price vs 50-Day Moving Average')
st.plotly_chart(fig2, use_container_width=True)

# ------------------ CANDLESTICK ------------------
st.subheader("🕯️ Candlestick Chart")

fig3 = go.Figure(data=[go.Candlestick(
    x=filtered['Date'],
    open=filtered['Open'],
    high=filtered['High'],
    low=filtered['Low'],
    close=filtered['Close']
)])

fig3.update_layout(title="Candlestick View")
st.plotly_chart(fig3, use_container_width=True)

# ------------------ RETURNS ------------------
st.subheader("📊 Daily Returns Distribution")

fig4 = px.histogram(filtered, x='Daily_Return', nbins=50,
                    title="Daily Returns")
st.plotly_chart(fig4, use_container_width=True)

# ------------------ VOLATILITY ------------------
st.subheader("⚠️ Rolling Volatility")

filtered['Volatility'] = filtered['Daily_Return'].rolling(30).std()

fig5 = px.line(filtered, x='Date', y='Volatility',
               title='30-Day Rolling Volatility')
st.plotly_chart(fig5, use_container_width=True)

# ------------------ CORRELATION ------------------
st.subheader("🔗 Correlation Between Stocks")

pivot = df.pivot_table(
    index='Date',
    columns='Ticker',
    values='Close',
    aggfunc='last'
)

corr = pivot.corr()

fig6 = px.imshow(
    corr,
    color_continuous_scale='RdBu_r',
    zmin=-1,
    zmax=1,
    title="Correlation Matrix"
)

st.plotly_chart(fig6, use_container_width=True)

# ------------------ SHARPE RATIO ------------------
st.subheader("📊 Risk-Adjusted Performance (Sharpe Ratio)")

risk_free_rate = 0.05 / 252

sharpe = df.groupby('Ticker')['Daily_Return'].agg(['mean', 'std']).reset_index()
sharpe['Sharpe'] = (sharpe['mean'] - risk_free_rate) / sharpe['std']

top_sharpe = sharpe.sort_values(by='Sharpe', ascending=False).head(10)

st.dataframe(top_sharpe[['Ticker', 'Sharpe']])

# ------------------ MARKET MOVERS ------------------
st.subheader("🚀 Market Movers")

latest_date = df['Date'].max()
latest = df[df['Date'] == latest_date]

top_gainers = latest.sort_values(by='Daily_Return', ascending=False).head(5)
top_losers = latest.sort_values(by='Daily_Return').head(5)

col1, col2 = st.columns(2)

with col1:
    st.write("Top Gainers")
    st.dataframe(top_gainers[['Ticker', 'Daily_Return']])

with col2:
    st.write("Top Losers")
    st.dataframe(top_losers[['Ticker', 'Daily_Return']])