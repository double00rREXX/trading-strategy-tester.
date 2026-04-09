import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="QuantTest Pro", layout="wide")
st.title("📈 QuantTest Pro: SPY & Futures Strategy")

asset = st.sidebar.selectbox("Select Asset", ["SPY", "ES=F", "NQ=F"])
strat = st.sidebar.selectbox("Strategy", ["SMA Crossover", "RSI Mean Reversion"])
timeframe = st.sidebar.selectbox("Lookback", ["1y", "2y", "5y"])

@st.cache_data
def get_data(ticker, p):
    return yf.download(ticker, period=p)

data = get_data(asset, timeframe)

# Heavy Math Logic
df = data.copy()
df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))

if strat == "SMA Crossover":
    df['S'] = np.where(df['Close'].rolling(50).mean() > df['Close'].rolling(200).mean(), 1, 0)
else:
    delta = df['Close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rsi = 100 - (100 / (1 + up/down))
    df['S'] = np.where(rsi < 30, 1, np.where(rsi > 70, 0, np.nan))
    df['S'] = df['S'].ffill().fillna(0)

df['Strat_Ret'] = df['Returns'] * df['S'].shift(1)
df = df.dropna()

# Accurate Stats
avg = df['Strat_Ret'].mean()
ci = stats.t.interval(0.95, len(df)-1, loc=avg, scale=stats.sem(df['Strat_Ret']))

st.metric("Total Strategy Return", f"{np.exp(df['Strat_Ret'].sum())-1:.2%}")
st.write(f"95% Confidence Interval (Daily): {ci[0]:.4f} to {ci[1]:.4f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Strat_Ret'].cumsum().apply(np.exp), name="Strategy"))
st.plotly_chart(fig)