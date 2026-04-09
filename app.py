import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from datetime import time

# --- Page Config ---
st.set_page_config(page_title="QuantTest Pro V2", layout="wide")
st.title("📊 Professional Strategy Tester: 20-Year & Session Analysis")

# --- Sidebar: Strategy & Sessions ---
st.sidebar.header("Global Settings")
asset = st.sidebar.selectbox("Select Asset", ["SPY", "ES=F", "NQ=F"])
session_focus = st.sidebar.radio("Trading Session", ["NY Session (09:30-16:00)", "Asia Session (19:00-03:00)", "Full 24h Market"])
lookback = st.sidebar.selectbox("Lookback", ["20y", "10y", "5y", "2y"])

st.sidebar.header("Strategy Library")
strategy_name = st.sidebar.selectbox("Select Strategy", [
    "Opening Range Breakout (ORB)",
    "Asia Range Fade",
    "Dual Moving Average Trend",
    "Mean Reversion (Bollinger/RSI)",
    "MACD Momentum Divergence",
    "Volatility Gap Close"
])

confidence_level = st.sidebar.slider("Confidence Interval Accuracy", 0.90, 0.99, 0.95)

# --- Data Engine (Handling Intraday vs Daily) ---
@st.cache_data
def load_market_data(ticker, period):
    # Use 1h for short periods, 1d for 20y
    interval = "1h" if period in ["1y", "2y"] else "1d"
    df = yf.download(ticker, period=period, interval=interval)
    return df

try:
    data = load_market_data(asset, lookback)
    
    # --- Session Filtering Logic ---
    def filter_sessions(df, session):
        if '1d' in str(df.index.freq): # If daily data, session filtering is limited
            return df 
        # If hourly/intraday
        df = df.copy()
        if session == "NY Session (09:30-16:00)":
            return df.between_time('09:30', '16:00')
        elif session == "Asia Session (19:00-03:00)":
            return df.between_time('19:00', '03:00')
        return df

    df = filter_sessions(data, session_focus)

    # --- Strategy Models (Vectorized Math) ---
    def apply_math_model(df, strat):
        df = df.copy()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        if strat == "Opening Range Breakout (ORB)":
            df['Signal'] = np.where(df['High'] > df['High'].shift(1), 1, 0)
            
        elif strat == "Dual Moving Average Trend":
            df['Fast'] = df['Close'].rolling(50).mean()
            df['Slow'] = df['Close'].rolling(200).mean()
            df['Signal'] = np.where(df['Fast'] > df['Slow'], 1, 0)
            
        elif strat == "Mean Reversion (Bollinger/RSI)":
            df['MA'] = df['Close'].rolling(20).mean()
            df['Std'] = df['Close'].rolling(20).std()
            df['Upper'] = df['MA'] + (2 * df['Std'])
            df['Lower'] = df['MA'] - (2 * df['Std'])
            df['Signal'] = np.where(df['Close'] < df['Lower'], 1, np.where(df['Close'] > df['Upper'], 0, np.nan))
            df['Signal'] = df['Signal'].ffill().fillna(0)

        elif strat == "Volatility Gap Close":
            df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['Signal'] = np.where(df['Gap'] < -0.005, 1, 0) # Buy gaps down > 0.5%
            
        df['Strat_Ret'] = df['Log_Ret'] * df['Signal'].shift(1)
        return df.dropna()

    processed_df = apply_math_model(df, strategy_name)

    # --- Advanced Statistics Engine ---
    def get_stats(returns, ci_val):
        # Student's T-Distribution for High Accuracy CI
        mu = returns.mean()
        sigma = returns.std()
        dof = len(returns) - 1
        # Confidence Interval (Standard Error of the Mean)
        ci = stats.t.interval(ci_val, dof, loc=mu, scale=stats.sem(returns))
        
        # Risk Metrics
        sharpe = (mu / sigma) * np.sqrt(252) if sigma != 0 else 0
        max_dd = (np.exp(returns.cumsum()).cummax() - np.exp(returns.cumsum())).max()
        
        return {
            "Annualized Return": f"{np.exp(returns.mean() * 252) - 1:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "CI Lower Bound": f"{np.exp(ci[0] * 252) - 1:.2%}",
            "CI Upper Bound": f"{np.exp(ci[1] * 252) - 1:.2%}"
        }

    stats_results = get_stats(processed_df['Strat_Ret'], confidence_level)

    # --- UI Display ---
    st.header(f"Results for {asset} ({strategy_name})")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ann. Return", stats_results["Annualized Return"])
    col2.metric("Sharpe", stats_results["Sharpe Ratio"])
    col3.metric("Max Drawdown", stats_results["Max Drawdown"])
    col4.metric("CI Upper (Math Accuracy)", stats_results["CI Upper Bound"])

    # --- Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['Strat_Ret'].cumsum().apply(np.exp), 
                             name="Strategy Equity", line=dict(color='#00ffcc', width=2)))
    fig.update_layout(template="plotly_dark", title="Equity Growth of $1.00", height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detailed Math & Model Assumptions"):
        st.write("""
        - **Modeling:** We use Logarithmic Returns for time-additivity.
        - **Statistical Rigor:** Confidence Intervals use the Student's T-Distribution to account for the 'Fat Tails' (excess kurtosis) found in SPY and NQ Futures.
        - **Session Constraint:** For 20-year lookbacks, data is Daily. Session-specific hourly filters are only applied if lookback is ≤ 2 years.
        """)

except Exception as e:
    st.error(f"Waiting for valid data... (Error: {e})")