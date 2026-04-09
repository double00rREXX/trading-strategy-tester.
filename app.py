import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

st.set_page_config(page_title="QuantPro: 20-Year Session Tester", layout="wide")

# --- Institutional UI ---
st.title("📈 Realistic Session Backtester (20-Year Analysis)")
st.markdown("""
*This model reconstructs sessions by analyzing the **Opening Gap** (Asia/Europe) vs. **Intraday Range** (NY).*
""")

# --- Sidebar ---
st.sidebar.header("Execution Settings")
asset = st.sidebar.selectbox("Select Asset", ["ES=F", "NQ=F", "SPY"])
lookback = st.sidebar.selectbox("History", ["20y", "10y", "5y"])
session_to_trade = st.sidebar.radio("Session to Trade", ["NY Session (Open-to-Close)", "ASIA/Overnight (Close-to-Open)"])

st.sidebar.header("Strategy Parameters")
strat_type = st.sidebar.selectbox("Strategy Logic", [
    "Momentum (Follow previous session direction)",
    "Mean Reversion (Fade previous session move)",
    "Volatility Breakout (Standard Deviation)"
])
ci_level = st.sidebar.slider("CI Accuracy (Student's T)", 0.90, 0.99, 0.95)

@st.cache_data
def load_data(ticker, period):
    data = yf.download(ticker, period=period, interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

try:
    df = load_data(asset, lookback)
    
    # --- The "Realistic" Session Reconstruction ---
    # Log returns for mathematical accuracy
    df['NY_Return'] = np.log(df['Close'] / df['Open'])
    df['ASIA_Return'] = np.log(df['Open'] / df['Close'].shift(1))
    
    # Define "Signal" based on the PREVIOUS session's behavior
    if strat_type == "Momentum (Follow previous session direction)":
        # If Asia was UP, buy NY session
        if session_to_trade == "NY Session (Open-to-Close)":
            df['Signal'] = np.where(df['ASIA_Return'] > 0, 1, -1)
        else: # If NY was UP yesterday, buy Asia session today
            df['Signal'] = np.where(df['NY_Return'].shift(1) > 0, 1, -1)
            
    elif strat_type == "Mean Reversion (Fade previous session move)":
        # If Asia was UP, short NY session
        if session_to_trade == "NY Session (Open-to-Close)":
            df['Signal'] = np.where(df['ASIA_Return'] > 0, -1, 1)
        else: # If NY was UP yesterday, short Asia session
            df['Signal'] = np.where(df['NY_Return'].shift(1) > 0, -1, 1)

    elif strat_type == "Volatility Breakout (Standard Deviation)":
        std_window = 20
        df['Vol'] = df['NY_Return'].rolling(std_window).std()
        df['Signal'] = np.where(df['NY_Return'].abs() > df['Vol'], 1, 0)

    # --- Calculation of Strategy Returns ---
    if session_to_trade == "NY Session (Open-to-Close)":
        df['Strategy_Returns'] = df['NY_Return'] * df['Signal']
        benchmark = df['NY_Return']
    else:
        df['Strategy_Returns'] = df['ASIA_Return'] * df['Signal']
        benchmark = df['ASIA_Return']

    df = df.dropna()

    # --- High Accuracy Math: Student's T-Distribution ---
    # We use T-Distribution because financial returns have 'Fat Tails'
    mu = df['Strategy_Returns'].mean()
    se = stats.sem(df['Strategy_Returns']) # Standard Error
    dof = len(df) - 1
    
    # CI for daily expected return
    ci = stats.t.interval(ci_level, dof, loc=mu, scale=se)
    
    # Annualizing results
    ann_return = np.exp(df['Strategy_Returns'].mean() * 252) - 1
    ann_vol = df['Strategy_Returns'].std() * np.sqrt(252)
    sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)

    # --- Display Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualized Return", f"{ann_return:.2%}")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c3.metric("95% CI Lower", f"{np.exp(ci[0]*252)-1:.2%}")
    c4.metric("95% CI Upper", f"{np.exp(ci[1]*252)-1:.2%}")

    # --- Equity Curve Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Strategy_Returns'].cumsum().apply(np.exp), 
                             name="Strategy Equity", line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=df.index, y=benchmark.cumsum().apply(np.exp), 
                             name="Session Benchmark", line=dict(color='gray', dash='dash')))
    
    fig.update_layout(template="plotly_dark", title=f"20-Year {session_to_trade} Performance", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics Deep Dive
    with st.expander("Show Scientific Statistical Model"):
        st.write(f"""
        **Model Details:**
        - **Degrees of Freedom:** {dof}
        - **Daily Mean (Log):** {mu:.6f}
        - **Standard Error:** {se:.6f}
        - **Confidence Interval Method:** Student's T (More accurate for NQ/ES fat-tails than Gaussian models).
        - **Session Split:** NY defined as Open-to-Close. ASIA/London defined as Prev Close-to-Open.
        """)

except Exception as e:
    st.error(f"Error: {e}")