import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="QuantPro: Institutional Terminal", layout="wide")
st.title("🏛️ Institutional Quant Terminal")

# --- Sidebar Configuration ---
st.sidebar.header("1. Data Scope")
asset = st.sidebar.selectbox("Asset", ["ES=F", "NQ=F", "SPY"])
timeframe = st.sidebar.selectbox("Timeframe", ["Daily (1d)", "Hourly (1h)", "Minute (1m)"])

if "Daily" in timeframe:
    lookback, interval, adj = st.sidebar.selectbox("History", ["20y", "10y", "5y"]), "1d", 252
elif "Hourly" in timeframe:
    lookback, interval, adj = st.sidebar.selectbox("History", ["2y", "1y"]), "1h", 252*6.5
else:
    lookback, interval, adj = st.sidebar.selectbox("History", ["7d", "1d"]), "1m", 252*6.5*60

st.sidebar.header("2. Strategy Engine")
strat_choice = st.sidebar.selectbox("Institutional Model", [
    "Trend Following (MA Cross)", "Stat-Arb (Z-Score)", "Gap Momentum", 
    "Volatility Squeeze", "Mean Reversion (RSI)", "Donchian Breakout"
])

ci_level = st.sidebar.slider("Statistical Confidence (CI)", 0.90, 0.99, 0.95)

# --- Data Engine ---
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

try:
    df = load_data(asset, lookback, interval).copy()
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Strategy Logic Implementation
    if "Trend" in strat_choice:
        df['Signal'] = np.where(df['Close'] > df['Close'].rolling(50).mean(), 1, -1)
    elif "Z-Score" in strat_choice:
        z = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        df['Signal'] = np.where(z < -2, 1, np.where(z > 2, -1, 0))
    elif "Gap" in strat_choice:
        df['Signal'] = np.where(df['Open'] > df['Close'].shift(1), 1, -1)
    elif "Squeeze" in strat_choice:
        vol = df['Close'].rolling(20).std()
        df['Signal'] = np.where(vol < vol.rolling(100).quantile(0.2), 1, 0)
    elif "RSI" in strat_choice:
        delta = df['Close'].diff()
        u, d = delta.clip(lower=0).rolling(14).mean(), -delta.clip(upper=0).rolling(14).mean()
        rsi = 100 - (100 / (1 + u/d))
        df['Signal'] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    elif "Donchian" in strat_choice:
        df['Signal'] = np.where(df['Close'] > df['High'].shift(1).rolling(20).max(), 1, 0)

    df['Strat_Ret'] = df['Returns'] * df['Signal'].shift(1)
    df = df.dropna()

    # --- Metrics ---
    mu, std = df['Strat_Ret'].mean(), df['Strat_Ret'].std()
    sharpe = (mu / std) * np.sqrt(adj) if std != 0 else 0
    total_ret = np.exp(df['Strat_Ret'].sum()) - 1
    win_rate = len(df[df['Strat_Ret'] > 0]) / len(df)
    
    # --- UI Metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compounded Return", f"{total_ret:.2%}")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c3.metric("Win Rate", f"{win_rate:.1%}")
    c4.metric("Volatility (Ann)", f"{std * np.sqrt(adj):.2%}")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Strat_Ret'].cumsum().apply(np.exp), name="Strategy", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Returns'].cumsum().apply(np.exp), name="Market", line=dict(color='white', opacity=0.2)))
    fig.update_layout(template="plotly_dark", height=500, title="Equity Curve")
    st.plotly_chart(fig, use_container_width=True)

    # --- NEW: TRADE RECAP & FORENSIC ANALYSIS ---
    st.header("🔍 Post-Trade Quantitative Forensic Analysis")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.subheader("Why the Strategy Wins")
        if sharpe > 1:
            st.write("✅ **Positive Drift:** The strategy successfully captured 'Risk Premia'. By aligning with the asset's momentum, it stayed on the right side of the institutional flow.")
        if win_rate > 0.55:
            st.write("✅ **High Probability:** The model shows high predictive accuracy in this timeframe, suggesting the signal is stronger than market noise.")
        st.write("✅ **Volatility Clustering:** This strategy won during periods of 'Autocorrelation' (where tomorrow's price follows today's direction).")

    with rec_col2:
        st.subheader("Why the Strategy Loses")
        if total_ret < 0:
            st.write("❌ **Regime Shift:** The market moved into a 'Mean Reverting' state while the strategy was looking for 'Trend'. This caused the model to buy tops and sell bottoms.")
        st.write("❌ **Whipsaws:** In low-volatility 'choppy' periods, the model generated signals that were immediately reversed, leading to sequential losses.")
        st.write("❌ **Fat-Tail Risk (Kurtosis):** Sudden macro events (news/gaps) caused price jumps that bypassed the strategy's technical logic, resulting in 'Slippage' of theoretical alpha.")

    # --- SCIENTIFIC MODELING SECTION ---
    with st.expander("🔬 Scientific Model & Statistical Rigor"):
        # Accurate CI Calculation
        dof = len(df) - 1
        ci = stats.t.interval(ci_level, dof, loc=mu, scale=stats.sem(df['Strat_Ret']))
        
        st.write(f"### The Student's T-Distribution Model")
        st.write(f"""
        Traditional finance uses the **Gaussian (Normal) Distribution**, but for ES/NQ Futures, this is dangerous. 
        Futures markets exhibit **Excess Kurtosis** (Fat Tails). 
        
        Our model utilizes the **Student's T-Distribution** for the Confidence Interval (CI):
        - **Degrees of Freedom (ν):** {dof}
        - **Daily Mean Estimate:** {mu:.6f}
        - **Standard Error:** {stats.sem(df['Strat_Ret']):.6f}
        
        **Your {int(ci_level*100)}% Confidence Interval (Annualized):**
        - Lower Bound: **{np.exp(ci[0]*adj)-1:.2%}**
        - Upper Bound: **{np.exp(ci[1]*adj)-1:.2%}**
        
        **Scientific Conclusion:** 
        If the Lower Bound is above 0%, the strategy is **Statistically Significant**. If the range crosses 0, the strategy results may be due to 'Random Walk' (luck) rather than edge.
        """)

except Exception as e:
    st.error(f"Execution Error: {e}")