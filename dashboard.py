# dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import get_stock_data
from src.mc_simulation import monte_carlo_simulation
from src.var_calc import historical_var, parametric_var
from src.lstm_model import train_lstm_model
from src.portfolio_opt import simulate_portfolios, plot_efficient_frontier

st.set_page_config(page_title="Finance ML Dashboard", layout="wide")
st.title("📈 ML-Powered Financial Risk Dashboard")

# Sidebar settings
st.sidebar.header("Simulation Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
days = st.sidebar.slider("Simulation Days", 10, 365, 30)
n_sim = st.sidebar.slider("Number of Monte Carlo Simulations", 100, 5000, 1000)

# Load data
st.subheader(f"📊 Loading Data for {ticker}")
data = get_stock_data(ticker)
st.line_chart(data['Close'])

# Monte Carlo
st.subheader("🔁 Monte Carlo Simulation")
returns = data['log_returns']
mu = returns.mean()
sigma = returns.std()
S0 = data['Close'].iloc[-1]

simulations = monte_carlo_simulation(S0, mu, sigma, days=days, n_sim=n_sim)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(simulations, alpha=0.05)
ax.set_title("Monte Carlo Simulated Price Paths")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
st.pyplot(fig)

# Value at Risk
st.subheader("⚠️ Value at Risk (VaR)")
hist_var = historical_var(returns)
param_var = parametric_var(returns)
st.metric(label="Historical VaR (95%)", value=f"{hist_var:.4f}")
st.metric(label="Parametric VaR (95%)", value=f"{param_var:.4f}")

# LSTM Prediction
st.subheader("🤖 LSTM Next-Day Prediction")
series = data['Close'].values
model, scaler = train_lstm_model(series, epochs=5)  # Use fewer epochs for speed
window = 60
input_seq = scaler.transform(series[-window:].reshape(-1,1)).reshape(1, window, 1)
pred = model.predict(input_seq)
pred_price = scaler.inverse_transform(pred)[0][0]
st.metric(label="Predicted Price (Next Day)", value=f"${pred_price:.2f}")
# LSTM prediction chart
last_prices = series[-60:]
extended_prices = np.append(last_prices, pred_price)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(range(60), last_prices, label="Past Prices")
ax3.plot(60, pred_price, 'ro', label="Predicted Next Day")
ax3.set_title(f"{ticker} - LSTM Price Forecast")
ax3.set_xlabel("Days")
ax3.set_ylabel("Price")
ax3.legend()
st.pyplot(fig3)

# Portfolio Optimization
st.subheader("💼 Portfolio Optimization")
tickers = st.multiselect("Choose Tickers", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], default=['AAPL', 'MSFT', 'GOOGL'])

if st.button("Run Portfolio Optimization"):
    log_return_list = []
    for t in tickers:
        df = get_stock_data(t)
        log_return_list.append(df['log_returns'])
    log_returns_df = pd.concat(log_return_list, axis=1)
    log_returns_df.columns = tickers
    results, _ = simulate_portfolios(log_returns_df)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sc = ax2.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5)
    ax2.set_xlabel("Volatility")
    ax2.set_ylabel("Expected Return")
    ax2.set_title("Efficient Frontier")
    plt.colorbar(sc, ax=ax2, label="Sharpe Ratio")
    st.pyplot(fig2)
