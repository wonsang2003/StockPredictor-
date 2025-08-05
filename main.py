# main.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import get_stock_data
from src.mc_simulation import monte_carlo_simulation, plot_simulation
from src.var_calc import historical_var, parametric_var
from src.lstm_model import train_lstm_model
from src.portfolio_opt import simulate_portfolios, plot_efficient_frontier

import warnings
warnings.filterwarnings("ignore")

def run_monte_carlo(data):
    returns = data['log_returns']
    mu = returns.mean()
    sigma = returns.std()
    S0 = data['Close'].iloc[-1]

    simulations = monte_carlo_simulation(S0, mu, sigma)
    plot_simulation(simulations)
    plt.savefig("results/monte_carlo.png")
    plt.close()

    return returns

def run_var(returns):
    hist_var = historical_var(returns)
    param_var = parametric_var(returns)
    print(f"[VaR] Historical (95%): {hist_var:.4f}")
    print(f"[VaR] Parametric (95%): {param_var:.4f}")

def run_lstm(data):
    close_series = data['Close'].values
    model, scaler = train_lstm_model(close_series)

    # Predict next day
    window = 60
    input_seq = scaler.transform(close_series[-window:].reshape(-1, 1)).reshape(1, window, 1)
    pred_scaled = model.predict(input_seq)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    print(f"[LSTM] Predicted next day price: {pred_price:.2f}")

def run_portfolio_optimization(tickers):
    log_return_list = []
    for ticker in tickers:
        df = get_stock_data(ticker)
        log_return_list.append(df['log_returns'])

    log_returns_df = pd.concat(log_return_list, axis=1)
    log_returns_df.columns = tickers

    results, _ = simulate_portfolios(log_returns_df)
    plot_efficient_frontier(results)
    plt.savefig("results/efficient_frontier.png")
    plt.close()

    print(f"[Portfolio] Optimized with {len(tickers)} tickers.")

if __name__ == "__main__":
    print("==== Machine Learning in Finance Pipeline ====\n")

    # Ensure results folder exists
    if not os.path.exists("results"):
        os.makedirs("results")

    print("▶ Loading data for AAPL...")
    data = get_stock_data("AAPL")

    print("\n▶ Running Monte Carlo Simulation...")
    returns = run_monte_carlo(data)

    print("\n▶ Calculating Value at Risk (VaR)...")
    run_var(returns)

    print("\n▶ Training LSTM Model...")
    run_lstm(data)

    print("\n▶ Running Portfolio Optimization...")
    run_portfolio_optimization(["AAPL", "MSFT", "GOOGL", "AMZN"])

    print("\n✅ All tasks completed. Check results/ folder for charts.")
