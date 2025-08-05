# src/portfolio_opt.py
import numpy as np
import matplotlib.pyplot as plt

def simulate_portfolios(log_returns, n_portfolios=10000, risk_free_rate=0.01):
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    results = np.zeros((3, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        weights = np.random.random(len(log_returns.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = sharpe_ratio

    return results, weights_record

def plot_efficient_frontier(results):
    plt.figure(figsize=(10, 6))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.colorbar(label='Sharpe Ratio')
    plt.grid(True)
    plt.show()
