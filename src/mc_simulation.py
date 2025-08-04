import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_simulation(S0, mu, sigma, days=30, n_sim=1000):
    """
    S0: Initial stock price
    mu: Daily return
    sigma: Daily volatility
    days: Days to simulate
    n_sim: Number of simulations
    """
    dt = 1
    simulations = np.zeros((days, n_sim))

    for sim in range(n_sim):
        prices = [S0]
        for _ in range(1, days):
            random_shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
            price = prices[-1] * np.exp(random_shock)
            prices.append(price)
        simulations[:, sim] = prices

    return simulations

def plot_simulation(simulations):
    plt.figure(figsize=(10, 6))
    plt.plot(simulations, alpha=0.1, color='blue')
    plt.title("Monte Carlo Simulation of Stock Prices")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()
