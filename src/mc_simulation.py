# src/mc_simulation.py
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_simulation(S0, mu, sigma, days=30, n_sim=1000):
    dt = 1
    simulations = np.zeros((days, n_sim))

    for sim in range(n_sim):
        prices = np.zeros(days)
        prices[0] = S0
        for t in range(1, days):
            shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
            prices[t] = prices[t-1] * np.exp(shock)
        simulations[:, sim] = prices

    return simulations

def plot_simulation(simulations):
    plt.figure(figsize=(10, 6))
    plt.plot(simulations, alpha=0.05, color='blue')
    plt.title("Monte Carlo Simulation of Stock Prices")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()
