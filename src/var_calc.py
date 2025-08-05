# src/var_calc.py
import numpy as np
from scipy.stats import norm

def historical_var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)

def parametric_var(returns, confidence=0.95):
    mu = np.mean(returns)
    sigma = np.std(returns)
    return norm.ppf(1 - confidence, mu, sigma)
