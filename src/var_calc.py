import numpy as np

def historical_var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)

def parametric_var(returns, confidence=0.95):
    mean = np.mean(returns)
    std_dev = np.std(returns)
    from scipy.stats import norm
    return norm.ppf(1 - confidence, mean, std_dev)
