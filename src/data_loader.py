# src/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np

def get_stock_data(ticker, start='2018-01-01', end='2024-12-31'):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    return df
