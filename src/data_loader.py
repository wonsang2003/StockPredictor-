import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start='2018-01-01', end='2024-12-31'):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']]
    data['log_returns'] = (data['Close'] / data['Close'].shift(1)).apply(lambda x: np.log(x))
    data.dropna(inplace=True)
    return data
