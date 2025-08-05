# src/lstm_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def create_dataset(series, window_size=60):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

def train_lstm_model(series, window_size=60, epochs=10):
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1))

    X, y = create_dataset(scaled_series, window_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=False, input_shape=(window_size, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)

    return model, scaler
