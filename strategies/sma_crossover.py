import pandas as pd
import numpy as np
from ta.trend import SMAIndicator


class SMACrossoverStrategy:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def prepare_data(self, data):
        data['SMA_Short'] = SMAIndicator(close=data['Close'], window=self.short_window).sma_indicator()
        data['SMA_Long'] = SMAIndicator(close=data['Close'], window=self.long_window).sma_indicator()
        data['SMA_Crossover'] = (data['SMA_Short'] > data['SMA_Long']).astype(int)
        return data

    def generate_signals(self, data):
        data['Signal'] = np.where(data['SMA_Crossover'].diff() > 0, 2, 0)  # Buy signal
        data['Signal'] = np.where(data['SMA_Crossover'].diff() < 0, 1, data['Signal'])  # Sell signal

        # Calculate RSI-based signal strength
        data['SignalStrength'] = (data['RSI'] - 30) / 40  # Normalize RSI to 0-1 range
        data['SignalStrength'] = data['SignalStrength'].clip(0, 1)  # Ensure it's between 0 and 1

        return data['Signal'], data['SignalStrength']