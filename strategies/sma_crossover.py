import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange


class SMACrossoverStrategy:
    def __init__(self, short_window, long_window, adx_window=14, adx_threshold=25):
        self.short_window = short_window
        self.long_window = long_window
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold

    def prepare_data(self, data):
        data['SMA_Short'] = SMAIndicator(close=data['Close'], window=self.short_window).sma_indicator()
        data['SMA_Long'] = SMAIndicator(close=data['Close'], window=self.long_window).sma_indicator()
        data['SMA_Crossover'] = (data['SMA_Short'] > data['SMA_Long']).astype(int)

        # Add ADX
        adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=self.adx_window)
        data['ADX'] = adx.adx()

        # Add ATR for dynamic stop-loss and take-profit
        data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'],
                                       window=14).average_true_range()

        return data

    def generate_signals(self, data):
        data['Signal'] = np.where(data['SMA_Crossover'].diff() > 0, 2, 0)  # Buy signal
        data['Signal'] = np.where(data['SMA_Crossover'].diff() < 0, 1, data['Signal'])  # Sell signal

        # Add ADX filter
        data['Signal'] = np.where((data['Signal'] == 2) & (data['ADX'] > self.adx_threshold), 2, data['Signal'])
        data['Signal'] = np.where((data['Signal'] == 1) & (data['ADX'] > self.adx_threshold), 1, data['Signal'])

        # Calculate RSI-based signal strength
        data['SignalStrength'] = (data['RSI'] - 30) / 40  # Normalize RSI to 0-1 range
        data['SignalStrength'] = data['SignalStrength'].clip(0, 1)  # Ensure it's between 0 and 1

        return data['Signal'], data['SignalStrength']