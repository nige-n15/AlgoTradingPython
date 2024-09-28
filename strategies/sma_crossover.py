import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator


class SMACrossoverStrategy:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def prepare_data(self, data):
        # Calculate short-term and long-term SMAs
        data['SMA_Short'] = SMAIndicator(close=data['Close'], window=self.short_window).sma_indicator()
        data['SMA_Long'] = SMAIndicator(close=data['Close'], window=self.long_window).sma_indicator()
        data['SMA_Crossover'] = data['SMA_Short'] - data['SMA_Long']

        # Calculate RSI for signal strength
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()

        return data

    def generate_signals(self, data):
        data['SMA_Signal'] = 0

        # Generate Buy Signals
        buy_signal = (data['SMA_Crossover'] > 0) & (data['SMA_Crossover'].shift(1) <= 0)
        data.loc[buy_signal, 'SMA_Signal'] = 1  # Buy signal

        # Generate Sell Signals
        sell_signal = (data['SMA_Crossover'] < 0) & (data['SMA_Crossover'].shift(1) >= 0)
        data.loc[sell_signal, 'SMA_Signal'] = -1  # Sell signal

        # Calculate RSI-based signal strength
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        data['SignalStrength'] = (data['RSI'] - 30) / 40  # Normalize RSI to 0-1 range
        data['SignalStrength'] = data['SignalStrength'].clip(0, 1)  # Ensure it's between 0 and 1

        # Fill NaN values in SignalStrength
        data['SignalStrength'] = data['SignalStrength'].fillna(0.5)  # Default to 0.5 if NaN

        return data['SMA_Signal'], data['SignalStrength']