import pandas as pd
import numpy as np
from ta.trend import SMAIndicator


class IchiMokuCloud:
    def __init__(self):
        pass

    def prepare_data(self,data):
        # Tenkan-sen (Conversion Line)
        high9 = data['High'].rolling(window=9).max()
        low9 = data['Low'].rolling(window=9).min()
        data['tenkan_sen'] = (high9 + low9) / 2

        # Kijun-sen (Base Line)
        high26 = data['High'].rolling(window=26).max()
        low26 = data['Low'].rolling(window=26).min()
        data['kijun_sen'] = (high26 + low26) / 2

        # Senkou Span A (Leading Span A)
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        high52 = data['High'].rolling(window=52).max()
        low52 = data['Low'].rolling(window=52).min()
        data['senkou_span_b'] = ((high52 + low52) / 2).shift(26)

        # Chikou Span (Lagging Span)
        data['chikou_span'] = data['Close'].shift(-26)
        return data

    def generate_signals(self, data):
        data['ichimoku_Signal'] = 0

        # Buy Signal
        buy_signal = (
            (data['tenkan_sen'] > data['kijun_sen']) &
            (data['tenkan_sen'].shift(1) <= data['kijun_sen'].shift(1)) &
            (data['Close'] > data['senkou_span_a']) &
            (data['Close'] > data['senkou_span_b']) &
            (data['chikou_span'] > data['Close'].shift(26))
        )

        # Sell Signal
        sell_signal = (
            (data['tenkan_sen'] < data['kijun_sen']) &
            (data['tenkan_sen'].shift(1) >= data['kijun_sen'].shift(1)) &
            (data['Close'] < data['senkou_span_a']) &
            (data['Close'] < data['senkou_span_b']) &
            (data['chikou_span'] < data['Close'].shift(26))
        )

        data.loc[buy_signal, 'ichimoku_Signal'] = 1
        data.loc[sell_signal, 'ichimoku_Signal'] = -1

        return data
