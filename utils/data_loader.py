import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator


def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    macd = MACD(close=data['Close'])
    rsi = RSIIndicator(close=data['Close'])

    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()
    data['MACD'] = macd.macd_diff()
    data['RSI'] = rsi.rsi()

    return data