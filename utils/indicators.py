from ta.trend import SMAIndicator

def add_sma_indicators(data, short_window, long_window):
    data['SMA_Short'] = SMAIndicator(close=data['Close'], window=short_window).sma_indicator()
    data['SMA_Long'] = SMAIndicator(close=data['Close'], window=long_window).sma_indicator()
    data['SMA_Crossover'] = (data['SMA_Short'] > data['SMA_Long']).astype(int)
    return data
