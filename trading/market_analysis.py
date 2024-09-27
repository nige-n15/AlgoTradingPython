import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import ta
from datetime import datetime
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from datetime import timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer



def add_technical_indicators(df):
    df = df.copy()
    # Calculate SMA
    df['SMA_20'] = SMAIndicator(df['price'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['price'], window=50).sma_indicator()
    # Calculate EMA
    df['EMA_20'] = EMAIndicator(df['price'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(df['price'], window=50).ema_indicator()
    # Calculate RSI
    df['RSI_14'] = RSIIndicator(df['price'], window=14).rsi()
    # Calculate MACD
    macd = MACD(df['price'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    return df

def identify_buy_signals(df):
    df = df.copy()
    df['Buy_Signal'] = False

    # Bullish SMA Crossover
    df['SMA_Crossover'] = df['SMA_20'] > df['SMA_50']
    df['Prev_SMA_Crossover'] = df['SMA_Crossover'].shift(1)
    df.loc[(df['SMA_Crossover'] == True) & (df['Prev_SMA_Crossover'] == False), 'Buy_Signal'] = True

    # RSI Oversold Condition
    df['RSI_Oversold'] = df['RSI_14'] < 30
    df['Buy_Signal'] = df['Buy_Signal'] | df['RSI_Oversold']

    # MACD Bullish Crossover
    df['MACD_Crossover'] = df['MACD'] > df['MACD_Signal']
    df['Prev_MACD_Crossover'] = df['MACD_Crossover'].shift(1)
    df.loc[(df['MACD_Crossover'] == True) & (df['Prev_MACD_Crossover'] == False), 'Buy_Signal'] = True

    return df


def fetch_market_data(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
    response = requests.get(url, params={'localization': 'false', 'tickers': 'false', 'market_data': 'true', 'community_data': 'false', 'developer_data': 'false', 'sparkline': 'false'})
    data = response.json()
    market_cap = data['market_data']['market_cap']['usd']
    total_volume = data['market_data']['total_volume']['usd']
    return market_cap, total_volume


def analyze_sentiment(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
    response = requests.get(url)
    data = response.json()
    sentiment_analyzer = SentimentIntensityAnalyzer()
    description = data['description']['en']
    sentiment_score = sentiment_analyzer.polarity_scores(description)
    return sentiment_score


def fetch_historical_data(coin_id, days):
    """
    Fetch historical market data for a given cryptocurrency from CoinGecko.

    Parameters:
        coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., 'bitcoin').
        days (int): Number of days of data to retrieve.

    Returns:
        DataFrame: A pandas DataFrame containing the historical data.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    print(data)
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)

    return df

def plot_coin_data(df, coin_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df['price'], label='Price')
    plt.plot(df['SMA_20'], label='SMA 20')
    plt.plot(df['SMA_50'], label='SMA 50')
    plt.title(f"{coin_name.capitalize()} Price and SMA")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(df['RSI_14'], label='RSI 14', color='purple')
    plt.axhline(30, linestyle='--', color='grey')
    plt.axhline(70, linestyle='--', color='grey')
    plt.title(f"{coin_name.capitalize()} RSI")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(df['MACD'], label='MACD', color='red')
    plt.plot(df['MACD_Signal'], label='Signal Line', color='blue')
    plt.bar(df.index, df['MACD_Diff'], label='MACD Histogram', color='grey')
    plt.title(f"{coin_name.capitalize()} MACD")
    plt.legend()
    plt.show()
coins = ['bitcoin', 'ethereum', 'cardano', 'ripple', 'solana']
days = 30
market_data = {}
for coin in coins:
    df = fetch_historical_data(coin, days)
    market_data[coin] = df
    print(f"Fetched data for {coin}")

for coin in market_data:
    market_data[coin] = add_technical_indicators(market_data[coin])
    print(f"Calculated technical indicators for {coin}")

# Apply buy signal identification
for coin in market_data:
    market_data[coin] = identify_buy_signals(market_data[coin])
    print(f"Identified buy signals for {coin}")


potential_investments = []

for coin, df in market_data.items():
    latest_data = df.iloc[-1]
    if latest_data['Buy_Signal']:
        print(f"Potential Buy Signal for {coin.capitalize()} on {latest_data.name}")
        potential_investments.append({
            'coin': coin,
            'date': latest_data.name,
            'price': latest_data['price']
        })

if not potential_investments:
    print("No potential investments found based on the current criteria.")

# Plot data for coins with potential buy signals
for investment in potential_investments:
    coin = investment['coin']
    df = market_data[coin]
    plot_coin_data(df, coin)

# Example usage:
for investment in potential_investments:
    coin = investment['coin']
    market_cap, total_volume = fetch_market_data(coin)
    print(f"{coin.capitalize()} Market Cap: ${market_cap:,.2f}, Total Volume: ${total_volume:,.2f}")

nltk.download('vader_lexicon')

# Example usage:
for investment in potential_investments:
    coin = investment['coin']
    sentiment = analyze_sentiment(coin)
    print(f"{coin.capitalize()} Sentiment Score: {sentiment}")
