import time
import requests
import urllib.parse
import hashlib
import hmac
import base64
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from ta.trend import SMAIndicator

with open(("keys"),"r") as f:
	lines = f.read().splitlines()
	api_key=lines[0]
	api_sec=lines[1]

api_url="https://api.kraken.com/"

def get_kraken_signature(urlpath, data, secret):
	postdata = urllib.parse.urlencode(data)
	encoded = (str(data['nonce']) + postdata).encode()
	message = urlpath.encode() + hashlib.sha256(encoded).digest()
	mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
	sigdigest = base64.b64encode(mac.digest())
	return sigdigest.decode()

def kraken_request(url_path, data, api_key, api_sec):
	headers={"API-Key":api_key, "API-Sign": get_kraken_signature(url_path, data, api_sec ) }
	resp = requests.post((api_url + url_path), headers=headers, data=data)
	return resp



#resp = kraken_request('/0/private/TradeBalance', {
#	"nonce": str(int(time.time() * 1000000)),
#	"asset": "GBP"
#}, api_key, api_sec)

def market_buy(buy_amt):
	resp = kraken_request("/0/private/AddOrder", {
	"nonce": str(int(time.time() * 1000000)),
	"ordertype": "market",
	"type": "buy",
	"volume": buy_amt,
	"pair": "XBTUSD",
	}, api_key, api_sec)
	return resp

def kraken_cancellall():
	resp = kraken_request("/0/private/CancelAll", {
	"nonce": str(int(time.time() * 1000000)),
	},api_key,api_sec)
	print(resp.json())

def market_sell(sell_amt):
	resp = kraken_request("/0/private/AddOrder", {
	"nonce": str(int(time.time() * 1000000)),
	"ordertype": "market",
	"type": "sell",
	"volume": sell_amt,
	"pair": "XBTUSD"
	}, api_key, api_sec)
	return resp



#
# buy_limit = 21500
# sell_limit = 22000
# buy_amt = 0.001
# sell_amt = 0.001
#
# while True:
# 	current_price = requests.get("https://api.kraken.com/0/public/Ticker?pair=BTCUSD").json()['result']['XXBTZUSD']['c'][0]
# 	position_open=False
# 	if (float(current_price) < buy_limit) and not position_open:
# 		print("buying {buy_amt} of BTC at {current_price}!")
# 		resp = kraken_buy(buy_amt,current_price,"market")
# 		print(current_price)
# 		if not resp.json()['error']:
# 			position_open=True
# 			print('Bought BTC!')
# 		else:
# 			print(f"error:{resp.json()['error']}")
# 	elif float(current_price) > sell_limit:
# 		print(f"Selling {sell_amount} of BTC at {current_price}")
# 		resp = market_sell(sell_amt)
# 		if not resp.json()['error']:
# 			print("Successfully sold BTC!")
# 		else:
# 			print(f"Error: {resp.json()['error']}")
# 	else:
# 		print(f"Current Price: {current_price}, not buying or selling")
# 	time.sleep(10)

def generate_signals_with_strength(model, X, dates):
	hidden_states = model.predict(X)
	state_probs = model.predict_proba(X)

	signals = pd.Series(hidden_states, index=dates, name='Signal')
	signal_strengths = pd.Series(np.max(state_probs, axis=1), index=dates, name='SignalStrength')

	return signals, signal_strengths

def train_hmm(X, n_components=2):
	model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
	model.fit(X)
	return model


# Make sure to update the generate_signals function to work with the modified data structure
def generate_signals(data):
	# Generate buy signals when SMA_50 crosses above SMA_200
	data['Signal'] = np.where(data['SMA_Crossover'].diff() > 0, 2, 0)
	# Generate sell signals when SMA_50 crosses below SMA_200
	data['Signal'] = np.where(data['SMA_Crossover'].diff() < 0, 1, data['Signal'])

	# Calculate RSI-based signal strength
	data['SignalStrength'] = (data['RSI'] - 30) / 40  # Normalize RSI to 0-1 range
	data['SignalStrength'] = data['SignalStrength'].clip(0, 1)  # Ensure it's between 0 and 1

	return data['Signal'], data['SignalStrength']

def calculate_position_size(account_balance, signal_strength, volatility, min_trade=50, max_trade=250):
    # Ensure signal_strength is between 0 and 1
    signal_strength = max(0, min(signal_strength, 1))

    # Calculate base position size
    base_position_size = min_trade + (max_trade - min_trade) * signal_strength

    # Adjust position size based on volatility
    volatility_factor = 1 / (1 + volatility)  # Lower position size for higher volatility
    adjusted_position_size = base_position_size * volatility_factor

    # Ensure the position size doesn't exceed the account balance
    position_size = min(adjusted_position_size, account_balance)

    # Implement maximum position size as a percentage of account balance
    max_position_pct = 0.2  # Maximum 20% of account balance per trade
    position_size = min(position_size, account_balance * max_position_pct)

    return position_size


def backtest_strategy(data, initial_balance=2000, stop_loss=0.05, take_profit=0.1, trailing_stop=0.05):
    account_balance = initial_balance
    position = 0  # Represents the amount of ETH held
    trades = []
    entry_price = None
    highest_price = None

    # Calculate returns
    data['Returns'] = data['Close'].pct_change()

    # Calculate volatility
    data['Volatility'] = data['Returns'].rolling(window=20).std()

    for date, row in data.iterrows():
        if position > 0:  # We're in a trade
            current_value = position * row['Close']

            # Update highest price seen during this trade
            if highest_price is None or row['Close'] > highest_price:
                highest_price = row['Close']

            # Check for stop loss, take profit, or trailing stop
            if (current_value <= entry_price * position * (1 - stop_loss) or
                    current_value >= entry_price * position * (1 + take_profit) or
                    row['Close'] <= highest_price * (1 - trailing_stop)):
                # Close the position
                trade_type = 'SELL'
                trade_return = (current_value - entry_price * position) / (entry_price * position)
                account_balance += current_value
                trades.append({
                    'Date': date,
                    'Type': trade_type,
                    'Price': row['Close'],
                    'Size': current_value,
                    'Amount': position,
                    'Return': trade_return,
                    'Balance': account_balance
                })
                position = 0
                entry_price = None
                highest_price = None

        if position == 0 and row['Signal'] == 2:  # Enter a new position
            trade_type = 'BUY'
            position_size = calculate_position_size(account_balance, row['SignalStrength'], row['Volatility'])
            position = position_size / row['Close']  # Calculate fractional ETH amount
            entry_price = row['Close']
            highest_price = row['Close']
            account_balance -= position_size
            trades.append({
                'Date': date,
                'Type': trade_type,
                'Price': row['Close'],
                'Size': position_size,
                'Amount': position,
                'Balance': account_balance
            })

        # Calculate strategy returns
        data.loc[date, 'Strategy_Returns'] = position * row['Returns'] if position > 0 else 0

    # Close any open position at the end of the backtest
    if position > 0:
        final_value = position * data.iloc[-1]['Close']
        trade_return = (final_value - entry_price * position) / (entry_price * position)
        account_balance += final_value
        trades.append({
            'Date': data.index[-1],
            'Type': 'SELL',
            'Price': data.iloc[-1]['Close'],
            'Size': final_value,
            'Amount': position,
            'Return': trade_return,
            'Balance': account_balance
        })

    trades_df = pd.DataFrame(trades)
    total_return = (account_balance - initial_balance) / initial_balance

    return total_return, trades_df, data

def prepare_data(data):
	# Calculate indicators
	bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
	macd = MACD(close=data['Close'])
	rsi = RSIIndicator(close=data['Close'])
	sma_50 = SMAIndicator(close=data['Close'], window=50)
	sma_200 = SMAIndicator(close=data['Close'], window=200)

	data['BB_high'] = bb.bollinger_hband()
	data['BB_low'] = bb.bollinger_lband()
	data['MACD'] = macd.macd_diff()
	data['RSI'] = rsi.rsi()
	data['SMA_50'] = sma_50.sma_indicator()
	data['SMA_200'] = sma_200.sma_indicator()

	# Add SMA crossover feature
	data['SMA_Crossover'] = (data['SMA_50'] > data['SMA_200']).astype(int)

	return data


def optimize_parameters(data, sma_short_range, sma_long_range, stop_loss_range, take_profit_range, trailing_stop_range):
    best_return = -np.inf
    best_params = {}

    total_iterations = len(sma_short_range) * len(sma_long_range) * len(stop_loss_range) * len(take_profit_range) * len(
        trailing_stop_range)

    with tqdm(total=total_iterations, desc="Optimizing") as pbar:
        for sma_short in sma_short_range:
            for sma_long in sma_long_range:
                for stop_loss in stop_loss_range:
                    for take_profit in take_profit_range:
                        for trailing_stop in trailing_stop_range:
                            # Prepare data with current parameters
                            data_copy = data.copy()
                            data_copy['SMA_Short'] = SMAIndicator(close=data_copy['Close'],
                                                                  window=sma_short).sma_indicator()
                            data_copy['SMA_Long'] = SMAIndicator(close=data_copy['Close'],
                                                                 window=sma_long).sma_indicator()
                            data_copy['SMA_Crossover'] = (data_copy['SMA_Short'] > data_copy['SMA_Long']).astype(int)

                            # Generate signals
                            signals, signal_strengths = generate_signals(data_copy)
                            data_copy['Signal'] = signals
                            data_copy['SignalStrength'] = signal_strengths

                            # Calculate volatility
                            data_copy['Volatility'] = data_copy['Close'].pct_change().rolling(window=20).std()

                            # Backtest with current parameters
                            total_return, _, _ = backtest_strategy(data_copy, initial_balance=2000,
                                                                   stop_loss=stop_loss,
                                                                   take_profit=take_profit,
                                                                   trailing_stop=trailing_stop)

                            if total_return > best_return:
                                best_return = total_return
                                best_params = {
                                    'sma_short': sma_short,
                                    'sma_long': sma_long,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'trailing_stop': trailing_stop
                                }

                            pbar.update(1)

    return best_params, best_return

def analyze_performance(trades_df, strategy_data):
    # Calculate additional metrics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['Return'] > 0]
    losing_trades = trades_df[trades_df['Return'] <= 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    avg_win = winning_trades['Return'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['Return'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    max_drawdown = (
                strategy_data['Strategy_Returns'].cumsum() - strategy_data['Strategy_Returns'].cumsum().cummax()).min()

    # Print results
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average win: {avg_win:.2%}")
    print(f"Average loss: {avg_loss:.2%}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_data.index, strategy_data['Strategy_Returns'].cumsum())
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

    # Plot drawdown
    drawdown = strategy_data['Strategy_Returns'].cumsum() - strategy_data['Strategy_Returns'].cumsum().cummax()
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_data.index, drawdown)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.show()

def visualize_strategy(data, trades_df):
	plt.figure(figsize=(12, 8))
	plt.plot(data.index, data['Close'], label='ETH Price')
	plt.plot(data.index, data['SMA_50'], label='SMA 50')
	plt.plot(data.index, data['SMA_200'], label='SMA 200')

	for _, trade in trades_df.iterrows():
		if trade['Type'] == 'BUY':
			plt.scatter(trade['Date'], trade['Price'], color='g', marker='^', s=100)
		else:  # SELL
			plt.scatter(trade['Date'], trade['Price'], color='r', marker='v', s=100)

	plt.title('ETH Price with SMA Crossover Strategy')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()
	plt.tight_layout()
	plt.savefig('strategy_visualization.png')
	plt.show()
def main():
    # Load and prepare data
    data = pd.read_csv('data/ETH-USD_data_7years.csv', index_col='Date', parse_dates=True)
    data = prepare_data(data)

    # Define parameter ranges for optimization
    sma_short_range = range(5, 30, 5)
    sma_long_range = range(30, 150, 20)
    stop_loss_range = [0.02, 0.03, 0.05, 0.07]
    take_profit_range = [0.05, 0.1, 0.15, 0.2]
    trailing_stop_range = [0.03, 0.05, 0.07, 0.1]

    print("Starting parameter optimization...")
    # Optimize parameters
    best_params, best_return = optimize_parameters(data, sma_short_range, sma_long_range,
                                                   stop_loss_range, take_profit_range, trailing_stop_range)
    print(f"Best parameters: {best_params}")
    print(f"Best return during optimization: {best_return:.2%}")

    print("\nPreparing data with best parameters for final backtest...")
    # Prepare data with best parameters
    data['SMA_Short'] = SMAIndicator(close=data['Close'], window=best_params['sma_short']).sma_indicator()
    data['SMA_Long'] = SMAIndicator(close=data['Close'], window=best_params['sma_long']).sma_indicator()
    data['SMA_Crossover'] = (data['SMA_Short'] > data['SMA_Long']).astype(int)

    signals, signal_strengths = generate_signals(data)
    data['Signal'] = signals
    data['SignalStrength'] = signal_strengths

    # Calculate volatility
    data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()

    print("Running final backtest with optimized parameters...")
    # Run final backtest
    total_return, trades_df, strategy_data = backtest_strategy(data, initial_balance=2000,
                                                               stop_loss=best_params['stop_loss'],
                                                               take_profit=best_params['take_profit'],
                                                               trailing_stop=best_params['trailing_stop'])

    print(f"\nFinal backtest results:")
    print(f"Total return: {total_return:.2%}")
    print(f"Number of trades: {len(trades_df)}")

    # Analyze and display results
    analyze_performance(trades_df, strategy_data)

    # Save results
    strategy_data.to_csv('strategy_data.csv')
    trades_df.to_csv('trades.csv')
    print("\nDetailed strategy data saved to 'strategy_data.csv'")
    print("Trade log saved to 'trades.csv'")

    # Visualize the strategy
    visualize_strategy(data, trades_df)

if __name__ == "__main__":
    main()