import numpy as np
import pandas as pd


class Backtester:
    def __init__(self, initial_balance=2000, stop_loss=0.05, take_profit=0.1, trailing_stop=0.05):
        self.initial_balance = initial_balance
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop

    def calculate_position_size(self, account_balance, signal_strength, volatility, min_trade=650, max_trade=1000):
        # Ensure signal_strength is a valid number
        if np.isnan(signal_strength):
            print("Warning: signal_strength is NaN, setting to default value of 1.")
            signal_strength = 1
        signal_strength = max(0, min(signal_strength, 1))

        # Ensure volatility is a valid number
        if np.isnan(volatility):
            print("Warning: volatility is NaN, setting to default value of 0.")
            volatility = 0

        # Calculate base position size
        base_position_size = min_trade + (max_trade - min_trade) * signal_strength

        # Calculate volatility factor
        volatility_factor = 1 / (1 + volatility) if volatility != 0 else 1

        # Calculate adjusted position size
        adjusted_position_size = base_position_size * volatility_factor

        # Limit position size to account balance and max position percentage
        position_size = min(adjusted_position_size, account_balance * 0.4, account_balance)

        # Check for NaN or invalid position_size
        if np.isnan(position_size) or position_size <= 0:
            print("Warning: position_size is invalid (NaN or non-positive), setting to zero.")
            position_size = 0  # Set to 0 to prevent trading

        return position_size

    def backtest_strategy(self, data, signal_column='Signal', signal_strength_column='SignalStrength'):
        account_balance = self.initial_balance
        position = 0
        trades = []
        entry_price = None
        highest_price = None

        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        #data['Volatility'].fillna(method='bfill', inplace=True)
        data['Volatility'] = data['Volatility'].fillna(0)


        for date, row in data.iterrows():
            if position > 0:
                current_value = position * row['Close']
                if highest_price is None or row['Close'] > highest_price:
                    highest_price = row['Close']
                if (current_value <= entry_price * position * (1 - self.stop_loss) or
                    current_value >= entry_price * position * (1 + self.take_profit) or
                    row['Close'] <= highest_price * (1 - self.trailing_stop) or
                    row[signal_column] == -1):
                    trade_type = 'SELL'
                    trade_return = (current_value - entry_price * position) / (entry_price * position) if entry_price * position != 0 else 0
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

            if position == 0 and row[signal_column] == 1:
                trade_type = 'BUY'
                original_account_balance = account_balance  # Save the original balance

                signal_strength = row.get(signal_strength_column, 1)
                if np.isnan(signal_strength):
                    print(f"Warning: signal_strength is NaN on date {date}, setting to default value of 1.")
                    signal_strength = 1
                volatility = row['Volatility']
                if np.isnan(volatility):
                    print(f"Warning: volatility is NaN on date {date}, setting to default value of 0.")
                    volatility = 0
                position_size = self.calculate_position_size(account_balance, signal_strength, volatility)

                if position_size == 0:
                    print(f"Warning: position_size is zero on date {date}, skipping trade.")
                    continue  # Skip this iteration

                # Ensure account_balance is valid before subtraction
                if np.isnan(account_balance):
                    print(f"Error: account_balance is NaN before buying on date {date}, skipping trade.")
                    continue  # Skip this iteration

                account_balance -= position_size

                # Check if account_balance became invalid after subtraction
                if np.isnan(account_balance) or account_balance < 0:
                    print(f"Error: account_balance became invalid after buying on date {date}, restoring previous balance.")
                    account_balance = original_account_balance  # Restore previous balance
                    continue  # Skip this iteration

                position = position_size / row['Close']
                if np.isnan(position) or position <= 0:
                    print(f"Error: position is invalid on date {date}, restoring account_balance and skipping trade.")
                    account_balance = original_account_balance  # Restore previous balance
                    continue  # Skip this iteration

                entry_price = row['Close']
                highest_price = row['Close']
                trades.append({
                    'Date': date,
                    'Type': trade_type,
                    'Price': row['Close'],
                    'Size': position_size,
                    'Amount': position,
                    'Return': np.nan,
                    'Balance': account_balance
                })

            data.loc[date, 'Strategy_Returns'] = position * row['Returns'] if position > 0 else 0

        # Close any open positions at the end
        if position > 0:
            final_value = position * data.iloc[-1]['Close']
            trade_return = (final_value - entry_price * position) / (entry_price * position) if entry_price * position != 0 else 0
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
        total_return = (account_balance - self.initial_balance) / self.initial_balance

        return total_return, trades_df, data