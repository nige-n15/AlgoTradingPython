import pandas as pd
import numpy as np

class Backtester:
   def __init__(self, initial_balance=2000, stop_loss=0.05, take_profit=0.1, trailing_stop=0.05):
       self.initial_balance = initial_balance
       self.stop_loss = stop_loss
       self.take_profit = take_profit
       self.trailing_stop = trailing_stop

   def calculate_position_size(self, account_balance, signal_strength, volatility, min_trade=50, max_trade=250):
       signal_strength = max(0, min(signal_strength, 1))
       base_position_size = min_trade + (max_trade - min_trade) * signal_strength
       volatility_factor = 1 / (1 + volatility)
       adjusted_position_size = base_position_size * volatility_factor
       position_size = min(adjusted_position_size, account_balance)
       max_position_pct = 0.2
       position_size = min(position_size, account_balance * max_position_pct)
       return position_size

   def backtest_strategy(self, data):
       account_balance = self.initial_balance
       position = 0
       trades = []
       entry_price = None
       highest_price = None

       data['Returns'] = data['Close'].pct_change()
       data['Volatility'] = data['Returns'].rolling(window=20).std()

       for date, row in data.iterrows():
           if position > 0:
               current_value = position * row['Close']
               if highest_price is None or row['Close'] > highest_price:
                   highest_price = row['Close']
               if (current_value <= entry_price * position * (1 - self.stop_loss) or
                   current_value >= entry_price * position * (1 + self.take_profit) or
                   row['Close'] <= highest_price * (1 - self.trailing_stop)):
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

           if position == 0 and row['Signal'] == 2:
               trade_type = 'BUY'
               position_size = self.calculate_position_size(account_balance, row['SignalStrength'], row['Volatility'])
               position = position_size / row['Close']
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

           data.loc[date, 'Strategy_Returns'] = position * row['Returns'] if position > 0 else 0

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
       total_return = (account_balance - self.initial_balance) / self.initial_balance

       return total_return, trades_df, data