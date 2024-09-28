import pandas as pd
import numpy as np
import itertools
import multiprocessing
from itertools import product
from tqdm import tqdm
import logging
from strategies.ichimoku_cloud import IchiMokuCloud
from strategies.sma_crossover import SMACrossoverStrategy
from trading.backtester import Backtester
from utils.data_loader import load_and_prepare_data
from utils.indicators import add_sma_indicators
from utils.visualization import visualize_strategy, plot_equity_curve, plot_drawdown

def evaluate_parameters(args):
    data, sma_short, sma_long, stop_loss, take_profit, trailing_stop = args

    # Prepare data
    strategy = SMACrossoverStrategy(sma_short, sma_long)
    data_prepared = strategy.prepare_data(data.copy())
    signals, signal_strengths = strategy.generate_signals(data_prepared)
    data_prepared['Signal'] = signals
    data_prepared['SignalStrength'] = signal_strengths

    # Backtest
    backtester = Backtester(stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop)
    total_return, _, _ = backtester.backtest_strategy(data_prepared, signal_column='Signal', signal_strength_column='SignalStrength')

    return (total_return, {'sma_short': sma_short, 'sma_long': sma_long,
                           'stop_loss': stop_loss, 'take_profit': take_profit,
                           'trailing_stop': trailing_stop})

def optimize_parameters(data, sma_short_range, sma_long_range, stop_loss_range, take_profit_range, trailing_stop_range):
    # Generate all possible combinations
    parameter_combinations = list(itertools.product(
        sma_short_range,
        sma_long_range,
        stop_loss_range,
        take_profit_range,
        trailing_stop_range
    ))

    # Filter out invalid combinations (e.g., sma_short >= sma_long)
    parameter_combinations = [comb for comb in parameter_combinations if comb[0] < comb[1]]

    # Create a list of arguments for each process
    tasks = [(data, sma_short, sma_long, stop_loss, take_profit, trailing_stop)
             for sma_short, sma_long, stop_loss, take_profit, trailing_stop in parameter_combinations]

    # Use multiprocessing Pool
    with multiprocessing.Pool() as pool:
        results = pool.map(evaluate_parameters, tasks)

    # Find the best result
    best_return = float('-inf')
    best_params = None

    for total_return, params in results:
        if total_return > best_return:
            best_return = total_return
            best_params = params

    return best_params, best_return
def analyze_performance(trades_df, strategy_data):
    # Filter only SELL trades
    print(trades_df)
    if len(trades_df) == 0:
        return
    sell_trades = trades_df[trades_df['Type'] == 'SELL']

    total_trades = len(sell_trades)
    winning_trades = sell_trades[sell_trades['Return'] > 0]
    losing_trades = sell_trades[sell_trades['Return'] <= 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    avg_win = winning_trades['Return'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['Return'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    max_drawdown = (
        strategy_data['Strategy_Returns'].cumsum() - strategy_data['Strategy_Returns'].cumsum().cummax()
    ).min()

    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average win: {avg_win:.2%}")
    print(f"Average loss: {avg_loss:.2%}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Maximum drawdown: {max_drawdown:.2%}")

    plot_equity_curve(strategy_data)
    plot_drawdown(strategy_data)

def main():
    # Load and prepare data
    # Load and prepare data
    data = load_and_prepare_data('data/ETH-USD_data_7years.csv')
    data.index = pd.to_datetime(data.index)

    # Filter data to the last year's data
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(years=1)
    data = data.loc[start_date:end_date]
    print(f"Data filtered from {start_date.date()} to {end_date.date()}")

    # Define parameter ranges for optimization
    sma_short_range = range(5, 30, 5)
    sma_long_range = range(30, 150, 20)
    stop_loss_range = [0.02, 0.03, 0.05, 0.07]
    take_profit_range = [0.15, 0.19, 0.25, 0.32]
    trailing_stop_range = [0.07, 0.09, 0.13, 0.19]

    print("Starting parameter optimization...")
    best_params, best_return = optimize_parameters(
        data,
        sma_short_range,
        sma_long_range,
        stop_loss_range,
        take_profit_range,
        trailing_stop_range
    )
    print(f"Best parameters: {best_params}")
    print(f"Best return during optimization: {best_return:.2%}")

    print("\nPreparing data with best parameters for final backtest...")
    strategy = SMACrossoverStrategy(best_params['sma_short'], best_params['sma_long'])
    data = strategy.prepare_data(data)
    signals, signal_strengths = strategy.generate_signals(data)
    data['SMA_Signal'] = signals
    data['SignalStrength'] = signal_strengths

    ichimoku = IchiMokuCloud()
    data = ichimoku.prepare_data(data)
    data = ichimoku.generate_signals(data)  # Adds 'ichimoku_Signal'
    print("SMA Signal Counts:")
    print(data['SMA_Signal'].value_counts())

    print("\nIchimoku Signal Counts:")
    print(data['ichimoku_Signal'].value_counts())
    # Remove NaN values
    data.dropna(inplace=True)

    # Combine signals using OR condition
    data['final_signal'] = 0
    data.loc[(data['SMA_Signal'] == 1) | (data['ichimoku_Signal'] == 1), 'final_signal'] = 1
    data.loc[(data['SMA_Signal'] == -1) | (data['ichimoku_Signal'] == -1), 'final_signal'] = -1

    print("Running final backtest with optimized parameters...")
    backtester = Backtester(
        stop_loss=best_params['stop_loss'],
        take_profit=best_params['take_profit'],
        trailing_stop=best_params['trailing_stop']
    )

    # Specify the signal column when calling backtest_strategy
    total_return, trades_df, strategy_data = backtester.backtest_strategy(
        data,
        signal_column='final_signal',
        signal_strength_column='SignalStrength'  # This is optional
    )
    print(trades_df.head())
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
    visualize_strategy(data, trades_df, signal_column='final_signal')

if __name__ == "__main__":
    main()