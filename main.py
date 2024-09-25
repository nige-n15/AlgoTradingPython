import pandas as pd
import numpy as np
from strategies.sma_crossover import SMACrossoverStrategy
from trading.backtester import Backtester
from utils.data_loader import load_and_prepare_data
from utils.indicators import add_sma_indicators
from utils.visualization import visualize_strategy, plot_equity_curve, plot_drawdown
import multiprocessing
from itertools import product
from tqdm import tqdm


def evaluate_params(params, data):
    sma_short, sma_long, stop_loss, take_profit, trailing_stop, adx_threshold = params
    strategy = SMACrossoverStrategy(sma_short, sma_long, adx_threshold=adx_threshold)
    data_copy = strategy.prepare_data(data.copy())
    signals, signal_strengths = strategy.generate_signals(data_copy)
    data_copy['Signal'] = signals
    data_copy['SignalStrength'] = signal_strengths

    backtester = Backtester(stop_loss_multiplier=stop_loss,
                            take_profit_multiplier=take_profit,
                            trailing_stop=trailing_stop)
    total_return, _, _ = backtester.backtest_strategy(data_copy)
    return total_return, params


def optimize_parameters(data, sma_short_range, sma_long_range, stop_loss_range, take_profit_range, trailing_stop_range,
                        adx_threshold_range):
    param_combinations = list(product(sma_short_range, sma_long_range, stop_loss_range,
                                      take_profit_range, trailing_stop_range, adx_threshold_range))

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    results = []
    with tqdm(total=len(param_combinations), desc="Optimizing") as pbar:
        for result in pool.imap_unordered(lambda p: evaluate_params(p, data), param_combinations):
            results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    best_return, best_params = max(results, key=lambda x: x[0])
    best_params = {
        'sma_short': best_params[0],
        'sma_long': best_params[1],
        'stop_loss': best_params[2],
        'take_profit': best_params[3],
        'trailing_stop': best_params[4],
        'adx_threshold': best_params[5]
    }

    return best_params, best_return


# In your main function:
if __name__ == '__main__':
    multiprocessing.freeze_support()  # This is necessary if you're using PyInstaller or similar tools
    main()

def analyze_performance(trades_df, strategy_data):
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['Return'] > 0]
    losing_trades = trades_df[trades_df['Return'] <= 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    avg_win = winning_trades['Return'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['Return'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    max_drawdown = (strategy_data['Strategy_Returns'].cumsum() - strategy_data['Strategy_Returns'].cumsum().cummax()).min()

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
    data = load_and_prepare_data('data/ETH-USD_data_7years.csv')
    # Define parameter ranges for optimization
    sma_short_range = range(5, 30, 5)
    sma_long_range = range(30, 150, 20)
    stop_loss_range = [0.5, 1, 1.5, 2]  # Multipliers for ATR
    take_profit_range = [1, 1.5, 2, 2.5]  # Multipliers for ATR
    trailing_stop_range = [0.02, 0.03, 0.05, 0.07]
    adx_threshold_range = range(20, 35, 5)

    print("Starting parameter optimization...")
    # Define parameter ranges for optimization
    sma_short_range = range(5, 30, 5)
    sma_long_range = range(30, 150, 20)
    stop_loss_range = [0.02, 0.03, 0.05, 0.07]
    take_profit_range = [0.05, 0.1, 0.15, 0.2]
    trailing_stop_range = [0.03, 0.05, 0.07, 0.1]

    print("Starting parameter optimization...")
    best_params, best_return = optimize_parameters(data, sma_short_range, sma_long_range,
                                                   stop_loss_range, take_profit_range,
                                                   trailing_stop_range, adx_threshold_range)
    print(f"Best parameters: {best_params}")
    print(f"Best return during optimization: {best_return:.2%}")


    print("\nPreparing data with best parameters for final backtest...")
    strategy = SMACrossoverStrategy(best_params['sma_short'], best_params['sma_long'],
                                    adx_threshold=best_params['adx_threshold'])
    data = strategy.prepare_data(data)
    signals, signal_strengths = strategy.generate_signals(data)
    data['Signal'] = signals
    data['SignalStrength'] = signal_strengths

    backtester = Backtester(stop_loss_multiplier=best_params['stop_loss'],
                            take_profit_multiplier=best_params['take_profit'],
                            trailing_stop=best_params['trailing_stop'])
    total_return, trades_df, strategy_data = backtester.backtest_strategy(data)

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