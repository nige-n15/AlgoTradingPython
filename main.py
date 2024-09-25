import pandas as pd
import numpy as np
from tqdm import tqdm
from strategies.sma_crossover import SMACrossoverStrategy
from trading.backtester import Backtester
from utils.data_loader import load_and_prepare_data
from utils.indicators import add_sma_indicators
from utils.visualization import visualize_strategy, plot_equity_curve, plot_drawdown

def optimize_parameters(data, sma_short_range, sma_long_range, stop_loss_range, take_profit_range, trailing_stop_range):
    best_return = -np.inf
    best_params = {}

    total_iterations = len(sma_short_range) * len(sma_long_range) * len(stop_loss_range) * len(take_profit_range) * len(trailing_stop_range)

    with tqdm(total=total_iterations, desc="Optimizing") as pbar:
        for sma_short in sma_short_range:
            for sma_long in sma_long_range:
                for stop_loss in stop_loss_range:
                    for take_profit in take_profit_range:
                        for trailing_stop in trailing_stop_range:
                            strategy = SMACrossoverStrategy(sma_short, sma_long)
                            data_copy = strategy.prepare_data(data.copy())
                            signals, signal_strengths = strategy.generate_signals(data_copy)
                            data_copy['Signal'] = signals
                            data_copy['SignalStrength'] = signal_strengths

                            backtester = Backtester(stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop)
                            total_return, _, _ = backtester.backtest_strategy(data_copy)

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
    stop_loss_range = [0.02, 0.03, 0.05, 0.07]
    take_profit_range = [0.05, 0.1, 0.15, 0.2]
    trailing_stop_range = [0.03, 0.05, 0.07, 0.1]

    print("Starting parameter optimization...")
    best_params, best_return = optimize_parameters(data, sma_short_range, sma_long_range,
                                                   stop_loss_range, take_profit_range, trailing_stop_range)
    print(f"Best parameters: {best_params}")
    print(f"Best return during optimization: {best_return:.2%}")

    print("\nPreparing data with best parameters for final backtest...")
    strategy = SMACrossoverStrategy(best_params['sma_short'], best_params['sma_long'])
    data = strategy.prepare_data(data)
    signals, signal_strengths = strategy.generate_signals(data)
    data['Signal'] = signals
    data['SignalStrength'] = signal_strengths

    print("Running final backtest with optimized parameters...")
    backtester = Backtester(stop_loss=best_params['stop_loss'],
                            take_profit=best_params['take_profit'],
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