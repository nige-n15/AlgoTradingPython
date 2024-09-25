import matplotlib.pyplot as plt

def visualize_strategy(data, trades_df):
    plt.figure(figsize=(12, 8))
    plt.plot(data.index, data['Close'], label='ETH Price')
    plt.plot(data.index, data['SMA_Short'], label=f'SMA Short')
    plt.plot(data.index, data['SMA_Long'], label=f'SMA Long')

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

def plot_equity_curve(strategy_data):
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_data.index, strategy_data['Strategy_Returns'].cumsum())
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

def plot_drawdown(strategy_data):
    drawdown = strategy_data['Strategy_Returns'].cumsum() - strategy_data['Strategy_Returns'].cumsum().cummax()
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_data.index, drawdown)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.show()