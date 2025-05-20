import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go


class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover Trading Strategy

    This class implements a simple moving average crossover strategy
    where buy signals are generated when the short-term MA crosses above
    the long-term MA, and sell signals when it crosses below.
    """

    def __init__(self, ticker, short_window=50, long_window=200,
                 start_date=None, end_date=None, initial_capital=100000):
        """
        Initialize the strategy with parameters

        Parameters:
        -----------
        ticker : str
            The stock ticker symbol
        short_window : int
            Short-term moving average window (default: 50 days)
        long_window : int
            Long-term moving average window (default: 200 days)
        start_date : str
            Start date for the backtest in 'YYYY-MM-DD' format
        end_date : str
            End date for the backtest in 'YYYY-MM-DD' format
        initial_capital : float
            Initial capital for the backtest
        """
        self.ticker = ticker
        self.short_window = short_window
        self.long_window = long_window

        # Set default dates if not provided
        if start_date is None:
            self.start_date = (
                datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date

        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.positions = None
        self.portfolio = None

    def fetch_data(self):
        """Fetch historical stock price data"""
        print(
            f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        self.data = yf.download(
            self.ticker, start=self.start_date, end=self.end_date)
        print(f"Retrieved {len(self.data)} data points")

        # Save the data
        os.makedirs('./data', exist_ok=True)
        self.data.to_csv(f'./data/{self.ticker}_data.csv')
        return self.data

    def generate_signals(self):
        """Generate trading signals based on moving average crossover"""
        if self.data is None:
            self.fetch_data()

        print("Generating trading signals...")
        # Create a copy of the data
        self.signals = self.data.copy()

        # Calculate moving averages
        self.signals[f'SMA_{self.short_window}'] = self.signals['Close'].rolling(
            window=self.short_window, min_periods=1).mean()
        self.signals[f'SMA_{self.long_window}'] = self.signals['Close'].rolling(
            window=self.long_window, min_periods=1).mean()

        # Initialize the signal column
        self.signals['Signal'] = 0.0

        # Generate signals: 1.0 for buy, 0.0 for hold/sell
        # Use .loc to avoid SettingWithCopyWarning
        self.signals.loc[self.signals.index[self.short_window:], 'Signal'] = np.where(
            self.signals[f'SMA_{self.short_window}'][self.short_window:] >
            self.signals[f'SMA_{self.long_window}'][self.short_window:],
            1.0, 0.0)

        # Generate trading orders
        self.signals['Position'] = self.signals['Signal'].diff()

        # Save signals
        os.makedirs('./results', exist_ok=True)
        self.signals.to_csv(f'./results/{self.ticker}_signals.csv')

        return self.signals

    def backtest_strategy(self):
        """Backtest the trading strategy"""
        if self.signals is None:
            self.generate_signals()

        print("Backtesting the strategy...")

        # Create a copy of signals DataFrame as a new DataFrame to avoid issues
        self.positions = pd.DataFrame(index=self.signals.index)

        # Copy necessary columns
        self.positions['Close'] = self.signals['Close']
        self.positions['Signal'] = self.signals['Signal']
        self.positions['Position'] = self.signals['Position']

        # Calculate holdings (current position value)
        self.positions['Holdings'] = self.positions['Signal'] * \
            self.positions['Close']

        # Calculate cash position
        position_diff = self.positions['Signal'].diff().fillna(0)
        trade_values = position_diff * self.positions['Close']
        self.positions['Cash'] = self.initial_capital - trade_values.cumsum()

        # Calculate total portfolio value
        self.positions['Total'] = self.positions['Holdings'] + \
            self.positions['Cash']

        # Calculate daily returns
        self.positions['Returns'] = self.positions['Total'].pct_change().fillna(
            0)

        # Calculate metrics
        self.calculate_metrics()

        # Save positions
        self.positions.to_csv(f'./results/{self.ticker}_backtest.csv')

        return self.positions

    def calculate_metrics(self):
        """Calculate performance metrics"""
        if self.positions is None:
            self.backtest_strategy()

        print("Calculating performance metrics...")
        # Calculate daily returns
        daily_returns = self.positions['Returns'].dropna()

        # Calculate metrics
        total_days = len(daily_returns)
        sharpe_ratio = np.sqrt(
            252) * (daily_returns.mean() / daily_returns.std())
        cum_returns = (1 + daily_returns).cumprod().iloc[-1] - 1
        annual_returns = (1 + cum_returns) ** (252 / total_days) - 1

        # Calculate drawdowns
        wealth_index = (1 + daily_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()

        # Calculate win rate
        trades = self.positions[self.positions['Position'] != 0].copy()
        buy_trades = trades[trades['Position'] > 0]
        sell_trades = trades[trades['Position'] < 0]

        # Pair buy and sell trades
        trade_pairs = min(len(buy_trades), len(sell_trades))

        # Match buy and sell pairs to calculate profit
        profits = []
        for i in range(trade_pairs):
            try:
                buy_price = buy_trades.iloc[i]['Close']
                sell_price = sell_trades.iloc[i]['Close']
                profit = (sell_price - buy_price) / buy_price
                profits.append(profit)
            except IndexError:
                pass

        profits = np.array(profits)
        win_rate = len(profits[profits > 0]) / \
            len(profits) if len(profits) > 0 else 0

        # Create a performance metrics dictionary
        self.metrics = {
            'Initial Capital': self.initial_capital,
            'Final Portfolio Value': self.positions['Total'].iloc[-1],
            'Total Return (%)': cum_returns * 100,
            'Annualized Return (%)': annual_returns * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Win Rate (%)': win_rate * 100,
            # Divide by 2 to count buy-sell pairs as one trade
            'Number of Trades': len(trades) // 2,
        }

        # Print metrics
        print("\nPerformance Metrics:")
        for key, value in self.metrics.items():
            print(f"{key}: {value:.2f}")

        # Save metrics
        pd.DataFrame([self.metrics]).to_csv(
            f'./results/{self.ticker}_metrics.csv', index=False)

        return self.metrics

    def plot_strategy(self):
        """Plot the strategy results with buy/sell signals"""
        if self.signals is None:
            self.generate_signals()

        print("Plotting strategy results...")
        # Create a figure
        fig = go.Figure()

        # Add price
        fig.add_trace(go.Scatter(
            x=self.signals.index,
            y=self.signals['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ))

        # Add moving averages
        fig.add_trace(go.Scatter(
            x=self.signals.index,
            y=self.signals[f'SMA_{self.short_window}'],
            mode='lines',
            name=f'{self.short_window}-day SMA',
            line=dict(color='blue', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=self.signals.index,
            y=self.signals[f'SMA_{self.long_window}'],
            mode='lines',
            name=f'{self.long_window}-day SMA',
            line=dict(color='red', width=1)
        ))

        # Add buy signals
        buy_signals = self.signals[self.signals['Position'] > 0]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
            )
        ))

        # Add sell signals
        sell_signals = self.signals[self.signals['Position'] < 0]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
            )
        ))

        # Update layout
        fig.update_layout(
            title=f'{self.ticker} Moving Average Crossover Strategy',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Save the figure
        fig.write_html(f'./results/{self.ticker}_strategy_plot.html')

        # Plot performance
        if self.positions is not None:
            # Create portfolio value plot
            fig2 = go.Figure()

            # Add portfolio value
            fig2.add_trace(go.Scatter(
                x=self.positions.index,
                y=self.positions['Total'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))

            # Update layout
            fig2.update_layout(
                title=f'{self.ticker} Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Save the figure
            fig2.write_html(f'./results/{self.ticker}_portfolio_plot.html')

        print(
            f"Plots saved to ./results/{self.ticker}_strategy_plot.html and ./results/{self.ticker}_portfolio_plot.html")

        return fig


def run_strategy(ticker='AAPL', short_window=50, long_window=200,
                 start_date=None, end_date=None, initial_capital=100000):
    """
    Run the Moving Average Crossover Strategy

    Parameters:
    -----------
    ticker : str
        The stock ticker symbol
    short_window : int
        Short-term moving average window (default: 50 days)
    long_window : int
        Long-term moving average window (default: 200 days)
    start_date : str
        Start date for the backtest in 'YYYY-MM-DD' format
    end_date : str
        End date for the backtest in 'YYYY-MM-DD' format
    initial_capital : float
        Initial capital for the backtest
    """
    # Initialize the strategy
    strategy = MovingAverageCrossoverStrategy(
        ticker=ticker,
        short_window=short_window,
        long_window=long_window,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    # Run the strategy
    strategy.fetch_data()
    strategy.generate_signals()
    strategy.backtest_strategy()
    strategy.plot_strategy()

    return strategy


if __name__ == "__main__":
    # Run the strategy
    strategy = run_strategy(
        ticker='AAPL',  # Change to your preferred ticker
        short_window=50,
        long_window=200,
        start_date='2015-01-01',
        end_date=None,  # Use None for current date
        initial_capital=100000
    )
