import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moving_avg_crossover import MovingAverageCrossoverStrategy, run_strategy
import visualization_styles as vs
import matplotlib.dates as mdates
import argparse


def create_animated_strategy_visualization(ticker='AAPL', short_window=50, long_window=200,
                                           start_date='2018-01-01', end_date=None,
                                           initial_capital=100000, save_path=None):
    """
    Create an animated visualization of the Moving Average Crossover strategy using matplotlib
    """
    print(
        f"Creating animated visualization for {ticker} with {short_window}-day and {long_window}-day moving averages...")

    # Run the strategy to get the data
    strategy = run_strategy(
        ticker=ticker,
        short_window=short_window,
        long_window=long_window,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    signals = strategy.signals
    positions = strategy.positions
    colors = vs.COLORS

    step = max(1, len(signals) // 300)
    frame_dates = signals.index[::step].tolist() + [signals.index[-1]]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9),
                                        sharex=True, gridspec_kw={'height_ratios': [0.5, 0.2, 0.3]})
    fig.suptitle(f"{ticker} Moving Average Crossover Strategy Animation",
                 fontsize=vs.THEME['title_font_size'])

    # Prepare lines
    price_line, = ax1.plot([], [], color=colors['price'], label='Price', lw=2)
    short_ma_line, = ax1.plot(
        [], [], color=colors['short_ma'], label=f'{short_window}-day SMA', lw=1.5)
    long_ma_line, = ax1.plot(
        [], [], color=colors['long_ma'], label=f'{long_window}-day SMA', lw=1.5)
    buy_scatter = ax1.scatter(
        [], [], color=colors['buy_signal'], marker='^', s=80, label='Buy Signal')
    sell_scatter = ax1.scatter(
        [], [], color=colors['sell_signal'], marker='v', s=80, label='Sell Signal')
    signal_line, = ax2.plot(
        [], [], color=colors['signal_line'], label='Position', lw=2)
    portfolio_line, = ax3.plot(
        [], [], color=colors['portfolio'], label='Portfolio Value', lw=2)

    ax1.set_ylabel('Price ($)')
    ax2.set_ylabel('Position')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.set_xlabel('Date')
    ax2.set_ylim(-0.1, 1.1)
    ax1.legend(loc='upper left')
    ax3.legend(loc='upper left')
    ax1.grid(True, color=colors['grid'])
    ax2.grid(True, color=colors['grid'])
    ax3.grid(True, color=colors['grid'])

    def animate(i):
        current_date = frame_dates[i]
        mask = signals.index <= current_date
        current_signals = signals[mask]
        current_positions = positions[mask]
        buy_signals = current_signals[current_signals['Position'] > 0]
        sell_signals = current_signals[current_signals['Position'] < 0]
        # Price and MAs
        price_line.set_data(current_signals.index, current_signals['Close'])
        short_ma_line.set_data(current_signals.index,
                               current_signals[f'SMA_{short_window}'])
        long_ma_line.set_data(current_signals.index,
                              current_signals[f'SMA_{long_window}'])
        # Buy/Sell signals
        if not buy_signals.empty:
            buy_x = mdates.date2num(buy_signals.index)
            buy_y = buy_signals['Close'].values
            buy_scatter.set_offsets(np.column_stack([buy_x, buy_y]))
        else:
            buy_scatter.set_offsets(np.empty((0, 2)))
        if not sell_signals.empty:
            sell_x = mdates.date2num(sell_signals.index)
            sell_y = sell_signals['Close'].values
            sell_scatter.set_offsets(np.column_stack([sell_x, sell_y]))
        else:
            sell_scatter.set_offsets(np.empty((0, 2)))
        # Signal line
        signal_line.set_data(current_signals.index, current_signals['Signal'])
        # Portfolio value
        portfolio_line.set_data(current_positions.index,
                                current_positions['Total'])
        # Set axis limits only if there is data in the current frame
        if not current_signals.empty:
            ax1.set_xlim(current_signals.index[0], current_signals.index[-1])
            min_close = float(current_signals['Close'].min())
            max_close = float(current_signals['Close'].max())
            ax1.set_ylim(min_close*0.95, max_close*1.05)
        if not current_positions.empty:
            min_total = float(current_positions['Total'].min())
            max_total = float(current_positions['Total'].max())
            ax3.set_ylim(min_total*0.95, max_total*1.05)
        return price_line, short_ma_line, long_ma_line, buy_scatter, sell_scatter, signal_line, portfolio_line

    anim = FuncAnimation(fig, animate, frames=len(
        frame_dates), interval=100, blit=False)
    # Use MP4 if ffmpeg is available, otherwise fallback to GIF
    if save_path is None:
        save_path = f"./results/{ticker}_animated_strategy_{short_window}_{long_window}.mp4"
    try:
        anim.save(save_path, writer='ffmpeg', fps=15)
    except (ValueError, RuntimeError):
        save_path = save_path.replace('.mp4', '.gif')
        anim.save(save_path, writer='pillow', fps=10)
    print(f"Animation saved to {save_path}")
    plt.close(fig)


def create_performance_summary(strategy, ticker, short_window, long_window, save_path):
    """Create a summary chart with performance metrics"""

    # Extract metrics
    metrics = strategy.metrics

    # Create a figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Cumulative Returns",
            "Monthly Returns Heatmap",
            "Drawdown",
            "Performance Metrics"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # Calculate cumulative returns
    returns = strategy.positions['Returns']
    cum_returns = (1 + returns).cumprod()

    # Add cumulative returns trace
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Cumulative Return'
        ),
        row=1, col=1
    )

    # Calculate drawdown
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    # Add drawdown trace
    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns,
            mode='lines',
            line=dict(color='red', width=2),
            name='Drawdown',
            fill='tozeroy'
        ),
        row=2, col=1
    )

    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns_matrix = monthly_returns.groupby(
        [monthly_returns.index.year, monthly_returns.index.month]).first().unstack()

    # Convert to proper format for heatmap
    years = monthly_returns_matrix.index.tolist()
    months = monthly_returns_matrix.columns.tolist()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                   'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create month labels and values for heatmap
    month_labels = []
    monthly_values = []

    for year in years:
        for month in months:
            if not pd.isna(monthly_returns_matrix.loc[year, month]):
                month_labels.append(f"{month_names[month-1]} {year}")
                monthly_values.append(monthly_returns_matrix.loc[year, month])

    # Add monthly returns heatmap
    fig.add_trace(
        go.Heatmap(
            z=monthly_returns_matrix.values,
            x=month_names,
            y=years,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Monthly Return"),
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>"
        ),
        row=1, col=2
    )

    # Add performance metrics table
    metrics_table = pd.DataFrame({
        'Metric': [
            'Initial Capital',
            'Final Portfolio Value',
            'Total Return (%)',
            'Annualized Return (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)',
            'Number of Trades'
        ],
        'Value': [
            f"${metrics['Initial Capital']:,.2f}",
            f"${metrics['Final Portfolio Value']:,.2f}",
            f"{metrics['Total Return (%)']:.2f}%",
            f"{metrics['Annualized Return (%)']:.2f}%",
            f"{metrics['Sharpe Ratio']:.2f}",
            f"{metrics['Max Drawdown (%)']:.2f}%",
            f"{metrics['Win Rate (%)']:.2f}%",
            f"{metrics['Number of Trades']:.0f}"
        ]
    })

    # Add table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color=vs.TABLE_STYLE['header']['fill_color'],
                align='left',
                font=dict(
                    size=vs.TABLE_STYLE['header']['font']['size'],
                    color=vs.TABLE_STYLE['header']['font']['color'],
                    family=vs.TABLE_STYLE['header']['font']['family']
                ),
                height=vs.TABLE_STYLE['header']['height']
            ),
            cells=dict(
                values=[metrics_table['Metric'], metrics_table['Value']],
                fill_color=vs.TABLE_STYLE['cells']['fill_color'],
                align='left',
                font=dict(
                    size=vs.TABLE_STYLE['cells']['font']['size'],
                    color=vs.TABLE_STYLE['cells']['font']['color'],
                    family=vs.TABLE_STYLE['cells']['font']['family']
                ),
                height=vs.TABLE_STYLE['cells']['height'],
                line=vs.TABLE_STYLE['cells']['line']
            )
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"{ticker} Performance Summary - {short_window}/{long_window} Day MA Crossover",
        height=800,
        width=1200,
        showlegend=False,
        plot_bgcolor=vs.LAYOUT['plot_bgcolor'],
        paper_bgcolor=vs.LAYOUT['paper_bgcolor'],
        font=vs.LAYOUT['font'],
    )

    # Apply theme
    vs.apply_theme_to_figure(fig)
    vs.create_watermark(fig)

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)

    # Save the figure
    fig.write_html(save_path)
    print(f"Performance summary saved to {save_path}")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate MA Crossover Strategy")
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol')
    parser.add_argument('--short-window', type=int, default=20,
                        help='Short-term moving average window')
    parser.add_argument('--long-window', type=int, default=50,
                        help='Long-term moving average window')
    parser.add_argument('--start-date', type=str,
                        default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float,
                        default=100000, help='Initial capital for backtest')
    args = parser.parse_args()
    create_animated_strategy_visualization(
        ticker=args.ticker,
        short_window=args.short_window,
        long_window=args.long_window,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital
    )
