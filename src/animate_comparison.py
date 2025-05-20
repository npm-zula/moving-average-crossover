import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moving_avg_crossover import run_strategy
import visualization_styles as vs


def create_strategy_comparison_animation(ticker='AAPL', strategies=None,
                                         start_date='2018-01-01', end_date=None,
                                         initial_capital=100000, save_path=None):
    """
    Create an animated comparison of different Moving Average Crossover strategies using matplotlib
    """
    if strategies is None:
        strategies = [
            {'short_window': 5, 'long_window': 20,
                'color': vs.COLORS['strategy1']},
            {'short_window': 10, 'long_window': 30,
                'color': vs.COLORS['strategy2']},
            {'short_window': 20, 'long_window': 50,
                'color': vs.COLORS['strategy3']},
            {'short_window': 50, 'long_window': 200,
                'color': vs.COLORS['strategy4']}
        ]

    print(f"Creating animated strategy comparison for {ticker}...")

    # Run all strategies to get data
    strategy_results = []
    for strategy_config in strategies:
        short_window = strategy_config['short_window']
        long_window = strategy_config['long_window']
        color = strategy_config['color']
        strategy = run_strategy(
            ticker=ticker,
            short_window=short_window,
            long_window=long_window,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        strategy_results.append({
            'short_window': short_window,
            'long_window': long_window,
            'color': color,
            'strategy': strategy,
            'signals': strategy.signals,
            'positions': strategy.positions,
            'name': f"SMA({short_window},{long_window})"
        })

    common_dates = strategy_results[0]['signals'].index
    step = max(1, len(common_dates) // 300)
    frame_dates = common_dates[::step].tolist() + [common_dates[-1]]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9),
                                        sharex=True, gridspec_kw={'height_ratios': [0.4, 0.2, 0.4]})
    fig.suptitle(f"{ticker} Moving Average Crossover Strategy Comparison",
                 fontsize=vs.THEME['title_font_size'])

    # Prepare lines for each strategy
    price_line, = ax1.plot(
        [], [], color=vs.COLORS['price'], label='Price', lw=2)
    ma_lines = []
    pos_lines = []
    port_lines = []
    for strat in strategy_results:
        color = strat['color']
        name = strat['name']
        ma_lines.append((ax1.plot([], [], color=color, lw=1.5, ls='--', label=f'{name} Short MA')[0],
                         ax1.plot([], [], color=color, lw=1.5, label=f'{name} Long MA')[0]))
        pos_lines.append(ax2.plot([], [], color=color,
                         lw=2, label=f'{name} Position')[0])
        port_lines.append(ax3.plot([], [], color=color,
                          lw=2, label=f'{name} Return')[0])

    ax1.set_ylabel('Price ($)')
    ax2.set_ylabel('Position')
    ax3.set_ylabel('Portfolio Return (Starting=100)')
    ax3.set_xlabel('Date')
    ax2.set_ylim(-0.1, 1.1)
    ax1.legend(loc='upper left')
    ax3.legend(loc='upper left')
    ax1.grid(True, color=vs.COLORS['grid'])
    ax2.grid(True, color=vs.COLORS['grid'])
    ax3.grid(True, color=vs.COLORS['grid'])

    price_data = strategy_results[0]['signals']['Close']

    def animate(i):
        current_date = frame_dates[i]
        mask = common_dates <= current_date
        current_dates = common_dates[mask]
        price_line.set_data(current_dates, price_data[mask])
        for idx, strat in enumerate(strategy_results):
            short_window = strat['short_window']
            long_window = strat['long_window']
            signals = strat['signals'][mask]
            positions = strat['positions'][mask]
            ma_lines[idx][0].set_data(
                current_dates, signals[f'SMA_{short_window}'])
            ma_lines[idx][1].set_data(
                current_dates, signals[f'SMA_{long_window}'])
            pos_lines[idx].set_data(current_dates, signals['Signal'])
            norm_port = 100 * positions['Total'] / initial_capital
            port_lines[idx].set_data(current_dates, norm_port)
        ax1.set_xlim(common_dates[0], common_dates[-1])
        ax1.set_ylim(price_data.min()*0.95, price_data.max()*1.05)
        ax3.set_ylim(90, 110 + max([100 * strat['positions']
                     ['Total'].max() / initial_capital for strat in strategy_results]))
        return [price_line] + [l for pair in ma_lines for l in pair] + pos_lines + port_lines

    anim = FuncAnimation(fig, animate, frames=len(
        frame_dates), interval=100, blit=False)

    if save_path is None:
        save_path = f"./results/{ticker}_animated_strategy_comparison.mp4"
    anim.save(save_path, writer='ffmpeg', fps=15)
    print(f"Comparison animation saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Create animated strategy comparison
    strategies = [
        {'short_window': 5, 'long_window': 20, 'color': 'green'},
        {'short_window': 10, 'long_window': 30, 'color': 'blue'},
        {'short_window': 20, 'long_window': 50, 'color': 'purple'},
        {'short_window': 50, 'long_window': 200, 'color': 'red'}
    ]

    create_strategy_comparison_animation(
        ticker='AAPL',
        strategies=strategies,
        start_date='2020-01-01',  # Using more recent data for clearer visualization
        end_date=None
    )
