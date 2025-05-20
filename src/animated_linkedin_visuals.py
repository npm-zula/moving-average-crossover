"""
Animated LinkedIn-optimized visualizations for the Moving Average Crossover Strategy

This module provides functions for creating animated, visually appealing, 
LinkedIn-friendly visualizations of trading strategies with enhanced styling 
and templates designed for social media sharing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moving_avg_crossover import run_strategy
import visualization_styles as vs


def create_animated_linkedin_strategy_card(ticker='AAPL', short_window=50, long_window=200,
                                           start_date='2018-01-01', end_date=None,
                                           initial_capital=100000, save_path=None):
    """
    Create an animated LinkedIn-optimized visualization card showing strategy performance over time using matplotlib
    """
    print(
        f"Creating animated LinkedIn-optimized visualization for {ticker}...")
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
    metrics = strategy.metrics
    colors = vs.COLORS
    step = max(1, len(signals) // 200)
    frame_dates = signals.index[::step].tolist() + [signals.index[-1]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   sharex=True, gridspec_kw={'height_ratios': [0.7, 0.3]})
    fig.suptitle(f"{ticker} Price with {short_window}/{long_window} MA Crossover",
                 fontsize=vs.THEME['title_font_size'])
    price_line, = ax1.plot([], [], color=colors['price'], label='Price', lw=2)
    short_ma_line, = ax1.plot(
        [], [], color=colors['short_ma'], label=f'{short_window}-day SMA', lw=1.5)
    long_ma_line, = ax1.plot(
        [], [], color=colors['long_ma'], label=f'{long_window}-day SMA', lw=1.5)
    buy_scatter = ax1.scatter(
        [], [], color=colors['buy_signal'], marker='^', s=80, label='Buy Signal')
    sell_scatter = ax1.scatter(
        [], [], color=colors['sell_signal'], marker='v', s=80, label='Sell Signal')
    portfolio_line, = ax2.plot(
        [], [], color=colors['portfolio'], label='Portfolio Value', lw=2)
    ax1.set_ylabel('Price ($)')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_xlabel('Date')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax1.grid(True, color=colors['grid'])
    ax2.grid(True, color=colors['grid'])

    def animate(i):
        current_date = frame_dates[i]
        mask = signals.index <= current_date
        current_signals = signals[mask]
        current_positions = positions[mask]
        buy_signals = current_signals[current_signals['Position'] > 0]
        sell_signals = current_signals[current_signals['Position'] < 0]
        price_line.set_data(current_signals.index, current_signals['Close'])
        short_ma_line.set_data(current_signals.index,
                               current_signals[f'SMA_{short_window}'])
        long_ma_line.set_data(current_signals.index,
                              current_signals[f'SMA_{long_window}'])
        buy_scatter.set_offsets(np.c_[buy_signals.index, buy_signals['Close']]
                                ) if not buy_signals.empty else buy_scatter.set_offsets([])
        sell_scatter.set_offsets(np.c_[sell_signals.index, sell_signals['Close']]
                                 ) if not sell_signals.empty else sell_scatter.set_offsets([])
        portfolio_line.set_data(current_positions.index,
                                current_positions['Total'])
        ax1.set_xlim(signals.index[0], signals.index[-1])
        ax1.set_ylim(signals['Close'].min()*0.95, signals['Close'].max()*1.05)
        ax2.set_ylim(positions['Total'].min()*0.95,
                     positions['Total'].max()*1.05)
        return price_line, short_ma_line, long_ma_line, buy_scatter, sell_scatter, portfolio_line
    anim = FuncAnimation(fig, animate, frames=len(
        frame_dates), interval=100, blit=False)
    if save_path is None:
        save_path = f"./results/{ticker}_animated_linkedin_card_{short_window}_{long_window}.mp4"
    anim.save(save_path, writer='ffmpeg', fps=15)
    print(f"Animated LinkedIn visualization saved to {save_path}")
    plt.close(fig)


def create_animated_strategy_comparison_card(ticker='AAPL', strategies=None,
                                             start_date='2018-01-01', end_date=None,
                                             initial_capital=100000, save_path=None):
    """
    Create an animated LinkedIn-optimized visualization card comparing multiple strategies using matplotlib
    """
    if strategies is None:
        strategies = [
            {'short_window': 5, 'long_window': 20,
                'name': 'Fast (5/20)', 'color': vs.COLORS['strategy1']},
            {'short_window': 20, 'long_window': 50,
                'name': 'Medium (20/50)', 'color': vs.COLORS['strategy3']},
            {'short_window': 50, 'long_window': 200,
                'name': 'Slow (50/200)', 'color': vs.COLORS['strategy4']}
        ]
    print(f"Creating animated strategy comparison card for {ticker}...")
    results = []
    for strategy in strategies:
        short_window = strategy['short_window']
        long_window = strategy['long_window']
        name = strategy.get('name', f"SMA {short_window}/{long_window}")
        strat_result = run_strategy(
            ticker=ticker,
            short_window=short_window,
            long_window=long_window,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        results.append({
            'strategy': strat_result,
            'name': name,
            'short_window': short_window,
            'long_window': long_window,
            'color': strategy.get('color', vs.COLORS['strategy1'])
        })
    reference_signals = results[0]['strategy'].signals
    step = max(1, len(reference_signals) // 200)
    frame_dates = reference_signals.index[::step].tolist(
    ) + [reference_signals.index[-1]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   sharex=True, gridspec_kw={'height_ratios': [0.6, 0.4]})
    fig.suptitle(f"{ticker} Moving Average Crossover Strategy Comparison",
                 fontsize=vs.THEME['title_font_size'])
    price_line, = ax1.plot(
        [], [], color=vs.COLORS['price'], label='Price', lw=2)
    ma_lines = []
    port_lines = []
    for result in results:
        color = result['color']
        name = result['name']
        short_window = result['short_window']
        long_window = result['long_window']
        ma_lines.append((ax1.plot([], [], color=color, lw=1.5, ls='--', label=f'{name} {short_window}d SMA')[0],
                         ax1.plot([], [], color=color, lw=1.5, label=f'{name} {long_window}d SMA')[0]))
        port_lines.append(ax2.plot([], [], color=color, lw=2,
                          label=f'{name} Portfolio')[0])
    ax1.set_ylabel('Price ($)')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_xlabel('Date')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax1.grid(True, color=vs.COLORS['grid'])
    ax2.grid(True, color=vs.COLORS['grid'])

    def animate(i):
        current_date = frame_dates[i]
        mask = reference_signals.index <= current_date
        current_price_data = reference_signals[mask]
        price_line.set_data(current_price_data.index,
                            current_price_data['Close'])
        for idx, result in enumerate(results):
            signals = result['strategy'].signals[mask]
            positions = result['strategy'].positions[mask]
            short_window = result['short_window']
            long_window = result['long_window']
            ma_lines[idx][0].set_data(
                signals.index, signals[f'SMA_{short_window}'])
            ma_lines[idx][1].set_data(
                signals.index, signals[f'SMA_{long_window}'])
            port_lines[idx].set_data(positions.index, positions['Total'])
        ax1.set_xlim(reference_signals.index[0], reference_signals.index[-1])
        ax1.set_ylim(reference_signals['Close'].min(
        )*0.95, reference_signals['Close'].max()*1.05)
        ax2.set_ylim(min([r['strategy'].positions['Total'].min() for r in results])*0.95,
                     max([r['strategy'].positions['Total'].max() for r in results])*1.05)
        return [price_line] + [l for pair in ma_lines for l in pair] + port_lines
    anim = FuncAnimation(fig, animate, frames=len(
        frame_dates), interval=100, blit=False)
    if save_path is None:
        save_path = f"./results/{ticker}_animated_strategy_comparison.mp4"
    anim.save(save_path, writer='ffmpeg', fps=15)
    print(f"Animated strategy comparison visualization saved to {save_path}")
    plt.close(fig)
