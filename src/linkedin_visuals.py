"""
LinkedIn-optimized visualizations for the Moving Average Crossover Strategy

This module provides functions for creating visually appealing, LinkedIn-friendly
visualizations of trading strategies with enhanced styling and templates designed
for social media sharing.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from moving_avg_crossover import MovingAverageCrossoverStrategy, run_strategy
import visualization_styles as vs
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_linkedin_strategy_card(ticker='AAPL', short_window=50, long_window=200,
                                  start_date='2018-01-01', end_date=None,
                                  initial_capital=100000, save_path=None):
    """
    Create a LinkedIn-optimized visualization card showing strategy performance

    This generates a compact, visually appealing summary of a trading strategy
    designed specifically for sharing on LinkedIn.

    Parameters:
    -----------
    ticker : str
        The stock ticker symbol
    short_window : int
        Short-term moving average window
    long_window : int
        Long-term moving average window
    start_date : str
        Start date for the backtest in 'YYYY-MM-DD' format
    end_date : str
        End date for the backtest in 'YYYY-MM-DD' format
    initial_capital : float
        Initial capital for the backtest
    save_path : str
        Path to save the visualization HTML file
    """
    logger.info("Creating LinkedIn-optimized visualization for %s (%s/%s)...",
                ticker, short_window, long_window)

    # Run the strategy
    strategy = run_strategy(
        ticker=ticker,
        short_window=short_window,
        long_window=long_window,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    # Get key data
    signals = strategy.signals
    # positions = strategy.positions # Removed unused variable
    metrics = strategy.metrics

    # Create a figure with 2x2 subplots layout
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "indicator"}, {"type": "indicator"}]
        ],
        vertical_spacing=0.1,
        subplot_titles=(
            f"{ticker} Price with {short_window}/{long_window} MA Crossover", "", "")
    )

    # Add price and MA traces
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals['Close'],
            mode='lines',
            line=dict(color=vs.COLORS['price'], width=2),
            name='Price'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals[f'SMA_{short_window}'],
            mode='lines',
            line=dict(color=vs.COLORS['short_ma'], width=1.5),
            name=f'{short_window}-day SMA'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals[f'SMA_{long_window}'],
            mode='lines',
            line=dict(color=vs.COLORS['long_ma'], width=1.5),
            name=f'{long_window}-day SMA'
        ),
        row=1, col=1
    )

    # Add buy signals
    buy_signals = signals[signals['Position'] > 0]
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            marker=dict(color=vs.COLORS['buy_signal'],
                        size=10, symbol='triangle-up'),
            name='Buy Signal'
        ),
        row=1, col=1
    )

    # Add sell signals
    sell_signals = signals[signals['Position'] < 0]
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            marker=dict(color=vs.COLORS['sell_signal'],
                        size=10, symbol='triangle-down'),
            name='Sell Signal'
        ),
        row=1, col=1
    )

    # Add indicator for total return
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=metrics['Final Portfolio Value'],
            number={"prefix": "$", "valueformat": ",.2f"},
            delta={"reference": initial_capital,
                   "valueformat": ".2%", "relative": True},
            title={
                "text": "Portfolio Value<br><span style='font-size:0.8em;color:gray'>Final Value</span>"},
            domain={'row': 1, 'column': 0}
        ),
        row=2, col=1
    )

    # Add indicator for annualized return
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=metrics['Annualized Return (%)'] / 100,
            number={"valueformat": ".2%"},
            title={"text": "Annualized<br>Return"},
            domain={'row': 1, 'column': 1}
        ),
        row=2, col=2
    )

    # Update layout with LinkedIn-friendly styling
    fig.update_layout(
        title={
            'text': f"<b>{ticker} Moving Average Crossover Strategy</b><br><span style='font-size:0.9em;'>Short MA: {short_window} days, Long MA: {long_window} days</span>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=650,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.55,
            xanchor="center",
            x=0.5
        )
    )

    # Apply consistent styling
    vs.apply_theme_to_figure(fig)
    vs.create_watermark(fig)

    # Add LinkedIn-specific branding and call-to-action
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="<b>Follow for more Trading Strategy Insights!</b><br>Strategy Metrics: "
             f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f} | "
             f"Max Drawdown: {metrics['Max Drawdown (%)']:.2f}% | "
             f"Win Rate: {metrics['Win Rate (%)']:.2f}%",
        showarrow=False,
        font=dict(size=12),
        bordercolor=vs.COLORS['grid'],
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(250, 250, 250, 0.8)",
        align="center"
    )

    # Save to file
    if save_path is None:
        save_path = f"./results/{ticker}_linkedin_card_{short_window}_{long_window}.html"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    logger.info("LinkedIn visualization saved to %s", save_path)

    return fig


def create_strategy_comparison_card(ticker='AAPL', strategies=None,
                                    start_date='2018-01-01', end_date=None,
                                    initial_capital=100000, save_path=None):
    """
    Create a LinkedIn-optimized comparison card of different MA strategies

    Parameters:
    -----------
    ticker : str
        The stock ticker symbol
    strategies : list of dict
        List of strategy configurations with short_window and long_window
    start_date : str
        Start date for the backtest in 'YYYY-MM-DD' format
    end_date : str
        End date for the backtest in 'YYYY-MM-DD' format
    initial_capital : float
        Initial capital for the backtest
    save_path : str
        Path to save the visualization HTML file
    """
    if strategies is None:
        # Use default from visualization_styles
        strategies = vs.LINKEDIN_COMPARISON_STRATEGIES

    logger.info(
        "Creating LinkedIn-optimized strategy comparison for %s...", ticker)

    # Run all strategies
    strategy_results = []
    for _, config in enumerate(strategies):  # Replaced i with _
        short_window = config['short_window']
        long_window = config['long_window']
        name = config.get('name', f'SMA({short_window},{long_window})')
        color = config.get('color', vs.get_strategy_color(
            short_window, long_window))

        logger.info("Running strategy: %s...", name)
        strategy_obj = run_strategy(  # Renamed to avoid conflict with strategies list
            ticker=ticker,
            short_window=short_window,
            long_window=long_window,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )

        strategy_results.append({
            'name': name,
            'color': color,
            'strategy': strategy_obj,
            'signals': strategy_obj.signals,
            'positions': strategy_obj.positions,
            'metrics': strategy_obj.metrics
        })

    # Create a figure with 2 subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"{ticker} Strategy Performance Comparison",
            "Portfolio Value Over Time"
        ),
        row_heights=[0.3, 0.7]
    )

    # Add total returns bar chart to the first subplot
    returns = [result['metrics']
               ['Total Return (%)'] for result in strategy_results]
    names = [result['name'] for result in strategy_results]
    colors = [result['color'] for result in strategy_results]

    fig.add_trace(
        go.Bar(
            x=names,
            y=returns,
            marker_color=colors,
            text=[f"{ret:.2f}%" for ret in returns],
            textposition='auto',
            name='Total Return'
        ),
        row=1, col=1
    )

    # Add portfolio value lines to the second subplot
    for result in strategy_results:
        positions = result['positions']
        # Normalize portfolio value to start at 100 for comparison
        normalized_portfolio = 100 * positions['Total'] / initial_capital

        fig.add_trace(
            go.Scatter(
                x=positions.index,
                y=normalized_portfolio,
                mode='lines',
                line=dict(color=result['color'], width=2),
                name=f"{result['name']} (Return: {result['metrics']['Total Return (%)']:.2f}%)"
            ),
            row=2, col=1
        )

    # Add buy/hold benchmark
    first_price = strategy_results[0]['signals']['Close'].iloc[0]
    last_price = strategy_results[0]['signals']['Close'].iloc[-1]
    buy_hold_return = (last_price / first_price - 1) * 100

    fig.add_trace(
        go.Scatter(
            # Use signals index for x-axis
            x=strategy_results[0]['signals'].index,
            # Normalized buy & hold
            y=100 * (strategy_results[0]['signals']['Close'] / first_price),
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name=f"Buy & Hold (Return: {buy_hold_return:.2f}%)"
        ),
        row=2, col=1
    )

    # Create a metrics comparison table for annotation
    table_html = vs.TABLE_HTML_STYLE
    table_html += "<table class='metrics-table'>"
    table_html += "<tr><th>Strategy</th><th>Total Return</th><th>Annual Return</th>"
    table_html += "<th>Sharpe Ratio</th><th>Max Drawdown</th><th>Win Rate</th></tr>"

    for result in strategy_results:
        metrics = result['metrics']
        color = result['color']
        name = result['name']

        total_return = metrics['Total Return (%)']
        annual_return = metrics['Annualized Return (%)']

        table_html += "<tr>"  # Not an f-string
        table_html += f"<td class='strategy-name' style='color:{color};'>{name}</td>"
        table_html += f"<td class=\"{'positive-value' if total_return > 0 else 'negative-value'}\">{total_return:.2f}%</td>"
        table_html += f"<td class=\"{'positive-value' if annual_return > 0 else 'negative-value'}\">{annual_return:.2f}%</td>"
        table_html += f"<td>{metrics['Sharpe Ratio']:.2f}</td>"
        table_html += f"<td>{metrics['Max Drawdown (%)']:.2f}%</td>"
        table_html += f"<td>{metrics['Win Rate (%)']:.2f}%</td>"
        table_html += "</tr>"  # Not an f-string

    # Add buy and hold to the table
    table_html += "<tr>"  # Not an f-string
    # Not an f-string
    table_html += "<td class='strategy-name' style='color:gray;'>Buy & Hold</td>"
    table_html += f"<td class=\"{'positive-value' if buy_hold_return > 0 else 'negative-value'}\">{buy_hold_return:.2f}%</td>"

    date_diff_days = (
        strategy_results[0]['signals'].index[-1] - strategy_results[0]['signals'].index[0]).days
    if date_diff_days == 0:  # Avoid division by zero if start and end date are the same
        annual_buy_hold = 0.0
    else:
        annual_buy_hold = (((1 + buy_hold_return/100) **
                           (365 / date_diff_days)) - 1) * 100

    table_html += f"<td class=\"{'positive-value' if annual_buy_hold > 0 else 'negative-value'}\">{annual_buy_hold:.2f}%</td>"
    table_html += "<td>-</td><td>-</td><td>-</td>"  # Not an f-string
    table_html += "</tr>"  # Not an f-string
    table_html += "</table>"  # Not an f-string

    # Update layout
    fig.update_layout(
        title={
            'text': f"<b>{ticker} Moving Average Crossover Strategy Comparison</b>",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=800,
        width=1000,
        showlegend=True,
        yaxis_title="Total Return (%)",
        yaxis2_title="Portfolio Value (Starting=100)",
    )

    # Apply consistent styling
    vs.apply_theme_to_figure(fig)
    vs.create_watermark(fig)

    # Add metrics table annotation
    fig.add_annotation(
        x=0.5,
        y=-0.20,  # Adjusted y to prevent overlap if watermark is also low
        xref="paper",
        yref="paper",
        text=f"<b>Strategy Performance Metrics</b><br>{table_html}",
        showarrow=False,
        align="center"
    )

    # Add LinkedIn-specific call-to-action
    fig.add_annotation(
        x=0.5,
        y=-0.33,  # Adjusted y
        xref="paper",
        yref="paper",
        text="<b>✨ Follow for more Trading Strategy Insights! ✨</b><br>"
             "Leave a comment with your favorite strategy or parameter combination!",
        showarrow=False,
        font=dict(size=14),
        bordercolor=vs.COLORS['grid'],
        borderwidth=1,
        borderpad=8,
        bgcolor="rgba(250, 250, 250, 0.8)",
        align="center"
    )

    # Save to file
    if save_path is None:
        save_path = f"./results/{ticker}_linkedin_comparison_card.html"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    logger.info("LinkedIn comparison card saved to %s", save_path)

    return fig

# Main CLI function (previously in create_linkedin_visuals.py)


def main_cli():
    parser = argparse.ArgumentParser(
        description='Create LinkedIn-optimized visualizations for Moving Average Crossover Strategy')

    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str,
                        default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD), None for current date')
    parser.add_argument('--type', type=str, choices=['single', 'compare', 'all'], default='all',
                        help='Type of visualization: single strategy, comparison, or all')
    parser.add_argument('--short-window', type=int, default=50,
                        help='Short-term moving average window for single strategy')
    parser.add_argument('--long-window', type=int, default=200,
                        help='Long-term moving average window for single strategy')
    parser.add_argument('--initial-capital', type=float, default=100000,
                        help='Initial capital for backtests')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)  # Use module-level logger

    # Ensure results directory exists
    os.makedirs('./results', exist_ok=True)

    if args.type in ['single', 'all']:
        try:
            logger.info("=== Creating LinkedIn Strategy Card ===")
            create_linkedin_strategy_card(
                ticker=args.ticker,
                short_window=args.short_window,
                long_window=args.long_window,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital
            )

            # Create another standard card if user params are different and type is 'all' or 'single'
            # This ensures a 20/50 card is generated if not the primary one.
            if (args.short_window != 20 or args.long_window != 50):
                create_linkedin_strategy_card(  # Corrected indentation
                    ticker=args.ticker,
                    short_window=20,  # A common medium-term strategy
                    long_window=50,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    initial_capital=args.initial_capital
                )
            logger.info("LinkedIn strategy card(s) created successfully.")
        except Exception as e:
            logger.error("Error creating LinkedIn strategy cards: %s",
                         e, exc_info=args.debug)
            if args.debug:
                raise

    if args.type in ['compare', 'all']:
        try:
            logger.info("=== Creating LinkedIn Strategy Comparison Card ===")
            create_strategy_comparison_card(
                ticker=args.ticker,
                strategies=vs.LINKEDIN_COMPARISON_STRATEGIES,  # Using predefined strategies
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital
            )
            logger.info(
                "LinkedIn strategy comparison card created successfully.")
        except Exception as e:
            logger.error(
                "Error creating LinkedIn strategy comparison card: %s", e, exc_info=args.debug)
            if args.debug:
                raise

    logger.info("\n=== LinkedIn Visualization Process Completed ===")
    logger.info("Check the 'results' directory for output files.")
    # ... (removed LinkedIn sharing tips for brevity, can be added back if desired)


if __name__ == "__main__":
    main_cli()
