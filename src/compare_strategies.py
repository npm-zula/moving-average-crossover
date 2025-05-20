import os
import pandas as pd
import matplotlib.pyplot as plt
from moving_avg_crossover import run_strategy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualization_styles import DEFAULT_STRATEGIES
import visualization_styles as vs


def compare_strategies(ticker='AAPL', strategies=None, start_date='2015-01-01', end_date=None):
    """
    Compare different moving average crossover strategies

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
    """
    if strategies is None:
        strategies = DEFAULT_STRATEGIES

    # Create directory for results
    os.makedirs('./results', exist_ok=True)

    # Run strategies
    results = []
    portfolios = []

    for strategy_config in strategies:
        short_window = strategy_config['short_window']
        long_window = strategy_config['long_window']

        print(
            f"\nRunning strategy with {short_window}-day and {long_window}-day moving averages...")
        strategy = run_strategy(
            ticker=ticker,
            short_window=short_window,
            long_window=long_window,
            start_date=start_date,
            end_date=end_date
        )

        # Add strategy info to metrics
        strategy.metrics['Short Window'] = short_window
        strategy.metrics['Long Window'] = long_window
        strategy.metrics['Strategy'] = f"SMA({short_window},{long_window})"

        # Append results
        results.append(strategy.metrics)
        portfolios.append({
            'portfolio': strategy.positions['Total'],
            'name': f"SMA({short_window},{long_window})"
        })

    # Create comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('Strategy')

    # Save comparison
    comparison_df.to_csv(f'./results/{ticker}_strategy_comparison.csv')

    # Plot comparison
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio Value Comparison', 'Return Comparison')
    )

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']

    # Add portfolio values
    for i, portfolio_data in enumerate(portfolios):
        portfolio = portfolio_data['portfolio']
        name = portfolio_data['name']
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=portfolio,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ),
            row=1, col=1
        )

        # Calculate and add normalized returns (starting from 100)
        normalized = 100 * portfolio / portfolio.iloc[0]

        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=normalized,
                mode='lines',
                name=f"{name} (Return)",
                line=dict(color=color, width=2, dash='dash')
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title=f'{ticker} Moving Average Crossover Strategy Comparison',
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, tickformat='.2f')
    fig.update_yaxes(title_text="Return (Starting from 100)", row=2, col=1, tickformat='.2f')

    # Apply the global theme
    vs.apply_theme_to_figure(fig)

    # Save the figure
    fig.write_html(f'./results/{ticker}_strategy_comparison.html')

    print(
        f"\nComparison results saved to ./results/{ticker}_strategy_comparison.csv")
    print(
        f"Comparison plot saved to ./results/{ticker}_strategy_comparison.html")

    return comparison_df


if __name__ == "__main__":
    # Define strategies to compare
    strategies = DEFAULT_STRATEGIES

    # Compare strategies
    compare_strategies(
        ticker='AAPL',
        strategies=DEFAULT_STRATEGIES,  # Use DEFAULT_STRATEGIES
        start_date='2018-01-01'
    )
