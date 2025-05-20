# Moving Average Crossover Strategy Project

## Enhanced LinkedIn Visualizations

This project implements a Moving Average Crossover Trading Strategy with advanced visualizations optimized for LinkedIn sharing. The strategy uses short and long-term moving averages to generate buy/sell signals, with comprehensive backtesting and performance metrics.

## Key Features

- **Trading Strategy Implementation**: Complete moving average crossover strategy with customizable parameters
- **Interactive Visualizations**: Animated charts showing strategy performance over time
- **Performance Metrics**: Comprehensive metrics including Sharpe ratio, drawdown, win rate, and more
- **LinkedIn-Optimized Visuals**: Professional designs specifically created for sharing on LinkedIn
- **Strategy Comparison**: Tools to compare different MA parameter combinations

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run a simple backtest: `python3 src/main.py --ticker AAPL --short-window 20 --long-window 50`
4. Generate visualizations: `python3 src/create_animations.py --ticker AAPL --type all`
5. Create LinkedIn visuals: `python3 src/create_linkedin_visuals.py --ticker AAPL --type all`

## Documentation

- `README.md`: Complete project documentation and usage guide
- `ANIMATION_GUIDE.md`: Guide for creating and sharing LinkedIn visualizations

## Project Structure

- `src/moving_avg_crossover.py`: Core strategy implementation
- `src/animate_strategy.py`: Generate animated strategy visualizations
- `src/animate_comparison.py`: Create strategy parameter comparisons
- `src/linkedin_visuals.py`: LinkedIn-optimized visualization cards
- `src/visualization_styles.py`: Styling module for consistent design

## Visualization Types

1. **Strategy Animations**: Watch the strategy unfold over time
2. **Performance Summaries**: Comprehensive performance analysis
3. **Strategy Comparisons**: Compare different MA parameters
4. **LinkedIn Cards**: Compact strategy cards optimized for social sharing
5. **Comparison Cards**: Side-by-side strategy comparisons
