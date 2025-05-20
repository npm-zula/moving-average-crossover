import argparse
from moving_avg_crossover import run_strategy
from compare_strategies import compare_strategies
from visualization_styles import DEFAULT_STRATEGIES


def main():
    parser = argparse.ArgumentParser(
        description='Moving Average Crossover Trading Strategy')

    # Add arguments
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol')
    parser.add_argument('--short-window', type=int, default=50,
                        help='Short-term moving average window')
    parser.add_argument('--long-window', type=int, default=200,
                        help='Long-term moving average window')
    parser.add_argument('--start-date', type=str,
                        default='2015-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD), None for current date')
    parser.add_argument('--initial-capital', type=float,
                        default=100000, help='Initial capital')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different MA strategies')

    # Parse arguments
    args = parser.parse_args()

    if args.compare:
        # Compare different strategies
        # Use a copy to avoid modifying the original list
        strategies = DEFAULT_STRATEGIES.copy()

        # Add the user-specified strategy if it's not in the list
        user_strategy = {'short_window': args.short_window,
                         'long_window': args.long_window}
        if user_strategy not in strategies:
            strategies.append(user_strategy)

        compare_strategies(
            ticker=args.ticker,
            strategies=strategies,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        # Run a single strategy
        run_strategy(
            ticker=args.ticker,
            short_window=args.short_window,
            long_window=args.long_window,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital
        )


if __name__ == "__main__":
    main()
