import argparse
import logging
import os
from visualization_styles import ANIMATION_COMPARISON_STRATEGIES

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Create animated visualizations for Moving Average Crossover Strategy')

    # Add arguments
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str,
                        default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD), None for current date')
    parser.add_argument('--type', type=str, choices=['single', 'compare', 'linkedin', 'animated_linkedin', 'all'], default='all',
                        help='Type of animation to create: single strategy, comparison, linkedin optimized, animated linkedin, or all')
    parser.add_argument('--short-window', type=int, default=50,
                        help='Short-term moving average window for single strategy')
    parser.add_argument('--long-window', type=int, default=200,
                        help='Long-term moving average window for single strategy')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    # Parse arguments
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    os.makedirs('./results', exist_ok=True)

    if args.type in ['single', 'all']:
        try:
            logger.info("=== Creating Single Strategy Animation ===")
            from animate_strategy import create_animated_strategy_visualization

            create_animated_strategy_visualization(
                ticker=args.ticker,
                short_window=args.short_window,
                long_window=args.long_window,
                start_date=args.start_date,
                end_date=args.end_date
            )
            logger.info("Single strategy animation created successfully")
        except RuntimeError as e:  # Catch specific exception
            logger.error(
                "Error creating single strategy animation: %s", str(e))
            if args.debug:
                raise

    if args.type in ['compare', 'all']:
        try:
            logger.info("=== Creating Strategy Comparison Animation ===")
            from animate_comparison import create_strategy_comparison_animation

            create_strategy_comparison_animation(
                ticker=args.ticker,
                strategies=ANIMATION_COMPARISON_STRATEGIES,
                start_date=args.start_date,
                end_date=args.end_date
            )
            logger.info("Strategy comparison animation created successfully")
        except RuntimeError as e:  # Catch specific exception
            logger.error(
                "Error creating strategy comparison animation: %s", str(e))
            if args.debug:
                raise

    if args.type in ['linkedin', 'all']:
        try:
            logger.info("=== Creating LinkedIn Visuals ===")
            # Assuming create_linkedin_visuals.py has a main function or similar entry point
            # This might require refactoring create_linkedin_visuals.py to be importable
            # For now, we'll assume it can be called or its relevant functions imported
            # from create_linkedin_visuals import create_linkedin_visuals_main # Placeholder
            # create_linkedin_visuals_main(args) # Placeholder
            logger.info("LinkedIn visuals created successfully")
        except RuntimeError as e:  # Catch specific exception
            logger.error("Error creating LinkedIn visuals: %s", str(e))
            if args.debug:
                raise

    if args.type in ['animated_linkedin', 'all']:
        try:
            logger.info("=== Creating Animated LinkedIn Visuals ===")
            from animated_linkedin_visuals import create_animated_linkedin_strategy_card, create_animated_strategy_comparison_card
            create_animated_linkedin_strategy_card(
                ticker=args.ticker,
                short_window=args.short_window,
                long_window=args.long_window,
                start_date=args.start_date,
                end_date=args.end_date
            )
            create_animated_strategy_comparison_card(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date
            )
            logger.info("Animated LinkedIn visuals created successfully")
        except RuntimeError as e:  # Catch specific exception
            logger.error(
                "Error creating animated LinkedIn visuals: %s", str(e))
            if args.debug:
                raise

    logger.info("\nFor LinkedIn sharing:")
    logger.info("1. Open the HTML files in your browser")
    logger.info(
        "2. Use screen recording software to capture the animation playing")
    logger.info("3. Convert to GIF or MP4 format for LinkedIn posts")
    logger.info("4. Add captions explaining the trading strategy")


if __name__ == "__main__":
    main()
