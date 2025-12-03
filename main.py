"""
Main entry point for the Tweet-Informed Prediction Market Forecasting project.
Orchestrates the full pipeline or runs individual components.
"""

import argparse
import sys
from pathlib import Path


def run_tweet_collection():
    """Run tweet collection"""
    from src.data_collection.tweet_scraper import main
    main()


def run_market_data_collection():
    """Run market data collection"""
    from src.data_collection.market_data import main
    main()


def run_preprocessing():
    """Run tweet preprocessing"""
    from src.preprocessing.clean_tweets import main
    main()


def main():
    parser = argparse.ArgumentParser(
        description="Tweet-Informed Prediction Market Forecasting Pipeline"
    )
    
    parser.add_argument(
        'step',
        choices=['collect-tweets', 'collect-markets', 'preprocess', 'all'],
        help='Pipeline step to run'
    )
    
    args = parser.parse_args()
    
    if args.step == 'collect-tweets':
        print("Collecting tweets...")
        run_tweet_collection()
    elif args.step == 'collect-markets':
        print("Collecting market data...")
        run_market_data_collection()
    elif args.step == 'preprocess':
        print("Preprocessing tweets...")
        run_preprocessing()
    elif args.step == 'all':
        print("Running full data collection and preprocessing pipeline...")
        run_tweet_collection()
        run_market_data_collection()
        run_preprocessing()
        print("\nData collection complete!")
        print("Next: Label data and train model in Google Colab")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

