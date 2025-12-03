"""
Temporal alignment of tweets and market data.
Aligns tweet timestamps with market price timestamps for correlation analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime, timedelta
from typing import Dict, List


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def align_tweets_to_markets(tweets_df: pd.DataFrame, markets_df: pd.DataFrame, 
                            time_window_hours: int = 6) -> pd.DataFrame:
    """
    Align tweets with market data based on timestamps.
    
    Args:
        tweets_df: DataFrame with tweets (must have 'date' or 'timestamp' column)
        markets_df: DataFrame with market data (must have 'timestamp' and 'price' columns)
        time_window_hours: Time window in hours to match tweets to market prices
    
    Returns:
        DataFrame with tweets aligned to market prices
    """
    # Ensure timestamps are datetime
    if 'date' in tweets_df.columns:
        tweets_df['timestamp'] = pd.to_datetime(tweets_df['date'])
    elif 'timestamp' not in tweets_df.columns:
        raise ValueError("Tweets DataFrame must have 'date' or 'timestamp' column")
    
    markets_df['timestamp'] = pd.to_datetime(markets_df['timestamp'])
    
    # Sort by timestamp
    tweets_df = tweets_df.sort_values('timestamp').copy()
    markets_df = markets_df.sort_values('timestamp').copy()
    
    # For each tweet, find the market price at that time and future price
    aligned_data = []
    
    print(f"Aligning {len(tweets_df)} tweets to market data...")
    print(f"Time window: {time_window_hours} hours")
    
    for idx, tweet in tweets_df.iterrows():
        tweet_time = tweet['timestamp']
        
        # Find closest market price before or at tweet time
        market_before = markets_df[markets_df['timestamp'] <= tweet_time]
        
        if len(market_before) > 0:
            # Get most recent market price
            current_price = market_before.iloc[-1]['price']
            current_volume = market_before.iloc[-1].get('volume', 0)
            
            # Find future price (for prediction target)
            future_time = tweet_time + timedelta(hours=time_window_hours)
            market_after = markets_df[markets_df['timestamp'] > tweet_time]
            market_future = market_after[market_after['timestamp'] <= future_time]
            
            if len(market_future) > 0:
                future_price = market_future.iloc[-1]['price']
                price_change = future_price - current_price
                price_direction = 'up' if price_change > 0 else ('down' if price_change < 0 else 'neutral')
            else:
                future_price = np.nan
                price_change = np.nan
                price_direction = 'unknown'
            
            # Create aligned record
            aligned_record = tweet.to_dict()
            aligned_record['market_timestamp'] = market_before.iloc[-1]['timestamp']
            aligned_record['current_price'] = current_price
            aligned_record['current_volume'] = current_volume
            aligned_record['future_price'] = future_price
            aligned_record['price_change'] = price_change
            aligned_record['price_direction'] = price_direction
            aligned_record['time_to_future'] = (future_time - tweet_time).total_seconds() / 3600
            
            aligned_data.append(aligned_record)
    
    aligned_df = pd.DataFrame(aligned_data)
    
    print(f"Aligned {len(aligned_df)} tweets to market data")
    print(f"  Tweets with future prices: {aligned_df['price_direction'].notna().sum()}")
    
    return aligned_df


def create_time_windows(tweets_df: pd.DataFrame, markets_df: pd.DataFrame,
                       window_hours: List[int] = [1, 6, 24]) -> pd.DataFrame:
    """
    Create time-windowed aggregations of tweets and market data.
    
    Args:
        tweets_df: DataFrame with tweets
        markets_df: DataFrame with market data
        window_hours: List of time windows in hours
    
    Returns:
        DataFrame with aggregated features per time window
    """
    # Ensure timestamps
    if 'date' in tweets_df.columns:
        tweets_df['timestamp'] = pd.to_datetime(tweets_df['date'])
    elif 'timestamp' not in tweets_df.columns:
        tweets_df['timestamp'] = pd.to_datetime(tweets_df.get('id', 0), unit='s', errors='coerce')
    
    markets_df['timestamp'] = pd.to_datetime(markets_df['timestamp'])
    
    # Round timestamps to hour for aggregation
    tweets_df['hour'] = tweets_df['timestamp'].dt.floor('H')
    markets_df['hour'] = markets_df['timestamp'].dt.floor('H')
    
    windowed_data = []
    
    # Get time range
    min_time = min(tweets_df['hour'].min(), markets_df['hour'].min())
    max_time = max(tweets_df['hour'].max(), markets_df['hour'].max())
    
    print(f"Creating time windows from {min_time} to {max_time}")
    
    # For each time window size
    for window_hours in window_hours:
        print(f"  Processing {window_hours}-hour windows...")
        
        current_time = min_time
        while current_time <= max_time:
            window_end = current_time + timedelta(hours=window_hours)
            
            # Get tweets in window
            tweets_in_window = tweets_df[
                (tweets_df['hour'] >= current_time) & 
                (tweets_df['hour'] < window_end)
            ]
            
            # Get market data in window
            markets_in_window = markets_df[
                (markets_df['hour'] >= current_time) & 
                (markets_df['hour'] < window_end)
            ]
            
            if len(tweets_in_window) > 0 or len(markets_in_window) > 0:
                window_record = {
                    'window_start': current_time,
                    'window_end': window_end,
                    'window_hours': window_hours,
                    'tweet_count': len(tweets_in_window),
                    'avg_market_price': markets_in_window['price'].mean() if len(markets_in_window) > 0 else np.nan,
                    'market_price_start': markets_in_window['price'].iloc[0] if len(markets_in_window) > 0 else np.nan,
                    'market_price_end': markets_in_window['price'].iloc[-1] if len(markets_in_window) > 0 else np.nan,
                    'market_volume': markets_in_window['volume'].sum() if len(markets_in_window) > 0 else 0,
                }
                
                windowed_data.append(window_record)
            
            # Move to next window
            current_time += timedelta(hours=window_hours)
    
    windowed_df = pd.DataFrame(windowed_data)
    
    if len(windowed_df) > 0:
        # Calculate price changes
        windowed_df['price_change'] = windowed_df['market_price_end'] - windowed_df['market_price_start']
        windowed_df['price_change_pct'] = (windowed_df['price_change'] / windowed_df['market_price_start']) * 100
        windowed_df['price_direction'] = windowed_df['price_change'].apply(
            lambda x: 'up' if x > 0 else ('down' if x < 0 else 'neutral')
        )
    
    print(f"Created {len(windowed_df)} time windows")
    
    return windowed_df


def main():
    """Main function to align tweets and market data"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Align tweets with market data")
    parser.add_argument('--tweets', '-t', type=str, required=True,
                       help='Tweets file (parquet or json)')
    parser.add_argument('--markets', '-m', type=str, required=True,
                       help='Market data file (parquet)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file for aligned data')
    parser.add_argument('--window', '-w', type=int, default=6,
                       help='Time window in hours for alignment (default: 6)')
    parser.add_argument('--create-windows', action='store_true',
                       help='Also create time-windowed aggregations')
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Load data
    print("Loading data...")
    tweets_path = Path(args.tweets)
    markets_path = Path(args.markets)
    
    if tweets_path.suffix == '.parquet':
        tweets_df = pd.read_parquet(tweets_path)
    else:
        tweets_df = pd.read_json(tweets_path, lines=True)
    
    markets_df = pd.read_parquet(markets_path)
    
    print(f"Loaded {len(tweets_df)} tweets and {len(markets_df)} market data points")
    
    # Align data
    aligned_df = align_tweets_to_markets(tweets_df, markets_df, time_window_hours=args.window)
    
    # Save aligned data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        aligned_df.to_parquet(output_path, index=False)
    else:
        aligned_df.to_json(output_path, orient='records', lines=True)
    
    print(f"\n✓ Saved aligned data to {output_path}")
    
    # Create time windows if requested
    if args.create_windows:
        print("\nCreating time-windowed aggregations...")
        window_hours = config['features']['time_windows_hours']
        windowed_df = create_time_windows(tweets_df, markets_df, window_hours=window_hours)
        
        windows_output = output_path.parent / f"{output_path.stem}_windows{output_path.suffix}"
        if windows_output.suffix == '.parquet':
            windowed_df.to_parquet(windows_output, index=False)
        else:
            windowed_df.to_json(windows_output, orient='records', lines=True)
        
        print(f"✓ Saved time windows to {windows_output}")


if __name__ == "__main__":
    main()

