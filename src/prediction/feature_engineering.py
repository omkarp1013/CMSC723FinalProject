"""
Feature engineering for prediction model.
Combines sentiment features with market features into time-series.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import List, Dict


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def combine_sentiment_and_market_features(sentiment_features: pd.DataFrame, 
                                         market_features: pd.DataFrame,
                                         time_window_hours: int = 6) -> pd.DataFrame:
    """
    Combine sentiment features with market features.
    
    Args:
        sentiment_features: DataFrame with sentiment features (from feature_extractor)
        market_features: DataFrame with market data
        time_window_hours: Time window for alignment
    
    Returns:
        Combined feature DataFrame
    """
    # Ensure timestamps
    sentiment_features['timestamp'] = pd.to_datetime(sentiment_features['timestamp'])
    market_features['timestamp'] = pd.to_datetime(market_features['timestamp'])
    
    # Round to hour for alignment
    sentiment_features['hour'] = sentiment_features['timestamp'].dt.floor('H')
    market_features['hour'] = market_features['timestamp'].dt.floor('H')
    
    # Aggregate sentiment features by hour
    sentiment_agg = sentiment_features.groupby('hour').agg({
        'sentiment_polarity': ['mean', 'std', 'count'],
        'stance_score': ['mean', 'std'],
        'betting_score': ['mean'],
        'confidence': ['mean'],
        'combined_sentiment': ['mean'],
    }).reset_index()
    
    # Flatten column names
    sentiment_agg.columns = [
        'hour' if col == 'hour' else f"{col[0]}_{col[1]}"
        for col in sentiment_agg.columns.values
    ]
    
    # Aggregate market features by hour
    market_agg = market_features.groupby('hour').agg({
        'price': ['mean', 'std', 'first', 'last'],
        'volume': ['sum', 'mean'],
    }).reset_index()
    
    market_agg.columns = [
        'hour' if col == 'hour' else f"{col[0]}_{col[1]}"
        for col in market_agg.columns.values
    ]
    
    # Merge
    combined = sentiment_agg.merge(market_agg, on='hour', how='outer')
    combined = combined.sort_values('hour')
    
    # Calculate price change (target variable)
    combined['price_change'] = combined['price_last'] - combined['price_first']
    combined['price_change_pct'] = (combined['price_change'] / combined['price_first']) * 100
    combined['price_direction'] = combined['price_change'].apply(
        lambda x: 1 if x > 0 else (0 if x < 0 else 0.5)
    )
    
    # Calculate rolling statistics
    window = time_window_hours
    combined['sentiment_rolling_mean'] = combined['sentiment_polarity_mean'].rolling(window=window).mean()
    combined['sentiment_rolling_std'] = combined['sentiment_polarity_mean'].rolling(window=window).std()
    combined['price_rolling_mean'] = combined['price_mean'].rolling(window=window).mean()
    
    # Momentum features
    combined['sentiment_momentum'] = combined['sentiment_polarity_mean'].diff()
    combined['price_momentum'] = combined['price_mean'].diff()
    
    # Volume features
    combined['volume_ratio'] = combined['volume_sum'] / combined['volume_sum'].rolling(window=window*2).mean()
    
    return combined


def create_prediction_features(combined_df: pd.DataFrame, 
                              prediction_horizon: int = 6) -> pd.DataFrame:
    """
    Create features for prediction model.
    Shifts target variable by prediction horizon.
    
    Args:
        combined_df: Combined sentiment and market features
        prediction_horizon: Hours ahead to predict
    
    Returns:
        DataFrame with features and shifted target
    """
    features_df = combined_df.copy()
    
    # Shift target variable by prediction horizon
    features_df['target_price'] = features_df['price_mean'].shift(-prediction_horizon)
    features_df['target_price_change'] = features_df['target_price'] - features_df['price_mean']
    features_df['target_direction'] = (features_df['target_price_change'] > 0).astype(int)
    
    # Remove rows where target is NaN (end of time series)
    features_df = features_df[features_df['target_price'].notna()]
    
    # Select feature columns
    feature_columns = [
        'sentiment_polarity_mean', 'sentiment_polarity_std', 'sentiment_polarity_count',
        'stance_score_mean', 'stance_score_std',
        'betting_score_mean',
        'confidence_mean',
        'combined_sentiment_mean',
        'price_mean', 'price_std', 'price_first',
        'volume_sum', 'volume_mean',
        'sentiment_rolling_mean', 'sentiment_rolling_std',
        'price_rolling_mean',
        'sentiment_momentum', 'price_momentum',
        'volume_ratio',
    ]
    
    # Keep only available columns
    available_features = [col for col in feature_columns if col in features_df.columns]
    
    # Fill NaN values
    features_df[available_features] = features_df[available_features].fillna(0)
    
    return features_df[['hour'] + available_features + ['target_direction', 'target_price_change']]


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Engineer features for prediction model")
    parser.add_argument('--sentiment', '-s', type=str, required=True,
                       help='Sentiment features file')
    parser.add_argument('--market', '-m', type=str, required=True,
                       help='Market data file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file for engineered features')
    parser.add_argument('--horizon', type=int, default=6,
                       help='Prediction horizon in hours (default: 6)')
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Load data
    print("Loading data...")
    sentiment_df = pd.read_parquet(args.sentiment) if Path(args.sentiment).suffix == '.parquet' else pd.read_json(args.sentiment, lines=True)
    market_df = pd.read_parquet(args.market)
    
    print(f"Loaded {len(sentiment_df)} sentiment features and {len(market_df)} market data points")
    
    # Combine features
    print("\nCombining features...")
    combined_df = combine_sentiment_and_market_features(sentiment_df, market_df)
    
    # Create prediction features
    print("Creating prediction features...")
    prediction_horizon = args.horizon or config['prediction']['prediction_horizon_hours']
    features_df = create_prediction_features(combined_df, prediction_horizon=prediction_horizon)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        features_df.to_parquet(output_path, index=False)
    else:
        features_df.to_json(output_path, orient='records', lines=True)
    
    print(f"\n✓ Saved {len(features_df)} feature vectors to {output_path}")
    print(f"  Features: {len(features_df.columns) - 3}")  # Exclude hour, target_direction, target_price_change
    print(f"  Target: price direction (up/down)")


if __name__ == "__main__":
    main()

