"""
Feature extraction from LLM predictions.
Extracts sentiment, stance, confidence, and other features from model outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_sentiment_features(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sentiment features from model predictions.
    
    Args:
        predictions_df: DataFrame with model predictions (sentiment, stance, etc.)
    
    Returns:
        DataFrame with extracted features
    """
    features_df = predictions_df.copy()
    
    # Sentiment polarity mapping
    sentiment_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
    features_df['sentiment_polarity'] = features_df['sentiment_predicted'].map(
        sentiment_map
    ).fillna(0.0)
    
    # Stance mapping
    stance_map = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}
    features_df['stance_score'] = features_df['stance_predicted'].map(
        stance_map
    ).fillna(0.0)
    
    # Betting direction mapping
    betting_map = {'up': 1.0, 'down': -1.0, 'neutral': 0.0}
    if 'betting_direction_predicted' in features_df.columns:
        features_df['betting_score'] = features_df['betting_direction_predicted'].map(
            betting_map
        ).fillna(0.0)
    else:
        features_df['betting_score'] = features_df['stance_score']  # Use stance as proxy
    
    # Confidence score (if available)
    if 'confidence' not in features_df.columns:
        features_df['confidence'] = 0.5  # Default
    
    # Combined sentiment-stance score
    features_df['combined_sentiment'] = (
        features_df['sentiment_polarity'] * 0.5 + 
        features_df['stance_score'] * 0.5
    )
    
    # Volume (tweet count per time period - will be aggregated later)
    features_df['tweet_volume'] = 1
    
    return features_df


def aggregate_time_features(features_df: pd.DataFrame, time_windows: List[int] = [1, 6, 24]) -> pd.DataFrame:
    """
    Aggregate features over time windows.
    
    Args:
        features_df: DataFrame with per-tweet features
        time_windows: List of time windows in hours
    
    Returns:
        DataFrame with aggregated features per time window
    """
    # Ensure timestamp column
    if 'timestamp' not in features_df.columns:
        if 'date' in features_df.columns:
            features_df['timestamp'] = pd.to_datetime(features_df['date'])
        else:
            raise ValueError("No timestamp or date column found")
    
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
    features_df = features_df.sort_values('timestamp')
    
    # Round to hour for aggregation
    features_df['hour'] = features_df['timestamp'].dt.floor('H')
    
    aggregated_data = []
    
    for window_hours in time_windows:
        print(f"  Aggregating {window_hours}-hour windows...")
        
        # Group by time windows
        features_df[f'hour_{window_hours}h'] = (
            features_df['hour'].dt.floor(f'{window_hours}H')
        )
        
        grouped = features_df.groupby(f'hour_{window_hours}h').agg({
            'sentiment_polarity': ['mean', 'std', 'count'],
            'stance_score': ['mean', 'std'],
            'betting_score': ['mean', 'std'],
            'confidence': ['mean'],
            'combined_sentiment': ['mean', 'std'],
            'tweet_volume': 'sum',
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [
            'timestamp' if 'hour' in str(col) else f"{col[0]}_{col[1]}_{window_hours}h"
            for col in grouped.columns.values
        ]
        
        # Rename for clarity
        grouped = grouped.rename(columns={
            f'sentiment_polarity_mean_{window_hours}h': f'avg_sentiment_{window_hours}h',
            f'sentiment_polarity_std_{window_hours}h': f'std_sentiment_{window_hours}h',
            f'sentiment_polarity_count_{window_hours}h': f'tweet_count_{window_hours}h',
            f'stance_score_mean_{window_hours}h': f'avg_stance_{window_hours}h',
            f'betting_score_mean_{window_hours}h': f'avg_betting_{window_hours}h',
            f'confidence_mean_{window_hours}h': f'avg_confidence_{window_hours}h',
            f'combined_sentiment_mean_{window_hours}h': f'avg_combined_{window_hours}h',
            f'tweet_volume_sum_{window_hours}h': f'total_volume_{window_hours}h',
        })
        
        # Calculate momentum (change over time)
        grouped = grouped.sort_values('timestamp')
        grouped[f'sentiment_momentum_{window_hours}h'] = grouped[f'avg_sentiment_{window_hours}h'].diff()
        grouped[f'stance_momentum_{window_hours}h'] = grouped[f'avg_stance_{window_hours}h'].diff()
        
        aggregated_data.append(grouped)
    
    # Merge all time windows
    if aggregated_data:
        result = aggregated_data[0]
        for df in aggregated_data[1:]:
            result = result.merge(df, on='timestamp', how='outer')
        
        result = result.sort_values('timestamp')
        return result
    else:
        return pd.DataFrame()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from model predictions")
    parser.add_argument('--predictions', '-p', type=str, required=True,
                       help='Model predictions file (from Colab inference)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file for extracted features')
    parser.add_argument('--aggregate', '-a', action='store_true',
                       help='Also create time-windowed aggregations')
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    pred_path = Path(args.predictions)
    
    if pred_path.suffix == '.parquet':
        predictions_df = pd.read_parquet(pred_path)
    else:
        predictions_df = pd.read_json(pred_path, lines=True)
    
    print(f"Loaded {len(predictions_df)} predictions")
    
    # Extract features
    print("\nExtracting sentiment features...")
    features_df = extract_sentiment_features(predictions_df)
    
    # Save per-tweet features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        features_df.to_parquet(output_path, index=False)
    else:
        features_df.to_json(output_path, orient='records', lines=True)
    
    print(f"✓ Saved features to {output_path}")
    
    # Aggregate if requested
    if args.aggregate:
        print("\nCreating time-windowed aggregations...")
        time_windows = config['features']['time_windows_hours']
        aggregated_df = aggregate_time_features(features_df, time_windows=time_windows)
        
        agg_output = output_path.parent / f"{output_path.stem}_aggregated{output_path.suffix}"
        if agg_output.suffix == '.parquet':
            aggregated_df.to_parquet(agg_output, index=False)
        else:
            aggregated_df.to_json(agg_output, orient='records', lines=True)
        
        print(f"✓ Saved aggregated features to {agg_output}")


if __name__ == "__main__":
    main()

