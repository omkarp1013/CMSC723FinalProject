"""
Naive sentiment baseline for comparison.
Uses simple sentiment analysis (TextBlob/VADER) to predict market movements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_textblob_sentiment(text: str) -> float:
    """Calculate sentiment using TextBlob"""
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity  # Range: -1 to 1
    except:
        return 0.0


def calculate_vader_sentiment(text: str) -> float:
    """Calculate sentiment using VADER"""
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(str(text))
        return scores['compound']  # Range: -1 to 1
    except:
        return 0.0


def create_baseline_features(tweets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create baseline features using simple sentiment analysis.
    
    Args:
        tweets_df: DataFrame with tweets
    
    Returns:
        DataFrame with baseline sentiment features
    """
    print("Calculating baseline sentiment features...")
    
    # Ensure content column
    if 'content_cleaned' in tweets_df.columns:
        content_col = 'content_cleaned'
    elif 'content' in tweets_df.columns:
        content_col = 'content'
    else:
        raise ValueError("No content column found")
    
    # Calculate sentiment
    print("  Using TextBlob...")
    tweets_df['textblob_sentiment'] = tweets_df[content_col].apply(calculate_textblob_sentiment)
    
    print("  Using VADER...")
    tweets_df['vader_sentiment'] = tweets_df[content_col].apply(calculate_vader_sentiment)
    
    # Average sentiment
    tweets_df['baseline_sentiment'] = (
        tweets_df['textblob_sentiment'] + tweets_df['vader_sentiment']
    ) / 2
    
    # Ensure timestamp
    if 'timestamp' not in tweets_df.columns:
        if 'date' in tweets_df.columns:
            tweets_df['timestamp'] = pd.to_datetime(tweets_df['date'])
        else:
            tweets_df['timestamp'] = pd.to_datetime(tweets_df.get('id', 0), unit='s', errors='coerce')
    
    tweets_df['timestamp'] = pd.to_datetime(tweets_df['timestamp'])
    tweets_df['hour'] = tweets_df['timestamp'].dt.floor('H')
    
    # Aggregate by hour
    baseline_features = tweets_df.groupby('hour').agg({
        'baseline_sentiment': ['mean', 'std', 'count'],
        'textblob_sentiment': 'mean',
        'vader_sentiment': 'mean',
    }).reset_index()
    
    baseline_features.columns = [
        'hour' if col == 'hour' else f"{col[0]}_{col[1]}"
        for col in baseline_features.columns.values
    ]
    
    return baseline_features


def predict_with_baseline(baseline_features: pd.DataFrame, 
                         market_df: pd.DataFrame,
                         threshold: float = 0.0) -> pd.DataFrame:
    """
    Predict market direction using baseline sentiment.
    
    Simple rule: positive sentiment -> price up, negative -> price down
    
    Args:
        baseline_features: Baseline sentiment features
        market_df: Market data
        threshold: Sentiment threshold for prediction
    
    Returns:
        DataFrame with predictions
    """
    # Align by hour
    baseline_features['hour'] = pd.to_datetime(baseline_features['hour'])
    market_df['hour'] = pd.to_datetime(market_df['timestamp']).dt.floor('H')
    
    market_agg = market_df.groupby('hour').agg({
        'price': ['mean', 'first', 'last'],
    }).reset_index()
    
    market_agg.columns = [
        'hour' if col == 'hour' else f"{col[0]}_{col[1]}"
        for col in market_agg.columns.values
    ]
    
    # Merge
    combined = baseline_features.merge(market_agg, on='hour', how='inner')
    combined = combined.sort_values('hour')
    
    # Calculate actual price direction
    combined['price_change'] = combined['price_last'] - combined['price_first']
    combined['actual_direction'] = (combined['price_change'] > 0).astype(int)
    
    # Predict: positive sentiment -> up, negative -> down
    combined['predicted_direction'] = (combined['baseline_sentiment_mean'] > threshold).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(combined['actual_direction'], combined['predicted_direction'])
    precision = precision_score(combined['actual_direction'], combined['predicted_direction'], zero_division=0)
    recall = recall_score(combined['actual_direction'], combined['predicted_direction'], zero_division=0)
    f1 = f1_score(combined['actual_direction'], combined['predicted_direction'], zero_division=0)
    
    print("\n" + "=" * 60)
    print("BASELINE MODEL RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 60)
    
    return combined


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline sentiment model")
    parser.add_argument('--tweets', '-t', type=str, required=True,
                       help='Tweets file')
    parser.add_argument('--markets', '-m', type=str, required=True,
                       help='Market data file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file for baseline predictions')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Sentiment threshold (default: 0.0)')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    tweets_df = pd.read_parquet(args.tweets) if Path(args.tweets).suffix == '.parquet' else pd.read_json(args.tweets, lines=True)
    market_df = pd.read_parquet(args.markets)
    
    print(f"Loaded {len(tweets_df)} tweets and {len(market_df)} market data points")
    
    # Create baseline features
    baseline_features = create_baseline_features(tweets_df)
    
    # Predict
    predictions = predict_with_baseline(baseline_features, market_df, threshold=args.threshold)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        predictions.to_parquet(output_path, index=False)
    else:
        predictions.to_json(output_path, orient='records', lines=True)
    
    print(f"\n✓ Saved baseline predictions to {output_path}")


if __name__ == "__main__":
    main()

