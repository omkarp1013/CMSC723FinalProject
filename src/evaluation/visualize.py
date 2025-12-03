"""
Visualization scripts for sentiment and market data.
Creates plots showing sentiment trends, price movements, and correlations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from datetime import datetime


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_sentiment_vs_price(sentiment_df: pd.DataFrame, 
                           market_df: pd.DataFrame,
                           output_path: Path):
    """
    Plot sentiment time series vs. market prices.
    
    Args:
        sentiment_df: DataFrame with sentiment features
        market_df: DataFrame with market data
        output_path: Path to save plot
    """
    # Prepare data
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df.get('timestamp', sentiment_df.get('hour', sentiment_df.get('date'))))
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
    
    # Aggregate to hourly
    sentiment_df['hour'] = sentiment_df['timestamp'].dt.floor('H')
    market_df['hour'] = market_df['timestamp'].dt.floor('H')
    
    sentiment_agg = sentiment_df.groupby('hour')['sentiment_polarity'].mean().reset_index()
    market_agg = market_df.groupby('hour')['price'].mean().reset_index()
    
    # Merge
    combined = sentiment_agg.merge(market_agg, on='hour', how='inner')
    combined = combined.sort_values('hour')
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot sentiment
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Sentiment Polarity', color=color)
    ax1.plot(combined['hour'], combined['sentiment_polarity'], color=color, label='Sentiment', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Plot price on second axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Market Price', color=color)
    ax2.plot(combined['hour'], combined['price'], color=color, label='Price', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Sentiment vs. Market Price Over Time', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved sentiment vs. price plot to {output_path}")


def plot_prediction_accuracy(predictions_df: pd.DataFrame, output_path: Path):
    """
    Plot prediction accuracy over time.
    
    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to save plot
    """
    # Prepare data
    if 'hour' in predictions_df.columns:
        time_col = 'hour'
    elif 'timestamp' in predictions_df.columns:
        time_col = 'timestamp'
    else:
        raise ValueError("No time column found")
    
    predictions_df[time_col] = pd.to_datetime(predictions_df[time_col])
    predictions_df = predictions_df.sort_values(time_col)
    
    # Calculate rolling accuracy
    window = 24  # 24-hour window
    predictions_df['correct'] = (predictions_df['actual_direction'] == predictions_df['predicted_direction']).astype(int)
    predictions_df['rolling_accuracy'] = predictions_df['correct'].rolling(window=window).mean()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(predictions_df[time_col], predictions_df['rolling_accuracy'], 
            label=f'{window}-hour Rolling Accuracy', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random (0.5)', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Accuracy')
    ax.set_title('Prediction Accuracy Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved accuracy plot to {output_path}")


def plot_correlation_heatmap(features_df: pd.DataFrame, output_path: Path):
    """
    Plot correlation heatmap of features.
    
    Args:
        features_df: DataFrame with features
        output_path: Path to save plot
    """
    # Select numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and ID columns
    exclude = ['target_direction', 'target_price_change', 'id', 'hour']
    feature_cols = [col for col in numeric_cols if col not in exclude]
    
    # Calculate correlation
    corr_matrix = features_df[feature_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved correlation heatmap to {output_path}")


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path, top_n: int = 15):
    """
    Plot feature importance from model.
    
    Args:
        importance_df: DataFrame with feature importance
        output_path: Path to save plot
        top_n: Number of top features to show
    """
    # Sort by importance
    top_features = importance_df.nlargest(top_n, 'abs_coefficient')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
    ax.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Coefficient Value')
    ax.set_title(f'Top {top_n} Feature Importance (Logistic Regression)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature importance plot to {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations")
    parser.add_argument('--sentiment', type=str, default=None,
                       help='Sentiment features file')
    parser.add_argument('--market', type=str, default=None,
                       help='Market data file')
    parser.add_argument('--predictions', type=str, default=None,
                       help='Predictions file')
    parser.add_argument('--features', type=str, default=None,
                       help='Feature importance file')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # Plot sentiment vs. price
    if args.sentiment and args.market:
        sentiment_df = pd.read_parquet(args.sentiment) if Path(args.sentiment).suffix == '.parquet' else pd.read_json(args.sentiment, lines=True)
        market_df = pd.read_parquet(args.market)
        plot_sentiment_vs_price(sentiment_df, market_df, 
                                output_dir / 'sentiment_vs_price.png')
    
    # Plot prediction accuracy
    if args.predictions:
        predictions_df = pd.read_parquet(args.predictions) if Path(args.predictions).suffix == '.parquet' else pd.read_json(args.predictions, lines=True)
        plot_prediction_accuracy(predictions_df, 
                                output_dir / 'prediction_accuracy.png')
    
    # Plot feature importance
    if args.features:
        importance_df = pd.read_csv(args.features)
        plot_feature_importance(importance_df, 
                               output_dir / 'feature_importance.png')
    
    # Plot correlation heatmap (if features available)
    if args.predictions:
        predictions_df = pd.read_parquet(args.predictions) if Path(args.predictions).suffix == '.parquet' else pd.read_json(args.predictions, lines=True)
        plot_correlation_heatmap(predictions_df, 
                                output_dir / 'correlation_heatmap.png')
    
    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

