"""
Automated labeling using pre-trained sentiment models.
Uses TextBlob, VADER, and optionally a base LLM for zero-shot classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import re


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def classify_sentiment(text: str) -> str:
    """
    Classify sentiment using TextBlob and VADER.
    
    Args:
        text: Tweet text
    
    Returns:
        'positive', 'negative', or 'neutral'
    """
    try:
        # TextBlob sentiment
        blob = TextBlob(str(text))
        textblob_score = blob.sentiment.polarity
        
        # VADER sentiment
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(str(text))
        vader_score = vader_scores['compound']
        
        # Average sentiment
        avg_sentiment = (textblob_score + vader_score) / 2
        
        # Classify
        if avg_sentiment > 0.1:
            return 'positive'
        elif avg_sentiment < -0.1:
            return 'negative'
        else:
            return 'neutral'
    except:
        return 'neutral'


def classify_stance(text: str, sentiment: str) -> str:
    """
    Classify stance (bullish/bearish) based on text and sentiment.
    Uses keyword matching and sentiment to determine if price will go up or down.
    
    Args:
        text: Tweet text
        sentiment: Sentiment classification
    
    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    text_lower = str(text).lower()
    
    # Bullish keywords (suggest price will go up)
    bullish_keywords = [
        'win', 'wins', 'winning', 'victory', 'leading', 'ahead', 'up', 'rise', 'rising',
        'gain', 'gains', 'increase', 'higher', 'boost', 'surge', 'momentum',
        'favorite', 'favored', 'likely', 'probable', 'expected', 'predicted',
        'odds favor', 'betting on', 'confident', 'optimistic'
    ]
    
    # Bearish keywords (suggest price will go down)
    bearish_keywords = [
        'lose', 'loses', 'losing', 'defeat', 'behind', 'down', 'fall', 'falling',
        'drop', 'drops', 'decrease', 'lower', 'decline', 'crash', 'collapse',
        'underdog', 'unlikely', 'doubt', 'skeptical', 'pessimistic', 'worried',
        'odds against', 'unfavorable', 'struggling', 'trouble'
    ]
    
    # Count keyword matches
    bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
    bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
    
    # Determine stance
    if bullish_count > bearish_count:
        return 'bullish'
    elif bearish_count > bullish_count:
        return 'bearish'
    elif sentiment == 'positive':
        return 'bullish'  # Positive sentiment often correlates with bullish
    elif sentiment == 'negative':
        return 'bearish'  # Negative sentiment often correlates with bearish
    else:
        return 'neutral'


def classify_betting_direction(stance: str, sentiment: str) -> str:
    """
    Classify betting direction based on stance.
    
    Args:
        stance: Bullish/bearish/neutral
        sentiment: Positive/negative/neutral
    
    Returns:
        'up', 'down', or 'neutral'
    """
    if stance == 'bullish':
        return 'up'
    elif stance == 'bearish':
        return 'down'
    else:
        return 'neutral'


def auto_label_tweets(tweets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically label tweets using pre-trained models.
    
    Args:
        tweets_df: DataFrame with tweets
    
    Returns:
        DataFrame with labels added
    """
    print("Automatically labeling tweets...")
    print(f"Processing {len(tweets_df)} tweets...")
    
    # Ensure content column
    if 'content_cleaned' in tweets_df.columns:
        content_col = 'content_cleaned'
    elif 'content' in tweets_df.columns:
        content_col = 'content'
    else:
        raise ValueError("No content column found")
    
    labeled_tweets = tweets_df.copy()
    
    # Classify sentiment
    print("  Classifying sentiment...")
    tqdm.pandas(desc="Sentiment")
    labeled_tweets['sentiment'] = labeled_tweets[content_col].progress_apply(classify_sentiment)
    
    # Classify stance
    print("  Classifying stance...")
    tqdm.pandas(desc="Stance")
    labeled_tweets['stance'] = labeled_tweets.progress_apply(
        lambda row: classify_stance(row[content_col], row['sentiment']), axis=1
    )
    
    # Classify betting direction
    print("  Classifying betting direction...")
    labeled_tweets['betting_direction'] = labeled_tweets.apply(
        lambda row: classify_betting_direction(row['stance'], row['sentiment']), axis=1
    )
    
    # Add metadata
    labeled_tweets['labeled_by'] = 'auto_labeler'
    labeled_tweets['label_date'] = pd.Timestamp.now().isoformat()
    
    # Print summary
    print("\n" + "=" * 60)
    print("AUTOMATED LABELING SUMMARY")
    print("=" * 60)
    print("Sentiment distribution:")
    print(labeled_tweets['sentiment'].value_counts().to_string())
    print("\nStance distribution:")
    print(labeled_tweets['stance'].value_counts().to_string())
    print("\nBetting direction distribution:")
    print(labeled_tweets['betting_direction'].value_counts().to_string())
    print("=" * 60)
    
    return labeled_tweets


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automatically label tweets using pre-trained models")
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input tweet file (parquet or json)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file for labeled tweets')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Load tweets
    print(f"Loading tweets from {input_path}...")
    if input_path.suffix == '.parquet':
        tweets_df = pd.read_parquet(input_path)
    else:
        tweets_df = pd.read_json(input_path, lines=True)
    
    print(f"Loaded {len(tweets_df)} tweets")
    
    # Auto-label
    labeled_df = auto_label_tweets(tweets_df)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        labeled_df.to_parquet(output_path, index=False)
    else:
        labeled_df.to_json(output_path, orient='records', lines=True)
    
    print(f"\n✓ Saved {len(labeled_df)} labeled tweets to {output_path}")


if __name__ == "__main__":
    main()

