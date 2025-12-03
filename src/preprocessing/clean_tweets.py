"""
Tweet cleaning and preprocessing pipeline.
Removes URLs, normalizes text, preserves emojis.
"""

import pandas as pd
import re
from pathlib import Path
import yaml
from tqdm import tqdm


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clean_tweet_text(text: str, config: dict) -> str:
    """
    Clean a single tweet text.
    
    Args:
        text: Raw tweet text
        config: Preprocessing configuration
    
    Returns:
        Cleaned tweet text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs if configured
    if config['preprocessing']['remove_urls']:
        text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions if configured
    if config['preprocessing']['remove_mentions']:
        text = re.sub(r'@\w+', '', text)
    
    # Preserve emojis (they're already in the text, just don't remove them)
    # Emojis are preserved by default
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Lowercase if configured
    if config['preprocessing']['lowercase']:
        text = text.lower()
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def filter_tweets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter tweets based on length constraints.
    
    Args:
        df: DataFrame with tweets
        config: Preprocessing configuration
    
    Returns:
        Filtered DataFrame
    """
    min_length = config['preprocessing']['min_tweet_length']
    max_length = config['preprocessing']['max_tweet_length']
    
    # Filter by length
    mask = (df['content_cleaned'].str.len() >= min_length) & \
           (df['content_cleaned'].str.len() <= max_length)
    
    filtered_df = df[mask].copy()
    
    print(f"Filtered {len(df) - len(filtered_df)} tweets based on length constraints")
    print(f"Remaining tweets: {len(filtered_df)}")
    
    return filtered_df


def main():
    """Main preprocessing function"""
    config = load_config()
    
    # Input and output directories
    input_dir = Path(config['data']['tweets_dir'])
    output_dir = Path(config['data']['processed_dir']) / "cleaned_tweets"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tweet files
    tweet_files = list(input_dir.glob("*.parquet")) + list(input_dir.glob("*.json"))
    
    if not tweet_files:
        print(f"No tweet files found in {input_dir}")
        return
    
    all_cleaned = []
    
    for tweet_file in tqdm(tweet_files, desc="Processing tweet files"):
        print(f"\nProcessing: {tweet_file.name}")
        
        # Load tweets
        if tweet_file.suffix == '.parquet':
            df = pd.read_parquet(tweet_file)
        elif tweet_file.suffix == '.json':
            df = pd.read_json(tweet_file, lines=True)
        else:
            continue
        
        print(f"  Loaded {len(df)} tweets")
        
        # Clean tweet text
        tqdm.pandas(desc="Cleaning tweets")
        df['content_cleaned'] = df['content'].progress_apply(
            lambda x: clean_tweet_text(x, config)
        )
        
        # Filter tweets
        df = filter_tweets(df, config)
        
        # Save cleaned tweets
        output_path = output_dir / f"cleaned_{tweet_file.stem}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  Saved cleaned tweets to {output_path}")
        
        all_cleaned.append(df)
    
    # Save combined cleaned tweets
    if all_cleaned:
        combined_df = pd.concat(all_cleaned, ignore_index=True)
        combined_path = output_dir / f"cleaned_tweets_combined_{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
        combined_df.to_parquet(combined_path, index=False)
        print(f"\nTotal cleaned tweets: {len(combined_df)}")
        print(f"Saved to: {combined_path}")


if __name__ == "__main__":
    main()

