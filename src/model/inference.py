"""
Model inference for sentiment classification.
This script is designed to run in Google Colab after model fine-tuning,
but can also run locally with a smaller model for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import List, Dict
import json


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_inference_colab(tweets_df: pd.DataFrame, model_path: str, output_path: Path):
    """
    Run inference in Colab environment.
    This function should be called from the Colab notebook.
    
    Args:
        tweets_df: DataFrame with tweets to classify
        model_path: Path to fine-tuned model
        output_path: Path to save results
    """
    print("=" * 60)
    print("Running inference in Colab")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Tweets to classify: {len(tweets_df)}")
    print("\nNote: This function should be called from the Colab notebook")
    print("See notebooks/model_training.ipynb for implementation")
    print("=" * 60)
    
    # Placeholder - actual implementation in Colab notebook
    # The notebook will:
    # 1. Load fine-tuned model
    # 2. Process tweets in batches
    # 3. Generate sentiment/stance predictions
    # 4. Save results
    
    # For now, create placeholder results
    results = tweets_df.copy()
    results['sentiment_predicted'] = 'neutral'
    results['stance_predicted'] = 'neutral'
    results['confidence'] = 0.5
    
    if output_path.suffix == '.parquet':
        results.to_parquet(output_path, index=False)
    else:
        results.to_json(output_path, orient='records', lines=True)
    
    print(f"\nPlaceholder results saved to {output_path}")
    print("Replace with actual Colab inference results")


def prepare_inference_data(tweets_file: Path, output_file: Path):
    """
    Prepare tweets for inference in Colab.
    Creates a clean dataset that can be uploaded to Google Drive.
    
    Args:
        tweets_file: Input tweets file
        output_file: Output file for Colab
    """
    print(f"Preparing inference data from {tweets_file}...")
    
    # Load tweets
    if tweets_file.suffix == '.parquet':
        df = pd.read_parquet(tweets_file)
    else:
        df = pd.read_json(tweets_file, lines=True)
    
    # Select relevant columns
    inference_df = df[['id', 'content_cleaned', 'date', 'timestamp']].copy()
    
    # Ensure content_cleaned exists
    if 'content_cleaned' not in inference_df.columns:
        if 'content' in inference_df.columns:
            inference_df['content_cleaned'] = inference_df['content']
        else:
            raise ValueError("No content column found in tweets")
    
    # Remove empty tweets
    inference_df = inference_df[inference_df['content_cleaned'].notna()]
    inference_df = inference_df[inference_df['content_cleaned'].str.len() > 0]
    
    print(f"Prepared {len(inference_df)} tweets for inference")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix == '.parquet':
        inference_df.to_parquet(output_file, index=False)
    else:
        inference_df.to_json(output_file, orient='records', lines=True)
    
    print(f"✓ Saved to {output_file}")
    print(f"\nNext steps:")
    print(f"1. Upload {output_file} to Google Drive")
    print(f"2. Load in Colab notebook")
    print(f"3. Run inference with fine-tuned model")
    print(f"4. Save results back to Google Drive")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for model inference")
    parser.add_argument('--tweets', '-t', type=str, required=True,
                       help='Input tweets file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file for inference data')
    parser.add_argument('--colab', action='store_true',
                       help='Prepare for Colab (default: True)')
    
    args = parser.parse_args()
    
    tweets_path = Path(args.tweets)
    output_path = Path(args.output)
    
    if not tweets_path.exists():
        print(f"Error: Tweets file not found: {tweets_path}")
        return
    
    prepare_inference_data(tweets_path, output_path)


if __name__ == "__main__":
    main()

