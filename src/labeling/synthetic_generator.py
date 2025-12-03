"""
Synthetic data generation using LLM few-shot prompting.
Generates additional labeled tweets based on seed examples.
"""

import pandas as pd
from pathlib import Path
import yaml
from typing import List, Dict
import json
from tqdm import tqdm


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_few_shot_prompt(seed_tweets: List[Dict], num_examples: int = 5) -> str:
    """
    Create a few-shot prompt for LLM-based synthetic generation.
    
    Args:
        seed_tweets: List of manually labeled tweets
        num_examples: Number of examples to include in prompt
    
    Returns:
        Prompt string
    """
    examples = seed_tweets[:num_examples]
    
    prompt = """You are generating synthetic tweets about the 2024 US Presidential Election and prediction markets. 

Generate tweets that are similar in style and content to the examples below. Each tweet should be labeled with:
1. Sentiment: positive, negative, or neutral
2. Stance: bullish (suggests price will go up), bearish (suggests price will go down), or neutral
3. Betting Direction: up, down, or neutral

Examples:
"""
    
    for i, tweet in enumerate(examples, 1):
        content = tweet.get('content_cleaned', tweet.get('content', ''))
        prompt += f"""
Example {i}:
Tweet: "{content}"
Sentiment: {tweet.get('sentiment', 'neutral')}
Stance: {tweet.get('stance', 'neutral')}
Betting Direction: {tweet.get('betting_direction', 'neutral')}
"""
    
    prompt += """
Now generate a new tweet about the 2024 election or prediction markets. Make it realistic and varied.
Return your response in JSON format:
{
    "tweet": "the tweet text here",
    "sentiment": "positive|negative|neutral",
    "stance": "bullish|bearish|neutral",
    "betting_direction": "up|down|neutral"
}
"""
    
    return prompt


def generate_synthetic_with_llm(prompt: str, model_name: str = "gpt-3.5-turbo"):
    """
    Generate synthetic tweet using LLM.
    
    Note: This is a placeholder. In practice, you would:
    1. Use OpenAI API, Anthropic API, or local LLM
    2. For local: Use transformers library with a small model
    3. For Colab: Can use the same model you'll fine-tune
    
    Args:
        prompt: Few-shot prompt
        model_name: Model to use
    
    Returns:
        Generated tweet dictionary
    """
    # Placeholder implementation
    # In production, replace with actual LLM call
    
    print(f"  Note: Using placeholder LLM generation")
    print(f"  To use real LLM, implement API calls or use local model")
    print(f"  Suggested: Use OpenAI API or local Llama model")
    
    # For now, return a template that shows the structure
    # In actual implementation, call the LLM here
    return {
        "tweet": "Sample synthetic tweet - implement LLM call here",
        "sentiment": "neutral",
        "stance": "neutral",
        "betting_direction": "neutral"
    }


def generate_synthetic_tweets(seed_file: Path, output_file: Path, num_synthetic: int = 100):
    """
    Generate synthetic labeled tweets from automatically labeled seed data.
    
    Args:
        seed_file: Path to automatically labeled seed tweets
        output_file: Path to save synthetic tweets
        num_synthetic: Number of synthetic tweets to generate
    """
    config = load_config()
    
    # Load seed tweets
    print(f"Loading seed tweets from {seed_file}...")
    if seed_file.suffix == '.parquet':
        seed_df = pd.read_parquet(seed_file)
    else:
        seed_df = pd.read_json(seed_file, lines=True)
    
    print(f"Loaded {len(seed_df)} seed tweets")
    
    if len(seed_df) < 5:
        print("Warning: Need at least 5 seed tweets for few-shot generation")
        print("Note: Synthetic generation is optional. You can skip this step.")
        return
    
    # Get seed tweets as list
    seed_tweets = seed_df.to_dict('records')
    
    # Generate synthetic tweets
    print(f"\nGenerating {num_synthetic} synthetic tweets...")
    synthetic_tweets = []
    
    # Create few-shot prompt
    prompt = create_few_shot_prompt(seed_tweets, num_examples=min(10, len(seed_tweets)))
    
    for i in tqdm(range(num_synthetic), desc="Generating"):
        # Generate synthetic tweet
        # In production, this would call an LLM
        synthetic = generate_synthetic_with_llm(prompt)
        
        # Add metadata
        synthetic['id'] = f"synthetic_{i}"
        synthetic['content'] = synthetic['tweet']
        synthetic['content_cleaned'] = synthetic['tweet']
        synthetic['date'] = pd.Timestamp.now().isoformat()
        synthetic['user'] = 'synthetic'
        synthetic['labeled_by'] = 'synthetic_llm'
        synthetic['label_date'] = pd.Timestamp.now().isoformat()
        
        synthetic_tweets.append(synthetic)
    
    # Save synthetic tweets
    if synthetic_tweets:
        synthetic_df = pd.DataFrame(synthetic_tweets)
        
        if output_file.suffix == '.parquet':
            synthetic_df.to_parquet(output_file, index=False)
        else:
            synthetic_df.to_json(output_file, orient='records', lines=True)
        
        print(f"\n✓ Saved {len(synthetic_tweets)} synthetic tweets to {output_file}")
        print(f"\nNote: These are placeholder tweets. Implement LLM generation for real synthetic data.")
    else:
        print("No synthetic tweets generated.")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic labeled tweets")
    parser.add_argument('--seed', '-s', type=str, required=True,
                       help='Seed file with manually labeled tweets')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file for synthetic tweets')
    parser.add_argument('--num', '-n', type=int, default=100,
                       help='Number of synthetic tweets to generate (default: 100)')
    
    args = parser.parse_args()
    
    seed_path = Path(args.seed)
    output_path = Path(args.output)
    
    if not seed_path.exists():
        print(f"Error: Seed file not found: {seed_path}")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_synthetic_tweets(seed_path, output_path, num_synthetic=args.num)


if __name__ == "__main__":
    main()

