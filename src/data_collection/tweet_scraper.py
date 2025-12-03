"""
Tweet scraper using snscrape library.
Collects tweets related to prediction markets without requiring API keys.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from tqdm import tqdm
import snscrape.modules.twitter as sntwitter


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def scrape_tweets(query, max_tweets=1000, since_date=None, until_date=None):
    """
    Scrape tweets for a given query.
    Optimized for 2024 US Presidential Election data collection.
    
    Args:
        query: Search query string (can include hashtags)
        max_tweets: Maximum number of tweets to collect
        since_date: Start date for collection (datetime object)
        until_date: End date for collection (datetime object)
    
    Returns:
        List of tweet dictionaries
    """
    tweets = []
    search_query = query
    
    # Add date filters if provided
    if since_date:
        search_query += f" since:{since_date.strftime('%Y-%m-%d')}"
    if until_date:
        search_query += f" until:{until_date.strftime('%Y-%m-%d')}"
    
    print(f"Scraping tweets for query: {search_query}")
    
    try:
        for i, tweet in enumerate(tqdm(sntwitter.TwitterSearchScraper(search_query).get_items(), 
                                       total=max_tweets, desc=f"Collecting {query}")):
            if i >= max_tweets:
                break
            
            tweet_data = {
                'id': tweet.id,
                'date': tweet.date.isoformat(),
                'content': tweet.rawContent,
                'user': tweet.user.username,
                'retweet_count': tweet.retweetCount,
                'like_count': tweet.likeCount,
                'reply_count': tweet.replyCount,
                'quote_count': tweet.quoteCount,
                'hashtags': tweet.hashtags if tweet.hashtags else [],
                'urls': [url for url in tweet.outlinks] if tweet.outlinks else [],
                'query': query
            }
            tweets.append(tweet_data)
    
    except Exception as e:
        print(f"Error scraping {query}: {e}")
    
    return tweets


def save_tweets(tweets, output_path, format='parquet'):
    """
    Save tweets to file.
    
    Args:
        tweets: List of tweet dictionaries
        output_path: Path to save file
        format: 'parquet' or 'json'
    """
    df = pd.DataFrame(tweets)
    
    if format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved {len(tweets)} tweets to {output_path}")


def main():
    """Main function to collect tweets based on configuration"""
    config = load_config()
    
    # Setup output directory
    output_dir = Path(config['data']['tweets_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get collection settings
    queries = config['tweet_collection']['search_queries']
    max_tweets = config['tweet_collection']['max_tweets_per_query']
    date_range = config['tweet_collection']['date_range_days']
    output_format = config['tweet_collection']['output_format']
    
    # Calculate date range - focus on 2024 election period
    # Default: collect from start of 2024 to election day (Nov 5, 2024)
    # Or use date_range_days if specified
    if date_range:
        since_date = datetime.now() - timedelta(days=date_range)
    else:
        # Focus on 2024 election year
        since_date = datetime(2024, 1, 1)
    
    # Set end date to election day (Nov 5, 2024) or current date if later
    election_day = datetime(2024, 11, 5)
    until_date = min(election_day, datetime.now()) if datetime.now() > election_day else datetime.now()
    
    print(f"\nDate range: {since_date.strftime('%Y-%m-%d')} to {until_date.strftime('%Y-%m-%d')}")
    print(f"Total queries: {len(queries)}")
    print(f"Max tweets per query: {max_tweets}")
    print("=" * 60)
    
    all_tweets = []
    
    # Collect tweets for each query
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}")
        
        tweets = scrape_tweets(query, max_tweets=max_tweets, since_date=since_date, until_date=until_date)
        all_tweets.extend(tweets)
        
        # Save individual query results
        query_safe = query.replace(' ', '_').replace('"', '').lower()
        output_path = output_dir / f"tweets_{query_safe}_{datetime.now().strftime('%Y%m%d')}.{output_format}"
        save_tweets(tweets, output_path, format=output_format)
    
    # Save combined results
    if all_tweets:
        combined_path = output_dir / f"tweets_combined_{datetime.now().strftime('%Y%m%d')}.{output_format}"
        save_tweets(all_tweets, combined_path, format=output_format)
        print(f"\nTotal tweets collected: {len(all_tweets)}")
    else:
        print("\nNo tweets collected. Check your queries and date range.")


if __name__ == "__main__":
    main()

