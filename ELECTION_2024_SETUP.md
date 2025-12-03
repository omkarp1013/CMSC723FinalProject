# 2024 US Presidential Election - Data Collection Guide

This project is configured to collect data specifically for the **2024 US Presidential Election** from Twitter and prediction markets (Kalshi and Polymarket).

## Data Sources

### 1. Twitter/X Data
- **Collection Method**: `snscrape` (no API keys needed)
- **Focus**: Election-related tweets, candidate mentions, prediction market discussions
- **Date Range**: January 1, 2024 to November 5, 2024 (Election Day)
- **Queries**: See `config/config.yaml` for full list

### 2. Prediction Market Data

#### Polymarket
- **API**: Public GraphQL API (no authentication required for market data)
- **Endpoint**: `https://clob.polymarket.com/`
- **Contracts**: 
  - Presidential Election Winner
  - Trump-specific contracts
  - Biden/Harris-specific contracts
- **Data Collected**: Prices, volumes, timestamps

#### Kalshi
- **API**: Requires authentication (sign up at https://kalshi.com)
- **Endpoint**: `https://trading-api.kalshi.com/trade-api/v2`
- **Note**: Current implementation uses fallback data. To use real Kalshi data:
  1. Sign up for Kalshi account
  2. Get API credentials
  3. Update `src/data_collection/market_data.py` with your credentials

## Quick Start

### Step 1: Collect Tweets
```bash
python src/data_collection/tweet_scraper.py
```

This will collect tweets using queries like:
- "2024 election"
- "Trump 2024"
- "Biden 2024"
- "#2024Election"
- "election odds"
- And more (see config.yaml)

### Step 2: Collect Market Data
```bash
python src/data_collection/market_data.py
```

This will:
- Attempt to fetch real data from Polymarket API
- Use fallback data for Kalshi (until API credentials are added)
- Save data to `data/raw/markets/`

### Step 3: Clean and Process
```bash
python src/preprocessing/clean_tweets.py
```

## Configuration

Edit `config/config.yaml` to customize:

### Tweet Collection
- **search_queries**: Add/remove election-related queries
- **date_range_days**: Adjust collection period (default: 90 days)
- **max_tweets_per_query**: Limit per query (default: 10,000)

### Market Data
- **contracts**: Add/remove election contracts
- **polymarket_slug**: Update with actual Polymarket contract slugs
- **kalshi_ticker**: Update with actual Kalshi tickers
- **start_date/end_date**: Set specific date ranges

## Getting Real Market Data

### Polymarket
The current implementation attempts to use Polymarket's public API. If you need more historical data:
1. Check Polymarket's API documentation: https://docs.polymarket.com/
2. Some contracts may require different endpoints
3. Consider using their GraphQL API directly

### Kalshi
To get real Kalshi data:
1. Sign up at https://kalshi.com
2. Navigate to API settings
3. Generate API credentials
4. Update `fetch_kalshi_data()` in `market_data.py`:
   ```python
   headers = {
       "Authorization": f"Bearer {YOUR_API_KEY}",
       "Content-Type": "application/json"
   }
   ```

## Data Structure

### Tweets
- `id`: Tweet ID
- `date`: Timestamp
- `content`: Tweet text
- `user`: Username
- `retweet_count`, `like_count`, etc.
- `query`: Which search query found this tweet

### Market Data
- `timestamp`: Price timestamp
- `contract_name`: Contract identifier
- `price`: Market price (0-1 probability)
- `volume`: Trading volume
- `source`: "polymarket" or "kalshi"

## Expected Data Volume

For the 2024 election period (Jan-Nov 2024):
- **Tweets**: 50,000 - 500,000+ tweets (depending on queries)
- **Market Data**: Hourly prices for multiple contracts
- **Storage**: ~100MB - 1GB (depending on collection period)

## Troubleshooting

### No tweets collected?
- Check your internet connection
- Verify snscrape is installed: `pip install snscrape`
- Try reducing `max_tweets_per_query`
- Check if date range is valid

### Polymarket API errors?
- The API may have rate limits - the code includes delays
- Some contract slugs may not exist - check Polymarket website
- Fallback data will be generated if API fails

### Kalshi data is placeholder?
- This is expected - you need API credentials
- The fallback data is realistic but not real
- See "Getting Real Market Data" section above

## Next Steps

After data collection:
1. **Label tweets** (200-500 examples) for sentiment/stance
2. **Generate synthetic data** using LLM
3. **Fine-tune model** in Google Colab
4. **Extract features** and build prediction model
5. **Evaluate** correlation between sentiment and market movements

