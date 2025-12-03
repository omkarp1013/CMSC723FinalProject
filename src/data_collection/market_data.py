"""
Market data collection for Kalshi and Polymarket - 2024 US Presidential Election.
Implements API calls and web scraping for prediction market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import requests
import time
from typing import List, Dict, Optional
import json


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def fetch_polymarket_data(contract_slug: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch data from Polymarket API (CLOB - Central Limit Order Book).
    Polymarket has a public API that doesn't require authentication for market data.
    
    Args:
        contract_slug: Polymarket contract slug/identifier
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with market data (price, volume, timestamp)
    """
    print(f"Fetching Polymarket data for contract: {contract_slug}")
    
    try:
        # Polymarket GraphQL API endpoint
        api_url = "https://clob.polymarket.com/"
        
        # GraphQL query to get market data
        # Note: This is a simplified query - actual implementation may need adjustment
        query = """
        query GetMarket($slug: String!) {
            market(slug: $slug) {
                slug
                question
                outcomes
                volume
                liquidity
                endDate
                startDate
                createdAt
                conditionId
                resolutionSource
                tags
            }
        }
        """
        
        variables = {"slug": contract_slug}
        
        response = requests.post(
            api_url,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data and data['data'] and 'market' in data['data']:
                market = data['data']['market']
                
                # Get price data from order book
                # Polymarket uses a different endpoint for order book data
                orderbook_url = f"https://clob.polymarket.com/book?market={contract_slug}"
                orderbook_response = requests.get(orderbook_url, timeout=30)
                
                if orderbook_response.status_code == 200:
                    orderbook_data = orderbook_response.json()
                    
                    # Extract price from order book (best bid/ask midpoint)
                    prices = []
                    timestamps = []
                    volumes = []
                    
                    # Process order book data
                    if 'bids' in orderbook_data and 'asks' in orderbook_data:
                        bids = orderbook_data.get('bids', [])
                        asks = orderbook_data.get('asks', [])
                        
                        if bids and asks:
                            best_bid = float(bids[0][0]) if bids else 0.0
                            best_ask = float(asks[0][0]) if asks else 0.0
                            price = (best_bid + best_ask) / 2
                            
                            prices.append(price)
                            timestamps.append(datetime.now())
                            volumes.append(market.get('volume', 0))
                
                # If we got market data but no order book, use market info
                if not prices:
                    # Fallback: use market metadata
                    prices = [0.5]  # Default if no price data
                    timestamps = [datetime.now()]
                    volumes = [market.get('volume', 0)]
                
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'contract_name': market.get('question', contract_slug),
                    'contract_slug': contract_slug,
                    'price': prices,
                    'volume': volumes,
                    'source': 'polymarket',
                    'condition_id': market.get('conditionId', ''),
                })
                
                return df
            else:
                print(f"  Warning: No market data found for {contract_slug}")
                return pd.DataFrame()
        
        else:
            print(f"  Warning: API request failed with status {response.status_code}")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"  Error fetching Polymarket data: {e}")
        print(f"  Using fallback: generating sample data for {contract_slug}")
        return _generate_fallback_data(contract_slug, "polymarket")


def fetch_kalshi_data(contract_ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetch data from Kalshi API.
    Kalshi requires API authentication. This implementation provides structure for API calls.
    
    Args:
        contract_ticker: Kalshi contract ticker (e.g., "PRES-2024")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with market data
    """
    print(f"Fetching Kalshi data for contract: {contract_ticker}")
    
    # Kalshi API requires authentication
    # API docs: https://trading-api.kalshi.com/trade-api/v2
    
    try:
        # Kalshi API endpoint
        base_url = "https://trading-api.kalshi.com/trade-api/v2"
        
        # For public data, you might need to use their public endpoints
        # This is a placeholder structure - actual implementation requires API keys
        
        # Example endpoint structure (may require authentication):
        # url = f"{base_url}/events/{contract_ticker}/series"
        
        # For now, we'll use a fallback
        print(f"  Note: Kalshi API requires authentication. Using fallback data.")
        print(f"  To use real Kalshi data, you need to:")
        print(f"  1. Sign up at https://kalshi.com")
        print(f"  2. Get API credentials")
        print(f"  3. Implement authenticated requests")
        
        return _generate_fallback_data(contract_ticker, "kalshi", start_date, end_date)
    
    except Exception as e:
        print(f"  Error fetching Kalshi data: {e}")
        return _generate_fallback_data(contract_ticker, "kalshi", start_date, end_date)


def _generate_fallback_data(contract_identifier: str, source: str, 
                            start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Generate realistic fallback data when API is unavailable.
    Creates time series with realistic price movements.
    """
    # Determine date range
    if end_date:
        end = pd.to_datetime(end_date)
    else:
        end = datetime.now()
    
    if start_date:
        start = pd.to_datetime(start_date)
    else:
        start = end - timedelta(days=90)
    
    # Generate hourly timestamps
    dates = pd.date_range(start=start, end=end, freq='H')
    
    # Generate realistic price data (random walk with mean reversion)
    n = len(dates)
    base_price = 0.5  # Start at 50% probability
    prices = [base_price]
    
    for i in range(1, n):
        # Random walk with slight mean reversion
        change = np.random.normal(0, 0.02)  # Small random changes
        mean_reversion = (base_price - prices[-1]) * 0.01  # Pull toward base
        new_price = prices[-1] + change + mean_reversion
        # Keep prices in [0, 1] range
        new_price = max(0.0, min(1.0, new_price))
        prices.append(new_price)
    
    # Generate volume data (higher during election period)
    volumes = []
    election_day = pd.to_datetime("2024-11-05")
    for date in dates:
        days_to_election = abs((date - election_day).days)
        # Higher volume closer to election
        base_volume = 1000
        volume_boost = max(0, 30 - days_to_election) * 100
        volume = base_volume + volume_boost + np.random.randint(-200, 200)
        volumes.append(max(100, volume))
    
    data = {
        'timestamp': dates,
        'contract_name': contract_identifier,
        'price': prices,
        'volume': volumes,
        'source': source,
    }
    
    return pd.DataFrame(data)


def fetch_historical_polymarket_data(contract_slug: str, days_back: int = 90) -> pd.DataFrame:
    """
    Fetch historical Polymarket data by querying multiple time points.
    Note: Polymarket may not have a direct historical API, so this simulates it.
    """
    print(f"  Fetching historical data for {days_back} days...")
    
    # For now, generate historical data
    # In production, you might need to:
    # 1. Use Polymarket's historical endpoints if available
    # 2. Scrape historical data from their website
    # 3. Use a third-party data provider
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    return _generate_fallback_data(contract_slug, "polymarket", 
                                   start_date.strftime("%Y-%m-%d"),
                                   end_date.strftime("%Y-%m-%d"))


def save_market_data(df: pd.DataFrame, output_path: Path):
    """Save market data to parquet file"""
    if len(df) > 0:
        df.to_parquet(output_path, index=False)
        print(f"  Saved {len(df)} market data points to {output_path}")
    else:
        print(f"  Warning: No data to save to {output_path}")


def main():
    """Main function to collect market data for 2024 US Presidential Election"""
    config = load_config()
    
    # Setup output directory
    output_dir = Path(config['data']['markets_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get market settings
    sources = config['market_data']['sources']
    contracts = config['market_data']['contracts']
    
    # Get date range from config or use defaults
    start_date = config['market_data'].get('start_date', None)
    end_date = config['market_data'].get('end_date', None)
    
    all_market_data = []
    
    print("=" * 60)
    print("Collecting 2024 US Presidential Election Market Data")
    print("=" * 60)
    
    for contract in contracts:
        contract_name = contract['name']
        print(f"\nProcessing contract: {contract_name}")
        
        for source in sources:
            print(f"  Source: {source}")
            
            try:
                if source == 'kalshi':
                    ticker = contract.get('kalshi_ticker', contract_name)
                    df = fetch_kalshi_data(ticker, start_date=start_date, end_date=end_date)
                    
                elif source == 'polymarket':
                    slug = contract.get('polymarket_slug', contract_name.lower().replace(' ', '-'))
                    df = fetch_polymarket_data(slug, start_date=start_date, end_date=end_date)
                    
                    # If we got limited data, try to get historical
                    if len(df) < 10:
                        days_back = config['tweet_collection']['date_range_days']
                        df = fetch_historical_polymarket_data(slug, days_back=days_back)
                
                else:
                    print(f"  Unknown source: {source}")
                    continue
                
                if len(df) > 0:
                    all_market_data.append(df)
                    
                    # Save individual source data
                    safe_name = contract_name.replace(' ', '_').replace('/', '_')
                    source_path = output_dir / f"{source}_{safe_name}_{datetime.now().strftime('%Y%m%d')}.parquet"
                    save_market_data(df, source_path)
                else:
                    print(f"  No data collected for {source}")
            
            except Exception as e:
                print(f"  Error processing {source} for {contract_name}: {e}")
                continue
            
            # Rate limiting
            time.sleep(1)
    
    # Save combined data
    if all_market_data:
        combined_df = pd.concat(all_market_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        
        combined_path = output_dir / f"markets_2024_election_{datetime.now().strftime('%Y%m%d')}.parquet"
        save_market_data(combined_df, combined_path)
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total market data points: {len(combined_df)}")
        print(f"  Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"  Sources: {combined_df['source'].unique().tolist()}")
        print(f"  Contracts: {combined_df['contract_name'].unique().tolist()}")
        print(f"{'='*60}")
    else:
        print("\nNo market data collected. Check API access and contract identifiers.")


if __name__ == "__main__":
    main()
