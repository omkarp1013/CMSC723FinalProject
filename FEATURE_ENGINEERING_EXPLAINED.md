# Feature Engineering Explained

This document explains how feature engineering works in this project, step by step.

## Overview

Feature engineering transforms raw data (tweets + market prices) into numerical features that a machine learning model can use to predict price movements.

**Goal**: Create features that capture the relationship between Twitter sentiment and market price movements.

---

## The Feature Engineering Pipeline

The process happens in **three main stages**:

1. **Feature Extraction** - Convert LLM predictions to numbers
2. **Time Aggregation** - Group features by time windows
3. **Feature Combination** - Merge sentiment + market features

---

## Stage 1: Feature Extraction (`feature_extractor.py`)

### Input
- LLM predictions: `sentiment_predicted`, `stance_predicted`, `confidence`, etc.
- Per-tweet data

### What It Does

Converts categorical predictions into numerical features:

#### 1. Sentiment Polarity
```python
# Convert: "positive" → 1.0, "negative" → -1.0, "neutral" → 0.0
sentiment_polarity = {
    'positive': 1.0,
    'negative': -1.0,
    'neutral': 0.0
}
```

**Example**:
- Tweet: "Trump will win the election!"
- LLM predicts: `sentiment = "positive"`
- Feature: `sentiment_polarity = 1.0`

#### 2. Stance Score
```python
# Convert: "bullish" → 1.0, "bearish" → -1.0, "neutral" → 0.0
stance_score = {
    'bullish': 1.0,   # Suggests price will go up
    'bearish': -1.0,  # Suggests price will go down
    'neutral': 0.0
}
```

**Example**:
- Tweet: "The odds are in favor of Trump"
- LLM predicts: `stance = "bullish"`
- Feature: `stance_score = 1.0`

#### 3. Betting Score
```python
# Convert: "up" → 1.0, "down" → -1.0, "neutral" → 0.0
betting_score = {
    'up': 1.0,
    'down': -1.0,
    'neutral': 0.0
}
```

#### 4. Combined Sentiment
```python
# Weighted combination of sentiment and stance
combined_sentiment = (sentiment_polarity * 0.5) + (stance_score * 0.5)
```

**Why?** Combines both sentiment and directional bias into one score.

### Output
Per-tweet numerical features:
- `sentiment_polarity`: -1.0 to 1.0
- `stance_score`: -1.0 to 1.0
- `betting_score`: -1.0 to 1.0
- `confidence`: 0.0 to 1.0
- `combined_sentiment`: -1.0 to 1.0

---

## Stage 2: Time Aggregation (`feature_extractor.py`)

### Input
- Per-tweet features (from Stage 1)
- Timestamps for each tweet

### What It Does

Aggregates features over time windows (1 hour, 6 hours, 24 hours).

### Why Aggregate?

Individual tweets are noisy. Aggregating over time:
- Smooths out noise
- Captures trends
- Creates stable features for prediction

### How It Works

#### Example: 6-Hour Window

**Raw tweets** (per hour):
```
Hour 1: sentiment = [0.8, 0.5, -0.2, 0.9]  → mean = 0.5
Hour 2: sentiment = [0.6, 0.7, 0.4]         → mean = 0.57
Hour 3: sentiment = [0.3, 0.1, -0.1]         → mean = 0.1
Hour 4: sentiment = [-0.2, -0.5, 0.0]       → mean = -0.23
Hour 5: sentiment = [-0.3, -0.4, -0.6]      → mean = -0.43
Hour 6: sentiment = [-0.5, -0.7, -0.8]      → mean = -0.67
```

**Aggregated features** (6-hour window):
```python
avg_sentiment_6h = mean([0.5, 0.57, 0.1, -0.23, -0.43, -0.67]) = -0.03
std_sentiment_6h = std([0.5, 0.57, 0.1, -0.23, -0.43, -0.67]) = 0.45
tweet_count_6h = 18  # Total tweets in 6 hours
```

### Features Created

For each time window (1h, 6h, 24h):

1. **Average sentiment** (`avg_sentiment_1h`, `avg_sentiment_6h`, `avg_sentiment_24h`)
   - Mean sentiment over the window
   - Captures overall sentiment level

2. **Sentiment standard deviation** (`std_sentiment_1h`, etc.)
   - Measures sentiment volatility
   - High std = mixed/uncertain sentiment

3. **Tweet count** (`tweet_count_1h`, etc.)
   - Volume of discussion
   - More tweets = more attention/interest

4. **Sentiment momentum** (`sentiment_momentum_6h`)
   - Change in sentiment over time
   - Positive momentum = sentiment improving
   - Negative momentum = sentiment worsening

### Output
Time-aggregated features per hour:
- `avg_sentiment_1h`, `avg_sentiment_6h`, `avg_sentiment_24h`
- `std_sentiment_1h`, `std_sentiment_6h`, `std_sentiment_24h`
- `tweet_count_1h`, `tweet_count_6h`, `tweet_count_24h`
- `sentiment_momentum_6h`, etc.

---

## Stage 3: Feature Combination (`feature_engineering.py`)

### Input
- Aggregated sentiment features (from Stage 2)
- Market data (prices, volumes, timestamps)

### What It Does

Combines sentiment features with market features and creates derived features.

### Step 1: Align by Time

Both sentiment and market data are aligned to hourly timestamps:

```python
# Sentiment features (per hour)
hour | avg_sentiment_6h | tweet_count_6h
-----|------------------|----------------
10:00| 0.5              | 150
11:00| 0.6              | 180
12:00| 0.4              | 200

# Market data (per hour)
hour | price | volume
-----|-------|-------
10:00| 0.55  | 10000
11:00| 0.57  | 12000
12:00| 0.54  | 11000

# Combined (merged on hour)
hour | avg_sentiment_6h | tweet_count_6h | price | volume
-----|------------------|-----------------|-------|--------
10:00| 0.5              | 150            | 0.55  | 10000
11:00| 0.6              | 180            | 0.57  | 12000
12:00| 0.4              | 200            | 0.54  | 11000
```

### Step 2: Create Market Features

#### Price Statistics
```python
price_mean = mean(prices in hour)      # Average price
price_std = std(prices in hour)        # Price volatility
price_first = first price in hour      # Opening price
price_last = last price in hour        # Closing price
```

#### Volume Features
```python
volume_sum = sum(volume in hour)       # Total trading volume
volume_mean = mean(volume in hour)     # Average volume
```

### Step 3: Create Derived Features

#### Rolling Statistics
```python
# 6-hour rolling mean of sentiment
sentiment_rolling_mean = mean([sentiment_t-5, sentiment_t-4, ..., sentiment_t])

# 6-hour rolling mean of price
price_rolling_mean = mean([price_t-5, price_t-4, ..., price_t])
```

**Why?** Captures trends and smooths short-term fluctuations.

#### Momentum Features
```python
# Change in sentiment from previous hour
sentiment_momentum = sentiment_t - sentiment_t-1

# Change in price from previous hour
price_momentum = price_t - price_t-1
```

**Why?** Captures direction and speed of change.

#### Volume Ratio
```python
# Current volume vs. average volume
volume_ratio = volume_current / volume_rolling_mean
```

**Why?** Identifies unusual trading activity (spikes in volume).

### Step 4: Create Target Variable

```python
# Price 6 hours ahead
target_price = price[t + 6]

# Price change
target_price_change = target_price - current_price

# Direction (1 = up, 0 = down)
target_direction = 1 if target_price_change > 0 else 0
```

**Example**:
```
Current time: 10:00, price = 0.55
Future time: 16:00, price = 0.58
→ target_price_change = 0.03
→ target_direction = 1 (price went up)
```

---

## Final Feature Set

After all three stages, we have **~20 features**:

### Sentiment Features
1. `sentiment_polarity_mean` - Average sentiment
2. `sentiment_polarity_std` - Sentiment volatility
3. `sentiment_polarity_count` - Tweet volume
4. `stance_score_mean` - Average stance
5. `stance_score_std` - Stance volatility
6. `betting_score_mean` - Average betting direction
7. `confidence_mean` - Average confidence
8. `combined_sentiment_mean` - Combined score

### Market Features
9. `price_mean` - Average price
10. `price_std` - Price volatility
11. `price_first` - Opening price
12. `volume_sum` - Total volume
13. `volume_mean` - Average volume

### Derived Features
14. `sentiment_rolling_mean` - Rolling sentiment trend
15. `sentiment_rolling_std` - Rolling sentiment volatility
16. `price_rolling_mean` - Rolling price trend
17. `sentiment_momentum` - Sentiment change rate
18. `price_momentum` - Price change rate
19. `volume_ratio` - Volume spike indicator

### Target
20. `target_direction` - Price direction 6 hours ahead (1 = up, 0 = down)

---

## Example: Complete Feature Vector

For a single hour (e.g., 2024-10-15 14:00):

```python
{
    'hour': '2024-10-15 14:00',
    
    # Sentiment features
    'sentiment_polarity_mean': 0.65,      # Positive sentiment
    'sentiment_polarity_std': 0.25,       # Moderate volatility
    'sentiment_polarity_count': 200,      # 200 tweets this hour
    'stance_score_mean': 0.7,             # Bullish stance
    'betting_score_mean': 0.6,            # Betting on up
    'confidence_mean': 0.75,              # High confidence
    'combined_sentiment_mean': 0.675,     # Combined score
    
    # Market features
    'price_mean': 0.58,                   # Current price
    'price_std': 0.02,                    # Low volatility
    'price_first': 0.57,                  # Opening price
    'volume_sum': 15000,                  # High volume
    'volume_mean': 15000,
    
    # Derived features
    'sentiment_rolling_mean': 0.6,        # Trending positive
    'sentiment_momentum': 0.1,            # Sentiment improving
    'price_momentum': 0.01,               # Price rising
    'volume_ratio': 1.5,                  # 50% above average
    
    # Target (what we're predicting)
    'target_direction': 1                  # Price went up 6 hours later
}
```

---

## Why This Feature Engineering?

### 1. **Captures Multiple Signals**
- Sentiment (what people feel)
- Stance (directional bias)
- Market state (current price, volume)
- Trends (momentum, rolling means)

### 2. **Handles Time**
- Aggregates over windows (smooths noise)
- Captures momentum (direction of change)
- Aligns sentiment with market timing

### 3. **Creates Predictive Signals**
- Sentiment momentum → predicts price momentum
- Volume spikes → predicts volatility
- Combined features → captures interactions

### 4. **Makes Features Interpretable**
- Can see which features matter most
- Understand what drives predictions
- Debug and improve the model

---

## The Complete Flow

```
LLM Predictions (per tweet)
    ↓
[Feature Extraction]
    → sentiment_polarity, stance_score, confidence
    ↓
[Time Aggregation]
    → avg_sentiment_6h, tweet_count_6h, momentum
    ↓
[Feature Combination]
    → sentiment + market + derived features
    ↓
Final Feature Vector (20 features)
    ↓
[Logistic Regression]
    → Predicts target_direction (up/down)
```

---

## Key Insights

1. **Individual tweets are noisy** → Aggregate over time
2. **Sentiment alone isn't enough** → Combine with market data
3. **Trends matter** → Use momentum and rolling statistics
4. **Timing matters** → Align sentiment with market timestamps
5. **Multiple signals** → Combine sentiment, stance, volume, price

This feature engineering pipeline transforms raw tweets and prices into a rich feature set that captures the relationship between social media sentiment and market movements.

