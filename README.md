# Tweet-Informed Prediction Market Forecasting

This project uses Twitter sentiment analysis (via fine-tuned LLM) to predict short-term movements in prediction markets (Kalshi/Polymarket) for the 2024 US Presidential Election.

## Project Overview

**Goal**: Predict short-term price movements in prediction markets by analyzing Twitter sentiment.

**Approach**:
1. Collect tweets about the 2024 election and prediction markets
2. Automatically label tweets with sentiment, stance, and betting direction
3. Fine-tune an LLM in Google Colab to improve sentiment classification
4. Extract sentiment features and combine with market data
5. Train a logistic regression model to predict price movements
6. Evaluate and compare with baseline models

## Project Structure

```
CMSC723FinalProject/
├── data/                    # Data directory
│   ├── raw/                 # Raw collected data (tweets, markets)
│   ├── processed/           # Processed and labeled data
│   └── models/              # Trained model checkpoints
├── src/                     # Source code
│   ├── data_collection/    # Tweet and market data collection
│   ├── preprocessing/      # Data cleaning and alignment
│   ├── labeling/           # Automated labeling tools
│   ├── model/              # Model inference and feature extraction
│   ├── prediction/         # Prediction models (baseline, logistic regression)
│   └── evaluation/         # Evaluation and visualization
├── notebooks/               # Jupyter notebooks (Colab)
├── config/                  # Configuration files
└── requirements.txt         # Python dependencies
```

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create Data Directories
```bash
python setup.py
```

This creates all necessary directories automatically.

## Complete Workflow - Step by Step

### Phase 1: Data Collection (Local)

#### Step 1.1: Collect Tweets
```bash
python src/data_collection/tweet_scraper.py
```
**What it does**: 
- Uses `snscrape` to collect tweets (no API keys needed)
- Searches for election-related queries (see `config/config.yaml`)
- Collects tweet text, timestamps, user info, engagement metrics
- Saves to `data/raw/tweets/` in Parquet format
- **Time**: 10-30 minutes depending on volume

#### Step 1.2: Collect Market Data
```bash
python src/data_collection/market_data.py
```
**What it does**:
- Fetches prediction market data from Polymarket API (real data)
- Generates realistic fallback data for Kalshi (add API credentials for real data)
- Collects prices, volumes, timestamps for election contracts
- Saves to `data/raw/markets/` in Parquet format
- **Time**: 5-10 minutes

#### Step 1.3: Clean Tweets
```bash
python src/preprocessing/clean_tweets.py
```
**What it does**:
- Removes URLs, normalizes whitespace
- Preserves emojis (useful for sentiment)
- Filters by length (min 10, max 280 characters)
- Saves cleaned tweets to `data/processed/cleaned_tweets/`
- **Time**: 2-5 minutes

### Phase 2: Automated Labeling (Local)

#### Step 2.1: Auto-Label Tweets
```bash
python src/labeling/auto_labeler.py \
    --input data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    --output data/processed/labeled_tweets/auto_labels.parquet
```
**What it does**:
- Uses TextBlob + VADER for sentiment classification (positive/negative/neutral)
- Uses keyword matching to determine stance (bullish/bearish/neutral)
- Maps stance to betting direction (up/down/neutral)
- Labels ALL tweets automatically - no manual work required
- Saves labeled data to `data/processed/labeled_tweets/`
- **Time**: 5-15 minutes (depends on dataset size)

### Phase 3: Data Alignment (Local)

#### Step 3.1: Align Tweets with Market Data
```bash
python src/preprocessing/align_data.py \
    --tweets data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    --markets data/raw/markets/markets_2024_election_*.parquet \
    --output data/processed/aligned_data.parquet \
    --window 6 \
    --create-windows
```
**What it does**:
- Aligns tweet timestamps with market price timestamps
- For each tweet, finds current market price and future price (6 hours ahead)
- Creates time-windowed aggregations (1h, 6h, 24h windows)
- Calculates price changes and directions for correlation analysis
- Saves aligned data to `data/processed/aligned_data.parquet`
- **Time**: 5-10 minutes

### Phase 4: Baseline Model (Local)

#### Step 4.1: Run Baseline Model
```bash
python src/prediction/baseline.py \
    --tweets data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    --markets data/raw/markets/markets_2024_election_*.parquet \
    --output data/processed/predictions/baseline_predictions.parquet
```
**What it does**:
- Uses TextBlob + VADER for sentiment (same as auto-labeler)
- Simple rule: positive sentiment → price up, negative → price down
- Aggregates sentiment by hour and compares to actual price movements
- Calculates accuracy, precision, recall, F1-score
- Provides baseline for comparison with LLM-based model
- Saves predictions to `data/processed/predictions/`
- **Time**: 5-10 minutes

### Phase 5: Model Training (Google Colab)

#### Step 5.1: Prepare Data for Colab
```bash
python src/model/inference.py \
    --tweets data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    --output data/processed/inference_data.parquet
```
**What it does**:
- Prepares clean dataset for Colab inference
- Selects only necessary columns (id, content, timestamp)
- Removes empty tweets
- Creates file ready for Google Drive upload

#### Step 5.2: Upload to Google Drive
1. Upload `data/processed/inference_data.parquet` to Google Drive
2. Upload `data/processed/labeled_tweets/auto_labels.parquet` to Google Drive
3. Upload `config/config.yaml` to Google Drive

#### Step 5.3: Train Model in Colab
1. Open `notebooks/model_training.ipynb` in Google Colab
2. Mount Google Drive
3. Install Unsloth and dependencies
4. Load auto-labeled data
5. Fine-tune Llama 3.2 3B model using LoRA
6. Run inference on all tweets
7. Save predictions to Google Drive

**What it does**:
- Fine-tunes LLM on auto-labeled data to improve sentiment/stance classification
- Uses Unsloth for efficient training (LoRA, 4-bit quantization)
- Runs inference on all collected tweets
- Generates sentiment, stance, and confidence scores

#### Step 5.4: Download Results
- Download `model_predictions.parquet` from Google Drive
- Save to `data/processed/model_predictions.parquet`

### Phase 6: Feature Extraction (Local)

#### Step 6.1: Extract Features from Predictions
```bash
python src/model/feature_extractor.py \
    --predictions data/processed/model_predictions.parquet \
    --output data/processed/features/extracted_features.parquet \
    --aggregate
```
**What it does**:
- Extracts sentiment features from LLM predictions
- Maps sentiment/stance to numerical scores (polarity, stance_score, betting_score)
- Creates time-windowed aggregations (1h, 6h, 24h)
- Calculates rolling statistics (mean, std, momentum)
- Saves features to `data/processed/features/`
- **Time**: 5-10 minutes

### Phase 7: Feature Engineering (Local)

#### Step 7.1: Combine Sentiment and Market Features
```bash
python src/prediction/feature_engineering.py \
    --sentiment data/processed/features/extracted_features_aggregated.parquet \
    --market data/raw/markets/markets_2024_election_*.parquet \
    --output data/processed/features/engineered_features.parquet \
    --horizon 6
```
**What it does**:
- Combines sentiment features with market data
- Creates prediction features (sentiment + price + volume + momentum)
- Sets up target variable: price direction 6 hours ahead
- Calculates rolling statistics and momentum features
- Prepares data for logistic regression training
- Saves engineered features to `data/processed/features/`
- **Time**: 2-5 minutes

### Phase 8: Prediction Model (Local)

#### Step 8.1: Train Logistic Regression
```bash
python src/prediction/market_predictor.py \
    --features data/processed/features/engineered_features.parquet \
    --output-model data/models/logistic_regression_model.pkl \
    --output-scaler data/models/scaler.pkl
```
**What it does**:
- Trains logistic regression on engineered features
- Splits data: 70% train, 15% validation, 15% test
- Scales features using StandardScaler
- Evaluates on test set (accuracy, precision, recall, F1)
- Saves model, scaler, and feature importance
- **Time**: 1-2 minutes

### Phase 9: Evaluation (Local)

#### Step 9.1: Evaluate Models
```bash
python src/evaluation/evaluate_prediction.py \
    --llm-predictions data/processed/predictions/llm_predictions.parquet \
    --baseline-predictions data/processed/predictions/baseline_predictions.parquet \
    --output data/results/evaluation_results.csv \
    --correlation
```
**What it does**:
- Compares LLM-based model vs. baseline
- Calculates accuracy, precision, recall, F1-score for both
- Computes correlation between sentiment and price movements
- Performs statistical tests (Pearson/Spearman correlation)
- Saves evaluation results to `data/results/`
- **Time**: 1-2 minutes

### Phase 10: Visualization (Local)

#### Step 10.1: Create Visualizations
```bash
python src/evaluation/visualize.py \
    --sentiment data/processed/features/extracted_features.parquet \
    --market data/raw/markets/markets_2024_election_*.parquet \
    --predictions data/processed/predictions/llm_predictions.parquet \
    --features data/models/logistic_regression_model_feature_importance.csv \
    --output-dir data/results/plots/
```
**What it does**:
- Creates sentiment vs. price time-series plot
- Shows prediction accuracy over time
- Displays feature importance from logistic regression
- Generates correlation heatmap of features
- Saves all plots to `data/results/plots/`
- **Time**: 1-2 minutes

## Quick Start - All Commands in Sequence

```bash
# Setup
python setup.py
pip install -r requirements.txt

# Phase 1: Data Collection
python src/data_collection/tweet_scraper.py
python src/data_collection/market_data.py
python src/preprocessing/clean_tweets.py

# Phase 2: Labeling
python src/labeling/auto_labeler.py \
    -i data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    -o data/processed/labeled_tweets/auto_labels.parquet

# Phase 3: Alignment
python src/preprocessing/align_data.py \
    -t data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    -m data/raw/markets/markets_2024_election_*.parquet \
    -o data/processed/aligned_data.parquet --create-windows

# Phase 4: Baseline
python src/prediction/baseline.py \
    -t data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    -m data/raw/markets/markets_2024_election_*.parquet \
    -o data/processed/predictions/baseline_predictions.parquet

# Phase 5: Prepare for Colab
python src/model/inference.py \
    -t data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    -o data/processed/inference_data.parquet

# [Upload to Google Drive and run Colab notebook]

# Phase 6: Feature Extraction (after Colab)
python src/model/feature_extractor.py \
    -p data/processed/model_predictions.parquet \
    -o data/processed/features/extracted_features.parquet --aggregate

# Phase 7: Feature Engineering
python src/prediction/feature_engineering.py \
    -s data/processed/features/extracted_features_aggregated.parquet \
    -m data/raw/markets/markets_2024_election_*.parquet \
    -o data/processed/features/engineered_features.parquet

# Phase 8: Prediction Model
python src/prediction/market_predictor.py \
    -f data/processed/features/engineered_features.parquet \
    -o data/models/logistic_regression_model.pkl

# Phase 9: Evaluation
python src/evaluation/evaluate_prediction.py \
    --llm-predictions data/processed/predictions/llm_predictions.parquet \
    --baseline-predictions data/processed/predictions/baseline_predictions.parquet \
    -o data/results/evaluation_results.csv --correlation

# Phase 10: Visualization
python src/evaluation/visualize.py \
    --sentiment data/processed/features/extracted_features.parquet \
    --market data/raw/markets/markets_2024_election_*.parquet \
    --predictions data/processed/predictions/llm_predictions.parquet \
    -o data/results/plots/
```

## What Each Component Does (High Level)

### Data Collection
- **Tweet Scraper**: Collects tweets using `snscrape` (no API needed). Searches for election-related keywords and hashtags.
- **Market Data Collector**: Fetches prediction market prices from Polymarket API. Can use Kalshi API with credentials.

### Preprocessing
- **Tweet Cleaner**: Removes URLs, normalizes text, preserves emojis. Filters by length.
- **Data Aligner**: Matches tweet timestamps with market prices. Creates time windows for correlation analysis.

### Labeling
- **Auto Labeler**: Uses TextBlob + VADER for sentiment. Keyword matching for stance. Fully automated - no manual work.

### Model Training (Colab)
- **Fine-tuning**: Improves LLM's ability to classify sentiment/stance on election tweets using auto-labeled data.
- **Inference**: Runs fine-tuned model on all tweets to generate predictions.

### Feature Extraction
- **Feature Extractor**: Converts LLM predictions into numerical features (sentiment polarity, stance scores, confidence).
- **Time Aggregation**: Creates hourly/daily aggregations of sentiment features.

### Feature Engineering
- **Feature Combiner**: Merges sentiment features with market data (prices, volumes).
- **Target Creation**: Sets up prediction target (price direction 6 hours ahead).

### Prediction Models
- **Baseline**: Simple rule-based model (positive sentiment → price up). Uses TextBlob/VADER.
- **Logistic Regression**: ML model that learns weights for sentiment + market features to predict price movements.

### Evaluation
- **Model Comparison**: Compares LLM-based model vs. baseline (accuracy, precision, recall, F1).
- **Correlation Analysis**: Measures correlation between sentiment indicators and price movements.

### Visualization
- **Plots**: Creates time-series plots, accuracy charts, feature importance, correlation heatmaps.

## Configuration

Edit `config/config.yaml` to customize:
- **Tweet queries**: Search terms for tweet collection
- **Date ranges**: Collection period (default: 90 days)
- **Model parameters**: LLM fine-tuning settings
- **Time windows**: Feature aggregation windows (1h, 6h, 24h)
- **Prediction horizon**: Hours ahead to predict (default: 6)

## Expected Results

- **Baseline Model**: ~50-55% accuracy (slightly better than random)
- **LLM-based Model**: Should outperform baseline by 5-10% accuracy
- **Correlation**: Sentiment indicators should show statistically significant correlation with price movements

## Time Estimates

- **Data Collection**: 1-2 hours
- **Auto-labeling**: 5-15 minutes (automatic)
- **Feature Engineering**: 10-30 minutes
- **Model Training (Colab)**: 1-3 hours
- **Evaluation**: 5-10 minutes
- **Total Local Time**: ~2-3 hours
- **Total Colab Time**: ~1-3 hours

## Notes

- **No API Keys Needed**: Tweet collection uses `snscrape` (free, no authentication)
- **Colab Required**: Model fine-tuning needs GPU (free T4 GPU in Colab)
- **Automated Labeling**: All labeling is automated - no manual work required
- **Market Data**: Polymarket uses real API; Kalshi uses fallback (add credentials for real data)

## Troubleshooting

- **No tweets collected?** Check internet connection and search queries in config
- **Colab GPU not available?** Select Runtime > Change runtime type > GPU
- **Import errors?** Make sure all dependencies are installed: `pip install -r requirements.txt`

## See Also

- `WORKFLOW.md` - Detailed workflow guide
- `ELECTION_2024_SETUP.md` - Election-specific setup instructions
- `QUICKSTART.md` - Quick reference guide
