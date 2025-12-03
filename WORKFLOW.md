# Complete Workflow Guide

This guide walks you through the entire pipeline from data collection to evaluation, excluding model training (which is done in Colab).

## Phase 1: Data Collection

### 1.1 Collect Tweets
```bash
python src/data_collection/tweet_scraper.py
```
- Collects tweets using election-focused queries
- Saves to `data/raw/tweets/`
- Takes 10-30 minutes depending on volume

### 1.2 Collect Market Data
```bash
python src/data_collection/market_data.py
```
- Fetches Polymarket data (real API)
- Generates fallback Kalshi data
- Saves to `data/raw/markets/`

### 1.3 Clean Tweets
```bash
python src/preprocessing/clean_tweets.py
```
- Cleans and normalizes tweet text
- Removes URLs, normalizes whitespace
- Saves to `data/processed/cleaned_tweets/`

## Phase 2: Automated Labeling

### 2.1 Auto-Label Tweets
```bash
python src/labeling/auto_labeler.py \
    --input data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    --output data/processed/labeled_tweets/auto_labels.parquet
```
- **Automated**: Uses TextBlob + VADER for sentiment
- **Keyword matching**: Determines stance (bullish/bearish)
- **No manual work required**: Processes all tweets automatically
- Takes 5-15 minutes depending on dataset size

### 2.2 Generate Synthetic Data (Optional)
```bash
python src/labeling/synthetic_generator.py \
    --seed data/processed/labeled_tweets/auto_labels.parquet \
    --output data/processed/labeled_tweets/synthetic_labels.parquet \
    --num 2000
```
- **Note**: Currently uses placeholder. Implement LLM call for real generation.
- Expands dataset with synthetic examples

## Phase 3: Data Alignment

### 3.1 Align Tweets with Market Data
```bash
python src/preprocessing/align_data.py \
    --tweets data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    --markets data/raw/markets/markets_2024_election_*.parquet \
    --output data/processed/aligned_data.parquet \
    --window 6 \
    --create-windows
```
- Aligns tweet timestamps with market prices
- Creates time-windowed aggregations
- Essential for correlation analysis

## Phase 4: Model Training (Google Colab)

### 4.1 Prepare Data for Colab
```bash
python src/model/inference.py \
    --tweets data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    --output data/processed/inference_data.parquet
```

### 4.2 Upload to Google Drive
1. Upload `data/processed/inference_data.parquet` to Google Drive
2. Upload `data/processed/labeled_tweets/` to Google Drive
3. Open `notebooks/model_training.ipynb` in Colab

### 4.3 Train Model in Colab
- Follow notebook instructions
- Fine-tune LLM on labeled data
- Run inference on all tweets
- Save predictions to Google Drive

### 4.4 Download Results
- Download model predictions from Google Drive
- Save to `data/processed/model_predictions.parquet`

## Phase 5: Feature Extraction

### 5.1 Extract Features from Predictions
```bash
python src/model/feature_extractor.py \
    --predictions data/processed/model_predictions.parquet \
    --output data/processed/features/extracted_features.parquet \
    --aggregate
```
- Extracts sentiment, stance, confidence features
- Creates time-windowed aggregations
- Prepares features for prediction model

## Phase 6: Feature Engineering

### 6.1 Combine Sentiment and Market Features
```bash
python src/prediction/feature_engineering.py \
    --sentiment data/processed/features/extracted_features_aggregated.parquet \
    --market data/raw/markets/markets_2024_election_*.parquet \
    --output data/processed/features/engineered_features.parquet \
    --horizon 6
```
- Combines sentiment features with market data
- Creates prediction features
- Sets up target variable (price direction)

## Phase 7: Baseline Model

### 7.1 Run Baseline
```bash
python src/prediction/baseline.py \
    --tweets data/processed/cleaned_tweets/cleaned_tweets_combined_*.parquet \
    --markets data/raw/markets/markets_2024_election_*.parquet \
    --output data/processed/predictions/baseline_predictions.parquet
```
- Uses TextBlob + VADER for sentiment
- Simple rule: positive sentiment → price up
- Provides comparison baseline

## Phase 8: Prediction Model

### 8.1 Train Logistic Regression
```bash
python src/prediction/market_predictor.py \
    --features data/processed/features/engineered_features.parquet \
    --output-model data/models/logistic_regression_model.pkl \
    --output-scaler data/models/scaler.pkl
```
- Trains logistic regression on engineered features
- Evaluates on test set
- Saves model and scaler

## Phase 9: Evaluation

### 9.1 Evaluate Models
```bash
python src/evaluation/evaluate_prediction.py \
    --llm-predictions data/processed/predictions/llm_predictions.parquet \
    --baseline-predictions data/processed/predictions/baseline_predictions.parquet \
    --output data/results/evaluation_results.csv \
    --correlation
```
- Compares LLM-based model vs. baseline
- Calculates accuracy, precision, recall, F1
- Computes sentiment-price correlations

## Phase 10: Visualization

### 10.1 Create Visualizations
```bash
python src/evaluation/visualize.py \
    --sentiment data/processed/features/extracted_features.parquet \
    --market data/raw/markets/markets_2024_election_*.parquet \
    --predictions data/processed/predictions/llm_predictions.parquet \
    --features data/models/logistic_regression_model_feature_importance.csv \
    --output-dir data/results/plots/
```
- Creates sentiment vs. price plots
- Shows prediction accuracy over time
- Displays feature importance
- Generates correlation heatmaps

## Quick Reference: All Commands

```bash
# Data Collection
python src/data_collection/tweet_scraper.py
python src/data_collection/market_data.py
python src/preprocessing/clean_tweets.py

# Labeling
python src/labeling/manual_labeler.py -i <tweets> -o <output> -n 300
python src/labeling/synthetic_generator.py -s <seed> -o <output> -n 2000

# Alignment
python src/preprocessing/align_data.py -t <tweets> -m <markets> -o <output> --create-windows

# Feature Extraction (after Colab)
python src/model/feature_extractor.py -p <predictions> -o <output> --aggregate

# Feature Engineering
python src/prediction/feature_engineering.py -s <sentiment> -m <market> -o <output>

# Models
python src/prediction/baseline.py -t <tweets> -m <markets> -o <output>
python src/prediction/market_predictor.py -f <features> -o <model>

# Evaluation
python src/evaluation/evaluate_prediction.py --llm-predictions <llm> --baseline-predictions <baseline> -o <output> --correlation
python src/evaluation/visualize.py --sentiment <sentiment> --market <market> --predictions <pred> -o <plots>
```

## Expected File Structure After Completion

```
data/
├── raw/
│   ├── tweets/              # Collected tweets
│   └── markets/             # Market data
├── processed/
│   ├── cleaned_tweets/      # Cleaned tweets
│   ├── labeled_tweets/      # Manual + synthetic labels
│   ├── aligned_data.parquet # Aligned tweets + markets
│   ├── inference_data.parquet # For Colab
│   ├── model_predictions.parquet # From Colab
│   └── features/
│       ├── extracted_features.parquet
│       └── engineered_features.parquet
├── models/
│   ├── logistic_regression_model.pkl
│   └── scaler.pkl
└── results/
    ├── evaluation_results.csv
    └── plots/               # Visualizations
```

## Notes

- **Model Training**: Done in Google Colab (see `notebooks/model_training.ipynb`)
- **Synthetic Data**: Currently placeholder - implement LLM API call
- **Market Data**: Polymarket uses real API, Kalshi uses fallback (add API credentials for real data)
- **Time Estimates**: 
  - Data collection: 1-2 hours
  - Auto-labeling: 5-15 minutes (automatic, no manual work)
  - Feature engineering: 10-30 minutes
  - Model training: 1-3 hours (Colab)
  - Evaluation: 5-10 minutes

