# Tweet-Informed Prediction Market Forecasting

This project uses Twitter sentiment analysis (via fine-tuned LLM) to predict short-term movements in prediction markets (Kalshi/Polymarket).

## Project Structure

```
CMSC723FinalProject/
├── data/                    # Data directory (create manually)
│   ├── raw/                 # Raw collected data
│   ├── processed/           # Processed and labeled data
│   └── models/              # Model checkpoints
├── src/                     # Source code
├── notebooks/               # Jupyter notebooks
├── config/                  # Configuration files
└── requirements.txt         # Python dependencies
```

## Setup

### Local Environment (Data Collection)

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create data directories:
```bash
mkdir -p data/raw/tweets data/raw/markets data/processed/labeled_tweets data/processed/features data/models
```

### Google Colab Setup (Model Training)

1. Upload this project folder to Google Drive
2. Open `notebooks/model_training.ipynb` in Google Colab
3. Mount Google Drive and install Colab dependencies (see notebook)

## Workflow

1. **Data Collection** (Local): Run tweet scraper and market data collection
2. **Preprocessing** (Local): Clean and align data
3. **Labeling** (Local): Manually label seed data
4. **Model Training** (Colab): Fine-tune LLM in Google Colab
5. **Feature Extraction** (Colab): Extract sentiment features
6. **Prediction** (Local/Colab): Build and evaluate prediction models
7. **Evaluation** (Local): Analyze results and create visualizations

## Quick Start

1. Collect tweets:
```bash
python src/data_collection/tweet_scraper.py
```

2. Clean data:
```bash
python src/preprocessing/clean_tweets.py
```

3. For model training, use the Colab notebook: `notebooks/model_training.ipynb`

## Configuration

Edit `config/config.yaml` to customize:
- Search queries for tweet collection
- Model parameters
- Time windows for feature aggregation
- Prediction horizons

## Notes

- This project uses `snscrape` for tweet collection (no API keys needed)
- Model fine-tuning requires Google Colab (free GPU access)
- Data is synced between local and Colab via Google Drive

