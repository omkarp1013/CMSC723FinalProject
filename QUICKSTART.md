# Quick Start Guide

## Initial Setup (5 minutes)

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup script:**
   ```bash
   python setup.py
   ```
   This creates all necessary directories.

## Data Collection (Local)

### Step 1: Collect Tweets
```bash
python src/data_collection/tweet_scraper.py
```
This will:
- Scrape tweets using snscrape (no API needed)
- Search for queries in `config/config.yaml`
- Save to `data/raw/tweets/`

**Note:** This may take 10-30 minutes depending on how many tweets you're collecting.

### Step 2: Collect Market Data
```bash
python src/data_collection/market_data.py
```
**Note:** This currently uses placeholder data. You'll need to implement actual API calls for Kalshi/Polymarket.

### Step 3: Clean Tweets
```bash
python src/preprocessing/clean_tweets.py
```
This cleans and normalizes the collected tweets.

### Or run all at once:
```bash
python main.py all
```

## Model Training (Google Colab)

1. **Upload project to Google Drive:**
   - Upload the entire `CMSC723FinalProject` folder to `MyDrive/`

2. **Open Colab notebook:**
   - Open `notebooks/model_training.ipynb` in Google Colab
   - Or create a new notebook and copy the cells

3. **Run the notebook:**
   - Mount Google Drive
   - Install dependencies
   - Load your labeled data
   - Fine-tune the model
   - Save model to Google Drive

## Next Steps

After data collection:
1. **Auto-label tweets** - runs automatically using TextBlob + VADER
2. **Generate synthetic data** (optional) using LLM
3. **Fine-tune model** in Colab using auto-labeled data
4. **Extract features** and build prediction model

## Troubleshooting

- **snscrape not working?** Make sure you have the latest version: `pip install --upgrade snscrape`
- **No tweets collected?** Check your search queries in `config/config.yaml`
- **Colab GPU not available?** Make sure Runtime > Change runtime type > GPU is selected

## Configuration

Edit `config/config.yaml` to customize:
- Search queries for tweets
- Date ranges
- Model parameters
- Time windows

