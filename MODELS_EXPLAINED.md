# Models Used in This Project - Complete Explanation

This document explains all the models used in the project and how they differ.

## Overview: Three Types of Models

This project uses **three different types of models** for different purposes:

1. **Baseline Model** - Simple rule-based sentiment analysis (for comparison)
2. **LLM (Large Language Model)** - Fine-tuned for sentiment/stance classification
3. **Logistic Regression** - Final prediction model (predicts market movements)

---

## 1. Baseline Model (`baseline.py`)

### What It Does
- **Purpose**: Provides a simple baseline for comparison
- **Task**: Predicts market price direction using basic sentiment analysis
- **Method**: Rule-based (no machine learning)

### How It Works
1. Uses **TextBlob** and **VADER** (pre-trained sentiment analyzers)
2. Calculates sentiment score for each tweet (-1 to +1)
3. Aggregates sentiment by hour
4. **Simple rule**: 
   - Positive sentiment → Predict price goes UP
   - Negative sentiment → Predict price goes DOWN
   - Neutral sentiment → Predict no change

### Models Used
- **TextBlob**: Python library for sentiment analysis (uses pattern matching)
- **VADER**: Valence Aware Dictionary and sEntiment Reasoner (designed for social media)

### Expected Performance
- **Accuracy**: ~50-55% (slightly better than random 50%)
- **Why it's limited**: Doesn't learn from data, just uses simple rules

### When It Runs
- Runs **locally** (no GPU needed)
- Uses the same tweets as the LLM model
- Provides comparison point to show if LLM improves predictions

---

## 2. LLM Model (Large Language Model) - Fine-Tuned

### What It Does
- **Purpose**: Classifies sentiment, stance, and betting direction more accurately
- **Task**: Takes tweet text → outputs sentiment/stance/confidence scores
- **Method**: Fine-tuned transformer model (deep learning)

### Model Used
- **Base Model**: `Llama-3.2-3B-Instruct` (3 billion parameters)
- **Fine-tuning**: Uses **LoRA** (Low-Rank Adaptation) for efficient training
- **Framework**: Unsloth (optimized for Colab)

### How It Works
1. **Pre-training**: Llama 3.2 was pre-trained on massive text data (general knowledge)
2. **Fine-tuning**: We fine-tune it on auto-labeled election tweets to improve:
   - Sentiment classification (positive/negative/neutral)
   - Stance detection (bullish/bearish/neutral)
   - Context understanding (election-specific language)
3. **Inference**: Runs fine-tuned model on all tweets to generate predictions

### Why Use LLM Instead of Baseline?
- **Better context understanding**: Understands election-specific language, sarcasm, context
- **More nuanced**: Can detect subtle sentiment that TextBlob/VADER miss
- **Stance detection**: Better at determining if sentiment suggests price up/down
- **Confidence scores**: Provides confidence in predictions

### Expected Performance
- **Sentiment accuracy**: ~70-80% (better than baseline)
- **Stance accuracy**: ~65-75%

### When It Runs
- **Training**: Google Colab (needs GPU)
- **Inference**: Google Colab (needs GPU)
- **Output**: Predictions saved to Google Drive, downloaded locally

### Key Difference from Baseline
- Baseline: Simple rule (positive → up, negative → down)
- LLM: Learns complex patterns, understands context, provides confidence scores

---

## 3. Logistic Regression Model (`market_predictor.py`)

### What It Does
- **Purpose**: Final prediction model that predicts market price movements
- **Task**: Takes sentiment features + market features → predicts price direction (up/down)
- **Method**: Supervised machine learning (logistic regression)

### How It Works
1. **Input Features**:
   - Sentiment features from LLM (polarity, stance, confidence)
   - Market features (current price, volume, momentum)
   - Time-aggregated features (rolling means, trends)
   - Combined features (sentiment × market interactions)

2. **Target Variable**: 
   - Price direction 6 hours ahead (1 = up, 0 = down)

3. **Training**:
   - Learns weights for each feature
   - Determines which features are most important
   - Optimizes to maximize prediction accuracy

4. **Prediction**:
   - Takes current sentiment + market state
   - Outputs probability of price increase
   - If probability > 0.5 → predict UP, else DOWN

### Model Used
- **Logistic Regression**: Simple but effective linear classifier
- **Why logistic regression?**
  - Interpretable (can see which features matter)
  - Fast training and inference
  - Works well with time-series features
  - Good baseline before trying more complex models

### Expected Performance
- **Accuracy**: 55-65% (better than baseline, uses LLM features)
- **Improvement over baseline**: 5-10% accuracy gain

### When It Runs
- **Training**: Locally (no GPU needed)
- **Inference**: Locally (fast, no GPU needed)

### Key Difference from Baseline/LLM
- **Baseline/LLM**: Classify sentiment/stance (text → sentiment)
- **Logistic Regression**: Predicts market movements (features → price direction)

---

## Model Pipeline Flow

```
Tweets
  ↓
[Baseline Model] → Simple sentiment → Predict price direction
  ↓
[Auto-Labeler] → TextBlob/VADER → Labels for fine-tuning
  ↓
[LLM Fine-Tuning] → Learn from labels → Better sentiment/stance classifier
  ↓
[LLM Inference] → Run on all tweets → Generate sentiment features
  ↓
[Feature Engineering] → Combine sentiment + market data → Create features
  ↓
[Logistic Regression] → Learn from features → Predict price movements
  ↓
[Evaluation] → Compare all models → Measure performance
```

---

## Comparison Table

| Model | Type | Purpose | Input | Output | Runs Where |
|-------|------|---------|-------|--------|------------|
| **Baseline** | Rule-based | Compare against | Tweet text | Price direction | Local |
| **TextBlob/VADER** | Pre-trained | Auto-labeling | Tweet text | Sentiment score | Local |
| **LLM (Llama 3.2)** | Fine-tuned transformer | Sentiment classification | Tweet text | Sentiment/stance/confidence | Colab |
| **Logistic Regression** | Supervised ML | Price prediction | Features (sentiment + market) | Price direction | Local |

---

## Why Three Models?

1. **Baseline Model**: 
   - Shows that simple sentiment analysis isn't enough
   - Provides comparison point
   - Demonstrates improvement from LLM

2. **LLM Model**:
   - Generates high-quality sentiment features
   - Better than baseline at understanding context
   - Provides features for final prediction model

3. **Logistic Regression**:
   - Combines sentiment features with market data
   - Learns which features predict price movements
   - Final model that makes actual predictions

---

## Summary

- **Baseline**: Simple rule (positive sentiment → price up). Uses TextBlob/VADER.
- **LLM**: Fine-tuned Llama 3.2 for better sentiment/stance classification. Generates features.
- **Logistic Regression**: Learns from LLM features + market data to predict price movements.

The **baseline** is for comparison, the **LLM** improves sentiment analysis, and **logistic regression** makes the final predictions using all the features.

