"""
Evaluation scripts for prediction model.
Compares LLM-based model with baseline and calculates correlation metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_predictions(predictions_df: pd.DataFrame, 
                        actual_col: str = 'actual_direction',
                        pred_col: str = 'predicted_direction') -> dict:
    """
    Evaluate prediction accuracy.
    
    Args:
        predictions_df: DataFrame with predictions and actual values
        actual_col: Column name for actual values
        pred_col: Column name for predictions
    
    Returns:
        Dictionary with evaluation metrics
    """
    actual = predictions_df[actual_col]
    predicted = predictions_df[pred_col]
    
    # Remove NaN
    valid_mask = ~(actual.isna() | predicted.isna())
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]
    
    metrics = {
        'accuracy': accuracy_score(actual, predicted),
        'precision': precision_score(actual, predicted, zero_division=0),
        'recall': recall_score(actual, predicted, zero_division=0),
        'f1_score': f1_score(actual, predicted, zero_division=0),
        'n_samples': len(actual)
    }
    
    return metrics


def calculate_correlation(sentiment_features: pd.Series, 
                        price_changes: pd.Series,
                        method: str = 'pearson') -> dict:
    """
    Calculate correlation between sentiment and price movements.
    
    Args:
        sentiment_features: Sentiment feature series
        price_changes: Price change series
        method: 'pearson' or 'spearman'
    
    Returns:
        Dictionary with correlation statistics
    """
    # Remove NaN
    valid_mask = ~(sentiment_features.isna() | price_changes.isna())
    sentiment = sentiment_features[valid_mask]
    prices = price_changes[valid_mask]
    
    if len(sentiment) < 2:
        return {'correlation': np.nan, 'p_value': np.nan, 'n_samples': len(sentiment)}
    
    if method == 'pearson':
        corr, p_value = pearsonr(sentiment, prices)
    elif method == 'spearman':
        corr, p_value = spearmanr(sentiment, prices)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'n_samples': len(sentiment),
        'method': method
    }


def compare_models(llm_predictions: pd.DataFrame,
                   baseline_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compare LLM-based model with baseline.
    
    Args:
        llm_predictions: Predictions from LLM-based model
        baseline_predictions: Predictions from baseline model
    
    Returns:
        DataFrame with comparison metrics
    """
    # Evaluate LLM model
    llm_metrics = evaluate_predictions(llm_predictions)
    llm_metrics['model'] = 'LLM-based'
    
    # Evaluate baseline
    baseline_metrics = evaluate_predictions(baseline_predictions)
    baseline_metrics['model'] = 'Baseline'
    
    # Create comparison
    comparison = pd.DataFrame([llm_metrics, baseline_metrics])
    
    # Calculate improvement
    comparison['improvement'] = comparison['accuracy'] - comparison.loc[comparison['model'] == 'Baseline', 'accuracy'].values[0]
    
    return comparison


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate prediction models")
    parser.add_argument('--llm-predictions', type=str, required=True,
                       help='LLM-based model predictions file')
    parser.add_argument('--baseline-predictions', type=str, required=True,
                       help='Baseline model predictions file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output file for evaluation results')
    parser.add_argument('--correlation', action='store_true',
                       help='Also calculate sentiment-price correlation')
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Load predictions
    print("Loading predictions...")
    llm_df = pd.read_parquet(args.llm_predictions) if Path(args.llm_predictions).suffix == '.parquet' else pd.read_json(args.llm_predictions, lines=True)
    baseline_df = pd.read_parquet(args.baseline_predictions) if Path(args.baseline_predictions).suffix == '.parquet' else pd.read_json(args.baseline_predictions, lines=True)
    
    print(f"Loaded {len(llm_df)} LLM predictions and {len(baseline_df)} baseline predictions")
    
    # Compare models
    print("\nComparing models...")
    comparison = compare_models(llm_df, baseline_df)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison[['model', 'accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False))
    print("=" * 60)
    
    # Calculate correlation if requested
    if args.correlation:
        print("\nCalculating sentiment-price correlations...")
        
        correlation_method = config['evaluation']['correlation_method']
        
        # LLM model correlation
        if 'sentiment_polarity_mean' in llm_df.columns and 'price_change' in llm_df.columns:
            llm_corr = calculate_correlation(
                llm_df['sentiment_polarity_mean'],
                llm_df['price_change'],
                method=correlation_method
            )
            llm_corr['model'] = 'LLM-based'
            print(f"LLM Correlation ({correlation_method}): {llm_corr['correlation']:.4f} (p={llm_corr['p_value']:.4f})")
        
        # Baseline correlation
        if 'baseline_sentiment_mean' in baseline_df.columns and 'price_change' in baseline_df.columns:
            baseline_corr = calculate_correlation(
                baseline_df['baseline_sentiment_mean'],
                baseline_df['price_change'],
                method=correlation_method
            )
            baseline_corr['model'] = 'Baseline'
            print(f"Baseline Correlation ({correlation_method}): {baseline_corr['correlation']:.4f} (p={baseline_corr['p_value']:.4f})")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.csv':
        comparison.to_csv(output_path, index=False)
    else:
        comparison.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()

