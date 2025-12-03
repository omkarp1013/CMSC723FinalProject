"""
Logistic regression model for predicting market movements from sentiment features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_features(features_df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for model training.
    
    Args:
        features_df: DataFrame with engineered features
    
    Returns:
        X (features), y (target)
    """
    # Exclude non-feature columns
    exclude_cols = ['hour', 'timestamp', 'target_direction', 'target_price', 'target_price_change']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_cols].copy()
    y = features_df['target_direction'].copy()
    
    # Remove NaN rows
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Fill remaining NaN with 0
    X = X.fillna(0)
    
    return X, y, feature_cols


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> tuple:
    """
    Train logistic regression model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        Trained model and scaler
    """
    print("Training logistic regression model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val_scaled)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return model, scaler


def evaluate_model(model, scaler, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate model on test set"""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train logistic regression model for market prediction")
    parser.add_argument('--features', '-f', type=str, required=True,
                       help='Engineered features file')
    parser.add_argument('--output-model', '-o', type=str, required=True,
                       help='Output path for trained model')
    parser.add_argument('--output-scaler', type=str, default=None,
                       help='Output path for scaler')
    parser.add_argument('--test-split', type=float, default=None,
                       help='Test split ratio (overrides config)')
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Load features
    print("Loading features...")
    features_df = pd.read_parquet(args.features) if Path(args.features).suffix == '.parquet' else pd.read_json(args.features, lines=True)
    
    print(f"Loaded {len(features_df)} feature vectors")
    
    # Prepare features
    X, y, feature_cols = prepare_features(features_df)
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    test_split = args.test_split or config['prediction']['test_split']
    val_split = config['prediction']['validation_split']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    
    val_size = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X):.2%})")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X):.2%})")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X):.2%})")
    
    # Train model
    model, scaler = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    results = evaluate_model(model, scaler, X_test, y_test)
    
    # Save model
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved model to {output_path}")
    
    # Save scaler
    scaler_path = Path(args.output_scaler) if args.output_scaler else output_path.parent / f"{output_path.stem}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler to {scaler_path}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    importance_path = output_path.parent / f"{output_path.stem}_feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Saved feature importance to {importance_path}")
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

