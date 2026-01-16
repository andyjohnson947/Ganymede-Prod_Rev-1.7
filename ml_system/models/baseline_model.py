#!/usr/bin/env python3
"""
Day 6: Baseline Model Training
Trains first Random Forest model with default parameters
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)


def train_baseline_model(
    data_file='ml_system/data/training_data.csv',
    model_file='ml_system/models/baseline_rf.pkl',
    test_size=0.2,
    random_state=42
):
    """
    Train baseline Random Forest model.

    Args:
        data_file: Path to training data CSV
        model_file: Path to save trained model
        test_size: Proportion for test set (0.2 = 20%)
        random_state: Random seed for reproducibility
    """

    print("=" * 80)
    print("DAY 6: BASELINE MODEL TRAINING")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {data_file}")
    print(f"Model output: {model_file}")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/6] Loading training data...")
    try:
        df = pd.read_csv(data_file)
        print(f"[OK] Loaded {len(df)} samples")
        print(f"  Features: {len(df.columns) - 1}")
        print(f"  Win rate: {df['target'].mean():.2%}")
    except FileNotFoundError:
        print(f"[ERROR] Error: {data_file} not found!")
        print("  Run create_dataset.py first (Day 5)")
        return None

    # Step 2: Split features and target
    print("\n[2/6] Preparing data...")
    X = df.drop('target', axis=1)
    y = df['target']

    print(f"[OK] Features shape: {X.shape}")
    print(f"[OK] Target shape: {y.shape}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")

    # Step 3: Train/val split
    print(f"\n[3/6] Splitting data (test_size={test_size})...")

    if len(df) < 5:
        print("  [WARN] Warning: Very small dataset, using all data for training")
        X_train, X_val = X, X
        y_train, y_val = y, y
        has_val_set = False
    else:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            has_val_set = True
        except ValueError:
            # Stratify failed (not enough samples per class)
            print("  [WARN] Warning: Cannot stratify with small dataset")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            has_val_set = True

    print(f"[OK] Training samples: {len(X_train)}")
    if has_val_set:
        print(f"[OK] Validation samples: {len(X_val)}")
        print(f"  Train win rate: {y_train.mean():.2%}")
        print(f"  Val win rate: {y_val.mean():.2%}")

    # Step 4: Train baseline model
    print("\n[4/6] Training Random Forest...")
    print("  Hyperparameters:")
    print("    n_estimators: 100")
    print("    max_depth: 10")
    print("    random_state: 42")
    print("    n_jobs: -1")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("[OK] Model trained successfully!")

    # Step 5: Evaluate
    print("\n[5/6] Evaluating model...")

    # Training metrics
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    print(f"\n  Training Accuracy: {train_accuracy:.2%}")

    if has_val_set:
        # Validation metrics
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]

        val_accuracy = accuracy_score(y_val, y_val_pred)

        print(f"  Validation Accuracy: {val_accuracy:.2%}")

        # Detailed metrics (if we have both classes)
        unique_classes = len(np.unique(y_val))
        if unique_classes > 1:
            try:
                val_precision = precision_score(y_val, y_val_pred)
                val_recall = recall_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred)
                val_roc_auc = roc_auc_score(y_val, y_val_prob)

                print(f"  Validation Precision: {val_precision:.2%}")
                print(f"  Validation Recall: {val_recall:.2%}")
                print(f"  Validation F1: {val_f1:.2%}")
                print(f"  Validation ROC-AUC: {val_roc_auc:.3f}")

                print("\n  Classification Report (Validation):")
                print(classification_report(y_val, y_val_pred, target_names=['Loss', 'Win']))

                print("\n  Confusion Matrix (Validation):")
                cm = confusion_matrix(y_val, y_val_pred)
                print(cm)
                print("               Predicted")
                print("             Loss    Win")
                print(f"  Actual Loss    {cm[0,0] if len(cm) > 0 and cm.shape[0] > 1 else 0}      {cm[0,1] if len(cm) > 0 and cm.shape[1] > 1 else cm[0,0] if len(cm) > 0 else 0}")
                print(f"         Win     {cm[1,0] if len(cm) > 1 else 0}      {cm[1,1] if len(cm) > 1 else cm[0,0] if len(cm) == 1 else 0}")

            except Exception as e:
                print(f"  [WARN] Could not compute all metrics: {e}")
        else:
            print(f"  [WARN] Validation set only contains one class")

    # Step 6: Feature importance
    print("\n[6/6] Analyzing feature importance...")

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n  Top 15 Most Important Features:")
    print("  " + "-" * 76)
    for i, row in feature_importance.head(15).iterrows():
        bar_length = int(row['importance'] * 50)
        bar = '#' * bar_length
        print(f"  {row['feature']:30s} {row['importance']:6.4f} {bar}")

    # Save feature importance
    importance_file = Path(model_file).parent / 'feature_importance_baseline.csv'
    feature_importance.to_csv(importance_file, index=False)
    print(f"\n[OK] Saved feature importance to: {importance_file}")

    # Save model
    print("\n  Saving model...")
    Path(model_file).parent.mkdir(parents=True, exist_ok=True)

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] Model saved to: {model_file}")

    # Save model metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 10,
        'training_samples': len(X_train),
        'validation_samples': len(X_val) if has_val_set else 0,
        'features': len(X.columns),
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy) if has_val_set else None,
        'val_roc_auc': float(val_roc_auc) if has_val_set and unique_classes > 1 else None,
    }

    import json
    metadata_file = Path(model_file).parent / 'baseline_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Model metadata saved to: {metadata_file}")

    # Summary
    print("\n" + "=" * 80)
    print("BASELINE MODEL SUMMARY")
    print("=" * 80)
    print(f"\n[OK] Successfully trained baseline Random Forest model")
    print(f"[OK] Training accuracy: {train_accuracy:.2%}")
    if has_val_set:
        print(f"[OK] Validation accuracy: {val_accuracy:.2%}")
        if unique_classes > 1:
            print(f"[OK] Validation ROC-AUC: {val_roc_auc:.3f}")

    print("\n Top 5 Features:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {i+1}. {row['feature']} ({row['importance']:.3f})")

    # Success criteria check
    print("\n[OK] Success Criteria:")
    if has_val_set and val_accuracy >= 0.60:
        print(f"  [OK] Validation accuracy > 60% ({val_accuracy:.1%})")
    elif has_val_set:
        print(f"  [WARN] Validation accuracy < 60% ({val_accuracy:.1%})")
        print(f"     Consider collecting more training data")
    else:
        print(f"  [WARN] No validation set (dataset too small)")

    print("\n[OK] Ready for Day 7: Model Evaluation & Analysis")
    print("=" * 80)

    return model, feature_importance


if __name__ == '__main__':
    model, importance = train_baseline_model()

    if model is not None:
        print("\n[OK] Day 6 Complete!")
    else:
        print("\n[ERROR] Day 6 Failed - check errors above")
