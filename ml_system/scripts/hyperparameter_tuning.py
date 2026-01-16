#!/usr/bin/env python3
"""
Day 8: Hyperparameter Tuning
Optimize Random Forest parameters using GridSearchCV
"""

import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def tune_hyperparameters(
    data_file='ml_system/data/training_data.csv',
    output_file='ml_system/models/tuned_rf.pkl',
    cv_folds=3,
    n_jobs=-1
):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        data_file: Path to training data
        output_file: Path to save tuned model
        cv_folds: Number of cross-validation folds (default 3 for small dataset)
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
    """

    print("=" * 80)
    print("DAY 8: HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {data_file}")
    print(f"Output: {output_file}")
    print(f"CV Folds: {cv_folds}")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/5] Loading training data...")
    try:
        df = pd.read_csv(data_file)
        print(f"[OK] Loaded {len(df)} samples")
        print(f"  Features: {len(df.columns) - 1}")
        print(f"  Win rate: {df['target'].mean():.2%}")
    except FileNotFoundError:
        print(f"[ERROR] Error: {data_file} not found!")
        return None

    X = df.drop('target', axis=1)
    y = df['target']

    # Check dataset size
    if len(df) < 10:
        print(f"\n  [WARN] Warning: Very small dataset ({len(df)} samples)")
        print(f"     Hyperparameter tuning may not be reliable")
        print(f"     Consider collecting more data for better results")

    # Step 2: Define parameter grid
    print("\n[2/5] Defining hyperparameter grid...")

    # Adjusted for small dataset
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    print("  Parameter grid:")
    for param, values in param_grid.items():
        print(f"    {param}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n  Total combinations: {total_combinations}")
    print(f"  Total fits: {total_combinations * cv_folds}")

    # Step 3: Set up GridSearchCV
    print(f"\n[3/5] Setting up GridSearchCV...")

    # Use accuracy as scoring metric (simpler for small datasets)
    scoring = 'accuracy'

    base_model = RandomForestClassifier(random_state=42, n_jobs=1)

    # Adjust CV folds if dataset is too small
    actual_cv_folds = min(cv_folds, len(df) // 2, len(y[y==0]), len(y[y==1]))
    if actual_cv_folds < cv_folds:
        print(f"  [WARN] Adjusting CV folds from {cv_folds} to {actual_cv_folds} (limited by dataset size)")
        cv_folds = max(2, actual_cv_folds)

    print(f"  Scoring: {scoring}")
    print(f"  CV Folds: {cv_folds}")
    print(f"  N Jobs: {n_jobs}")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )

    # Step 4: Run grid search
    print(f"\n[4/5] Running grid search...")
    print("  This may take a few minutes...")
    print()

    try:
        grid_search.fit(X, y)
        print("\n[OK] Grid search completed!")
    except Exception as e:
        print(f"\n[ERROR] Error during grid search: {e}")
        print("  Trying with 2-fold CV...")
        try:
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=2,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                return_train_score=True
            )
            grid_search.fit(X, y)
            print("\n[OK] Grid search completed with 2-fold CV!")
        except Exception as e2:
            print(f"\n[ERROR] Error: {e2}")
            print("  Dataset may be too small for hyperparameter tuning")
            return None

    # Step 5: Analyze results
    print("\n[5/5] Analyzing results...")

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print("\n  🏆 Best Parameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")

    print(f"\n   Best CV Score: {best_score:.3f} ({scoring})")

    # Compare with baseline
    baseline_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    }

    print("\n   Comparison with Baseline:")
    print("    Baseline parameters:")
    for param, value in baseline_params.items():
        print(f"      {param}: {value}")

    # Show top 5 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')

    print("\n   Top 5 Parameter Combinations:")
    print("  " + "-" * 76)
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"  {i+1}. Score: {row['mean_test_score']:.3f} (±{row['std_test_score']:.3f})")
        print(f"     Params: {row['params']}")

    # Save tuned model
    print(f"\n   Saving tuned model...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  [OK] Saved to: {output_file}")

    # Save results
    results_file = Path(output_file).parent / 'tuning_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"  [OK] Saved tuning results to: {results_file}")

    # Save metadata
    import json
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'best_params': {k: int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in best_params.items()},
        'best_score': float(best_score),
        'scoring': scoring,
        'cv_folds': int(cv_folds),
        'training_samples': int(len(X)),
        'total_combinations_tested': int(total_combinations)
    }

    metadata_file = Path(output_file).parent / 'tuned_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Saved metadata to: {metadata_file}")

    # Summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 80)

    print(f"\n[OK] Successfully tuned Random Forest model")
    print(f"[OK] Best CV Score: {best_score:.3f}")
    print(f"[OK] Tested {total_combinations} parameter combinations")

    print("\n🔑 Key Parameters Changed:")
    for param in best_params:
        if best_params[param] != baseline_params.get(param):
            print(f"  * {param}: {baseline_params.get(param)} -> {best_params[param]}")

    # Check if tuning helped
    improvement = best_score - 1.0  # Compare with baseline (assuming baseline was 1.0)
    if best_score >= 0.95:
        print(f"\n[OK] Excellent performance! ({best_score:.1%})")
    elif best_score >= 0.70:
        print(f"\n[OK] Good performance! ({best_score:.1%})")
    elif best_score >= 0.60:
        print(f"\n[WARN] Moderate performance ({best_score:.1%})")
        print(f"   Consider collecting more training data")
    else:
        print(f"\n[WARN] Low performance ({best_score:.1%})")
        print(f"   More training data needed for reliable model")

    print("\n[OK] Ready for Day 9: Alternative Models")
    print("=" * 80)

    return best_model, best_params, best_score


if __name__ == '__main__':
    result = tune_hyperparameters()

    if result is not None:
        print("\n[OK] Day 8 Complete!")
    else:
        print("\n[ERROR] Day 8 Failed - check errors above")
