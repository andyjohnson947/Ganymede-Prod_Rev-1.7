#!/usr/bin/env python3
"""
Day 5: Dataset Creation Script
Creates clean ML dataset from continuous trade log
"""

import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_system.features.extractor import FeatureExtractor


def create_dataset(
    input_file='ml_system/outputs/continuous_trade_log.jsonl',
    output_file='ml_system/data/training_data.csv',
    test_size=0.2,
    val_size=0.2
):
    """
    Create ML dataset from trade log.

    Args:
        input_file: Path to continuous trade log
        output_file: Path to save dataset CSV
        test_size: Proportion for test set (0.2 = 20%)
        val_size: Proportion for validation set (0.2 = 20%)
    """

    print("=" * 80)
    print("DAY 5: DATASET CREATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("=" * 80)

    # Step 1: Load trades
    print("\n[1/5] Loading trades from log...")
    trades = []

    try:
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return None

    print(f"[OK] Loaded {len(trades)} trades")

    # Step 2: Extract features
    print("\n[2/5] Extracting features...")
    extractor = FeatureExtractor()

    feature_rows = []
    targets = []
    skipped = 0

    for trade in trades:
        try:
            # Extract features
            features = extractor.extract_features(trade)
            target = extractor.extract_target(trade)

            # Only include closed trades (with outcomes)
            if target is not None:
                # Validate features
                if extractor.validate_features(features):
                    feature_rows.append(features)
                    targets.append(target)
                else:
                    print(f"Warning: Invalid features for trade {trade.get('ticket')}")
                    skipped += 1
        except Exception as e:
            print(f"Warning: Error processing trade {trade.get('ticket')}: {e}")
            skipped += 1

    if not feature_rows:
        print("\n[ERROR] No valid closed trades found!")
        print("  Need closed trades (with outcomes) to create dataset.")
        return None

    print(f"[OK] Extracted features from {len(feature_rows)} closed trades")
    if skipped > 0:
        print(f"  Skipped {skipped} trades due to errors or validation failures")

    # Step 3: Create DataFrame
    print("\n[3/5] Creating dataset...")
    df = pd.DataFrame(feature_rows)
    df['target'] = targets

    print(f"[OK] Created dataset with {len(df)} samples")
    print(f"  Features: {len(df.columns) - 1}")
    print(f"  Target: {df.columns[-1]}")

    # Dataset statistics
    print("\n  Dataset Statistics:")
    print(f"    Total samples: {len(df)}")
    print(f"    Win rate: {df['target'].mean():.2%}")
    print(f"    Wins: {df['target'].sum()}")
    print(f"    Losses: {len(df) - df['target'].sum()}")

    # Check for class imbalance
    win_rate = df['target'].mean()
    if win_rate > 0.9 or win_rate < 0.1:
        print(f"\n  [WARN] Warning: Severe class imbalance detected ({win_rate:.1%} win rate)")
        print(f"     Consider collecting more diverse data")

    # Step 4: Feature statistics
    print("\n[4/5] Analyzing features...")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n  [WARN] Missing values detected:")
        for col in missing[missing > 0].index:
            print(f"    {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")
    else:
        print("  [OK] No missing values")

    # Feature value ranges
    print("\n  Feature Value Ranges:")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols[:5]:  # Show first 5 numeric features
        print(f"    {col:30s} [{df[col].min():.2f}, {df[col].max():.2f}]")

    # Step 5: Save dataset
    print("\n[5/5] Saving dataset...")

    # Create data directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    df.to_csv(output_file, index=False)
    print(f"[OK] Saved to: {output_file}")

    # Save dataset info
    info_file = output_path.parent / 'dataset_info.txt'
    with open(info_file, 'w') as f:
        f.write("ML DATASET INFORMATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {input_file}\n")
        f.write(f"Output: {output_file}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Features: {len(df.columns) - 1}\n")
        f.write(f"Win rate: {df['target'].mean():.2%}\n")
        f.write(f"Wins: {df['target'].sum()}\n")
        f.write(f"Losses: {len(df) - df['target'].sum()}\n\n")
        f.write("Feature List:\n")
        for i, col in enumerate(df.columns[:-1], 1):
            f.write(f"  {i:2d}. {col}\n")
        f.write("\nTarget Variable: target (0=loss, 1=win)\n")

    print(f"[OK] Saved dataset info to: {info_file}")

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"\n[OK] Successfully created ML dataset")
    print(f"[OK] {len(df)} samples with {len(df.columns)-1} features")
    print(f"[OK] {df['target'].mean():.1%} win rate")
    print(f"[OK] No missing values" if missing.sum() == 0 else f"[ERROR] {missing.sum()} missing values")

    print("\nFeature Categories:")
    print(f"  VWAP features: {len([c for c in df.columns if 'vwap' in c])}")
    print(f"  Volume profile features: {len([c for c in df.columns if 'at_' in c or 'poc' in c or 'lvn' in c])}")
    print(f"  HTF features: {len([c for c in df.columns if 'htf' in c or 'prev_day' in c or 'prev_week' in c or 'weekly' in c or 'daily_hvn' in c])}")
    print(f"  Market/trend features: {len([c for c in df.columns if 'adx' in c or 'di' in c or 'trend' in c or 'regime' in c])}")
    print(f"  Temporal features: {len([c for c in df.columns if 'hour' in c or 'day' in c or 'week' in c or 'session' in c or 'overlap' in c])}")

    print("\n[OK] Ready for Day 6: Baseline Model Training")
    print("=" * 80)

    return df


if __name__ == '__main__':
    df = create_dataset()

    if df is not None:
        print("\n[OK] Day 5 Complete!")
    else:
        print("\n[ERROR] Day 5 Failed - check errors above")
        sys.exit(1)
