#!/usr/bin/env python3
"""
A/B Testing Framework for ML vs Baseline
Compares ML model decisions with rule-based baseline decisions in shadow mode
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import sys
from pathlib import Path

# Add project root to path dynamically
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_system.features.extractor import FeatureExtractor
from ml_system.models.model_loader import ModelLoader

class ABTest:
    """
    A/B test ML predictions vs baseline (confluence threshold) decisions
    """

    def __init__(self, baseline_threshold=15):
        """
        Args:
            baseline_threshold: Minimum confluence score for baseline to take trade
        """
        self.baseline_threshold = baseline_threshold
        self.feature_extractor = FeatureExtractor()

        # Load ML model
        loader = ModelLoader()
        self.ml_model = loader.load_production_model()

        # Store results
        self.results = []

    def evaluate_trade(self, trade_record: Dict) -> Dict:
        """
        Evaluate a single trade with both ML and baseline

        Returns:
            Dict with decisions and outcome
        """
        # Get actual outcome
        outcome = trade_record.get('outcome', {})
        if not outcome or outcome.get('status') != 'closed':
            return None  # Skip open trades

        actual_profit = outcome.get('profit', 0)
        actual_win = 1 if actual_profit > 0 else 0

        # Baseline decision: Based on confluence score
        confluence_score = trade_record.get('confluence_score', 0)
        baseline_decision = 'TAKE' if confluence_score >= self.baseline_threshold else 'SKIP'

        # ML decision: Based on model prediction
        features = self.feature_extractor.extract_features(trade_record)
        X = pd.DataFrame([features])
        ml_prob = self.ml_model.predict_proba(X)[0, 1]
        ml_decision = 'TAKE' if ml_prob >= 0.70 else 'SKIP'

        return {
            'ticket': trade_record.get('ticket'),
            'entry_time': trade_record.get('entry_time'),
            'confluence_score': confluence_score,
            'actual_win': actual_win,
            'actual_profit': actual_profit,
            'baseline_decision': baseline_decision,
            'ml_decision': ml_decision,
            'ml_confidence': ml_prob,
            'baseline_correct': (baseline_decision == 'TAKE' and actual_win == 1) or (baseline_decision == 'SKIP' and actual_win == 0),
            'ml_correct': (ml_decision == 'TAKE' and actual_win == 1) or (ml_decision == 'SKIP' and actual_win == 0),
        }

    def run_test(self, trade_log_path='ml_system/outputs/continuous_trade_log.jsonl'):
        """Run A/B test on all closed trades"""

        print(f"Running A/B test...")
        print(f"Baseline threshold: {self.baseline_threshold}")
        print(f"ML threshold: 0.70")
        print()

        # Load trades
        with open(trade_log_path, 'r') as f:
            for line in f:
                trade = json.loads(line)
                result = self.evaluate_trade(trade)
                if result:
                    self.results.append(result)

        print(f"Evaluated {len(self.results)} closed trades")
        return self.results

    def analyze_results(self) -> Dict:
        """Analyze A/B test results"""

        if not self.results:
            print("No results to analyze")
            return {}

        df = pd.DataFrame(self.results)

        # Overall stats
        baseline_takes = (df['baseline_decision'] == 'TAKE').sum()
        ml_takes = (df['ml_decision'] == 'TAKE').sum()

        # Win rates
        baseline_wins = df[df['baseline_decision'] == 'TAKE']['actual_win'].sum()
        ml_wins = df[df['ml_decision'] == 'TAKE']['actual_win'].sum()

        baseline_win_rate = baseline_wins / baseline_takes if baseline_takes > 0 else 0
        ml_win_rate = ml_wins / ml_takes if ml_takes > 0 else 0

        # Profit
        baseline_profit = df[df['baseline_decision'] == 'TAKE']['actual_profit'].sum()
        ml_profit = df[df['ml_decision'] == 'TAKE']['actual_profit'].sum()

        # Agreement
        agreement = (df['baseline_decision'] == df['ml_decision']).sum()
        agreement_rate = agreement / len(df)

        results = {
            'total_trades': len(df),
            'baseline': {
                'takes': int(baseline_takes),
                'wins': int(baseline_wins),
                'win_rate': baseline_win_rate,
                'profit': baseline_profit,
                'avg_profit_per_trade': baseline_profit / baseline_takes if baseline_takes > 0 else 0
            },
            'ml': {
                'takes': int(ml_takes),
                'wins': int(ml_wins),
                'win_rate': ml_win_rate,
                'profit': ml_profit,
                'avg_profit_per_trade': ml_profit / ml_takes if ml_takes > 0 else 0
            },
            'comparison': {
                'agreement_rate': agreement_rate,
                'win_rate_improvement': ml_win_rate - baseline_win_rate,
                'profit_improvement': ml_profit - baseline_profit,
                'better_system': 'ML' if ml_win_rate > baseline_win_rate else 'Baseline'
            }
        }

        return results

    def print_report(self, results: Dict):
        """Print formatted A/B test report"""

        print()
        print("=" * 80)
        print("A/B TEST RESULTS: ML vs BASELINE")
        print("=" * 80)
        print()

        print(f"Total Trades Evaluated: {results['total_trades']}")
        print()

        print("BASELINE (Rule-Based, Confluence >= 15):")
        print(f"  Trades Taken: {results['baseline']['takes']}/{results['total_trades']}")
        print(f"  Wins: {results['baseline']['wins']}")
        print(f"  Win Rate: {results['baseline']['win_rate']*100:.1f}%")
        print(f"  Total Profit: ${results['baseline']['profit']:.2f}")
        print(f"  Avg Profit/Trade: ${results['baseline']['avg_profit_per_trade']:.2f}")
        print()

        print("ML MODEL (Ensemble, Confidence >= 70%):")
        print(f"  Trades Taken: {results['ml']['takes']}/{results['total_trades']}")
        print(f"  Wins: {results['ml']['wins']}")
        print(f"  Win Rate: {results['ml']['win_rate']*100:.1f}%")
        print(f"  Total Profit: ${results['ml']['profit']:.2f}")
        print(f"  Avg Profit/Trade: ${results['ml']['avg_profit_per_trade']:.2f}")
        print()

        print("COMPARISON:")
        print(f"  Agreement Rate: {results['comparison']['agreement_rate']*100:.1f}%")
        print(f"  Win Rate Improvement: {results['comparison']['win_rate_improvement']*100:+.1f}%")
        print(f"  Profit Improvement: ${results['comparison']['profit_improvement']:+.2f}")
        print(f"  Better System: {results['comparison']['better_system']}")
        print()

        # Statistical significance note
        print(" Statistical Significance:")
        if results['total_trades'] < 30:
            print("  [WARN]  Sample size too small (<30) for statistical significance")
            print("  -> Continue collecting data")
        else:
            print("  [OK] Sample size adequate for initial analysis")

        print()
        print("=" * 80)

    def save_results(self, results: Dict, output_path='ml_system/outputs/ab_test_results.json'):
        """Save A/B test results to file"""

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Also save detailed results
        df = pd.DataFrame(self.results)
        df.to_csv('ml_system/outputs/ab_test_detailed.csv', index=False)

        print(f"[OK] Results saved to {output_path}")
        print(f"[OK] Detailed results saved to ml_system/outputs/ab_test_detailed.csv")

def main():
    """Run A/B test"""
    ab_test = ABTest(baseline_threshold=15)
    ab_test.run_test()
    results = ab_test.analyze_results()
    ab_test.print_report(results)
    ab_test.save_results(results)

if __name__ == '__main__':
    main()
