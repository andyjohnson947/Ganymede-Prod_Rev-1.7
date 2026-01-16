"""
Day 13: Real-time ML Prediction Service
Provides fast, low-latency predictions for live trading
"""

import time
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime

# Import our modules
from ml_system.features.extractor import FeatureExtractor
from ml_system.models.model_loader import load_latest_model


class MLPredictor:
    """
    Real-time ML prediction service for trading decisions.

    Features:
    - Fast prediction (<100ms target)
    - Error handling and fallback logic
    - Prediction logging
    - Model hot-swapping support
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.6):
        """
        Initialize ML Predictor.

        Args:
            model_path: Optional path to model file. If None, loads production model.
            confidence_threshold: Minimum probability to take trade (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = FeatureExtractor()

        # Performance tracking
        self.prediction_count = 0
        self.total_latency = 0.0
        self.errors = 0

        # Load model
        self._load_model(model_path)

        # Prediction log
        self.log_file = Path('ml_system/outputs/predictions.jsonl')

    def _load_model(self, model_path: str = None):
        """Load ML model"""
        start_time = time.time()

        if model_path:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[OK] Loaded model from: {model_path}")
        else:
            self.model = load_latest_model()
            print("[OK] Loaded production model")

        load_time = (time.time() - start_time) * 1000
        print(f"  Load time: {load_time:.1f}ms")

    def predict(self, trade_record: Dict[str, Any], log_prediction: bool = True) -> Dict[str, Any]:
        """
        Make real-time prediction for a trade.

        Args:
            trade_record: Trade data dictionary
            log_prediction: Whether to log this prediction

        Returns:
            Dict with keys:
                - decision: 'TAKE' or 'SKIP'
                - confidence: Probability of win (0-1)
                - latency_ms: Prediction latency in milliseconds
                - timestamp: Prediction timestamp
        """
        start_time = time.time()

        try:
            # Extract features
            features = self.feature_extractor.extract_features(trade_record)

            # Convert to DataFrame
            X = pd.DataFrame([features])

            # Get prediction
            prob = self.model.predict_proba(X)[0, 1]

            # Make decision
            decision = 'TAKE' if prob >= self.confidence_threshold else 'SKIP'

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Update stats
            self.prediction_count += 1
            self.total_latency += latency_ms

            # Build result
            result = {
                'decision': decision,
                'confidence': float(prob),
                'latency_ms': float(latency_ms),
                'timestamp': datetime.now().isoformat()
            }

            # Log prediction
            if log_prediction:
                self._log_prediction(trade_record, result)

            return result

        except Exception as e:
            self.errors += 1
            print(f"[ERROR] Prediction error: {e}")

            # Fallback decision (based on confluence)
            confluence_score = trade_record.get('confluence_score', 0)
            decision = 'TAKE' if confluence_score >= 5 else 'SKIP'

            return {
                'decision': decision,
                'confidence': 0.5,  # Neutral confidence for fallback
                'latency_ms': 0.0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def predict_batch(self, trade_records: list) -> list:
        """
        Make predictions for multiple trades.

        Args:
            trade_records: List of trade data dictionaries

        Returns:
            List of prediction results
        """
        start_time = time.time()

        results = []
        for trade in trade_records:
            result = self.predict(trade, log_prediction=False)
            results.append(result)

        batch_time = (time.time() - start_time) * 1000
        avg_latency = batch_time / len(trade_records) if trade_records else 0

        print(f"[OK] Batch prediction: {len(trade_records)} trades in {batch_time:.1f}ms")
        print(f"  Average latency: {avg_latency:.1f}ms per trade")

        return results

    def _log_prediction(self, trade_record: Dict[str, Any], result: Dict[str, Any]):
        """Log prediction to file"""
        try:
            log_entry = {
                'timestamp': result['timestamp'],
                'ticket': trade_record.get('ticket'),
                'symbol': trade_record.get('symbol'),
                'confluence_score': trade_record.get('confluence_score'),
                'decision': result['decision'],
                'confidence': result['confidence'],
                'latency_ms': result['latency_ms']
            }

            with open(self.log_file, 'a', encoding='utf-8', errors='ignore') as f:
                import json
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            print(f"[WARN] Error logging prediction: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get predictor performance statistics.

        Returns:
            Dict with performance metrics
        """
        avg_latency = self.total_latency / self.prediction_count if self.prediction_count > 0 else 0

        return {
            'predictions_made': self.prediction_count,
            'average_latency_ms': avg_latency,
            'errors': self.errors,
            'error_rate': self.errors / self.prediction_count if self.prediction_count > 0 else 0,
            'model_type': type(self.model).__name__,
            'confidence_threshold': self.confidence_threshold
        }

    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        self.confidence_threshold = threshold
        print(f"[OK] Confidence threshold updated to: {threshold:.1%}")

    def reload_model(self, model_path: str = None):
        """
        Hot-swap the ML model (for live updates).

        Args:
            model_path: Path to new model, or None for latest production model
        """
        print("🔄 Reloading model...")
        self._load_model(model_path)
        print("[OK] Model reloaded successfully")


# Convenience function for quick predictions
def predict_trade(trade_record: Dict[str, Any], confidence_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Quick prediction function.

    Args:
        trade_record: Trade data dictionary
        confidence_threshold: Minimum probability to take trade

    Returns:
        Prediction result dict
    """
    predictor = MLPredictor(confidence_threshold=confidence_threshold)
    return predictor.predict(trade_record)
