"""
Day 10: Model Loading Utility
Provides utilities for loading and managing ML models
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ModelLoader:
    """
    Utility class for loading and managing ML models.

    Supports:
    - Loading models by file path
    - Loading models by version from registry
    - Getting model metadata
    - Listing available models
    """

    def __init__(self, models_dir: str = 'ml_system/models'):
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / 'registry.json'
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from JSON file"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8', errors='ignore') as f:
                return json.load(f)
        return {'models': []}

    def _save_registry(self):
        """Save model registry to JSON file"""
        with open(self.registry_file, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(self.registry, f, indent=2)

    def load_model(self, model_path: str):
        """
        Load a model from pickle file.

        Args:
            model_path: Path to model pickle file

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        return model

    def load_model_by_version(self, version: str):
        """
        Load a model by version from registry.

        Args:
            version: Model version (e.g., '1.0', '1.1')

        Returns:
            Loaded model object

        Raises:
            ValueError: If version not found in registry
        """
        # Find model in registry
        for model_info in self.registry['models']:
            if model_info['version'] == version:
                model_file = self.models_dir / model_info['filename']
                return self.load_model(str(model_file))

        raise ValueError(f"Model version {version} not found in registry")

    def load_production_model(self):
        """
        Load the current production model.

        Returns:
            Loaded production model

        Raises:
            ValueError: If no production model found
        """
        # Find production model
        for model_info in self.registry['models']:
            if model_info.get('status') == 'production':
                model_file = self.models_dir / model_info['filename']
                return self.load_model(str(model_file))

        raise ValueError("No production model found in registry")

    def get_model_metadata(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific model version.

        Args:
            version: Model version

        Returns:
            Model metadata dictionary or None if not found
        """
        for model_info in self.registry['models']:
            if model_info['version'] == version:
                return model_info
        return None

    def list_models(self) -> list:
        """
        List all models in registry.

        Returns:
            List of model info dictionaries
        """
        return self.registry['models']

    def register_model(
        self,
        version: str,
        model_type: str,
        filename: str,
        validation_metrics: Dict[str, float],
        features: list,
        status: str = 'staging'
    ):
        """
        Register a new model in the registry.

        Args:
            version: Model version (e.g., '1.0')
            model_type: Type of model (e.g., 'RandomForest')
            filename: Model pickle filename
            validation_metrics: Dict of validation metrics
            features: List of feature names
            status: Model status ('staging', 'production', 'archived')
        """
        model_info = {
            'version': version,
            'type': model_type,
            'filename': filename,
            'trained_date': datetime.now().strftime('%Y-%m-%d'),
            'validation_metrics': validation_metrics,
            'num_features': len(features),
            'features': features,
            'status': status
        }

        # Add to registry
        self.registry['models'].append(model_info)
        self._save_registry()

    def promote_to_production(self, version: str):
        """
        Promote a model version to production.

        This will:
        1. Set all other models to 'archived' status
        2. Set the specified version to 'production' status

        Args:
            version: Model version to promote

        Raises:
            ValueError: If version not found
        """
        found = False

        for model_info in self.registry['models']:
            if model_info['version'] == version:
                model_info['status'] = 'production'
                found = True
            elif model_info.get('status') == 'production':
                model_info['status'] = 'archived'

        if not found:
            raise ValueError(f"Model version {version} not found")

        self._save_registry()

    def get_production_version(self) -> Optional[str]:
        """
        Get the current production model version.

        Returns:
            Production model version or None
        """
        for model_info in self.registry['models']:
            if model_info.get('status') == 'production':
                return model_info['version']
        return None


def load_latest_model(models_dir: str = 'ml_system/models'):
    """
    Convenience function to load the latest production model.

    Args:
        models_dir: Directory containing models

    Returns:
        Loaded model object
    """
    loader = ModelLoader(models_dir)
    return loader.load_production_model()
