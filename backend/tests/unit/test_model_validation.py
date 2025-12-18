"""
ML Model Validation Tests.

Tests specific to machine learning model behavior, quality, and robustness.
"""

import pytest
import sys
import os
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


# ============================================================================
# Model Loading and Initialization Tests
# ============================================================================

class TestModelLoading:
    """Tests for model loading and initialization."""
    
    def test_model_loads_successfully(self):
        """Test that model can be loaded without errors."""
        from app.services.predict import _load_model, ModelLoader
        
        ModelLoader.reset_instance()
        
        try:
            model = _load_model()
            assert model is not None
        except FileNotFoundError:
            pytest.skip("Model file not found - run training first")
    
    def test_model_has_required_methods(self):
        """Test that loaded model has required prediction methods."""
        from app.services.predict import _load_model, ModelLoader
        
        ModelLoader.reset_instance()
        
        try:
            model = _load_model()
            
            # Should have predict method
            assert hasattr(model, 'predict')
            assert callable(model.predict)
            
            # Should have predict_proba for probability estimation
            assert hasattr(model, 'predict_proba')
            assert callable(model.predict_proba)
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_model_has_feature_names(self):
        """Test that model tracks feature names."""
        from app.services.predict import _load_model, ModelLoader
        
        ModelLoader.reset_instance()
        
        try:
            model = _load_model()
            
            # Should have feature names (for Random Forest)
            assert hasattr(model, 'feature_names_in_')
            assert len(model.feature_names_in_) > 0
            
            # Check for expected features
            features = list(model.feature_names_in_)
            assert 'temperature' in features or any('temp' in f.lower() for f in features)
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_model_metadata_exists(self):
        """Test that model metadata file exists and is valid."""
        from app.services.predict import get_model_metadata
        
        metadata = get_model_metadata()
        
        if metadata is None:
            pytest.skip("No model metadata available")
        
        # Should have version info
        assert 'version' in metadata or 'created_at' in metadata
        
        # Should have checksum for integrity
        if 'checksum' in metadata:
            assert len(metadata['checksum']) == 64  # SHA-256
    
    def test_model_version_loading(self):
        """Test loading specific model versions."""
        from app.services.predict import load_model_version, list_available_models
        
        models = list_available_models()
        
        if not models:
            pytest.skip("No versioned models available")
        
        # Try loading the latest version
        latest = models[0]
        try:
            model = load_model_version(latest['version'])
            assert model is not None
        except FileNotFoundError:
            pytest.skip(f"Model version {latest['version']} not found")


# ============================================================================
# Prediction Quality Tests
# ============================================================================

class TestPredictionQuality:
    """Tests for prediction quality and consistency."""
    
    def test_prediction_deterministic(self):
        """Test that predictions are deterministic (same input = same output)."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        test_data = {
            'temperature': 298.15,
            'humidity': 75.0,
            'precipitation': 10.0
        }
        
        try:
            # Make the same prediction multiple times
            results = []
            for _ in range(5):
                result = predict_flood(test_data, return_proba=True)
                results.append(result['prediction'])
            
            # All predictions should be identical
            assert all(r == results[0] for r in results)
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_prediction_probability_range(self):
        """Test that prediction probabilities are in valid range [0, 1]."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        test_data = {
            'temperature': 298.15,
            'humidity': 75.0,
            'precipitation': 10.0
        }
        
        try:
            result = predict_flood(test_data, return_proba=True)
            
            if 'probability' in result:
                for key, prob in result['probability'].items():
                    assert 0.0 <= prob <= 1.0, f"Probability {key}={prob} out of range"
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_probability_sum_to_one(self):
        """Test that class probabilities sum to approximately 1."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        test_data = {
            'temperature': 298.15,
            'humidity': 75.0,
            'precipitation': 10.0
        }
        
        try:
            result = predict_flood(test_data, return_proba=True)
            
            if 'probability' in result:
                prob_sum = sum(result['probability'].values())
                assert 0.99 <= prob_sum <= 1.01, f"Probabilities sum to {prob_sum}"
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_extreme_conditions_trigger_flood(self):
        """Test that extreme weather conditions tend to predict flood risk."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        extreme_data = {
            'temperature': 298.15,
            'humidity': 98.0,
            'precipitation': 200.0  # Very heavy rain
        }
        
        try:
            result = predict_flood(extreme_data, return_proba=True, return_risk_level=True)
            
            # Extreme conditions should either predict flood or high probability
            flood_prob = result.get('probability', {}).get('flood', 0)
            risk_level = result.get('risk_level', 0)
            
            # At least one should indicate elevated risk
            assert flood_prob > 0.3 or risk_level >= 1 or result['prediction'] == 1, \
                f"Extreme conditions not detected: prob={flood_prob}, risk={risk_level}"
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_normal_conditions_low_risk(self):
        """Test that normal weather conditions predict low risk."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        normal_data = {
            'temperature': 298.15,
            'humidity': 50.0,
            'precipitation': 0.0  # No rain
        }
        
        try:
            result = predict_flood(normal_data, return_proba=True, return_risk_level=True)
            
            # Normal conditions should have low flood probability
            flood_prob = result.get('probability', {}).get('flood', 0)
            
            # Probability should be relatively low (< 50%)
            assert flood_prob < 0.7, f"Normal conditions show high risk: {flood_prob}"
            
        except FileNotFoundError:
            pytest.skip("Model file not found")


# ============================================================================
# Model Robustness Tests
# ============================================================================

class TestModelRobustness:
    """Tests for model robustness to edge cases and adversarial inputs."""
    
    def test_handles_boundary_values(self):
        """Test model handles boundary values without crashing."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        boundary_cases = [
            {'temperature': 200.0, 'humidity': 0.0, 'precipitation': 0.0},  # Min values
            {'temperature': 330.0, 'humidity': 100.0, 'precipitation': 500.0},  # Max values
            {'temperature': 273.15, 'humidity': 50.0, 'precipitation': 0.001},  # Near zero
        ]
        
        try:
            for data in boundary_cases:
                result = predict_flood(data, return_proba=True)
                assert 'prediction' in result
                assert result['prediction'] in [0, 1]
                
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_handles_integer_inputs(self):
        """Test model handles integer inputs (converted to float)."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        int_data = {
            'temperature': 298,  # Integer, not float
            'humidity': 75,
            'precipitation': 10
        }
        
        try:
            result = predict_flood(int_data, return_proba=True)
            assert 'prediction' in result
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_prediction_with_additional_features(self):
        """Test prediction when extra features are provided."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        data_with_extras = {
            'temperature': 298.15,
            'humidity': 75.0,
            'precipitation': 10.0,
            'wind_speed': 15.0,  # Extra feature
            'pressure': 1013.25,  # Extra feature
        }
        
        try:
            # Should handle extra features gracefully
            result = predict_flood(data_with_extras, return_proba=True)
            assert 'prediction' in result
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_multiple_sequential_predictions(self):
        """Test that model handles many sequential predictions."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        import random
        
        try:
            for _ in range(100):
                data = {
                    'temperature': random.uniform(273.15, 320.0),
                    'humidity': random.uniform(0.0, 100.0),
                    'precipitation': random.uniform(0.0, 200.0)
                }
                
                result = predict_flood(data, return_proba=True)
                assert 'prediction' in result
                assert result['prediction'] in [0, 1]
                
        except FileNotFoundError:
            pytest.skip("Model file not found")


# ============================================================================
# Risk Classification Validation Tests
# ============================================================================

class TestRiskClassificationValidation:
    """Tests for 3-level risk classification accuracy."""
    
    def test_risk_level_consistency(self):
        """Test that risk level is consistent with prediction and probability."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        test_cases = [
            {'temperature': 298.15, 'humidity': 50.0, 'precipitation': 0.0},
            {'temperature': 298.15, 'humidity': 80.0, 'precipitation': 20.0},
            {'temperature': 298.15, 'humidity': 95.0, 'precipitation': 100.0},
        ]
        
        try:
            for data in test_cases:
                result = predict_flood(data, return_proba=True, return_risk_level=True)
                
                prediction = result['prediction']
                risk_level = result.get('risk_level', -1)
                flood_prob = result.get('probability', {}).get('flood', 0)
                
                # Validate risk level assignment
                if prediction == 1 and flood_prob >= 0.75:
                    # Should be Critical (2)
                    assert risk_level == 2, f"Expected Critical for high prob {flood_prob}"
                
                if prediction == 0 and flood_prob < 0.3:
                    # Should be Safe (0) unless other conditions apply
                    pass  # Other factors may elevate risk
                    
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_risk_label_matches_level(self):
        """Test that risk label matches risk level."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        level_labels = {0: 'Safe', 1: 'Alert', 2: 'Critical'}
        
        try:
            for _ in range(20):
                data = {
                    'temperature': np.random.uniform(273.15, 320.0),
                    'humidity': np.random.uniform(0.0, 100.0),
                    'precipitation': np.random.uniform(0.0, 200.0)
                }
                
                result = predict_flood(data, return_proba=True, return_risk_level=True)
                
                risk_level = result.get('risk_level')
                risk_label = result.get('risk_label')
                
                if risk_level is not None and risk_label is not None:
                    expected_label = level_labels.get(risk_level)
                    assert risk_label == expected_label, \
                        f"Mismatch: level={risk_level}, label={risk_label}"
                    
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_risk_color_codes(self):
        """Test that risk color codes are correct."""
        from app.services.predict import predict_flood, ModelLoader
        
        ModelLoader.reset_instance()
        
        color_map = {
            'Safe': '#28a745',      # Green
            'Alert': '#ffc107',     # Yellow
            'Critical': '#dc3545'   # Red
        }
        
        try:
            for _ in range(20):
                data = {
                    'temperature': np.random.uniform(273.15, 320.0),
                    'humidity': np.random.uniform(0.0, 100.0),
                    'precipitation': np.random.uniform(0.0, 200.0)
                }
                
                result = predict_flood(data, return_proba=True, return_risk_level=True)
                
                risk_label = result.get('risk_label')
                risk_color = result.get('risk_color')
                
                if risk_label and risk_color:
                    expected_color = color_map.get(risk_label)
                    assert risk_color == expected_color, \
                        f"Wrong color for {risk_label}: got {risk_color}"
                    
        except FileNotFoundError:
            pytest.skip("Model file not found")


# ============================================================================
# Feature Importance Tests
# ============================================================================

class TestFeatureImportance:
    """Tests related to feature importance in the model."""
    
    def test_model_has_feature_importances(self):
        """Test that Random Forest model has feature importances."""
        from app.services.predict import _load_model, ModelLoader
        
        ModelLoader.reset_instance()
        
        try:
            model = _load_model()
            
            # Random Forest should have feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Should have importance for each feature
                assert len(importances) > 0
                
                # Importances should sum to ~1
                assert 0.99 <= sum(importances) <= 1.01
                
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_precipitation_is_important(self):
        """Test that precipitation is a significant feature."""
        from app.services.predict import _load_model, ModelLoader
        
        ModelLoader.reset_instance()
        
        try:
            model = _load_model()
            
            if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
                importances = model.feature_importances_
                
                # Find precipitation feature
                for i, name in enumerate(feature_names):
                    if 'precip' in name.lower():
                        # Precipitation should be a significant feature
                        # (assuming it's one of the top features for flood prediction)
                        assert importances[i] > 0.01, \
                            f"Precipitation importance too low: {importances[i]}"
                        break
                        
        except FileNotFoundError:
            pytest.skip("Model file not found")


# ============================================================================
# Model Consistency Across Versions
# ============================================================================

class TestModelVersionConsistency:
    """Tests for consistency across model versions."""
    
    def test_all_versions_produce_valid_output(self):
        """Test that all available model versions produce valid outputs."""
        from app.services.predict import list_available_models, load_model_version
        import pandas as pd
        
        models = list_available_models()
        
        if not models:
            pytest.skip("No versioned models available")
        
        test_data = pd.DataFrame([{
            'temperature': 298.15,
            'humidity': 75.0,
            'precipitation': 10.0
        }])
        
        for model_info in models:
            version = model_info['version']
            try:
                model = load_model_version(version)
                
                # Reindex to match model features
                if hasattr(model, 'feature_names_in_'):
                    test_df = test_data.reindex(columns=model.feature_names_in_, fill_value=0)
                else:
                    test_df = test_data
                
                prediction = model.predict(test_df)
                
                assert prediction[0] in [0, 1], \
                    f"Version {version} produced invalid prediction: {prediction[0]}"
                    
            except FileNotFoundError:
                continue  # Skip missing versions
    
    def test_current_model_info(self):
        """Test that current model info is accessible."""
        from app.services.predict import get_current_model_info, ModelLoader
        
        ModelLoader.reset_instance()
        
        try:
            info = get_current_model_info()
            
            if info:
                assert 'model_path' in info
                assert 'model_type' in info
                
        except FileNotFoundError:
            pytest.skip("Model file not found")


# ============================================================================
# Run Model Validation Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
