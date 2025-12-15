import joblib
import pandas as pd
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from app.services.risk_classifier import classify_risk_level, RISK_LEVELS

logger = logging.getLogger(__name__)

# Lazy load model
_model = None
_model_path = os.path.join('models', 'flood_rf_model.joblib')
_model_metadata = None
_model_checksum = None  # Store checksum for integrity verification


def get_model_metadata(model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get metadata for a model."""
    if model_path is None:
        model_path = _model_path
    
    metadata_path = Path(model_path).with_suffix('.json')
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {str(e)}")
    return None


def compute_model_checksum(model_path: str) -> str:
    """
    Compute SHA-256 checksum of a model file for integrity verification.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        str: Hex-encoded SHA-256 checksum
    """
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_model_integrity(model_path: str, expected_checksum: Optional[str] = None) -> bool:
    """
    Verify model file integrity using checksum.
    
    Args:
        model_path: Path to the model file
        expected_checksum: Expected SHA-256 checksum (from metadata)
        
    Returns:
        bool: True if checksum matches or no expected checksum provided
    """
    if not expected_checksum:
        # Try to get from metadata
        metadata = get_model_metadata(model_path)
        if metadata:
            expected_checksum = metadata.get('checksum')
    
    if not expected_checksum:
        logger.warning(f"No checksum available for model: {model_path}")
        return True  # Allow if no checksum to verify against
    
    actual_checksum = compute_model_checksum(model_path)
    if actual_checksum != expected_checksum:
        logger.error(
            f"Model integrity check FAILED for {model_path}! "
            f"Expected: {expected_checksum}, Got: {actual_checksum}"
        )
        return False
    
    logger.info(f"Model integrity verified: {model_path}")
    return True


def save_model_with_checksum(model, model_path: str, metadata: Dict[str, Any]) -> str:
    """
    Save a model with checksum in metadata for integrity verification.
    
    Args:
        model: The model object to save
        model_path: Path to save the model
        metadata: Metadata dictionary
        
    Returns:
        str: The computed checksum
    """
    # Save model first
    joblib.dump(model, model_path)
    
    # Compute checksum
    checksum = compute_model_checksum(model_path)
    
    # Update metadata with checksum
    metadata['checksum'] = checksum
    metadata['checksum_algorithm'] = 'SHA-256'
    
    # Save metadata
    metadata_path = Path(model_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved with checksum: {checksum[:16]}...")
    return checksum


def list_available_models(models_dir: str = 'models') -> list:
    """List all available model versions."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    models = []
    for file in models_path.glob('flood_rf_model_v*.joblib'):
        try:
            version_str = file.stem.split('_v')[-1]
            version = int(version_str)
            metadata = get_model_metadata(str(file))
            models.append({
                'version': version,
                'path': str(file),
                'metadata': metadata
            })
        except (ValueError, IndexError):
            continue
    
    # Sort by version
    models.sort(key=lambda x: x['version'], reverse=True)
    return models


def get_latest_model_version(models_dir: str = 'models') -> Optional[int]:
    """Get the latest model version number."""
    models = list_available_models(models_dir)
    if models:
        return models[0]['version']
    
    # Check if latest model exists
    latest_path = Path(models_dir) / 'flood_rf_model.joblib'
    if latest_path.exists():
        metadata = get_model_metadata(str(latest_path))
        if metadata and 'version' in metadata:
            return metadata['version']
    
    return None


def _load_model(model_path: Optional[str] = None, force_reload: bool = False, verify_integrity: bool = True):
    """
    Load the trained model with optional integrity verification.
    
    Args:
        model_path: Path to the model file
        force_reload: Force reload even if already loaded
        verify_integrity: Verify model checksum before loading
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If integrity check fails
    """
    global _model, _model_path, _model_metadata, _model_checksum
    
    # Use provided path or default
    if model_path is None:
        model_path = _model_path
    else:
        _model_path = model_path
    
    # Reload if path changed or force reload
    if _model is None or force_reload or model_path != _model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please train the model first using train.py"
            )
        
        # Verify integrity if enabled
        if verify_integrity and os.getenv('VERIFY_MODEL_INTEGRITY', 'True').lower() == 'true':
            metadata = get_model_metadata(model_path)
            expected_checksum = metadata.get('checksum') if metadata else None
            
            if expected_checksum:
                if not verify_model_integrity(model_path, expected_checksum):
                    raise ValueError(
                        f"Model integrity verification failed for {model_path}. "
                        "The model file may have been tampered with."
                    )
            else:
                logger.warning(
                    f"No checksum in metadata for {model_path}. "
                    "Consider re-training with checksum enabled."
                )
        
        try:
            _model = joblib.load(model_path)
            _model_metadata = get_model_metadata(model_path)
            _model_checksum = compute_model_checksum(model_path)
            
            logger.info(f"Model loaded successfully from {model_path}")
            if _model_metadata:
                logger.info(f"Model version: {_model_metadata.get('version', 'unknown')}")
            logger.info(f"Model checksum: {_model_checksum[:16]}...")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    return _model


def load_model_version(version: int, models_dir: str = 'models', force_reload: bool = False):
    """Load a specific model version."""
    model_path = Path(models_dir) / f'flood_rf_model_v{version}.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model version {version} not found at {model_path}")
    
    return _load_model(str(model_path), force_reload=force_reload)

def predict_flood(input_data, model_version: Optional[int] = None, return_proba: bool = False, return_risk_level: bool = True):
    """
    Predict flood based on input data.
    
    Args:
        input_data: dict with keys 'temperature', 'humidity', 'precipitation', etc.
        model_version: Optional model version number (uses latest if None)
        return_proba: If True, also return prediction probabilities
        return_risk_level: If True, return 3-level risk classification (Safe/Alert/Critical)
    
    Returns:
        int or dict: 0 (no flood) or 1 (flood predicted), or dict with prediction, probabilities, and risk level
    """
    if not isinstance(input_data, dict):
        raise ValueError("input_data must be a dictionary")
    
    # Validate required fields
    required_fields = ['temperature', 'humidity', 'precipitation']
    missing_fields = [field for field in required_fields if field not in input_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Load model (specific version or latest)
    if model_version is not None:
        model = load_model_version(model_version, force_reload=True)
    else:
        model = _load_model()
    
    try:
        # Convert input_data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Get model feature names if available
        if hasattr(model, 'feature_names_in_'):
            # Ensure columns match model expectations
            expected_features = model.feature_names_in_
            # Use reasonable defaults for missing features
            fill_values = {}
            for feature in expected_features:
                if feature not in input_data:
                    # Provide reasonable defaults based on feature name
                    if 'wind' in feature.lower():
                        fill_values[feature] = 10.0  # Default wind speed
                    elif 'temperature' in feature.lower():
                        fill_values[feature] = 298.15  # Default temperature (25°C)
                    elif 'humidity' in feature.lower():
                        fill_values[feature] = 50.0  # Default humidity
                    elif 'precipitation' in feature.lower():
                        fill_values[feature] = 0.0  # Default precipitation
                    else:
                        fill_values[feature] = 0.0  # Default for other features
            
            # Reindex with custom fill values
            df = df.reindex(columns=expected_features)
            # Apply custom fill values for missing features
            for feature, value in fill_values.items():
                df[feature] = df[feature].fillna(value)
        
        # Make prediction
        prediction = model.predict(df)
        result = int(prediction[0])
        
        # Get probabilities if requested or if risk level classification is needed
        probability_dict = None
        if (return_proba or return_risk_level) and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            probability_dict = {
                'no_flood': float(proba[0]),
                'flood': float(proba[1]) if len(proba) > 1 else 0.0
            }
        
        # Classify risk level if requested
        risk_classification = None
        if return_risk_level:
            risk_classification = classify_risk_level(
                prediction=result,
                probability=probability_dict,
                precipitation=input_data.get('precipitation'),
                humidity=input_data.get('humidity')
            )
        
        # Build response
        if return_proba or return_risk_level:
            response = {
                'prediction': result,
                'model_version': _model_metadata.get('version') if _model_metadata else None
            }
            if probability_dict:
                response['probability'] = probability_dict
            if risk_classification:
                response['risk_level'] = risk_classification['risk_level']
                response['risk_label'] = risk_classification['risk_label']
                response['risk_color'] = risk_classification['risk_color']
                response['risk_description'] = risk_classification['description']
                response['confidence'] = risk_classification['confidence']
            
            logger.info(f"Prediction: {result}, Risk Level: {risk_classification['risk_label'] if risk_classification else 'N/A'}")
            return response
        else:
            logger.info(f"Prediction made: {result} (0=no flood, 1=flood)")
            return result
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise


def get_current_model_info() -> Optional[Dict[str, Any]]:
    """Get information about the currently loaded model."""
    if _model is None:
        try:
            _load_model()
        except:
            return None
    
    info = {
        'model_path': _model_path,
        'model_type': type(_model).__name__ if _model else None,
        'metadata': _model_metadata,
        'checksum': _model_checksum[:16] + '...' if _model_checksum else None,
        'integrity_verified': bool(_model_checksum)
    }
    
    if _model and hasattr(_model, 'feature_names_in_'):
        info['features'] = list(_model.feature_names_in_)
    
    return info
