# Model Versioning & A/B Testing Guide

This document describes the enhanced model versioning system with semantic versioning, A/B testing capabilities, and automated rollback features.

## Table of Contents

1. [Overview](#overview)
2. [Semantic Versioning](#semantic-versioning)
3. [A/B Testing](#ab-testing)
4. [Performance Monitoring](#performance-monitoring)
5. [Automated Rollback](#automated-rollback)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)

## Overview

The model versioning system provides enterprise-grade capabilities for managing ML models in production:

- **Semantic Versioning**: MAJOR.MINOR.PATCH versioning with prerelease and build metadata support
- **A/B Testing**: Compare model performance with multiple traffic splitting strategies
- **Performance Monitoring**: Track accuracy, latency, error rates, and confidence
- **Automated Rollback**: Automatic rollback when performance degrades below thresholds

## Semantic Versioning

### Version Format

Models use semantic versioning (SemVer) format: `MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`

- **MAJOR**: Breaking changes, significant retraining, incompatible feature changes
- **MINOR**: New capabilities, significant accuracy improvements
- **PATCH**: Bug fixes, minor tuning, metadata updates

Examples:
- `1.0.0` - Initial release
- `1.1.0` - New feature added
- `1.1.1` - Bug fix
- `2.0.0-beta` - Major version beta release
- `1.2.3+build456` - With build metadata

### Version Bumping

```python
from app.services.model_versioning import get_version_manager, VersionBumpType

manager = get_version_manager()

# Get next version suggestions
next_major = manager.get_next_version(VersionBumpType.MAJOR)  # 2.0.0
next_minor = manager.get_next_version(VersionBumpType.MINOR)  # 1.1.0
next_patch = manager.get_next_version(VersionBumpType.PATCH)  # 1.0.1
```

### Registering New Versions

```python
from app.services.model_versioning import (
    get_version_manager, 
    SemanticVersion
)

manager = get_version_manager()

# Register a new model version
manager.register_version(
    model_path='models/flood_rf_model_v2.joblib',
    version=SemanticVersion(2, 0, 0),
    metadata={
        'algorithm': 'RandomForest',
        'accuracy': 0.92,
        'training_date': '2026-01-12'
    }
)
```

### Promoting Versions

```python
# Promote a version to production
success = manager.promote_version(
    version=SemanticVersion(2, 0, 0),
    backup_current=True  # Keep previous version for rollback
)
```

## A/B Testing

### Traffic Splitting Strategies

| Strategy | Description |
|----------|-------------|
| `RANDOM` | Random assignment based on variant weights |
| `ROUND_ROBIN` | Sequential alternating assignment |
| `STICKY` | Same user always gets same variant |
| `CANARY` | Gradually increase new variant traffic |

### Creating an A/B Test

```python
from app.services.model_versioning import (
    get_version_manager,
    SemanticVersion,
    TrafficSplitStrategy
)

manager = get_version_manager()

# Create test comparing v1 vs v2
test = manager.create_ab_test(
    test_id='model_comparison_2026q1',
    name='Q1 Model Comparison',
    description='Comparing v2.0.0 against v1.0.0 baseline',
    control_version=SemanticVersion(1, 0, 0),
    treatment_version=SemanticVersion(2, 0, 0),
    traffic_split=0.2,  # 20% to treatment
    strategy=TrafficSplitStrategy.CANARY,
    target_sample_size=5000
)

# Start the test
manager.start_ab_test('model_comparison_2026q1')
```

### Using A/B Test for Predictions

```python
# Get variant for prediction
model, variant = manager.get_ab_test_variant(
    test_id='model_comparison_2026q1',
    user_id='user_123'  # Optional for sticky assignments
)

# Make prediction with selected variant
result = model.predict(input_data)

# Record the prediction
manager.record_ab_prediction(
    test_id='model_comparison_2026q1',
    variant_name=variant.name,
    latency_ms=45.2,
    confidence=0.87,
    risk_level='medium'
)
```

### Using the Decorator

```python
from app.services.model_versioning import ab_test_prediction

@ab_test_prediction('model_comparison_2026q1')
def make_prediction(model, input_data):
    return model.predict(input_data)

# Prediction is automatically routed to A/B test variant
result = make_prediction(model, data, user_id='user_123')

# Result includes A/B test metadata
# result['_ab_test'] = {'test_id': '...', 'variant': 'treatment', 'version': '2.0.0'}
```

### Recording Feedback

```python
# Record whether prediction was correct (for accuracy tracking)
manager.record_ab_feedback(
    test_id='model_comparison_2026q1',
    variant_name='treatment',
    was_correct=True
)
```

### Concluding Tests

```python
# Conclude and optionally promote winner
results = manager.conclude_ab_test(
    test_id='model_comparison_2026q1',
    promote_winner=True
)

print(f"Winner: {results['winner']}")
print(f"Statistical significance: {results['statistical_significance']}")
```

## Performance Monitoring

### Setting Thresholds

```python
from app.services.model_versioning import (
    get_version_manager,
    PerformanceThresholds
)

manager = get_version_manager()

# Configure rollback thresholds
thresholds = PerformanceThresholds(
    min_accuracy=0.80,          # Minimum 80% accuracy
    max_error_rate=0.05,        # Maximum 5% error rate
    max_latency_ms=200.0,       # Maximum 200ms latency
    min_confidence=0.60,        # Minimum 60% confidence
    evaluation_window=100,      # Evaluate every 100 predictions
    consecutive_failures=3      # Rollback after 3 consecutive failures
)

manager.set_performance_thresholds(thresholds)
```

### Recording Predictions

```python
# Record each prediction for monitoring
rollback_event = manager.record_prediction(
    latency_ms=45.0,
    confidence=0.92,
    risk_level='low',
    error=False
)

# If rollback was triggered, returns RollbackEvent
if rollback_event:
    print(f"Rolled back: {rollback_event.reason}")
```

### Getting Performance Data

```python
# Current performance
perf = manager.get_current_performance()
print(f"Current version: {perf['version']}")
print(f"Predictions: {perf['metrics']['predictions_count']}")
print(f"Average latency: {perf['metrics']['average_latency_ms']}ms")

# Historical performance
history = manager.get_performance_history()
for snapshot in history:
    print(f"{snapshot['version']}: {snapshot['accuracy']}")
```

## Automated Rollback

### Rollback Triggers

Automatic rollback occurs when:

1. **Accuracy Degradation**: Accuracy falls below `min_accuracy` threshold
2. **High Error Rate**: Error rate exceeds `max_error_rate` threshold
3. **High Latency**: Average latency exceeds `max_latency_ms` threshold
4. **Low Confidence**: Average confidence below `min_confidence` threshold
5. **Consecutive Failures**: Number of consecutive errors exceeds `consecutive_failures`

### Manual Rollback

```python
from app.services.model_versioning import SemanticVersion

# Manual rollback to specific version
event = manager.manual_rollback(
    to_version=SemanticVersion(1, 0, 0),
    details="Rolling back due to customer complaints"
)

print(f"Rolled back from {event.from_version} to {event.to_version}")
```

### Rollback Callbacks

```python
def on_rollback(event):
    # Send alert
    send_slack_alert(
        f"Model rollback: {event.from_version} -> {event.to_version}"
        f"\nReason: {event.reason.value}"
        f"\nDetails: {event.details}"
    )

manager.register_rollback_callback(on_rollback)
```

### Rollback History

```python
history = manager.get_rollback_history()
for event in history:
    print(f"{event['timestamp']}: {event['from_version']} -> {event['to_version']}")
    print(f"  Reason: {event['reason']}")
    print(f"  Automatic: {event['automatic']}")
```

## API Reference

### REST Endpoints

#### Version Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/versioning/api/models/versions` | List all versions |
| GET | `/api/v1/versioning/api/models/versions/current` | Get current version |
| GET | `/api/v1/versioning/api/models/versions/next` | Get next version numbers |
| POST | `/api/v1/versioning/api/models/versions/{version}/promote` | Promote version |

#### A/B Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/versioning/api/ab-tests` | List all A/B tests |
| POST | `/api/v1/versioning/api/ab-tests` | Create new A/B test |
| GET | `/api/v1/versioning/api/ab-tests/{id}` | Get test details |
| POST | `/api/v1/versioning/api/ab-tests/{id}/start` | Start test |
| POST | `/api/v1/versioning/api/ab-tests/{id}/conclude` | Conclude test |
| POST | `/api/v1/versioning/api/ab-tests/{id}/feedback` | Record feedback |

#### Performance & Rollback

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/versioning/api/models/performance` | Get current metrics |
| GET | `/api/v1/versioning/api/models/performance/history` | Get history |
| GET | `/api/v1/versioning/api/models/performance/thresholds` | Get thresholds |
| PUT | `/api/v1/versioning/api/models/performance/thresholds` | Update thresholds |
| POST | `/api/v1/versioning/api/models/rollback` | Manual rollback |
| GET | `/api/v1/versioning/api/models/rollback/history` | Rollback history |
| POST | `/api/v1/versioning/api/models/feedback` | Record prediction feedback |

## Usage Examples

### Example: Canary Deployment

```python
from app.services.model_versioning import (
    get_version_manager,
    SemanticVersion,
    TrafficSplitStrategy
)

manager = get_version_manager()

# 1. Register new model version
manager.register_version(
    model_path='models/flood_rf_model_v2.joblib',
    version=SemanticVersion(2, 0, 0),
    metadata={'accuracy': 0.94}
)

# 2. Create canary test (starts at 10%, increases 10% each phase)
test = manager.create_ab_test(
    test_id='canary_v2',
    name='v2.0.0 Canary Deployment',
    description='Gradual rollout of v2.0.0',
    control_version=SemanticVersion(1, 0, 0),
    treatment_version=SemanticVersion(2, 0, 0),
    strategy=TrafficSplitStrategy.CANARY,
    target_sample_size=10000
)

# 3. Start canary
manager.start_ab_test('canary_v2')

# 4. Monitor and increment canary percentage
# (could be automated based on performance)
test.increment_canary()  # Now at 20%
test.increment_canary()  # Now at 30%
# ... continue until 100%

# 5. If successful, conclude and promote
results = manager.conclude_ab_test('canary_v2', promote_winner=True)
```

### Example: Integration with Prediction Endpoint

```python
from flask import request, jsonify
from app.services.model_versioning import get_version_manager

@app.route('/predict', methods=['POST'])
def predict():
    manager = get_version_manager()
    data = request.get_json()
    
    # Check for active A/B test
    ab_test_id = data.get('ab_test_id')
    
    if ab_test_id:
        model, variant = manager.get_ab_test_variant(
            ab_test_id, 
            user_id=data.get('user_id')
        )
        if model:
            # Use A/B test variant
            start = time.time()
            result = make_prediction(model, data)
            latency = (time.time() - start) * 1000
            
            manager.record_ab_prediction(
                test_id=ab_test_id,
                variant_name=variant.name,
                latency_ms=latency,
                confidence=result.get('confidence')
            )
            
            result['variant'] = variant.name
            return jsonify(result)
    
    # Normal prediction with production model
    start = time.time()
    result = make_prediction(get_production_model(), data)
    latency = (time.time() - start) * 1000
    
    # Record for monitoring (may trigger rollback)
    rollback = manager.record_prediction(
        latency_ms=latency,
        confidence=result.get('confidence'),
        risk_level=result.get('risk_level')
    )
    
    if rollback:
        result['warning'] = f"Model rolled back: {rollback.reason.value}"
    
    return jsonify(result)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ModelVersionManager                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Version   │  │   A/B Test  │  │    Performance         │  │
│  │   Registry  │  │   Manager   │  │    Monitor             │  │
│  │             │  │             │  │                        │  │
│  │ - Semantic  │  │ - Tests     │  │ - Metrics tracking     │  │
│  │   versions  │  │ - Variants  │  │ - Threshold checking   │  │
│  │ - Promotion │  │ - Metrics   │  │ - Auto rollback        │  │
│  │ - Registry  │  │ - Traffic   │  │ - History              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ModelLoader                              │
│                    (Basic Singleton)                            │
│                                                                 │
│  - Model loading/caching                                        │
│  - Integrity verification                                       │
│  - Metadata management                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Best Practices

1. **Always use semantic versioning** for clarity on change magnitude
2. **Start with canary deployments** for major version changes
3. **Set appropriate thresholds** based on your SLAs
4. **Monitor rollback history** to identify problematic patterns
5. **Use sticky assignments** for user-facing applications to ensure consistent experience
6. **Record feedback** when ground truth becomes available for accurate accuracy tracking
7. **Register rollback callbacks** to integrate with alerting systems
