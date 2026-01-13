"""Feature Flags API Routes.

Provides REST API endpoints for managing feature flags:
- List all feature flags
- Get flag status
- Update flag configuration
- Set emergency bypass
- View flag statistics
"""

from flask import Blueprint, jsonify, request, g
from app.services.feature_flags import (
    get_feature_flag_service,
    is_feature_enabled,
    get_alert_thresholds,
    should_use_model_v2,
    should_bypass_external_api,
    FeatureFlagType
)
from app.api.middleware.rate_limit import limiter, get_endpoint_limit
from app.api.middleware.api_key import require_api_key, require_admin
from functools import wraps
import logging
import os

logger = logging.getLogger(__name__)

feature_flags_bp = Blueprint('feature_flags', __name__)


def require_feature_flag_admin(f):
    """Decorator to require admin access for feature flag management."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for admin API key or internal token
        api_key = request.headers.get('X-API-Key')
        internal_token = request.headers.get('X-Internal-Token')
        expected_internal = os.getenv('INTERNAL_API_TOKEN')
        
        # Admin API keys (comma-separated list)
        admin_keys = os.getenv('ADMIN_API_KEYS', '').split(',')
        admin_keys = [k.strip() for k in admin_keys if k.strip()]
        
        is_admin = False
        
        if api_key and api_key in admin_keys:
            is_admin = True
        elif internal_token and expected_internal and internal_token == expected_internal:
            is_admin = True
        elif getattr(g, 'user_role', None) == 'admin':
            is_admin = True
        
        if not is_admin:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'ADMIN_REQUIRED',
                    'message': 'Admin access required for feature flag management'
                }
            }), 403
        
        return f(*args, **kwargs)
    return decorated_function


@feature_flags_bp.route('/', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
def list_flags():
    """
    List all feature flags.
    
    Query Parameters:
        tags: Comma-separated list of tags to filter by
        enabled_only: If true, only return enabled flags
    
    Returns:
        200: List of feature flags
    ---
    tags:
      - Feature Flags
    parameters:
      - name: tags
        in: query
        type: string
        description: Comma-separated tags to filter by
      - name: enabled_only
        in: query
        type: boolean
        description: Only return enabled flags
    responses:
      200:
        description: List of feature flags
    """
    service = get_feature_flag_service()
    
    # Parse query parameters
    tags = request.args.get('tags')
    tags_list = [t.strip() for t in tags.split(',')] if tags else None
    enabled_only = request.args.get('enabled_only', 'false').lower() == 'true'
    
    flags = service.list_flags(tags=tags_list)
    
    if enabled_only:
        flags = [f for f in flags if f.enabled]
    
    return jsonify({
        'success': True,
        'data': {
            'flags': [f.to_dict() for f in flags],
            'count': len(flags)
        }
    }), 200


@feature_flags_bp.route('/<flag_name>', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
def get_flag(flag_name: str):
    """
    Get a specific feature flag.
    
    Path Parameters:
        flag_name: Name of the feature flag
    
    Query Parameters:
        user_id: Optional user ID for percentage/experiment evaluation
        segment: Optional user segment
    
    Returns:
        200: Feature flag details and evaluation result
        404: Flag not found
    ---
    tags:
      - Feature Flags
    parameters:
      - name: flag_name
        in: path
        type: string
        required: true
      - name: user_id
        in: query
        type: string
      - name: segment
        in: query
        type: string
    responses:
      200:
        description: Feature flag details
      404:
        description: Flag not found
    """
    service = get_feature_flag_service()
    flag = service.get_flag(flag_name)
    
    if flag is None:
        return jsonify({
            'success': False,
            'error': {
                'code': 'FLAG_NOT_FOUND',
                'message': f'Feature flag "{flag_name}" not found'
            }
        }), 404
    
    # Get evaluation parameters
    user_id = request.args.get('user_id')
    segment = request.args.get('segment')
    
    # Evaluate flag
    is_enabled = service.is_enabled(flag_name, user_id=user_id, segment=segment)
    
    response_data = {
        'flag': flag.to_dict(),
        'evaluation': {
            'enabled': is_enabled,
            'user_id': user_id,
            'segment': segment
        }
    }
    
    # Add experiment group if applicable
    if flag.flag_type == FeatureFlagType.EXPERIMENT and user_id:
        group = service.get_experiment_group(flag_name, user_id)
        config = service.get_experiment_config(flag_name, user_id)
        response_data['evaluation']['experiment_group'] = group
        response_data['evaluation']['experiment_config'] = config
    
    return jsonify({
        'success': True,
        'data': response_data
    }), 200


@feature_flags_bp.route('/<flag_name>', methods=['PATCH'])
@limiter.limit("10 per minute")
@require_feature_flag_admin
def update_flag(flag_name: str):
    """
    Update a feature flag.
    
    Path Parameters:
        flag_name: Name of the feature flag
    
    Request Body:
        enabled: Boolean to enable/disable flag
        rollout_percentage: Integer 0-100 for percentage rollout
        force_value: Boolean for emergency override
    
    Returns:
        200: Updated flag
        400: Invalid request
        403: Admin access required
        404: Flag not found
    ---
    tags:
      - Feature Flags
    parameters:
      - name: flag_name
        in: path
        type: string
        required: true
      - name: body
        in: body
        schema:
          type: object
          properties:
            enabled:
              type: boolean
            rollout_percentage:
              type: integer
              minimum: 0
              maximum: 100
            force_value:
              type: boolean
    responses:
      200:
        description: Updated flag
      400:
        description: Invalid request
      403:
        description: Admin access required
      404:
        description: Flag not found
    """
    service = get_feature_flag_service()
    flag = service.get_flag(flag_name)
    
    if flag is None:
        return jsonify({
            'success': False,
            'error': {
                'code': 'FLAG_NOT_FOUND',
                'message': f'Feature flag "{flag_name}" not found'
            }
        }), 404
    
    data = request.get_json() or {}
    
    # Validate input
    enabled = data.get('enabled')
    rollout_percentage = data.get('rollout_percentage')
    force_value = data.get('force_value')
    
    if rollout_percentage is not None:
        if not isinstance(rollout_percentage, int) or rollout_percentage < 0 or rollout_percentage > 100:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_PERCENTAGE',
                    'message': 'rollout_percentage must be an integer between 0 and 100'
                }
            }), 400
    
    # Update flag
    success = service.update_flag(
        flag_name,
        enabled=enabled,
        rollout_percentage=rollout_percentage,
        force_value=force_value
    )
    
    if success:
        logger.info(f"Feature flag '{flag_name}' updated by {getattr(g, 'user_id', 'unknown')}")
        
        return jsonify({
            'success': True,
            'data': {
                'flag': service.get_flag(flag_name).to_dict(),
                'message': f'Feature flag "{flag_name}" updated successfully'
            }
        }), 200
    
    return jsonify({
        'success': False,
        'error': {
            'code': 'UPDATE_FAILED',
            'message': 'Failed to update feature flag'
        }
    }), 500


@feature_flags_bp.route('/<flag_name>/emergency', methods=['POST'])
@limiter.limit("5 per minute")
@require_feature_flag_admin
def set_emergency_bypass(flag_name: str):
    """
    Set emergency bypass for a feature flag.
    
    This is used for emergency situations where a feature needs to be
    immediately enabled or disabled, bypassing normal rollout rules.
    
    Path Parameters:
        flag_name: Name of the feature flag
    
    Request Body:
        bypass: Boolean to enable/disable emergency bypass
        reason: String explaining the reason for bypass
    
    Returns:
        200: Bypass set successfully
        403: Admin access required
        404: Flag not found
    ---
    tags:
      - Feature Flags
    parameters:
      - name: flag_name
        in: path
        type: string
        required: true
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - bypass
          properties:
            bypass:
              type: boolean
            reason:
              type: string
    responses:
      200:
        description: Bypass set successfully
      403:
        description: Admin access required
      404:
        description: Flag not found
    """
    service = get_feature_flag_service()
    flag = service.get_flag(flag_name)
    
    if flag is None:
        return jsonify({
            'success': False,
            'error': {
                'code': 'FLAG_NOT_FOUND',
                'message': f'Feature flag "{flag_name}" not found'
            }
        }), 404
    
    data = request.get_json() or {}
    bypass = data.get('bypass')
    reason = data.get('reason', 'No reason provided')
    
    if bypass is None:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MISSING_PARAMETER',
                'message': 'bypass parameter is required'
            }
        }), 400
    
    success = service.set_emergency_bypass(flag_name, bypass)
    
    if success:
        logger.warning(
            f"EMERGENCY BYPASS: Flag '{flag_name}' set to {bypass} "
            f"by {getattr(g, 'user_id', 'unknown')}. Reason: {reason}"
        )
        
        return jsonify({
            'success': True,
            'data': {
                'flag': service.get_flag(flag_name).to_dict(),
                'message': f'Emergency bypass {"enabled" if bypass else "disabled"} for "{flag_name}"'
            }
        }), 200
    
    return jsonify({
        'success': False,
        'error': {
            'code': 'BYPASS_FAILED',
            'message': 'Failed to set emergency bypass'
        }
    }), 500


@feature_flags_bp.route('/stats', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
@require_feature_flag_admin
def get_stats():
    """
    Get feature flag evaluation statistics.
    
    Returns:
        200: Statistics for all flags
        403: Admin access required
    ---
    tags:
      - Feature Flags
    responses:
      200:
        description: Flag statistics
      403:
        description: Admin access required
    """
    service = get_feature_flag_service()
    stats = service.get_stats()
    
    return jsonify({
        'success': True,
        'data': {
            'stats': stats,
            'total_flags': len(service.list_flags())
        }
    }), 200


@feature_flags_bp.route('/evaluate', methods=['POST'])
@limiter.limit(get_endpoint_limit('predict'))
def evaluate_flags():
    """
    Evaluate multiple feature flags for a user.
    
    Request Body:
        flags: List of flag names to evaluate
        user_id: Optional user identifier
        segment: Optional user segment
    
    Returns:
        200: Evaluation results for requested flags
    ---
    tags:
      - Feature Flags
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - flags
          properties:
            flags:
              type: array
              items:
                type: string
            user_id:
              type: string
            segment:
              type: string
    responses:
      200:
        description: Evaluation results
    """
    data = request.get_json() or {}
    
    flag_names = data.get('flags', [])
    user_id = data.get('user_id')
    segment = data.get('segment')
    
    if not flag_names:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MISSING_FLAGS',
                'message': 'flags parameter is required'
            }
        }), 400
    
    service = get_feature_flag_service()
    results = {}
    
    for flag_name in flag_names:
        results[flag_name] = {
            'enabled': service.is_enabled(flag_name, user_id=user_id, segment=segment)
        }
        
        # Add experiment info if applicable
        flag = service.get_flag(flag_name)
        if flag and flag.flag_type == FeatureFlagType.EXPERIMENT and user_id:
            results[flag_name]['experiment_group'] = service.get_experiment_group(flag_name, user_id)
    
    return jsonify({
        'success': True,
        'data': {
            'results': results,
            'user_id': user_id,
            'segment': segment
        }
    }), 200


@feature_flags_bp.route('/experiments/thresholds', methods=['GET'])
@limiter.limit(get_endpoint_limit('predict'))
def get_experiment_thresholds():
    """
    Get alert thresholds for A/B experiment.
    
    Query Parameters:
        user_id: User identifier for experiment assignment
    
    Returns:
        200: Alert thresholds for the user's experiment group
    ---
    tags:
      - Feature Flags
      - Experiments
    parameters:
      - name: user_id
        in: query
        type: string
        required: true
    responses:
      200:
        description: Alert thresholds
    """
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MISSING_USER_ID',
                'message': 'user_id parameter is required'
            }
        }), 400
    
    service = get_feature_flag_service()
    thresholds = get_alert_thresholds(user_id)
    experiment_group = service.get_experiment_group("alert_threshold_experiment", user_id)
    
    return jsonify({
        'success': True,
        'data': {
            'thresholds': thresholds,
            'experiment_group': experiment_group,
            'user_id': user_id
        }
    }), 200


@feature_flags_bp.route('/model/version', methods=['GET'])
@limiter.limit(get_endpoint_limit('predict'))
def get_model_version():
    """
    Get which model version to use for a user.
    
    Query Parameters:
        user_id: Optional user identifier for rollout evaluation
    
    Returns:
        200: Model version information
    ---
    tags:
      - Feature Flags
      - Models
    parameters:
      - name: user_id
        in: query
        type: string
    responses:
      200:
        description: Model version information
    """
    user_id = request.args.get('user_id')
    
    use_v2 = should_use_model_v2(user_id)
    
    service = get_feature_flag_service()
    full_release = service.is_enabled("model_v2_full_release")
    rollout_flag = service.get_flag("model_v2_rollout")
    
    return jsonify({
        'success': True,
        'data': {
            'model_version': 'v2' if use_v2 else 'v1',
            'use_v2': use_v2,
            'rollout_percentage': rollout_flag.rollout_percentage if rollout_flag else 0,
            'full_release': full_release,
            'user_id': user_id
        }
    }), 200


@feature_flags_bp.route('/api-bypass/status', methods=['GET'])
@limiter.limit(get_endpoint_limit('status'))
def get_api_bypass_status():
    """
    Get status of external API bypass flags.
    
    Returns:
        200: Status of all API bypass flags
    ---
    tags:
      - Feature Flags
      - Emergency
    responses:
      200:
        description: API bypass status
    """
    apis = ['openweathermap', 'weatherstack', 'worldtides']
    
    status = {
        'global_bypass': should_bypass_external_api('all_external_apis'),
        'apis': {}
    }
    
    for api in apis:
        status['apis'][api] = {
            'bypassed': should_bypass_external_api(api)
        }
    
    return jsonify({
        'success': True,
        'data': status
    }), 200
