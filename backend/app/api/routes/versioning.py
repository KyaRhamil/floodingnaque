"""
Model Versioning & A/B Testing API Routes.

Provides REST endpoints for:
- Semantic version management
- A/B test creation and management
- Performance monitoring
- Rollback operations
"""

import logging

from app.api.middleware.rate_limit import get_endpoint_limit, limiter
from app.services.model_versioning import (
    ABTestStatus,
    PerformanceThresholds,
    SemanticVersion,
    TrafficSplitStrategy,
    VersionBumpType,
    get_version_manager,
)
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

versioning_bp = Blueprint("versioning", __name__)


# =============================================================================
# Version Management Endpoints
# =============================================================================


@versioning_bp.route("/api/models/versions", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def list_model_versions():
    """
    List all available model versions with semantic versioning.

    Returns detailed version information including semantic version,
    metadata, and whether each version is currently active.

    Returns:
        200: List of model versions
        500: Internal server error
    ---
    tags:
      - Model Versioning
    produces:
      - application/json
    responses:
      200:
        description: List of model versions
        schema:
          type: object
          properties:
            versions:
              type: array
              items:
                type: object
                properties:
                  version:
                    type: object
                    properties:
                      major:
                        type: integer
                      minor:
                        type: integer
                      patch:
                        type: integer
                      version_string:
                        type: string
                  path:
                    type: string
                  is_current:
                    type: boolean
            current_version:
              type: string
            total_count:
              type: integer
    """
    try:
        manager = get_version_manager()
        versions = manager.list_versions()
        current = manager.get_current_version()

        request_id = getattr(request, "request_id", "unknown")
        return (
            jsonify(
                {
                    "versions": versions,
                    "current_version": str(current) if current else None,
                    "total_count": len(versions),
                    "request_id": request_id,
                }
            ),
            200,
        )

    except Exception as e:
        request_id = getattr(request, "request_id", "unknown")
        logger.error(f"Error listing versions [{request_id}]: {str(e)}")
        return jsonify({"error": "Failed to list model versions", "request_id": request_id}), 500


@versioning_bp.route("/api/models/versions/current", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def get_current_version():
    """
    Get the current active model version.

    Returns:
        200: Current version information
        404: No active version
    ---
    tags:
      - Model Versioning
    """
    try:
        manager = get_version_manager()
        current = manager.get_current_version()

        if current is None:
            return (
                jsonify({"error": "No active model version", "request_id": getattr(request, "request_id", "unknown")}),
                404,
            )

        return jsonify({"version": current.to_dict(), "request_id": getattr(request, "request_id", "unknown")}), 200

    except Exception as e:
        logger.error(f"Error getting current version: {str(e)}")
        return jsonify({"error": "Failed to get current version"}), 500


@versioning_bp.route("/api/models/versions/next", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def get_next_version():
    """
    Get the next version number for each bump type.

    Query Parameters:
        bump_type: Optional. One of 'major', 'minor', 'patch'. Default is 'minor'.

    Returns:
        200: Next version information
    ---
    tags:
      - Model Versioning
    """
    try:
        manager = get_version_manager()
        bump_type_str = request.args.get("bump_type", "minor")

        try:
            bump_type = VersionBumpType(bump_type_str)
        except ValueError:
            return jsonify({"error": f"Invalid bump_type: {bump_type_str}. Must be 'major', 'minor', or 'patch'"}), 400

        next_version = manager.get_next_version(bump_type)

        return (
            jsonify(
                {
                    "current_version": str(manager.get_current_version()) if manager.get_current_version() else None,
                    "next_version": next_version.to_dict(),
                    "bump_type": bump_type.value,
                    "all_next_versions": {
                        "major": str(manager.get_next_version(VersionBumpType.MAJOR)),
                        "minor": str(manager.get_next_version(VersionBumpType.MINOR)),
                        "patch": str(manager.get_next_version(VersionBumpType.PATCH)),
                    },
                    "request_id": getattr(request, "request_id", "unknown"),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error calculating next version: {str(e)}")
        return jsonify({"error": "Failed to calculate next version"}), 500


@versioning_bp.route("/api/models/versions/<version>/promote", methods=["POST"])
@limiter.limit(get_endpoint_limit("models"))
def promote_version(version: str):
    """
    Promote a specific version to be the active production model.

    Args:
        version: Semantic version string (e.g., "1.2.3")

    Request Body:
        backup_current: boolean - Whether to backup current version (default: true)

    Returns:
        200: Promotion successful
        400: Invalid version or promotion failed
        404: Version not found
    ---
    tags:
      - Model Versioning
    """
    try:
        manager = get_version_manager()
        data = request.get_json() or {}
        backup_current = data.get("backup_current", True)

        target_version = SemanticVersion.parse(version)

        # Check if version exists
        versions = manager.list_versions()
        version_exists = any(v["version"]["version_string"] == str(target_version) for v in versions)

        if not version_exists:
            return (
                jsonify(
                    {
                        "error": f"Version {version} not found",
                        "available_versions": [v["version"]["version_string"] for v in versions],
                        "request_id": getattr(request, "request_id", "unknown"),
                    }
                ),
                404,
            )

        success = manager.promote_version(target_version, backup_current=backup_current)

        if success:
            logger.info(f"Promoted version {version} to production")
            return (
                jsonify(
                    {
                        "message": f"Successfully promoted version {version}",
                        "version": target_version.to_dict(),
                        "request_id": getattr(request, "request_id", "unknown"),
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "error": f"Failed to promote version {version}",
                        "request_id": getattr(request, "request_id", "unknown"),
                    }
                ),
                400,
            )

    except Exception as e:
        logger.error(f"Error promoting version {version}: {str(e)}")
        return jsonify({"error": "Failed to promote version"}), 500


# =============================================================================
# A/B Testing Endpoints
# =============================================================================


@versioning_bp.route("/api/ab-tests", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def list_ab_tests():
    """
    List all A/B tests.

    Query Parameters:
        status: Optional. Filter by status ('created', 'running', 'paused', 'completed', 'cancelled')

    Returns:
        200: List of A/B tests
    ---
    tags:
      - A/B Testing
    """
    try:
        manager = get_version_manager()

        status_filter = request.args.get("status")
        status = None
        if status_filter:
            try:
                status = ABTestStatus(status_filter)
            except ValueError:
                return jsonify({"error": f"Invalid status: {status_filter}"}), 400

        tests = manager.list_ab_tests(status)

        return (
            jsonify(
                {"tests": tests, "total_count": len(tests), "request_id": getattr(request, "request_id", "unknown")}
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error listing A/B tests: {str(e)}")
        return jsonify({"error": "Failed to list A/B tests"}), 500


@versioning_bp.route("/api/ab-tests", methods=["POST"])
@limiter.limit(get_endpoint_limit("models"))
def create_ab_test():
    """
    Create a new A/B test.

    Request Body:
        test_id: string - Unique test identifier
        name: string - Human-readable name
        description: string - Test description
        control_version: string - Control version (e.g., "1.0.0")
        treatment_version: string - Treatment version (e.g., "2.0.0")
        traffic_split: float - Traffic percentage for treatment (0.0-1.0)
        strategy: string - Traffic split strategy ('random', 'round_robin', 'sticky', 'canary')
        target_sample_size: integer - Number of predictions needed

    Returns:
        201: Test created
        400: Invalid request
    ---
    tags:
      - A/B Testing
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        required_fields = ["test_id", "name", "control_version", "treatment_version"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        # Parse strategy
        strategy_str = data.get("strategy", "random")
        try:
            strategy = TrafficSplitStrategy(strategy_str)
        except ValueError:
            return (
                jsonify(
                    {"error": f"Invalid strategy: {strategy_str}. Valid options: random, round_robin, sticky, canary"}
                ),
                400,
            )

        manager = get_version_manager()

        test = manager.create_ab_test(
            test_id=data["test_id"],
            name=data["name"],
            description=data.get("description", ""),
            control_version=SemanticVersion.parse(data["control_version"]),
            treatment_version=SemanticVersion.parse(data["treatment_version"]),
            traffic_split=data.get("traffic_split", 0.5),
            strategy=strategy,
            target_sample_size=data.get("target_sample_size", 1000),
        )

        logger.info(f"Created A/B test: {data['test_id']}")
        return (
            jsonify(
                {
                    "message": "A/B test created",
                    "test": test.to_dict(),
                    "request_id": getattr(request, "request_id", "unknown"),
                }
            ),
            201,
        )

    except ValueError as e:
        return jsonify({"error": "Invalid request parameters"}), 400
    except Exception as e:
        logger.error(f"Error creating A/B test: {str(e)}")
        return jsonify({"error": "Failed to create A/B test"}), 500


@versioning_bp.route("/api/ab-tests/<test_id>", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def get_ab_test(test_id: str):
    """
    Get details of a specific A/B test.

    Returns:
        200: Test details
        404: Test not found
    ---
    tags:
      - A/B Testing
    """
    try:
        manager = get_version_manager()
        test = manager.get_ab_test(test_id)

        if test is None:
            return (
                jsonify(
                    {"error": f"Test {test_id} not found", "request_id": getattr(request, "request_id", "unknown")}
                ),
                404,
            )

        return jsonify({"test": test, "request_id": getattr(request, "request_id", "unknown")}), 200

    except Exception as e:
        logger.error(f"Error getting A/B test {test_id}: {str(e)}")
        return jsonify({"error": "Failed to get A/B test"}), 500


@versioning_bp.route("/api/ab-tests/<test_id>/start", methods=["POST"])
@limiter.limit(get_endpoint_limit("models"))
def start_ab_test(test_id: str):
    """
    Start an A/B test.

    Returns:
        200: Test started
        400: Failed to start
        404: Test not found
    ---
    tags:
      - A/B Testing
    """
    try:
        manager = get_version_manager()

        test = manager.get_ab_test(test_id)
        if test is None:
            return jsonify({"error": f"Test {test_id} not found"}), 404

        success = manager.start_ab_test(test_id)

        if success:
            logger.info(f"Started A/B test: {test_id}")
            return (
                jsonify(
                    {
                        "message": f"A/B test {test_id} started",
                        "test": manager.get_ab_test(test_id),
                        "request_id": getattr(request, "request_id", "unknown"),
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "error": f"Failed to start A/B test {test_id}",
                        "request_id": getattr(request, "request_id", "unknown"),
                    }
                ),
                400,
            )

    except Exception as e:
        logger.error(f"Error starting A/B test {test_id}: {str(e)}")
        return jsonify({"error": "Failed to start A/B test"}), 500


@versioning_bp.route("/api/ab-tests/<test_id>/conclude", methods=["POST"])
@limiter.limit(get_endpoint_limit("models"))
def conclude_ab_test(test_id: str):
    """
    Conclude an A/B test and optionally promote the winner.

    Request Body:
        promote_winner: boolean - Whether to promote the winning variant (default: false)

    Returns:
        200: Test concluded with results
        404: Test not found
    ---
    tags:
      - A/B Testing
    """
    try:
        manager = get_version_manager()
        data = request.get_json() or {}
        promote_winner = data.get("promote_winner", False)

        test = manager.get_ab_test(test_id)
        if test is None:
            return jsonify({"error": f"Test {test_id} not found"}), 404

        results = manager.conclude_ab_test(test_id, promote_winner=promote_winner)

        logger.info(f"Concluded A/B test: {test_id}. Winner: {results.get('winner')}")
        return (
            jsonify(
                {
                    "message": f"A/B test {test_id} concluded",
                    "results": results,
                    "request_id": getattr(request, "request_id", "unknown"),
                }
            ),
            200,
        )

    except ValueError as e:
        return jsonify({"error": "Test not found or invalid parameters"}), 404
    except Exception as e:
        logger.error(f"Error concluding A/B test {test_id}: {str(e)}")
        return jsonify({"error": "Failed to conclude A/B test"}), 500


@versioning_bp.route("/api/ab-tests/<test_id>/feedback", methods=["POST"])
@limiter.limit(get_endpoint_limit("predict"))
def record_ab_feedback(test_id: str):
    """
    Record feedback for A/B test accuracy tracking.

    Request Body:
        variant_name: string - The variant that made the prediction
        was_correct: boolean - Whether the prediction was correct

    Returns:
        200: Feedback recorded
        400: Invalid request
    ---
    tags:
      - A/B Testing
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        required_fields = ["variant_name", "was_correct"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        manager = get_version_manager()
        manager.record_ab_feedback(test_id=test_id, variant_name=data["variant_name"], was_correct=data["was_correct"])

        return jsonify({"message": "Feedback recorded", "request_id": getattr(request, "request_id", "unknown")}), 200

    except Exception as e:
        logger.error(f"Error recording A/B feedback: {str(e)}")
        return jsonify({"error": "Failed to record feedback"}), 500


# =============================================================================
# Performance & Rollback Endpoints
# =============================================================================


@versioning_bp.route("/api/models/performance", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def get_performance():
    """
    Get current model performance metrics.

    Returns:
        200: Performance metrics
    ---
    tags:
      - Performance Monitoring
    """
    try:
        manager = get_version_manager()
        performance = manager.get_current_performance()

        return jsonify({"performance": performance, "request_id": getattr(request, "request_id", "unknown")}), 200

    except Exception as e:
        logger.error(f"Error getting performance: {str(e)}")
        return jsonify({"error": "Failed to get performance metrics"}), 500


@versioning_bp.route("/api/models/performance/history", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def get_performance_history():
    """
    Get historical performance snapshots.

    Returns:
        200: Performance history
    ---
    tags:
      - Performance Monitoring
    """
    try:
        manager = get_version_manager()
        history = manager.get_performance_history()

        return (
            jsonify(
                {
                    "history": history,
                    "total_count": len(history),
                    "request_id": getattr(request, "request_id", "unknown"),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting performance history: {str(e)}")
        return jsonify({"error": "Failed to get performance history"}), 500


@versioning_bp.route("/api/models/performance/thresholds", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def get_thresholds():
    """
    Get current performance thresholds for auto-rollback.

    Returns:
        200: Current thresholds
    ---
    tags:
      - Performance Monitoring
    """
    try:
        manager = get_version_manager()
        performance = manager.get_current_performance()

        return (
            jsonify({"thresholds": performance["thresholds"], "request_id": getattr(request, "request_id", "unknown")}),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting thresholds: {str(e)}")
        return jsonify({"error": "Failed to get thresholds"}), 500


@versioning_bp.route("/api/models/performance/thresholds", methods=["PUT"])
@limiter.limit(get_endpoint_limit("models"))
def update_thresholds():
    """
    Update performance thresholds for auto-rollback.

    Request Body:
        min_accuracy: float - Minimum acceptable accuracy (0.0-1.0)
        max_error_rate: float - Maximum acceptable error rate (0.0-1.0)
        max_latency_ms: float - Maximum acceptable latency in milliseconds
        min_confidence: float - Minimum acceptable confidence (0.0-1.0)
        evaluation_window: integer - Number of predictions to evaluate
        consecutive_failures: integer - Failures before rollback

    Returns:
        200: Thresholds updated
        400: Invalid values
    ---
    tags:
      - Performance Monitoring
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        # Validate values
        if "min_accuracy" in data and not (0 <= data["min_accuracy"] <= 1):
            return jsonify({"error": "min_accuracy must be between 0 and 1"}), 400
        if "max_error_rate" in data and not (0 <= data["max_error_rate"] <= 1):
            return jsonify({"error": "max_error_rate must be between 0 and 1"}), 400
        if "max_latency_ms" in data and data["max_latency_ms"] <= 0:
            return jsonify({"error": "max_latency_ms must be positive"}), 400

        manager = get_version_manager()

        # Create new thresholds with provided values
        thresholds = PerformanceThresholds(
            min_accuracy=data.get("min_accuracy", 0.70),
            max_error_rate=data.get("max_error_rate", 0.10),
            max_latency_ms=data.get("max_latency_ms", 500.0),
            min_confidence=data.get("min_confidence", 0.50),
            evaluation_window=data.get("evaluation_window", 100),
            consecutive_failures=data.get("consecutive_failures", 3),
        )

        manager.set_performance_thresholds(thresholds)

        logger.info("Updated performance thresholds")
        return (
            jsonify(
                {
                    "message": "Thresholds updated",
                    "thresholds": manager.get_current_performance()["thresholds"],
                    "request_id": getattr(request, "request_id", "unknown"),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error updating thresholds: {str(e)}")
        return jsonify({"error": "Failed to update thresholds"}), 500


@versioning_bp.route("/api/models/rollback", methods=["POST"])
@limiter.limit(get_endpoint_limit("models"))
def manual_rollback():
    """
    Manually trigger a rollback to a specific version.

    Request Body:
        to_version: string - Target version to rollback to (e.g., "1.0.0")
        reason: string - Optional reason for rollback

    Returns:
        200: Rollback successful
        400: Invalid version or rollback failed
        404: Version not found
    ---
    tags:
      - Rollback
    """
    try:
        data = request.get_json()
        if not data or "to_version" not in data:
            return jsonify({"error": "to_version is required"}), 400

        manager = get_version_manager()
        target_version = SemanticVersion.parse(data["to_version"])
        reason = data.get("reason", "Manual rollback requested")

        # Check if version exists
        versions = manager.list_versions()
        version_exists = any(v["version"]["version_string"] == str(target_version) for v in versions)

        if not version_exists:
            return (
                jsonify(
                    {
                        "error": f"Version {data['to_version']} not found",
                        "available_versions": [v["version"]["version_string"] for v in versions],
                    }
                ),
                404,
            )

        event = manager.manual_rollback(target_version, details=reason)

        logger.warning(f"Manual rollback executed: {event.from_version} -> {event.to_version}")
        return (
            jsonify(
                {
                    "message": "Rollback successful",
                    "event": event.to_dict(),
                    "request_id": getattr(request, "request_id", "unknown"),
                }
            ),
            200,
        )

    except ValueError as e:
        return jsonify({"error": "Invalid version format"}), 400
    except Exception as e:
        logger.error(f"Error executing rollback: {str(e)}")
        return jsonify({"error": "Failed to execute rollback"}), 500


@versioning_bp.route("/api/models/rollback/history", methods=["GET"])
@limiter.limit(get_endpoint_limit("models"))
def get_rollback_history():
    """
    Get history of all rollback events.

    Returns:
        200: Rollback history
    ---
    tags:
      - Rollback
    """
    try:
        manager = get_version_manager()
        history = manager.get_rollback_history()

        return (
            jsonify(
                {
                    "history": history,
                    "total_count": len(history),
                    "request_id": getattr(request, "request_id", "unknown"),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting rollback history: {str(e)}")
        return jsonify({"error": "Failed to get rollback history"}), 500


@versioning_bp.route("/api/models/feedback", methods=["POST"])
@limiter.limit(get_endpoint_limit("predict"))
def record_prediction_feedback():
    """
    Record feedback for production model accuracy tracking.

    This endpoint allows recording whether predictions were correct,
    which is used for accuracy calculation and auto-rollback decisions.

    Request Body:
        was_correct: boolean - Whether the prediction was correct

    Returns:
        200: Feedback recorded
        400: Invalid request
    ---
    tags:
      - Performance Monitoring
    """
    try:
        data = request.get_json()
        if not data or "was_correct" not in data:
            return jsonify({"error": "was_correct is required"}), 400

        manager = get_version_manager()
        manager.record_feedback(data["was_correct"])

        return jsonify({"message": "Feedback recorded", "request_id": getattr(request, "request_id", "unknown")}), 200

    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        return jsonify({"error": "Failed to record feedback"}), 500
