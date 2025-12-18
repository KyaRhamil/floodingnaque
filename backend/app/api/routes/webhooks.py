"""
Webhook Management API.

Endpoints for registering and managing webhooks for flood alerts.
"""

from flask import Blueprint, request, jsonify
from app.models.db import Webhook, get_db_session
from app.api.middleware.auth import require_api_key
from app.api.middleware.rate_limit import limiter
import logging
import secrets
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

webhooks_bp = Blueprint('webhooks', __name__, url_prefix='/webhooks')


@webhooks_bp.route('/register', methods=['POST'])
@require_api_key
@limiter.limit("10 per hour")
def register_webhook():
    """
    Register a new webhook for flood alerts.
    
    Request Body:
    {
        "url": "https://your-system.com/flood-alert",
        "events": ["flood_detected", "critical_risk", "high_risk"],
        "secret": "optional_custom_secret"
    }
    
    Returns:
        201: Webhook registered successfully
        400: Invalid request data
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'url' not in data or 'events' not in data:
            return jsonify({
                'error': 'Missing required fields: url, events'
            }), 400
        
        url = data['url']
        events = data['events']
        
        # Validate URL
        if not url.startswith('http://') and not url.startswith('https://'):
            return jsonify({
                'error': 'Invalid URL format. Must start with http:// or https://'
            }), 400
        
        # Validate events
        valid_events = ['flood_detected', 'critical_risk', 'high_risk', 'medium_risk', 'low_risk']
        if not isinstance(events, list) or not events:
            return jsonify({
                'error': 'Events must be a non-empty list'
            }), 400
        
        for event in events:
            if event not in valid_events:
                return jsonify({
                    'error': f'Invalid event: {event}. Valid events: {valid_events}'
                }), 400
        
        # Generate or use provided secret
        secret = data.get('secret') or secrets.token_urlsafe(32)
        
        # Create webhook
        webhook = Webhook(
            url=url,
            events=json.dumps(events),
            secret=secret,
            is_active=True
        )
        
        with get_db_session() as session:
            session.add(webhook)
            session.commit()
            webhook_id = webhook.id
        
        logger.info(f"Webhook registered: {url} for events {events}")
        
        return jsonify({
            'message': 'Webhook registered successfully',
            'webhook_id': webhook_id,
            'url': url,
            'events': events,
            'secret': secret,
            'is_active': True
        }), 201
        
    except Exception as e:
        logger.error(f"Error registering webhook: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@webhooks_bp.route('/list', methods=['GET'])
@require_api_key
@limiter.limit("30 per minute")
def list_webhooks():
    """
    List all registered webhooks.
    
    Returns:
        200: List of webhooks
    """
    try:
        with get_db_session() as session:
            webhooks = session.query(Webhook).filter_by(is_deleted=False).all()
            
            webhook_list = []
            for webhook in webhooks:
                webhook_list.append({
                    'id': webhook.id,
                    'url': webhook.url,
                    'events': json.loads(webhook.events),
                    'is_active': webhook.is_active,
                    'failure_count': webhook.failure_count,
                    'last_triggered_at': webhook.last_triggered_at.isoformat() if webhook.last_triggered_at else None,
                    'created_at': webhook.created_at.isoformat()
                })
        
        return jsonify({
            'webhooks': webhook_list,
            'count': len(webhook_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@webhooks_bp.route('/<int:webhook_id>', methods=['DELETE'])
@require_api_key
@limiter.limit("10 per hour")
def delete_webhook(webhook_id):
    """
    Delete a webhook.
    
    Args:
        webhook_id: ID of the webhook to delete
        
    Returns:
        200: Webhook deleted successfully
        404: Webhook not found
    """
    try:
        with get_db_session() as session:
            webhook = session.query(Webhook).filter_by(
                id=webhook_id,
                is_deleted=False
            ).first()
            
            if not webhook:
                return jsonify({'error': 'Webhook not found'}), 404
            
            # Soft delete
            webhook.is_deleted = True
            webhook.deleted_at = datetime.now(timezone.utc)
            session.commit()
        
        logger.info(f"Webhook deleted: {webhook_id}")
        
        return jsonify({
            'message': 'Webhook deleted successfully',
            'webhook_id': webhook_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting webhook: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@webhooks_bp.route('/<int:webhook_id>/toggle', methods=['POST'])
@require_api_key
@limiter.limit("30 per minute")
def toggle_webhook(webhook_id):
    """
    Enable or disable a webhook.
    
    Args:
        webhook_id: ID of the webhook to toggle
        
    Returns:
        200: Webhook toggled successfully
        404: Webhook not found
    """
    try:
        with get_db_session() as session:
            webhook = session.query(Webhook).filter_by(
                id=webhook_id,
                is_deleted=False
            ).first()
            
            if not webhook:
                return jsonify({'error': 'Webhook not found'}), 404
            
            webhook.is_active = not webhook.is_active
            webhook.updated_at = datetime.now(timezone.utc)
            session.commit()
            
            new_status = webhook.is_active
        
        logger.info(f"Webhook {webhook_id} {'enabled' if new_status else 'disabled'}")
        
        return jsonify({
            'message': f'Webhook {"enabled" if new_status else "disabled"} successfully',
            'webhook_id': webhook_id,
            'is_active': new_status
        }), 200
        
    except Exception as e:
        logger.error(f"Error toggling webhook: {e}")
        return jsonify({'error': 'Internal server error'}), 500
