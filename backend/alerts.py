"""
Alert System Module for Flood Detection and Early Warning
Supports SMS, email, and web dashboard notifications
Aligned with research objectives for Parañaque City.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from risk_classifier import format_alert_message, RISK_LEVELS

logger = logging.getLogger(__name__)

# Parañaque City coordinates (default location)
PARANAQUE_COORDS = {
    'lat': 14.4793,
    'lon': 121.0198,
    'name': 'Parañaque City',
    'region': 'Metro Manila',
    'country': 'Philippines'
}


class AlertSystem:
    """Alert system for flood warnings."""
    
    def __init__(self, sms_enabled: bool = False, email_enabled: bool = False):
        """
        Initialize alert system.
        
        Args:
            sms_enabled: Enable SMS notifications (requires SMS gateway API)
            email_enabled: Enable email notifications (requires SMTP config)
        """
        self.sms_enabled = sms_enabled
        self.email_enabled = email_enabled
        self.alert_history = []
    
    def send_alert(
        self,
        risk_data: Dict,
        location: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        alert_type: str = 'web'
    ) -> Dict[str, any]:
        """
        Send flood alert notification.
        
        Args:
            risk_data: Risk classification data from risk_classifier
            location: Location name (default: Parañaque City)
            recipients: List of phone numbers or email addresses
            alert_type: 'web', 'sms', 'email', or 'all'
        
        Returns:
            dict: Alert delivery status
        """
        if not location:
            location = PARANAQUE_COORDS['name']
        
        risk_label = risk_data.get('risk_label', 'Unknown')
        
        # Format alert message
        message = format_alert_message(risk_data, location)
        
        # Create alert record
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'risk_level': risk_data.get('risk_level'),
            'risk_label': risk_label,
            'message': message,
            'delivery_status': {}
        }
        
        # Send based on alert type
        if alert_type in ['web', 'all']:
            alert_record['delivery_status']['web'] = 'delivered'
            logger.info(f"Web alert sent: {risk_label} for {location}")
        
        if alert_type in ['sms', 'all'] and self.sms_enabled and recipients:
            sms_status = self._send_sms(recipients, message)
            alert_record['delivery_status']['sms'] = sms_status
            logger.info(f"SMS alert sent: {risk_label} for {location}")
        
        if alert_type in ['email', 'all'] and self.email_enabled and recipients:
            email_status = self._send_email(recipients, risk_label, message)
            alert_record['delivery_status']['email'] = email_status
            logger.info(f"Email alert sent: {risk_label} for {location}")
        
        # Store in history
        self.alert_history.append(alert_record)
        
        return alert_record
    
    def _send_sms(self, recipients: List[str], message: str) -> str:
        """
        Send SMS notification (placeholder for SMS gateway integration).
        
        Args:
            recipients: List of phone numbers
            message: Alert message
        
        Returns:
            str: Delivery status
        """
        # TODO: Integrate with SMS gateway (e.g., Twilio, Nexmo, local provider)
        # For now, log the message
        logger.info(f"SMS would be sent to {recipients}: {message[:50]}...")
        return 'not_implemented'
    
    def _send_email(self, recipients: List[str], subject: str, message: str) -> str:
        """
        Send email notification (placeholder for SMTP integration).
        
        Args:
            recipients: List of email addresses
            subject: Email subject
            message: Alert message
        
        Returns:
            str: Delivery status
        """
        # TODO: Integrate with SMTP server
        # For now, log the message
        logger.info(f"Email would be sent to {recipients}: {subject}")
        return 'not_implemented'
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get recent alert history."""
        return self.alert_history[-limit:]
    
    def get_alerts_by_risk_level(self, risk_level: int) -> List[Dict]:
        """Get alerts filtered by risk level."""
        return [alert for alert in self.alert_history if alert['risk_level'] == risk_level]


# Global alert system instance
_alert_system = AlertSystem()


def get_alert_system() -> AlertSystem:
    """Get the global alert system instance."""
    return _alert_system


def send_flood_alert(
    risk_data: Dict,
    location: Optional[str] = None,
    recipients: Optional[List[str]] = None,
    alert_type: str = 'web'
) -> Dict:
    """
    Convenience function to send flood alert.
    
    Args:
        risk_data: Risk classification data
        location: Location name
        recipients: List of recipients
        alert_type: 'web', 'sms', 'email', or 'all'
    
    Returns:
        dict: Alert delivery status
    """
    return _alert_system.send_alert(risk_data, location, recipients, alert_type)

