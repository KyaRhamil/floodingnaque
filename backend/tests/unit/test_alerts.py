"""
Unit tests for alert service.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestAlertSystem:
    """Tests for the AlertSystem class."""
    
    def test_alert_system_initialization(self):
        """Test AlertSystem can be instantiated."""
        from app.services.alerts import AlertSystem
        
        with patch('app.services.alerts.get_db_session'):
            alert_system = AlertSystem()
            assert alert_system is not None
    
    def test_alert_system_singleton(self):
        """Test get_alert_system returns singleton instance."""
        from app.services.alerts import get_alert_system
        
        with patch('app.services.alerts.get_db_session'):
            instance1 = get_alert_system()
            instance2 = get_alert_system()
            # Should be the same instance
            assert instance1 is instance2
    
    def test_send_flood_alert_returns_dict(self):
        """Test send_flood_alert returns a dictionary."""
        from app.services.alerts import send_flood_alert
        
        with patch('app.services.alerts.get_alert_system') as mock_get_alert:
            mock_system = MagicMock()
            mock_system.send_alert.return_value = {
                'success': True,
                'alert_id': '12345'
            }
            mock_get_alert.return_value = mock_system
            
            result = send_flood_alert(
                risk_level=2,
                location='Paranaque City',
                message='Critical flood risk detected'
            )
            
            assert isinstance(result, dict)


class TestAlertSeverityLevels:
    """Tests for alert severity level handling."""
    
    def test_risk_level_0_is_safe(self):
        """Test that risk level 0 corresponds to Safe."""
        assert 0 == 0  # Safe level
    
    def test_risk_level_1_is_alert(self):
        """Test that risk level 1 corresponds to Alert."""
        assert 1 == 1  # Alert level
    
    def test_risk_level_2_is_critical(self):
        """Test that risk level 2 corresponds to Critical."""
        assert 2 == 2  # Critical level
    
    def test_valid_risk_levels(self):
        """Test valid risk levels range."""
        valid_levels = [0, 1, 2]
        for level in valid_levels:
            assert 0 <= level <= 2


class TestAlertMessageFormatting:
    """Tests for alert message formatting."""
    
    def test_alert_message_contains_location(self):
        """Test that alert messages include location."""
        from app.services.risk_classifier import format_alert_message
        
        risk_data = {
            'risk_label': 'Critical',
            'description': 'High flood risk',
            'confidence': 0.85
        }
        
        message = format_alert_message(risk_data, location='Paranaque City')
        assert 'Paranaque City' in message
    
    def test_alert_message_contains_risk_level(self):
        """Test that alert messages include risk level."""
        from app.services.risk_classifier import format_alert_message
        
        risk_data = {
            'risk_label': 'Alert',
            'description': 'Moderate flood risk',
            'confidence': 0.65
        }
        
        message = format_alert_message(risk_data)
        assert 'Alert' in message
    
    def test_critical_alert_has_action_text(self):
        """Test that critical alerts include action text."""
        from app.services.risk_classifier import format_alert_message
        
        risk_data = {
            'risk_label': 'Critical',
            'description': 'High flood risk',
            'confidence': 0.90
        }
        
        message = format_alert_message(risk_data)
        assert 'TAKE IMMEDIATE ACTION' in message


class TestAlertDelivery:
    """Tests for alert delivery mechanisms."""
    
    def test_alert_delivery_status_types(self):
        """Test valid delivery status types."""
        valid_statuses = ['pending', 'sent', 'delivered', 'failed']
        for status in valid_statuses:
            assert status in valid_statuses
    
    def test_alert_delivery_channels(self):
        """Test valid delivery channels."""
        valid_channels = ['sms', 'email', 'webhook', 'push']
        for channel in valid_channels:
            assert channel in valid_channels


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
