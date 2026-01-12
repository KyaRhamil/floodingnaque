# Real-time Subscription Guide

This document describes real-time data streaming patterns using Server-Sent Events (SSE) in the Floodingnaque API.

## Table of Contents

1. [Overview](#overview)
2. [SSE vs WebSocket](#sse-vs-websocket)
3. [Available Streams](#available-streams)
4. [Connection Management](#connection-management)
5. [Event Types](#event-types)
6. [Frontend Implementation](#frontend-implementation)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)

---

## Overview

Floodingnaque uses **Server-Sent Events (SSE)** for real-time communication. SSE provides a simple, efficient one-way channel from server to client, perfect for flood alert notifications.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SSE Communication Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    GET /sse/alerts    ┌─────────┐                 │
│  │ Frontend│───────────────────────▶│   API   │                 │
│  │         │◀───────────────────────│         │                 │
│  └─────────┘    text/event-stream  └─────────┘                 │
│       │                                  │                      │
│       │◀─────── event: connected ────────│                      │
│       │                                  │                      │
│       │◀─────── event: heartbeat ────────│  (every 30s)        │
│       │                                  │                      │
│       │◀─────── event: alert ────────────│  (when alerts occur)│
│       │                                  │                      │
│       ▼                                  ▼                      │
│    Process events                   Broadcast alerts            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## SSE vs WebSocket

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server → Client only | Bidirectional |
| Protocol | HTTP | WS/WSS |
| Auto-reconnect | Built-in | Manual |
| Browser support | Excellent | Excellent |
| Complexity | Simple | More complex |
| Best for | Notifications, alerts | Chat, gaming |

**Why SSE for Floodingnaque?**
- Alerts only flow from server to clients
- Built-in reconnection handling
- Works through HTTP proxies
- Simpler implementation

---

## Available Streams

### 1. Alert Stream

**Endpoint:** `GET /sse/alerts`

**Purpose:** Stream real-time flood alerts to connected clients.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `risk_level` | int | Filter by minimum risk level (0-2) |

**Example:**
```
GET /sse/alerts?risk_level=1
Accept: text/event-stream
```

**Response Headers:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

---

### 2. SSE Status

**Endpoint:** `GET /sse/status`

**Rate Limit:** 60 per minute

**Purpose:** Check SSE service status and connected client count.

**Response:**
```json
{
  "success": true,
  "status": "operational",
  "connected_clients": 42,
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "abc123"
}
```

---

### 3. Recent Alerts

**Endpoint:** `GET /sse/alerts/recent`

**Rate Limit:** 30 per minute

**Purpose:** Fetch recent alerts for clients on initial connection.

**Query Parameters:**
| Parameter | Type | Default | Max |
|-----------|------|---------|-----|
| `limit` | int | 10 | 50 |
| `since` | datetime | - | - |

**Response:**
```json
{
  "success": true,
  "alerts": [
    {
      "id": 1,
      "risk_level": 2,
      "risk_label": "Critical",
      "location": "Parañaque, NCR",
      "message": "Severe flooding detected in coastal areas",
      "created_at": "2024-01-15T10:25:00Z"
    }
  ],
  "count": 1,
  "request_id": "abc123"
}
```

---

### 4. Test Alert Broadcast

**Endpoint:** `POST /sse/alerts/test`

**Rate Limit:** 5 per minute

**Purpose:** Send test alerts (development/testing only).

**Request:**
```json
{
  "risk_level": 1,
  "message": "Test flood alert",
  "location": "Test Location"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Test alert broadcast to 5 connected clients",
  "alert": {
    "id": "test_1705312200",
    "risk_level": 1,
    "risk_label": "Alert",
    "location": "Test Location",
    "message": "Test flood alert",
    "is_test": true,
    "created_at": "2024-01-15T10:30:00Z"
  },
  "connected_clients": 5,
  "request_id": "abc123"
}
```

---

## Event Types

### 1. connected

Sent immediately when client connects.

```
event: connected
data: {"client_id":"192.168.1.1_1705312200","timestamp":"2024-01-15T10:30:00Z","message":"Connected to flood alert stream","request_id":"abc123"}
```

**Data Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `client_id` | string | Unique client identifier |
| `timestamp` | string | Connection timestamp (ISO 8601) |
| `message` | string | Welcome message |
| `request_id` | string | Request correlation ID |

---

### 2. heartbeat

Sent every 30 seconds to keep connection alive.

```
event: heartbeat
data: {"timestamp":"2024-01-15T10:30:30Z","status":"connected"}
```

**Data Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | Heartbeat timestamp |
| `status` | string | Connection status |

---

### 3. alert

Sent when a flood alert is generated.

```
event: alert
data: {"timestamp":"2024-01-15T10:35:00Z","alert":{"id":123,"risk_level":2,"risk_label":"Critical","location":"Parañaque, NCR","message":"Severe flooding detected in coastal areas","created_at":"2024-01-15T10:35:00Z"}}
```

**Alert Data Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | int/string | Alert identifier |
| `risk_level` | int | Risk level (0=Safe, 1=Alert, 2=Critical) |
| `risk_label` | string | Human-readable risk label |
| `location` | string | Affected location |
| `message` | string | Alert message |
| `is_test` | boolean | Whether this is a test alert |
| `created_at` | string | Alert creation timestamp |

**Risk Levels:**
| Level | Label | Color | Description |
|-------|-------|-------|-------------|
| 0 | Safe | Green | Normal conditions |
| 1 | Alert | Yellow/Orange | Elevated risk, monitor conditions |
| 2 | Critical | Red | Immediate action required |

---

## Connection Management

### Connection Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SSE Connection Lifecycle                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Client                                                              │
│     │                                                                 │
│     │  new EventSource('/sse/alerts')                                 │
│     ▼                                                                 │
│   ┌───────────────┐                                                   │
│   │  CONNECTING   │ ────────────▶ onopen()                           │
│   └───────────────┘                                                   │
│          │                                                            │
│          │ Connected                                                  │
│          ▼                                                            │
│   ┌───────────────┐                                                   │
│   │     OPEN      │ ────────────▶ onmessage()                        │
│   └───────────────┘              onerror() → auto-reconnect          │
│          │                                                            │
│          │ Connection lost                                            │
│          ▼                                                            │
│   ┌───────────────┐                                                   │
│   │  CONNECTING   │ ◀──── Browser auto-reconnects                    │
│   └───────────────┘                                                   │
│          │                                                            │
│          │ eventSource.close()                                        │
│          ▼                                                            │
│   ┌───────────────┐                                                   │
│   │    CLOSED     │                                                   │
│   └───────────────┘                                                   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Server-Side Management

The server maintains a client registry with the following features:

1. **Client Queue**: Each client has a message queue (max 100 messages)
2. **Heartbeat**: 30-second keepalive prevents proxy timeouts
3. **Slow Client Detection**: Clients with full queues are disconnected
4. **Broadcast**: Messages sent to all connected clients simultaneously

---

## Frontend Implementation

### Basic Implementation

```javascript
// Basic SSE connection
const eventSource = new EventSource('/sse/alerts');

eventSource.onopen = () => {
  console.log('Connected to alert stream');
};

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

eventSource.onerror = (error) => {
  console.error('SSE error:', error);
  // Browser will auto-reconnect
};

// Listen for specific event types
eventSource.addEventListener('alert', (event) => {
  const data = JSON.parse(event.data);
  showAlertNotification(data.alert);
});

eventSource.addEventListener('heartbeat', (event) => {
  console.log('Heartbeat received');
});
```

### Complete Alert Service

```javascript
// services/alertService.js
class AlertService {
  constructor(baseUrl = '/sse') {
    this.baseUrl = baseUrl;
    this.eventSource = null;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectDelay = 1000;
    this.isConnected = false;
  }

  // Connect to alert stream
  connect(options = {}) {
    const { riskLevel } = options;
    let url = `${this.baseUrl}/alerts`;
    
    if (riskLevel !== undefined) {
      url += `?risk_level=${riskLevel}`;
    }

    this.disconnect(); // Close existing connection
    
    this.eventSource = new EventSource(url);
    
    this.eventSource.onopen = () => {
      console.log('Alert stream connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.emit('connected');
    };

    this.eventSource.onerror = (error) => {
      console.error('Alert stream error:', error);
      this.isConnected = false;
      this.emit('error', error);
      
      // Check if we should attempt manual reconnect
      if (this.eventSource.readyState === EventSource.CLOSED) {
        this.handleReconnect(options);
      }
    };

    // Event handlers
    this.eventSource.addEventListener('connected', (event) => {
      const data = JSON.parse(event.data);
      console.log('Connected:', data);
      this.emit('connected', data);
    });

    this.eventSource.addEventListener('alert', (event) => {
      const data = JSON.parse(event.data);
      console.log('Alert received:', data);
      this.emit('alert', data.alert);
    });

    this.eventSource.addEventListener('heartbeat', (event) => {
      const data = JSON.parse(event.data);
      this.emit('heartbeat', data);
    });

    return this;
  }

  // Disconnect from stream
  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.isConnected = false;
      this.emit('disconnected');
    }
  }

  // Manual reconnect with exponential backoff
  handleReconnect(options) {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      this.emit('maxReconnectReached');
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    console.log(`Reconnecting in ${delay}ms...`);
    
    this.reconnectAttempts++;
    
    setTimeout(() => {
      this.connect(options);
    }, delay);
  }

  // Event subscription
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
    return () => this.off(event, callback);
  }

  // Remove listener
  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  // Emit event
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (e) {
          console.error('Listener error:', e);
        }
      });
    }
  }

  // Fetch recent alerts on connection
  async fetchRecentAlerts(limit = 10) {
    const response = await fetch(`${this.baseUrl}/alerts/recent?limit=${limit}`);
    const data = await response.json();
    return data.alerts;
  }

  // Get connection status
  getStatus() {
    return {
      connected: this.isConnected,
      readyState: this.eventSource?.readyState,
      reconnectAttempts: this.reconnectAttempts
    };
  }
}

export default new AlertService();
```

### React Hook

```javascript
// hooks/useAlerts.js
import { useState, useEffect, useCallback, useRef } from 'react';
import alertService from '../services/alertService';

export function useAlerts(options = {}) {
  const [alerts, setAlerts] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const optionsRef = useRef(options);

  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const handleAlert = useCallback((alert) => {
    setAlerts(prev => [alert, ...prev].slice(0, 100)); // Keep last 100
    
    // Show notification if browser supports it
    if (Notification.permission === 'granted') {
      new Notification(`Flood Alert: ${alert.risk_label}`, {
        body: alert.message,
        icon: '/alert-icon.png',
        tag: `alert-${alert.id}`
      });
    }
  }, []);

  const handleConnected = useCallback(() => {
    setIsConnected(true);
    setError(null);
  }, []);

  const handleError = useCallback((err) => {
    setError(err);
  }, []);

  const handleDisconnected = useCallback(() => {
    setIsConnected(false);
  }, []);

  useEffect(() => {
    // Subscribe to events
    const unsubAlert = alertService.on('alert', handleAlert);
    const unsubConnected = alertService.on('connected', handleConnected);
    const unsubError = alertService.on('error', handleError);
    const unsubDisconnected = alertService.on('disconnected', handleDisconnected);

    // Connect
    alertService.connect(optionsRef.current);

    // Fetch recent alerts
    alertService.fetchRecentAlerts(10).then(recentAlerts => {
      setAlerts(recentAlerts);
    }).catch(console.error);

    // Cleanup
    return () => {
      unsubAlert();
      unsubConnected();
      unsubError();
      unsubDisconnected();
      alertService.disconnect();
    };
  }, [handleAlert, handleConnected, handleError, handleDisconnected]);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  const reconnect = useCallback(() => {
    alertService.connect(optionsRef.current);
  }, []);

  return {
    alerts,
    isConnected,
    error,
    clearAlerts,
    reconnect
  };
}
```

### React Component Example

```jsx
// components/AlertPanel.jsx
import React from 'react';
import { useAlerts } from '../hooks/useAlerts';

function AlertPanel() {
  const { alerts, isConnected, error, clearAlerts, reconnect } = useAlerts({
    riskLevel: 1 // Only show Alert and Critical
  });

  const getRiskColor = (level) => {
    switch (level) {
      case 0: return 'green';
      case 1: return 'orange';
      case 2: return 'red';
      default: return 'gray';
    }
  };

  return (
    <div className="alert-panel">
      <div className="status-bar">
        <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? '● Connected' : '○ Disconnected'}
        </span>
        {!isConnected && (
          <button onClick={reconnect}>Reconnect</button>
        )}
        <button onClick={clearAlerts}>Clear</button>
      </div>

      {error && (
        <div className="error-banner">
          Connection error. Auto-reconnecting...
        </div>
      )}

      <div className="alert-list">
        {alerts.length === 0 ? (
          <p className="no-alerts">No alerts</p>
        ) : (
          alerts.map(alert => (
            <div 
              key={alert.id} 
              className="alert-item"
              style={{ borderLeftColor: getRiskColor(alert.risk_level) }}
            >
              <div className="alert-header">
                <span className="risk-label" style={{ color: getRiskColor(alert.risk_level) }}>
                  {alert.risk_label}
                </span>
                <span className="timestamp">
                  {new Date(alert.created_at).toLocaleTimeString()}
                </span>
              </div>
              <div className="alert-location">{alert.location}</div>
              <div className="alert-message">{alert.message}</div>
              {alert.is_test && (
                <span className="test-badge">TEST</span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default AlertPanel;
```

### CSS Styling

```css
/* Alert Panel Styles */
.alert-panel {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  overflow: hidden;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 15px;
  background: #f5f5f5;
  border-bottom: 1px solid #ddd;
}

.status-indicator {
  font-size: 14px;
}

.status-indicator.connected {
  color: green;
}

.status-indicator.disconnected {
  color: red;
}

.error-banner {
  background: #ffebee;
  color: #c62828;
  padding: 10px 15px;
  font-size: 14px;
}

.alert-list {
  max-height: 400px;
  overflow-y: auto;
}

.alert-item {
  padding: 15px;
  border-left: 4px solid;
  border-bottom: 1px solid #eee;
}

.alert-item:last-child {
  border-bottom: none;
}

.alert-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.risk-label {
  font-weight: bold;
  font-size: 14px;
}

.timestamp {
  color: #666;
  font-size: 12px;
}

.alert-location {
  font-weight: 500;
  margin-bottom: 5px;
}

.alert-message {
  color: #333;
  font-size: 14px;
}

.test-badge {
  display: inline-block;
  background: #2196f3;
  color: white;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  margin-top: 8px;
}

.no-alerts {
  text-align: center;
  color: #999;
  padding: 30px;
}
```

---

## Error Handling

### Connection Errors

| Error State | Cause | Action |
|-------------|-------|--------|
| `readyState: 0` | Connecting | Wait for connection |
| `readyState: 2` | Connection closed | Browser auto-reconnects |
| Network error | Server unreachable | Show status, auto-reconnect |
| CORS error | Server misconfigured | Check server CORS settings |

### Reconnection Strategy

The browser's EventSource automatically reconnects, but you may want additional control:

```javascript
// Custom reconnection with exponential backoff
class ReconnectingEventSource {
  constructor(url, options = {}) {
    this.url = url;
    this.maxRetries = options.maxRetries || 10;
    this.baseDelay = options.baseDelay || 1000;
    this.maxDelay = options.maxDelay || 30000;
    this.retries = 0;
  }

  connect() {
    this.eventSource = new EventSource(this.url);
    
    this.eventSource.onopen = () => {
      this.retries = 0;
      this.onopen?.();
    };
    
    this.eventSource.onerror = (error) => {
      this.onerror?.(error);
      
      if (this.eventSource.readyState === EventSource.CLOSED) {
        this.scheduleReconnect();
      }
    };
  }

  scheduleReconnect() {
    if (this.retries >= this.maxRetries) {
      this.onmaxretries?.();
      return;
    }

    const delay = Math.min(
      this.baseDelay * Math.pow(2, this.retries),
      this.maxDelay
    );
    
    this.retries++;
    console.log(`Reconnecting in ${delay}ms (attempt ${this.retries})`);
    
    setTimeout(() => this.connect(), delay);
  }

  close() {
    this.eventSource?.close();
  }
}
```

---

## Best Practices

### 1. Request Notification Permission

```javascript
// Request permission on user interaction
async function requestNotificationPermission() {
  if (!('Notification' in window)) {
    console.log('Notifications not supported');
    return false;
  }
  
  if (Notification.permission === 'granted') {
    return true;
  }
  
  if (Notification.permission !== 'denied') {
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  }
  
  return false;
}
```

### 2. Handle Visibility Changes

```javascript
// Pause/resume connection based on page visibility
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    // Optionally disconnect to save resources
    // alertService.disconnect();
  } else {
    // Reconnect if disconnected
    if (!alertService.isConnected) {
      alertService.connect();
    }
  }
});
```

### 3. Fetch Missed Alerts

```javascript
// Fetch alerts that may have been missed during disconnection
alertService.on('connected', async () => {
  const lastAlertTime = getLastAlertTime();
  const recentAlerts = await alertService.fetchRecentAlerts(50);
  
  const missedAlerts = recentAlerts.filter(
    alert => new Date(alert.created_at) > lastAlertTime
  );
  
  missedAlerts.forEach(alert => {
    handleAlert(alert);
  });
});
```

### 4. Debounce Reconnection

```javascript
// Prevent rapid reconnection attempts
const debouncedReconnect = debounce(() => {
  alertService.connect();
}, 1000);
```

### 5. Clean Up on Unmount

```javascript
// React cleanup
useEffect(() => {
  alertService.connect();
  
  return () => {
    alertService.disconnect();
  };
}, []);
```

---

## Debugging

### Browser DevTools

1. **Network Tab**: Look for SSE connections (EventSource)
2. **Console**: Check for connection/error logs
3. **Event Listener**: View incoming events in real-time

### Test Connection

```javascript
// Quick test in browser console
const testSource = new EventSource('/sse/alerts');
testSource.onopen = () => console.log('Connected');
testSource.onmessage = (e) => console.log('Message:', e.data);
testSource.onerror = (e) => console.log('Error:', e);

// Send test alert (from another terminal/tab)
// POST /sse/alerts/test
```

---

## See Also

- [FRONTEND_API_GUIDE.md](FRONTEND_API_GUIDE.md) - Complete API integration guide
- [AUTH_FLOW.md](AUTH_FLOW.md) - JWT authentication
- [ERROR_CODES.md](ERROR_CODES.md) - Error handling reference
