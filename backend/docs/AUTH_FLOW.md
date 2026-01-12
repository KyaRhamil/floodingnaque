# JWT Authentication Flow Guide

This document describes the JWT (JSON Web Token) authentication flow for the Floodingnaque API, specifically designed for frontend integration.

## Table of Contents

1. [Overview](#overview)
2. [Token Types](#token-types)
3. [Authentication Endpoints](#authentication-endpoints)
4. [Token Lifecycle](#token-lifecycle)
5. [Token Refresh Flow](#token-refresh-flow)
6. [Frontend Implementation](#frontend-implementation)
7. [Security Best Practices](#security-best-practices)

---

## Overview

The Floodingnaque API uses a dual-token authentication system:

- **Access Token**: Short-lived token (15 minutes) for API requests
- **Refresh Token**: Long-lived token (7 days) for obtaining new access tokens

```
┌─────────────────────────────────────────────────────────────────┐
│                    Authentication Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐     Login      ┌─────────┐                        │
│  │ Frontend│───────────────▶│   API   │                        │
│  │         │◀───────────────│         │                        │
│  └─────────┘  access_token  └─────────┘                        │
│               refresh_token                                     │
│                                                                 │
│  ┌─────────┐   API Request  ┌─────────┐                        │
│  │ Frontend│───────────────▶│   API   │                        │
│  │         │  + Bearer token│         │                        │
│  └─────────┘◀───────────────└─────────┘                        │
│                  Response                                       │
│                                                                 │
│  ┌─────────┐ Token Expired  ┌─────────┐                        │
│  │ Frontend│───────────────▶│   API   │                        │
│  │         │ refresh_token  │         │                        │
│  └─────────┘◀───────────────└─────────┘                        │
│              new access_token                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Token Types

### Access Token

| Property | Value |
|----------|-------|
| Type | JWT (HS256) |
| Lifetime | 15 minutes |
| Storage | Memory or secure storage |
| Purpose | API request authorization |

**Payload Structure:**
```json
{
  "sub": "123",           // User ID
  "email": "user@example.com",
  "role": "user",         // user/admin/operator
  "type": "access",
  "iat": 1705311600,      // Issued at (Unix timestamp)
  "exp": 1705312500,      // Expires (Unix timestamp)
  "jti": "unique-token-id"
}
```

### Refresh Token

| Property | Value |
|----------|-------|
| Type | JWT (HS256) |
| Lifetime | 7 days |
| Storage | HttpOnly cookie or secure storage |
| Purpose | Obtain new access tokens |

**Payload Structure:**
```json
{
  "sub": "123",           // User ID
  "type": "refresh",
  "iat": 1705311600,
  "exp": 1705916400,      // 7 days from issue
  "jti": "unique-refresh-id"
}
```

---

## Authentication Endpoints

### 1. User Registration

**Endpoint:** `POST /api/users/register`

**Rate Limit:** 5 per hour

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!@#",
  "full_name": "John Doe",
  "phone_number": "+639123456789"
}
```

**Password Requirements:**
- Minimum 12 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)

**Response (201 Created):**
```json
{
  "success": true,
  "message": "User registered successfully",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "John Doe",
    "role": "user",
    "is_active": true
  },
  "request_id": "abc123"
}
```

**Error Responses:**
- `400`: Invalid email format or weak password
- `409`: Email already exists

---

### 2. User Login

**Endpoint:** `POST /api/users/login`

**Rate Limit:** 10 per minute

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!@#"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Login successful",
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 900,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "John Doe",
    "role": "user"
  },
  "request_id": "abc123"
}
```

**Error Responses:**
- `401`: Invalid credentials
- `403`: Account disabled
- `423`: Account locked (too many failed attempts)

---

### 3. Token Refresh

**Endpoint:** `POST /api/users/refresh`

**Rate Limit:** 30 per hour

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 900,
  "request_id": "abc123"
}
```

**Error Responses:**
- `400`: Missing refresh token
- `401`: Invalid, expired, or revoked refresh token

---

### 4. Logout

**Endpoint:** `POST /api/users/logout`

**Rate Limit:** 30 per hour

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Logged out successfully",
  "request_id": "abc123"
}
```

**Note:** Logout invalidates the refresh token server-side. Even if the access token is expired, logout will succeed.

---

### 5. Get Current User

**Endpoint:** `GET /api/users/me`

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "success": true,
  "user": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "John Doe",
    "role": "user",
    "is_active": true,
    "is_verified": true,
    "last_login_at": "2024-01-15T10:30:00Z"
  },
  "request_id": "abc123"
}
```

---

### 6. Password Reset Request

**Endpoint:** `POST /api/users/password-reset/request`

**Rate Limit:** 3 per hour

**Request:**
```json
{
  "email": "user@example.com"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "If an account exists with this email, a password reset link has been sent",
  "request_id": "abc123"
}
```

**Note:** Always returns success to prevent email enumeration attacks.

---

### 7. Password Reset Confirm

**Endpoint:** `POST /api/users/password-reset/confirm`

**Rate Limit:** 5 per hour

**Request:**
```json
{
  "email": "user@example.com",
  "token": "reset_token_from_email",
  "new_password": "NewSecurePassword123!@#"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Password reset successful. Please login with your new password.",
  "request_id": "abc123"
}
```

**Note:** After password reset, all existing refresh tokens are invalidated.

---

## Token Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Token Lifecycle                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Login                                                               │
│     │                                                                 │
│     ▼                                                                 │
│   ┌─────────────────────────────────────────────────────┐            │
│   │ Access Token (15 min)    Refresh Token (7 days)     │            │
│   └────────────────┬────────────────┬───────────────────┘            │
│                    │                │                                 │
│                    ▼                │                                 │
│              Use for API            │                                 │
│               requests              │                                 │
│                    │                │                                 │
│                    ▼                │                                 │
│            Access Token             │                                 │
│              Expires                │                                 │
│                    │                │                                 │
│                    ▼                │                                 │
│           ┌────────┴────────┐       │                                 │
│           │  Still have     │       │                                 │
│           │ refresh token?  │       │                                 │
│           └────────┬────────┘       │                                 │
│              YES   │                │                                 │
│                    ▼                │                                 │
│           POST /refresh ◀───────────┘                                 │
│                    │                                                  │
│                    ▼                                                  │
│           New Access Token                                            │
│           (continue using)                                            │
│                    │                                                  │
│                    ▼                                                  │
│           Refresh Token                                               │
│             Expires?                                                  │
│                    │                                                  │
│                YES │                                                  │
│                    ▼                                                  │
│           Redirect to Login                                           │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Token Refresh Flow

### When to Refresh

1. **Proactive Refresh**: Refresh before token expires (recommended)
2. **Reactive Refresh**: Refresh after receiving 401 error

### Proactive Refresh Strategy

```javascript
// Check token expiration 1 minute before actual expiry
const TOKEN_REFRESH_THRESHOLD = 60; // seconds

function isTokenExpiringSoon(token) {
  const payload = decodeToken(token);
  const expiresAt = payload.exp * 1000; // Convert to milliseconds
  const now = Date.now();
  
  return (expiresAt - now) < (TOKEN_REFRESH_THRESHOLD * 1000);
}

// Check before each API request
async function makeRequest(config) {
  const accessToken = getAccessToken();
  
  if (accessToken && isTokenExpiringSoon(accessToken)) {
    await refreshAccessToken();
  }
  
  return axios(config);
}
```

### Reactive Refresh Strategy

```javascript
// Handle 401 responses
api.interceptors.response.use(
  response => response,
  async error => {
    const originalRequest = error.config;
    
    // If 401 and not already retried
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        await refreshAccessToken();
        // Retry original request with new token
        return api(originalRequest);
      } catch (refreshError) {
        // Refresh failed, redirect to login
        clearTokens();
        redirectToLogin();
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(error);
  }
);
```

### Handling Concurrent Requests

When multiple requests fail with 401 simultaneously:

```javascript
let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach(prom => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

api.interceptors.response.use(
  response => response,
  async error => {
    const originalRequest = error.config;
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        // Queue this request
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then(token => {
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return api(originalRequest);
        });
      }
      
      originalRequest._retry = true;
      isRefreshing = true;
      
      try {
        const newToken = await refreshAccessToken();
        processQueue(null, newToken);
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError, null);
        clearTokens();
        redirectToLogin();
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }
    
    return Promise.reject(error);
  }
);
```

---

## Frontend Implementation

### Complete Authentication Service

```javascript
// services/auth.js
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

class AuthService {
  constructor() {
    this.accessToken = null;
    this.refreshToken = null;
    this.user = null;
  }

  // Initialize from stored tokens
  init() {
    this.accessToken = localStorage.getItem('accessToken');
    this.refreshToken = localStorage.getItem('refreshToken');
    const userData = localStorage.getItem('user');
    this.user = userData ? JSON.parse(userData) : null;
  }

  // Register new user
  async register(email, password, fullName, phoneNumber) {
    const response = await axios.post(`${API_URL}/users/register`, {
      email,
      password,
      full_name: fullName,
      phone_number: phoneNumber
    });
    return response.data;
  }

  // Login user
  async login(email, password) {
    const response = await axios.post(`${API_URL}/users/login`, {
      email,
      password
    });

    if (response.data.success) {
      this.setTokens(
        response.data.access_token,
        response.data.refresh_token
      );
      this.user = response.data.user;
      localStorage.setItem('user', JSON.stringify(this.user));
    }

    return response.data;
  }

  // Refresh access token
  async refreshAccessToken() {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await axios.post(`${API_URL}/users/refresh`, {
      refresh_token: this.refreshToken
    });

    if (response.data.success) {
      this.accessToken = response.data.access_token;
      localStorage.setItem('accessToken', this.accessToken);
    }

    return response.data.access_token;
  }

  // Logout user
  async logout() {
    try {
      await axios.post(`${API_URL}/users/logout`, null, {
        headers: {
          Authorization: `Bearer ${this.accessToken}`
        }
      });
    } catch (error) {
      // Continue with local cleanup even if API call fails
      console.warn('Logout API call failed:', error);
    }

    this.clearTokens();
  }

  // Get current user
  async getCurrentUser() {
    const response = await axios.get(`${API_URL}/users/me`, {
      headers: {
        Authorization: `Bearer ${this.accessToken}`
      }
    });

    if (response.data.success) {
      this.user = response.data.user;
      localStorage.setItem('user', JSON.stringify(this.user));
    }

    return response.data.user;
  }

  // Request password reset
  async requestPasswordReset(email) {
    const response = await axios.post(`${API_URL}/users/password-reset/request`, {
      email
    });
    return response.data;
  }

  // Confirm password reset
  async confirmPasswordReset(email, token, newPassword) {
    const response = await axios.post(`${API_URL}/users/password-reset/confirm`, {
      email,
      token,
      new_password: newPassword
    });
    return response.data;
  }

  // Helper methods
  setTokens(accessToken, refreshToken) {
    this.accessToken = accessToken;
    this.refreshToken = refreshToken;
    localStorage.setItem('accessToken', accessToken);
    localStorage.setItem('refreshToken', refreshToken);
  }

  clearTokens() {
    this.accessToken = null;
    this.refreshToken = null;
    this.user = null;
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('user');
  }

  isAuthenticated() {
    return !!this.accessToken;
  }

  getAccessToken() {
    return this.accessToken;
  }

  getUser() {
    return this.user;
  }
}

export default new AuthService();
```

### React Hook Example

```javascript
// hooks/useAuth.js
import { useState, useEffect, useContext, createContext } from 'react';
import authService from '../services/auth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    authService.init();
    if (authService.isAuthenticated()) {
      setUser(authService.getUser());
    }
    setLoading(false);
  }, []);

  const login = async (email, password) => {
    const result = await authService.login(email, password);
    setUser(result.user);
    return result;
  };

  const logout = async () => {
    await authService.logout();
    setUser(null);
  };

  const refreshToken = async () => {
    return await authService.refreshAccessToken();
  };

  const value = {
    user,
    loading,
    isAuthenticated: !!user,
    login,
    logout,
    refreshToken,
    register: authService.register.bind(authService)
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
```

---

## Security Best Practices

### 1. Token Storage

| Storage Option | Security Level | Use Case |
|----------------|----------------|----------|
| Memory (state) | Most secure | SPAs, tab-only persistence |
| HttpOnly Cookie | Very secure | If backend sets cookie |
| localStorage | Less secure | If memory not feasible |
| sessionStorage | Moderate | Tab-specific storage |

**Recommendation:** Use memory for access token, secure storage for refresh token.

### 2. HTTPS Only

Always use HTTPS in production to prevent token interception.

### 3. Token Validation

```javascript
// Validate token before use
function isValidToken(token) {
  if (!token) return false;
  
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.exp * 1000 > Date.now();
  } catch {
    return false;
  }
}
```

### 4. Account Lockout Handling

```javascript
// Handle locked accounts
if (error.response?.status === 423) {
  const retryAfter = error.response.data.retry_after;
  showLockoutMessage(retryAfter);
  // Optionally disable login form
  disableLoginForm(retryAfter);
}
```

### 5. Secure Logout

```javascript
// Clean up on logout
async function secureLogout() {
  // 1. Call logout API
  await authService.logout();
  
  // 2. Clear all stored data
  localStorage.clear();
  sessionStorage.clear();
  
  // 3. Clear in-memory state
  // (handled by AuthService)
  
  // 4. Redirect to login
  window.location.href = '/login';
}
```

### 6. XSS Prevention

- Never store tokens in cookies without HttpOnly flag
- Sanitize all user input
- Use Content Security Policy headers

---

## Troubleshooting

### Common Issues

1. **Token refresh loop**
   - Check that refresh token is valid and not expired
   - Ensure refresh endpoint isn't rate limited

2. **401 after refresh**
   - Verify new token is being stored correctly
   - Check Authorization header format: `Bearer <token>`

3. **CORS issues**
   - Ensure backend allows your origin
   - Check that Authorization header is in allowed headers

4. **Account keeps locking**
   - Implement proper error handling to stop retries
   - Check for automated processes causing failed logins

---

## See Also

- [ERROR_CODES.md](ERROR_CODES.md) - Complete error code reference
- [FRONTEND_API_GUIDE.md](FRONTEND_API_GUIDE.md) - Full API integration guide
- [REALTIME_GUIDE.md](REALTIME_GUIDE.md) - SSE subscription guide
