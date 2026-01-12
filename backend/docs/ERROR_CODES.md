# Floodingnaque API Error Codes Reference

This document provides a comprehensive reference of all error codes returned by the Floodingnaque API for frontend error handling.

## Response Format

All API errors follow RFC 7807 Problem Details format:

```json
{
  "success": false,
  "error": {
    "type": "/errors/validation",
    "title": "Validation Failed",
    "status": 400,
    "detail": "Temperature must be between 173.15K and 333.15K",
    "code": "ValidationError",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "errors": [
      {
        "field": "temperature",
        "message": "Value out of range",
        "code": "invalid_value"
      }
    ]
  },
  "request_id": "abc123"
}
```

## HTTP Status Codes

| Status | Category | Description |
|--------|----------|-------------|
| 200 | Success | Request completed successfully |
| 201 | Success | Resource created successfully |
| 202 | Success | Request accepted for processing |
| 400 | Client Error | Bad request - invalid input |
| 401 | Client Error | Authentication required |
| 403 | Client Error | Access forbidden |
| 404 | Client Error | Resource not found |
| 409 | Client Error | Resource conflict |
| 422 | Client Error | Unprocessable entity |
| 423 | Client Error | Account locked |
| 429 | Client Error | Rate limit exceeded |
| 500 | Server Error | Internal server error |
| 502 | Server Error | External service error |
| 503 | Server Error | Service unavailable |

---

## Error Code Reference

### Client Errors (4xx)

#### ValidationError
**Status Code:** 400

Indicates invalid input data that doesn't meet validation requirements.

```json
{
  "error": {
    "code": "ValidationError",
    "title": "Validation Failed",
    "detail": "Humidity must be between 0 and 100"
  }
}
```

**Common Causes:**
- Missing required fields
- Values out of acceptable range
- Invalid date format
- Malformed JSON

**Frontend Handling:**
```javascript
if (error.code === 'ValidationError') {
  // Display validation errors to user
  if (error.errors) {
    error.errors.forEach(err => {
      showFieldError(err.field, err.message);
    });
  } else {
    showToast(error.detail, 'warning');
  }
}
```

---

#### BadRequestError
**Status Code:** 400

Generic bad request error for malformed requests.

```json
{
  "error": {
    "code": "BadRequestError",
    "title": "Bad Request",
    "detail": "Invalid request format"
  }
}
```

**Frontend Handling:**
```javascript
if (error.code === 'BadRequestError') {
  showToast('Request could not be processed. Please check your input.', 'error');
}
```

---

#### InvalidJSON
**Status Code:** 400

JSON parsing failed in request body.

```json
{
  "error": {
    "code": "InvalidJSON",
    "title": "Bad Request",
    "detail": "Please ensure your JSON is properly formatted"
  }
}
```

---

#### UnauthorizedError / AuthenticationError
**Status Code:** 401

Authentication is required but not provided or invalid.

```json
{
  "error": {
    "code": "UnauthorizedError",
    "title": "Authentication Required",
    "detail": "Invalid or expired access token"
  }
}
```

**Frontend Handling:**
```javascript
if (error.status === 401) {
  // Try to refresh token
  const newToken = await refreshAccessToken();
  if (newToken) {
    // Retry original request
    return retryRequest(originalRequest);
  } else {
    // Redirect to login
    redirectToLogin();
  }
}
```

---

#### InvalidToken
**Status Code:** 401

JWT token validation failed.

```json
{
  "error": {
    "code": "InvalidToken",
    "title": "Authentication Required",
    "detail": "Token has expired"
  }
}
```

**Possible Detail Messages:**
- "Token has expired"
- "Invalid token format"
- "Token has been revoked"
- "Not a refresh token"

---

#### InvalidCredentials
**Status Code:** 401

Login credentials are incorrect.

```json
{
  "error": {
    "code": "InvalidCredentials",
    "title": "Authentication Required",
    "detail": "Invalid email or password"
  }
}
```

**Note:** This generic message prevents email enumeration attacks.

---

#### ForbiddenError / AuthorizationError
**Status Code:** 403

User is authenticated but lacks permission.

```json
{
  "error": {
    "code": "ForbiddenError",
    "title": "Access Denied",
    "detail": "Insufficient permissions to access this resource"
  }
}
```

**Frontend Handling:**
```javascript
if (error.status === 403) {
  showToast('You do not have permission to perform this action', 'error');
  // Optionally redirect to appropriate page
}
```

---

#### AccountDisabled
**Status Code:** 403

User account has been disabled by administrator.

```json
{
  "error": {
    "code": "AccountDisabled",
    "title": "Access Denied",
    "detail": "Account is disabled"
  }
}
```

---

#### NotFoundError
**Status Code:** 404

Requested resource does not exist.

```json
{
  "error": {
    "code": "NotFoundError",
    "title": "Resource Not Found",
    "detail": "Weather data with id 123 not found"
  }
}
```

---

#### ModelNotFound
**Status Code:** 404

ML model file is not available.

```json
{
  "error": {
    "code": "ModelNotFound",
    "title": "Resource Not Found",
    "detail": "Model version v2.0 not found"
  }
}
```

---

#### ConflictError / EmailExists
**Status Code:** 409

Resource already exists or conflicts with existing data.

```json
{
  "error": {
    "code": "EmailExists",
    "title": "Resource Conflict",
    "detail": "An account with this email already exists"
  }
}
```

---

#### AccountLocked
**Status Code:** 423

Account is temporarily locked due to failed login attempts.

```json
{
  "success": false,
  "error": "AccountLocked",
  "message": "Account is locked. Try again in 15 minutes",
  "retry_after": 900
}
```

**Frontend Handling:**
```javascript
if (error.status === 423) {
  const minutes = Math.ceil(error.retry_after / 60);
  showToast(`Account locked. Please try again in ${minutes} minutes.`, 'warning');
  disableLoginForm(error.retry_after);
}
```

---

#### RateLimitExceededError / RateLimitError
**Status Code:** 429

Too many requests in a given time period.

```json
{
  "error": {
    "code": "RateLimitExceededError",
    "title": "Rate Limit Exceeded",
    "detail": "Rate limit exceeded. Please retry after 60 seconds",
    "retry_after_seconds": 60
  }
}
```

**Response Headers:**
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

**Frontend Handling:**
```javascript
if (error.status === 429) {
  const retryAfter = error.retry_after_seconds || 60;
  showToast(`Too many requests. Please wait ${retryAfter} seconds.`, 'warning');
  
  // Implement exponential backoff for retries
  await delay(retryAfter * 1000);
  return retryRequest(originalRequest);
}
```

---

### Server Errors (5xx)

#### InternalServerError
**Status Code:** 500

Unexpected server error occurred.

```json
{
  "error": {
    "code": "InternalServerError",
    "title": "Internal Server Error",
    "detail": "An unexpected error occurred",
    "error_id": "abc123"
  }
}
```

**Frontend Handling:**
```javascript
if (error.status >= 500) {
  // Log error for debugging
  console.error('Server error:', error.error_id);
  
  showToast('Server error. Please try again later.', 'error');
  
  // Report to monitoring (if configured)
  reportError(error);
}
```

---

#### ModelError
**Status Code:** 500

ML model processing failed.

```json
{
  "error": {
    "code": "ModelError",
    "title": "Model Processing Error",
    "detail": "Model prediction failed"
  }
}
```

---

#### DatabaseError
**Status Code:** 500

Database operation failed.

```json
{
  "error": {
    "code": "DatabaseError",
    "title": "Database Error",
    "detail": "Database operation failed"
  }
}
```

---

#### ExternalServiceError / ExternalAPIError
**Status Code:** 502

External API call failed.

```json
{
  "error": {
    "code": "ExternalServiceError",
    "title": "External Service Error",
    "detail": "Weather API temporarily unavailable",
    "service_name": "OpenWeatherMap",
    "retryable": true
  }
}
```

**Frontend Handling:**
```javascript
if (error.code === 'ExternalServiceError') {
  if (error.retryable) {
    showToast('External service unavailable. Retrying...', 'info');
    await delay(5000);
    return retryRequest(originalRequest, { maxRetries: 3 });
  }
}
```

---

#### ServiceUnavailableError
**Status Code:** 503

Service is temporarily unavailable.

```json
{
  "error": {
    "code": "ServiceUnavailableError",
    "title": "Service Unavailable",
    "detail": "Service temporarily unavailable",
    "retry_after_seconds": 30
  }
}
```

---

## Complete Error Code Table

| Error Code | HTTP Status | Category | Retryable |
|------------|-------------|----------|-----------|
| ValidationError | 400 | Client | No |
| BadRequestError | 400 | Client | No |
| InvalidJSON | 400 | Client | No |
| NoInput | 400 | Client | No |
| UnauthorizedError | 401 | Client | No* |
| AuthenticationError | 401 | Client | No* |
| InvalidToken | 401 | Client | No* |
| InvalidCredentials | 401 | Client | No |
| TokenExpired | 401 | Client | Yes** |
| ForbiddenError | 403 | Client | No |
| AuthorizationError | 403 | Client | No |
| AccountDisabled | 403 | Client | No |
| NotFoundError | 404 | Client | No |
| ModelNotFound | 404 | Client | No |
| ConflictError | 409 | Client | No |
| EmailExists | 409 | Client | No |
| AccountLocked | 423 | Client | Yes*** |
| RateLimitExceededError | 429 | Client | Yes |
| RateLimitError | 429 | Client | Yes |
| InternalServerError | 500 | Server | Yes |
| ModelError | 500 | Server | Yes |
| DatabaseError | 500 | Server | Yes |
| PredictionFailed | 500 | Server | Yes |
| ConfigurationError | 500 | Server | No |
| ExternalServiceError | 502 | Server | Yes |
| ExternalAPIError | 502 | Server | Yes |
| ServiceUnavailableError | 503 | Server | Yes |

\* Retry after refreshing token  
\** Use refresh token to get new access token  
\*** Retry after `retry_after` seconds

---

## Frontend Error Handling Best Practices

### 1. Global Error Handler

```javascript
// api/errorHandler.js
export async function handleApiError(error, originalRequest) {
  const { status, code, detail, retry_after } = error;
  
  switch (status) {
    case 401:
      if (code === 'TokenExpired' || code === 'InvalidToken') {
        return await handleTokenRefresh(originalRequest);
      }
      return redirectToLogin();
      
    case 403:
      showToast('Access denied', 'error');
      return null;
      
    case 404:
      showToast('Resource not found', 'warning');
      return null;
      
    case 423:
      showToast(`Account locked. Try again in ${Math.ceil(retry_after/60)} minutes`, 'warning');
      return null;
      
    case 429:
      showToast('Rate limited. Please wait...', 'warning');
      await delay((retry_after || 60) * 1000);
      return retryRequest(originalRequest);
      
    case 500:
    case 502:
    case 503:
      showToast('Server error. Please try again.', 'error');
      return null;
      
    default:
      showToast(detail || 'An error occurred', 'error');
      return null;
  }
}
```

### 2. Request Interceptor with Token Refresh

```javascript
// api/client.js
let isRefreshing = false;
let failedQueue = [];

api.interceptors.response.use(
  response => response,
  async error => {
    const originalRequest = error.config;
    
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
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
        const newToken = await refreshToken();
        processQueue(null, newToken);
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError, null);
        redirectToLogin();
      } finally {
        isRefreshing = false;
      }
    }
    
    return Promise.reject(error);
  }
);
```

### 3. Form Validation Error Display

```javascript
// components/FormErrorDisplay.jsx
function FormErrorDisplay({ errors }) {
  if (!errors?.length) return null;
  
  return (
    <div className="validation-errors">
      {errors.map((err, idx) => (
        <div key={idx} className="error-item">
          <span className="field">{err.field}:</span>
          <span className="message">{err.message}</span>
        </div>
      ))}
    </div>
  );
}
```

---

## Rate Limiting Information

### Default Rate Limits

| Endpoint Category | Limit | Window |
|-------------------|-------|--------|
| Authentication (login) | 10 | 1 minute |
| Authentication (register) | 5 | 1 hour |
| Token refresh | 30 | 1 hour |
| Prediction | 60 | 1 hour |
| Data queries | 100 | 1 minute |
| SSE stream | 5 | 1 minute |
| Webhooks | 10 | 1 minute |

### Rate Limit Headers

All responses include rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312200
```

---

## Debugging Tips

1. **Always check `request_id`**: Include this in bug reports for tracing
2. **Check `error.errors` array**: Contains field-specific validation errors
3. **Monitor `error_id`**: For server errors, this helps backend debugging
4. **Use retry headers**: `retry_after_seconds` tells you when to retry

## See Also

- [AUTH_FLOW.md](AUTH_FLOW.md) - JWT authentication flow
- [FRONTEND_API_GUIDE.md](FRONTEND_API_GUIDE.md) - Complete API integration guide
- [REALTIME_GUIDE.md](REALTIME_GUIDE.md) - SSE real-time subscriptions
