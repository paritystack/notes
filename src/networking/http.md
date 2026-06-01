# HTTP/HTTPS

## Overview

HTTP (HyperText Transfer Protocol) is the foundation of data communication on the web. HTTPS adds [TLS/SSL](tls_ssl.md) encryption for secure communication. [HTTP/2](http2.md) is the binary successor that multiplexes requests; [HTTP/3](quic.md) runs over QUIC; [WebSocket](websocket.md) upgrades an HTTP connection to a full-duplex channel; all run over [TCP](tcp.md).

## HTTP Basics

### Request-Response Model

```
Client                          Server

    HTTP Request               ->

  <-          HTTP Response

```

### HTTP Methods

| Method | Purpose | Idempotent | Safe |
|--------|---------|-----------|------|
| **GET** | Retrieve resource | Yes | Yes |
| **POST** | Create resource | No | No |
| **PUT** | Replace resource | Yes | No |
| **PATCH** | Partial update | No | No |
| **DELETE** | Remove resource | Yes | No |
| **HEAD** | Like GET, no body | Yes | Yes |
| **OPTIONS** | Describe options | Yes | Yes |

### Status Codes

| Code | Meaning | Examples |
|------|---------|----------|
| **1xx** | Informational | 100 Continue |
| **2xx** | Success | 200 OK, 201 Created |
| **3xx** | Redirection | 301 Moved, 304 Not Modified |
| **4xx** | Client Error | 400 Bad Request, 404 Not Found |
| **5xx** | Server Error | 500 Server Error, 503 Unavailable |

### Headers

**Request Headers**:
```
Host: example.com
User-Agent: Mozilla/5.0
Accept: application/json
Authorization: Bearer token123
Cookie: session=abc123
Content-Type: application/json
```

**Response Headers**:
```
Content-Type: application/json
Content-Length: 256
Set-Cookie: session=def456
Cache-Control: max-age=3600
ETag: "12345abc"
```

## HTTP Versions

| Version | Released | Features |
|---------|----------|----------|
| **HTTP/1.1** | 1997 | Keep-alive, chunked transfer |
| **HTTP/2** | 2015 | Multiplexing, server push, binary |
| **HTTP/3** | 2022 | QUIC protocol, faster |

## REST API Design

### Resource-Oriented
```
 GET /users           - List users
 POST /users          - Create user
 GET /users/123       - Get user 123
 PUT /users/123       - Update user 123
 DELETE /users/123    - Delete user 123

 GET /getUser?id=123  - Procedural (bad)
 POST /createUser     - Procedural (bad)
```

### Request/Response Example

```python
# Request
GET /users/123 HTTP/1.1
Host: api.example.com
Authorization: Bearer token

# Response
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 156

{
  "id": 123,
  "name": "John",
  "email": "john@example.com"
}
```

## HTTPS (Secure HTTP)

Adds TLS encryption on top of HTTP:

```
HTTP over TLS = HTTPS
```

### Benefits
- **Encryption**: Data unreadable to eavesdroppers
- **Authentication**: Verify server identity
- **Integrity**: Detect tampering

### Certificate Process
```
1. Generate private/public key pair
2. Request certificate from CA
3. CA verifies and signs certificate
4. Browser verifies signature with CA's public key
```

## Caching

### Cache Headers
```
Cache-Control: max-age=3600     # Cache for 1 hour
Cache-Control: no-cache         # Validate before use
Cache-Control: no-store         # Don't cache
Cache-Control: private          # Only browser cache
Cache-Control: public           # Any cache can store
ETag: "12345"                   # Resource version
```

### Conditional Requests
```
If-None-Match: "12345"
-> Returns 304 Not Modified if unchanged
```

## Authentication

### Basic Auth
```
Authorization: Basic base64(username:password)
```

### Bearer Token
```
Authorization: Bearer eyJhbGc...
```

### OAuth 2.0
Multi-step authorization flow for 3rd party apps

## CORS (Cross-Origin Resource Sharing)

Enable browser to access cross-origin APIs:

```
Server Response:
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST
Access-Control-Allow-Headers: Content-Type
```

## Common Issues

### 404 Not Found
Resource doesn't exist

### 401 Unauthorized
Missing/invalid authentication

### 403 Forbidden
Authenticated but not allowed

### 429 Too Many Requests
Rate limit exceeded

### 503 Service Unavailable
Server temporarily down

## Best Practices

### 1. Use Appropriate Methods
```
 GET for reading (no side effects)
 POST for creating
 PUT for full replacement
 PATCH for partial update
```

### 2. Meaningful Status Codes
```
 200 OK for success
 201 Created for new resource
 204 No Content for delete
 200 for everything (bad)
```

### 3. Versioning
```
/api/v1/users
/api/v2/users
```

### 4. Error Responses
```json
{
  "error": "Invalid input",
  "details": {
    "email": "Email format invalid"
  }
}
```

## ELI10

HTTP is like sending letters:
- **GET**: "What's the address of 123 Main St?"
- **POST**: "Please add my address to your system"
- **PUT**: "Update my address to..."
- **DELETE**: "Remove my address"

The server sends back a number (status code):
- **200**: "Got it, here's what you asked for!"
- **404**: "Can't find that address"
- **500**: "I have a problem..."

HTTPS adds a sealed envelope so only the right person can read it!

## Further Resources

- [MDN HTTP Guide](https://developer.mozilla.org/en-US/docs/Web/HTTP)
## Where this connects

- [HTTP/2](http2.md) — binary successor that multiplexes requests over a single TCP connection
- [QUIC](quic.md) — HTTP/3 runs HTTP semantics over QUIC instead of TCP
- [TLS/SSL](tls_ssl.md) — HTTPS = HTTP over TLS; required for HTTP/2 in browsers
- [WebSocket](websocket.md) — upgrades an HTTP/1.1 connection to full-duplex
- [gRPC](grpc.md) — uses HTTP/2 as transport for RPC calls

- [HTTP Status Codes](https://httpstat.us/)
- [REST API Best Practices](https://restfulapi.net/)
