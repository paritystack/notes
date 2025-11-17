# API Design Guide

## Table of Contents

- [Introduction](#introduction)
- [RESTful Principles Deep Dive](#restful-principles-deep-dive)
- [Resource Naming Conventions](#resource-naming-conventions)
- [HTTP Methods Usage](#http-methods-usage)
- [HTTP Status Codes](#http-status-codes)
- [API Versioning Strategies](#api-versioning-strategies)
- [Pagination Strategies](#pagination-strategies)
- [Filtering and Sorting](#filtering-and-sorting)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Authentication and Authorization](#authentication-and-authorization)
- [HATEOAS Principles](#hateoas-principles)
- [API Documentation](#api-documentation)
- [Idempotency](#idempotency)
- [Caching Strategies](#caching-strategies)
- [Webhooks vs Polling](#webhooks-vs-polling)
- [GraphQL vs REST Trade-offs](#graphql-vs-rest-trade-offs)
- [gRPC Use Cases](#grpc-use-cases)
- [API Gateway Patterns](#api-gateway-patterns)
- [Backward Compatibility](#backward-compatibility)
- [Deprecation Strategies](#deprecation-strategies)
- [Best Practices Summary](#best-practices-summary)

---

## Introduction

API (Application Programming Interface) design is a critical aspect of modern software development. A well-designed API is intuitive, consistent, maintainable, and provides a great developer experience. This guide covers comprehensive best practices for designing robust, scalable, and user-friendly APIs.

### What Makes a Good API?

**1. Intuitive and Consistent**
- Easy to understand and predict
- Follows established conventions
- Consistent naming and structure

**2. Well-Documented**
- Clear, comprehensive documentation
- Code examples in multiple languages
- Interactive API explorers

**3. Versioned**
- Backward compatible when possible
- Clear versioning strategy
- Deprecation policies

**4. Secure**
- Proper authentication and authorization
- Rate limiting and abuse prevention
- Input validation and sanitization

**5. Performant**
- Efficient data retrieval
- Proper caching mechanisms
- Pagination for large datasets

**6. Developer-Friendly**
- Helpful error messages
- Consistent response formats
- SDKs and libraries

---

## RESTful Principles Deep Dive

REST (Representational State Transfer) is an architectural style for designing networked applications. It relies on a stateless, client-server protocol, typically HTTP.

### Six Constraints of REST

#### 1. Client-Server Architecture

**Principle:** Separation of concerns between client and server.

```
Client (UI/Presentation)  ←→  Server (Data/Business Logic)
```

**Benefits:**
- Independent evolution of client and server
- Improved scalability through separation
- Multiple clients can use the same API

**Example:**
```javascript
// Client (React)
const fetchUser = async (userId) => {
  const response = await fetch(`/api/users/${userId}`);
  return response.json();
};

// Server (Express)
app.get('/api/users/:id', (req, res) => {
  const user = database.getUser(req.params.id);
  res.json(user);
});
```

#### 2. Stateless

**Principle:** Each request contains all information needed to understand and process it.

```
❌ Stateful (Bad):
Request 1: Login user → Server stores session
Request 2: Get profile → Server uses stored session

✅ Stateless (Good):
Request 1: Login → Return token
Request 2: Get profile (with token) → Server validates token
```

**Implementation:**
```javascript
// Client includes authentication in every request
const headers = {
  'Authorization': `Bearer ${accessToken}`,
  'Content-Type': 'application/json'
};

fetch('/api/profile', { headers })
  .then(res => res.json());

// Server validates each request independently
app.get('/api/profile', authenticateToken, (req, res) => {
  // Each request is self-contained
  const user = getUserFromToken(req.token);
  res.json(user);
});
```

**Benefits:**
- Scalability: No server-side session storage
- Reliability: No session state to lose
- Visibility: Complete request information

#### 3. Cacheable

**Principle:** Responses must define themselves as cacheable or non-cacheable.

```http
HTTP/1.1 200 OK
Cache-Control: max-age=3600, public
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
Last-Modified: Wed, 15 Nov 2023 12:00:00 GMT

{
  "id": 123,
  "name": "John Doe"
}
```

**Implementation:**
```javascript
// Server sets cache headers
app.get('/api/users/:id', (req, res) => {
  const user = database.getUser(req.params.id);

  // Cache for 1 hour
  res.set('Cache-Control', 'public, max-age=3600');
  res.set('ETag', generateETag(user));
  res.json(user);
});

// Client respects cache headers
const response = await fetch('/api/users/123');
// Browser automatically caches based on headers
```

**Benefits:**
- Reduced server load
- Improved performance
- Lower latency

#### 4. Uniform Interface

**Principle:** Consistent, standardized interface between client and server.

**Four Sub-Constraints:**

**a) Resource Identification**
```
✅ Good: /users/123
✅ Good: /orders/456/items/789
❌ Bad: /getUserById?id=123
```

**b) Resource Manipulation Through Representations**
```json
GET /users/123
{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com"
}

PUT /users/123
{
  "name": "John Smith",
  "email": "john.smith@example.com"
}
```

**c) Self-Descriptive Messages**
```http
POST /users HTTP/1.1
Content-Type: application/json
Accept: application/json

{
  "name": "Jane Doe",
  "email": "jane@example.com"
}
```

**d) HATEOAS (Hypermedia As The Engine Of Application State)**
```json
{
  "id": 123,
  "name": "John Doe",
  "links": {
    "self": "/users/123",
    "posts": "/users/123/posts",
    "friends": "/users/123/friends"
  }
}
```

#### 5. Layered System

**Principle:** Client cannot tell if connected directly to end server or intermediary.

```
Client → Load Balancer → API Gateway → Cache Layer → Application Server → Database
```

**Benefits:**
- Security through encapsulation
- Load balancing
- Shared caches
- Legacy system encapsulation

**Example:**
```javascript
// Client doesn't know about intermediary layers
fetch('https://api.example.com/users')
  .then(res => res.json());

// Request might go through:
// 1. CDN
// 2. Load balancer
// 3. API gateway
// 4. Application server
// 5. Database server
```

#### 6. Code on Demand (Optional)

**Principle:** Server can extend client functionality by transferring executable code.

**Example:**
```html
<!-- Server sends JavaScript to client -->
<script src="https://api.example.com/widget.js"></script>

<!-- Client executes server-provided code -->
<div id="widget"></div>
```

**Note:** This constraint is optional and rarely used in modern API design.

---

## Resource Naming Conventions

Resource naming is critical for API usability. Good names make APIs intuitive and self-documenting.

### General Principles

#### 1. Use Nouns, Not Verbs

```
✅ Good:
GET    /users
POST   /users
GET    /users/123
PUT    /users/123
DELETE /users/123

❌ Bad:
GET    /getUsers
POST   /createUser
GET    /getUserById/123
PUT    /updateUser/123
DELETE /deleteUser/123
```

#### 2. Use Plural Nouns for Collections

```
✅ Good:
GET /users          # Collection
GET /users/123      # Single resource
GET /orders         # Collection
GET /orders/456     # Single resource

❌ Bad:
GET /user
GET /user/123
```

**Reasoning:** Consistency and predictability. Plural form works for both collection and individual resources.

#### 3. Use Lowercase and Hyphens

```
✅ Good:
/user-profiles
/order-items
/api-keys

❌ Bad:
/userProfiles      (camelCase)
/UserProfiles      (PascalCase)
/user_profiles     (snake_case - acceptable but less common)
```

**Exception:** Some teams prefer snake_case for consistency with backend languages (Python, Ruby).

#### 4. Use Forward Slashes for Hierarchy

```
✅ Good:
/users/123/orders
/users/123/orders/456
/users/123/orders/456/items

Structure represents relationship:
User 123 → Orders → Order 456 → Items
```

#### 5. Avoid Trailing Slashes

```
✅ Good: /users
❌ Bad:  /users/
```

**Implementation:**
```javascript
// Redirect trailing slashes
app.use((req, res, next) => {
  if (req.path.endsWith('/') && req.path.length > 1) {
    const query = req.url.slice(req.path.length);
    res.redirect(301, req.path.slice(0, -1) + query);
  } else {
    next();
  }
});
```

### Resource Relationships

#### 1. Nested Resources (Parent-Child)

```
GET /users/123/posts           # Posts belonging to user 123
GET /users/123/posts/456       # Post 456 of user 123
GET /orders/789/items          # Items in order 789
GET /teams/5/members           # Members of team 5
```

**When to Use:**
- Clear parent-child relationship
- Child rarely exists independently
- Limited nesting depth (2-3 levels max)

#### 2. Independent Resources with Filters

```
GET /posts?userId=123          # Alternative to /users/123/posts
GET /comments?postId=456       # Alternative to /posts/456/comments
GET /items?orderId=789         # Alternative to /orders/789/items
```

**When to Use:**
- Resource can exist independently
- Multiple filtering options needed
- Avoiding deep nesting

#### 3. Many-to-Many Relationships

```
# Users and teams (many-to-many)
GET /users/123/teams           # Teams for user 123
GET /teams/5/members           # Members of team 5

# Alternative: Membership resource
GET /memberships?userId=123    # Memberships for user 123
GET /memberships?teamId=5      # Memberships in team 5
POST /memberships              # Create membership
{
  "userId": 123,
  "teamId": 5,
  "role": "admin"
}
```

### Special Endpoints

#### 1. Actions That Don't Fit CRUD

When an action doesn't map to standard CRUD operations, use a verb as a resource:

```
POST /users/123/activate       # Activate user
POST /orders/456/cancel        # Cancel order
POST /invoices/789/send        # Send invoice
POST /passwords/reset          # Reset password
```

**Alternative Approach (More RESTful):**
```
# Use resource state change
PATCH /users/123
{ "status": "active" }

PATCH /orders/456
{ "status": "cancelled" }

# Use sub-resources
POST /users/123/password-resets
POST /invoices/789/email-deliveries
```

#### 2. Search and Complex Queries

```
GET /search?q=john&type=users
GET /users/search?name=john&age=25
POST /users/search              # Complex search with body
{
  "filters": {
    "age": { "min": 25, "max": 35 },
    "city": "New York",
    "skills": ["JavaScript", "Python"]
  }
}
```

#### 3. Batch Operations

```
POST /users/batch
{
  "operation": "delete",
  "ids": [1, 2, 3, 4, 5]
}

PATCH /users/bulk-update
{
  "updates": [
    { "id": 1, "status": "active" },
    { "id": 2, "status": "inactive" }
  ]
}
```

### Naming Examples by Domain

#### E-commerce

```
/products
/products/123
/products/123/reviews
/products/123/images
/categories
/categories/electronics
/categories/electronics/products
/carts
/carts/456/items
/orders
/orders/789
/orders/789/items
/orders/789/shipments
/customers
/customers/123/addresses
/customers/123/payment-methods
```

#### Social Media

```
/users
/users/123
/users/123/posts
/users/123/followers
/users/123/following
/posts
/posts/456
/posts/456/comments
/posts/456/likes
/posts/456/shares
/hashtags
/hashtags/trending
/messages
/conversations
/conversations/789/messages
```

#### Project Management

```
/projects
/projects/123
/projects/123/tasks
/projects/123/members
/tasks
/tasks/456
/tasks/456/comments
/tasks/456/attachments
/teams
/teams/789
/teams/789/projects
/teams/789/members
/milestones
/timesheets
```

---

## HTTP Methods Usage

HTTP methods (verbs) define the type of operation to perform on a resource.

### Standard CRUD Operations

| Operation | HTTP Method | Example | Idempotent | Safe |
|-----------|-------------|---------|------------|------|
| Create | POST | POST /users | No | No |
| Read | GET | GET /users/123 | Yes | Yes |
| Update (full) | PUT | PUT /users/123 | Yes | No |
| Update (partial) | PATCH | PATCH /users/123 | No | No |
| Delete | DELETE | DELETE /users/123 | Yes | No |

**Idempotent:** Multiple identical requests have the same effect as a single request
**Safe:** Request doesn't modify server state

### GET - Retrieve Resources

**Purpose:** Retrieve resource representation without side effects.

**Characteristics:**
- Safe: Doesn't modify state
- Idempotent: Same result every time
- Cacheable: Responses can be cached

```http
GET /users HTTP/1.1
Host: api.example.com

HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: max-age=3600

[
  { "id": 1, "name": "John" },
  { "id": 2, "name": "Jane" }
]
```

```http
GET /users/123 HTTP/1.1
Host: api.example.com

HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com"
}
```

**Query Parameters:**
```http
GET /users?page=2&limit=20&sort=name&filter=active HTTP/1.1
```

**Implementation:**
```javascript
// Express.js
app.get('/users', async (req, res) => {
  const { page = 1, limit = 20, sort, filter } = req.query;

  const users = await database.getUsers({
    page: parseInt(page),
    limit: parseInt(limit),
    sort,
    filter
  });

  res.json(users);
});

app.get('/users/:id', async (req, res) => {
  const user = await database.getUser(req.params.id);

  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  res.json(user);
});
```

**Best Practices:**
- Never use GET for operations with side effects
- Use query parameters for filtering, sorting, pagination
- Support conditional requests (ETag, If-Modified-Since)
- Implement proper caching headers

### POST - Create Resources

**Purpose:** Create new resources.

**Characteristics:**
- Not safe: Modifies state
- Not idempotent: Multiple requests create multiple resources
- Response often includes Location header

```http
POST /users HTTP/1.1
Host: api.example.com
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com"
}

HTTP/1.1 201 Created
Location: /users/123
Content-Type: application/json

{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com",
  "createdAt": "2023-11-15T10:00:00Z"
}
```

**Implementation:**
```javascript
app.post('/users', async (req, res) => {
  // Validate input
  const errors = validateUser(req.body);
  if (errors.length > 0) {
    return res.status(400).json({ errors });
  }

  // Create user
  const user = await database.createUser(req.body);

  // Return 201 with Location header
  res.status(201)
    .location(`/users/${user.id}`)
    .json(user);
});
```

**Use Cases:**
- Creating new resources
- Submitting forms
- Triggering operations
- Uploading files

**POST vs PUT for Creation:**
```
POST /users              → Server generates ID (e.g., /users/123)
PUT /users/john-doe      → Client specifies ID
```

### PUT - Replace Resources

**Purpose:** Replace entire resource or create if doesn't exist.

**Characteristics:**
- Not safe: Modifies state
- Idempotent: Multiple identical requests have same effect
- Requires full resource representation

```http
PUT /users/123 HTTP/1.1
Host: api.example.com
Content-Type: application/json

{
  "name": "John Smith",
  "email": "john.smith@example.com",
  "age": 30,
  "city": "New York"
}

HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": 123,
  "name": "John Smith",
  "email": "john.smith@example.com",
  "age": 30,
  "city": "New York",
  "updatedAt": "2023-11-15T10:00:00Z"
}
```

**Implementation:**
```javascript
app.put('/users/:id', async (req, res) => {
  const { id } = req.params;

  // Validate full resource
  const errors = validateUser(req.body);
  if (errors.length > 0) {
    return res.status(400).json({ errors });
  }

  // Check if resource exists
  const exists = await database.userExists(id);

  // Replace resource
  const user = await database.replaceUser(id, req.body);

  // Return 200 if updated, 201 if created
  res.status(exists ? 200 : 201).json(user);
});
```

**Important:**
- Client must send complete resource
- Missing fields will be removed
- Use PATCH for partial updates

### PATCH - Partial Update

**Purpose:** Partially modify a resource.

**Characteristics:**
- Not safe: Modifies state
- Not necessarily idempotent (depends on implementation)
- Accepts partial representation

```http
PATCH /users/123 HTTP/1.1
Host: api.example.com
Content-Type: application/json

{
  "email": "new.email@example.com"
}

HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": 123,
  "name": "John Doe",
  "email": "new.email@example.com",
  "age": 30,
  "city": "New York",
  "updatedAt": "2023-11-15T10:00:00Z"
}
```

**JSON Patch (RFC 6902):**
```http
PATCH /users/123 HTTP/1.1
Content-Type: application/json-patch+json

[
  { "op": "replace", "path": "/email", "value": "new@example.com" },
  { "op": "add", "path": "/phone", "value": "+1234567890" },
  { "op": "remove", "path": "/age" }
]
```

**Implementation:**
```javascript
app.patch('/users/:id', async (req, res) => {
  const { id } = req.params;

  // Check if user exists
  const user = await database.getUser(id);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  // Validate partial update
  const errors = validatePartialUser(req.body);
  if (errors.length > 0) {
    return res.status(400).json({ errors });
  }

  // Update specific fields
  const updatedUser = await database.updateUser(id, req.body);

  res.json(updatedUser);
});
```

**PATCH vs PUT:**
```
PUT /users/123          → Must send complete resource
PATCH /users/123        → Send only fields to update
```

### DELETE - Remove Resources

**Purpose:** Delete a resource.

**Characteristics:**
- Not safe: Modifies state
- Idempotent: Deleting same resource multiple times has same effect
- May return 204 No Content or 200 OK with body

```http
DELETE /users/123 HTTP/1.1
Host: api.example.com

HTTP/1.1 204 No Content
```

**With Response Body:**
```http
DELETE /users/123 HTTP/1.1

HTTP/1.1 200 OK
Content-Type: application/json

{
  "message": "User deleted successfully",
  "deletedAt": "2023-11-15T10:00:00Z"
}
```

**Implementation:**
```javascript
app.delete('/users/:id', async (req, res) => {
  const { id } = req.params;

  // Check if user exists
  const user = await database.getUser(id);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  // Delete user
  await database.deleteUser(id);

  // Return 204 No Content or 200 with message
  res.status(204).send();
  // OR
  // res.status(200).json({ message: 'User deleted' });
});
```

**Soft Delete:**
```javascript
app.delete('/users/:id', async (req, res) => {
  await database.updateUser(id, {
    deletedAt: new Date(),
    status: 'deleted'
  });

  res.status(204).send();
});
```

### Other HTTP Methods

#### HEAD - Retrieve Headers Only

```http
HEAD /users/123 HTTP/1.1

HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 256
Last-Modified: Wed, 15 Nov 2023 10:00:00 GMT
```

**Use Cases:**
- Check if resource exists
- Get metadata without downloading body
- Verify cache freshness

#### OPTIONS - Describe Communication Options

```http
OPTIONS /users HTTP/1.1

HTTP/1.1 200 OK
Allow: GET, POST, OPTIONS
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
```

**Use Cases:**
- CORS preflight requests
- Discover supported methods

---

## HTTP Status Codes

Status codes communicate the result of an HTTP request. Proper status codes improve API usability and debugging.

### Status Code Ranges

| Range | Category | Meaning |
|-------|----------|---------|
| 1xx | Informational | Request received, continuing process |
| 2xx | Success | Request successfully received, understood, accepted |
| 3xx | Redirection | Further action needed to complete request |
| 4xx | Client Error | Request contains bad syntax or cannot be fulfilled |
| 5xx | Server Error | Server failed to fulfill valid request |

### 2xx Success

#### 200 OK

**Usage:** General success for GET, PUT, PATCH, DELETE (with body).

```http
GET /users/123
HTTP/1.1 200 OK

{
  "id": 123,
  "name": "John Doe"
}
```

#### 201 Created

**Usage:** Resource successfully created (POST, sometimes PUT).

```http
POST /users
HTTP/1.1 201 Created
Location: /users/123

{
  "id": 123,
  "name": "John Doe"
}
```

**Best Practice:** Include Location header with new resource URL.

#### 202 Accepted

**Usage:** Request accepted but processing not complete (async operations).

```http
POST /reports/generate
HTTP/1.1 202 Accepted

{
  "message": "Report generation started",
  "statusUrl": "/reports/status/abc123"
}
```

#### 204 No Content

**Usage:** Success but no content to return (DELETE, PUT, PATCH).

```http
DELETE /users/123
HTTP/1.1 204 No Content
```

**Note:** No response body.

### 3xx Redirection

#### 301 Moved Permanently

**Usage:** Resource permanently moved to new URL.

```http
GET /users/123
HTTP/1.1 301 Moved Permanently
Location: /v2/users/123
```

#### 302 Found / 307 Temporary Redirect

**Usage:** Resource temporarily at different URL.

```http
GET /users/123
HTTP/1.1 307 Temporary Redirect
Location: /users/temp/123
```

#### 304 Not Modified

**Usage:** Cached version is still valid.

```http
GET /users/123
If-None-Match: "33a64df551425fcc55e4d42a148795d9f25f89d4"

HTTP/1.1 304 Not Modified
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
```

### 4xx Client Errors

#### 400 Bad Request

**Usage:** Invalid request syntax or parameters.

```http
POST /users
{
  "email": "invalid-email"
}

HTTP/1.1 400 Bad Request
{
  "error": "Validation failed",
  "details": {
    "email": "Invalid email format"
  }
}
```

**When to Use:**
- Invalid JSON
- Missing required fields
- Invalid field values
- Malformed parameters

#### 401 Unauthorized

**Usage:** Authentication required or failed.

```http
GET /users/123
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="api"

{
  "error": "Authentication required",
  "message": "Please provide a valid access token"
}
```

**Note:** Despite the name, this means "unauthenticated".

#### 403 Forbidden

**Usage:** Authenticated but not authorized.

```http
DELETE /users/123
Authorization: Bearer valid-token

HTTP/1.1 403 Forbidden
{
  "error": "Insufficient permissions",
  "message": "You don't have permission to delete this user"
}
```

**401 vs 403:**
```
401: You need to log in
403: You're logged in, but you can't do this
```

#### 404 Not Found

**Usage:** Resource doesn't exist.

```http
GET /users/999
HTTP/1.1 404 Not Found

{
  "error": "Resource not found",
  "message": "User with ID 999 does not exist"
}
```

#### 405 Method Not Allowed

**Usage:** HTTP method not supported for resource.

```http
POST /users/123
HTTP/1.1 405 Method Not Allowed
Allow: GET, PUT, PATCH, DELETE

{
  "error": "Method not allowed",
  "message": "POST is not supported for this resource. Use PUT or PATCH to update."
}
```

#### 409 Conflict

**Usage:** Request conflicts with current state.

```http
POST /users
{
  "email": "existing@example.com"
}

HTTP/1.1 409 Conflict
{
  "error": "User already exists",
  "message": "A user with this email already exists"
}
```

**Use Cases:**
- Duplicate resource
- Version conflict
- Business rule violation

#### 422 Unprocessable Entity

**Usage:** Valid syntax but semantic errors.

```http
POST /users
{
  "age": -5,
  "email": "valid@example.com"
}

HTTP/1.1 422 Unprocessable Entity
{
  "error": "Validation failed",
  "details": {
    "age": "Age must be a positive number"
  }
}
```

**400 vs 422:**
```
400: Syntax error (invalid JSON, wrong type)
422: Semantic error (valid JSON, invalid business logic)
```

#### 429 Too Many Requests

**Usage:** Rate limit exceeded.

```http
GET /users
HTTP/1.1 429 Too Many Requests
Retry-After: 60

{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Please try again in 60 seconds."
}
```

### 5xx Server Errors

#### 500 Internal Server Error

**Usage:** Generic server error.

```http
GET /users/123
HTTP/1.1 500 Internal Server Error

{
  "error": "Internal server error",
  "message": "An unexpected error occurred",
  "requestId": "abc123"
}
```

**Best Practice:** Log detailed error, return generic message to client.

#### 502 Bad Gateway

**Usage:** Invalid response from upstream server.

```http
GET /users
HTTP/1.1 502 Bad Gateway

{
  "error": "Bad gateway",
  "message": "Error communicating with database"
}
```

#### 503 Service Unavailable

**Usage:** Server temporarily unavailable.

```http
GET /users
HTTP/1.1 503 Service Unavailable
Retry-After: 300

{
  "error": "Service unavailable",
  "message": "Server is under maintenance. Please try again in 5 minutes."
}
```

#### 504 Gateway Timeout

**Usage:** Upstream server timeout.

```http
GET /reports/large
HTTP/1.1 504 Gateway Timeout

{
  "error": "Gateway timeout",
  "message": "Request took too long to process"
}
```

### Status Code Decision Tree

```
Request received
├─ Valid request?
│  ├─ No → 400 Bad Request
│  └─ Yes
│     ├─ Authenticated?
│     │  ├─ No → 401 Unauthorized
│     │  └─ Yes
│     │     ├─ Authorized?
│     │     │  ├─ No → 403 Forbidden
│     │     │  └─ Yes
│     │     │     ├─ Resource exists?
│     │     │     │  ├─ No → 404 Not Found
│     │     │     │  └─ Yes
│     │     │     │     ├─ Method allowed?
│     │     │     │     │  ├─ No → 405 Method Not Allowed
│     │     │     │     │  └─ Yes
│     │     │     │     │     ├─ Business logic valid?
│     │     │     │     │     │  ├─ No → 422 Unprocessable Entity / 409 Conflict
│     │     │     │     │     │  └─ Yes
│     │     │     │     │     │     ├─ Server error?
│     │     │     │     │     │     │  ├─ Yes → 500 Internal Server Error
│     │     │     │     │     │     │  └─ No → 200 OK / 201 Created / 204 No Content
```

### Implementation Example

```javascript
const express = require('express');
const app = express();

app.get('/users/:id', async (req, res, next) => {
  try {
    // Validate ID format
    if (!isValidId(req.params.id)) {
      return res.status(400).json({
        error: 'Invalid ID format'
      });
    }

    // Check authentication
    if (!req.headers.authorization) {
      return res.status(401).json({
        error: 'Authentication required'
      });
    }

    // Verify token
    const user = await verifyToken(req.headers.authorization);
    if (!user) {
      return res.status(401).json({
        error: 'Invalid token'
      });
    }

    // Get resource
    const targetUser = await database.getUser(req.params.id);
    if (!targetUser) {
      return res.status(404).json({
        error: 'User not found'
      });
    }

    // Check authorization
    if (!canAccessUser(user, targetUser)) {
      return res.status(403).json({
        error: 'Insufficient permissions'
      });
    }

    // Success
    res.status(200).json(targetUser);

  } catch (error) {
    // Server error
    console.error('Error:', error);
    res.status(500).json({
      error: 'Internal server error',
      requestId: req.id
    });
  }
});

app.post('/users', async (req, res) => {
  // Validate input
  const errors = validateUser(req.body);
  if (errors.length > 0) {
    return res.status(422).json({
      error: 'Validation failed',
      details: errors
    });
  }

  // Check for duplicates
  const existing = await database.findUserByEmail(req.body.email);
  if (existing) {
    return res.status(409).json({
      error: 'User already exists'
    });
  }

  // Create user
  const user = await database.createUser(req.body);

  res.status(201)
    .location(`/users/${user.id}`)
    .json(user);
});
```

---

## API Versioning Strategies

API versioning allows you to evolve your API while maintaining backward compatibility for existing clients.

### Why Version APIs?

- **Breaking Changes:** Modify response structure, rename fields, change behavior
- **Backward Compatibility:** Support old clients while adding new features
- **Gradual Migration:** Give clients time to upgrade
- **Multiple Client Versions:** Mobile apps can't force immediate updates

### Versioning Strategies

#### 1. URL Path Versioning

**Most Common Approach**

```
https://api.example.com/v1/users
https://api.example.com/v2/users
https://api.example.com/v3/users
```

**Pros:**
- Simple and explicit
- Easy to route and cache
- Clear in URLs and documentation
- Browser-friendly

**Cons:**
- URLs change between versions
- Can lead to URL bloat

**Implementation:**
```javascript
const express = require('express');
const app = express();

// Version 1
app.get('/v1/users/:id', (req, res) => {
  const user = getUserV1(req.params.id);
  res.json({
    id: user.id,
    name: user.name,
    email: user.email
  });
});

// Version 2 - added phone field
app.get('/v2/users/:id', (req, res) => {
  const user = getUserV2(req.params.id);
  res.json({
    id: user.id,
    fullName: user.name,  // renamed field
    email: user.email,
    phone: user.phone     // new field
  });
});

// Version 3 - restructured response
app.get('/v3/users/:id', (req, res) => {
  const user = getUserV3(req.params.id);
  res.json({
    user: {
      id: user.id,
      profile: {
        fullName: user.name,
        email: user.email,
        phone: user.phone
      },
      metadata: {
        createdAt: user.createdAt,
        updatedAt: user.updatedAt
      }
    }
  });
});
```

**Best Practices:**
```
✅ Use major versions only: v1, v2, v3
✅ Include version in base path: /v1/
❌ Avoid minor versions in URL: /v1.2/users
```

#### 2. Header Versioning

**Custom Header**

```http
GET /users/123 HTTP/1.1
Host: api.example.com
API-Version: 2
```

**Accept Header (Content Negotiation)**

```http
GET /users/123 HTTP/1.1
Host: api.example.com
Accept: application/vnd.example.v2+json
```

**Pros:**
- Clean URLs
- Better adherence to REST principles
- Supports multiple versioning dimensions

**Cons:**
- Less visible
- Harder to test (can't just paste URL in browser)
- More complex routing

**Implementation:**
```javascript
// Custom header versioning
app.use((req, res, next) => {
  req.apiVersion = req.headers['api-version'] || '1';
  next();
});

app.get('/users/:id', (req, res) => {
  const user = getUser(req.params.id);

  if (req.apiVersion === '1') {
    res.json({
      id: user.id,
      name: user.name,
      email: user.email
    });
  } else if (req.apiVersion === '2') {
    res.json({
      id: user.id,
      fullName: user.name,
      email: user.email,
      phone: user.phone
    });
  }
});

// Accept header versioning
app.get('/users/:id', (req, res) => {
  const user = getUser(req.params.id);
  const accept = req.headers.accept;

  if (accept.includes('vnd.example.v2+json')) {
    res.json({ /* v2 format */ });
  } else {
    res.json({ /* v1 format */ });
  }
});
```

#### 3. Query Parameter Versioning

```
https://api.example.com/users?version=2
https://api.example.com/users?v=2
https://api.example.com/users?api-version=2
```

**Pros:**
- Simple to implement
- Easy to test
- Flexible

**Cons:**
- Mixes versioning with filtering parameters
- Less clean than URL path versioning
- Not RESTful

**Implementation:**
```javascript
app.get('/users/:id', (req, res) => {
  const version = req.query.version || '1';
  const user = getUser(req.params.id);

  const formatters = {
    '1': formatUserV1,
    '2': formatUserV2,
    '3': formatUserV3
  };

  const formatter = formatters[version] || formatters['1'];
  res.json(formatter(user));
});
```

#### 4. Content Negotiation (Media Type)

```http
GET /users/123 HTTP/1.1
Accept: application/vnd.example.user.v2+json
```

**Pros:**
- RESTful approach
- Standard HTTP mechanism
- Can version individual resources

**Cons:**
- Complex
- Less intuitive
- Harder to debug

### Version Management Strategies

#### Default Version

```javascript
// Default to latest stable version
app.use((req, res, next) => {
  if (!req.path.startsWith('/v')) {
    return res.redirect(`/v2${req.path}`);
  }
  next();
});

// Or serve default version directly
app.get('/users/:id', (req, res) => {
  // Default to v2
  req.apiVersion = '2';
  // Handle request...
});
```

#### Version Sunset/Deprecation

```http
GET /v1/users/123 HTTP/1.1

HTTP/1.1 200 OK
Sunset: Sat, 31 Dec 2023 23:59:59 GMT
Deprecation: true
Link: </v2/users/123>; rel="successor-version"
Warning: 299 - "API version 1 is deprecated and will be removed on Dec 31, 2023"

{
  "id": 123,
  "name": "John Doe"
}
```

**Implementation:**
```javascript
app.use('/v1/*', (req, res, next) => {
  res.set({
    'Sunset': 'Sat, 31 Dec 2023 23:59:59 GMT',
    'Deprecation': 'true',
    'Warning': '299 - "API version 1 is deprecated"'
  });
  next();
});
```

### Versioning Best Practices

#### 1. Version the Entire API

```
✅ /v1/users, /v1/orders, /v1/products (consistent)
❌ /v1/users, /v2/orders, /users/products (inconsistent)
```

#### 2. Make Non-Breaking Changes When Possible

**Non-Breaking Changes:**
- Adding new endpoints
- Adding optional fields to requests
- Adding fields to responses
- Adding optional query parameters

**Breaking Changes:**
- Removing fields from responses
- Renaming fields
- Changing field types
- Changing validation rules
- Removing endpoints

```javascript
// Non-breaking: Adding optional field
// Version 1
{ "name": "John" }

// Version 1 (updated)
{ "name": "John", "phone": "123" }  // phone is optional

// Breaking: Renaming field
// Version 1
{ "name": "John" }

// Version 2 (requires new version)
{ "fullName": "John" }
```

#### 3. Support Multiple Versions

```javascript
const v1Routes = require('./routes/v1');
const v2Routes = require('./routes/v2');
const v3Routes = require('./routes/v3');

app.use('/v1', v1Routes);
app.use('/v2', v2Routes);
app.use('/v3', v3Routes);
```

#### 4. Document Version Differences

```markdown
# API Versions

## Version 3 (Current)
- Restructured user response
- Added pagination metadata
- Breaking: Changed date format to ISO 8601

## Version 2 (Deprecated - Sunset: Dec 31, 2023)
- Added phone field
- Renamed `name` to `fullName`

## Version 1 (Deprecated - Sunset: Dec 31, 2023)
- Initial version
```

### Migration Example

**Version 1 → Version 2 Migration**

```javascript
// v1/users.js
router.get('/:id', async (req, res) => {
  const user = await db.getUser(req.params.id);
  res.json({
    id: user.id,
    name: user.name,
    email: user.email,
    created: user.createdAt.getTime()  // timestamp
  });
});

// v2/users.js
router.get('/:id', async (req, res) => {
  const user = await db.getUser(req.params.id);
  res.json({
    id: user.id,
    fullName: user.name,              // renamed
    email: user.email,
    phone: user.phone,                // added
    createdAt: user.createdAt.toISOString()  // changed format
  });
});

// Migration helper for clients
router.get('/:id/migrate', async (req, res) => {
  const user = await db.getUser(req.params.id);
  res.json({
    v1: {
      id: user.id,
      name: user.name,
      email: user.email,
      created: user.createdAt.getTime()
    },
    v2: {
      id: user.id,
      fullName: user.name,
      email: user.email,
      phone: user.phone,
      createdAt: user.createdAt.toISOString()
    },
    migration: {
      'name → fullName': 'Field renamed',
      'phone': 'New field added',
      'created → createdAt': 'Format changed to ISO 8601'
    }
  });
});
```

---

## Pagination Strategies

Pagination is essential for APIs that return large datasets. It improves performance, reduces bandwidth, and provides better user experience.

### Why Paginate?

- **Performance:** Avoid loading entire dataset into memory
- **Bandwidth:** Reduce data transfer
- **User Experience:** Faster response times
- **Database Load:** Reduce query complexity

### Pagination Strategies

#### 1. Offset-Based Pagination

**Most Common Approach**

```
GET /users?limit=20&offset=40
GET /users?page=3&per_page=20
```

**Response:**
```json
{
  "data": [
    { "id": 41, "name": "User 41" },
    { "id": 42, "name": "User 42" },
    ...
  ],
  "pagination": {
    "total": 1000,
    "page": 3,
    "perPage": 20,
    "totalPages": 50,
    "hasNext": true,
    "hasPrev": true
  }
}
```

**Implementation:**
```javascript
app.get('/users', async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const perPage = parseInt(req.query.per_page) || 20;
  const offset = (page - 1) * perPage;

  // Get paginated data
  const users = await db.query(
    'SELECT * FROM users LIMIT $1 OFFSET $2',
    [perPage, offset]
  );

  // Get total count
  const total = await db.query('SELECT COUNT(*) FROM users');
  const totalUsers = parseInt(total.rows[0].count);
  const totalPages = Math.ceil(totalUsers / perPage);

  res.json({
    data: users.rows,
    pagination: {
      total: totalUsers,
      page,
      perPage,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1
    }
  });
});
```

**Pros:**
- Simple to implement
- Easy to jump to specific page
- Shows total count and pages

**Cons:**
- Performance degrades with large offsets
- Inconsistent results if data changes during pagination
- Expensive COUNT(*) queries

**Performance Issue:**
```sql
-- Fast
SELECT * FROM users LIMIT 20 OFFSET 0;

-- Slow (must scan 1,000,000 rows)
SELECT * FROM users LIMIT 20 OFFSET 1000000;
```

#### 2. Cursor-Based Pagination

**Best for Real-Time Data**

```
GET /users?limit=20
GET /users?limit=20&cursor=eyJpZCI6MTIzfQ==
```

**Response:**
```json
{
  "data": [
    { "id": 124, "name": "User 124" },
    { "id": 125, "name": "User 125" },
    ...
  ],
  "pagination": {
    "nextCursor": "eyJpZCI6MTQzfQ==",
    "hasMore": true
  }
}
```

**Implementation:**
```javascript
app.get('/users', async (req, res) => {
  const limit = parseInt(req.query.limit) || 20;
  const cursor = req.query.cursor
    ? JSON.parse(Buffer.from(req.query.cursor, 'base64').toString())
    : null;

  // Build query
  let query = 'SELECT * FROM users';
  let params = [limit + 1];  // Fetch one extra to check if more exist

  if (cursor) {
    query += ' WHERE id > $2';
    params.push(cursor.id);
  }

  query += ' ORDER BY id ASC LIMIT $1';

  const users = await db.query(query, params);
  const hasMore = users.rows.length > limit;

  // Remove extra item if exists
  if (hasMore) {
    users.rows.pop();
  }

  // Create next cursor
  let nextCursor = null;
  if (hasMore && users.rows.length > 0) {
    const lastUser = users.rows[users.rows.length - 1];
    nextCursor = Buffer.from(
      JSON.stringify({ id: lastUser.id })
    ).toString('base64');
  }

  res.json({
    data: users.rows,
    pagination: {
      nextCursor,
      hasMore
    }
  });
});
```

**Pros:**
- Consistent performance (no offset)
- Handles real-time data well
- No missing/duplicate items during pagination

**Cons:**
- Can't jump to specific page
- No total count
- More complex implementation

**When to Use:**
- Infinite scroll
- Real-time feeds
- Large datasets
- Chat messages

#### 3. Keyset Pagination

**Variation of Cursor-Based**

```
GET /users?limit=20&after_id=123
GET /posts?limit=20&after_date=2023-11-15T10:00:00Z
```

**Implementation:**
```javascript
app.get('/posts', async (req, res) => {
  const limit = parseInt(req.query.limit) || 20;
  const afterDate = req.query.after_date;

  let query = 'SELECT * FROM posts';
  const params = [limit + 1];

  if (afterDate) {
    query += ' WHERE created_at < $2';
    params.push(afterDate);
  }

  query += ' ORDER BY created_at DESC LIMIT $1';

  const posts = await db.query(query, params);
  const hasMore = posts.rows.length > limit;

  if (hasMore) {
    posts.rows.pop();
  }

  res.json({
    data: posts.rows,
    pagination: {
      hasMore,
      nextAfterDate: hasMore
        ? posts.rows[posts.rows.length - 1].created_at
        : null
    }
  });
});
```

**Pros:**
- Human-readable cursor
- Efficient database queries
- Predictable performance

**Cons:**
- Requires indexed column
- Can't jump to specific page

### Link Header Pagination (GitHub Style)

```http
GET /users?page=3&per_page=20 HTTP/1.1

HTTP/1.1 200 OK
Link: <https://api.example.com/users?page=1&per_page=20>; rel="first",
      <https://api.example.com/users?page=2&per_page=20>; rel="prev",
      <https://api.example.com/users?page=4&per_page=20>; rel="next",
      <https://api.example.com/users?page=50&per_page=20>; rel="last"

[
  { "id": 41, "name": "User 41" },
  ...
]
```

**Implementation:**
```javascript
app.get('/users', async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const perPage = parseInt(req.query.per_page) || 20;

  // Get data and total
  const users = await getUsers(page, perPage);
  const total = await getUserCount();
  const totalPages = Math.ceil(total / perPage);

  // Build Link header
  const baseUrl = `${req.protocol}://${req.get('host')}${req.path}`;
  const links = [];

  // First page
  links.push(`<${baseUrl}?page=1&per_page=${perPage}>; rel="first"`);

  // Previous page
  if (page > 1) {
    links.push(`<${baseUrl}?page=${page-1}&per_page=${perPage}>; rel="prev"`);
  }

  // Next page
  if (page < totalPages) {
    links.push(`<${baseUrl}?page=${page+1}&per_page=${perPage}>; rel="next"`);
  }

  // Last page
  links.push(`<${baseUrl}?page=${totalPages}&per_page=${perPage}>; rel="last"`);

  res.set('Link', links.join(', '));
  res.json(users);
});
```

### Pagination Best Practices

#### 1. Set Default and Maximum Limits

```javascript
app.get('/users', (req, res) => {
  const limit = Math.min(
    parseInt(req.query.limit) || 20,  // default
    100                                // maximum
  );

  if (limit < 1 || limit > 100) {
    return res.status(400).json({
      error: 'Limit must be between 1 and 100'
    });
  }

  // Continue with pagination...
});
```

#### 2. Include Pagination Metadata

```json
{
  "data": [...],
  "meta": {
    "total": 1000,
    "page": 3,
    "perPage": 20,
    "totalPages": 50,
    "links": {
      "first": "/users?page=1",
      "prev": "/users?page=2",
      "self": "/users?page=3",
      "next": "/users?page=4",
      "last": "/users?page=50"
    }
  }
}
```

#### 3. Handle Edge Cases

```javascript
// Empty results
{
  "data": [],
  "pagination": {
    "total": 0,
    "page": 1,
    "perPage": 20,
    "totalPages": 0,
    "hasNext": false,
    "hasPrev": false
  }
}

// Out of range page
if (page > totalPages && totalPages > 0) {
  return res.status(404).json({
    error: 'Page not found',
    message: `Page ${page} does not exist. Total pages: ${totalPages}`
  });
}
```

#### 4. Support Sorting with Pagination

```javascript
GET /users?page=2&sort=name&order=asc
GET /users?page=2&sort=-created_at  // minus sign for descending

app.get('/users', async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const perPage = 20;
  const sort = req.query.sort || 'id';
  const order = req.query.order === 'desc' ? 'DESC' : 'ASC';

  // Validate sort field
  const allowedSortFields = ['id', 'name', 'created_at'];
  if (!allowedSortFields.includes(sort)) {
    return res.status(400).json({ error: 'Invalid sort field' });
  }

  const users = await db.query(
    `SELECT * FROM users ORDER BY ${sort} ${order} LIMIT $1 OFFSET $2`,
    [perPage, (page - 1) * perPage]
  );

  res.json({ data: users.rows });
});
```

### Choosing a Pagination Strategy

| Use Case | Strategy | Reason |
|----------|----------|--------|
| Admin dashboards | Offset-based | Need page numbers and total count |
| Social media feeds | Cursor-based | Real-time data, infinite scroll |
| Search results | Offset-based | Jump to specific pages |
| Chat messages | Cursor-based | Real-time, chronological |
| Reports | Offset-based | Need total count |
| Activity logs | Keyset | Time-based, efficient |

---

## Filtering and Sorting

Filtering and sorting allow clients to retrieve specific subsets of data in the desired order.

### Filtering

#### Basic Filtering

```
GET /users?status=active
GET /users?role=admin&status=active
GET /products?category=electronics&price_min=100&price_max=500
```

**Implementation:**
```javascript
app.get('/users', async (req, res) => {
  const { status, role, city } = req.query;

  let query = 'SELECT * FROM users WHERE 1=1';
  const params = [];
  let paramIndex = 1;

  if (status) {
    query += ` AND status = $${paramIndex++}`;
    params.push(status);
  }

  if (role) {
    query += ` AND role = $${paramIndex++}`;
    params.push(role);
  }

  if (city) {
    query += ` AND city = $${paramIndex++}`;
    params.push(city);
  }

  const users = await db.query(query, params);
  res.json({ data: users.rows });
});
```

#### Range Filtering

```
GET /products?price_min=100&price_max=500
GET /events?start_date=2023-01-01&end_date=2023-12-31
GET /users?age_gte=18&age_lte=65
```

**Implementation:**
```javascript
app.get('/products', async (req, res) => {
  const { price_min, price_max, stock_gt, stock_lt } = req.query;

  let query = 'SELECT * FROM products WHERE 1=1';
  const params = [];
  let paramIndex = 1;

  if (price_min) {
    query += ` AND price >= $${paramIndex++}`;
    params.push(parseFloat(price_min));
  }

  if (price_max) {
    query += ` AND price <= $${paramIndex++}`;
    params.push(parseFloat(price_max));
  }

  if (stock_gt) {
    query += ` AND stock > $${paramIndex++}`;
    params.push(parseInt(stock_gt));
  }

  if (stock_lt) {
    query += ` AND stock < $${paramIndex++}`;
    params.push(parseInt(stock_lt));
  }

  const products = await db.query(query, params);
  res.json({ data: products.rows });
});
```

**Operators Convention:**
```
_min, _max    → Inclusive range
_gt, _lt      → Greater than, less than
_gte, _lte    → Greater/less than or equal
_ne           → Not equal
_in           → In array
```

#### Array Filtering

```
GET /products?category=electronics,books,toys
GET /users?role_in=admin,moderator
GET /posts?tags=javascript,nodejs
```

**Implementation:**
```javascript
app.get('/products', async (req, res) => {
  const { category, tags } = req.query;

  let query = 'SELECT * FROM products WHERE 1=1';
  const params = [];

  if (category) {
    const categories = category.split(',');
    query += ' AND category = ANY($1)';
    params.push(categories);
  }

  if (tags) {
    const tagList = tags.split(',');
    query += ' AND tags && $2';  // PostgreSQL array overlap
    params.push(tagList);
  }

  const products = await db.query(query, params);
  res.json({ data: products.rows });
});
```

#### Search/Text Filtering

```
GET /users?search=john
GET /products?q=laptop
GET /posts?title_contains=api
```

**Implementation:**
```javascript
app.get('/users', async (req, res) => {
  const { search } = req.query;

  if (search) {
    // Full-text search
    const users = await db.query(
      `SELECT * FROM users
       WHERE name ILIKE $1
          OR email ILIKE $1`,
      [`%${search}%`]
    );
    return res.json({ data: users.rows });
  }

  // Regular query...
});

// Advanced: Full-text search with PostgreSQL
app.get('/posts', async (req, res) => {
  const { q } = req.query;

  if (q) {
    const posts = await db.query(
      `SELECT *, ts_rank(search_vector, query) AS rank
       FROM posts, to_tsquery($1) query
       WHERE search_vector @@ query
       ORDER BY rank DESC`,
      [q.split(' ').join(' & ')]
    );
    return res.json({ data: posts.rows });
  }
});
```

#### Complex Filtering (Filter Query Language)

```
GET /users?filter={"age":{"$gte":18},"city":"NYC"}
GET /products?filter={"$or":[{"category":"electronics"},{"price":{"$lt":100}}]}
```

**Implementation:**
```javascript
// Using MongoDB-style query language
app.get('/users', async (req, res) => {
  const { filter } = req.query;

  if (filter) {
    try {
      const filterObj = JSON.parse(filter);
      const users = await db.collection('users').find(filterObj).toArray();
      return res.json({ data: users });
    } catch (error) {
      return res.status(400).json({ error: 'Invalid filter' });
    }
  }
});

// Custom filter builder
function buildWhereClause(filter) {
  const conditions = [];
  const params = [];
  let paramIndex = 1;

  for (const [key, value] of Object.entries(filter)) {
    if (typeof value === 'object') {
      for (const [operator, operand] of Object.entries(value)) {
        switch (operator) {
          case '$gte':
            conditions.push(`${key} >= $${paramIndex++}`);
            params.push(operand);
            break;
          case '$lte':
            conditions.push(`${key} <= $${paramIndex++}`);
            params.push(operand);
            break;
          case '$gt':
            conditions.push(`${key} > $${paramIndex++}`);
            params.push(operand);
            break;
          case '$lt':
            conditions.push(`${key} < $${paramIndex++}`);
            params.push(operand);
            break;
          case '$ne':
            conditions.push(`${key} != $${paramIndex++}`);
            params.push(operand);
            break;
          case '$in':
            conditions.push(`${key} = ANY($${paramIndex++})`);
            params.push(operand);
            break;
        }
      }
    } else {
      conditions.push(`${key} = $${paramIndex++}`);
      params.push(value);
    }
  }

  return {
    where: conditions.join(' AND '),
    params
  };
}
```

### Sorting

#### Basic Sorting

```
GET /users?sort=name
GET /users?sort=created_at
GET /products?sort=price
```

**Implementation:**
```javascript
app.get('/users', async (req, res) => {
  const { sort = 'id', order = 'asc' } = req.query;

  // Whitelist allowed sort fields
  const allowedSortFields = ['id', 'name', 'email', 'created_at'];
  if (!allowedSortFields.includes(sort)) {
    return res.status(400).json({
      error: 'Invalid sort field',
      allowed: allowedSortFields
    });
  }

  const orderDir = order.toLowerCase() === 'desc' ? 'DESC' : 'ASC';

  const users = await db.query(
    `SELECT * FROM users ORDER BY ${sort} ${orderDir}`
  );

  res.json({ data: users.rows });
});
```

#### Multi-field Sorting

```
GET /users?sort=last_name,first_name
GET /products?sort=-price,name  // - prefix for descending
GET /posts?sort=pinned:desc,created_at:desc
```

**Implementation:**
```javascript
app.get('/users', async (req, res) => {
  const { sort } = req.query;

  if (!sort) {
    return res.json({ data: await getAllUsers() });
  }

  const sortFields = sort.split(',');
  const orderClauses = [];
  const allowedFields = ['id', 'name', 'created_at', 'age'];

  for (const field of sortFields) {
    let fieldName = field;
    let direction = 'ASC';

    // Handle - prefix for descending
    if (field.startsWith('-')) {
      fieldName = field.substring(1);
      direction = 'DESC';
    }

    // Handle :desc/:asc suffix
    if (field.includes(':')) {
      const parts = field.split(':');
      fieldName = parts[0];
      direction = parts[1].toUpperCase();
    }

    // Validate field
    if (!allowedFields.includes(fieldName)) {
      return res.status(400).json({
        error: `Invalid sort field: ${fieldName}`
      });
    }

    orderClauses.push(`${fieldName} ${direction}`);
  }

  const users = await db.query(
    `SELECT * FROM users ORDER BY ${orderClauses.join(', ')}`
  );

  res.json({ data: users.rows });
});
```

### Combining Filtering, Sorting, and Pagination

```
GET /products?category=electronics&price_min=100&price_max=500&sort=-price&page=2&limit=20
```

**Complete Implementation:**
```javascript
app.get('/products', async (req, res) => {
  const {
    // Filtering
    category,
    price_min,
    price_max,
    in_stock,
    // Sorting
    sort = 'id',
    order = 'asc',
    // Pagination
    page = 1,
    limit = 20
  } = req.query;

  // Build WHERE clause
  const conditions = ['1=1'];
  const params = [];
  let paramIndex = 1;

  if (category) {
    conditions.push(`category = $${paramIndex++}`);
    params.push(category);
  }

  if (price_min) {
    conditions.push(`price >= $${paramIndex++}`);
    params.push(parseFloat(price_min));
  }

  if (price_max) {
    conditions.push(`price <= $${paramIndex++}`);
    params.push(parseFloat(price_max));
  }

  if (in_stock === 'true') {
    conditions.push('stock > 0');
  }

  const whereClause = conditions.join(' AND ');

  // Validate and build ORDER BY
  const allowedSortFields = ['id', 'name', 'price', 'created_at', 'stock'];
  if (!allowedSortFields.includes(sort)) {
    return res.status(400).json({ error: 'Invalid sort field' });
  }

  const orderDir = order.toLowerCase() === 'desc' ? 'DESC' : 'ASC';
  const orderClause = `${sort} ${orderDir}`;

  // Pagination
  const pageNum = parseInt(page);
  const limitNum = Math.min(parseInt(limit), 100);  // max 100
  const offset = (pageNum - 1) * limitNum;

  params.push(limitNum, offset);

  // Execute query
  const query = `
    SELECT * FROM products
    WHERE ${whereClause}
    ORDER BY ${orderClause}
    LIMIT $${paramIndex++} OFFSET $${paramIndex++}
  `;

  const products = await db.query(query, params);

  // Get total count for pagination
  const countQuery = `SELECT COUNT(*) FROM products WHERE ${whereClause}`;
  const totalResult = await db.query(countQuery, params.slice(0, -2));
  const total = parseInt(totalResult.rows[0].count);

  res.json({
    data: products.rows,
    meta: {
      page: pageNum,
      limit: limitNum,
      total,
      totalPages: Math.ceil(total / limitNum),
      filters: { category, price_min, price_max, in_stock },
      sort: { field: sort, order: orderDir }
    }
  });
});
```

### Filter and Sort Best Practices

#### 1. Validate Input

```javascript
function validateFilter(filter) {
  const allowedFields = ['name', 'email', 'status', 'role'];
  const allowedOperators = ['$eq', '$ne', '$gt', '$gte', '$lt', '$lte', '$in'];

  for (const field of Object.keys(filter)) {
    if (!allowedFields.includes(field)) {
      throw new Error(`Invalid filter field: ${field}`);
    }

    if (typeof filter[field] === 'object') {
      for (const operator of Object.keys(filter[field])) {
        if (!allowedOperators.includes(operator)) {
          throw new Error(`Invalid operator: ${operator}`);
        }
      }
    }
  }
}
```

#### 2. Prevent SQL Injection

```javascript
// ❌ BAD - SQL injection vulnerability
app.get('/users', (req, res) => {
  const { sort } = req.query;
  const query = `SELECT * FROM users ORDER BY ${sort}`;  // DANGEROUS!
  db.query(query);
});

// ✅ GOOD - Whitelist allowed fields
app.get('/users', (req, res) => {
  const { sort } = req.query;
  const allowedFields = ['id', 'name', 'created_at'];

  if (!allowedFields.includes(sort)) {
    return res.status(400).json({ error: 'Invalid sort field' });
  }

  const query = `SELECT * FROM users ORDER BY ${sort}`;  // Safe
  db.query(query);
});
```

#### 3. Document Available Filters

```json
GET /products/filters

{
  "filters": {
    "category": {
      "type": "string",
      "description": "Product category",
      "example": "electronics"
    },
    "price_min": {
      "type": "number",
      "description": "Minimum price",
      "example": 100
    },
    "price_max": {
      "type": "number",
      "description": "Maximum price",
      "example": 500
    },
    "in_stock": {
      "type": "boolean",
      "description": "Filter by stock availability",
      "example": true
    }
  },
  "sortFields": ["id", "name", "price", "created_at", "stock"],
  "examples": [
    "/products?category=electronics&price_min=100&sort=-price",
    "/products?in_stock=true&sort=name&page=2"
  ]
}
```

---

## Rate Limiting

Rate limiting protects APIs from abuse and ensures fair usage among clients.

### Why Rate Limit?

- **Prevent Abuse:** Stop malicious users from overwhelming the API
- **Ensure Fair Usage:** Distribute resources equitably
- **Cost Control:** Manage infrastructure costs
- **Quality of Service:** Maintain performance for all users
- **Security:** Mitigate DDoS attacks

### Rate Limiting Strategies

#### 1. Fixed Window

**Algorithm:** Count requests in fixed time windows (e.g., per minute, per hour).

```
Window 1: 00:00-00:59 → 100 requests allowed
Window 2: 01:00-01:59 → 100 requests allowed (counter resets)
```

**Pros:**
- Simple to implement
- Low memory usage

**Cons:**
- Burst traffic at window boundaries
- Not smooth rate limiting

**Implementation:**
```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 60 * 1000,  // 1 minute
  max: 100,             // max 100 requests per window
  message: {
    error: 'Too many requests',
    retryAfter: 60
  },
  standardHeaders: true,  // Return rate limit info in headers
  legacyHeaders: false
});

app.use('/api/', limiter);
```

**Response Headers:**
```http
HTTP/1.1 200 OK
RateLimit-Limit: 100
RateLimit-Remaining: 75
RateLimit-Reset: 1699876800

HTTP/1.1 429 Too Many Requests
RateLimit-Limit: 100
RateLimit-Remaining: 0
RateLimit-Reset: 1699876800
Retry-After: 60

{
  "error": "Too many requests",
  "retryAfter": 60
}
```

#### 2. Sliding Window

**Algorithm:** Count requests in a rolling time window.

```
At 00:30: Count requests from 23:30-00:30
At 00:31: Count requests from 23:31-00:31
```

**Pros:**
- Smoother rate limiting
- No burst issues

**Cons:**
- More complex
- Higher memory usage

**Implementation (Redis):**
```javascript
const Redis = require('ioredis');
const redis = new Redis();

async function slidingWindowRateLimit(userId, limit, windowMs) {
  const key = `ratelimit:${userId}`;
  const now = Date.now();
  const windowStart = now - windowMs;

  // Start transaction
  const pipeline = redis.pipeline();

  // Remove old entries
  pipeline.zremrangebyscore(key, 0, windowStart);

  // Count requests in window
  pipeline.zcard(key);

  // Add current request
  pipeline.zadd(key, now, `${now}-${Math.random()}`);

  // Set expiry
  pipeline.expire(key, Math.ceil(windowMs / 1000));

  const results = await pipeline.exec();
  const count = results[1][1];

  if (count >= limit) {
    throw new Error('Rate limit exceeded');
  }

  return {
    allowed: true,
    remaining: limit - count - 1
  };
}

// Middleware
app.use(async (req, res, next) => {
  try {
    const userId = req.user?.id || req.ip;
    const result = await slidingWindowRateLimit(userId, 100, 60000);

    res.set({
      'X-RateLimit-Limit': 100,
      'X-RateLimit-Remaining': result.remaining
    });

    next();
  } catch (error) {
    res.status(429).json({
      error: 'Rate limit exceeded'
    });
  }
});
```

#### 3. Token Bucket

**Algorithm:** Bucket fills with tokens at fixed rate. Each request consumes a token.

```
Bucket capacity: 100 tokens
Refill rate: 10 tokens/second
Request: Consumes 1 token
```

**Pros:**
- Allows short bursts
- Smooth long-term rate
- Flexible

**Cons:**
- Complex implementation

**Implementation:**
```javascript
class TokenBucket {
  constructor(capacity, refillRate) {
    this.capacity = capacity;
    this.tokens = capacity;
    this.refillRate = refillRate;  // tokens per second
    this.lastRefill = Date.now();
  }

  refill() {
    const now = Date.now();
    const timePassed = (now - this.lastRefill) / 1000;  // seconds
    const tokensToAdd = timePassed * this.refillRate;

    this.tokens = Math.min(this.capacity, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }

  tryConsume(tokens = 1) {
    this.refill();

    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return true;
    }

    return false;
  }

  getAvailableTokens() {
    this.refill();
    return Math.floor(this.tokens);
  }
}

// Usage
const buckets = new Map();

app.use((req, res, next) => {
  const userId = req.user?.id || req.ip;

  if (!buckets.has(userId)) {
    buckets.set(userId, new TokenBucket(100, 10));  // 100 capacity, 10/sec
  }

  const bucket = buckets.get(userId);

  if (bucket.tryConsume(1)) {
    res.set({
      'X-RateLimit-Limit': 100,
      'X-RateLimit-Remaining': bucket.getAvailableTokens()
    });
    next();
  } else {
    res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: Math.ceil((1 - bucket.tokens) / bucket.refillRate)
    });
  }
});
```

#### 4. Leaky Bucket

**Algorithm:** Requests enter bucket and leak out at constant rate.

```
Bucket capacity: 100 requests
Leak rate: 10 requests/second
Incoming requests queue up
```

**Pros:**
- Smooth output rate
- Good for protecting downstream services

**Cons:**
- Requests may queue (latency)
- Complex implementation

### Rate Limit Tiers

#### Different Limits for Different Users

```javascript
function getRateLimit(user) {
  if (!user) {
    return { limit: 10, window: 60000 };      // Anonymous: 10/min
  }

  switch (user.tier) {
    case 'free':
      return { limit: 100, window: 60000 };   // Free: 100/min
    case 'pro':
      return { limit: 1000, window: 60000 };  // Pro: 1000/min
    case 'enterprise':
      return { limit: 10000, window: 60000 }; // Enterprise: 10000/min
    default:
      return { limit: 100, window: 60000 };
  }
}

app.use(async (req, res, next) => {
  const limits = getRateLimit(req.user);
  const userId = req.user?.id || req.ip;

  const allowed = await checkRateLimit(userId, limits.limit, limits.window);

  if (!allowed) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      tier: req.user?.tier || 'anonymous',
      limit: limits.limit
    });
  }

  res.set({
    'X-RateLimit-Limit': limits.limit,
    'X-RateLimit-Tier': req.user?.tier || 'anonymous'
  });

  next();
});
```

### Rate Limit by Endpoint

```javascript
// Different limits for different endpoints
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 5,                     // 5 requests
  message: 'Too many login attempts'
});

const apiLimiter = rateLimit({
  windowMs: 60 * 1000,  // 1 minute
  max: 100              // 100 requests
});

const uploadLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,  // 1 hour
  max: 10                     // 10 uploads
});

app.post('/auth/login', authLimiter, loginHandler);
app.use('/api', apiLimiter);
app.post('/upload', uploadLimiter, uploadHandler);
```

### Rate Limiting Best Practices

#### 1. Return Proper Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 75
X-RateLimit-Reset: 1699876800
Retry-After: 60
```

```javascript
res.set({
  'X-RateLimit-Limit': limit,
  'X-RateLimit-Remaining': remaining,
  'X-RateLimit-Reset': Math.ceil(resetTime / 1000),
  'Retry-After': retryAfter
});
```

#### 2. Provide Clear Error Messages

```json
{
  "error": "Rate limit exceeded",
  "message": "You have exceeded the rate limit of 100 requests per minute",
  "limit": 100,
  "windowMs": 60000,
  "retryAfter": 45,
  "resetAt": "2023-11-15T10:00:00Z"
}
```

#### 3. Document Rate Limits

```markdown
# Rate Limits

## Free Tier
- 100 requests/minute
- 5,000 requests/day

## Pro Tier
- 1,000 requests/minute
- 50,000 requests/day

## Enterprise Tier
- 10,000 requests/minute
- Unlimited daily requests

## Endpoint-Specific Limits
- POST /auth/login: 5 requests/15 minutes
- POST /upload: 10 requests/hour
```

#### 4. Use Distributed Rate Limiting

```javascript
// Redis-based distributed rate limiting
const Redis = require('ioredis');
const redis = new Redis();

async function checkRateLimit(key, limit, window) {
  const current = await redis.incr(key);

  if (current === 1) {
    await redis.expire(key, Math.ceil(window / 1000));
  }

  return {
    allowed: current <= limit,
    remaining: Math.max(0, limit - current),
    current
  };
}
```

---

## Error Handling

Consistent, informative error handling improves API usability and debugging.

### Error Response Format

#### Standard Error Response

```json
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User with ID 123 was not found",
    "details": {
      "userId": 123
    },
    "timestamp": "2023-11-15T10:00:00Z",
    "path": "/users/123",
    "requestId": "abc-123-def-456"
  }
}
```

#### Implementation

```javascript
class APIError extends Error {
  constructor(code, message, statusCode = 500, details = {}) {
    super(message);
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
  }

  toJSON() {
    return {
      error: {
        code: this.code,
        message: this.message,
        details: this.details,
        timestamp: new Date().toISOString()
      }
    };
  }
}

// Error handler middleware
app.use((err, req, res, next) => {
  if (err instanceof APIError) {
    return res.status(err.statusCode).json({
      ...err.toJSON(),
      path: req.path,
      requestId: req.id
    });
  }

  // Generic error
  console.error('Unexpected error:', err);
  res.status(500).json({
    error: {
      code: 'INTERNAL_SERVER_ERROR',
      message: 'An unexpected error occurred',
      requestId: req.id
    }
  });
});

// Usage
app.get('/users/:id', async (req, res, next) => {
  const user = await db.getUser(req.params.id);

  if (!user) {
    throw new APIError(
      'USER_NOT_FOUND',
      `User with ID ${req.params.id} was not found`,
      404,
      { userId: req.params.id }
    );
  }

  res.json(user);
});
```

### Common Error Codes

```javascript
const ErrorCodes = {
  // Authentication & Authorization
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  INVALID_TOKEN: 'INVALID_TOKEN',
  TOKEN_EXPIRED: 'TOKEN_EXPIRED',

  // Validation
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  INVALID_INPUT: 'INVALID_INPUT',
  MISSING_REQUIRED_FIELD: 'MISSING_REQUIRED_FIELD',

  // Resources
  RESOURCE_NOT_FOUND: 'RESOURCE_NOT_FOUND',
  RESOURCE_ALREADY_EXISTS: 'RESOURCE_ALREADY_EXISTS',
  RESOURCE_CONFLICT: 'RESOURCE_CONFLICT',

  // Rate Limiting
  RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',

  // Server Errors
  INTERNAL_SERVER_ERROR: 'INTERNAL_SERVER_ERROR',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  DATABASE_ERROR: 'DATABASE_ERROR',
  EXTERNAL_SERVICE_ERROR: 'EXTERNAL_SERVICE_ERROR'
};
```

### Validation Errors

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "email": {
        "value": "invalid-email",
        "message": "Must be a valid email address",
        "rule": "email"
      },
      "age": {
        "value": -5,
        "message": "Must be a positive number",
        "rule": "min",
        "params": { "min": 0 }
      }
    }
  }
}
```

**Implementation:**
```javascript
const Joi = require('joi');

const userSchema = Joi.object({
  name: Joi.string().required(),
  email: Joi.string().email().required(),
  age: Joi.number().min(0).max(150)
});

app.post('/users', async (req, res, next) => {
  const { error, value } = userSchema.validate(req.body, {
    abortEarly: false
  });

  if (error) {
    const details = {};
    error.details.forEach(err => {
      details[err.path[0]] = {
        value: err.context.value,
        message: err.message,
        rule: err.type
      };
    });

    return res.status(422).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Request validation failed',
        details
      }
    });
  }

  // Process valid data
  const user = await createUser(value);
  res.status(201).json(user);
});
```

### Error Handling Best Practices

#### 1. Never Expose Internal Details

```javascript
// ❌ BAD - Exposes stack trace and internal paths
{
  "error": "Error: ENOENT: no such file or directory, open '/var/app/data/user-123.json'",
  "stack": "Error: ENOENT...\n    at Object.openSync (fs.js:476:3)\n    ..."
}

// ✅ GOOD - Generic message, log details server-side
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User not found",
    "requestId": "abc-123"
  }
}

// Server-side logging
console.error('File read error:', {
  error: err.message,
  stack: err.stack,
  path: filepath,
  requestId: req.id,
  userId: req.user?.id
});
```

#### 2. Provide Actionable Messages

```javascript
// ❌ BAD - Vague
{
  "error": "Invalid input"
}

// ✅ GOOD - Specific and actionable
{
  "error": {
    "code": "INVALID_EMAIL",
    "message": "The email address 'user@invalid' is not valid. Please provide a valid email address (e.g., user@example.com)",
    "field": "email",
    "value": "user@invalid"
  }
}
```

#### 3. Use Consistent Format

```javascript
// Always use the same structure
function formatError(code, message, statusCode, details = {}) {
  return {
    error: {
      code,
      message,
      ...(Object.keys(details).length > 0 && { details }),
      timestamp: new Date().toISOString()
    }
  };
}

// Validation error
res.status(422).json(formatError(
  'VALIDATION_ERROR',
  'Validation failed',
  422,
  validationErrors
));

// Not found error
res.status(404).json(formatError(
  'NOT_FOUND',
  'Resource not found',
  404
));
```

#### 4. Include Request ID for Debugging

```javascript
const { v4: uuidv4 } = require('uuid');

// Add request ID middleware
app.use((req, res, next) => {
  req.id = uuidv4();
  res.set('X-Request-ID', req.id);
  next();
});

// Include in error responses
app.use((err, req, res, next) => {
  res.status(err.statusCode || 500).json({
    error: {
      code: err.code,
      message: err.message,
      requestId: req.id
    }
  });
});

// Log with request ID
console.error(`[${req.id}] Error:`, err);
```

#### 5. Handle Different Error Types

```javascript
app.use((err, req, res, next) => {
  // Validation errors
  if (err.name === 'ValidationError') {
    return res.status(422).json(formatValidationError(err));
  }

  // Database errors
  if (err.name === 'SequelizeUniqueConstraintError') {
    return res.status(409).json({
      error: {
        code: 'DUPLICATE_ENTRY',
        message: 'A record with this value already exists',
        field: err.errors[0].path
      }
    });
  }

  // JWT errors
  if (err.name === 'JsonWebTokenError') {
    return res.status(401).json({
      error: {
        code: 'INVALID_TOKEN',
        message: 'Invalid authentication token'
      }
    });
  }

  if (err.name === 'TokenExpiredError') {
    return res.status(401).json({
      error: {
        code: 'TOKEN_EXPIRED',
        message: 'Authentication token has expired'
      }
    });
  }

  // Default
  res.status(500).json({
    error: {
      code: 'INTERNAL_SERVER_ERROR',
      message: 'An unexpected error occurred',
      requestId: req.id
    }
  });
});
```

---

## Authentication and Authorization

See also: [OAuth 2.0](../security/oauth2.md), [JWT](../security/jwt.md)

### Authentication Methods

#### 1. API Keys

```http
GET /users HTTP/1.1
X-API-Key: sk_live_abc123def456
```

**Pros:**
- Simple
- Easy to implement
- Good for server-to-server

**Cons:**
- No expiration
- Hard to rotate
- All-or-nothing permissions

**Implementation:**
```javascript
app.use('/api', (req, res, next) => {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey) {
    return res.status(401).json({
      error: 'API key required'
    });
  }

  const key = await db.query(
    'SELECT * FROM api_keys WHERE key = $1 AND active = true',
    [apiKey]
  );

  if (!key.rows[0]) {
    return res.status(401).json({
      error: 'Invalid API key'
    });
  }

  req.apiKey = key.rows[0];
  req.user = await db.getUser(key.rows[0].user_id);
  next();
});
```

#### 2. JWT (JSON Web Tokens)

```http
GET /users HTTP/1.1
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Pros:**
- Stateless
- Contains user info
- Expiration built-in
- Industry standard

**Cons:**
- Can't revoke easily
- Token size
- Need secure storage

**Implementation:**
```javascript
const jwt = require('jsonwebtoken');

// Generate token
app.post('/auth/login', async (req, res) => {
  const { email, password } = req.body;

  const user = await authenticateUser(email, password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  const token = jwt.sign(
    {
      userId: user.id,
      email: user.email,
      role: user.role
    },
    process.env.JWT_SECRET,
    { expiresIn: '1h' }
  );

  res.json({ token, expiresIn: 3600 });
});

// Verify token middleware
function authenticateJWT(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Token required' });
  }

  const token = authHeader.substring(7);

  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET);
    req.user = payload;
    next();
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired' });
    }
    return res.status(401).json({ error: 'Invalid token' });
  }
}

app.get('/profile', authenticateJWT, (req, res) => {
  res.json({ user: req.user });
});
```

#### 3. OAuth 2.0

See [OAuth 2.0 Guide](../security/oauth2.md) for complete implementation.

### Authorization

#### Role-Based Access Control (RBAC)

```javascript
const roles = {
  admin: ['read', 'write', 'delete', 'manage_users'],
  moderator: ['read', 'write', 'delete'],
  user: ['read', 'write'],
  guest: ['read']
};

function authorize(...requiredPermissions) {
  return (req, res, next) => {
    const userRole = req.user?.role || 'guest';
    const userPermissions = roles[userRole] || [];

    const hasPermission = requiredPermissions.every(permission =>
      userPermissions.includes(permission)
    );

    if (!hasPermission) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        required: requiredPermissions,
        current: userPermissions
      });
    }

    next();
  };
}

// Usage
app.get('/posts', authenticateJWT, authorize('read'), getPosts);
app.post('/posts', authenticateJWT, authorize('write'), createPost);
app.delete('/posts/:id', authenticateJWT, authorize('delete'), deletePost);
app.get('/admin/users', authenticateJWT, authorize('manage_users'), getUsers);
```

#### Resource-Based Authorization

```javascript
app.delete('/posts/:id', authenticateJWT, async (req, res) => {
  const post = await db.getPost(req.params.id);

  if (!post) {
    return res.status(404).json({ error: 'Post not found' });
  }

  // Check ownership or admin role
  const canDelete = post.authorId === req.user.userId ||
                    req.user.role === 'admin';

  if (!canDelete) {
    return res.status(403).json({
      error: 'You can only delete your own posts'
    });
  }

  await db.deletePost(req.params.id);
  res.status(204).send();
});
```

---

## HATEOAS Principles

HATEOAS (Hypermedia As The Engine Of Application State) is a constraint of REST that says clients interact with an application entirely through hypermedia provided dynamically by the application.

### Basic HATEOAS

```json
{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com",
  "_links": {
    "self": {
      "href": "/users/123"
    },
    "posts": {
      "href": "/users/123/posts"
    },
    "followers": {
      "href": "/users/123/followers"
    },
    "following": {
      "href": "/users/123/following"
    }
  }
}
```

### HAL (Hypertext Application Language)

```json
{
  "_links": {
    "self": { "href": "/orders/123" },
    "customer": { "href": "/customers/456" },
    "items": { "href": "/orders/123/items" }
  },
  "id": 123,
  "total": 99.99,
  "status": "pending",
  "_embedded": {
    "items": [
      {
        "_links": {
          "self": { "href": "/items/789" }
        },
        "id": 789,
        "name": "Product A",
        "price": 49.99
      }
    ]
  }
}
```

### Implementation

```javascript
function addLinks(resource, type, req) {
  const baseUrl = `${req.protocol}://${req.get('host')}`;

  resource._links = {
    self: { href: `${baseUrl}${req.path}` }
  };

  switch (type) {
    case 'user':
      resource._links.posts = {
        href: `${baseUrl}/users/${resource.id}/posts`
      };
      resource._links.followers = {
        href: `${baseUrl}/users/${resource.id}/followers`
      };
      break;

    case 'post':
      resource._links.author = {
        href: `${baseUrl}/users/${resource.authorId}`
      };
      resource._links.comments = {
        href: `${baseUrl}/posts/${resource.id}/comments`
      };
      break;
  }

  return resource;
}

app.get('/users/:id', async (req, res) => {
  const user = await db.getUser(req.params.id);
  res.json(addLinks(user, 'user', req));
});
```

---

## API Documentation

Good documentation is crucial for API adoption and developer experience.

### OpenAPI/Swagger

```yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
  description: API for managing users

paths:
  /users:
    get:
      summary: List users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'

    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '422':
          description: Validation error

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        email:
          type: string

    CreateUserRequest:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
        email:
          type: string
          format: email
```

### Generating OpenAPI Docs

```javascript
const swaggerJsdoc = require('swagger-jsdoc');
const swaggerUi = require('swagger-ui-express');

const options = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'User API',
      version: '1.0.0',
    },
  },
  apis: ['./routes/*.js'],
};

const specs = swaggerJsdoc(options);
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(specs));

/**
 * @swagger
 * /users:
 *   get:
 *     summary: Retrieve users
 *     responses:
 *       200:
 *         description: List of users
 */
app.get('/users', getUsers);
```

---

## Idempotency

Idempotent operations produce the same result regardless of how many times they're executed.

### Idempotent Methods

| Method | Idempotent | Example |
|--------|------------|---------|
| GET | Yes | Multiple reads return same data |
| PUT | Yes | Replacing resource produces same state |
| DELETE | Yes | Deleting already-deleted resource is safe |
| POST | No | Multiple creates = multiple resources |
| PATCH | Sometimes | Depends on implementation |

### Implementing Idempotency for POST

```javascript
const idempotencyCache = new Map();

app.post('/orders', async (req, res) => {
  const idempotencyKey = req.headers['idempotency-key'];

  if (!idempotencyKey) {
    return res.status(400).json({
      error: 'Idempotency-Key header required'
    });
  }

  // Check cache
  if (idempotencyCache.has(idempotencyKey)) {
    const cachedResponse = idempotencyCache.get(idempotencyKey);
    return res.status(cachedResponse.status).json(cachedResponse.body);
  }

  // Process request
  const order = await createOrder(req.body);

  // Cache response
  const response = { status: 201, body: order };
  idempotencyCache.set(idempotencyKey, response);

  // Expire after 24 hours
  setTimeout(() => {
    idempotencyCache.delete(idempotencyKey);
  }, 24 * 60 * 60 * 1000);

  res.status(201).json(order);
});
```

---

## Caching Strategies

### ETag

```http
GET /users/123 HTTP/1.1

HTTP/1.1 200 OK
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"

{ "id": 123, "name": "John" }
```

```http
GET /users/123 HTTP/1.1
If-None-Match: "33a64df551425fcc55e4d42a148795d9f25f89d4"

HTTP/1.1 304 Not Modified
```

### Cache-Control

```http
HTTP/1.1 200 OK
Cache-Control: public, max-age=3600
Expires: Wed, 15 Nov 2023 11:00:00 GMT
```

---

## Webhooks vs Polling

### Polling
```javascript
// Client polls every 30 seconds
setInterval(async () => {
  const status = await fetch('/api/job/123/status');
  if (status.completed) {
    // Process result
  }
}, 30000);
```

### Webhooks
```javascript
// Server notifies client when ready
app.post('/jobs', async (req, res) => {
  const job = await createJob(req.body);

  // Process asynchronously
  processJob(job.id).then(result => {
    // Notify via webhook
    fetch(req.body.webhookUrl, {
      method: 'POST',
      body: JSON.stringify({ jobId: job.id, result })
    });
  });

  res.status(202).json({ jobId: job.id });
});
```

---

## GraphQL vs REST Trade-offs

See also: [GraphQL Guide](graphql.md), [REST APIs](rest_apis.md)

| Aspect | REST | GraphQL |
|--------|------|---------|
| **Endpoints** | Multiple | Single |
| **Data Fetching** | Fixed responses | Client specifies |
| **Over-fetching** | Common | Eliminated |
| **Under-fetching** | Requires multiple requests | Single request |
| **Caching** | Easy (HTTP) | Complex |
| **Learning Curve** | Low | Moderate |
| **Tooling** | Mature | Growing |

---

## gRPC Use Cases

See also: [gRPC Guide](grpc.md)

**When to Use gRPC:**
- Microservices communication
- Real-time streaming
- Performance-critical applications
- Polyglot environments

**When to Use REST:**
- Public APIs
- Browser clients
- Simple CRUD operations

---

## API Gateway Patterns

API gateways provide a single entry point for API requests.

**Features:**
- Request routing
- Authentication/Authorization
- Rate limiting
- Caching
- Protocol translation
- Analytics

```
Client → API Gateway → [Microservice A, Microservice B, Microservice C]
```

---

## Backward Compatibility

**Non-Breaking Changes:**
- Adding optional fields
- Adding new endpoints
- Adding optional parameters

**Breaking Changes:**
- Removing fields
- Renaming fields
- Changing field types
- Removing endpoints

---

## Deprecation Strategies

```http
HTTP/1.1 200 OK
Sunset: Sat, 31 Dec 2023 23:59:59 GMT
Deprecation: true
Link: </v2/users>; rel="successor-version"
Warning: 299 - "This endpoint is deprecated"
```

---

## Best Practices Summary

1. **Use nouns for resources, not verbs**
2. **Proper HTTP methods and status codes**
3. **Versioning from day one**
4. **Comprehensive error handling**
5. **Authentication and authorization**
6. **Rate limiting**
7. **Pagination for large datasets**
8. **Caching when appropriate**
9. **Thorough documentation**
10. **Consistent response formats**

---

## References

- [REST APIs](rest_apis.md)
- [GraphQL](graphql.md)
- [gRPC](grpc.md)
- [OAuth 2.0](../security/oauth2.md)
- [JWT](../security/jwt.md)
- [RESTful API Design](https://restfulapi.net/)
- [HTTP Status Codes](https://httpstat.us/)
- [OpenAPI Specification](https://swagger.io/specification/)
