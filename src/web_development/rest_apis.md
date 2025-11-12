# REST APIs

## Overview

REST (Representational State Transfer) is an architectural style for building web services using HTTP.

## Core Principles

1. **Client-Server**: Separation of concerns
2. **Stateless**: Each request contains all info
3. **Uniform Interface**: Consistent API design
4. **Cacheable**: Responses can be cached
5. **Layered**: Client unaware of layers

## HTTP Methods

| Method | Purpose | Idempotent |
|--------|---------|-----------|
| **GET** | Retrieve resource | ✓ |
| **POST** | Create resource | ✗ |
| **PUT** | Replace resource | ✓ |
| **PATCH** | Partial update | ✗ |
| **DELETE** | Remove resource | ✓ |

## Status Codes

- **2xx**: Success (200 OK, 201 Created)
- **3xx**: Redirection (301, 304)
- **4xx**: Client error (400, 404, 401)
- **5xx**: Server error (500, 503)

## Resource-Oriented Design

```
✓ GET /users          - List users
✓ POST /users         - Create user
✓ GET /users/123      - Get user 123
✓ PUT /users/123      - Replace user 123
✓ PATCH /users/123    - Partial update
✓ DELETE /users/123   - Delete user 123

✗ GET /getUser?id=123 - Procedural (bad)
```

## Request/Response

```python
# Request
GET /api/v1/users/123 HTTP/1.1
Host: api.example.com
Authorization: Bearer token
Content-Type: application/json

# Response
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": 123,
  "name": "John",
  "email": "john@example.com"
}
```

## Error Handling

```json
{
  "error": "Validation failed",
  "details": {
    "email": "Invalid email format"
  },
  "status": 400
}
```

## Pagination

```
GET /users?page=2&limit=20
GET /users?offset=40&limit=20
GET /users?cursor=abc123
```

## Versioning

```
/api/v1/users    (stable)
/api/v2/users    (new version)
/api/beta/users  (experimental)
```

## Best Practices

1. **Use appropriate methods** for operations
2. **Meaningful status codes** for responses
3. **Consistent naming** conventions
4. **Pagination** for large datasets
5. **Rate limiting** to protect API
6. **Authentication/Authorization**
7. **Documentation** (Swagger/OpenAPI)

## Express.js Example

```javascript
const express = require('express');
const app = express();

// Get all users
app.get('/users', (req, res) => {
  res.json(users);
});

// Get user by ID
app.get('/users/:id', (req, res) => {
  const user = users.find(u => u.id == req.params.id);
  res.json(user);
});

// Create user
app.post('/users', (req, res) => {
  const user = req.body;
  users.push(user);
  res.status(201).json(user);
});

// Update user
app.patch('/users/:id', (req, res) => {
  const user = users.find(u => u.id == req.params.id);
  Object.assign(user, req.body);
  res.json(user);
});

// Delete user
app.delete('/users/:id', (req, res) => {
  users = users.filter(u => u.id != req.params.id);
  res.status(204).send();
});

app.listen(3000);
```

## Testing

```bash
# Using curl
curl -X GET http://localhost:3000/users
curl -X POST http://localhost:3000/users -H "Content-Type: application/json" -d '{"name":"John"}'
```

## ELI10

REST API is like a restaurant menu:
- **GET**: View menu/food
- **POST**: Place new order
- **PUT**: Replace entire order
- **PATCH**: Modify order slightly
- **DELETE**: Cancel order

Standard ways to order without confusion!

## Further Resources

- [REST API Best Practices](https://restfulapi.net/)
- [HTTP Status Codes](https://httpstat.us/)
- [OpenAPI/Swagger](https://swagger.io/)
