# JWT (JSON Web Tokens)

JSON Web Token (JWT) is an open standard (RFC 7519) that defines a compact and self-contained way for securely transmitting information between parties as a JSON object. This information can be verified and trusted because it is digitally signed.

## Table of Contents
- [Introduction](#introduction)
- [JWT Structure](#jwt-structure)
- [How JWT Works](#how-jwt-works)
- [Creating and Verifying JWTs](#creating-and-verifying-jwts)
- [JWT Authentication Flow](#jwt-authentication-flow)
- [Refresh Tokens](#refresh-tokens)
- [Security Best Practices](#security-best-practices)
- [Common Vulnerabilities](#common-vulnerabilities)

---

## Introduction

**What is JWT?**
A JWT is a string of three Base64-URL encoded parts separated by dots (`.`), representing:
1. Header
2. Payload
3. Signature

**Example JWT:**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

**Use Cases:**
- Authentication
- Information Exchange
- Authorization
- Single Sign-On (SSO)
- Stateless API authentication

**Benefits:**
- Compact size
- Self-contained (all info in the token)
- Stateless (no server-side session storage)
- Cross-domain/CORS friendly
- Mobile-friendly

---

## JWT Structure

### Header

Contains the type of token and the signing algorithm.

```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**Common Algorithms:**
- `HS256` (HMAC with SHA-256) - Symmetric
- `RS256` (RSA with SHA-256) - Asymmetric
- `ES256` (ECDSA with SHA-256) - Asymmetric

### Payload

Contains the claims (statements about an entity and additional data).

```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "email": "john@example.com",
  "iat": 1516239022,
  "exp": 1516242622,
  "roles": ["user", "admin"]
}
```

**Registered Claims:**
- `iss` (issuer)
- `sub` (subject)
- `aud` (audience)
- `exp` (expiration time)
- `nbf` (not before)
- `iat` (issued at)
- `jti` (JWT ID)

**Custom Claims:**
Any additional data you want to include.

### Signature

Created by taking:
```
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  secret
)
```

---

## How JWT Works

### Authentication Flow

```
1. User logs in with credentials
   ↓
2. Server validates credentials
   ↓
3. Server creates JWT with user info
   ↓
4. Server sends JWT to client
   ↓
5. Client stores JWT (localStorage/cookie)
   ↓
6. Client sends JWT with each request
   ↓
7. Server verifies JWT signature
   ↓
8. Server grants/denies access
```

---

## Creating and Verifying JWTs

### Node.js Implementation

**Installation:**
```bash
npm install jsonwebtoken
```

**Creating a JWT:**
```javascript
const jwt = require('jsonwebtoken');

const SECRET_KEY = process.env.JWT_SECRET;

function generateToken(user) {
  const payload = {
    sub: user.id,
    email: user.email,
    name: user.name,
    roles: user.roles,
  };

  const options = {
    expiresIn: '1h',
    issuer: 'your-app-name',
    audience: 'your-app-users',
  };

  return jwt.sign(payload, SECRET_KEY, options);
}

// Usage
const token = generateToken({
  id: 123,
  email: 'john@example.com',
  name: 'John Doe',
  roles: ['user'],
});

console.log(token);
```

**Verifying a JWT:**
```javascript
function verifyToken(token) {
  try {
    const decoded = jwt.verify(token, SECRET_KEY, {
      issuer: 'your-app-name',
      audience: 'your-app-users',
    });

    return decoded;
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      throw new Error('Token expired');
    }
    if (error.name === 'JsonWebTokenError') {
      throw new Error('Invalid token');
    }
    throw error;
  }
}

// Usage
try {
  const decoded = verifyToken(token);
  console.log('User:', decoded);
} catch (error) {
  console.error('Verification failed:', error.message);
}
```

### Express Middleware

```javascript
const jwt = require('jsonwebtoken');

function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  try {
    const user = jwt.verify(token, process.env.JWT_SECRET);
    req.user = user;
    next();
  } catch (error) {
    return res.status(403).json({ error: 'Invalid or expired token' });
  }
}

// Protected route
app.get('/api/protected', authenticateToken, (req, res) => {
  res.json({
    message: 'Protected data',
    user: req.user,
  });
});
```

### Python Implementation (PyJWT)

```python
import jwt
import datetime
from functools import wraps
from flask import request, jsonify

SECRET_KEY = "your-secret-key"

def generate_token(user_id, email):
    payload = {
        'sub': user_id,
        'email': email,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
        'iat': datetime.datetime.utcnow(),
        'iss': 'your-app-name'
    }

    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_token(token):
    try:
        decoded = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=['HS256'],
            issuer='your-app-name'
        )
        return decoded
    except jwt.ExpiredSignatureError:
        raise Exception('Token expired')
    except jwt.InvalidTokenError:
        raise Exception('Invalid token')

# Decorator for protected routes
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'error': 'Token missing'}), 401

        try:
            token = token.split(' ')[1]  # Remove 'Bearer '
            decoded = verify_token(token)
            request.user = decoded
        except Exception as e:
            return jsonify({'error': str(e)}), 403

        return f(*args, **kwargs)

    return decorated

# Protected route
@app.route('/api/protected')
@token_required
def protected():
    return jsonify({
        'message': 'Protected data',
        'user': request.user
    })
```

---

## JWT Authentication Flow

### Complete Implementation

**auth.js:**
```javascript
const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const router = express.Router();

const SECRET_KEY = process.env.JWT_SECRET;
const REFRESH_SECRET = process.env.REFRESH_SECRET;

// Login
router.post('/login', async (req, res) => {
  const { email, password } = req.body;

  // Find user in database
  const user = await User.findOne({ email });

  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // Verify password
  const isValidPassword = await bcrypt.compare(password, user.password);

  if (!isValidPassword) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // Generate tokens
  const accessToken = jwt.sign(
    {
      sub: user.id,
      email: user.email,
      roles: user.roles,
    },
    SECRET_KEY,
    { expiresIn: '15m' }
  );

  const refreshToken = jwt.sign(
    { sub: user.id },
    REFRESH_SECRET,
    { expiresIn: '7d' }
  );

  // Store refresh token in database
  await RefreshToken.create({
    token: refreshToken,
    userId: user.id,
    expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
  });

  // Send tokens
  res.json({
    accessToken,
    refreshToken,
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
    },
  });
});

// Refresh token
router.post('/refresh', async (req, res) => {
  const { refreshToken } = req.body;

  if (!refreshToken) {
    return res.status(401).json({ error: 'Refresh token required' });
  }

  try {
    // Verify refresh token
    const decoded = jwt.verify(refreshToken, REFRESH_SECRET);

    // Check if refresh token exists in database
    const storedToken = await RefreshToken.findOne({
      token: refreshToken,
      userId: decoded.sub,
    });

    if (!storedToken) {
      return res.status(403).json({ error: 'Invalid refresh token' });
    }

    // Get user
    const user = await User.findById(decoded.sub);

    // Generate new access token
    const accessToken = jwt.sign(
      {
        sub: user.id,
        email: user.email,
        roles: user.roles,
      },
      SECRET_KEY,
      { expiresIn: '15m' }
    );

    res.json({ accessToken });
  } catch (error) {
    return res.status(403).json({ error: 'Invalid refresh token' });
  }
});

// Logout
router.post('/logout', authenticateToken, async (req, res) => {
  const { refreshToken } = req.body;

  // Remove refresh token from database
  await RefreshToken.deleteOne({
    token: refreshToken,
    userId: req.user.sub,
  });

  res.json({ message: 'Logged out successfully' });
});

module.exports = router;
```

### Client-Side Implementation

```javascript
class AuthService {
  constructor() {
    this.accessToken = null;
    this.refreshToken = localStorage.getItem('refreshToken');
  }

  async login(email, password) {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      throw new Error('Login failed');
    }

    const data = await response.json();
    this.accessToken = data.accessToken;
    this.refreshToken = data.refreshToken;

    localStorage.setItem('refreshToken', data.refreshToken);

    return data.user;
  }

  async refreshAccessToken() {
    if (!this.refreshToken) {
      throw new Error('No refresh token');
    }

    const response = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refreshToken: this.refreshToken }),
    });

    if (!response.ok) {
      this.logout();
      throw new Error('Token refresh failed');
    }

    const data = await response.json();
    this.accessToken = data.accessToken;

    return this.accessToken;
  }

  async makeAuthenticatedRequest(url, options = {}) {
    if (!this.accessToken) {
      await this.refreshAccessToken();
    }

    const response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${this.accessToken}`,
      },
    });

    // If token expired, refresh and retry
    if (response.status === 401) {
      await this.refreshAccessToken();

      return fetch(url, {
        ...options,
        headers: {
          ...options.headers,
          'Authorization': `Bearer ${this.accessToken}`,
        },
      });
    }

    return response;
  }

  logout() {
    this.accessToken = null;
    this.refreshToken = null;
    localStorage.removeItem('refreshToken');
  }
}

// Usage
const auth = new AuthService();

// Login
await auth.login('user@example.com', 'password');

// Make authenticated request
const response = await auth.makeAuthenticatedRequest('/api/user/profile');
const profile = await response.json();

// Logout
auth.logout();
```

---

## Refresh Tokens

### Why Use Refresh Tokens?

1. **Short-lived access tokens** reduce the window of opportunity for token theft
2. **Long-lived refresh tokens** improve user experience (don't have to login frequently)
3. **Revocable** - Can invalidate refresh tokens without affecting other sessions

### Implementation Strategy

```javascript
// Token lifetimes
const ACCESS_TOKEN_LIFETIME = '15m';
const REFRESH_TOKEN_LIFETIME = '7d';

// Store refresh tokens in database
const refreshTokenSchema = new mongoose.Schema({
  token: { type: String, required: true, unique: true },
  userId: { type: ObjectId, ref: 'User', required: true },
  expiresAt: { type: Date, required: true },
  createdAt: { type: Date, default: Date.now },
  revokedAt: { type: Date },
  replacedByToken: { type: String },
});

// Automatic cleanup of expired tokens
refreshTokenSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

// Token rotation
async function rotateRefreshToken(oldRefreshToken) {
  // Verify old token
  const decoded = jwt.verify(oldRefreshToken, REFRESH_SECRET);

  // Find old token in database
  const oldToken = await RefreshToken.findOne({
    token: oldRefreshToken,
    userId: decoded.sub,
  });

  if (!oldToken || oldToken.revokedAt) {
    throw new Error('Invalid refresh token');
  }

  // Create new refresh token
  const newRefreshToken = jwt.sign(
    { sub: decoded.sub },
    REFRESH_SECRET,
    { expiresIn: REFRESH_TOKEN_LIFETIME }
  );

  // Mark old token as revoked
  oldToken.revokedAt = new Date();
  oldToken.replacedByToken = newRefreshToken;
  await oldToken.save();

  // Store new token
  await RefreshToken.create({
    token: newRefreshToken,
    userId: decoded.sub,
    expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
  });

  return newRefreshToken;
}
```

---

## Security Best Practices

### 1. Use Strong Secrets

```javascript
// Generate a strong secret
const crypto = require('crypto');
const secret = crypto.randomBytes(64).toString('hex');

// Use environment variables
const SECRET_KEY = process.env.JWT_SECRET;

if (!SECRET_KEY || SECRET_KEY.length < 32) {
  throw new Error('JWT_SECRET must be at least 32 characters');
}
```

### 2. Short Expiration Times

```javascript
// Short-lived access tokens
const accessToken = jwt.sign(payload, SECRET_KEY, {
  expiresIn: '15m', // 15 minutes
});

// Long-lived refresh tokens
const refreshToken = jwt.sign(payload, REFRESH_SECRET, {
  expiresIn: '7d', // 7 days
});
```

### 3. Secure Token Storage

```javascript
// ❌ Bad: localStorage (vulnerable to XSS)
localStorage.setItem('token', accessToken);

// ✅ Good: httpOnly cookie (protected from XSS)
res.cookie('access_token', accessToken, {
  httpOnly: true,
  secure: true,
  sameSite: 'strict',
  maxAge: 15 * 60 * 1000, // 15 minutes
});

// ✅ Good: Memory (for SPAs)
class TokenStore {
  constructor() {
    this.token = null;
  }

  setToken(token) {
    this.token = token;
  }

  getToken() {
    return this.token;
  }

  clearToken() {
    this.token = null;
  }
}
```

### 4. Validate All Claims

```javascript
function validateToken(token) {
  const decoded = jwt.verify(token, SECRET_KEY, {
    issuer: 'your-app',
    audience: 'your-users',
  });

  // Check expiration
  if (decoded.exp < Date.now() / 1000) {
    throw new Error('Token expired');
  }

  // Check not before
  if (decoded.nbf && decoded.nbf > Date.now() / 1000) {
    throw new Error('Token not yet valid');
  }

  // Validate custom claims
  if (!decoded.roles || !Array.isArray(decoded.roles)) {
    throw new Error('Invalid token structure');
  }

  return decoded;
}
```

### 5. Use Asymmetric Algorithms for Distributed Systems

```javascript
const fs = require('fs');

// Generate RSA key pair
const { generateKeyPairSync } = require('crypto');
const { privateKey, publicKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
});

// Sign with private key
const token = jwt.sign(payload, privateKey, {
  algorithm: 'RS256',
  expiresIn: '1h',
});

// Verify with public key (can be shared with other services)
const decoded = jwt.verify(token, publicKey, {
  algorithms: ['RS256'],
});
```

### 6. Implement Token Blacklist for Logout

```javascript
const blacklist = new Set();

async function logout(token) {
  const decoded = jwt.decode(token);

  // Add to blacklist with expiration
  await redis.setex(
    `blacklist:${decoded.jti}`,
    decoded.exp - Math.floor(Date.now() / 1000),
    'true'
  );
}

async function isTokenBlacklisted(token) {
  const decoded = jwt.decode(token);
  const isBlacklisted = await redis.exists(`blacklist:${decoded.jti}`);
  return isBlacklisted === 1;
}

// Middleware
async function authenticateToken(req, res, next) {
  const token = extractToken(req);

  if (await isTokenBlacklisted(token)) {
    return res.status(401).json({ error: 'Token has been revoked' });
  }

  // Verify token...
  next();
}
```

---

## Common Vulnerabilities

### 1. Algorithm Confusion Attack

**Vulnerability:**
Attacker changes algorithm from RS256 to HS256 and uses public key as secret

**Mitigation:**
```javascript
// Always specify allowed algorithms
jwt.verify(token, secret, {
  algorithms: ['RS256'], // Only allow RS256
});

// Never use 'none' algorithm
jwt.sign(payload, secret, {
  algorithm: 'HS256', // Specify algorithm explicitly
});
```

### 2. Weak Secret Keys

**Vulnerability:**
Short or predictable secrets can be brute-forced

**Mitigation:**
```javascript
// Use strong, random secrets (at least 256 bits)
const crypto = require('crypto');
const secret = crypto.randomBytes(32).toString('hex');

// Store in environment variables
const SECRET_KEY = process.env.JWT_SECRET;

// Validate secret strength
if (SECRET_KEY.length < 32) {
  throw new Error('Secret key too short');
}
```

### 3. Token Leakage in URLs

**Vulnerability:**
Tokens in URL parameters are logged and visible

**Mitigation:**
```javascript
// ❌ Bad: Token in URL
fetch(`/api/data?token=${accessToken}`);

// ✅ Good: Token in header
fetch('/api/data', {
  headers: {
    'Authorization': `Bearer ${accessToken}`,
  },
});
```

### 4. Missing Expiration

**Vulnerability:**
Tokens without expiration never expire

**Mitigation:**
```javascript
// Always set expiration
const token = jwt.sign(payload, secret, {
  expiresIn: '15m',
});

// Verify expiration on the server
jwt.verify(token, secret, {
  clockTolerance: 0, // No tolerance for expired tokens
});
```

### 5. XSS Attacks

**Vulnerability:**
Tokens stored in localStorage can be stolen via XSS

**Mitigation:**
```javascript
// Use httpOnly cookies
res.cookie('token', token, {
  httpOnly: true,
  secure: true,
  sameSite: 'strict',
});

// Or store in memory (for SPAs)
// Never use localStorage or sessionStorage for sensitive tokens
```

### 6. Insufficient Token Validation

**Vulnerability:**
Not validating all claims or checking token blacklist

**Mitigation:**
```javascript
async function validateToken(token) {
  // 1. Verify signature
  const decoded = jwt.verify(token, SECRET_KEY);

  // 2. Check blacklist
  if (await isBlacklisted(decoded.jti)) {
    throw new Error('Token revoked');
  }

  // 3. Validate issuer
  if (decoded.iss !== EXPECTED_ISSUER) {
    throw new Error('Invalid issuer');
  }

  // 4. Validate audience
  if (decoded.aud !== EXPECTED_AUDIENCE) {
    throw new Error('Invalid audience');
  }

  // 5. Additional business logic checks
  const user = await User.findById(decoded.sub);
  if (!user || !user.isActive) {
    throw new Error('User not found or inactive');
  }

  return decoded;
}
```

---

## Resources

**Official Specifications:**
- [RFC 7519 - JWT](https://tools.ietf.org/html/rfc7519)
- [JWT.io](https://jwt.io/)

**Libraries:**
- [jsonwebtoken](https://github.com/auth0/node-jsonwebtoken) (Node.js)
- [PyJWT](https://pyjwt.readthedocs.io/) (Python)
- [jose](https://github.com/panva/jose) (Node.js, modern)
- [java-jwt](https://github.com/auth0/java-jwt) (Java)

**Tools:**
- [JWT Debugger](https://jwt.io/#debugger)
- [JWT Inspector](https://www.jwtinspector.io/)

**Learning Resources:**
- [Introduction to JWT](https://jwt.io/introduction)
- [JWT Handbook](https://auth0.com/resources/ebooks/jwt-handbook)
- [OWASP JWT Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html)

**Security:**
- [JWT Security Best Practices](https://curity.io/resources/learn/jwt-best-practices/)
- [Common JWT Security Mistakes](https://pragmaticwebsecurity.com/articles/apisecurity/jwt-security.html)
