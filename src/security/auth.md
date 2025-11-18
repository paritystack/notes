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
- [Single Sign-On (SSO)](#single-sign-on-sso)
- [Biometric Authentication](#biometric-authentication)
- [Authentication Patterns](#authentication-patterns)
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
