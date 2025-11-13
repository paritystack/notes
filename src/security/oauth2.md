# OAuth 2.0

OAuth 2.0 is an industry-standard authorization framework that enables applications to obtain limited access to user accounts on an HTTP service. It works by delegating user authentication to the service that hosts the user account and authorizing third-party applications to access the user account.

## Table of Contents
- [Introduction](#introduction)
- [OAuth 2.0 Roles](#oauth-20-roles)
- [Grant Types](#grant-types)
- [Authorization Code Flow](#authorization-code-flow)
- [Client Credentials Flow](#client-credentials-flow)
- [Implementing OAuth 2.0](#implementing-oauth-20)
- [OAuth 2.0 Providers](#oauth-20-providers)
- [Security Best Practices](#security-best-practices)
- [Common Vulnerabilities](#common-vulnerabilities)

---

## Introduction

**What is OAuth 2.0?**
OAuth 2.0 is an authorization framework, not an authentication protocol. It allows users to grant limited access to their resources on one site to another site, without sharing their credentials.

**Key Benefits:**
- Users don't share passwords with third-party apps
- Fine-grained access control (scopes)
- Time-limited access through tokens
- Revocable access
- Industry standard with wide support

**Use Cases:**
- Social login (Sign in with Google, Facebook, etc.)
- API access delegation
- Third-party application integration
- Microservices authentication
- Mobile app authentication

---

## OAuth 2.0 Roles

### 1. Resource Owner
The user who owns the data and can grant access to it.

```
Example: John who has a Google account with photos
```

### 2. Client
The application requesting access to resources.

```
Example: A photo printing service that wants access to John's photos
```

### 3. Authorization Server
Server that authenticates the resource owner and issues access tokens.

```
Example: Google's OAuth 2.0 authorization server
```

### 4. Resource Server
Server hosting the protected resources.

```
Example: Google Photos API server
```

---

## Grant Types

### 1. Authorization Code Flow

**Best for:** Server-side web applications

**Flow:**
```
1. Client redirects user to authorization server
2. User authenticates and grants permission
3. Authorization server redirects back with authorization code
4. Client exchanges code for access token
5. Client uses access token to access resources
```

**Benefits:**
- Most secure flow
- Refresh tokens supported
- Client secret never exposed to browser

### 2. Implicit Flow (Deprecated)

**Status:** Not recommended for new applications

**Flow:**
```
1. Client redirects user to authorization server
2. User authenticates and grants permission
3. Authorization server redirects with access token in URL fragment
```

**Issues:**
- Token exposed in browser history
- No refresh token
- Less secure

### 3. Client Credentials Flow

**Best for:** Server-to-server communication

**Flow:**
```
1. Client authenticates with client_id and client_secret
2. Authorization server returns access token
3. Client uses access token for API calls
```

**Use cases:**
- Microservices communication
- Batch jobs
- CLI tools

### 4. Resource Owner Password Credentials (Not Recommended)

**Flow:**
```
1. User provides username and password to client
2. Client exchanges credentials for access token
```

**Issues:**
- User shares credentials with client
- Defeats OAuth purpose
- Only for legacy systems

### 5. PKCE (Proof Key for Code Exchange)

**Best for:** Mobile and SPA applications

**Enhancement to Authorization Code Flow:**
```
1. Client generates code_verifier (random string)
2. Client creates code_challenge = hash(code_verifier)
3. Authorization request includes code_challenge
4. Token request includes code_verifier
5. Server verifies code_challenge matches code_verifier
```

**Benefits:**
- Protects against authorization code interception
- No client secret needed
- Secure for public clients

---

## Authorization Code Flow

### Step-by-Step Implementation

#### Step 1: Authorization Request

```http
GET /authorize?
  response_type=code&
  client_id=YOUR_CLIENT_ID&
  redirect_uri=https://yourapp.com/callback&
  scope=read:user read:email&
  state=random_string
  HTTP/1.1
Host: authorization-server.com
```

**Parameters:**
- `response_type`: Set to "code"
- `client_id`: Your application's client ID
- `redirect_uri`: Where to redirect after authorization
- `scope`: Requested permissions
- `state`: Random string to prevent CSRF

#### Step 2: User Authorization

User sees consent screen and approves/denies access.

#### Step 3: Authorization Response

```http
HTTP/1.1 302 Found
Location: https://yourapp.com/callback?
  code=AUTH_CODE&
  state=random_string
```

#### Step 4: Token Request

```http
POST /token HTTP/1.1
Host: authorization-server.com
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=AUTH_CODE&
redirect_uri=https://yourapp.com/callback&
client_id=YOUR_CLIENT_ID&
client_secret=YOUR_CLIENT_SECRET
```

#### Step 5: Token Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here",
  "scope": "read:user read:email"
}
```

#### Step 6: Using Access Token

```http
GET /api/user HTTP/1.1
Host: api.example.com
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Node.js Implementation (Express)

```javascript
const express = require('express');
const axios = require('axios');
const crypto = require('crypto');

const app = express();

const CLIENT_ID = process.env.CLIENT_ID;
const CLIENT_SECRET = process.env.CLIENT_SECRET;
const REDIRECT_URI = 'http://localhost:3000/callback';
const AUTHORIZATION_URL = 'https://authorization-server.com/authorize';
const TOKEN_URL = 'https://authorization-server.com/token';

// Step 1: Initiate authorization
app.get('/login', (req, res) => {
  const state = crypto.randomBytes(16).toString('hex');
  req.session.state = state;

  const authUrl = new URL(AUTHORIZATION_URL);
  authUrl.searchParams.append('response_type', 'code');
  authUrl.searchParams.append('client_id', CLIENT_ID);
  authUrl.searchParams.append('redirect_uri', REDIRECT_URI);
  authUrl.searchParams.append('scope', 'read:user read:email');
  authUrl.searchParams.append('state', state);

  res.redirect(authUrl.toString());
});

// Step 2: Handle callback
app.get('/callback', async (req, res) => {
  const { code, state } = req.query;

  // Verify state
  if (state !== req.session.state) {
    return res.status(400).send('Invalid state');
  }

  try {
    // Exchange code for token
    const tokenResponse = await axios.post(TOKEN_URL, {
      grant_type: 'authorization_code',
      code,
      redirect_uri: REDIRECT_URI,
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
    });

    const { access_token, refresh_token } = tokenResponse.data;

    // Store tokens securely
    req.session.access_token = access_token;
    req.session.refresh_token = refresh_token;

    res.redirect('/dashboard');
  } catch (error) {
    res.status(500).send('Authentication failed');
  }
});

// Step 3: Use access token
app.get('/api/user', async (req, res) => {
  const { access_token } = req.session;

  if (!access_token) {
    return res.status(401).send('Not authenticated');
  }

  try {
    const userResponse = await axios.get('https://api.example.com/user', {
      headers: {
        Authorization: `Bearer ${access_token}`,
      },
    });

    res.json(userResponse.data);
  } catch (error) {
    if (error.response?.status === 401) {
      // Token expired, refresh it
      return res.redirect('/refresh');
    }
    res.status(500).send('Failed to fetch user');
  }
});

// Refresh token
app.get('/refresh', async (req, res) => {
  const { refresh_token } = req.session;

  try {
    const tokenResponse = await axios.post(TOKEN_URL, {
      grant_type: 'refresh_token',
      refresh_token,
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
    });

    req.session.access_token = tokenResponse.data.access_token;

    res.redirect('/dashboard');
  } catch (error) {
    res.redirect('/login');
  }
});
```

---

## Client Credentials Flow

### Implementation Example

```javascript
const axios = require('axios');

async function getAccessToken() {
  const response = await axios.post(
    'https://authorization-server.com/token',
    {
      grant_type: 'client_credentials',
      client_id: process.env.CLIENT_ID,
      client_secret: process.env.CLIENT_SECRET,
      scope: 'api:read api:write',
    },
    {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    }
  );

  return response.data.access_token;
}

async function callAPI() {
  const token = await getAccessToken();

  const apiResponse = await axios.get('https://api.example.com/data', {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  return apiResponse.data;
}

// Usage
callAPI()
  .then(data => console.log(data))
  .catch(error => console.error(error));
```

---

## Implementing OAuth 2.0

### Building an OAuth 2.0 Server

**Using Node.js with oauth2-server:**

```bash
npm install express oauth2-server
```

**server.js:**
```javascript
const express = require('express');
const OAuth2Server = require('oauth2-server');
const Request = OAuth2Server.Request;
const Response = OAuth2Server.Response;

const app = express();

// OAuth2 model
const model = {
  getClient: async (clientId, clientSecret) => {
    // Fetch client from database
    const client = await db.clients.findOne({ clientId });

    if (!client || (clientSecret && client.clientSecret !== clientSecret)) {
      return null;
    }

    return {
      id: client.id,
      grants: ['authorization_code', 'refresh_token'],
      redirectUris: client.redirectUris,
    };
  },

  saveToken: async (token, client, user) => {
    // Save token to database
    await db.tokens.create({
      accessToken: token.accessToken,
      accessTokenExpiresAt: token.accessTokenExpiresAt,
      refreshToken: token.refreshToken,
      refreshTokenExpiresAt: token.refreshTokenExpiresAt,
      client: client.id,
      user: user.id,
    });

    return token;
  },

  getAccessToken: async (accessToken) => {
    const token = await db.tokens.findOne({ accessToken });

    if (!token) return null;

    return {
      accessToken: token.accessToken,
      accessTokenExpiresAt: token.accessTokenExpiresAt,
      client: { id: token.client },
      user: { id: token.user },
    };
  },

  getAuthorizationCode: async (authorizationCode) => {
    const code = await db.authCodes.findOne({ code: authorizationCode });

    if (!code) return null;

    return {
      code: code.code,
      expiresAt: code.expiresAt,
      redirectUri: code.redirectUri,
      client: { id: code.client },
      user: { id: code.user },
    };
  },

  saveAuthorizationCode: async (code, client, user) => {
    await db.authCodes.create({
      code: code.authorizationCode,
      expiresAt: code.expiresAt,
      redirectUri: code.redirectUri,
      client: client.id,
      user: user.id,
    });

    return code;
  },

  revokeAuthorizationCode: async (code) => {
    await db.authCodes.delete({ code: code.code });
    return true;
  },

  verifyScope: async (token, scope) => {
    if (!token.scope) return false;
    const requestedScopes = scope.split(' ');
    const authorizedScopes = token.scope.split(' ');
    return requestedScopes.every(s => authorizedScopes.includes(s));
  },
};

const oauth = new OAuth2Server({
  model: model,
  accessTokenLifetime: 3600,
  allowBearerTokensInQueryString: true,
});

// Authorization endpoint
app.get('/authorize', async (req, res) => {
  const request = new Request(req);
  const response = new Response(res);

  try {
    // Authenticate user (implement your own logic)
    const user = await authenticateUser(req);

    if (!user) {
      return res.redirect('/login');
    }

    const code = await oauth.authorize(request, response, {
      authenticateHandler: {
        handle: () => user,
      },
    });

    res.redirect(`${code.redirectUri}?code=${code.authorizationCode}&state=${req.query.state}`);
  } catch (error) {
    res.status(error.code || 500).json(error);
  }
});

// Token endpoint
app.post('/token', async (req, res) => {
  const request = new Request(req);
  const response = new Response(res);

  try {
    const token = await oauth.token(request, response);
    res.json(token);
  } catch (error) {
    res.status(error.code || 500).json(error);
  }
});

// Protected resource
app.get('/api/resource', async (req, res) => {
  const request = new Request(req);
  const response = new Response(res);

  try {
    const token = await oauth.authenticate(request, response);
    res.json({ message: 'Protected resource', user: token.user });
  } catch (error) {
    res.status(error.code || 401).json({ error: 'Unauthorized' });
  }
});
```

---

## OAuth 2.0 Providers

### Google OAuth 2.0

```javascript
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;

passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: "http://localhost:3000/auth/google/callback"
  },
  function(accessToken, refreshToken, profile, cb) {
    // Find or create user in your database
    User.findOrCreate({ googleId: profile.id }, function (err, user) {
      return cb(err, user);
    });
  }
));

app.get('/auth/google',
  passport.authenticate('google', { scope: ['profile', 'email'] })
);

app.get('/auth/google/callback',
  passport.authenticate('google', { failureRedirect: '/login' }),
  function(req, res) {
    res.redirect('/dashboard');
  }
);
```

### GitHub OAuth 2.0

```javascript
const GitHubStrategy = require('passport-github2').Strategy;

passport.use(new GitHubStrategy({
    clientID: process.env.GITHUB_CLIENT_ID,
    clientSecret: process.env.GITHUB_CLIENT_SECRET,
    callbackURL: "http://localhost:3000/auth/github/callback"
  },
  function(accessToken, refreshToken, profile, done) {
    User.findOrCreate({ githubId: profile.id }, function (err, user) {
      return done(err, user);
    });
  }
));

app.get('/auth/github',
  passport.authenticate('github', { scope: [ 'user:email' ] })
);

app.get('/auth/github/callback',
  passport.authenticate('github', { failureRedirect: '/login' }),
  function(req, res) {
    res.redirect('/dashboard');
  }
);
```

### Custom OAuth 2.0 Client

```javascript
class OAuth2Client {
  constructor(config) {
    this.clientId = config.clientId;
    this.clientSecret = config.clientSecret;
    this.redirectUri = config.redirectUri;
    this.authorizationUrl = config.authorizationUrl;
    this.tokenUrl = config.tokenUrl;
  }

  getAuthorizationUrl(state, scope) {
    const params = new URLSearchParams({
      response_type: 'code',
      client_id: this.clientId,
      redirect_uri: this.redirectUri,
      scope: scope.join(' '),
      state,
    });

    return `${this.authorizationUrl}?${params.toString()}`;
  }

  async exchangeCodeForToken(code) {
    const response = await fetch(this.tokenUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        code,
        redirect_uri: this.redirectUri,
        client_id: this.clientId,
        client_secret: this.clientSecret,
      }),
    });

    if (!response.ok) {
      throw new Error('Token exchange failed');
    }

    return await response.json();
  }

  async refreshToken(refreshToken) {
    const response = await fetch(this.tokenUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: this.clientId,
        client_secret: this.clientSecret,
      }),
    });

    if (!response.ok) {
      throw new Error('Token refresh failed');
    }

    return await response.json();
  }

  async getUserInfo(accessToken) {
    const response = await fetch('https://api.example.com/user', {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user info');
    }

    return await response.json();
  }
}

// Usage
const client = new OAuth2Client({
  clientId: process.env.CLIENT_ID,
  clientSecret: process.env.CLIENT_SECRET,
  redirectUri: 'http://localhost:3000/callback',
  authorizationUrl: 'https://provider.com/authorize',
  tokenUrl: 'https://provider.com/token',
});

// Generate authorization URL
const authUrl = client.getAuthorizationUrl('random_state', ['read:user', 'read:email']);

// Exchange code for token
const tokens = await client.exchangeCodeForToken('authorization_code');

// Get user info
const user = await client.getUserInfo(tokens.access_token);
```

---

## Security Best Practices

### 1. Always Use HTTPS
```
All OAuth 2.0 endpoints must use HTTPS to prevent token interception
```

### 2. Validate Redirect URIs
```javascript
function validateRedirectUri(redirectUri, registeredUris) {
  return registeredUris.includes(redirectUri);
}
```

### 3. Use State Parameter
```javascript
const state = crypto.randomBytes(32).toString('hex');
req.session.oauthState = state;

// Verify on callback
if (req.query.state !== req.session.oauthState) {
  throw new Error('Invalid state parameter');
}
```

### 4. Implement PKCE
```javascript
// Generate code verifier
const codeVerifier = crypto.randomBytes(32).toString('base64url');

// Generate code challenge
const codeChallenge = crypto
  .createHash('sha256')
  .update(codeVerifier)
  .digest('base64url');

// Store code verifier
req.session.codeVerifier = codeVerifier;

// Include in authorization request
const authUrl = `${AUTHORIZATION_URL}?code_challenge=${codeChallenge}&code_challenge_method=S256`;
```

### 5. Secure Token Storage
```javascript
// Never store tokens in localStorage or sessionStorage
// Use secure, httpOnly cookies
res.cookie('access_token', token, {
  httpOnly: true,
  secure: true,
  sameSite: 'strict',
  maxAge: 3600000,
});
```

### 6. Token Expiration
```javascript
// Always set token expiration
{
  "access_token": "...",
  "expires_in": 3600,
  "refresh_token": "..."
}

// Check expiration before use
if (Date.now() >= tokenExpiresAt) {
  // Refresh token
  await refreshAccessToken();
}
```

### 7. Scope Limitation
```javascript
// Request only necessary scopes
const scopes = ['read:user', 'read:email']; // Don't request write access if not needed

// Validate scopes on the server
function validateScopes(requestedScopes, userGrantedScopes) {
  return requestedScopes.every(scope => userGrantedScopes.includes(scope));
}
```

---

## Common Vulnerabilities

### 1. Authorization Code Interception

**Vulnerability:**
Attacker intercepts authorization code

**Mitigation:**
Use PKCE (Proof Key for Code Exchange)

```javascript
// Generate PKCE parameters
const codeVerifier = generateCodeVerifier();
const codeChallenge = generateCodeChallenge(codeVerifier);

// Store code_verifier securely
// Include code_challenge in authorization request
```

### 2. Redirect URI Manipulation

**Vulnerability:**
Attacker changes redirect_uri to malicious site

**Mitigation:**
```javascript
// Strictly validate redirect URIs
const ALLOWED_REDIRECT_URIS = [
  'https://app.example.com/callback',
  'https://app.example.com/oauth/callback'
];

function validateRedirectUri(uri) {
  return ALLOWED_REDIRECT_URIS.includes(uri);
}
```

### 3. CSRF Attacks

**Vulnerability:**
Attacker tricks user into authorizing their account

**Mitigation:**
```javascript
// Always use state parameter
const state = crypto.randomBytes(16).toString('hex');
req.session.state = state;

// Verify state on callback
if (req.query.state !== req.session.state) {
  throw new Error('CSRF detected');
}
```

### 4. Token Leakage

**Vulnerability:**
Tokens exposed in URLs, logs, or browser history

**Mitigation:**
```javascript
// Never include tokens in URLs
// ❌ Bad
window.location.href = `/api/data?token=${accessToken}`;

// ✅ Good
fetch('/api/data', {
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});
```

### 5. Insufficient Token Validation

**Vulnerability:**
Server doesn't properly validate tokens

**Mitigation:**
```javascript
async function validateToken(token) {
  // 1. Verify token signature
  // 2. Check expiration
  // 3. Verify issuer
  // 4. Verify audience
  // 5. Check revocation status

  if (token.exp < Date.now() / 1000) {
    throw new Error('Token expired');
  }

  if (token.iss !== EXPECTED_ISSUER) {
    throw new Error('Invalid issuer');
  }

  // Check if token is revoked
  const isRevoked = await checkRevocationList(token.jti);
  if (isRevoked) {
    throw new Error('Token revoked');
  }

  return true;
}
```

---

## Resources

**Official Specifications:**
- [RFC 6749 - OAuth 2.0](https://tools.ietf.org/html/rfc6749)
- [RFC 7636 - PKCE](https://tools.ietf.org/html/rfc7636)
- [OAuth 2.0 Security Best Practices](https://tools.ietf.org/html/draft-ietf-oauth-security-topics)

**Learning Resources:**
- [OAuth 2.0 Simplified](https://aaronparecki.com/oauth-2-simplified/)
- [OAuth.net](https://oauth.net/2/)
- [Auth0 Documentation](https://auth0.com/docs)

**Tools:**
- [OAuth 2.0 Playground](https://www.oauth.com/playground/)
- [OAuth Debugger](https://oauthdebugger.com/)

**Libraries:**
- [Passport.js](http://www.passportjs.org/) (Node.js)
- [OAuth2 Server](https://github.com/oauthjs/node-oauth2-server) (Node.js)
- [Authlib](https://authlib.org/) (Python)
- [Spring Security OAuth](https://spring.io/projects/spring-security-oauth) (Java)
