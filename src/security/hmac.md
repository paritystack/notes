# HMAC (Hash-based Message Authentication Code)

## Overview

HMAC is a mechanism for message authentication using cryptographic hash functions. It provides both **data integrity** (message hasn't been altered) and **authentication** (message came from someone with the secret key).

## HMAC Construction

### Formula

```
HMAC(K, m) = H((K' ⊕ opad) || H((K' ⊕ ipad) || m))
```

Where:
- **K** = secret key
- **m** = message
- **H** = cryptographic hash function (SHA-256, SHA-512, etc.)
- **K'** = key derived from K (padded/hashed to block size)
- **⊕** = XOR operation
- **||** = concatenation
- **opad** = outer padding (0x5c repeated)
- **ipad** = inner padding (0x36 repeated)

### Simplified Steps

```
1. If key is longer than block size, hash it
2. If key is shorter than block size, pad with zeros
3. XOR key with inner padding (ipad)
4. Append message to result
5. Hash the result (inner hash)
6. XOR key with outer padding (opad)
7. Append inner hash to result
8. Hash the result (outer hash) = HMAC
```

### Visual Representation

```
        Secret Key
             |
      +------+------+
      |             |
    XOR ipad     XOR opad
      |             |
      + Message     |
      |             |
   Hash (inner)     |
      |             |
      +-------------+
            |
         Hash (outer)
            |
          HMAC
```

## Why HMAC Instead of Hash(Key + Message)?

### Vulnerable Approaches

```python
# VULNERABLE 1: Simple concatenation
tag = sha256(key + message)
# Vulnerable to length extension attacks!

# VULNERABLE 2: Wrong order
tag = sha256(message + key)
# Attacker can append data!

# SECURE: Use HMAC
tag = hmac.new(key, message, sha256).digest()
```

### Length Extension Attack Example

```python
# With SHA-256 concatenation (VULNERABLE)
original = sha256(key + message)
# Attacker can compute: sha256(key + message + attacker_data)
# WITHOUT knowing the key!

# With HMAC (SECURE)
original = hmac(key, message)
# Attacker CANNOT extend the message without knowing the key
```

## Using HMAC

### Python Examples

#### Basic HMAC

```python
import hmac
import hashlib

# Create HMAC
key = b"secret-key-12345"
message = b"Important message"

# HMAC-SHA256
mac = hmac.new(key, message, hashlib.sha256)
tag = mac.hexdigest()
print(f"HMAC-SHA256: {tag}")

# HMAC-SHA512
mac = hmac.new(key, message, hashlib.sha512)
tag = mac.hexdigest()
print(f"HMAC-SHA512: {tag}")

# Digest as bytes
tag_bytes = hmac.new(key, message, hashlib.sha256).digest()
print(f"HMAC (bytes): {tag_bytes}")
```

#### Verify HMAC

```python
import hmac
import hashlib

def create_hmac(key, message):
    return hmac.new(key, message, hashlib.sha256).digest()

def verify_hmac(key, message, received_tag):
    expected_tag = hmac.new(key, message, hashlib.sha256).digest()
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(expected_tag, received_tag)

# Usage
key = b"secret-key"
message = b"Transfer $100 to Alice"

# Create tag
tag = create_hmac(key, message)
print(f"Tag: {tag.hex()}")

# Verify tag (correct)
if verify_hmac(key, message, tag):
    print("Message is authentic!")

# Verify tag (tampered message)
tampered = b"Transfer $999 to Alice"
if not verify_hmac(key, tampered, tag):
    print("Message has been tampered with!")
```

#### Incremental HMAC

```python
import hmac
import hashlib

# For large messages
mac = hmac.new(b"secret-key", digestmod=hashlib.sha256)

# Update incrementally
mac.update(b"Part 1 of message ")
mac.update(b"Part 2 of message ")
mac.update(b"Part 3 of message")

tag = mac.hexdigest()
print(f"Incremental HMAC: {tag}")

# Equivalent to
mac_full = hmac.new(b"secret-key", 
                    b"Part 1 of message Part 2 of message Part 3 of message",
                    hashlib.sha256)
print(f"Full HMAC: {mac_full.hexdigest()}")
```

### OpenSSL/Bash Examples

```bash
# Generate HMAC-SHA256
echo -n "Important message" | openssl dgst -sha256 -hmac "secret-key"

# HMAC-SHA512
echo -n "Important message" | openssl dgst -sha512 -hmac "secret-key"

# HMAC of a file
openssl dgst -sha256 -hmac "secret-key" document.pdf

# Output in different formats
echo -n "message" | openssl dgst -sha256 -hmac "key" -hex
echo -n "message" | openssl dgst -sha256 -hmac "key" -binary | base64
```

### JavaScript Example

```javascript
const crypto = require('crypto');

// Create HMAC
const key = 'secret-key-12345';
const message = 'Important message';

const hmac = crypto.createHmac('sha256', key);
hmac.update(message);
const tag = hmac.digest('hex');

console.log(`HMAC-SHA256: ${tag}`);

// Verify HMAC
function verifyHMAC(key, message, receivedTag) {
    const expectedTag = crypto.createHmac('sha256', key)
                             .update(message)
                             .digest('hex');
    
    // Constant-time comparison
    return crypto.timingSafeEqual(
        Buffer.from(expectedTag, 'hex'),
        Buffer.from(receivedTag, 'hex')
    );
}
```

## Message Authentication

### Sending Authenticated Messages

```python
import hmac
import hashlib
import json

class AuthenticatedMessage:
    def __init__(self, shared_key):
        self.key = shared_key
    
    def send(self, message):
        # Create HMAC tag
        tag = hmac.new(self.key, message.encode(), hashlib.sha256).hexdigest()
        
        # Package message with tag
        package = {
            'message': message,
            'hmac': tag
        }
        return json.dumps(package)
    
    def receive(self, package_json):
        # Unpack message
        package = json.loads(package_json)
        message = package['message']
        received_tag = package['hmac']
        
        # Verify HMAC
        expected_tag = hmac.new(self.key, message.encode(), hashlib.sha256).hexdigest()
        
        if hmac.compare_digest(expected_tag, received_tag):
            return message, True
        else:
            return None, False

# Usage
shared_key = b"shared-secret-key-between-alice-and-bob"

# Alice sends message
alice = AuthenticatedMessage(shared_key)
package = alice.send("Transfer $100 to Bob")
print(f"Sent: {package}")

# Bob receives message
bob = AuthenticatedMessage(shared_key)
message, is_authentic = bob.receive(package)

if is_authentic:
    print(f"Authentic message: {message}")
else:
    print("Warning: Message tampered!")

# Attacker tries to tamper
tampered_package = package.replace("$100", "$999")
message, is_authentic = bob.receive(tampered_package)
print(f"Tampered authentic: {is_authentic}")  # False
```

## Integrity Verification

### File Integrity with HMAC

```python
import hmac
import hashlib
import os

class FileIntegrityChecker:
    def __init__(self, key):
        self.key = key
    
    def compute_file_hmac(self, filepath):
        mac = hmac.new(self.key, digestmod=hashlib.sha256)
        
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                mac.update(chunk)
        
        return mac.hexdigest()
    
    def create_manifest(self, files):
        manifest = {}
        for filepath in files:
            manifest[filepath] = self.compute_file_hmac(filepath)
        return manifest
    
    def verify_files(self, manifest):
        results = {}
        for filepath, expected_hmac in manifest.items():
            if not os.path.exists(filepath):
                results[filepath] = "MISSING"
            else:
                actual_hmac = self.compute_file_hmac(filepath)
                if hmac.compare_digest(expected_hmac, actual_hmac):
                    results[filepath] = "OK"
                else:
                    results[filepath] = "MODIFIED"
        return results

# Usage
checker = FileIntegrityChecker(b"integrity-check-key")

# Create manifest
files = ['config.json', 'app.py', 'data.db']
manifest = checker.create_manifest(files)
print("Manifest created:", manifest)

# Later, verify files
results = checker.verify_files(manifest)
for file, status in results.items():
    print(f"{file}: {status}")
```

## API Authentication

### API Request Signing

```python
import hmac
import hashlib
import time
import requests
from urllib.parse import urlencode

class APIClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
    
    def generate_signature(self, method, path, params):
        # Create string to sign
        timestamp = str(int(time.time()))
        params['timestamp'] = timestamp
        params['api_key'] = self.api_key
        
        # Sort parameters
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)
        
        # String to sign: METHOD + PATH + QUERY_STRING
        message = f"{method}{path}{query_string}"
        
        # Generate HMAC signature
        signature = hmac.new(
            self.api_secret,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature, timestamp
    
    def make_request(self, method, path, params=None):
        if params is None:
            params = {}
        
        # Generate signature
        signature, timestamp = self.generate_signature(method, path, params)
        
        # Add authentication headers
        headers = {
            'X-API-Key': self.api_key,
            'X-API-Signature': signature,
            'X-API-Timestamp': timestamp
        }
        
        # Make request
        url = f"https://api.example.com{path}"
        response = requests.request(method, url, params=params, headers=headers)
        
        return response

# Server-side verification
class APIServer:
    def __init__(self):
        # In practice, look up secret from database based on API key
        self.api_secrets = {
            'key123': b'secret123'
        }
    
    def verify_signature(self, api_key, signature, timestamp, method, path, params):
        # Check timestamp (prevent replay attacks)
        current_time = int(time.time())
        request_time = int(timestamp)
        
        if abs(current_time - request_time) > 300:  # 5 minutes
            return False, "Request expired"
        
        # Get API secret
        if api_key not in self.api_secrets:
            return False, "Invalid API key"
        
        api_secret = self.api_secrets[api_key]
        
        # Reconstruct signed message
        params['timestamp'] = timestamp
        params['api_key'] = api_key
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)
        message = f"{method}{path}{query_string}"
        
        # Compute expected signature
        expected_signature = hmac.new(
            api_secret,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures (constant time)
        if hmac.compare_digest(expected_signature, signature):
            return True, "Valid"
        else:
            return False, "Invalid signature"

# Usage
client = APIClient('key123', 'secret123')
response = client.make_request('GET', '/api/users', {'limit': 10})
```

### REST API with HMAC Authentication

```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)

API_SECRETS = {
    'client1': b'secret1',
    'client2': b'secret2'
}

def verify_hmac_signature():
    api_key = request.headers.get('X-API-Key')
    signature = request.headers.get('X-Signature')
    
    if not api_key or not signature:
        return False
    
    if api_key not in API_SECRETS:
        return False
    
    # Reconstruct signed data
    # Method + Path + Body (for POST/PUT)
    data = request.method + request.path
    
    if request.data:
        data += request.data.decode()
    
    # Compute expected signature
    expected = hmac.new(
        API_SECRETS[api_key],
        data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected, signature)

@app.route('/api/data', methods=['POST'])
def post_data():
    if not verify_hmac_signature():
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Process request
    data = request.json
    return jsonify({'status': 'success', 'data': data})

# Client request example
import requests
import hmac
import hashlib

api_key = 'client1'
api_secret = b'secret1'
url = 'http://localhost:5000/api/data'
payload = {'key': 'value'}

# Create signature
data = 'POST' + '/api/data' + json.dumps(payload)
signature = hmac.new(api_secret, data.encode(), hashlib.sha256).hexdigest()

headers = {
    'X-API-Key': api_key,
    'X-Signature': signature,
    'Content-Type': 'application/json'
}

response = requests.post(url, json=payload, headers=headers)
```

## JWT (JSON Web Tokens)

JWTs use HMAC (or RSA) for signature verification.

### JWT Structure

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
|                                      |                                                                                |                                              |
            Header                                              Payload                                                              Signature
```

### JWT with HMAC

```python
import hmac
import hashlib
import json
import base64

class JWT:
    def __init__(self, secret):
        self.secret = secret.encode()
    
    def base64url_encode(self, data):
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode()
    
    def base64url_decode(self, data):
        padding = 4 - len(data) % 4
        data += '=' * padding
        return base64.urlsafe_b64decode(data)
    
    def create_token(self, payload):
        # Header
        header = {
            'alg': 'HS256',
            'typ': 'JWT'
        }
        
        # Encode header and payload
        header_encoded = self.base64url_encode(json.dumps(header).encode())
        payload_encoded = self.base64url_encode(json.dumps(payload).encode())
        
        # Create signature
        message = f"{header_encoded}.{payload_encoded}".encode()
        signature = hmac.new(self.secret, message, hashlib.sha256).digest()
        signature_encoded = self.base64url_encode(signature)
        
        # Combine
        token = f"{header_encoded}.{payload_encoded}.{signature_encoded}"
        return token
    
    def verify_token(self, token):
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None, False
            
            header_encoded, payload_encoded, signature_encoded = parts
            
            # Verify signature
            message = f"{header_encoded}.{payload_encoded}".encode()
            expected_signature = hmac.new(self.secret, message, hashlib.sha256).digest()
            received_signature = self.base64url_decode(signature_encoded)
            
            if not hmac.compare_digest(expected_signature, received_signature):
                return None, False
            
            # Decode payload
            payload = json.loads(self.base64url_decode(payload_encoded))
            return payload, True
            
        except Exception as e:
            return None, False

# Usage
jwt = JWT('my-secret-key')

# Create token
payload = {
    'user_id': 12345,
    'username': 'john_doe',
    'exp': int(time.time()) + 3600  # Expires in 1 hour
}

token = jwt.create_token(payload)
print(f"JWT: {token}")

# Verify token
payload, is_valid = jwt.verify_token(token)
if is_valid:
    print(f"Valid token! User: {payload['username']}")
else:
    print("Invalid token!")

# Using PyJWT library (recommended)
import jwt as pyjwt

# Create token
token = pyjwt.encode(payload, 'my-secret-key', algorithm='HS256')

# Verify token
try:
    decoded = pyjwt.decode(token, 'my-secret-key', algorithms=['HS256'])
    print(f"Valid! Payload: {decoded}")
except pyjwt.InvalidTokenError:
    print("Invalid token!")
```

## HMAC vs Other MACs

### Comparison

| Feature | HMAC | CBC-MAC | GMAC | Poly1305 |
|---------|------|---------|------|----------|
| **Based on** | Hash function | Block cipher | Block cipher | Universal hash |
| **Performance** | Moderate | Slow | Fast | Very fast |
| **Key reuse** | Safe | Dangerous | Safe | One-time key |
| **Standardized** | Yes (RFC 2104) | Yes | Yes (GCM) | Yes (ChaCha20) |
| **Use case** | General purpose | Legacy | AEAD | Modern crypto |

### HMAC-SHA256 vs HMAC-SHA512

```python
import hmac
import hashlib
import time

message = b"x" * 1000000  # 1 MB
key = b"secret-key"

# HMAC-SHA256
start = time.time()
for _ in range(100):
    hmac.new(key, message, hashlib.sha256).digest()
print(f"HMAC-SHA256: {time.time() - start:.3f}s")

# HMAC-SHA512
start = time.time()
for _ in range(100):
    hmac.new(key, message, hashlib.sha512).digest()
print(f"HMAC-SHA512: {time.time() - start:.3f}s")

# Output sizes
print(f"SHA256 output: {len(hmac.new(key, b'test', hashlib.sha256).digest())} bytes")
print(f"SHA512 output: {len(hmac.new(key, b'test', hashlib.sha512).digest())} bytes")
```

## Security Considerations

### 1. Key Length

```python
# Minimum key length = hash output size
# SHA-256: minimum 32 bytes
# SHA-512: minimum 64 bytes

# GOOD
key = os.urandom(32)  # 256 bits for HMAC-SHA256

# BAD - too short
key = b"secret"  # Only 48 bits!

# Better - derive from password
from hashlib import pbkdf2_hmac
key = pbkdf2_hmac('sha256', b'user-password', b'salt', 100000)
```

### 2. Constant-Time Comparison

```python
# VULNERABLE - timing attack
if computed_hmac == received_hmac:
    return True

# SECURE - constant time comparison
import hmac
if hmac.compare_digest(computed_hmac, received_hmac):
    return True
```

### 3. Prevent Replay Attacks

```python
import time

def verify_request(hmac_tag, timestamp, max_age=300):
    # Verify HMAC first
    if not verify_hmac(hmac_tag):
        return False
    
    # Check timestamp (prevent replays)
    current_time = int(time.time())
    request_time = int(timestamp)
    
    if abs(current_time - request_time) > max_age:
        return False  # Request too old
    
    # Optional: Track used nonces to prevent replay
    # if nonce in used_nonces:
    #     return False
    
    return True
```

### 4. Use Separate Keys

```python
# BAD - same key for different purposes
encryption_key = b"shared-key"
hmac_key = b"shared-key"

# GOOD - derive separate keys
from hashlib import sha256

master_key = b"master-secret-key"
encryption_key = sha256(master_key + b"encryption").digest()
hmac_key = sha256(master_key + b"authentication").digest()

# BETTER - use HKDF (HMAC-based Key Derivation)
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

master_key = b"master-secret-key"

hkdf = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b'encryption',
)
encryption_key = hkdf.derive(master_key)

hkdf = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b'authentication',
)
hmac_key = hkdf.derive(master_key)
```

### 5. Truncation

```python
# Full HMAC (recommended)
mac = hmac.new(key, message, hashlib.sha256).digest()  # 32 bytes

# Truncated HMAC (if needed)
mac_truncated = hmac.new(key, message, hashlib.sha256).digest()[:16]  # 16 bytes

# Minimum recommended: 128 bits (16 bytes)
# Never go below 80 bits (10 bytes)
```

## Best Practices

### 1. Always Use HMAC for Message Authentication

```python
# ✓ Use HMAC
tag = hmac.new(key, message, hashlib.sha256).digest()

# ✗ Don't use simple hash
tag = hashlib.sha256(key + message).digest()  # Vulnerable!
```

### 2. Choose Appropriate Hash Function

```python
# Modern: SHA-256 or SHA-512
hmac.new(key, message, hashlib.sha256)

# Avoid: MD5 or SHA-1
hmac.new(key, message, hashlib.md5)  # Don't use!
```

### 3. Protect the Key

```python
# Store keys securely
# - Use environment variables
# - Use key management service (AWS KMS, etc.)
# - Never hardcode in source code
# - Never commit to version control

import os
key = os.environ.get('HMAC_KEY').encode()

# Rotate keys periodically
# Support multiple active keys during rotation
```

### 4. Include All Relevant Data

```python
# Sign complete context
data = {
    'timestamp': timestamp,
    'user_id': user_id,
    'action': action,
    'nonce': nonce
}

message = json.dumps(data, sort_keys=True).encode()
signature = hmac.new(key, message, hashlib.sha256).hexdigest()
```

## Common Mistakes

### 1. Using == for Comparison

```python
# WRONG - timing attack
if hmac1 == hmac2:
    pass

# RIGHT - constant time
if hmac.compare_digest(hmac1, hmac2):
    pass
```

### 2. Not Including Timestamp

```python
# WRONG - vulnerable to replay
signature = hmac.new(key, message, sha256).hexdigest()

# RIGHT - include timestamp
data = f"{timestamp}:{message}"
signature = hmac.new(key, data.encode(), sha256).hexdigest()
```

### 3. Wrong Key Derivation

```python
# WRONG - weak key
key = b"password"

# RIGHT - derive from password
from hashlib import pbkdf2_hmac
key = pbkdf2_hmac('sha256', b'password', b'salt', 100000)
```

## ELI10

HMAC is like a secret handshake for messages:

Imagine you and your best friend have a secret code:
1. You write a message: "Meet at the treehouse at 3pm"
2. You add your secret code and mix it all together in a special way
3. You get a "stamp": `a7f9e4b2...`
4. You send: message + stamp

When your friend receives it:
1. They take the message
2. They add the SAME secret code and mix it the SAME way
3. They get their own stamp
4. If their stamp matches yours, they know:
   - The message really came from you (only you know the code!)
   - Nobody changed the message (the stamp would be different!)

**Why not just put the secret code in the message?**
- Anyone could copy your code!

**Why not just hash the message?**
- Anyone could make their own hash!

**HMAC is special because:**
- You need the secret code to make the stamp
- Even a tiny change makes a completely different stamp
- Nobody can make the right stamp without knowing your secret code!

**Real-world example:**
When you log into a website, your browser and the server use HMAC to:
- Make sure messages aren't tampered with
- Prove who sent the message
- Keep your session secure!

## Further Resources

- [RFC 2104 - HMAC Specification](https://tools.ietf.org/html/rfc2104)
- [HMAC Security Analysis](https://csrc.nist.gov/publications/detail/fips/198/1/final)
- [JWT Specification (RFC 7519)](https://tools.ietf.org/html/rfc7519)
- [API Authentication Best Practices](https://owasp.org/www-project-api-security/)
- [Timing Attack Prevention](https://codahale.com/a-lesson-in-timing-attacks/)
- [HKDF Specification (RFC 5869)](https://tools.ietf.org/html/rfc5869)
