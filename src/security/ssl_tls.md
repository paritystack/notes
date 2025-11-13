# SSL/TLS (Secure Sockets Layer / Transport Layer Security)

## Overview

TLS (Transport Layer Security) is a cryptographic protocol that provides secure communication over networks. SSL is the predecessor to TLS (now deprecated).

**Key Features:**
- **Confidentiality**: Encryption prevents eavesdropping
- **Integrity**: Detects message tampering
- **Authentication**: Verifies server (and optionally client) identity

## SSL/TLS History

| Version | Year | Status | Notes |
|---------|------|--------|-------|
| **SSL 1.0** | - | Never released | Internal Netscape protocol |
| **SSL 2.0** | 1995 | Deprecated | Serious security flaws |
| **SSL 3.0** | 1996 | Deprecated | POODLE attack (2014) |
| **TLS 1.0** | 1999 | Deprecated | Similar to SSL 3.0 |
| **TLS 1.1** | 2006 | Deprecated | Minor improvements |
| **TLS 1.2** | 2008 | Secure | Currently widely used |
| **TLS 1.3** | 2018 | Secure | Modern, fastest, most secure |

## TLS Handshake (TLS 1.2)

### Full Handshake Process

```
Client                                Server

1. ClientHello              -------->
   - TLS version
   - Cipher suites
   - Random bytes
   - Extensions

                            <--------  2. ServerHello
                                          - Chosen cipher suite
                                          - Random bytes
                                          - Session ID
                                       
                                       3. Certificate
                                          - Server certificate chain
                                       
                                       4. ServerKeyExchange
                                          - Key exchange parameters
                                       
                                       5. ServerHelloDone

6. ClientKeyExchange        -------->
   - Pre-master secret (encrypted)

7. ChangeCipherSpec         -------->
   - Switch to encrypted communication

8. Finished                 -------->
   - Verification message (encrypted)

                            <--------  9. ChangeCipherSpec

                            <--------  10. Finished

11. Encrypted Application Data <---> Encrypted Application Data
```

### Detailed Steps

#### 1. ClientHello

```
Client → Server:

TLS Version: 1.2
Cipher Suites:
  - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
  - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
  - TLS_RSA_WITH_AES_128_CBC_SHA256
Random: [28 bytes client random]
Session ID: [empty for new session]
Extensions:
  - server_name: example.com
  - supported_groups: P-256, P-384
  - signature_algorithms: RSA-PSS-SHA256, ECDSA-SHA256
```

#### 2. ServerHello

```
Server → Client:

TLS Version: 1.2
Cipher Suite: TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
Random: [28 bytes server random]
Session ID: [32 bytes for session resumption]
Extensions:
  - renegotiation_info
  - extended_master_secret
```

#### 3. Certificate

```
Server → Client:

Certificate Chain:
  1. Server certificate (example.com)
  2. Intermediate CA certificate
  [Root CA not sent - client has it]
```

#### 4. ServerKeyExchange (for ECDHE)

```
Server → Client:

Curve: P-256
Public Key: [server's ephemeral ECDH public key]
Signature: [signed with server's private key]
```

#### 5. ClientKeyExchange

```
Client → Server:

Pre-master Secret:
  [Encrypted with server's public key (RSA) OR
   Client's ephemeral ECDH public key (ECDHE)]
```

#### 6. Master Secret Derivation

```
Both compute:

Master Secret = PRF(
  pre-master secret,
  "master secret",
  ClientHello.random + ServerHello.random
)

Then derive:
  - Client write MAC key
  - Server write MAC key
  - Client write encryption key
  - Server write encryption key
  - Client write IV
  - Server write IV
```

### Visual TLS 1.2 Handshake

```
┌────────┐                           ┌────────┐
│ Client │                           │ Server │
└───┬────┘                           └───┬────┘
    │                                    │
    │ ClientHello                        │
    │ (ciphers, random, SNI)             │
    ├───────────────────────────────────>│
    │                                    │
    │                   ServerHello      │
    │              (chosen cipher, random)│
    │<───────────────────────────────────┤
    │                                    │
    │                 Certificate        │
    │         (server cert chain)        │
    │<───────────────────────────────────┤
    │                                    │
    │            ServerKeyExchange       │
    │        (DH params, signature)      │
    │<───────────────────────────────────┤
    │                                    │
    │            ServerHelloDone         │
    │<───────────────────────────────────┤
    │                                    │
    │ ClientKeyExchange                  │
    │ (pre-master secret)                │
    ├───────────────────────────────────>│
    │                                    │
    │ ChangeCipherSpec                   │
    ├───────────────────────────────────>│
    │                                    │
    │ Finished (encrypted)               │
    ├───────────────────────────────────>│
    │                                    │
    │            ChangeCipherSpec        │
    │<───────────────────────────────────┤
    │                                    │
    │         Finished (encrypted)       │
    │<───────────────────────────────────┤
    │                                    │
    │ Application Data (encrypted)       │
    │<──────────────────────────────────>│
    │                                    │
```

## TLS 1.3 Handshake

TLS 1.3 is **faster** - only 1 round-trip (vs 2 in TLS 1.2):

```
Client                                Server

1. ClientHello              -------->
   - Key share (DH)
   - Supported versions
   - Cipher suites

                            <--------  2. ServerHello
                                          - Key share (DH)
                                          - Chosen cipher
                                       
                                       {Certificate}*
                                       {CertificateVerify}*
                                       {Finished}
                                       [Application Data]

{Finished}                  -------->

[Application Data]          <------->  [Application Data]

* Encrypted with handshake traffic keys
[] Encrypted with application traffic keys
```

### Key Differences TLS 1.3 vs 1.2

| Feature | TLS 1.2 | TLS 1.3 |
|---------|---------|---------|
| **Round trips** | 2-RTT | 1-RTT |
| **0-RTT mode** | No | Yes (with risks) |
| **Cipher suites** | Many (weak ones) | Only 5 strong ones |
| **Key exchange** | RSA, DHE, ECDHE | Only (EC)DHE |
| **Encryption** | After handshake | Most of handshake encrypted |
| **Performance** | Slower | Faster |
| **Security** | Vulnerable configs | Secure by default |

### TLS 1.3 Improvements

1. **Faster handshake** (1-RTT instead of 2-RTT)
2. **0-RTT mode** (resume with no round trips)
3. **Removed weak crypto** (RC4, MD5, SHA-1, RSA key exchange)
4. **Forward secrecy** (mandatory ECDHE)
5. **Encrypted handshake** (server certificate encrypted)
6. **Simplified cipher suites**

## Cipher Suites

### Cipher Suite Format (TLS 1.2)

```
TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
|   |     |   |    |       |   |
|   |     |   |    |       |   +-- MAC algorithm (SHA-256)
|   |     |   |    |       +------ AEAD mode (GCM)
|   |     |   |    +-------------- Encryption (AES-128)
|   |     |   +------------------- "WITH"
|   |     +----------------------- Authentication (RSA)
|   +----------------------------- Key exchange (ECDHE)
+--------------------------------- Protocol (TLS)
```

### Common Cipher Suites (TLS 1.2)

#### Strong & Recommended

```
TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
```

#### Weak (Avoid)

```
TLS_RSA_WITH_RC4_128_SHA                    # RC4 broken
TLS_RSA_WITH_3DES_EDE_CBC_SHA               # 3DES weak
TLS_RSA_WITH_AES_128_CBC_SHA                # CBC mode, no forward secrecy
TLS_DHE_RSA_WITH_AES_128_CBC_SHA256         # CBC mode
```

### Cipher Suite Components

#### 1. Key Exchange

```
RSA:     No forward secrecy (deprecated)
DHE:     Diffie-Hellman Ephemeral (slow)
ECDHE:   Elliptic Curve DHE (fast, forward secrecy) ✓
```

#### 2. Authentication

```
RSA:     RSA certificate
ECDSA:   Elliptic Curve certificate (smaller, faster)
DSA:     Digital Signature Algorithm (obsolete)
```

#### 3. Encryption

```
AES-128-GCM:      Fast, secure, hardware accelerated ✓
AES-256-GCM:      Higher security ✓
ChaCha20-Poly1305: Fast on mobile (no AES hardware) ✓
AES-CBC:          Vulnerable to padding oracles (avoid)
3DES:             Obsolete (avoid)
RC4:              Broken (never use)
```

#### 4. MAC (Message Authentication Code)

```
SHA-256:  Secure ✓
SHA-384:  Secure ✓
SHA-1:    Weak (avoid)
MD5:      Broken (never use)

Note: AEAD modes (GCM, ChaCha20-Poly1305) don't need separate MAC
```

### TLS 1.3 Cipher Suites (Simplified)

```
TLS_AES_128_GCM_SHA256
TLS_AES_256_GCM_SHA384
TLS_CHACHA20_POLY1305_SHA256
TLS_AES_128_CCM_SHA256
TLS_AES_128_CCM_8_SHA256
```

Only 5 cipher suites! Key exchange and auth determined separately.

## Configuring TLS

### Nginx Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    # Certificates
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # TLS versions
    ssl_protocols TLSv1.2 TLSv1.3;

    # Cipher suites (TLS 1.2)
    ssl_ciphers 'ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305';
    ssl_prefer_server_ciphers on;

    # DH parameters (for DHE cipher suites)
    ssl_dhparam /etc/nginx/dhparam.pem;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/example.com/chain.pem;

    # Session tickets
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # HSTS (HTTP Strict Transport Security)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    location / {
        root /var/www/html;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name example.com;
    return 301 https://$server_name$request_uri;
}
```

### Apache Configuration

```apache
<VirtualHost *:443>
    ServerName example.com
    
    # Certificates
    SSLCertificateFile /etc/letsencrypt/live/example.com/cert.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/example.com/privkey.pem
    SSLCertificateChainFile /etc/letsencrypt/live/example.com/chain.pem
    
    # TLS versions
    SSLProtocol -all +TLSv1.2 +TLSv1.3
    
    # Cipher suites
    SSLCipherSuite ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305
    SSLHonorCipherOrder on
    
    # OCSP Stapling
    SSLUseStapling on
    SSLStaplingCache "shmcb:logs/ssl_stapling(32768)"
    
    # HSTS
    Header always set Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
    
    DocumentRoot /var/www/html
</VirtualHost>

# Redirect HTTP to HTTPS
<VirtualHost *:80>
    ServerName example.com
    Redirect permanent / https://example.com/
</VirtualHost>
```

### Python HTTPS Server

```python
import http.server
import ssl

# Simple HTTPS server
server_address = ('0.0.0.0', 4443)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)

# Create SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('cert.pem', 'key.pem')

# Optional: Configure cipher suites
context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

# Wrap socket with TLS
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print("Server running on https://localhost:4443")
httpd.serve_forever()
```

### Python Client with TLS

```python
import ssl
import socket

def https_request(hostname, path='/'):
    # Create SSL context
    context = ssl.create_default_context()
    
    # Optional: Verify certificate
    # context.check_hostname = True
    # context.verify_mode = ssl.CERT_REQUIRED
    
    # Optional: Pin certificate
    # context.load_verify_locations('ca-bundle.crt')
    
    # Connect
    with socket.create_connection((hostname, 443)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            # Send HTTP request
            request = f"GET {path} HTTP/1.1\r\nHost: {hostname}\r\nConnection: close\r\n\r\n"
            ssock.send(request.encode())
            
            # Receive response
            response = b''
            while True:
                data = ssock.recv(4096)
                if not data:
                    break
                response += data
            
            return response.decode()

# Usage
response = https_request('example.com', '/')
print(response)
```

### Using Python Requests Library

```python
import requests

# Basic HTTPS request (verifies certificates by default)
response = requests.get('https://example.com')

# Disable certificate verification (not recommended!)
response = requests.get('https://example.com', verify=False)

# Use custom CA bundle
response = requests.get('https://example.com', verify='/path/to/ca-bundle.crt')

# Client certificate authentication
response = requests.get('https://example.com', 
                       cert=('client.crt', 'client.key'))

# Specify TLS version
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_3
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', TLSAdapter())
response = session.get('https://example.com')
```

## Testing TLS Configuration

### OpenSSL Command-Line Tests

```bash
# Connect to server and show TLS info
openssl s_client -connect example.com:443 -servername example.com

# Test specific TLS version
openssl s_client -connect example.com:443 -tls1_2
openssl s_client -connect example.com:443 -tls1_3

# Test if old protocols are disabled
openssl s_client -connect example.com:443 -ssl3  # Should fail
openssl s_client -connect example.com:443 -tls1  # Should fail
openssl s_client -connect example.com:443 -tls1_1  # Should fail

# Test specific cipher suite
openssl s_client -connect example.com:443 -cipher 'ECDHE-RSA-AES128-GCM-SHA256'

# Show certificate chain
openssl s_client -connect example.com:443 -showcerts

# Check OCSP stapling
openssl s_client -connect example.com:443 -status

# Check certificate expiration
echo | openssl s_client -connect example.com:443 2>/dev/null | openssl x509 -noout -dates

# Full connection info
openssl s_client -connect example.com:443 -servername example.com </dev/null | grep -E 'Protocol|Cipher'
```

### Testing Tools

#### nmap

```bash
# Scan TLS versions
nmap --script ssl-enum-ciphers -p 443 example.com

# Check for vulnerabilities
nmap --script ssl-* -p 443 example.com
```

#### testssl.sh

```bash
# Install
git clone https://github.com/drwetter/testssl.sh.git
cd testssl.sh

# Run comprehensive test
./testssl.sh https://example.com

# Test specific features
./testssl.sh --protocols https://example.com
./testssl.sh --ciphers https://example.com
./testssl.sh --vulnerabilities https://example.com
```

#### SSL Labs

```bash
# Online tool (web interface)
# https://www.ssllabs.com/ssltest/

# API
curl "https://api.ssllabs.com/api/v3/analyze?host=example.com"
```

### Python TLS Testing

```python
import ssl
import socket

def test_tls_version(hostname, port=443):
    """Test TLS versions supported by server"""
    versions = {
        'TLS 1.0': ssl.PROTOCOL_TLSv1,
        'TLS 1.1': ssl.PROTOCOL_TLSv1_1,
        'TLS 1.2': ssl.PROTOCOL_TLSv1_2,
        'TLS 1.3': ssl.PROTOCOL_TLS,  # Tries highest available
    }
    
    for version_name, protocol in versions.items():
        try:
            context = ssl.SSLContext(protocol)
            with socket.create_connection((hostname, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    print(f"✓ {version_name}: Supported (cipher: {ssock.cipher()[0]})")
        except Exception as e:
            print(f"✗ {version_name}: Not supported")

def get_certificate_info(hostname, port=443):
    """Get server certificate information"""
    context = ssl.create_default_context()
    
    with socket.create_connection((hostname, port)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert = ssock.getpeercert()
            
            print(f"Subject: {dict(x[0] for x in cert['subject'])}")
            print(f"Issuer: {dict(x[0] for x in cert['issuer'])}")
            print(f"Version: {cert['version']}")
            print(f"Serial: {cert['serialNumber']}")
            print(f"Not Before: {cert['notBefore']}")
            print(f"Not After: {cert['notAfter']}")
            print(f"SANs: {', '.join([x[1] for x in cert.get('subjectAltName', [])])}")
            print(f"TLS Version: {ssock.version()}")
            print(f"Cipher: {ssock.cipher()}")

# Usage
test_tls_version('example.com')
print()
get_certificate_info('example.com')
```

## Common TLS Vulnerabilities

### 1. POODLE (Padding Oracle On Downgraded Legacy Encryption)

**Attack**: Forces downgrade to SSL 3.0, exploits CBC padding

**Mitigation**:
```nginx
# Disable SSL 3.0
ssl_protocols TLSv1.2 TLSv1.3;
```

### 2. BEAST (Browser Exploit Against SSL/TLS)

**Attack**: Exploits CBC mode in TLS 1.0

**Mitigation**:
```nginx
# Disable TLS 1.0, use modern cipher suites
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-RSA-AES128-GCM-SHA256:...';
```

### 3. CRIME (Compression Ratio Info-leak Made Easy)

**Attack**: Exploits TLS compression

**Mitigation**:
```nginx
# Disable TLS compression (usually disabled by default)
ssl_compression off;
```

### 4. Heartbleed

**Attack**: Buffer over-read in OpenSSL heartbeat extension

**Mitigation**:
```bash
# Update OpenSSL
sudo apt-get update
sudo apt-get upgrade openssl

# Check version (must be > 1.0.1g)
openssl version
```

### 5. Logjam

**Attack**: Weakness in DHE key exchange with small primes

**Mitigation**:
```bash
# Generate strong DH parameters
openssl dhparam -out /etc/nginx/dhparam.pem 2048

# Configure nginx
ssl_dhparam /etc/nginx/dhparam.pem;
```

### 6. FREAK (Factoring RSA Export Keys)

**Attack**: Forces use of weak export-grade encryption

**Mitigation**:
```nginx
# Disable export ciphers
ssl_ciphers 'ECDHE-RSA-AES128-GCM-SHA256:...';  # No EXPORT
```

### 7. DROWN (Decrypting RSA with Obsolete and Weakened eNcryption)

**Attack**: Exploits SSLv2 to break TLS

**Mitigation**:
```bash
# Ensure SSLv2 is disabled everywhere
# Check with:
nmap --script ssl-enum-ciphers -p 443 example.com
```

### Checking for Vulnerabilities

```bash
# Using testssl.sh
./testssl.sh --vulnerabilities https://example.com

# Using nmap
nmap --script ssl-heartbleed,ssl-poodle,ssl-dh-params -p 443 example.com
```

## Best Practices

### 1. Protocol Configuration

```nginx
# ✓ GOOD - Only modern protocols
ssl_protocols TLSv1.2 TLSv1.3;

# ✗ BAD - Includes old protocols
ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;
```

### 2. Cipher Suite Selection

```nginx
# ✓ GOOD - Strong, forward-secret ciphers
ssl_ciphers 'ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305';

# ✗ BAD - Includes weak ciphers
ssl_ciphers 'ALL:!aNULL:!MD5';
```

### 3. Certificate Management

```bash
# ✓ Use certificates from trusted CA (Let's Encrypt)
# ✓ Automate renewal
# ✓ Monitor expiration
# ✓ Include full certificate chain
# ✗ Don't use self-signed in production
# ✗ Don't let certificates expire
```

### 4. HSTS (HTTP Strict Transport Security)

```nginx
# Enforce HTTPS for all subdomains
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
```

### 5. OCSP Stapling

```nginx
# Enable OCSP stapling for faster certificate validation
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /path/to/chain.pem;
```

### 6. Session Management

```nginx
# Session resumption (performance)
ssl_session_timeout 1d;
ssl_session_cache shared:SSL:50m;

# Disable session tickets (forward secrecy)
ssl_session_tickets off;
```

### 7. Perfect Forward Secrecy

```
Use ECDHE or DHE key exchange:
- ECDHE: Fast, modern
- DHE: Slower, but compatible

Avoid RSA key exchange (no forward secrecy)
```

### 8. Regular Updates

```bash
# Keep OpenSSL updated
sudo apt-get update
sudo apt-get upgrade openssl libssl-dev

# Keep web server updated
sudo apt-get upgrade nginx  # or apache2
```

### 9. Monitoring and Testing

```bash
# Regular security scans
./testssl.sh https://example.com

# Monitor certificate expiration
curl https://crt.sh/?q=example.com

# Check SSL Labs rating
curl "https://api.ssllabs.com/api/v3/analyze?host=example.com"
```

## Security Checklist

```
Certificate:
  [✓] Valid and not expired
  [✓] From trusted CA
  [✓] Matches domain name
  [✓] Includes full chain
  [✓] Strong key (RSA 2048+ or ECDSA P-256+)

Protocol:
  [✓] TLS 1.2 minimum
  [✓] TLS 1.3 enabled
  [✓] SSL 3.0 disabled
  [✓] TLS 1.0/1.1 disabled

Cipher Suites:
  [✓] Only strong ciphers
  [✓] Forward secrecy (ECDHE)
  [✓] AEAD modes (GCM, ChaCha20-Poly1305)
  [✓] No weak ciphers (RC4, 3DES, etc.)

Headers:
  [✓] HSTS enabled
  [✓] Secure cookie flags

Features:
  [✓] OCSP stapling enabled
  [✓] Session tickets disabled
  [✓] HTTP → HTTPS redirect

Vulnerabilities:
  [✓] Not vulnerable to POODLE
  [✓] Not vulnerable to BEAST
  [✓] Not vulnerable to Heartbleed
  [✓] Not vulnerable to Logjam
  [✓] Not vulnerable to FREAK
  [✓] Not vulnerable to DROWN
```

## Common Mistakes

### 1. Mixed Content

```html
<!-- BAD - Loading HTTP resource on HTTPS page -->
<script src="http://example.com/script.js"></script>

<!-- GOOD - Use HTTPS -->
<script src="https://example.com/script.js"></script>

<!-- BETTER - Protocol-relative URL -->
<script src="//example.com/script.js"></script>
```

### 2. Weak Cipher Configuration

```nginx
# BAD - Allows weak ciphers
ssl_ciphers 'ALL:!aNULL:!MD5';

# GOOD - Only strong ciphers
ssl_ciphers 'ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384';
```

### 3. Missing Certificate Chain

```nginx
# BAD - Only server certificate
ssl_certificate /path/to/cert.pem;

# GOOD - Full chain (server + intermediate)
ssl_certificate /path/to/fullchain.pem;
```

### 4. Expired Certificates

```bash
# Check expiration regularly
echo | openssl s_client -connect example.com:443 2>/dev/null | openssl x509 -noout -dates

# Automate renewal
certbot renew
```

### 5. Not Redirecting HTTP to HTTPS

```nginx
# Missing HTTP → HTTPS redirect leaves users vulnerable

# GOOD - Redirect all HTTP to HTTPS
server {
    listen 80;
    server_name example.com;
    return 301 https://$server_name$request_uri;
}
```

## ELI10

TLS is like a secure tunnel for internet communication:

**Without TLS (HTTP):**
```
You: "My password is abc123"
  ↓ (anyone can read this!)
Server: "OK, logged in"
```
Bad guys can see everything!

**With TLS (HTTPS):**
```
Step 1: Build a secure tunnel
  You: "Let's talk securely!"
  Server: "Here's my ID card" (certificate)
  You: "OK, I trust you"
  Both: [Create secret code together]

Step 2: Talk through tunnel
  You: "xf9#k2@..." (encrypted password)
  ↓ (looks like gibberish to bad guys!)
  Server: "p8#nz..." (encrypted response)
```

**The Handshake (making friends):**
1. **You**: "Hi! I speak TLS 1.2 and TLS 1.3"
2. **Server**: "Great! Let's use TLS 1.3. Here's my ID card"
3. **You**: "ID looks good! Here's a secret number"
4. **Server**: "Got it! Here's my secret number"
5. **Both**: "Let's mix our secrets to make a key!"
6. **Both**: "Tunnel ready! Let's talk!"

**Why it's secure:**
- **Encryption**: Messages look like random gibberish
- **Authentication**: Server proves it's really who it claims to be
- **Integrity**: Detect if someone changes messages

**TLS 1.3 is better:**
- Faster (1 handshake step instead of 2)
- More secure (removed old, weak options)
- Simpler (fewer choices = fewer mistakes)

**Real-world analogy:**
- HTTP = Postcard (anyone can read it)
- HTTPS = Sealed letter with signature (secure and verified)

## Further Resources

- [TLS 1.3 RFC 8446](https://tools.ietf.org/html/rfc8446)
- [TLS 1.2 RFC 5246](https://tools.ietf.org/html/rfc5246)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [SSL Labs Server Test](https://www.ssllabs.com/ssltest/)
- [testssl.sh GitHub](https://github.com/drwetter/testssl.sh)
- [OWASP TLS Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Transport_Layer_Protection_Cheat_Sheet.html)
- [High Performance Browser Networking (Book)](https://hpbn.co/)
- [TLS Illustrated](https://tls.ulfheim.net/)
- [Cloudflare TLS 1.3 Guide](https://blog.cloudflare.com/rfc-8446-aka-tls-1-3/)
