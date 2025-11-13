# Digital Signatures

## Overview

A digital signature is a cryptographic mechanism that provides:
- **Authentication**: Proves who created the signature
- **Integrity**: Detects any changes to the signed data
- **Non-repudiation**: Signer cannot deny signing (unlike HMAC)

Digital signatures use **asymmetric cryptography** (public/private key pairs).

## Digital Signature vs HMAC

| Feature | Digital Signature | HMAC |
|---------|------------------|------|
| **Keys** | Public/Private key pair | Shared secret key |
| **Verification** | Anyone with public key | Only parties with secret |
| **Non-repudiation** | Yes | No |
| **Performance** | Slower | Faster |
| **Key distribution** | Public key can be shared | Secret must be protected |
| **Use case** | Documents, software, certificates | API auth, sessions |

## How Digital Signatures Work

### Signing Process

```
1. Hash the message
   Message → Hash Function → Digest

2. Encrypt digest with private key
   Digest → Private Key → Signature

3. Attach signature to message
   Message + Signature → Signed Document
```

### Verification Process

```
1. Hash the received message
   Message → Hash Function → Digest₁

2. Decrypt signature with public key
   Signature → Public Key → Digest₂

3. Compare digests
   If Digest₁ == Digest₂ → Valid Signature
```

### Visual Representation

```
SIGNING:
                    Message
                       |
                   Hash (SHA-256)
                       |
                    Digest
                       |
            Encrypt with Private Key
                       |
                   Signature
                       |
            Message + Signature


VERIFICATION:
            Message + Signature
                |           |
                |           |
           Hash (SHA-256)   Decrypt with Public Key
                |           |
             Digest₁     Digest₂
                |           |
                +-----+-----+
                      |
                  Compare
                      |
                Valid/Invalid
```

## RSA Signatures

### RSA Algorithm Overview

RSA uses modular arithmetic with large prime numbers:
- **Key Generation**: Create public (e, n) and private (d, n) keys
- **Signing**: signature = (hash)^d mod n
- **Verification**: hash = (signature)^e mod n

### Generating RSA Keys

#### Python (cryptography library)

```python
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,  # 2048 or 4096 bits
)

# Generate public key
public_key = private_key.public_key()

# Save private key
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

with open('private_key.pem', 'wb') as f:
    f.write(pem_private)

# Save public key
pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

with open('public_key.pem', 'wb') as f:
    f.write(pem_public)
```

#### OpenSSL (Bash)

```bash
# Generate private key (2048-bit RSA)
openssl genrsa -out private_key.pem 2048

# Generate private key with password protection
openssl genrsa -aes256 -out private_key.pem 2048

# Extract public key from private key
openssl rsa -in private_key.pem -pubout -out public_key.pem

# Generate 4096-bit key (more secure)
openssl genrsa -out private_key.pem 4096

# View key details
openssl rsa -in private_key.pem -text -noout
```

### Signing with RSA

#### Python Example

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization

# Load private key
with open('private_key.pem', 'rb') as f:
    private_key = serialization.load_pem_private_key(
        f.read(),
        password=None
    )

# Message to sign
message = b"This is an important document"

# Sign message (RSA-PSS with SHA-256)
signature = private_key.sign(
    message,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)

print(f"Signature: {signature.hex()}")
print(f"Signature length: {len(signature)} bytes")

# Save signature
with open('signature.bin', 'wb') as f:
    f.write(signature)
```

#### OpenSSL Example

```bash
# Sign a file
openssl dgst -sha256 -sign private_key.pem -out signature.bin document.txt

# Sign with different hash algorithms
openssl dgst -sha512 -sign private_key.pem -out signature.bin document.txt

# Create detached signature
openssl dgst -sha256 -sign private_key.pem -out document.sig document.pdf
```

### Verifying RSA Signatures

#### Python Example

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

# Load public key
with open('public_key.pem', 'rb') as f:
    public_key = serialization.load_pem_public_key(f.read())

# Load signature
with open('signature.bin', 'rb') as f:
    signature = f.read()

# Message to verify
message = b"This is an important document"

# Verify signature
try:
    public_key.verify(
        signature,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    print("✓ Signature is valid!")
except InvalidSignature:
    print("✗ Invalid signature!")

# Complete example
def verify_document(public_key_path, document, signature):
    with open(public_key_path, 'rb') as f:
        public_key = serialization.load_pem_public_key(f.read())
    
    try:
        public_key.verify(
            signature,
            document,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False
```

#### OpenSSL Example

```bash
# Verify signature
openssl dgst -sha256 -verify public_key.pem -signature signature.bin document.txt

# Output:
# Verified OK  (if valid)
# Verification Failure  (if invalid)

# Verify detached signature
openssl dgst -sha256 -verify public_key.pem -signature document.sig document.pdf
```

### RSA Padding Schemes

#### PKCS#1 v1.5 (Legacy)

```python
from cryptography.hazmat.primitives.asymmetric import padding

# Sign with PKCS#1 v1.5 (not recommended)
signature = private_key.sign(
    message,
    padding.PKCS1v15(),
    hashes.SHA256()
)
```

#### PSS (Recommended)

```python
# Sign with PSS (Probabilistic Signature Scheme)
signature = private_key.sign(
    message,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)
```

## ECDSA (Elliptic Curve Digital Signature Algorithm)

### Overview

ECDSA provides equivalent security to RSA with much smaller keys:
- **RSA 2048-bit** ≈ **ECDSA 224-bit**
- **RSA 3072-bit** ≈ **ECDSA 256-bit**
- **RSA 15360-bit** ≈ **ECDSA 512-bit**

**Benefits:**
- Smaller keys
- Faster signing
- Less bandwidth
- Less storage

### Common Curves

| Curve | Bits | Security | Use Case |
|-------|------|----------|----------|
| **P-256** (secp256r1) | 256 | ~128-bit | General purpose, TLS |
| **P-384** (secp384r1) | 384 | ~192-bit | High security |
| **P-521** (secp521r1) | 521 | ~256-bit | Maximum security |
| **secp256k1** | 256 | ~128-bit | Bitcoin, cryptocurrencies |

### Generating ECDSA Keys

#### Python Example

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

# Generate private key (P-256 curve)
private_key = ec.generate_private_key(ec.SECP256R1())

# Extract public key
public_key = private_key.public_key()

# Save private key
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

with open('ec_private_key.pem', 'wb') as f:
    f.write(pem_private)

# Save public key
pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

with open('ec_public_key.pem', 'wb') as f:
    f.write(pem_public)

# Different curves
# P-256 (most common)
key_p256 = ec.generate_private_key(ec.SECP256R1())

# P-384 (higher security)
key_p384 = ec.generate_private_key(ec.SECP384R1())

# P-521 (maximum security)
key_p521 = ec.generate_private_key(ec.SECP521R1())

# secp256k1 (Bitcoin)
key_secp256k1 = ec.generate_private_key(ec.SECP256K1())
```

#### OpenSSL Example

```bash
# Generate EC private key (P-256)
openssl ecparam -name prime256v1 -genkey -noout -out ec_private_key.pem

# Generate with P-384
openssl ecparam -name secp384r1 -genkey -noout -out ec_private_key.pem

# Generate with P-521
openssl ecparam -name secp521r1 -genkey -noout -out ec_private_key.pem

# Extract public key
openssl ec -in ec_private_key.pem -pubout -out ec_public_key.pem

# View key details
openssl ec -in ec_private_key.pem -text -noout

# List available curves
openssl ecparam -list_curves
```

### Signing with ECDSA

#### Python Example

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

# Load private key
with open('ec_private_key.pem', 'rb') as f:
    private_key = serialization.load_pem_private_key(
        f.read(),
        password=None
    )

# Message to sign
message = b"ECDSA signature example"

# Sign message
signature = private_key.sign(
    message,
    ec.ECDSA(hashes.SHA256())
)

print(f"ECDSA Signature: {signature.hex()}")
print(f"Signature length: {len(signature)} bytes")

# For P-256, signature is ~64 bytes (vs ~256 bytes for RSA-2048!)
```

#### OpenSSL Example

```bash
# Sign with ECDSA
openssl dgst -sha256 -sign ec_private_key.pem -out ecdsa_signature.bin document.txt

# Verify ECDSA signature
openssl dgst -sha256 -verify ec_public_key.pem -signature ecdsa_signature.bin document.txt
```

### Verifying ECDSA Signatures

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

# Load public key
with open('ec_public_key.pem', 'rb') as f:
    public_key = serialization.load_pem_public_key(f.read())

# Message and signature
message = b"ECDSA signature example"

with open('ecdsa_signature.bin', 'rb') as f:
    signature = f.read()

# Verify signature
try:
    public_key.verify(
        signature,
        message,
        ec.ECDSA(hashes.SHA256())
    )
    print("✓ ECDSA signature is valid!")
except InvalidSignature:
    print("✗ Invalid ECDSA signature!")
```

## EdDSA (Edwards-curve Digital Signature Algorithm)

### Overview

EdDSA is a modern signature scheme designed for high performance and security.

**Ed25519** (most common):
- 256-bit keys
- Fast signing and verification
- Deterministic (no random number needed)
- Resistant to side-channel attacks

### Generating Ed25519 Keys

```python
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Generate private key
private_key = ed25519.Ed25519PrivateKey.generate()

# Extract public key
public_key = private_key.public_key()

# Save private key
pem_private = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

with open('ed25519_private_key.pem', 'wb') as f:
    f.write(pem_private)

# Save public key
pem_public = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

with open('ed25519_public_key.pem', 'wb') as f:
    f.write(pem_public)

# Raw bytes format (32 bytes each)
private_bytes = private_key.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption()
)

public_bytes = public_key.public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw
)

print(f"Private key: {private_bytes.hex()} ({len(private_bytes)} bytes)")
print(f"Public key: {public_bytes.hex()} ({len(public_bytes)} bytes)")
```

### Signing with Ed25519

```python
from cryptography.hazmat.primitives.asymmetric import ed25519

# Generate key
private_key = ed25519.Ed25519PrivateKey.generate()

# Message to sign
message = b"Ed25519 is fast and secure!"

# Sign (deterministic, no hash function needed)
signature = private_key.sign(message)

print(f"Ed25519 Signature: {signature.hex()}")
print(f"Signature length: {len(signature)} bytes")  # Always 64 bytes

# Verify
public_key = private_key.public_key()

try:
    public_key.verify(signature, message)
    print("✓ Signature valid!")
except:
    print("✗ Invalid signature!")
```

### Performance Comparison

```python
import time
from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

message = b"Performance test message"

# RSA
rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
start = time.time()
for _ in range(1000):
    sig = rsa_key.sign(message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), 
                       salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
rsa_time = time.time() - start

# ECDSA
ec_key = ec.generate_private_key(ec.SECP256R1())
start = time.time()
for _ in range(1000):
    sig = ec_key.sign(message, ec.ECDSA(hashes.SHA256()))
ecdsa_time = time.time() - start

# Ed25519
ed_key = ed25519.Ed25519PrivateKey.generate()
start = time.time()
for _ in range(1000):
    sig = ed_key.sign(message)
ed25519_time = time.time() - start

print(f"RSA-2048:  {rsa_time:.3f}s")
print(f"ECDSA-256: {ecdsa_time:.3f}s")
print(f"Ed25519:   {ed25519_time:.3f}s")

# Typical results:
# RSA-2048:  5.234s  (slowest)
# ECDSA-256: 1.876s  (fast)
# Ed25519:   0.156s  (fastest!)
```

## Signature Verification

### Complete Verification Example

```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

class DocumentSigner:
    def __init__(self, private_key_path=None, public_key_path=None):
        if private_key_path:
            with open(private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
        
        if public_key_path:
            with open(public_key_path, 'rb') as f:
                self.public_key = serialization.load_pem_public_key(f.read())
    
    def sign_document(self, document):
        signature = self.private_key.sign(
            document,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_document(self, document, signature):
        try:
            self.public_key.verify(
                signature,
                document,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True, "Signature is valid"
        except InvalidSignature:
            return False, "Invalid signature - document may be tampered"
        except Exception as e:
            return False, f"Verification error: {str(e)}"
    
    def sign_file(self, filepath, signature_path):
        with open(filepath, 'rb') as f:
            document = f.read()
        
        signature = self.sign_document(document)
        
        with open(signature_path, 'wb') as f:
            f.write(signature)
        
        return signature
    
    def verify_file(self, filepath, signature_path):
        with open(filepath, 'rb') as f:
            document = f.read()
        
        with open(signature_path, 'rb') as f:
            signature = f.read()
        
        return self.verify_document(document, signature)

# Usage
# Signing
signer = DocumentSigner(private_key_path='private_key.pem')
document = b"Important contract: Alice pays Bob $1000"
signature = signer.sign_document(document)

# Verification
verifier = DocumentSigner(public_key_path='public_key.pem')
is_valid, message = verifier.verify_document(document, signature)
print(f"{message}")

# File signing
signer.sign_file('contract.pdf', 'contract.pdf.sig')
is_valid, message = verifier.verify_file('contract.pdf', 'contract.pdf.sig')
print(f"Contract verification: {message}")
```

## Code Signing

### Signing Software/Scripts

```bash
# Sign a Python script
openssl dgst -sha256 -sign private_key.pem -out script.py.sig script.py

# Create a signed package
tar -czf package.tar.gz files/
openssl dgst -sha256 -sign private_key.pem -out package.tar.gz.sig package.tar.gz

# Verification script
#!/bin/bash
FILE=$1
SIG=$2
PUBKEY=$3

openssl dgst -sha256 -verify $PUBKEY -signature $SIG $FILE

if [ $? -eq 0 ]; then
    echo "✓ Signature verified - safe to run"
else
    echo "✗ Invalid signature - DO NOT RUN"
    exit 1
fi
```

### Python Code Signing Example

```python
import os
import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

class CodeSigner:
    def __init__(self, private_key_path):
        with open(private_key_path, 'rb') as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
    
    def sign_file(self, filepath):
        # Read file
        with open(filepath, 'rb') as f:
            code = f.read()
        
        # Generate signature
        signature = self.private_key.sign(
            code,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Save signature
        sig_path = filepath + '.sig'
        with open(sig_path, 'wb') as f:
            f.write(signature)
        
        print(f"✓ Signed: {filepath}")
        print(f"✓ Signature: {sig_path}")
        
        return signature

class CodeVerifier:
    def __init__(self, public_key_path):
        with open(public_key_path, 'rb') as f:
            self.public_key = serialization.load_pem_public_key(f.read())
    
    def verify_file(self, filepath):
        sig_path = filepath + '.sig'
        
        # Read file and signature
        with open(filepath, 'rb') as f:
            code = f.read()
        
        with open(sig_path, 'rb') as f:
            signature = f.read()
        
        # Verify
        try:
            self.public_key.verify(
                signature,
                code,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            print(f"✓ Signature valid for {filepath}")
            return True
        except:
            print(f"✗ Invalid signature for {filepath}")
            return False

# Usage
signer = CodeSigner('private_key.pem')
signer.sign_file('important_script.py')

verifier = CodeVerifier('public_key.pem')
if verifier.verify_file('important_script.py'):
    # Safe to execute
    exec(open('important_script.py').read())
```

### macOS Code Signing

```bash
# Sign application
codesign -s "Developer ID" MyApp.app

# Verify signature
codesign -v MyApp.app

# Deep verification
codesign -v --deep MyApp.app

# Display signature info
codesign -d -vv MyApp.app
```

### Windows Code Signing

```powershell
# Sign executable with certificate
signtool sign /f certificate.pfx /p password /t http://timestamp.server.com app.exe

# Verify signature
signtool verify /pa app.exe
```

## Security Considerations

### 1. Key Size

```
RSA:
- Minimum: 2048 bits
- Recommended: 3072 bits
- High security: 4096 bits

ECDSA:
- Minimum: 256 bits (P-256)
- Recommended: 384 bits (P-384)
- High security: 521 bits (P-521)

Ed25519:
- Fixed: 256 bits (equivalent to ~128-bit security)
```

### 2. Hash Function

```python
# GOOD - SHA-256 or better
signature = private_key.sign(message, padding.PSS(...), hashes.SHA256())

# BETTER - SHA-512
signature = private_key.sign(message, padding.PSS(...), hashes.SHA512())

# BAD - SHA-1 (broken!)
signature = private_key.sign(message, padding.PSS(...), hashes.SHA1())
```

### 3. Random Number Generation

```python
# ECDSA requires good randomness
# Python's cryptography library handles this automatically

# NEVER implement your own random number generator!
# Use os.urandom() or secrets module for any manual crypto
import secrets
random_bytes = secrets.token_bytes(32)
```

### 4. Private Key Protection

```python
# Encrypt private key with password
from cryptography.hazmat.primitives import serialization

encrypted_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.BestAvailableEncryption(b'strong-password')
)

# Load encrypted key
with open('encrypted_key.pem', 'rb') as f:
    private_key = serialization.load_pem_private_key(
        f.read(),
        password=b'strong-password'
    )
```

### 5. Signature Malleability

```
Some signature schemes allow multiple valid signatures for the same message.

Ed25519: NOT malleable (good!)
ECDSA: Can be malleable (use canonical form)
RSA-PSS: Probabilistic (different signatures each time, but all valid)
```

## Best Practices

### 1. Use Modern Algorithms

```
✓ RSA-PSS (not PKCS#1 v1.5)
✓ ECDSA with P-256 or better
✓ Ed25519 (best choice for new systems)
✗ DSA (obsolete)
✗ RSA with PKCS#1 v1.5 (vulnerable)
```

### 2. Protect Private Keys

```
- Never commit to version control
- Use hardware security modules (HSM) for critical keys
- Use key management services (AWS KMS, Azure Key Vault)
- Encrypt keys at rest
- Limit access with proper permissions
```

### 3. Include Metadata

```python
import json
import time

def create_signed_document(content, private_key):
    metadata = {
        'content': content,
        'timestamp': int(time.time()),
        'signer': 'John Doe',
        'version': '1.0'
    }
    
    message = json.dumps(metadata, sort_keys=True).encode()
    signature = private_key.sign(message, ...)
    
    return {
        'metadata': metadata,
        'signature': signature.hex()
    }
```

### 4. Timestamp Signatures

```python
# Include timestamp to prevent replay attacks
import time

def sign_with_timestamp(message, private_key):
    timestamp = str(int(time.time()))
    data = f"{timestamp}:{message}".encode()
    signature = private_key.sign(data, ...)
    
    return {
        'message': message,
        'timestamp': timestamp,
        'signature': signature.hex()
    }

def verify_with_timestamp(signed_data, public_key, max_age=3600):
    timestamp = int(signed_data['timestamp'])
    current = int(time.time())
    
    # Check if too old
    if current - timestamp > max_age:
        return False, "Signature expired"
    
    # Verify signature
    data = f"{signed_data['timestamp']}:{signed_data['message']}".encode()
    # ... verify logic
```

## Common Mistakes

### 1. Signing Hash vs Message

```python
# WRONG - signing hash manually
hash_digest = hashlib.sha256(message).digest()
signature = private_key.sign(hash_digest, ...)  # May not work!

# RIGHT - let library handle hashing
signature = private_key.sign(message, ..., hashes.SHA256())
```

### 2. Not Validating Signatures

```python
# WRONG - trusting unsigned data
data = receive_data()
process(data)  # Danger!

# RIGHT - verify signature first
data, signature = receive_data_and_signature()
if verify_signature(data, signature):
    process(data)
else:
    reject()
```

### 3. Exposing Private Keys

```python
# WRONG
private_key = "-----BEGIN PRIVATE KEY-----\n..."  # Hardcoded!

# RIGHT
import os
key_path = os.environ.get('PRIVATE_KEY_PATH')
with open(key_path, 'rb') as f:
    private_key = load_key(f.read())
```

## ELI10

Digital signatures are like a special seal that only you can make:

**Regular signature (on paper):**
- Anyone can try to copy your signature
- Hard to prove it's really yours

**Digital signature:**
1. You have a special "stamp" that only you own (private key)
2. Anyone can see your "stamp pattern" (public key)
3. When you sign a document:
   - You use your secret stamp to make a unique mark
   - This mark is different for every document
4. Others can verify:
   - They use your public stamp pattern
   - If it matches, they know YOU signed it
   - Nobody else could have made that exact mark!

**Why it's secure:**
- Your secret stamp is like a lock that only you can use
- The public pattern lets others check your work
- Even if someone copies the signed document, they can't change it without your secret stamp!

**Real-world example:**
When you download software, the developer signs it:
- ✓ You can verify it's really from them
- ✓ Nobody tampered with the software
- ✓ The developer can't deny they released it

**Different from HMAC:**
- HMAC: Shared secret (like both having the same password)
- Digital Signature: Private/public keys (like a lock and key everyone can see fits)

## Further Resources

- [RSA Cryptography Explained](https://www.youtube.com/watch?v=wXB-V_Keiu8)
- [ECDSA Deep Dive](https://blog.cloudflare.com/ecdsa-the-digital-signature-algorithm-of-a-better-internet/)
- [Ed25519 Specification](https://ed25519.cr.yp.to/)
- [Digital Signatures Standard (DSS)](https://csrc.nist.gov/publications/detail/fips/186/4/final)
- [Cryptography Engineering (Book)](https://www.schneier.com/books/cryptography-engineering/)
- [Python Cryptography Library](https://cryptography.io/en/latest/)
- [OpenSSL Command Reference](https://www.openssl.org/docs/man1.1.1/man1/)
