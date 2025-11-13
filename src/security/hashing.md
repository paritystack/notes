# Cryptographic Hash Functions

## Overview

A cryptographic hash function is a mathematical algorithm that takes an input (message) of any size and produces a fixed-size output (hash digest). Hash functions are one-way functions designed to be computationally infeasible to reverse.

## Key Properties

### 1. Deterministic
Same input always produces the same output:
```
hash("hello") = 2cf24dba5fb0a30e...
hash("hello") = 2cf24dba5fb0a30e...  (always the same)
```

### 2. Fast Computation
Quick to compute hash for any input

### 3. Pre-image Resistance (One-way)
Given hash `h`, computationally infeasible to find message `m` where `hash(m) = h`

### 4. Collision Resistance
Computationally infeasible to find two different messages `m1` and `m2` where:
```
hash(m1) = hash(m2)
```

### 5. Avalanche Effect
Small change in input drastically changes output:
```
hash("hello")  = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
hash("helloX") = 9c70933a77f8d8d1eb5ba43c8f8c8b2e6f4c8e8a5b9e1b161e5c1fa7425e7304
```

## Common Hash Functions

| Algorithm | Output Size | Status | Use Cases |
|-----------|-------------|--------|-----------|
| **MD5** | 128 bits (16 bytes) | Broken | Checksums only |
| **SHA-1** | 160 bits (20 bytes) | Deprecated | Legacy systems |
| **SHA-256** | 256 bits (32 bytes) | Secure | General purpose |
| **SHA-512** | 512 bits (64 bytes) | Secure | High security |
| **SHA-3** | Variable | Secure | Modern alternative |
| **BLAKE2** | Variable | Secure | Fast, modern |
| **BLAKE3** | 256 bits | Secure | Fastest, modern |

## SHA-256 (Secure Hash Algorithm 256)

### Algorithm Overview

SHA-256 is part of the SHA-2 family, designed by the NSA and published in 2001.

**Process:**
1. Pad message to multiple of 512 bits
2. Initialize hash values (8 x 32-bit words)
3. Process message in 512-bit chunks
4. Each chunk goes through 64 rounds of operations
5. Produce final 256-bit hash

### Using SHA-256

#### Python Example

```python
import hashlib

# Hash a string
message = "Hello, World!"
hash_object = hashlib.sha256(message.encode())
hash_hex = hash_object.hexdigest()
print(f"SHA-256: {hash_hex}")
# Output: SHA-256: dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f

# Hash a file
def hash_file(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

file_hash = hash_file("document.pdf")
print(f"File hash: {file_hash}")

# Incremental hashing
hasher = hashlib.sha256()
hasher.update(b"Hello, ")
hasher.update(b"World!")
print(hasher.hexdigest())
# Same as hashing "Hello, World!" at once
```

#### Bash/OpenSSL Example

```bash
# Hash a string
echo -n "Hello, World!" | sha256sum
echo -n "Hello, World!" | openssl dgst -sha256

# Hash a file
sha256sum document.pdf
openssl dgst -sha256 document.pdf

# Verify file integrity
sha256sum document.pdf > checksum.txt
sha256sum -c checksum.txt

# Hash multiple files
sha256sum *.pdf > all_checksums.txt
```

### SHA-256 Output Format

```
Input: "Hello, World!"

Binary (256 bits):
11011111111111010110000000100001...

Hexadecimal (64 characters):
dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f

Base64 (44 characters):
3/1gIbsr1bCvZ2KQgJ7DpTGR3YHH9wpLKGiKNiGCmG8=
```

## SHA-3 (Keccak)

SHA-3 is based on a different construction (sponge function) than SHA-2, providing an alternative if SHA-2 is compromised.

### Using SHA-3

```python
import hashlib

message = "Hello, World!"

# SHA-3 variants
sha3_256 = hashlib.sha3_256(message.encode()).hexdigest()
sha3_512 = hashlib.sha3_512(message.encode()).hexdigest()

print(f"SHA3-256: {sha3_256}")
print(f"SHA3-512: {sha3_512}")

# SHAKE (extendable output)
shake = hashlib.shake_256(message.encode())
# Get 32 bytes of output
print(f"SHAKE256: {shake.hexdigest(32)}")
```

## BLAKE2

Faster than SHA-2 and SHA-3, with built-in keyed hashing and salting support.

### Using BLAKE2

```python
import hashlib

message = b"Hello, World!"

# BLAKE2b (optimized for 64-bit platforms)
blake2b = hashlib.blake2b(message).hexdigest()
print(f"BLAKE2b: {blake2b}")

# BLAKE2s (optimized for 8-32 bit platforms)
blake2s = hashlib.blake2s(message).hexdigest()
print(f"BLAKE2s: {blake2s}")

# Keyed hashing (MAC)
key = b"secret-key-123"
mac = hashlib.blake2b(message, key=key).hexdigest()
print(f"BLAKE2b MAC: {mac}")

# Custom digest size
digest = hashlib.blake2b(message, digest_size=16).hexdigest()
print(f"BLAKE2b-128: {digest}")

# With salt (for password hashing)
salt = b"random-salt-16bytes!"
h = hashlib.blake2b(message, salt=salt, digest_size=32)
print(f"BLAKE2b with salt: {h.hexdigest()}")
```

## Password Hashing

**WARNING**: Never use fast hashes (SHA-256, MD5) for passwords! Use specialized password hashing functions.

### Why Not SHA-256 for Passwords?

```python
# BAD - vulnerable to brute force
import hashlib
password = "password123"
hash = hashlib.sha256(password.encode()).hexdigest()
# Attacker can compute billions of SHA-256 hashes per second!
```

### Password Hashing Requirements

1. **Slow**: Intentionally slow to prevent brute force
2. **Salted**: Random salt prevents rainbow tables
3. **Adaptive**: Can increase work factor over time
4. **Memory-hard**: Requires significant memory (for some algorithms)

## bcrypt

### Overview

- Based on Blowfish cipher
- Adaptive (configurable work factor)
- Automatic salt generation
- Maximum password length: 72 bytes

### Using bcrypt

```python
import bcrypt

# Hash a password
password = b"my_secure_password"
salt = bcrypt.gensalt(rounds=12)  # 2^12 iterations
hashed = bcrypt.hashpw(password, salt)
print(f"Hashed: {hashed}")
# Output: b'$2b$12$KIXx8Z9...'

# Verify password
if bcrypt.checkpw(password, hashed):
    print("Password matches!")
else:
    print("Invalid password")

# Increase work factor over time
def needs_rehash(hashed_password, min_rounds=12):
    # Extract current rounds from hash
    parts = hashed_password.decode().split('$')
    current_rounds = int(parts[2])
    return current_rounds < min_rounds

# Complete example
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

# Usage
user_password = "SuperSecret123!"
stored_hash = hash_password(user_password)

# Later, during login
login_password = "SuperSecret123!"
if verify_password(login_password, stored_hash):
    print("Login successful!")
```

### bcrypt Hash Format

```
$2b$12$KIXx8Z9ByF7LHfG8z.yNH.Q5GF8Z9ByF7LHfG8z.yNH.Q5GF8Z9ByF7
 |  |  |                                                    |
 |  |  |                                                    |
 |  |  Salt (22 characters)                                Hash (31 chars)
 |  |
 |  Cost factor (2^12 iterations)
 |
 Algorithm identifier (2b = bcrypt)
```

## Argon2

### Overview

Winner of the Password Hashing Competition (2015). Memory-hard algorithm resistant to GPU/ASIC attacks.

**Variants:**
- **Argon2d**: Resistant to GPU attacks (not side-channel resistant)
- **Argon2i**: Resistant to side-channel attacks
- **Argon2id**: Hybrid (recommended)

### Using Argon2

```python
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Create hasher with default parameters
ph = PasswordHasher()

# Hash password
password = "my_secure_password"
hashed = ph.hash(password)
print(f"Hashed: {hashed}")
# Output: $argon2id$v=19$m=65536,t=3,p=4$...

# Verify password
try:
    ph.verify(hashed, password)
    print("Password matches!")
except VerifyMismatchError:
    print("Invalid password")

# Check if hash needs rehashing (parameters changed)
if ph.check_needs_rehash(hashed):
    new_hash = ph.hash(password)
    # Update in database

# Custom parameters
from argon2 import PasswordHasher
custom_ph = PasswordHasher(
    time_cost=3,        # Number of iterations
    memory_cost=65536,  # Memory usage in KiB (64 MB)
    parallelism=4,      # Number of parallel threads
    hash_len=32,        # Hash length in bytes
    salt_len=16         # Salt length in bytes
)

hashed = custom_ph.hash(password)
```

### Argon2 Hash Format

```
$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$hash_output_here
 |        |    |             |              |
 |        |    |             |              Hash output
 |        |    |             Salt (base64)
 |        |    Parameters (memory, time, parallelism)
 |        Version
 Variant (id, i, or d)
```

### Argon2 Parameters Guide

```python
# Low security (fast, for testing)
time_cost=1, memory_cost=8192, parallelism=1

# Medium security (default)
time_cost=3, memory_cost=65536, parallelism=4

# High security
time_cost=5, memory_cost=262144, parallelism=8

# Extreme security
time_cost=10, memory_cost=1048576, parallelism=16
```

## Salting

A salt is random data added to passwords before hashing to prevent rainbow table attacks.

### Without Salt (Vulnerable)

```python
# BAD - Same password = Same hash
hash("password123") = "abc123..."
hash("password123") = "abc123..."  # Attacker can precompute!
```

### With Salt (Secure)

```python
# GOOD - Same password = Different hashes
hash("password123" + "random_salt_1") = "xyz789..."
hash("password123" + "random_salt_2") = "def456..."
```

### Implementing Salt

```python
import hashlib
import os

def hash_password_with_salt(password):
    # Generate random salt (16 bytes = 128 bits)
    salt = os.urandom(16)
    
    # Combine password and salt
    pwdhash = hashlib.pbkdf2_hmac('sha256', 
                                  password.encode(), 
                                  salt, 
                                  100000)  # iterations
    
    # Store both salt and hash
    return salt + pwdhash

def verify_password(stored_password, provided_password):
    # Extract salt (first 16 bytes)
    salt = stored_password[:16]
    
    # Extract hash (remaining bytes)
    stored_hash = stored_password[16:]
    
    # Hash provided password with same salt
    pwdhash = hashlib.pbkdf2_hmac('sha256',
                                  provided_password.encode(),
                                  salt,
                                  100000)
    
    return pwdhash == stored_hash

# Usage
password = "my_password"
stored = hash_password_with_salt(password)

# Verify
if verify_password(stored, "my_password"):
    print("Correct password")
```

## PBKDF2 (Password-Based Key Derivation Function 2)

Standard algorithm for deriving cryptographic keys from passwords.

```python
import hashlib

password = b"my_password"
salt = b"random_salt_123"

# Derive key
key = hashlib.pbkdf2_hmac(
    'sha256',           # Hash algorithm
    password,           # Password
    salt,               # Salt
    100000,             # Iterations
    dklen=32            # Desired key length in bytes
)

print(f"Derived key: {key.hex()}")

# For password storage
def store_password(password):
    salt = os.urandom(16)
    hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    # Store: salt + hash
    return salt.hex() + '$' + hash.hex()

def check_password(password, stored):
    salt_hex, hash_hex = stored.split('$')
    salt = bytes.fromhex(salt_hex)
    hash = bytes.fromhex(hash_hex)
    
    new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return new_hash == hash
```

## Use Cases

### 1. File Integrity Verification

```bash
# Create checksum
sha256sum important_file.pdf > checksum.txt

# Later, verify file hasn't changed
sha256sum -c checksum.txt
```

### 2. Git Commits

Git uses SHA-1 (moving to SHA-256) to identify commits:
```bash
git log --oneline
# a1b2c3d Fix bug in authentication
```

### 3. Digital Signatures

Hash the message first, then sign the hash:
```
message -> hash -> encrypt with private key -> signature
```

### 4. Proof of Work (Blockchain)

```python
import hashlib
import time

def mine_block(data, difficulty=4):
    nonce = 0
    target = "0" * difficulty
    
    while True:
        message = f"{data}{nonce}"
        hash = hashlib.sha256(message.encode()).hexdigest()
        
        if hash.startswith(target):
            return nonce, hash
        
        nonce += 1

# Mine a block (find hash starting with 0000)
data = "Block data here"
nonce, hash = mine_block(data, difficulty=4)
print(f"Nonce: {nonce}, Hash: {hash}")
```

### 5. Message Deduplication

```python
import hashlib

def deduplicate_messages(messages):
    seen_hashes = set()
    unique_messages = []
    
    for msg in messages:
        msg_hash = hashlib.sha256(msg.encode()).hexdigest()
        if msg_hash not in seen_hashes:
            seen_hashes.add(msg_hash)
            unique_messages.append(msg)
    
    return unique_messages
```

### 6. Content-Addressable Storage

```python
import hashlib
import os

class ContentAddressableStorage:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def store(self, data):
        # Hash determines storage location
        hash = hashlib.sha256(data).hexdigest()
        path = os.path.join(self.storage_dir, hash)
        
        with open(path, 'wb') as f:
            f.write(data)
        
        return hash
    
    def retrieve(self, hash):
        path = os.path.join(self.storage_dir, hash)
        with open(path, 'rb') as f:
            return f.read()

# Usage
cas = ContentAddressableStorage('/tmp/cas')
content = b"Important document content"
hash = cas.store(content)
retrieved = cas.retrieve(hash)
```

## Hash Comparison

### Performance Benchmark (Python)

```python
import hashlib
import time

data = b"x" * 1000000  # 1 MB of data

algorithms = ['md5', 'sha1', 'sha256', 'sha512', 'sha3_256', 'blake2b']

for algo in algorithms:
    start = time.time()
    for _ in range(100):
        hashlib.new(algo, data).digest()
    elapsed = time.time() - start
    print(f"{algo:12} {elapsed:.3f}s")

# Typical results:
# md5          0.125s  (fastest, but insecure)
# sha1         0.156s  (fast, but deprecated)
# blake2b      0.187s  (fast and secure)
# sha256       0.234s  (standard, secure)
# sha512       0.187s  (fast on 64-bit, secure)
# sha3_256     0.876s  (slower, secure)
```

## Security Considerations

### 1. Never Use MD5 or SHA-1 for Security

```python
# VULNERABLE - collision attacks exist
md5_hash = hashlib.md5(data).hexdigest()
sha1_hash = hashlib.sha1(data).hexdigest()

# USE INSTEAD
sha256_hash = hashlib.sha256(data).hexdigest()
```

### 2. Always Salt Passwords

```python
# BAD
password_hash = hashlib.sha256(password.encode()).hexdigest()

# GOOD
import bcrypt
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
```

### 3. Use Appropriate Hash for Use Case

```
File integrity:     SHA-256, BLAKE2
Password storage:   bcrypt, Argon2, PBKDF2
General purpose:    SHA-256, SHA-3, BLAKE2
High performance:   BLAKE2, BLAKE3
Cryptographic:      SHA-256, SHA-3
```

### 4. Timing Attacks

```python
# VULNERABLE - timing attack
if hash1 == hash2:
    return True

# SAFE - constant time comparison
import hmac
if hmac.compare_digest(hash1, hash2):
    return True
```

### 5. Hash Length Extension Attacks

SHA-256 is vulnerable to length extension attacks. Use HMAC instead for authentication:

```python
# VULNERABLE
auth_tag = sha256(secret + message)

# SAFE
import hmac
auth_tag = hmac.new(secret, message, hashlib.sha256).digest()
```

## Best Practices

### 1. Password Hashing Checklist

```python
# ✓ Use specialized password hash (bcrypt, Argon2)
# ✓ Use random salt (automatic in bcrypt/Argon2)
# ✓ Use sufficient work factor
# ✓ Use constant-time comparison
# ✓ Plan for rehashing when parameters change

from argon2 import PasswordHasher
import hmac

ph = PasswordHasher()

def hash_password(password):
    return ph.hash(password)

def verify_password(password, hash):
    try:
        ph.verify(hash, password)
        return True
    except:
        return False
```

### 2. File Integrity

```bash
# Generate checksums for all files
find . -type f -exec sha256sum {} \; > checksums.txt

# Verify later
sha256sum -c checksums.txt
```

### 3. Secure Random Salt Generation

```python
import os

# Use cryptographically secure random
salt = os.urandom(16)  # 128 bits

# DON'T use regular random module
import random
salt = random.randbytes(16)  # NOT SECURE!
```

### 4. Database Schema for Passwords

```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,  -- Store full hash string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    password_updated_at TIMESTAMP
);

-- bcrypt example:
-- password_hash: $2b$12$KIXx8Z9ByF7LHfG8z.yNH.Q5GF8Z9ByF7...

-- Argon2 example:
-- password_hash: $argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$...
```

## Common Mistakes

### 1. Double Hashing

```python
# BAD - doesn't increase security
hash1 = sha256(password)
hash2 = sha256(hash1)  # No benefit!

# GOOD - use proper password hashing
hash = bcrypt.hashpw(password, bcrypt.gensalt())
```

### 2. Homemade Crypto

```python
# BAD - creating your own hash function
def my_hash(data):
    result = 0
    for byte in data:
        result = (result * 31 + byte) % 1000000007
    return result

# GOOD - use standard algorithms
import hashlib
hash = hashlib.sha256(data).hexdigest()
```

### 3. Insufficient Work Factor

```python
# BAD - too fast, vulnerable to brute force
hash = bcrypt.hashpw(password, bcrypt.gensalt(rounds=4))  # 2^4 = 16 iterations

# GOOD - sufficient work factor
hash = bcrypt.hashpw(password, bcrypt.gensalt(rounds=12))  # 2^12 = 4096 iterations
```

## ELI10

A hash function is like a magic blender for data:

1. **You put something in**: "Hello, World!"
2. **It blends it up**: The blender scrambles everything
3. **You get a unique smoothie**: "dffd6021bb2b..."

Special properties:
- **Always the same**: Same ingredients = Same smoothie
- **One-way**: Can't un-blend the smoothie to get ingredients back
- **Tiny changes matter**: "Hello, World!" vs "Hello, World?" = Completely different smoothies
- **Same size**: Whether you blend a strawberry or a watermelon, you always get the same size cup

**For passwords**, we use special slow blenders (bcrypt, Argon2):
- Regular blender: Makes 1 million smoothies per second (easy to guess passwords!)
- Password blender: Makes 10 smoothies per second (hard to guess passwords!)

**Salt** is like adding random spices:
- Without salt: Everyone who uses "password123" gets the same smoothie
- With salt: Everyone gets different random spices, so same password = different smoothies

## Further Resources

- [SHA-256 Specification (NIST)](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
- [Password Hashing Competition](https://password-hashing.net/)
- [Argon2 RFC 9106](https://www.rfc-editor.org/rfc/rfc9106.html)
- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [Hash Length Extension Attacks](https://blog.skullsecurity.org/2012/everything-you-need-to-know-about-hash-length-extension-attacks)
- [bcrypt Documentation](https://github.com/pyca/bcrypt/)
- [Argon2 Documentation](https://argon2-cffi.readthedocs.io/)
