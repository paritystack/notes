# Security

Comprehensive security reference covering cryptography, authentication, and secure communications.

## Cryptography

### [Encryption](encryption.md)
- Symmetric encryption (AES, ChaCha20)
- Asymmetric encryption (RSA, ECC)
- Encryption modes and best practices
- Key management

### [Hashing](hashing.md)
- Cryptographic hash functions
- SHA-256, SHA-3, BLAKE2
- Password hashing (bcrypt, Argon2)
- Hash-based applications

### [HMAC](hmac.md)
- Hash-based Message Authentication Code
- Message integrity and authenticity
- HMAC construction and usage
- Applications in APIs and tokens

## Authentication & Signatures

### [Digital Signatures](digital_signatures.md)
- RSA signatures
- ECDSA (Elliptic Curve Digital Signature Algorithm)
- EdDSA (Edwards-curve Digital Signature Algorithm)
- Signature verification
- Applications (code signing, documents)

### [Certificates](certificates.md)
- X.509 certificates
- Certificate Authorities (CAs)
- Certificate chains and trust
- Certificate management
- Let's Encrypt and ACME protocol

## Secure Communications

### [SSL/TLS](ssl_tls.md)
- TLS handshake process
- Cipher suites
- Certificate validation
- TLS 1.2 vs TLS 1.3
- Common vulnerabilities (BEAST, POODLE, Heartbleed)
- Best practices and configuration

## Quick Reference

### Common Algorithms

| Algorithm | Type | Key Size | Use Case |
|-----------|------|----------|----------|
| **AES** | Symmetric | 128/192/256-bit | General encryption |
| **ChaCha20** | Symmetric | 256-bit | Mobile/embedded |
| **RSA** | Asymmetric | 2048/4096-bit | Key exchange, signatures |
| **ECDSA** | Asymmetric | 256-bit | Signatures (Bitcoin) |
| **SHA-256** | Hash | N/A | Checksums, Bitcoin |
| **bcrypt** | Password Hash | N/A | Password storage |
| **Argon2** | Password Hash | N/A | Password storage (modern) |

### Security Best Practices

1. **Use Modern Algorithms**
   - AES-256 for symmetric encryption
   - RSA-2048 minimum, prefer ECC
   - SHA-256 or SHA-3 for hashing
   - Argon2 for password hashing

2. **Key Management**
   - Generate strong random keys
   - Rotate keys regularly
   - Use HSM for critical keys
   - Never hardcode secrets

3. **TLS Configuration**
   - Use TLS 1.2 minimum (prefer 1.3)
   - Disable weak cipher suites
   - Enable Perfect Forward Secrecy
   - Use strong certificate chains

4. **Password Storage**
   - Never store plaintext passwords
   - Use bcrypt or Argon2
   - Add unique salt per password
   - Use appropriate work factors

5. **API Security**
   - Use HMAC for message integrity
   - Implement rate limiting
   - Use short-lived tokens
   - Validate all inputs

## Common Tools

```bash
# OpenSSL
openssl enc -aes-256-cbc -in file.txt -out file.enc
openssl req -new -x509 -days 365 -key key.pem -out cert.pem

# Generate keys
ssh-keygen -t ed25519
openssl genrsa -out private.key 2048

# Hashing
sha256sum file.txt
openssl dgst -sha256 file.txt

# Certificate inspection
openssl x509 -in cert.pem -text -noout
openssl s_client -connect example.com:443
```

## Related Topics

- Network security (firewalls, VPNs)
- Application security (OWASP Top 10)
- Authentication protocols (OAuth, SAML)
- Blockchain and cryptocurrencies
