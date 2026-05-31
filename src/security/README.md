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

## Authentication & Authorization

### [OAuth 2.0](oauth2.md)
- Authorization framework and grant types
- Authorization Code, Client Credentials, PKCE
- Access tokens and refresh tokens
- OpenID Connect for authentication
- Implementation best practices

### [JWT (JSON Web Tokens)](jwt.md)
- Token structure (header, payload, signature)
- Signing algorithms (HS256, RS256, ES256)
- Token validation and verification
- Use cases and security considerations
- Best practices for token management

## Digital Signatures

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

## Application & API Security

### [API Security](api_security.md)
- OWASP API Security Top 10 (BOLA/IDOR, mass assignment, SSRF)
- Authentication (API keys, OAuth2, JWT, mTLS) vs authorization
- Input validation, rate limiting, and gateway controls
- GraphQL/gRPC specifics and API inventory

### [Threat Modeling](threat_modeling.md)
- Data-flow diagrams and trust boundaries
- STRIDE taxonomy and attack trees
- Risk ranking (DREAD, likelihood × impact)
- Mitigation strategies and tooling

## Key & Secrets Management

### [Key Management](key_management.md)
- Key lifecycle (generate → rotate → revoke → destroy)
- HSMs, cloud KMS, and envelope encryption
- Key hierarchy, separation, and rotation
- Standards (PKCS#11, KMIP, JWK, FIPS 140)

### [Secrets Management](secrets_management.md)
- Vault, cloud secret stores, sealed/external secrets
- Static vs dynamic (short-lived) secrets
- Workload identity and the secret-zero problem
- Leak detection and rotation

## Supply Chain & Future-Proofing

### [Supply Chain Security](supply_chain_security.md)
- SBOM, SLSA, and dependency hygiene
- Dependency confusion, typosquatting, build compromise
- Signing & provenance (Sigstore/cosign, in-toto)
- Securing the CI/CD pipeline

### [Post-Quantum Cryptography](post_quantum_crypto.md)
- Shor/Grover and the quantum threat model
- NIST standards (ML-KEM, ML-DSA, SLH-DSA)
- Hybrid deployment and crypto-agility
- Harvest-now-decrypt-later and migration

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
