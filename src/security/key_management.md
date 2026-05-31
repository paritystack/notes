# Key Management

## Overview

Cryptography moves the secret from the *algorithm* to the *key* — so the security of
[encryption](encryption.md), [HMAC](hmac.md), [digital signatures](digital_signatures.md),
and [TLS](ssl_tls.md) reduces almost entirely to **how well you manage keys**. A perfect
cipher with a leaked or never-rotated key is worthless. This note covers the key
lifecycle, where keys actually live (HSMs, KMS), and the envelope-encryption pattern that
makes large-scale key management tractable. For runtime distribution of keys and other
secrets to applications, see [secrets management](secrets_management.md).

```
Kerckhoffs's principle: the system should be secure even if everything about it
except the key is public knowledge. → All secrecy lives in the key.
```

## The key lifecycle

A key is not a static value; it has a managed life from birth to destruction. Most
breaches exploit a weak point in this chain, not the cipher.

```
 Generate ─► Distribute ─► Store ─► Use ─► Rotate ─► Revoke ─► Destroy
    │            │           │       │        │         │         │
  strong CSPRNG  TLS/wrap   HSM/KMS  least-   on a      compromise zeroize
  enough bits    never       never   priv     schedule  / expiry   memory
                 in plaintext  in code         or breach
```

```
Generate   use a cryptographically secure RNG; correct length (AES-256, RSA-3072/4096,
           ECC P-256). Never derive a key from a low-entropy password without a KDF.
Store      never in source, config files, or env vars in plaintext → HSM/KMS/vault.
Rotate     limits the blast radius and data exposed if a key leaks.
Revoke     a compromised key must be invalidated fast (CRL/OCSP for certs).
Destroy    securely erase old key material (zeroize) once no data depends on it.
```

## Where keys live

```
Risk ↑ / convenience ↓
┌───────────────────────────────────────────────────────────────────────┐
│ HSM (Hardware Security Module)   keys never leave the device; crypto    │  strongest
│                                  ops happen inside. FIPS 140-2/3.       │
│ Cloud KMS (AWS KMS, GCP KMS,     managed HSMs behind an API; you call   │
│   Azure Key Vault)               "encrypt/sign" without seeing the key. │
│ Secrets manager (Vault)          stores & leases keys/secrets to apps.  │
│ Encrypted file / keystore        e.g. PKCS#12, JKS — better than nothing│
│ Plaintext in env var / config    DON'T. One log line or leak = game over│  weakest
└───────────────────────────────────────────────────────────────────────┘
```

A **HSM** is tamper-resistant hardware that generates and uses keys without ever
exporting them — the gold standard for root/signing keys (CAs, code signing, payment).
**Cloud KMS** gives most of that benefit as an API: you ask the service to encrypt/decrypt
or sign, and the key material never reaches your application.

## Envelope encryption

The pattern that makes KMS scale. You don't encrypt gigabytes with the KMS-held key
directly; instead you use a two-tier hierarchy:

```
        ┌──────────────────────────────────────────────┐
        │  Root/Master Key (KEK)  — lives in HSM/KMS,   │
        │  never leaves, rarely rotates                 │
        └───────────────┬──────────────────────────────┘
                        │ encrypts (wraps)
        ┌───────────────▼──────────────────────────────┐
        │  Data Encryption Key (DEK) — random per       │
        │  file/record; encrypts the actual data        │
        └───────────────┬──────────────────────────────┘
                        │ encrypts
                     [ your data ]

Stored next to the data: the ENCRYPTED DEK + ciphertext.
To read: ask KMS to unwrap the DEK with the KEK, then decrypt locally.
```

Why it wins:

```
- Bulk data encrypted locally (fast); only the tiny DEK touches KMS.
- Rotate the KEK → just re-wrap DEKs, no need to re-encrypt all data.
- Per-record DEKs limit blast radius and enable crypto-shredding
  (delete the DEK → the data is unrecoverable, instant "secure delete").
```

## Key hierarchy & separation

```
Master key (KEK)       protects other keys; in HSM/KMS
  └─ Data keys (DEK)   protect data
Signing keys           separate from encryption keys (different EKU/purpose)
Per-environment keys   dev / staging / prod keys are NEVER shared
Per-tenant keys        isolate customers; enables per-tenant crypto-shred
```

Use a key for exactly one purpose — reusing an encryption key for signing (or across
environments) breaks isolation and complicates rotation.

## Rotation

```
Why:  limit data exposed by a single leaked key; meet compliance (PCI-DSS, etc.).
How:  keep key versions; new data uses the new version; old data stays readable under
      its (still-available) old version until lazily re-encrypted.
Crypto-agility: design so the algorithm/key can change without re-architecting —
      essential for the migration to post-quantum crypto (see below).
```

Automated rotation (KMS does this for you) beats manual — manual rotation that's "too
painful" simply never happens, which is the real-world failure mode.

## Symmetric vs asymmetric key handling

```
Symmetric (AES, HMAC)   same secret both sides → the hard problem is DISTRIBUTION.
                        Solved by wrapping with a public key, or KMS, or a KEK.
Asymmetric (RSA, ECC)   public key shared freely; PRIVATE key is the crown jewel →
                        the hard problem is PROTECTING & proving ownership.
                        Private keys belong in HSMs (CA roots, code signing).
```

Key exchange (ECDHE in [TLS 1.3](ssl_tls.md)) sidesteps distribution entirely by
*deriving* a fresh shared key per session — giving forward secrecy.

## Standards & interfaces

```
PKCS#11        standard API to talk to HSMs/tokens
KMIP           Key Management Interoperability Protocol (enterprise key servers)
JWK / JWKS     JSON key format; JWKS endpoint publishes public keys for JWT verification
FIPS 140-2/3   US validation levels for crypto modules / HSMs
```

## Where this connects

- **[Encryption](encryption.md)** — the keys this note manages.
- **[Certificates](certificates.md)** / **[TLS](ssl_tls.md)** — CA private-key protection,
  cert/key rotation, revocation.
- **[Secrets management](secrets_management.md)** — getting keys/secrets to running apps.
- **[Post-quantum crypto](post_quantum_crypto.md)** — crypto-agility makes the migration
  survivable.

## Pitfalls

- **Hardcoded keys** — in source/config/images; found by every secret scanner and attacker.
- **One key forever** — no rotation means a single leak exposes *all* historical data.
- **Same key everywhere** — sharing across environments/tenants destroys isolation.
- **Weak generation** — deriving keys from passwords without a KDF, or using a non-CSPRNG.
- **Forgetting destruction** — old keys lingering in backups/memory re-expose "deleted" data.
- **No crypto-agility** — algorithms hardwired so deeply that rotation or PQC migration
  means a rewrite.
