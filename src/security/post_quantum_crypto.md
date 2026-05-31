# Post-Quantum Cryptography

## Overview

Post-quantum cryptography (PQC) is the set of cryptographic algorithms designed to resist
attacks by **large-scale quantum computers**. The public-key crypto securing the internet
today — RSA, Diffie–Hellman, and elliptic curves underpinning [TLS](ssl_tls.md),
[certificates](certificates.md), and [digital signatures](digital_signatures.md) — relies
on math problems (integer factorization, discrete log; see [number theory](../maths/discrete_math.md))
that a quantum computer running **Shor's algorithm** would solve efficiently. PQC replaces
those with problems believed hard even for quantum machines. This is a *now* problem despite
quantum computers not yet being capable — because of "harvest now, decrypt later."

```
Quantum threat, split cleanly:
  PUBLIC-KEY (RSA, ECDH, ECDSA)  → BROKEN by Shor's algorithm. Must be replaced.
  SYMMETRIC (AES, hashes)        → weakened by Grover's (√ speedup) → just double key sizes.
                                   AES-256 stays safe; SHA-256/384 stay safe.
```

## Harvest now, decrypt later

The reason to act before quantum computers exist:

```
Today:   attacker records your encrypted traffic (TLS sessions, VPN, backups).
Future:  once a cryptographically-relevant quantum computer (CRQC) arrives, they
         decrypt the stored ciphertext retroactively.
⇒ Any data whose confidentiality must outlive ~10–20 years is ALREADY at risk now.
   (state secrets, health records, long-term IP, biometric data)
```

## Shor vs Grover

```
Shor's algorithm   factors integers & solves discrete log in polynomial time.
                   → demolishes RSA, DH, ECDH, ECDSA, EdDSA entirely.
Grover's algorithm gives a quadratic speedup for brute-force search.
                   → halves the effective key strength of symmetric crypto.
                   Counter: use AES-256 (gives 128-bit post-quantum security) and
                   SHA-384/SHA-512. Symmetric crypto does NOT need new algorithms.
```

## The NIST standards (2024)

After a multi-year competition, NIST standardized the first PQC algorithms in August 2024.
These are the ones to build toward:

```
FIPS 203  ML-KEM   (formerly Kyber)      Key Encapsulation — replaces ECDH key exchange.
FIPS 204  ML-DSA   (formerly Dilithium)  Signatures — general-purpose replacement for ECDSA.
FIPS 205  SLH-DSA  (formerly SPHINCS+)   Stateless hash-based signatures — conservative,
                                         larger/slower, but security rests only on hashes.
(coming)  FN-DSA   (Falcon)              compact lattice signatures.

Two jobs, two algorithm types:
  KEM (ML-KEM)        establish a shared secret  → the TLS handshake
  Signature (ML-DSA)  prove authenticity/identity → certificates, code signing
```

## Families of hard problems

PQC doesn't rely on factoring/discrete-log. The main approaches:

```
Lattice-based    ML-KEM, ML-DSA, Falcon. Best all-round size/speed → the mainstream choice.
                 Hardness: Learning With Errors (LWE) / shortest-vector problems.
Hash-based       SLH-DSA. Signatures only; security reduces to hash strength → very
                 conservative, but large signatures.
Code-based       Classic McEliece (KEM). Decades-old, trusted, but huge public keys.
Isogeny-based    once compact (SIKE) — BROKEN in 2022, a cautionary tale: "quantum-safe"
                 ≠ "classically safe"; new math needs years of cryptanalysis.
```

## Migration & crypto-agility

The practical engineering challenge — and where today's work happens:

```
Hybrid mode (the deployment strategy)
  run a CLASSICAL + a PQC algorithm together; the session is safe if EITHER holds.
  e.g. TLS 1.3 key exchange X25519MLKEM768 — already shipping in Chrome, OpenSSL 3.5,
  Cloudflare. Protects against both "PQC is broken later" and "classical is broken by
  quantum" while confidence in PQC builds.

Crypto-agility (the prerequisite)
  design systems so the algorithm can be swapped WITHOUT re-architecting:
    - no hardcoded key sizes / curve names
    - negotiate algorithms; version your crypto
    - inventory where crypto is used (you can't migrate what you can't find)
  This is the same property that makes routine key rotation possible — see
  ../security/key_management.md.
```

```
Migration order (by risk):
  1. Inventory all uses of public-key crypto (TLS, VPN, signing, tokens).
  2. Long-confidentiality data first (harvest-now-decrypt-later targets) → hybrid KEM.
  3. Signatures/PKI next (longer-lived roots; plan for larger keys & signatures).
  4. Watch performance: PQC keys/signatures are LARGER (KB not bytes) → bigger handshakes,
     certs, and packets; test MTU/fragmentation impact.
```

## Practical impact today

```
- TLS: hybrid ML-KEM key exchange is live in major browsers/servers (2024–2025).
- Signatures lag KEM: PKI/cert migration is slower (bigger ecosystem, larger artifacts).
- Most apps: rely on TLS libraries & cloud providers to adopt PQC underneath — your job
  is crypto-agility and staying on current libraries, not implementing lattices yourself.
- NEVER roll your own PQC — use vetted libraries (liboqs, OpenSSL 3.5+, BoringSSL).
```

## Where this connects

- **[Encryption](encryption.md)** / **[Digital signatures](digital_signatures.md)** — the
  primitives PQC replaces (public-key) or merely resizes (symmetric).
- **[TLS / SSL](ssl_tls.md)** — where hybrid PQC key exchange is rolling out first.
- **[Key management](key_management.md)** — crypto-agility makes the migration survivable.
- **[Number theory](../maths/discrete_math.md)** — the factoring/discrete-log assumptions
  Shor breaks.

## Pitfalls

- **"Quantum computers don't exist yet, so we're fine"** — harvest-now-decrypt-later makes
  long-lived secrets vulnerable *today*.
- **Quantum-safe ≠ proven safe** — SIKE was broken classically in 2022; prefer hybrids.
- **Rolling your own** — use standardized, vetted implementations only.
- **No crypto-agility** — hardcoded algorithms turn migration into a rewrite.
- **Forgetting size impact** — larger keys/signatures break assumptions about packet/cert
  sizes and handshake latency.
