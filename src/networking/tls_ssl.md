# TLS/SSL & PKI

## Overview

TLS (Transport Layer Security) is the protocol that turns an ordinary TCP byte stream
into an authenticated, confidential, integrity-protected channel. It's what the "S" in
HTTPS, the encryption under [SMTP](email_protocols.md) STARTTLS, the auth in
[gRPC](grpc.md) mTLS, and the inner tunnel of EAP-TLS ([802.1X](nac_8021x.md)) all rely
on. SSL (Secure Sockets Layer) is the obsolete predecessor — SSL 2.0/3.0 are broken and
disabled everywhere; the name "SSL" survives only colloquially.

```
What TLS gives you:
  Confidentiality  → eavesdroppers see only ciphertext
  Integrity        → tampering is detected (AEAD)
  Authentication   → you're really talking to example.com (certificates)
  (optional) mutual auth → server also verifies the client
```

TLS sits between the transport ([TCP](tcp.md)) and the application. For UDP there's DTLS
(Datagram TLS), used by [WebRTC](webrtc.md), [CoAP](iot_protocols.md), and
DTLS-SRTP ([SIP/VoIP](sip_voip.md)). [QUIC](quic.md) embeds the TLS 1.3 handshake
directly instead of running TLS as a layer.

## Why TLS 1.3 (and why not 1.2)

```
TLS 1.3 (RFC 8446, 2018) vs TLS 1.2:
  ✓ 1-RTT handshake (was 2-RTT) — faster connection setup
  ✓ 0-RTT resumption — send data on the first flight
  ✓ Forward secrecy is mandatory (ephemeral ECDHE only)
  ✓ Removed broken crypto: RSA key exchange, CBC, RC4, SHA-1, compression
  ✓ Only 5 AEAD cipher suites left — no footgun negotiation
  ✓ Encrypts most of the handshake (certificates are hidden)

  ✗ 0-RTT data is replayable — only safe for idempotent requests
  ✗ Middleboxes that did passive RSA decryption no longer work (by design)
```

Versions still seen: TLS 1.2 remains common for compatibility; TLS 1.0/1.1 are deprecated
(RFC 8996) and should be refused.

## How It Works

### TLS 1.3 handshake (1-RTT)

```
Client                                                    Server
  |  ClientHello                                              |
  |    - supported_versions: [1.3]                            |
  |    - key_share: client's ECDHE public key (X25519)        |
  |    - cipher_suites, signature_algorithms                  |
  |    - SNI: example.com   ALPN: [h2, http/1.1]              |
  | -------------------------------------------------------->  |
  |                                                            | picks group,
  |                          ServerHello                       | derives keys
  |                            - key_share: server ECDHE pub   |
  |   <------------------------------------------------------- |
  |   {EncryptedExtensions}  ← from here on, encrypted         |
  |   {Certificate}          ← server's cert chain             |
  |   {CertificateVerify}    ← signature over the transcript   |
  |   {Finished}                                               |
  |                                                            |
  | {Finished}                                                 |
  | [Application Data] --------------------------------------> |
  |                                                            |

  { } = encrypted with handshake keys
  [ ] = encrypted with application keys
  One round trip before app data flows.
```

The shared secret comes from **ECDHE** (Ephemeral Elliptic Curve Diffie-Hellman, usually
X25519 or P-256): both sides contribute an ephemeral key, so even if the server's
long-term private key leaks later, past sessions can't be decrypted — that's **Perfect
Forward Secrecy (PFS)**. `CertificateVerify` proves the server owns the private key for
the cert by signing the handshake transcript.

### TLS 1.2 handshake (2-RTT, for contrast)

```
Client                                  Server
  | ClientHello ----------------------->  |
  |        <----------------------------- ServerHello, Certificate,
  |                                        ServerKeyExchange, ServerHelloDone
  | ClientKeyExchange, ChangeCipherSpec    |
  | Finished --------------------------->  |
  |        <----------------------------- ChangeCipherSpec, Finished
  | Application Data <------------------>  |
   Two round trips. RSA key exchange (no PFS) was historically common here.
```

### Record layer

After the handshake, application bytes are split into **TLS records** (max 16 KB each),
each encrypted with an AEAD cipher. AEAD (AES-GCM, ChaCha20-Poly1305) provides
confidentiality and integrity in one operation — no separate MAC, no padding oracle.

```
TLS 1.3 cipher suites (the entire list):
  TLS_AES_128_GCM_SHA256
  TLS_AES_256_GCM_SHA384
  TLS_CHACHA20_POLY1305_SHA256      ← preferred on mobile/no-AES-HW
  TLS_AES_128_CCM_SHA256
  TLS_AES_128_CCM_8_SHA256
```

### Session resumption & 0-RTT

After a full handshake the server can issue a **session ticket** (PSK). On reconnect the
client presents it and skips certificate exchange (1-RTT, or 0-RTT if it sends
"early data" immediately).

```
0-RTT caveat:
  Early data can be REPLAYED by an attacker who captures it.
  → Only use for idempotent, non-state-changing requests (GET, not POST).
```

## Certificates & PKI

A certificate binds a **public key** to an **identity** (domain name), signed by a
**Certificate Authority (CA)** the client already trusts.

```
Chain of trust:

  Root CA (self-signed, in OS/browser trust store, offline)
      |  signs
  Intermediate CA (online, day-to-day issuance)
      |  signs
  Leaf / end-entity cert  (example.com)  ← served by the website

Client validates: leaf → intermediate → root, checking each signature,
                   validity dates, and that root is trusted.
```

### X.509 fields that matter

```
Subject / CN:            legacy name field (browsers ignore CN now)
Subject Alternative Name: the names that actually count (DNS:example.com, DNS:*.example.com)
Issuer:                  who signed this cert
Validity:                notBefore / notAfter
Public Key:              RSA-2048/3072 or ECDSA P-256
Key Usage / EKU:         e.g. serverAuth, clientAuth (EKU matters for mTLS)
Basic Constraints:       CA:TRUE/FALSE, path length
SCT:                     Certificate Transparency proofs (browser requirement)
```

### Validation levels & revocation

```
DV  (Domain Validation)     → proves control of the domain (most certs, e.g. Let's Encrypt)
OV  (Organization Validation)
EV  (Extended Validation)   → vetted org identity (no longer shown specially in browsers)

Revocation (when a key is compromised before expiry):
  CRL          → big list of revoked serials (clunky)
  OCSP         → online "is this cert still good?" query (privacy leak: CA sees your browsing)
  OCSP stapling→ server fetches OCSP response and staples it into the handshake (fast, private)
  Short-lived certs (90 days, ACME) → reduce reliance on revocation entirely
```

## Extensions: SNI, ALPN, ECH

```
SNI  (Server Name Indication): ClientHello says "I want example.com" so one IP can host
                                many TLS sites (virtual hosting). Sent in cleartext in TLS 1.3.
ALPN (Application-Layer Protocol Negotiation): client+server agree on h2 / http/1.1 / etc.
                                in the handshake — this is how HTTP/2 (see http2.md) is selected.
ECH  (Encrypted Client Hello): encrypts SNI (and the rest of ClientHello) so the visited
                                hostname isn't exposed on the wire. Successor to the failed ESNI.
```

## Mutual TLS (mTLS)

Normally only the client verifies the server. With **mTLS** the server also requests a
client certificate via `CertificateRequest`, and the client proves possession with its own
`CertificateVerify`. Common for service-to-service auth ([gRPC](grpc.md), service meshes),
VPNs, and EAP-TLS ([802.1X](nac_8021x.md)).

```
Server config (nginx):
  ssl_verify_client on;
  ssl_client_certificate /etc/nginx/ca.crt;   # CA that signs valid clients
```

## ACME / Let's Encrypt

ACME automates certificate issuance: prove domain control, get a 90-day cert, auto-renew.

```
HTTP-01 challenge:
  CA: "serve this token at http://example.com/.well-known/acme-challenge/<token>"
  client does so → CA fetches it → control proven → cert issued

DNS-01 challenge (needed for wildcards):
  prove control by publishing a TXT record _acme-challenge.example.com

Tools: certbot, acme.sh, caddy (automatic), lego
```

## Practical commands

```bash
# Inspect a server's live certificate and negotiated params
openssl s_client -connect example.com:443 -servername example.com </dev/null

# Show negotiated TLS version + cipher only
openssl s_client -connect example.com:443 -servername example.com </dev/null 2>/dev/null \
  | grep -E "Protocol|Cipher"

# Decode a PEM certificate
openssl x509 -in cert.pem -noout -text
openssl x509 -in cert.pem -noout -subject -issuer -dates -ext subjectAltName

# Check the full chain a server sends
openssl s_client -connect example.com:443 -showcerts </dev/null

# Test which TLS versions a server accepts
openssl s_client -connect example.com:443 -tls1_2 </dev/null
openssl s_client -connect example.com:443 -tls1_3 </dev/null

# Verify expiry quickly
echo | openssl s_client -connect example.com:443 2>/dev/null \
  | openssl x509 -noout -enddate

# Generate a self-signed cert (testing/dev)
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout key.pem -out cert.pem -days 90 -nodes -subj "/CN=localhost"
```

## Security notes & pitfalls

```
- Disable TLS 1.0/1.1 and SSLv3; prefer TLS 1.3, allow 1.2.
- Use ECDHE suites only (forward secrecy); never static-RSA key exchange.
- Validate the full chain AND the hostname (SAN), AND expiry — skipping any is a classic bug.
- Don't trust self-signed certs in production; pin a CA, not a leaf, if pinning.
- 0-RTT early data is replayable → restrict to idempotent requests.
- Mixed content / downgrade: enforce HTTPS with HSTS so attackers can't strip TLS.
- Certificate Transparency: browsers require SCTs; rogue certs become publicly auditable.
- Private keys: 0600 perms, never in git, rotate on suspicion of compromise.
```

## Related

- [HTTP/HTTPS](http.md), [HTTP/2](http2.md) — TLS is selected via ALPN here
- [QUIC](quic.md) — embeds the TLS 1.3 handshake instead of layering over TCP
- [SSH](ssh.md) — different handshake, same goals (auth + confidentiality)
- [802.1X / RADIUS](nac_8021x.md) — EAP-TLS reuses TLS for network access auth
- [Email protocols](email_protocols.md) — STARTTLS / implicit TLS for SMTP/IMAP
- [gRPC](grpc.md) — mTLS for service authentication
