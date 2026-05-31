# Email Protocols (SMTP / IMAP / POP3 + SPF/DKIM/DMARC)

## Overview

Email is split into two halves: **transport** (getting a message from sender to the
recipient's mailbox) and **access** (a user reading mail from their mailbox).

```
Transport:  SMTP  — server-to-server relay AND client submission
Access:     IMAP  — read mail that stays on the server (multi-device)
            POP3  — download mail to one device (older model)
Anti-abuse: SPF, DKIM, DMARC, MTA-STS, DANE — prove a message is legit
```

All three core protocols are old, text-based, line-oriented over [TCP](tcp.md), and were
designed without encryption — TLS ([see TLS note](tls_ssl.md)) was bolted on later via
STARTTLS and dedicated "implicit TLS" ports.

## The journey of an email

```
  Alice (alice@a.com)                                    Bob (bob@b.com)
       |                                                       ^
       | 1. Submission (SMTP, port 587, authenticated)         |
       v                                                       |
   MSA / outbound MTA  (a.com mail server)                     |
       |                                                       | 5. Access
       | 2. DNS MX lookup for b.com                            |    (IMAP 993
       | 3. Relay (SMTP, port 25, server-to-server)            |     or POP3 995)
       v                                                       |
   inbound MTA (b.com)  --4. local delivery (LMTP)-->  Mailbox (b.com store)

  MUA = Mail User Agent (Thunderbird, Apple Mail, webmail)
  MSA = Mail Submission Agent      MTA = Mail Transfer Agent
  MDA/LMTP = Mail Delivery Agent (writes to the mailbox)
```

The key DNS step: to deliver to `b.com`, the sending MTA queries the **MX records**
([DNS](dns.md)) for `b.com`, which list the receiving mail servers by priority.

```bash
dig MX b.com +short      # e.g. "10 mx1.b.com."  (lower number = higher priority)
```

## SMTP — Simple Mail Transfer Protocol

```
Ports:
  25   relay (MTA → MTA). Often blocked outbound by ISPs to fight spam.
  587  submission (MUA → MSA), authenticated, STARTTLS. The right port for clients.
  465  submission over implicit TLS (SMTPS) — back in favor (RFC 8314).

Envelope vs headers:
  MAIL FROM / RCPT TO  = the "envelope" — what actually routes the mail
  From:/To:/Subject:   = headers inside DATA — what the user sees
  These can differ (that's how mailing lists work — and how spoofing works).
```

### A real SMTP submission session

```
C: EHLO mua.a.com
S: 250-mail.a.com Hello
S: 250-STARTTLS
S: 250-AUTH LOGIN PLAIN
S: 250 SIZE 52428800
C: STARTTLS
S: 220 Ready to start TLS
   <TLS handshake — everything after is encrypted>
C: EHLO mua.a.com
C: AUTH LOGIN ...               # authenticate (only on submission ports)
C: MAIL FROM:<alice@a.com>      # envelope sender
S: 250 OK
C: RCPT TO:<bob@b.com>          # envelope recipient
S: 250 OK
C: DATA
S: 354 End with <CR><LF>.<CR><LF>
C: From: Alice <alice@a.com>
C: To: Bob <bob@b.com>
C: Subject: Hi
C:
C: Message body here.
C: .
S: 250 OK queued as ABC123
C: QUIT
```

STARTTLS upgrades a plaintext connection to TLS in-band; implicit TLS (465/993/995) does
TLS from the first byte. Implicit TLS is now preferred because STARTTLS can be stripped by
an active attacker (the "STARTTLS stripping" attack) unless MTA-STS/DANE enforce it.

## IMAP vs POP3 — accessing the mailbox

```
                    IMAP (993)                  POP3 (995)
  Model         mail stays on server        download then (usually) delete
  Multi-device  yes — folders sync          poor — one device owns the mail
  Folders       server-side folders/flags   inbox only
  Search        server-side                 client-side after download
  Offline       caches copies               full local copy
  Use today     the default                 legacy / low-resource setups
```

```
POP3 minimal session:           IMAP is stateful with folders/flags:
  USER bob                        a LOGIN bob secret
  PASS secret                     a SELECT INBOX
  STAT      (count, size)         a FETCH 1:* (FLAGS)
  RETR 1    (get message 1)       a SEARCH UNSEEN
  DELE 1                          a STORE 3 +FLAGS (\Seen)
  QUIT                            a LOGOUT
```

## Anti-spoofing: SPF, DKIM, DMARC

Because SMTP lets anyone claim any `From:`, three DNS-based mechanisms let receivers verify
that mail claiming to be from your domain is authorized.

### SPF — Sender Policy Framework

A DNS TXT record listing which IPs are allowed to send mail for the domain. The receiver
checks the **envelope** `MAIL FROM` domain against the sending IP.

```
a.com.  IN TXT  "v=spf1 ip4:198.51.100.0/24 include:_spf.google.com -all"
                                                                     ^
                                          -all = hard fail others   |
                                          ~all = soft fail
Limitation: SPF checks the envelope sender, not the visible From:, and breaks on
forwarding (the relay's IP isn't in your SPF).
```

### DKIM — DomainKeys Identified Mail

The sending server **signs** selected headers + body with a private key; the public key is
published in DNS. The receiver verifies the signature — proving the message wasn't altered
and really came from the domain. Survives forwarding.

```
Header added to the message:
  DKIM-Signature: v=1; a=rsa-sha256; d=a.com; s=mail2024;
                  h=from:to:subject:date; bh=<body hash>; b=<signature>

Public key in DNS:
  mail2024._domainkey.a.com.  IN TXT  "v=DKIM1; k=rsa; p=MIIBIj...<pubkey>"
```

### DMARC — alignment + policy + reporting

DMARC ties SPF/DKIM to the **visible From:** domain (requiring "alignment"), tells
receivers what to do on failure, and requests reports.

```
_dmarc.a.com.  IN TXT  "v=DMARC1; p=reject; rua=mailto:dmarc@a.com; pct=100; aspf=s; adkim=s"

  p=none        monitor only (start here)
  p=quarantine  send to spam
  p=reject      bounce outright
  Pass = SPF or DKIM passes AND aligns with the From: domain.
```

### Transport security policy: MTA-STS & DANE

```
MTA-STS  — publishes (via HTTPS + DNS) that your domain requires TLS for inbound SMTP,
           defeating STARTTLS stripping. Policy file at https://mta-sts.<domain>/.well-known/...
DANE     — uses DNSSEC (see dns.md) TLSA records to pin the receiving MTA's TLS cert.
TLS-RPT  — reporting for TLS delivery failures.
```

## Practical commands

```bash
# MX, SPF, DMARC, DKIM lookups
dig MX example.com +short
dig TXT example.com +short                       # SPF lives here
dig TXT _dmarc.example.com +short
dig TXT selector._domainkey.example.com +short

# Test SMTP delivery interactively (swaks is the easiest)
swaks --to bob@b.com --from alice@a.com --server mail.a.com:587 --tls --auth

# Raw STARTTLS check with openssl
openssl s_client -connect mail.a.com:587 -starttls smtp
openssl s_client -connect imap.a.com:993            # implicit TLS IMAP

# Read DKIM-Signature / Authentication-Results headers of a received message
# (most providers add an Authentication-Results: header summarizing spf/dkim/dmarc)
```

## Security notes & pitfalls

```
- Use submission ports (587/465) with AUTH + TLS for clients; never send creds in cleartext.
- STARTTLS without enforcement is strippable → deploy MTA-STS or DANE.
- Publish SPF + DKIM + DMARC; start DMARC at p=none, read reports, then tighten to reject.
- Open relays (accepting RCPT for any domain unauthenticated) get abused instantly — don't.
- Envelope From ≠ header From: spoofing exploits this; DMARC alignment is the fix.
- Spam/phishing defenses also include greylisting, rate limits, and RBLs (DNS blocklists).
```

## Related

- [DNS](dns.md) — MX, SPF/DKIM/DMARC TXT, and DANE TLSA records all live here
- [TLS/SSL](tls_ssl.md) — STARTTLS and implicit-TLS ports secure all three protocols
- [TCP](tcp.md) — transport for SMTP/IMAP/POP3
- [Firewalls](firewalls.md) — why port 25 is commonly blocked outbound
