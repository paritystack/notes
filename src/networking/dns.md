# DNS (Domain Name System)

## Overview

DNS is the internet's phonebook that translates human-readable domain names (like `example.com`) into IP addresses (like `93.184.216.34`) that computers use to identify each other on the network.

## DNS Hierarchy

```
                    Root (.)
                       |
         +-------------+-------------+
         |             |             |
       .com          .org          .net
         |             |             |
    example.com   wikipedia.org  archive.net
         |
    www.example.com
```

## DNS Record Types

| Record Type | Purpose | Example |
|-------------|---------|---------|
| **A** | IPv4 address | `example.com -> 93.184.216.34` |
| **AAAA** | IPv6 address | `example.com -> 2606:2800:220:1:...` |
| **CNAME** | Canonical name (alias) | `www.example.com -> example.com` |
| **MX** | Mail exchange server | `example.com -> mail.example.com` |
| **NS** | Name server | `example.com -> ns1.example.com` |
| **TXT** | Text information | SPF, DKIM records |
| **PTR** | Reverse DNS lookup | `34.216.184.93 -> example.com` |
| **SOA** | Start of authority | Zone information |
| **SRV** | Service location | `_service._proto.name` |

## DNS Query Process

```
1. User types "example.com" in browser

2. Browser checks local cache

3. If not cached, query DNS resolver (ISP or 8.8.8.8)

4. Resolver checks its cache

5. If not cached, recursive query:

   Resolver → Root DNS Server
   Root → "Ask .com server"

   Resolver → .com TLD Server
   TLD → "Ask example.com's nameserver"

   Resolver → example.com's Nameserver
   Nameserver → "IP is 93.184.216.34"

6. Resolver caches result and returns to browser

7. Browser connects to IP address
```

## DNS Message Format

```
+---------------------------+
|        Header             |  12 bytes
+---------------------------+
|        Question           |  Variable
+---------------------------+
|        Answer             |  Variable
+---------------------------+
|        Authority          |  Variable
+---------------------------+
|        Additional         |  Variable
+---------------------------+
```

### Header Format (12 bytes)

```
 0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                      ID                       |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|QR|   Opcode  |AA|TC|RD|RA|   Z    |   RCODE   |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    QDCOUNT                    |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    ANCOUNT                    |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    NSCOUNT                    |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    ARCOUNT                    |
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
```

**Fields:**
- **ID**: 16-bit identifier for matching requests/responses
- **QR**: Query (0) or Response (1)
- **Opcode**: Query type (0=standard, 1=inverse, 2=status)
- **AA**: Authoritative Answer
- **TC**: Truncated (message too long for UDP)
- **RD**: Recursion Desired
- **RA**: Recursion Available
- **RCODE**: Response code (0=no error, 3=name error)
- **QDCOUNT**: Number of questions
- **ANCOUNT**: Number of answers
- **NSCOUNT**: Number of authority records
- **ARCOUNT**: Number of additional records

## DNS Query Example

### Query (Request)

```
; DNS Query for example.com A record
; Header
ID: 0x1234
Flags: 0x0100 (standard query, recursion desired)
Questions: 1
Answer RRs: 0
Authority RRs: 0
Additional RRs: 0

; Question Section
example.com.    IN    A
```

**Hexadecimal representation:**
```
12 34  01 00  00 01  00 00  00 00  00 00
07 65 78 61 6d 70 6c 65 03 63 6f 6d 00
00 01  00 01
```

### Response

```
; DNS Response for example.com A record
; Header
ID: 0x1234
Flags: 0x8180 (response, recursion available)
Questions: 1
Answer RRs: 1
Authority RRs: 0
Additional RRs: 0

; Question Section
example.com.    IN    A

; Answer Section
example.com.    86400    IN    A    93.184.216.34
```

## DNS Query Types

### Recursive Query
Client asks DNS server to provide the final answer:
```
Client → Resolver: "What's example.com?"
Resolver → Root/TLD/Auth servers (multiple queries)
Resolver → Client: "It's 93.184.216.34"
```

### Iterative Query
DNS server returns best answer it knows:
```
Client → Root: "What's example.com?"
Root → Client: "Ask .com server at 192.5.6.30"

Client → TLD: "What's example.com?"
TLD → Client: "Ask ns1.example.com at 192.0.2.1"

Client → Auth: "What's example.com?"
Auth → Client: "It's 93.184.216.34"
```

## DNS Resource Record Format

```
Name: example.com
Type: A (1)
Class: IN (1) - Internet
TTL: 86400 (24 hours)
Data Length: 4
Data: 93.184.216.34
```

## Common DNS Operations

### Using dig (DNS lookup tool)

```bash
# Basic A record lookup
dig example.com

# Query specific record type
dig example.com MX
dig example.com AAAA

# Query specific DNS server
dig @8.8.8.8 example.com

# Reverse DNS lookup
dig -x 93.184.216.34

# Trace DNS resolution path
dig +trace example.com

# Short answer only
dig +short example.com
```

### Using nslookup

```bash
# Basic lookup
nslookup example.com

# Query specific server
nslookup example.com 8.8.8.8

# Query specific record type
nslookup -type=MX example.com
```

### Using host

```bash
# Simple lookup
host example.com

# Verbose output
host -v example.com

# Query MX records
host -t MX example.com
```

## DNS Caching

### Cache Levels

1. **Browser Cache**: Short-lived (seconds to minutes)
2. **OS Cache**: System-level DNS cache
3. **Router Cache**: Local network cache
4. **ISP Resolver Cache**: Hours to days
5. **Authoritative Server**: The source of truth

### TTL (Time To Live)

Controls how long records are cached:

```
example.com.  3600  IN  A  93.184.216.34
              ^^^^
              1 hour TTL
```

### Flushing DNS Cache

```bash
# Windows
ipconfig /flushdns

# macOS
sudo dscacheutil -flushcache

# Linux (systemd-resolved)
sudo systemd-resolve --flush-caches

# Linux (nscd)
sudo /etc/init.d/nscd restart
```

## DNS Security

### DNS Spoofing/Cache Poisoning

Attack where fake DNS responses are injected:

```
Attacker intercepts DNS query
Attacker sends fake response: "bank.com -> evil.com"
Victim connects to attacker's server
```

**Prevention**: DNSSEC

### DNSSEC (DNS Security Extensions)

Adds cryptographic signatures to DNS records:

```
1. Zone owner signs DNS records with private key
2. Public key published in DNS
3. Resolver verifies signature
4. Chain of trust from root to domain
```

**Record Types:**
- **RRSIG**: Contains signature
- **DNSKEY**: Public key
- **DS**: Delegation Signer (links parent to child)

### DNS over HTTPS (DoH)

Encrypts DNS queries using HTTPS:

```
Client → DoH Server (port 443)
Encrypted: "What's example.com?"
Encrypted: "It's 93.184.216.34"
```

**Providers:**
- Cloudflare: `https://1.1.1.1/dns-query`
- Google: `https://dns.google/dns-query`

### DNS over TLS (DoT)

Encrypts DNS queries using TLS:

```
Client → DoT Server (port 853)
TLS encrypted DNS query/response
```

## Public DNS Servers

| Provider | IPv4 | IPv6 | Features |
|----------|------|------|----------|
| **Google** | 8.8.8.8, 8.8.4.4 | 2001:4860:4860::8888 | Fast, reliable |
| **Cloudflare** | 1.1.1.1, 1.0.0.1 | 2606:4700:4700::1111 | Privacy-focused |
| **Quad9** | 9.9.9.9 | 2620:fe::fe | Malware blocking |
| **OpenDNS** | 208.67.222.222 | 2620:119:35::35 | Content filtering |

## DNS Load Balancing

Multiple A records for load distribution:

```
example.com.  300  IN  A  192.0.2.1
example.com.  300  IN  A  192.0.2.2
example.com.  300  IN  A  192.0.2.3
```

Round-robin or geographic distribution of requests.

## Common DNS Response Codes

| Code | Name | Meaning |
|------|------|---------|
| **0** | NOERROR | Query successful |
| **1** | FORMERR | Format error |
| **2** | SERVFAIL | Server failure |
| **3** | NXDOMAIN | Domain doesn't exist |
| **4** | NOTIMP | Not implemented |
| **5** | REFUSED | Query refused |

## DNS Best Practices

### 1. Use Multiple Nameservers
```
NS  ns1.example.com  (Primary)
NS  ns2.example.com  (Secondary)
```

### 2. Appropriate TTL Values
```
# Stable records (rarely change)
example.com.  86400  IN  A  93.184.216.34

# Dynamic records (may change soon)
staging.example.com.  300  IN  A  192.0.2.1
```

### 3. SPF Records for Email
```
example.com.  IN  TXT  "v=spf1 mx include:_spf.google.com ~all"
```

### 4. DKIM for Email Authentication
```
default._domainkey.example.com.  IN  TXT  "v=DKIM1; k=rsa; p=MIGfMA0..."
```

## DNS Troubleshooting

### Issue: Domain not resolving

```bash
# Check if domain exists
dig example.com

# Check all nameservers
dig example.com NS
dig @ns1.example.com example.com

# Check propagation
dig @8.8.8.8 example.com
dig @1.1.1.1 example.com
```

### Issue: Slow DNS resolution

```bash
# Test query time
dig example.com | grep "Query time"

# Compare different DNS servers
dig @8.8.8.8 example.com | grep "Query time"
dig @1.1.1.1 example.com | grep "Query time"
```

### Issue: NXDOMAIN (domain not found)

1. Check domain registration
2. Verify nameserver configuration
3. Check DNS propagation time (up to 48 hours)

## Zone File Example

```
$TTL 86400
@   IN  SOA  ns1.example.com. admin.example.com. (
            2024011301  ; Serial
            3600        ; Refresh
            1800        ; Retry
            604800      ; Expire
            86400 )     ; Minimum TTL

; Name servers
    IN  NS   ns1.example.com.
    IN  NS   ns2.example.com.

; Mail servers
    IN  MX   10 mail1.example.com.
    IN  MX   20 mail2.example.com.

; A records
@           IN  A    93.184.216.34
www         IN  A    93.184.216.34
mail1       IN  A    192.0.2.1
mail2       IN  A    192.0.2.2
ns1         IN  A    192.0.2.10
ns2         IN  A    192.0.2.11

; AAAA records (IPv6)
@           IN  AAAA 2606:2800:220:1:248:1893:25c8:1946

; CNAME records
ftp         IN  CNAME www.example.com.
webmail     IN  CNAME mail1.example.com.
```

## ELI10

DNS is like a phone book for the internet:

- **Without DNS**: "Visit 93.184.216.34" (hard to remember!)
- **With DNS**: "Visit example.com" (easy!)

When you type a website name:
1. Your computer asks "Where is example.com?"
2. DNS looks it up in its huge phone book
3. DNS says "It's at 93.184.216.34"
4. Your computer connects to that address

DNS servers are like helpers who:
- Remember answers (caching) so they can answer faster next time
- Ask other DNS servers if they don't know the answer
- Make sure everyone gets the same answer for the same website

## Further Resources

- [RFC 1035 - DNS Specification](https://tools.ietf.org/html/rfc1035)
- [How DNS Works (Comic)](https://howdns.works/)
- [DNS Lookup Tool](https://mxtoolbox.com/DNSLookup.aspx)
- [DNSSEC Explained](https://dnssec-analyzer.verisignlabs.com/)
