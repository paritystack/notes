# X.509 Certificates and PKI

## Overview

X.509 certificates are digital documents that bind public keys to identities. They enable:
- **Authentication**: Verify identity of servers/users
- **Encryption**: Establish secure connections
- **Trust**: Chain of trust through Certificate Authorities

## X.509 Certificate Structure

### Basic Components

```
Certificate:
  ├── Version (v3)
  ├── Serial Number (unique identifier)
  ├── Signature Algorithm (SHA-256 with RSA)
  ├── Issuer (who issued the certificate)
  ├── Validity Period
  │   ├── Not Before (start date)
  │   └── Not After (expiration date)
  ├── Subject (who the certificate is for)
  ├── Subject Public Key Info
  │   ├── Algorithm (RSA, ECDSA, etc.)
  │   └── Public Key (actual key data)
  ├── Extensions (v3)
  │   ├── Key Usage
  │   ├── Subject Alternative Names (SANs)
  │   ├── Basic Constraints
  │   └── Authority Key Identifier
  └── Signature (CA's signature)
```

### Certificate Fields

```
Subject: CN=example.com, O=Example Inc, C=US
  CN = Common Name (domain or person name)
  O  = Organization
  OU = Organizational Unit
  C  = Country
  ST = State/Province
  L  = Locality/City

Issuer: CN=Let's Encrypt Authority, O=Let's Encrypt, C=US
  (Who signed this certificate)

Validity:
  Not Before: Jan 1 00:00:00 2024 GMT
  Not After:  Apr 1 23:59:59 2024 GMT
  (Certificate valid period)

Public Key Algorithm: RSA 2048-bit
  (Type and size of public key)

Signature Algorithm: SHA-256 with RSA
  (How CA signed the certificate)
```

### Visual Representation

```
┌─────────────────────────────────────┐
│      X.509 Certificate              │
├─────────────────────────────────────┤
│ Version: 3                          │
│ Serial: 04:92:7f:63:ab:02:1e...     │
│                                     │
│ Issuer: CN=Let's Encrypt           │
│ Subject: CN=example.com             │
│                                     │
│ Valid: 2024-01-01 to 2024-04-01    │
│                                     │
│ Public Key: [RSA 2048-bit]         │
│   65537                             │
│   00:b8:7f:4e:91...                │
│                                     │
│ Extensions:                         │
│   - Key Usage: Digital Signature   │
│   - SANs: example.com, *.example.com│
│   - Basic Constraints: CA:FALSE    │
│                                     │
│ Signature Algorithm: sha256RSA     │
│ Signature: [CA's signature]        │
│   3a:7b:8c:9d...                   │
└─────────────────────────────────────┘
```

## Certificate Creation

### Creating a Self-Signed Certificate

#### OpenSSL (Bash)

```bash
# Generate private key and self-signed certificate in one command
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/C=US/ST=California/L=San Francisco/O=Example Inc/CN=example.com"

# Breakdown:
# -x509: Create self-signed certificate
# -newkey rsa:2048: Generate new 2048-bit RSA key
# -keyout: Output private key file
# -out: Output certificate file
# -days: Certificate validity period
# -nodes: Don't encrypt private key
# -subj: Certificate subject information

# View certificate details
openssl x509 -in cert.pem -text -noout

# Generate key and certificate separately
openssl genrsa -out key.pem 2048
openssl req -new -x509 -key key.pem -out cert.pem -days 365 \
  -subj "/CN=example.com"
```

#### Python

```python
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Create subject and issuer (same for self-signed)
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Example Inc"),
    x509.NameAttribute(NameOID.COMMON_NAME, "example.com"),
])

# Build certificate
cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    private_key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    datetime.datetime.utcnow() + datetime.timedelta(days=365)
).add_extension(
    x509.SubjectAlternativeName([
        x509.DNSName("example.com"),
        x509.DNSName("www.example.com"),
    ]),
    critical=False,
).sign(private_key, hashes.SHA256())

# Save certificate
with open("cert.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

# Save private key
with open("key.pem", "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ))

print("Certificate created successfully!")
```

### Creating a Certificate Signing Request (CSR)

#### OpenSSL

```bash
# Generate private key
openssl genrsa -out server.key 2048

# Create CSR
openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=CA/L=San Francisco/O=Example Inc/CN=example.com"

# View CSR
openssl req -in server.csr -text -noout

# Create CSR with Subject Alternative Names (using config file)
cat > san.cnf <<-END
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
C=US
ST=CA
L=San Francisco
O=Example Inc
CN=example.com

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = example.com
DNS.2 = www.example.com
DNS.3 = *.example.com
END

openssl req -new -key server.key -out server.csr -config san.cnf

# Verify CSR
openssl req -in server.csr -noout -verify
```

#### Python

```python
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Build CSR
csr = x509.CertificateSigningRequestBuilder().subject_name(x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Example Inc"),
    x509.NameAttribute(NameOID.COMMON_NAME, "example.com"),
])).add_extension(
    x509.SubjectAlternativeName([
        x509.DNSName("example.com"),
        x509.DNSName("www.example.com"),
        x509.DNSName("*.example.com"),
    ]),
    critical=False,
).sign(private_key, hashes.SHA256())

# Save CSR
with open("server.csr", "wb") as f:
    f.write(csr.public_bytes(serialization.Encoding.PEM))

print("CSR created successfully!")
```

## Certificate Authorities (CAs)

### CA Hierarchy

```
┌────────────────────────────┐
│      Root CA               │
│   (Self-signed)            │
│   Trust Anchor             │
└─────────────┬──────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
┌───▼──────────┐  ┌────▼───────────┐
│ Intermediate │  │ Intermediate   │
│ CA #1        │  │ CA #2          │
└───┬──────────┘  └────┬───────────┘
    │                  │
┌───▼──────┐     ┌────▼──────┐
│ End-User │     │ End-User  │
│ Cert #1  │     │ Cert #2   │
└──────────┘     └───────────┘
```

### Trust Chain

```
End-user certificate (example.com)
  ↓ Issued by
Intermediate CA certificate
  ↓ Issued by
Root CA certificate (in browser trust store)
  ✓ Trusted
```

### Setting Up a CA

#### Create Root CA

```bash
# Generate Root CA private key
openssl genrsa -aes256 -out rootCA.key 4096

# Create Root CA certificate
openssl req -x509 -new -nodes -key rootCA.key -sha256 -days 3650 \
  -out rootCA.crt \
  -subj "/C=US/ST=CA/O=Example Inc/CN=Example Root CA"

# View Root CA certificate
openssl x509 -in rootCA.crt -text -noout
```

#### Sign Certificate with CA

```bash
# You have: server.csr (from earlier)
# You have: rootCA.key and rootCA.crt

# Create extensions configuration
echo "
[ v3_req ]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = example.com
DNS.2 = www.example.com
DNS.3 = *.example.com
" > server_ext.cnf

# Sign CSR with CA
openssl x509 -req -in server.csr \
  -CA rootCA.crt -CAkey rootCA.key -CAcreateserial \
  -out server.crt -days 365 -sha256 \
  -extfile server_ext.cnf -extensions v3_req

# View signed certificate
openssl x509 -in server.crt -text -noout

# Verify certificate against CA
openssl verify -CAfile rootCA.crt server.crt
```

#### Python CA Implementation

```python
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime

class CertificateAuthority:
    def __init__(self):
        # Generate CA private key
        self.ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )

        # Create CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Example Inc"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Example Root CA"),
        ])

        self.ca_cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self.ca_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=3650)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).sign(self.ca_key, hashes.SHA256())

    def issue_certificate(self, csr, validity_days=365):
        """Issue a certificate from a CSR"""
        cert = x509.CertificateBuilder().subject_name(
            csr.subject
        ).issuer_name(
            self.ca_cert.subject
        ).public_key(
            csr.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=validity_days)
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )

        # Copy extensions from CSR
        for extension in csr.extensions:
            cert = cert.add_extension(extension.value, extension.critical)

        # Sign with CA key
        return cert.sign(self.ca_key, hashes.SHA256())

    def save_ca_cert(self, filename):
        with open(filename, "wb") as f:
            f.write(self.ca_cert.public_bytes(serialization.Encoding.PEM))

    def save_ca_key(self, filename, password=None):
        encryption = serialization.NoEncryption()
        if password:
            encryption = serialization.BestAvailableEncryption(password)

        with open(filename, "wb") as f:
            f.write(self.ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption
            ))

# Usage
ca = CertificateAuthority()
ca.save_ca_cert("ca.crt")
ca.save_ca_key("ca.key", password=b"secure-password")

# Load and sign a CSR
with open("server.csr", "rb") as f:
    csr = x509.load_pem_x509_csr(f.read())

cert = ca.issue_certificate(csr, validity_days=365)

with open("server.crt", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print("Certificate issued successfully!")
```

## Certificate Chains

### Understanding Certificate Chains

```
┌─────────────────────────────────┐
│  Server Certificate             │
│  Subject: CN=example.com        │
│  Issuer: CN=Intermediate CA     │
│  [Public Key]                   │
│  [Signature by Intermediate]    │
└────────────┬────────────────────┘
             │ Verified by
┌────────────▼────────────────────┐
│  Intermediate Certificate       │
│  Subject: CN=Intermediate CA    │
│  Issuer: CN=Root CA             │
│  [Public Key]                   │
│  [Signature by Root]            │
└────────────┬────────────────────┘
             │ Verified by
┌────────────▼────────────────────┐
│  Root Certificate               │
│  Subject: CN=Root CA            │
│  Issuer: CN=Root CA (self)      │
│  [Public Key]                   │
│  [Self Signature]               │
│  ✓ In Trust Store               │
└─────────────────────────────────┘
```

### Building Certificate Chain

```bash
# Create chain file (server cert + intermediate cert)
cat server.crt intermediate.crt > fullchain.pem

# Or with root CA (not usually needed)
cat server.crt intermediate.crt rootCA.crt > fullchain.pem

# Verify chain
openssl verify -CAfile rootCA.crt -untrusted intermediate.crt server.crt

# Display certificate chain
openssl s_client -connect example.com:443 -showcerts
```

### Verifying Certificate Chain in Python

```python
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

def verify_certificate_chain(cert_chain):
    """
    Verify a certificate chain
    cert_chain: list of certificates [leaf, intermediate, ..., root]
    """
    for i in range(len(cert_chain) - 1):
        cert = cert_chain[i]
        issuer_cert = cert_chain[i + 1]

        # Verify issuer name matches
        if cert.issuer != issuer_cert.subject:
            return False, f"Issuer mismatch at level {i}"

        # Verify signature
        try:
            issuer_public_key = issuer_cert.public_key()
            issuer_public_key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm,
            )
        except InvalidSignature:
            return False, f"Invalid signature at level {i}"

        # Verify validity period
        import datetime
        now = datetime.datetime.utcnow()
        if now < cert.not_valid_before or now > cert.not_valid_after:
            return False, f"Certificate expired or not yet valid at level {i}"

    return True, "Chain verified successfully"

# Load certificates
certs = []
for cert_file in ['server.crt', 'intermediate.crt', 'root.crt']:
    with open(cert_file, 'rb') as f:
        cert = x509.load_pem_x509_certificate(f.read())
        certs.append(cert)

# Verify chain
is_valid, message = verify_certificate_chain(certs)
print(message)
```

## Let's Encrypt

### Overview

Let's Encrypt is a free, automated Certificate Authority providing:
- Free SSL/TLS certificates
- 90-day validity (encourages automation)
- Domain Validation (DV) only
- Automated renewal

### ACME Protocol

```
1. Client requests certificate for example.com

2. Let's Encrypt challenges ownership:
   - HTTP-01: Place file at http://example.com/.well-known/acme-challenge/
   - DNS-01: Add TXT record to _acme-challenge.example.com
   - TLS-ALPN-01: Configure TLS server with special certificate

3. Let's Encrypt verifies challenge

4. If successful, issues certificate

5. Client installs certificate

6. Automated renewal before 90-day expiration
```

### Using Certbot

```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate (standalone)
sudo certbot certonly --standalone -d example.com -d www.example.com

# Obtain certificate (webroot - site already running)
sudo certbot certonly --webroot -w /var/www/html -d example.com

# Obtain certificate (DNS challenge)
sudo certbot certonly --manual --preferred-challenges dns -d example.com

# Obtain certificate (with automatic nginx configuration)
sudo certbot --nginx -d example.com -d www.example.com

# Obtain certificate (with automatic apache configuration)
sudo certbot --apache -d example.com

# List certificates
sudo certbot certificates

# Renew certificates (dry run)
sudo certbot renew --dry-run

# Renew certificates
sudo certbot renew

# Revoke certificate
sudo certbot revoke --cert-path /etc/letsencrypt/live/example.com/cert.pem

# Delete certificate
sudo certbot delete --cert-name example.com
```

### Automated Renewal

```bash
# Add to crontab (check renewal twice daily)
0 0,12 * * * certbot renew --quiet

# Systemd timer (if using systemd)
sudo systemctl enable certbot-renew.timer
sudo systemctl start certbot-renew.timer

# Test renewal
sudo certbot renew --dry-run
```

### Using acme.sh (Alternative)

```bash
# Install acme.sh
curl https://get.acme.sh | sh

# Issue certificate (HTTP validation)
acme.sh --issue -d example.com -w /var/www/html

# Issue certificate (DNS validation with Cloudflare)
export CF_Key="your-cloudflare-api-key"
export CF_Email="your@email.com"
acme.sh --issue --dns dns_cf -d example.com -d *.example.com

# Install certificate
acme.sh --install-cert -d example.com \
  --key-file /etc/nginx/ssl/example.com.key \
  --fullchain-file /etc/nginx/ssl/example.com.crt \
  --reloadcmd "systemctl reload nginx"

# Renew all certificates
acme.sh --renew-all

# Force renew
acme.sh --renew -d example.com --force
```

## Certificate Management

### Certificate Inspection

```bash
# View certificate details
openssl x509 -in cert.pem -text -noout

# View certificate dates
openssl x509 -in cert.pem -noout -dates

# View certificate subject
openssl x509 -in cert.pem -noout -subject

# View certificate issuer
openssl x509 -in cert.pem -noout -issuer

# View certificate fingerprint
openssl x509 -in cert.pem -noout -fingerprint -sha256

# Check certificate and key match
openssl x509 -noout -modulus -in cert.pem | openssl md5
openssl rsa -noout -modulus -in key.pem | openssl md5
# If md5 hashes match, cert and key are paired

# View certificate from server
openssl s_client -connect example.com:443 -showcerts

# Check certificate expiration
echo | openssl s_client -connect example.com:443 2>/dev/null | \
  openssl x509 -noout -dates
```

### Python Certificate Tools

```python
from cryptography import x509
from cryptography.hazmat.primitives import serialization
import datetime

def inspect_certificate(cert_path):
    with open(cert_path, 'rb') as f:
        cert = x509.load_pem_x509_certificate(f.read())

    print("Certificate Information:")
    print(f"Subject: {cert.subject.rfc4514_string()}")
    print(f"Issuer: {cert.issuer.rfc4514_string()}")
    print(f"Serial Number: {cert.serial_number}")
    print(f"Not Valid Before: {cert.not_valid_before}")
    print(f"Not Valid After: {cert.not_valid_after}")
    print(f"Signature Algorithm: {cert.signature_algorithm_oid._name}")

    # Check if expired
    now = datetime.datetime.utcnow()
    days_until_expiry = (cert.not_valid_after - now).days

    if now > cert.not_valid_after:
        print("⚠ Certificate EXPIRED!")
    elif days_until_expiry < 30:
        print(f"⚠ Certificate expires soon ({days_until_expiry} days)")
    else:
        print(f"✓ Certificate valid ({days_until_expiry} days remaining)")

    # Subject Alternative Names
    try:
        san_ext = cert.extensions.get_extension_for_oid(
            x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        )
        print(f"SANs: {', '.join([dns.value for dns in san_ext.value])}")
    except x509.ExtensionNotFound:
        print("No SANs found")

    return cert

# Usage
cert = inspect_certificate('cert.pem')
```

### Certificate Monitoring

```python
import ssl
import socket
from datetime import datetime

def check_certificate_expiry(hostname, port=443):
    """Check SSL certificate expiration"""
    context = ssl.create_default_context()

    with socket.create_connection((hostname, port)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert = ssock.getpeercert()

            # Parse expiration date
            expires = datetime.strptime(
                cert['notAfter'],
                '%b %d %H:%M:%S %Y %GMT'
            )

            days_remaining = (expires - datetime.now()).days

            print(f"Certificate for {hostname}:")
            print(f"  Subject: {dict(x[0] for x in cert['subject'])['commonName']}")
            print(f"  Issuer: {dict(x[0] for x in cert['issuer'])['commonName']}")
            print(f"  Expires: {expires}")
            print(f"  Days remaining: {days_remaining}")

            if days_remaining < 0:
                print("  ⚠ EXPIRED!")
            elif days_remaining < 30:
                print("  ⚠ Expiring soon!")
            else:
                print("  ✓ Valid")

            return days_remaining

# Check multiple sites
sites = ['google.com', 'github.com', 'example.com']
for site in sites:
    try:
        check_certificate_expiry(site)
        print()
    except Exception as e:
        print(f"Error checking {site}: {e}\n")
```

### Certificate Renewal Strategy

```bash
#!/bin/bash
# certificate-renewal.sh

# Check certificate expiration
check_expiry() {
    local domain=$1
    local days_until_expiry=$(echo | openssl s_client -connect $domain:443 2>/dev/null | \
        openssl x509 -noout -checkend 2592000)  # 30 days

    if [ $? -eq 0 ]; then
        echo "$domain: Certificate valid for at least 30 days"
        return 0
    else
        echo "$domain: Certificate expires within 30 days!"
        return 1
    fi
}

# Renew if needed
renew_certificate() {
    local domain=$1

    if ! check_expiry $domain; then
        echo "Renewing certificate for $domain..."
        certbot renew --cert-name $domain

        if [ $? -eq 0 ]; then
            echo "Certificate renewed successfully"
            systemctl reload nginx
        else
            echo "Certificate renewal failed!"
            # Send alert
        fi
    fi
}

# Check all domains
for domain in example.com api.example.com www.example.com; do
    renew_certificate $domain
done
```

## Certificate Revocation

### Certificate Revocation Lists (CRL)

```bash
# Download CRL
wget http://crl.example.com/example.crl

# View CRL
openssl crl -in example.crl -text -noout

# Check if certificate is revoked
openssl verify -crl_check -CRLfile example.crl -CAfile ca.crt cert.pem
```

### Online Certificate Status Protocol (OCSP)

```bash
# Get OCSP responder URL from certificate
openssl x509 -in cert.pem -noout -ocsp_uri

# Check certificate status via OCSP
openssl ocsp -issuer ca.crt -cert cert.pem \
  -url http://ocsp.example.com \
  -resp_text

# OCSP stapling check
openssl s_client -connect example.com:443 -status
```

### Revoking Certificate

```bash
# Revoke with certbot
sudo certbot revoke --cert-path /etc/letsencrypt/live/example.com/cert.pem

# Revoke with reason
sudo certbot revoke --cert-path cert.pem --reason keycompromise

# Revoke with custom CA
openssl ca -config ca.conf -revoke cert.pem -keyfile ca.key -cert ca.crt

# Generate CRL
openssl ca -config ca.conf -gencrl -out crl.pem
```

## Security Considerations

### 1. Key Size

```
RSA:
  Minimum: 2048 bits
  Recommended: 3072-4096 bits

ECDSA:
  Recommended: P-256 (256-bit)
  High security: P-384 (384-bit)

Ed25519:
  Fixed: 256-bit (recommended for new deployments)
```

### 2. Certificate Validity Period

```
Modern best practices:
- Maximum: 398 days (13 months) - enforced by browsers
- Recommended: 90 days (Let's Encrypt default)
- Automated renewal: Essential for short validity

Historical:
- Before 2020: Up to 2-3 years
- 2020: 398 days maximum
- Trend: Shorter validity periods
```

### 3. Subject Alternative Names (SANs)

```bash
# Include all domain variants
subjectAltName = DNS:example.com,DNS:www.example.com,DNS:*.example.com

# Don't rely on Common Name (CN) - deprecated
# Always use SANs
```

### 4. Certificate Pinning

```python
import ssl
import hashlib
import socket

def verify_certificate_pinning(hostname, expected_fingerprints):
    """Verify certificate matches expected fingerprint"""
    context = ssl.create_default_context()

    with socket.create_connection((hostname, 443)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            cert_der = ssock.getpeercert(binary_form=True)
            fingerprint = hashlib.sha256(cert_der).hexdigest()

            if fingerprint in expected_fingerprints:
                print(f"✓ Certificate pinning verified")
                return True
            else:
                print(f"✗ Certificate pinning failed!")
                print(f"  Expected: {expected_fingerprints}")
                print(f"  Got: {fingerprint}")
                return False

# Usage
expected_pins = [
    'a1b2c3d4e5f6...',  # Primary certificate
    '9a8b7c6d5e4f...',  # Backup certificate
]

verify_certificate_pinning('example.com', expected_pins)
```

## Best Practices

### 1. Automate Certificate Management

```
✓ Use Let's Encrypt for free certificates
✓ Automate renewal (certbot, acme.sh)
✓ Monitor expiration dates
✓ Test renewal process regularly
✓ Use short validity periods (90 days)
```

### 2. Secure Private Keys

```bash
# Restrict permissions
chmod 600 private.key

# Use hardware security modules (HSM) for critical keys
# Use encrypted private keys
openssl rsa -aes256 -in private.key -out private_encrypted.key

# Never commit to version control
echo "*.key" >> .gitignore
echo "*.pem" >> .gitignore
```

### 3. Use Strong Cryptography

```
✓ RSA 2048-bit minimum (prefer 3072+)
✓ ECDSA P-256 or better
✓ SHA-256 or SHA-512 for signatures
✗ Avoid MD5, SHA-1
✗ Avoid RSA <2048 bits
```

### 4. Implement Certificate Transparency

```bash
# Check if certificate is in CT logs
curl https://crt.sh/?q=example.com

# Monitor for unauthorized certificates
# Use tools like certstream, certificate-transparency-go
```

## Common Mistakes

### 1. Expired Certificates

```
Problem: Certificate expires unexpectedly
Solution: Automate monitoring and renewal
```

### 2. Missing Intermediate Certificates

```
Problem: Browser shows untrusted certificate
Solution: Include full chain (server + intermediate certs)

# Correct chain order
cat server.crt intermediate.crt > fullchain.pem
```

### 3. Certificate Name Mismatch

```
Problem: Certificate for wrong domain
Solution: Use proper SANs

# Include all domains
subjectAltName = DNS:example.com,DNS:www.example.com
```

### 4. Insecure Private Key

```
Problem: Private key readable by all users
Solution: Restrict permissions

chmod 600 private.key
chown root:root private.key
```

## ELI10

Certificates are like ID cards for websites:

**Without certificates:**
- You visit "bank.com"
- How do you know it's really your bank?
- Attackers could pretend to be your bank!

**With certificates:**
1. **Website has ID card** (certificate)
   - Says: "I'm bank.com"
   - Has a special seal (signature)

2. **Trusted Authority** (CA like Let's Encrypt)
   - Like a government issuing passports
   - Checks: "Yes, you really own bank.com"
   - Adds their official seal

3. **Your browser checks**:
   - Is the ID card real? ✓
   - Is it expired? ✓
   - Does it match the website name? ✓
   - Is the seal from a trusted authority? ✓

4. **Chain of Trust**:
   ```
   Browser trusts → Root CA
   Root CA trusts → Intermediate CA
   Intermediate CA trusts → Website Certificate
   Therefore, Browser trusts → Website!
   ```

**Let's Encrypt made it:**
- Free (used to cost $$$)
- Automatic (renews itself)
- Easy (simple commands)

**Real-world analogy:**
- Certificate = Passport
- CA = Government passport office
- Browser = Border control checking passports
- Expiration date = Passport validity
- Renewal = Getting new passport before expiry

## Further Resources

- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [X.509 Certificate Format (RFC 5280)](https://tools.ietf.org/html/rfc5280)
- [ACME Protocol (RFC 8555)](https://tools.ietf.org/html/rfc8555)
- [Certificate Transparency](https://certificate.transparency.dev/)
- [SSL Labs Server Test](https://www.ssllabs.com/ssltest/)
- [Certbot Documentation](https://certbot.eff.org/docs/)
- [OpenSSL Cookbook](https://www.feistyduck.com/library/openssl-cookbook/)
- [Public Key Infrastructure (PKI) Guide](https://en.wikipedia.org/wiki/Public_key_infrastructure)
