# SSH (Secure Shell)

## Overview

SSH is a cryptographic network protocol for secure remote login, command execution, and arbitrary tunneled traffic over an untrusted network. It replaced the insecure `telnet`, `rlogin`, and `rsh` protocols in the late 1990s and is now the universal tool for server administration. SSH runs over [TCP](tcp.md) port 22 by default. [TLS/SSL](tls_ssl.md) serves a similar authentication/encryption role for web traffic; [firewalls](firewalls.md) typically allow port 22 as a controlled access point.

## What SSH Provides

```
1. Confidentiality
   All traffic encrypted (AES, ChaCha20)

2. Integrity
   MAC ensures no tampering (HMAC, AEAD)

3. Authentication
   - Server auth via host key
   - Client auth via password, public key, certificate, MFA

4. Multiplexing
   One TCP connection carries multiple channels:
   - Interactive shell
   - Exec (single command)
   - Port forwards (local, remote, dynamic)
   - X11 forwarding
   - SFTP / SCP
```

## SSH Protocol Layers (RFC 4251)

```
┌─────────────────────────────────────────────┐
│  Connection Protocol (RFC 4254)             │  Channels (shell, exec, forward...)
├─────────────────────────────────────────────┤
│  User Authentication Protocol (RFC 4252)    │  pubkey, password, keyboard-interactive
├─────────────────────────────────────────────┤
│  Transport Protocol (RFC 4253)              │  KEX, encryption, server auth
├─────────────────────────────────────────────┤
│  TCP                                        │
└─────────────────────────────────────────────┘
```

## SSH Handshake

### 1. TCP Connection

```
Client → Server: TCP SYN to port 22
```

### 2. Version Exchange

Both sides send a banner:

```
Server → Client: SSH-2.0-OpenSSH_9.6
Client → Server: SSH-2.0-OpenSSH_9.6
```

### 3. Key Exchange (KEX)

```
Both sides:
  - Exchange supported algorithms (KEX_INIT)
  - Run a Diffie-Hellman variant
    → derives a shared session secret K
  - Server signs the exchange hash H with its host key
  - Client verifies the signature using cached host key
```

Modern KEX algorithms (in OpenSSH default order):
- `curve25519-sha256` (default, X25519 ECDH)
- `sntrup761x25519-sha512` (post-quantum hybrid)
- `diffie-hellman-group16-sha512`

### 4. Derive Symmetric Keys

```
From K and H, derive:
  - Encryption key (client→server)
  - Encryption key (server→client)
  - MAC key (each direction)
  - IV (each direction)
```

### 5. Host Key Verification

```
Client computes server host key fingerprint
Compare to ~/.ssh/known_hosts

If unknown:
  "The authenticity of host 'example.com' can't be established.
   ED25519 key fingerprint is SHA256:abc123...
   Are you sure you want to continue connecting (yes/no)?"

If different from cached → ALERT (man-in-the-middle?)
  "REMOTE HOST IDENTIFICATION HAS CHANGED!"
```

### 6. User Authentication

```
Client requests userauth service
Server lists allowed methods (e.g., "publickey,password")
Client tries methods in order until one succeeds
```

### 7. Channel Setup

```
Client opens a session channel
Requests shell, exec, sftp subsystem, or port forward
```

## Authentication Methods

### Password

```
Simple but vulnerable to brute force.
Send over established encrypted channel (not in plain).

Server checks against /etc/shadow or PAM.

Discouraged for internet-facing servers.
```

### Public Key (preferred)

```
1. Client generates a keypair locally:
     ssh-keygen -t ed25519 -C "alice@laptop"
     → ~/.ssh/id_ed25519       (private, keep secret!)
     → ~/.ssh/id_ed25519.pub   (public, share)

2. Public key copied to server:
     ssh-copy-id user@server
     → appends to ~/.ssh/authorized_keys on server

3. On login:
     Client signs a challenge with private key.
     Server verifies signature using public key in authorized_keys.
     No password ever transmitted.
```

### Key Algorithms

| Algorithm | Recommended | Notes |
|-----------|-------------|-------|
| **Ed25519** | ✓ (default) | Fast, small (256-bit), strong |
| **ECDSA** | ok | NIST curves, complex |
| **RSA-4096** | ok | Universal compatibility |
| **RSA-2048** | weak | Avoid for new keys |
| **DSA** | deprecated | Removed in OpenSSH 7+ |
| **RSA-1024** | broken | Never use |

### Certificates (better than raw keys at scale)

```
A CA signs short-lived user certificates.
Server trusts CA, not individual users.

Advantages:
  - No per-user authorized_keys management
  - Easy revocation via short TTL
  - Centralized identity (LDAP, IdP integration)

Tools: HashiCorp Vault, Smallstep, Teleport, BLESS
```

```bash
# Generate CA
ssh-keygen -t ed25519 -f user_ca

# Sign user key (validity 1h)
ssh-keygen -s user_ca -I alice@2026-05-26 -n alice -V +1h ~/.ssh/id_ed25519.pub
# → produces ~/.ssh/id_ed25519-cert.pub

# Server sshd_config
TrustedUserCAKeys /etc/ssh/user_ca.pub
```

### Keyboard-Interactive

Used for MFA (TOTP, push notification, hardware token).

```
PAM modules:
  pam_google_authenticator (TOTP)
  pam_duo (Duo Push)
  pam_u2f (YubiKey)
```

### Hardware-backed Keys

```bash
# Generate key on YubiKey (FIDO2 / SK)
ssh-keygen -t ed25519-sk -O resident -O verify-required
# Key requires physical touch to use; cannot be exfiltrated
```

## Authorized Keys

`~/.ssh/authorized_keys` on the server lists allowed public keys.

```
# Simple
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... alice@laptop

# With restrictions
command="/usr/local/bin/backup.sh",no-port-forwarding,no-X11-forwarding,no-pty,from="10.0.0.0/8" ssh-ed25519 AAA... backup@cronjob

# Common options:
#   command="..."        force a specific command (good for cron-style restricted access)
#   from="cidr,host"     limit source IPs
#   no-port-forwarding   block tunnel use
#   no-pty               no terminal
#   no-agent-forwarding  prevent agent passthrough
#   expiry-time="YYYYMMDDHHMM"  auto-expire key
```

## SSH Agent

The agent holds decrypted private keys in memory so you don't enter the passphrase repeatedly.

```bash
# Start agent (most systems start automatically)
eval "$(ssh-agent -s)"

# Add a key (prompts for passphrase once)
ssh-add ~/.ssh/id_ed25519

# List loaded keys
ssh-add -l

# Remove all
ssh-add -D

# Use a specific TTL (key auto-removed after time)
ssh-add -t 3600 ~/.ssh/id_ed25519
```

### Agent Forwarding (use with caution)

```bash
ssh -A user@bastion        # forward local agent into remote session
# Now `ssh otherhost` from bastion can use your local keys

# Risk: root on bastion can hijack your agent socket
# Better alternative: ProxyJump (-J)
ssh -J bastion user@target
```

## SSH Config (`~/.ssh/config`)

Per-host shortcuts and settings.

```
Host github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host bastion
    HostName bastion.example.com
    User alice
    Port 2222

Host prod-*
    User alice
    ProxyJump bastion
    ServerAliveInterval 60
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m

Host *
    HashKnownHosts yes
    AddKeysToAgent yes
    UseKeychain yes              # macOS only
```

### Connection Multiplexing

```
ControlMaster + ControlPersist:
  - First connection establishes a master socket
  - Subsequent `ssh prod-1` reuse the same TCP/auth
  - Massive speedup for repeated commands and scp
```

## Port Forwarding

### Local Forward (-L)

```
ssh -L LOCAL_PORT:remote_host:remote_port user@gateway

Example: access a database behind a bastion
  ssh -L 5432:db.internal:5432 alice@bastion
  # then locally: psql -h localhost -p 5432

Traffic flow:
  localhost:5432 → SSH tunnel → bastion → db.internal:5432
```

### Remote Forward (-R)

```
ssh -R REMOTE_PORT:local_host:local_port user@public

Example: expose local dev server temporarily
  ssh -R 8080:localhost:3000 alice@public-host
  # public-host:8080 now reaches your local :3000

Use case: webhook testing, NAT traversal
```

### Dynamic Forward (-D) — SOCKS Proxy

```
ssh -D 1080 user@gateway

Now point apps at SOCKS5 localhost:1080
All connections tunnel through gateway → arbitrary destinations
Effectively a poor man's VPN
```

### Gateway Ports

By default, forwarded ports listen on 127.0.0.1 only. Use `-g` or `GatewayPorts yes` to expose to other machines (cautiously).

## File Transfer

### scp (legacy)

```bash
scp file.txt user@host:/path/
scp -r dir/ user@host:/path/
scp user@host:/remote/file .
```

OpenSSH 9+ deprecates the scp protocol; use sftp under the hood:
```bash
scp -O ...   # force legacy
# default in modern versions uses SFTP transport
```

### sftp

```bash
sftp user@host
sftp> ls
sftp> get remote.txt
sftp> put local.txt
sftp> bye
```

### rsync over SSH (the workhorse)

```bash
rsync -avz --progress src/ user@host:/dest/
rsync -avz -e "ssh -p 2222" src/ user@host:/dest/
```

## SSH Tunneling Tricks

### ProxyJump (modern bastion)

```
# Single hop
ssh -J bastion target

# Multi-hop
ssh -J jump1,jump2 target

# In ~/.ssh/config
Host target
    ProxyJump bastion
```

Replaces older `ProxyCommand ssh -W %h:%p bastion`.

### Reverse SSH for Remote Access

```
On NATed home machine:
  ssh -fNR 2222:localhost:22 user@public-server

From anywhere:
  ssh -p 2222 home-user@public-server
  → reaches the home machine
```

### X11 Forwarding

```bash
ssh -X user@host       # untrusted X11
ssh -Y user@host       # trusted X11 (faster, less secure)
xeyes                  # opens locally despite running remotely
```

Often replaced by VNC/RDP in modern usage.

## Server (sshd) Configuration

`/etc/ssh/sshd_config` — apply with `systemctl reload sshd`.

```
Port 22
ListenAddress 0.0.0.0
ListenAddress ::

PermitRootLogin prohibit-password    # only root with key
PasswordAuthentication no            # keys only
PubkeyAuthentication yes
ChallengeResponseAuthentication no
KbdInteractiveAuthentication no
UsePAM yes

MaxAuthTries 3
MaxSessions 10
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2

AllowUsers alice bob
AllowGroups sshusers
DenyUsers root

# Restrict ciphers/KEX (modern defaults are good; only restrict if compliance)
KexAlgorithms curve25519-sha256,sntrup761x25519-sha512
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com

# Disable agent / tunnel forwarding if not needed
AllowAgentForwarding no
AllowTcpForwarding no
PermitTunnel no

# Match blocks for per-user/group rules
Match Group backups
    ForceCommand /usr/local/bin/restricted-backup.sh
    PermitTTY no
```

### Verify Configuration

```bash
sudo sshd -t                  # syntax check
sudo sshd -T                  # show effective config
```

## SSH Hardening Checklist

```
✓ Disable password auth (PasswordAuthentication no)
✓ Disable root login (PermitRootLogin no or prohibit-password)
✓ Use Ed25519 keys; rotate annually
✓ Use ssh-agent with timeout, not unencrypted keys on disk
✓ Move from port 22 → reduces log noise (not real security)
✓ fail2ban or sshguard against brute force
✓ Enable 2FA for human logins (pam_google_authenticator)
✓ Use ProxyJump bastions, not direct internet exposure
✓ Audit ~/.ssh/authorized_keys regularly
✓ Use SSH certificates with short TTL at scale
✓ Disable SSHv1 (long dead; verify with `nmap --script ssh2-enum-algos`)
✓ Limit cipher/KEX algorithms per compliance needs
✓ Centralized logging (auditd, journald → SIEM)
```

## Common Commands

```bash
# Connect
ssh user@host
ssh -p 2222 user@host
ssh -i ~/.ssh/special_key user@host
ssh -v user@host                    # verbose (auth debugging)
ssh -vvv user@host                  # super verbose

# Key management
ssh-keygen -t ed25519 -C "comment"
ssh-keygen -p -f ~/.ssh/id_ed25519  # change passphrase
ssh-keygen -l -f ~/.ssh/id_ed25519  # show fingerprint
ssh-keygen -y -f ~/.ssh/id_ed25519  # extract public from private
ssh-keygen -R hostname              # remove from known_hosts
ssh-keyscan host >> ~/.ssh/known_hosts

# Test connectivity
ssh -T git@github.com               # test without shell
nc -zv host 22                      # plain TCP check

# Run command, no shell
ssh user@host 'uptime'
ssh user@host -- ls -la /var/log

# Copy a public key
ssh-copy-id user@host
ssh-copy-id -i ~/.ssh/special.pub user@host
```

## SSH Agent Socket

```bash
echo $SSH_AUTH_SOCK
# /tmp/ssh-XXXX/agent.PID

# Manually use a forwarded agent
SSH_AUTH_SOCK=/tmp/ssh-foo/agent.123 ssh-add -l
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Permission denied (publickey) | Key not in authorized_keys / wrong perms | Check `~/.ssh/authorized_keys` and `~/.ssh` perms (700/600) |
| Connection refused | sshd not listening / firewall | `systemctl status sshd`, check iptables |
| Connection timed out | Network unreachable / firewall drop | tcpdump, traceroute |
| Host key verification failed | Server reinstalled or MITM | `ssh-keygen -R host` (verify legitimate first!) |
| Too many authentication failures | Agent tried too many keys | Use `IdentityFile` + `IdentitiesOnly yes` |
| Could not chdir to home directory | Home filesystem unmounted | Check `/etc/passwd` and mount status |

## File Permissions Matter

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 644 ~/.ssh/known_hosts
chmod 600 ~/.ssh/config
```

sshd refuses to use keys with loose permissions.

## SSH vs Related Protocols

| Protocol | Purpose | Encryption |
|----------|---------|------------|
| **SSH** | Remote shell / tunnel | Yes (KEX + symmetric) |
| **Telnet** | Plain remote shell | None (deprecated) |
| **SFTP** | File transfer (subsystem of SSH) | Via SSH |
| **SCP** | File copy (legacy SSH command) | Via SSH |
| **FTPS** | FTP over TLS | TLS |
| **rsync** | File sync (often over SSH) | Via SSH |
| **RDP** | Windows remote desktop | TLS |
| **VNC** | Cross-platform desktop | Optional (usually wrapped in SSH) |
| **mosh** | Roaming-friendly shell | SSH for setup, UDP for traffic |

## SSH-related Tools

```
sshpass         non-interactive password (avoid; use keys)
parallel-ssh    fan out commands to many hosts
ansible         config management built on SSH
mosh            roaming SSH alternative
autossh         auto-reconnect tunnels
sslh            multiplex SSH + HTTPS on port 443
chisel          HTTP-tunneled SSH-like (firewall bypass)
teleport        modern SSH access with audit and certs
```

## ELI10

SSH is like a sealed armored truck for talking to a faraway computer.

The truck has three guards: one checks the **destination address** (host key — "yep, this is really your friend's house, not a fake one"), one checks **your ID** (your public key matches one on the door's allowed list), and the third keeps the contents **secret** while in transit.

**Port forwarding** is like asking the truck driver to also pick up a package along the way: "while you're going to my friend's, swing by and grab their printer's signal so I can use it from home."

**Agent forwarding** is handing the truck driver your house keys so they can let you into other places too — convenient, but if the truck driver is a bad guy, your keys are exposed. Better to teach the next truck driver to make their own keys (certificates).

## Further Resources

- [RFC 4251 - SSH Protocol Architecture](https://tools.ietf.org/html/rfc4251)
- [RFC 4253 - SSH Transport Layer](https://tools.ietf.org/html/rfc4253)
- [OpenSSH Manual](https://man.openbsd.org/ssh)
## Where this connects

- [TCP](tcp.md) — SSH runs over TCP (port 22) for reliable ordered delivery
- [TLS/SSL](tls_ssl.md) — both serve authentication and encryption; TLS is for HTTP, SSH for shell/SFTP
- [Firewalls](firewalls.md) — SSH port 22 is a key firewall rule; jump hosts reduce attack surface
- [IPsec](ipsec.md) — alternative tunnel approach for site-to-site connectivity

- [SSH Mastery (book) by Michael W. Lucas](https://mwl.io/nonfiction/networking#ssh)
- [Mozilla SSH guidelines](https://infosec.mozilla.org/guidelines/openssh.html)
