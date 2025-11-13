# Firewalls

## Overview

A firewall is a network security system that monitors and controls incoming and outgoing network traffic based on predetermined security rules. It acts as a barrier between trusted internal networks and untrusted external networks (like the internet).

## Firewall Types

### 1. Packet Filtering Firewall

**How it works:**
- Inspects individual packets
- Makes decisions based on header information
- Stateless (doesn't track connections)

**Checks:**
- Source IP address
- Destination IP address
- Source port
- Destination port
- Protocol (TCP, UDP, ICMP)

**Example Rule:**
```
ALLOW TCP from 192.168.1.0/24 to any port 80
DENY TCP from any to any port 23
```

**Decision Process:**
```
Incoming packet:
  Src: 192.168.1.10:54321
  Dst: 10.0.0.5:80
  Protocol: TCP

Check rules top-to-bottom:
  Rule 1: Allow 192.168.1.0/24 to port 80 ’ MATCH
  Action: ALLOW

Packet forwarded
```

**Pros:**
- Fast (minimal inspection)
- Low resource usage
- Simple configuration

**Cons:**
- No state tracking
- Can't detect complex attacks
- Vulnerable to IP spoofing

### 2. Stateful Inspection Firewall

**How it works:**
- Tracks connection state
- Maintains state table
- Understands context of traffic

**State Table Example:**
```
Src IP        Src Port  Dst IP        Dst Port  State      Protocol
192.168.1.10  54321     93.184.216.34 80        ESTABLISHED TCP
192.168.1.11  54322     8.8.8.8       53        NEW         UDP
192.168.1.10  54323     10.0.0.5      22        SYN_SENT    TCP
```

**TCP Connection Tracking:**
```
Client ’ Server: SYN
  State: NEW

Server ’ Client: SYN-ACK
  State: ESTABLISHED

Client ’ Server: ACK
  State: ESTABLISHED

... data transfer ...

Client ’ Server: FIN
  State: CLOSING

Server ’ Client: FIN-ACK
  State: CLOSED
```

**Example Rule:**
```
# Outbound rule
ALLOW TCP from 192.168.1.0/24 to any port 80 STATE NEW,ESTABLISHED

# Return traffic automatically allowed
# (tracked in state table)
```

**Pros:**
- Understands connection context
- Better security than packet filtering
- Prevents spoofing attacks
- Allows related traffic

**Cons:**
- More resource intensive
- State table can be exhausted
- Performance impact at scale

### 3. Application Layer Firewall (Proxy Firewall)

**How it works:**
- Operates at Layer 7 (Application)
- Acts as intermediary (proxy)
- Deep packet inspection
- Understands application protocols

**Proxy Flow:**
```
Client ’ Proxy ’ Server

Client connects to proxy
Proxy inspects full request
Proxy makes decision
Proxy connects to server (if allowed)
Proxy relays response to client
```

**Inspection Capabilities:**
```
HTTP/HTTPS:
  - URL filtering
  - Content scanning
  - Malware detection
  - Data loss prevention

FTP:
  - Command filtering
  - File type restrictions

SMTP:
  - Spam filtering
  - Attachment scanning
```

**Example:**
```
HTTP Request:
  GET /admin.php HTTP/1.1
  Host: example.com

Proxy checks:
  1. Is /admin.php allowed? ’ NO
  2. Block request
  3. Return 403 Forbidden
```

**Pros:**
- Deep inspection
- Understands application protocols
- Can filter content
- Hides internal network
- Logging and auditing

**Cons:**
- Significant performance impact
- Complex configuration
- May break some applications
- Single point of failure

### 4. Next-Generation Firewall (NGFW)

**Combines:**
- Traditional firewall functions
- Intrusion Prevention System (IPS)
- Application awareness
- SSL/TLS inspection
- Advanced threat protection

**Features:**
```
1. Deep Packet Inspection (DPI)
   - Full packet content analysis

2. Application Control
   - Block Facebook but allow LinkedIn
   - Control by application, not just port

3. User Identity
   - Rules based on user/group
   - Active Directory integration

4. Threat Intelligence
   - Malware detection
   - Botnet protection
   - Zero-day protection

5. SSL Inspection
   - Decrypt HTTPS traffic
   - Inspect encrypted content
   - Re-encrypt and forward
```

**Example NGFW Rule:**
```
DENY application "BitTorrent" for group "Employees"
ALLOW application "Salesforce" for group "Sales"
BLOCK malware signature "Trojan.Generic.123"
```

## Firewall Architectures

### 1. Packet Filtering Router

```
Internet  ’ [Router with ACL]  ’ Internal Network

Simple, single layer of protection
```

### 2. Dual-Homed Host

```
Internet  ’ [Firewall with 2 NICs]  ’ Internal Network
             (All traffic through firewall)

Complete traffic control
```

### 3. Screened Host

```
Internet  ’ [Router]  ’ [Firewall Host]  ’ Internal Network

Router filters basic traffic
Firewall provides additional protection
```

### 4. Screened Subnet (DMZ)

```
Internet  ’ [External FW]  ’ [DMZ]  ’ [Internal FW]  ’ Internal Network
                               (Web, Mail)

Public services in DMZ
Internal network isolated
```

**DMZ Example:**
```
External Firewall Rules:
  - Allow HTTP/HTTPS to web server (DMZ)
  - Allow SMTP to mail server (DMZ)
  - Deny all to internal network

Internal Firewall Rules:
  - Allow web server to database (specific port)
  - Allow mail server to internal mail (specific port)
  - Deny all other DMZ traffic to internal
```

## Firewall Rules

### Rule Components

```
1. Source: Where traffic originates
2. Destination: Where traffic is going
3. Service/Port: What service (HTTP, SSH, etc.)
4. Action: Allow, Deny, Reject
5. Direction: Inbound, Outbound
6. State: NEW, ESTABLISHED, RELATED
```

### Rule Example (iptables)

```bash
# Allow SSH from specific network
iptables -A INPUT -s 192.168.1.0/24 -p tcp --dport 22 -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow HTTP and HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Drop everything else
iptables -A INPUT -j DROP
```

### Rule Ordering

**Important:** Rules processed top-to-bottom, first match wins

```bash
# WRONG ORDER:
1. DENY all
2. ALLOW HTTP port 80   Never reached!

# CORRECT ORDER:
1. ALLOW HTTP port 80
2. DENY all
```

### Default Policy

```bash
# Default DENY (whitelist approach - more secure)
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Then explicitly allow needed services

# Default ALLOW (blacklist approach - less secure)
iptables -P INPUT ACCEPT
iptables -P FORWARD ACCEPT
iptables -P OUTPUT ACCEPT

# Then explicitly block dangerous services
```

## Common Firewall Configurations

### 1. Linux iptables

**View rules:**
```bash
iptables -L -v -n
```

**Basic web server protection:**
```bash
# Flush existing rules
iptables -F

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (from specific network)
iptables -A INPUT -s 192.168.1.0/24 -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow ping
iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "DROPPED: "
iptables -A INPUT -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### 2. Linux ufw (Uncomplicated Firewall)

```bash
# Enable firewall
ufw enable

# Default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH
ufw allow 22/tcp

# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow from specific IP
ufw allow from 192.168.1.100

# Allow specific port from specific IP
ufw allow from 192.168.1.100 to any port 3306

# View rules
ufw status numbered

# Delete rule
ufw delete 5
```

### 3. Linux firewalld

```bash
# View zones
firewall-cmd --get-active-zones

# Add service to zone
firewall-cmd --zone=public --add-service=http
firewall-cmd --zone=public --add-service=https

# Add port
firewall-cmd --zone=public --add-port=8080/tcp

# Add rich rule
firewall-cmd --zone=public --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" service name="ssh" accept'

# Make permanent
firewall-cmd --runtime-to-permanent

# Reload
firewall-cmd --reload
```

### 4. Windows Firewall

```powershell
# View rules
Get-NetFirewallRule

# Enable firewall
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True

# Allow inbound port
New-NetFirewallRule -DisplayName "Allow HTTP" -Direction Inbound -LocalPort 80 -Protocol TCP -Action Allow

# Allow program
New-NetFirewallRule -DisplayName "My App" -Direction Inbound -Program "C:\App\myapp.exe" -Action Allow

# Block IP address
New-NetFirewallRule -DisplayName "Block IP" -Direction Inbound -RemoteAddress 10.0.0.5 -Action Block
```

## Port Knocking

**Concept:** Hidden service that opens after specific sequence

**Example:**
```bash
# Ports closed by default

# Client knocks sequence: 1234, 5678, 9012
nc -z server.com 1234
nc -z server.com 5678
nc -z server.com 9012

# Server detects sequence, opens SSH port 22 for client IP
# Client can now SSH

# After timeout, port closes again
```

**Configuration (knockd):**
```
[openSSH]
sequence = 1234,5678,9012
seq_timeout = 10
command = /sbin/iptables -I INPUT -s %IP% -p tcp --dport 22 -j ACCEPT
tcpflags = syn

[closeSSH]
sequence = 9012,5678,1234
seq_timeout = 10
command = /sbin/iptables -D INPUT -s %IP% -p tcp --dport 22 -j ACCEPT
tcpflags = syn
```

## NAT (Network Address Translation)

### Source NAT (SNAT) / Masquerading

**Purpose:** Hide internal IPs behind single public IP

```bash
# iptables NAT
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# Or specific IP
iptables -t nat -A POSTROUTING -o eth0 -j SNAT --to-source 203.0.113.5
```

**Traffic Flow:**
```
Internal: 192.168.1.10:5000 ’ Internet
External: 203.0.113.5:6000 ’ Internet

Firewall tracks connection:
  192.168.1.10:5000 ” 203.0.113.5:6000

Return traffic:
  Internet ’ 203.0.113.5:6000
  Firewall translates back to: 192.168.1.10:5000
```

### Destination NAT (DNAT) / Port Forwarding

**Purpose:** Expose internal service on public IP

```bash
# Forward public port 80 to internal web server
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 80 -j DNAT --to-destination 192.168.1.20:80

# Forward public port 2222 to internal SSH
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 2222 -j DNAT --to-destination 192.168.1.10:22
```

**Traffic Flow:**
```
Internet ’ 203.0.113.5:80
Firewall translates to: 192.168.1.20:80
Web server processes request
Response: 192.168.1.20:80 ’ Internet
Firewall translates from: 203.0.113.5:80 ’ Internet
```

## Firewall Evasion Techniques (for awareness)

### 1. Fragmentation

```
Split malicious payload across fragments
Some firewalls don't reassemble
```

### 2. IP Spoofing

```
Fake source IP address
Bypass source-based rules
```

### 3. Tunneling

```
Encapsulate forbidden traffic in allowed protocol
Example: SSH tunnel, DNS tunnel, ICMP tunnel
```

### 4. Encryption

```
Encrypt malicious traffic
Firewall can't inspect without SSL inspection
```

**Defense:**
- Fragment reassembly
- Anti-spoofing rules
- Protocol validation
- SSL/TLS inspection
- Deep packet inspection

## Firewall Logging

### What to Log

```
1. Blocked connections
2. Allowed critical connections
3. Rule changes
4. Authentication events
5. Anomalies (port scans, floods)
```

### iptables Logging

```bash
# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "DROPPED INPUT: " --log-level 4
iptables -A INPUT -j DROP

# Log accepted SSH
iptables -A INPUT -p tcp --dport 22 -j LOG --log-prefix "SSH ACCEPT: "
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
```

### Log Analysis

```bash
# View firewall logs (typical locations)
tail -f /var/log/syslog
tail -f /var/log/messages
tail -f /var/log/kern.log

# Search for dropped packets
grep "DROPPED" /var/log/syslog

# Count connections by source IP
grep "DROPPED" /var/log/syslog | awk '{print $NF}' | sort | uniq -c | sort -n
```

## Firewall Best Practices

### 1. Default Deny

```
Block everything by default
Explicitly allow only needed services
```

### 2. Principle of Least Privilege

```
Open only necessary ports
Restrict to specific sources when possible
Time-based rules when appropriate
```

### 3. Defense in Depth

```
Multiple layers:
  - Perimeter firewall
  - Host-based firewalls
  - Network segmentation
  - Application firewalls
```

### 4. Regular Updates

```
- Keep firewall software updated
- Review rules periodically
- Remove unused rules
- Update threat signatures (NGFW)
```

### 5. Monitoring and Alerts

```
- Enable logging
- Set up alerts for anomalies
- Regular log reviews
- Incident response plan
```

### 6. Testing

```
- Test rules before production
- Verify deny rules work
- Check for unintended access
- Regular security audits
```

## Troubleshooting Firewall Issues

### Can't connect to service

```bash
# 1. Check if service is running
systemctl status nginx

# 2. Check if service is listening
netstat -tuln | grep :80
ss -tuln | grep :80

# 3. Check firewall rules
iptables -L -n -v
ufw status
firewall-cmd --list-all

# 4. Check logs
tail -f /var/log/syslog | grep UFW
journalctl -f -u firewalld

# 5. Test from different source
curl http://server-ip
telnet server-ip 80
```

### Connection works locally but not remotely

```bash
# Likely firewall blocking external access

# Check INPUT chain
iptables -L INPUT -n -v

# Temporarily allow (testing only!)
iptables -I INPUT -p tcp --dport 80 -j ACCEPT

# If works, add permanent rule
```

### Rule not working

```bash
# Check rule order
iptables -L -n -v --line-numbers

# Rules processed top-to-bottom
# Earlier DENY rule might catch traffic before ALLOW

# Reorder rules
iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
```

## ELI10

A firewall is like a security guard at a building entrance:

**Security Guard (Firewall):**
- Checks everyone coming in and out
- Has a list of rules (who's allowed, who's not)
- Blocks suspicious people
- Keeps a log of who enters

**Types of Security:**

1. **Basic Guard (Packet Filter):**
   - Checks ID cards only
   - Fast but simple

2. **Smart Guard (Stateful):**
   - Remembers who entered
   - Allows them to leave
   - Tracks conversations

3. **Super Guard (Application Layer):**
   - Opens bags
   - Checks what you're carrying
   - Very thorough but slower

4. **AI Guard (NGFW):**
   - Facial recognition
   - Detects threats automatically
   - Learns from experience

**Rules Example:**
- "Allow employees" (like allowing HTTP port 80)
- "Block suspicious visitors" (like blocking unknown IPs)
- "Only executives can enter executive floor" (like restricting SSH to specific IPs)

**DMZ is like a reception area:**
- Visitors wait here
- Can't go into main office
- Receptionists (DMZ servers) handle requests

## Further Resources

- [iptables Tutorial](https://www.frozentux.net/iptables-tutorial/iptables-tutorial.html)
- [NIST Firewall Guide](https://csrc.nist.gov/publications/detail/sp/800-41/rev-1/final)
- [Firewall Best Practices](https://www.cisecurity.org/controls/)
- [pf (OpenBSD Firewall)](https://www.openbsd.org/faq/pf/)
