# WireGuard

WireGuard is a modern, high-performance VPN protocol that aims to be faster, simpler, and more secure than traditional VPN solutions like IPsec and OpenVPN. It has been integrated into the Linux kernel since version 5.6.

## Overview

WireGuard uses state-of-the-art cryptography and is designed with simplicity in mind, consisting of only about 4,000 lines of code compared to hundreds of thousands in older VPN implementations.

**Key Features:**
- **Performance**: Significantly faster than IPsec and OpenVPN
- **Security**: Modern cryptography with no configuration options (secure by default)
- **Simplicity**: Minimal attack surface and easy to audit
- **Cross-Platform**: Available on Linux, Windows, macOS, BSD, iOS, and Android
- **Kernel Integration**: Part of Linux kernel (5.6+)
- **Stealth**: Silent to port scanners when not actively communicating
- **Roaming**: Seamless IP address changes and network transitions

## Cryptography

WireGuard uses a fixed set of modern cryptographic primitives:

- **ChaCha20**: Symmetric encryption
- **Poly1305**: Message authentication
- **Curve25519**: Elliptic-curve Diffie-Hellman (ECDH)
- **BLAKE2s**: Cryptographic hash function
- **SipHash24**: Hash table keys
- **HKDF**: Key derivation

This approach eliminates cryptographic agility vulnerabilities and ensures consistent security.

## Protocol Architecture

### Noise Protocol Framework

WireGuard implements the Noise_IK handshake pattern from the Noise Protocol Framework:

- **I**: Initiator provides static public key
- **K**: Responder's static public key is known beforehand

This provides mutual authentication, forward secrecy, and identity hiding.

### Handshake Process

The handshake establishes a secure session between peers:

```
Initiator                                Responder
   |                                         |
   |  (1) Initiation Message                 |
   |  - Initiator's ephemeral public key     |
   |  - Encrypted static public key          |
   |  - Encrypted timestamp                  |
   | --------------------------------------> |
   |                                         |
   |  (2) Response Message                   |
   |  - Responder's ephemeral public key     |
   |  - Encrypted "empty" payload            |
   | <-------------------------------------- |
   |                                         |
   |  (3) Data Packets                       |
   |  - Encrypted with derived keys          |
   | <-------------------------------------> |
```

**Handshake Details:**

1. **Initiation (148 bytes)**:
   - Sender index (4 bytes)
   - Unencrypted ephemeral key (32 bytes)
   - Encrypted static key (32 + 16 bytes)
   - Encrypted timestamp (12 + 16 bytes)
   - MAC1 and MAC2 (16 + 16 bytes)

2. **Response (92 bytes)**:
   - Sender index (4 bytes)
   - Receiver index (4 bytes)
   - Unencrypted ephemeral key (32 bytes)
   - Encrypted empty (16 bytes)
   - MAC1 and MAC2 (16 + 16 bytes)

### Timer State Machine

WireGuard uses five timer states to manage connections:

1. **REKEY_AFTER_TIME** (120 seconds): Initiate new handshake
2. **REJECT_AFTER_TIME** (180 seconds): Reject packets, must handshake
3. **REKEY_ATTEMPT_TIME** (90 seconds): Retry handshake if no response
4. **REKEY_TIMEOUT** (5 seconds): Exponential backoff for retries
5. **KEEPALIVE_TIMEOUT** (10 seconds): Send keepalive if no traffic

### Key Rotation

WireGuard automatically rotates keys to maintain forward secrecy:

```
Time (seconds):    0        120       180
                   |         |         |
Key Pair 1:    [Active]--[Rekey]--[Reject]
                            |
Key Pair 2:              [Active]--[Rekey]--[Reject]
                                     |
Key Pair 3:                       [Active]-->
```

- Keys are valid for 180 seconds
- New handshake initiated at 120 seconds
- After 180 seconds, old keys are rejected
- Seamless transition with no connection interruption

### Packet Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Outgoing Packet Path                      │
└─────────────────────────────────────────────────────────────┘

Application Layer
       |
       v
Socket Buffer (SKB) Created
       |
       v
Routing Decision ─────> Check AllowedIPs (Cryptokey Routing)
       |                        |
       v                        v
  Match Found?            Select Peer
       |                        |
       v                        v
  [WireGuard Interface]         |
       |                        |
       v                        |
  Valid Handshake? <────────────┘
       |
       | Yes (use session keys)
       v
  Encrypt with ChaCha20-Poly1305
       |
       v
  Add WireGuard Header
  - Type (4 bytes)
  - Receiver index (4 bytes)
  - Counter (8 bytes)
  - Encrypted payload
  - Poly1305 tag (16 bytes)
       |
       v
  UDP Encapsulation (port 51820)
       |
       v
  Send via Physical Interface

┌─────────────────────────────────────────────────────────────┐
│                    Incoming Packet Path                      │
└─────────────────────────────────────────────────────────────┘

Physical Interface
       |
       v
UDP Packet Received (port 51820)
       |
       v
WireGuard Module
       |
       v
Packet Type Check
  ├─> Handshake Initiation (Type 1)
  ├─> Handshake Response (Type 2)
  ├─> Cookie Reply (Type 3)
  └─> Transport Data (Type 4)
       |
       v
  Lookup Receiver Index
       |
       v
  Verify Counter (anti-replay)
       |
       v
  Decrypt with ChaCha20-Poly1305
       |
       v
  Verify Poly1305 MAC
       |
       v
  Extract IP Packet
       |
       v
  Verify Source IP in AllowedIPs
       |
       v
  Forward to Network Stack
       |
       v
  Application Layer
```

### Cryptokey Routing

WireGuard's unique routing mechanism based on public keys:

```
Peer Configuration:
[Peer]
PublicKey = <peer_key>
AllowedIPs = 10.0.0.2/32, 192.168.1.0/24

Routing Logic:
1. Outgoing:
   Destination IP → Lookup in AllowedIPs → Select Peer → Encrypt

2. Incoming:
   Decrypt → Extract Source IP → Verify in Peer's AllowedIPs → Accept
```

This provides both routing and firewall functionality in one mechanism.

## Kernel Implementation

### Data Structures

Key kernel data structures in WireGuard:

```c
// Main device structure
struct wg_device {
    struct net_device *dev;
    struct list_head peer_list;
    struct mutex device_update_lock;
    struct sk_buff_head incoming_handshakes;
    // ... encryption keys, timers, etc.
};

// Peer structure
struct wg_peer {
    struct wg_device *device;
    struct endpoint endpoint;
    struct wireguard_peer *next;
    u8 public_key[NOISE_PUBLIC_KEY_LEN];
    // ... session keys, timers, allowedips, etc.
};

// Session keys
struct noise_keypair {
    u8 sending_key[CHACHA20POLY1305_KEY_SIZE];
    u8 receiving_key[CHACHA20POLY1305_KEY_SIZE];
    u64 sending_counter;
    u64 receiving_counter;
    // ... timestamps, validity
};
```

### Netlink Interface

WireGuard uses Generic Netlink for userspace communication:

```c
// Communication between wg-quick/wg and kernel module
Userspace (wg tool)
        |
        | Netlink messages
        v
    WG_CMD_GET_DEVICE
    WG_CMD_SET_DEVICE
        |
        v
Kernel Module (wireguard.ko)
        |
        v
    Process configuration
    Update peer lists
    Set keys and endpoints
```

**Netlink Commands:**
- `WG_CMD_GET_DEVICE`: Retrieve interface configuration
- `WG_CMD_SET_DEVICE`: Update interface configuration

### Network Stack Integration

WireGuard integrates as a network device:

```
┌───────────────────────────────────────────────┐
│        Linux Network Stack                     │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   eth0   │  │   wlan0  │  │   wg0    │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │              │          │
│       └─────────────┴──────────────┘          │
│                     │                         │
│              ┌──────▼──────┐                  │
│              │   Routing   │                  │
│              │   Decision  │                  │
│              └──────┬──────┘                  │
│                     │                         │
│              ┌──────▼──────┐                  │
│              │  iptables/  │                  │
│              │  netfilter  │                  │
│              └──────┬──────┘                  │
│                     │                         │
│              ┌──────▼──────┐                  │
│              │Application  │                  │
│              └─────────────┘                  │
└───────────────────────────────────────────────┘

WireGuard Operations:
- net_device_ops for interface operations
- Hard header length for encapsulation
- MTU handling for overhead
- Queueing discipline integration
```

### Performance Optimizations

Kernel-level optimizations in WireGuard:

1. **SIMD Acceleration**: Uses CPU SIMD instructions for ChaCha20
2. **Parallel Processing**: Multi-core capable for encryption/decryption
3. **Zero-Copy**: Minimizes memory copies in data path
4. **Lockless Operations**: Lock-free data structures where possible
5. **Batch Processing**: Handles multiple packets efficiently

## Installation

### Kernel Module (Linux 5.6+)

On modern kernels, WireGuard is built-in:

```bash
# Check if WireGuard is available
sudo modinfo wireguard

# Load the module if needed
sudo modprobe wireguard

# Verify module is loaded
lsmod | grep wireguard
```

### User-Space Tools

Install the WireGuard tools:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install wireguard wireguard-tools

# Fedora/RHEL/CentOS
sudo dnf install wireguard-tools

# Arch Linux
sudo pacman -S wireguard-tools

# Verify installation
wg --version
```

## Key Concepts

### Interface

WireGuard interfaces are network interfaces like any other (e.g., eth0, wlan0):

```bash
# Create a WireGuard interface
sudo ip link add dev wg0 type wireguard

# Delete a WireGuard interface
sudo ip link delete dev wg0
```

### Public/Private Keys

WireGuard uses public-key cryptography for authentication:

```bash
# Generate private key
wg genkey > privatekey

# Generate public key from private key
wg pubkey < privatekey > publickey

# Generate both keys
umask 077
wg genkey | tee privatekey | wg pubkey > publickey

# Generate pre-shared key (optional, for additional security)
wg genpsk > presharedkey
```

**Security Note**: Private keys should never be shared and must be protected with proper file permissions (600).

### Peers

Each WireGuard interface has a list of peers it can communicate with. Each peer is identified by their public key.

### Allowed IPs

The "allowed IPs" setting determines:
1. Which source IPs can be received from a peer (incoming traffic filtering)
2. Which destination IPs are routed to a peer (outgoing routing)

This dual purpose is a key WireGuard concept called "Cryptokey Routing".

## Configuration

### Configuration File Format

WireGuard configuration files are typically stored in `/etc/wireguard/`:

```ini
# /etc/wireguard/wg0.conf
[Interface]
# Interface settings
PrivateKey = <interface_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
# Optional settings
#PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
#PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
#DNS = 1.1.1.1, 8.8.8.8
#MTU = 1420

[Peer]
# Peer configuration
PublicKey = <peer_public_key>
# Optional pre-shared key for quantum resistance
#PresharedKey = <pre_shared_key>
# Which IPs this peer can use as source/destination
AllowedIPs = 10.0.0.2/32
# Peer's external endpoint
Endpoint = peer.example.com:51820
# Keep connection alive (useful for NAT traversal)
PersistentKeepalive = 25
```

### Interface Configuration Options

| Option | Description |
|--------|-------------|
| `PrivateKey` | Interface's private key (required) |
| `ListenPort` | UDP port to listen on (default: random) |
| `Address` | IP address(es) assigned to interface |
| `DNS` | DNS servers for the interface |
| `MTU` | Maximum transmission unit size |
| `Table` | Routing table to use (auto, off, or number) |
| `PreUp` | Command to run before bringing interface up |
| `PostUp` | Command to run after bringing interface up |
| `PreDown` | Command to run before bringing interface down |
| `PostDown` | Command to run after bringing interface down |

### Peer Configuration Options

| Option | Description |
|--------|-------------|
| `PublicKey` | Peer's public key (required) |
| `PresharedKey` | Pre-shared key for additional security |
| `AllowedIPs` | CIDR ranges for cryptokey routing |
| `Endpoint` | Peer's external IP/hostname and port |
| `PersistentKeepalive` | Interval in seconds for keepalive packets |

## Basic Setup Examples

### Point-to-Point VPN

**Server Configuration** (`/etc/wireguard/wg0.conf`):

```ini
[Interface]
PrivateKey = <server_private_key>
Address = 10.0.0.1/24
ListenPort = 51820

[Peer]
PublicKey = <client_public_key>
AllowedIPs = 10.0.0.2/32
```

**Client Configuration** (`/etc/wireguard/wg0.conf`):

```ini
[Interface]
PrivateKey = <client_private_key>
Address = 10.0.0.2/24

[Peer]
PublicKey = <server_public_key>
AllowedIPs = 10.0.0.0/24
Endpoint = server.example.com:51820
PersistentKeepalive = 25
```

### VPN Gateway (Route All Traffic)

**Server Configuration** (`/etc/wireguard/wg0.conf`):

```ini
[Interface]
PrivateKey = <server_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = <client_public_key>
AllowedIPs = 10.0.0.2/32
```

**Client Configuration** (`/etc/wireguard/wg0.conf`):

```ini
[Interface]
PrivateKey = <client_private_key>
Address = 10.0.0.2/24
DNS = 1.1.1.1

[Peer]
PublicKey = <server_public_key>
# Route all traffic through VPN
AllowedIPs = 0.0.0.0/0
Endpoint = server.example.com:51820
PersistentKeepalive = 25
```

**Server System Configuration**:

```bash
# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1
sudo sysctl -w net.ipv6.conf.all.forwarding=1

# Make permanent
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
echo "net.ipv6.conf.all.forwarding=1" | sudo tee -a /etc/sysctl.conf
```

### Site-to-Site VPN

**Site A Configuration**:

```ini
[Interface]
PrivateKey = <site_a_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT
PostDown = iptables -D FORWARD -i %i -j ACCEPT

[Peer]
PublicKey = <site_b_public_key>
# Allow traffic to Site B's local network
AllowedIPs = 192.168.2.0/24, 10.0.0.2/32
Endpoint = site-b.example.com:51820
PersistentKeepalive = 25
```

**Site B Configuration**:

```ini
[Interface]
PrivateKey = <site_b_private_key>
Address = 10.0.0.2/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT
PostDown = iptables -D FORWARD -i %i -j ACCEPT

[Peer]
PublicKey = <site_a_public_key>
# Allow traffic to Site A's local network
AllowedIPs = 192.168.1.0/24, 10.0.0.1/32
Endpoint = site-a.example.com:51820
PersistentKeepalive = 25
```

## Managing WireGuard

### Using wg-quick

The `wg-quick` utility simplifies interface management:

```bash
# Start VPN interface
sudo wg-quick up wg0

# Stop VPN interface
sudo wg-quick down wg0

# Restart VPN interface
sudo wg-quick down wg0 && sudo wg-quick up wg0
```

### Using systemd

Enable WireGuard to start on boot:

```bash
# Enable and start
sudo systemctl enable wg-quick@wg0.service
sudo systemctl start wg-quick@wg0.service

# Check status
sudo systemctl status wg-quick@wg0.service

# Stop and disable
sudo systemctl stop wg-quick@wg0.service
sudo systemctl disable wg-quick@wg0.service

# Restart
sudo systemctl restart wg-quick@wg0.service
```

### Manual Configuration

Configure WireGuard interfaces manually using `wg` and `ip` commands:

```bash
# Create interface
sudo ip link add dev wg0 type wireguard

# Configure interface
sudo wg setconf wg0 /etc/wireguard/wg0.conf
# Or configure directly
sudo wg set wg0 private-key /etc/wireguard/privatekey listen-port 51820

# Assign IP address
sudo ip address add dev wg0 10.0.0.1/24

# Bring interface up
sudo ip link set up dev wg0

# Show configuration
sudo wg show wg0
```

### Adding/Removing Peers Dynamically

```bash
# Add a peer
sudo wg set wg0 peer <peer_public_key> \
  allowed-ips 10.0.0.3/32 \
  endpoint peer.example.com:51820 \
  persistent-keepalive 25

# Remove a peer
sudo wg set wg0 peer <peer_public_key> remove

# Update peer's allowed IPs
sudo wg set wg0 peer <peer_public_key> \
  allowed-ips 10.0.0.3/32,10.0.1.0/24
```

## Monitoring and Troubleshooting

### Show Interface Status

```bash
# Show all WireGuard interfaces
sudo wg show

# Show specific interface
sudo wg show wg0

# Show in different formats
sudo wg show wg0 dump        # Machine-readable format
sudo wg show wg0 endpoints   # Show peer endpoints
sudo wg show wg0 allowed-ips # Show allowed IPs
sudo wg show wg0 latest-handshakes  # Show handshake times
sudo wg show wg0 transfer    # Show data transfer
sudo wg show wg0 persistent-keepalive  # Show keepalive settings
```

### Detailed Status Output

```bash
# View detailed interface information
sudo wg show wg0

# Example output:
# interface: wg0
#   public key: <public_key>
#   private key: (hidden)
#   listening port: 51820
#
# peer: <peer_public_key>
#   endpoint: 203.0.113.1:51820
#   allowed ips: 10.0.0.2/32
#   latest handshake: 1 minute, 23 seconds ago
#   transfer: 15.23 MiB received, 8.92 MiB sent
#   persistent keepalive: every 25 seconds
```

### Check Connectivity

```bash
# Ping peer through tunnel
ping 10.0.0.2

# Trace route through tunnel
traceroute 10.0.0.2

# Check if handshake is happening
sudo wg show wg0 latest-handshakes

# Monitor interface statistics
sudo ip -s link show wg0

# Check for errors
sudo dmesg | grep wireguard
sudo journalctl -u wg-quick@wg0
```

### Common Issues and Solutions

**1. No handshake occurring:**

```bash
# Check if WireGuard is running
sudo wg show

# Verify endpoint is reachable
ping -c 4 server.example.com
nc -u -v server.example.com 51820

# Check firewall rules
sudo iptables -L -n | grep 51820
sudo ufw status
```

**2. Handshake successful but no traffic:**

```bash
# Verify IP forwarding (on server)
sudo sysctl net.ipv4.ip_forward
sudo sysctl net.ipv6.conf.all.forwarding

# Check routing
ip route get 10.0.0.2
ip route show table all

# Verify iptables rules
sudo iptables -L FORWARD -n -v
sudo iptables -t nat -L POSTROUTING -n -v
```

**3. MTU issues (packet loss/slow performance):**

```bash
# Test MTU
ping -M do -s 1400 10.0.0.2

# Adjust MTU in configuration
# Add to [Interface] section:
# MTU = 1420

# Or manually:
sudo ip link set mtu 1420 dev wg0
```

**4. Connection drops after network change:**

```bash
# Ensure PersistentKeepalive is set (client side)
# Add to [Peer] section:
# PersistentKeepalive = 25

# Force handshake
sudo wg set wg0 peer <peer_public_key> endpoint new.example.com:51820
```

## Advanced Configurations

### Multiple Peers (Road Warrior Setup)

Server configuration for multiple clients:

```ini
[Interface]
PrivateKey = <server_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# Client 1
[Peer]
PublicKey = <client1_public_key>
AllowedIPs = 10.0.0.2/32

# Client 2
[Peer]
PublicKey = <client2_public_key>
AllowedIPs = 10.0.0.3/32

# Client 3
[Peer]
PublicKey = <client3_public_key>
AllowedIPs = 10.0.0.4/32
```

### Dynamic IP Assignment Script

For larger deployments, automate client IP assignment:

```bash
#!/bin/bash
# add-client.sh

CLIENT_NAME=$1
SERVER_PUBLIC_KEY="<server_public_key>"
SERVER_ENDPOINT="vpn.example.com:51820"
CONFIG_DIR="/etc/wireguard"
NETWORK="10.0.0"

# Find next available IP
NEXT_IP=$(wg show wg0 allowed-ips | \
  grep -oP "${NETWORK}\.\K\d+" | \
  sort -n | tail -1)
NEXT_IP=$((NEXT_IP + 1))

# Generate keys
umask 077
CLIENT_PRIVATE_KEY=$(wg genkey)
CLIENT_PUBLIC_KEY=$(echo "$CLIENT_PRIVATE_KEY" | wg pubkey)

# Add peer to server
wg set wg0 peer "$CLIENT_PUBLIC_KEY" \
  allowed-ips ${NETWORK}.${NEXT_IP}/32

# Generate client config
cat > "${CONFIG_DIR}/${CLIENT_NAME}.conf" <<EOF
[Interface]
PrivateKey = ${CLIENT_PRIVATE_KEY}
Address = ${NETWORK}.${NEXT_IP}/24
DNS = 1.1.1.1

[Peer]
PublicKey = ${SERVER_PUBLIC_KEY}
AllowedIPs = 0.0.0.0/0
Endpoint = ${SERVER_ENDPOINT}
PersistentKeepalive = 25
EOF

echo "Client ${CLIENT_NAME} added with IP ${NETWORK}.${NEXT_IP}"
echo "Config: ${CONFIG_DIR}/${CLIENT_NAME}.conf"

# Save to server config
wg-quick save wg0
```

### Split Tunneling

Route only specific traffic through VPN:

```ini
[Interface]
PrivateKey = <client_private_key>
Address = 10.0.0.2/24

[Peer]
PublicKey = <server_public_key>
# Only route specific networks through VPN
AllowedIPs = 10.0.0.0/24, 192.168.1.0/24
Endpoint = server.example.com:51820
PersistentKeepalive = 25
```

### IPv6 Support

Enable IPv6 in WireGuard:

```ini
[Interface]
PrivateKey = <private_key>
Address = 10.0.0.1/24, fd00::1/64
ListenPort = 51820

[Peer]
PublicKey = <peer_public_key>
AllowedIPs = 10.0.0.2/32, fd00::2/128
Endpoint = peer.example.com:51820
```

### Network Namespaces

Isolate WireGuard in a network namespace:

```bash
# Create namespace
sudo ip netns add wg_namespace

# Create WireGuard interface in namespace
sudo ip link add wg0 type wireguard
sudo ip link set wg0 netns wg_namespace

# Configure in namespace
sudo ip netns exec wg_namespace wg setconf wg0 /etc/wireguard/wg0.conf
sudo ip netns exec wg_namespace ip addr add 10.0.0.1/24 dev wg0
sudo ip netns exec wg_namespace ip link set wg0 up

# Run application in namespace
sudo ip netns exec wg_namespace sudo -u user firefox
```

### Pre-shared Keys for Quantum Resistance

Add pre-shared keys for post-quantum security:

```bash
# Generate pre-shared key
wg genpsk > presharedkey

# Add to peer configuration
sudo wg set wg0 peer <peer_public_key> \
  preshared-key /etc/wireguard/presharedkey
```

Configuration file:

```ini
[Peer]
PublicKey = <peer_public_key>
PresharedKey = <pre_shared_key>
AllowedIPs = 10.0.0.2/32
```

### High Availability Configurations

#### Active-Passive Failover

Primary and backup servers with automatic failover:

**Primary Server** (`/etc/wireguard/wg0.conf`):

```ini
[Interface]
PrivateKey = <primary_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
PostUp = ip route add 192.168.1.0/24 dev wg0

[Peer]
PublicKey = <client_public_key>
AllowedIPs = 10.0.0.2/32
```

**Backup Server** (`/etc/wireguard/wg0.conf`):

```ini
[Interface]
PrivateKey = <backup_private_key>  # Same or different key
Address = 10.0.0.1/24               # Same VPN IP
ListenPort = 51820
PostUp = ip route add 192.168.1.0/24 dev wg0

[Peer]
PublicKey = <client_public_key>
AllowedIPs = 10.0.0.2/32
```

**Client Configuration with Failover**:

```ini
[Interface]
PrivateKey = <client_private_key>
Address = 10.0.0.2/24

# Primary peer
[Peer]
PublicKey = <primary_public_key>
AllowedIPs = 0.0.0.0/0
Endpoint = primary.example.com:51820
PersistentKeepalive = 25

# Backup peer (activate manually or via script)
#[Peer]
#PublicKey = <backup_public_key>
#AllowedIPs = 0.0.0.0/0
#Endpoint = backup.example.com:51820
#PersistentKeepalive = 25
```

**Failover Script** (`/usr/local/bin/wg-failover.sh`):

```bash
#!/bin/bash

PRIMARY_ENDPOINT="primary.example.com:51820"
BACKUP_ENDPOINT="backup.example.com:51820"
PRIMARY_KEY="<primary_public_key>"
BACKUP_KEY="<backup_public_key>"
INTERFACE="wg0"
CHECK_INTERVAL=10

check_connection() {
    # Check if last handshake was recent (within 3 minutes)
    last_handshake=$(sudo wg show "$INTERFACE" latest-handshakes | \
        grep "$1" | awk '{print $2}')
    current_time=$(date +%s)

    if [ -z "$last_handshake" ]; then
        return 1
    fi

    time_diff=$((current_time - last_handshake))
    [ "$time_diff" -lt 180 ]
}

while true; do
    if ! check_connection "$PRIMARY_KEY"; then
        echo "Primary connection failed, switching to backup"
        sudo wg set "$INTERFACE" peer "$PRIMARY_KEY" remove
        sudo wg set "$INTERFACE" peer "$BACKUP_KEY" \
            endpoint "$BACKUP_ENDPOINT" \
            allowed-ips 0.0.0.0/0 \
            persistent-keepalive 25
    elif check_connection "$BACKUP_KEY"; then
        echo "Primary restored, switching back"
        sudo wg set "$INTERFACE" peer "$BACKUP_KEY" remove
        sudo wg set "$INTERFACE" peer "$PRIMARY_KEY" \
            endpoint "$PRIMARY_ENDPOINT" \
            allowed-ips 0.0.0.0/0 \
            persistent-keepalive 25
    fi
    sleep "$CHECK_INTERVAL"
done
```

Run as systemd service:

```ini
# /etc/systemd/system/wg-failover.service
[Unit]
Description=WireGuard Failover Service
After=wg-quick@wg0.service
Requires=wg-quick@wg0.service

[Service]
Type=simple
ExecStart=/usr/local/bin/wg-failover.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Load Balancing with Multiple Peers

Distribute traffic across multiple WireGuard servers:

**Client Configuration**:

```ini
[Interface]
PrivateKey = <client_private_key>
Address = 10.0.0.10/24
Table = off  # Disable automatic routing
PostUp = ip route add 10.0.1.0/24 via 10.0.0.1 dev wg0
PostUp = ip route add 10.0.2.0/24 via 10.0.0.2 dev wg0

# Server 1 - for network 10.0.1.0/24
[Peer]
PublicKey = <server1_public_key>
AllowedIPs = 10.0.0.1/32, 10.0.1.0/24
Endpoint = server1.example.com:51820
PersistentKeepalive = 25

# Server 2 - for network 10.0.2.0/24
[Peer]
PublicKey = <server2_public_key>
AllowedIPs = 10.0.0.2/32, 10.0.2.0/24
Endpoint = server2.example.com:51820
PersistentKeepalive = 25
```

### Multi-Hop and Complex Topologies

#### Multi-Hop VPN (Cascading)

Route traffic through multiple WireGuard servers:

```
Client → Server A → Server B → Internet
```

**Server A Configuration**:

```ini
[Interface]
PrivateKey = <server_a_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT
PostUp = iptables -t nat -A POSTROUTING -o wg1 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -o wg1 -j MASQUERADE

[Peer]
PublicKey = <client_public_key>
AllowedIPs = 10.0.0.2/32

# Second interface for connection to Server B
# /etc/wireguard/wg1.conf
[Interface]
PrivateKey = <server_a_wg1_private_key>
Address = 10.0.1.1/24

[Peer]
PublicKey = <server_b_public_key>
AllowedIPs = 0.0.0.0/0
Endpoint = server-b.example.com:51820
PersistentKeepalive = 25
```

**Client Configuration**:

```ini
[Interface]
PrivateKey = <client_private_key>
Address = 10.0.0.2/24
DNS = 1.1.1.1

[Peer]
PublicKey = <server_a_public_key>
AllowedIPs = 0.0.0.0/0
Endpoint = server-a.example.com:51820
PersistentKeepalive = 25
```

#### Hub-and-Spoke Network

Central hub with multiple spoke sites:

```
        Spoke 1 (192.168.1.0/24)
              |
              |
         [Hub Server]
         (10.0.0.1)
         /         \
        /           \
  Spoke 2         Spoke 3
(192.168.2.0/24) (192.168.3.0/24)
```

**Hub Configuration**:

```ini
[Interface]
PrivateKey = <hub_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
PostUp = sysctl -w net.ipv4.ip_forward=1

# Spoke 1
[Peer]
PublicKey = <spoke1_public_key>
AllowedIPs = 10.0.0.2/32, 192.168.1.0/24

# Spoke 2
[Peer]
PublicKey = <spoke2_public_key>
AllowedIPs = 10.0.0.3/32, 192.168.2.0/24

# Spoke 3
[Peer]
PublicKey = <spoke3_public_key>
AllowedIPs = 10.0.0.4/32, 192.168.3.0/24
```

**Spoke Configuration** (example for Spoke 1):

```ini
[Interface]
PrivateKey = <spoke1_private_key>
Address = 10.0.0.2/24
PostUp = ip route add 192.168.2.0/24 via 10.0.0.1 dev wg0
PostUp = ip route add 192.168.3.0/24 via 10.0.0.1 dev wg0

[Peer]
PublicKey = <hub_public_key>
AllowedIPs = 10.0.0.1/32, 192.168.2.0/24, 192.168.3.0/24
Endpoint = hub.example.com:51820
PersistentKeepalive = 25
```

#### Full Mesh Network

Every node connects to every other node:

```
     Node A ─────── Node B
       │  \       /  │
       │   \     /   │
       │    \   /    │
       │     \ /     │
       │      X      │
       │     / \     │
       │    /   \    │
       │   /     \   │
       │  /       \  │
     Node C ─────── Node D
```

**Node A Configuration**:

```ini
[Interface]
PrivateKey = <node_a_private_key>
Address = 10.0.0.1/24
ListenPort = 51820

[Peer]  # Node B
PublicKey = <node_b_public_key>
AllowedIPs = 10.0.0.2/32
Endpoint = node-b.example.com:51820
PersistentKeepalive = 25

[Peer]  # Node C
PublicKey = <node_c_public_key>
AllowedIPs = 10.0.0.3/32
Endpoint = node-c.example.com:51820
PersistentKeepalive = 25

[Peer]  # Node D
PublicKey = <node_d_public_key>
AllowedIPs = 10.0.0.4/32
Endpoint = node-d.example.com:51820
PersistentKeepalive = 25
```

### DNS Configuration

#### Split DNS Configuration

Route DNS queries based on domain:

```ini
[Interface]
PrivateKey = <private_key>
Address = 10.0.0.2/24
# Internal DNS for corporate domains
DNS = 10.0.0.53
PostUp = resolvconf -a %i -m 0 -x
PostUp = echo "search corporate.local" >> /etc/resolv.conf
PostUp = echo "nameserver 10.0.0.53" >> /etc/resolv.conf
PostDown = resolvconf -d %i

[Peer]
PublicKey = <server_public_key>
AllowedIPs = 10.0.0.0/24
Endpoint = vpn.example.com:51820
PersistentKeepalive = 25
```

#### DNS-Over-HTTPS Through Tunnel

**Server-side DNS-over-HTTPS Setup**:

```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
sudo chmod +x /usr/local/bin/cloudflared

# Configure DNS proxy
sudo cloudflared proxy-dns --port 53 --upstream https://1.1.1.1/dns-query

# Add to WireGuard config
# DNS = 10.0.0.1  (server's WireGuard IP)
```

#### Internal DNS Server

Set up a DNS server for the VPN network:

```bash
# Install dnsmasq
sudo apt install dnsmasq

# Configure /etc/dnsmasq.conf
interface=wg0
bind-interfaces
listen-address=10.0.0.1
domain=vpn.local
expand-hosts

# DNS records
address=/server1.vpn.local/10.0.0.10
address=/server2.vpn.local/10.0.0.11

# Upstream DNS
server=1.1.1.1
server=8.8.8.8

# Enable and start
sudo systemctl enable dnsmasq
sudo systemctl restart dnsmasq
```

**Client Configuration**:

```ini
[Interface]
PrivateKey = <client_private_key>
Address = 10.0.0.2/24
DNS = 10.0.0.1
PostUp = echo "search vpn.local" >> /etc/resolv.conf

[Peer]
PublicKey = <server_public_key>
AllowedIPs = 10.0.0.0/24
Endpoint = vpn.example.com:51820
PersistentKeepalive = 25
```

### Dynamic DNS for Endpoints

Update WireGuard endpoints when server IP changes:

**DDNS Update Script** (`/usr/local/bin/wg-ddns-update.sh`):

```bash
#!/bin/bash

INTERFACE="wg0"
PEER_KEY="<peer_public_key>"
DDNS_HOSTNAME="dynamic.example.com"
PORT="51820"
CHECK_INTERVAL=300  # 5 minutes

get_current_ip() {
    dig +short "$DDNS_HOSTNAME" A | tail -n1
}

get_configured_endpoint() {
    sudo wg show "$INTERFACE" endpoints | grep "$PEER_KEY" | awk '{print $2}'
}

while true; do
    current_ip=$(get_current_ip)
    configured_endpoint=$(get_configured_endpoint)
    expected_endpoint="${current_ip}:${PORT}"

    if [ "$configured_endpoint" != "$expected_endpoint" ] && [ -n "$current_ip" ]; then
        echo "Updating endpoint to $expected_endpoint"
        sudo wg set "$INTERFACE" peer "$PEER_KEY" endpoint "$expected_endpoint"
    fi

    sleep "$CHECK_INTERVAL"
done
```

## Security Best Practices

### Key Management

```bash
# Secure private key permissions
sudo chmod 600 /etc/wireguard/privatekey
sudo chmod 600 /etc/wireguard/wg0.conf
sudo chown root:root /etc/wireguard/*

# Store keys securely
# - Never commit to version control
# - Use encrypted storage
# - Rotate keys periodically
```

### Firewall Configuration

```bash
# Allow WireGuard traffic
sudo ufw allow 51820/udp

# Or with iptables
sudo iptables -A INPUT -p udp --dport 51820 -j ACCEPT

# Restrict to specific sources (more secure)
sudo ufw allow from 203.0.113.0/24 to any port 51820 proto udp
```

### Harden Server Configuration

```bash
# Disable IP forwarding for other interfaces
sudo iptables -A FORWARD -i wg0 -j ACCEPT
sudo iptables -A FORWARD -o wg0 -j ACCEPT
sudo iptables -A FORWARD -j DROP

# Rate limit new connections
sudo iptables -A INPUT -p udp --dport 51820 \
  -m state --state NEW -m recent --set
sudo iptables -A INPUT -p udp --dport 51820 \
  -m state --state NEW -m recent --update --seconds 60 --hitcount 10 -j DROP
```

### Monitoring and Auditing

```bash
# Log connections
sudo journalctl -u wg-quick@wg0 -f

# Monitor bandwidth
watch -n 1 'sudo wg show wg0 transfer'

# Track handshakes
watch -n 5 'sudo wg show wg0 latest-handshakes'

# Audit configuration
sudo wg show all
```

## Monitoring and Observability

### Prometheus Metrics Exporter

Monitor WireGuard with Prometheus using a custom exporter:

**Install wireguard_exporter**:

```bash
# Download and install
wget https://github.com/MindFlavor/prometheus_wireguard_exporter/releases/latest/download/prometheus_wireguard_exporter
sudo mv prometheus_wireguard_exporter /usr/local/bin/
sudo chmod +x /usr/local/bin/prometheus_wireguard_exporter

# Create systemd service
sudo tee /etc/systemd/system/prometheus-wireguard-exporter.service <<EOF
[Unit]
Description=Prometheus WireGuard Exporter
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/prometheus_wireguard_exporter -n /etc/wireguard/wg0.conf
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable prometheus-wireguard-exporter
sudo systemctl start prometheus-wireguard-exporter
```

**Prometheus Configuration** (`prometheus.yml`):

```yaml
scrape_configs:
  - job_name: 'wireguard'
    static_configs:
      - targets: ['localhost:9586']
        labels:
          instance: 'wg-server-01'
```

**Key Metrics Exposed**:
- `wireguard_sent_bytes_total`: Total bytes sent per peer
- `wireguard_received_bytes_total`: Total bytes received per peer
- `wireguard_latest_handshake_seconds`: Unix timestamp of last handshake
- `wireguard_peers`: Number of configured peers

### Grafana Dashboard

Create a Grafana dashboard for WireGuard:

**Example Dashboard JSON** (key panels):

```json
{
  "dashboard": {
    "title": "WireGuard Monitoring",
    "panels": [
      {
        "title": "Active Connections",
        "targets": [{
          "expr": "count(time() - wireguard_latest_handshake_seconds < 300)"
        }]
      },
      {
        "title": "Traffic per Peer",
        "targets": [{
          "expr": "rate(wireguard_sent_bytes_total[5m])"
        }, {
          "expr": "rate(wireguard_received_bytes_total[5m])"
        }]
      },
      {
        "title": "Handshake Freshness",
        "targets": [{
          "expr": "(time() - wireguard_latest_handshake_seconds) / 60"
        }]
      }
    ]
  }
}
```

### Logging Configuration

**Enhanced Logging with systemd**:

```bash
# Enable detailed logging
sudo mkdir -p /var/log/wireguard

# Create logging wrapper script
sudo tee /usr/local/bin/wg-quick-log <<'EOF'
#!/bin/bash
LOG_FILE="/var/log/wireguard/wg0.log"
ACTION=$1
INTERFACE=$2

{
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Action: $ACTION, Interface: $INTERFACE"
    /usr/bin/wg-quick "$ACTION" "$INTERFACE" 2>&1
    echo "Exit code: $?"
} | tee -a "$LOG_FILE"
EOF

sudo chmod +x /usr/local/bin/wg-quick-log

# Modify systemd service to use wrapper
sudo systemctl edit wg-quick@wg0 --full
# Change ExecStart to: /usr/local/bin/wg-quick-log up %i
```

**Connection Logging Script**:

```bash
#!/bin/bash
# /usr/local/bin/wg-connection-logger.sh

INTERFACE="wg0"
LOG_FILE="/var/log/wireguard/connections.log"
CHECK_INTERVAL=60

declare -A last_handshakes

while true; do
    while IFS= read -r line; do
        peer_key=$(echo "$line" | awk '{print $1}')
        handshake_time=$(echo "$line" | awk '{print $2}')

        if [ "${last_handshakes[$peer_key]}" != "$handshake_time" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - New handshake: $peer_key" >> "$LOG_FILE"
            last_handshakes[$peer_key]=$handshake_time
        fi
    done < <(sudo wg show "$INTERFACE" latest-handshakes)

    sleep "$CHECK_INTERVAL"
done
```

### Alert Rules

**Prometheus Alert Rules**:

```yaml
# /etc/prometheus/rules/wireguard.yml
groups:
  - name: wireguard
    interval: 30s
    rules:
      - alert: WireGuardPeerDown
        expr: (time() - wireguard_latest_handshake_seconds) > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "WireGuard peer {{ $labels.public_key }} is down"
          description: "No handshake in last 5 minutes"

      - alert: WireGuardHighLatency
        expr: wireguard_latest_handshake_seconds > 180
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "WireGuard peer {{ $labels.public_key }} high latency"

      - alert: WireGuardNoTraffic
        expr: rate(wireguard_received_bytes_total[5m]) == 0
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "No traffic on peer {{ $labels.public_key }}"
```

### Health Check Script

```bash
#!/bin/bash
# /usr/local/bin/wg-health-check.sh

INTERFACE="wg0"
MAX_HANDSHAKE_AGE=300  # 5 minutes
EXIT_CODE=0

echo "WireGuard Health Check - $(date)"
echo "======================================"

# Check if interface exists
if ! ip link show "$INTERFACE" &>/dev/null; then
    echo "❌ Interface $INTERFACE does not exist"
    exit 1
fi

# Check if interface is up
if ! ip link show "$INTERFACE" | grep -q "UP"; then
    echo "❌ Interface $INTERFACE is down"
    exit 1
fi

echo "✓ Interface $INTERFACE is up"

# Check peers
CURRENT_TIME=$(date +%s)
PEER_COUNT=0
HEALTHY_PEERS=0

while IFS= read -r line; do
    PEER_COUNT=$((PEER_COUNT + 1))
    peer_key=$(echo "$line" | awk '{print $1}')
    handshake_time=$(echo "$line" | awk '{print $2}')

    if [ -z "$handshake_time" ] || [ "$handshake_time" -eq 0 ]; then
        echo "⚠ Peer ${peer_key:0:16}... never completed handshake"
        EXIT_CODE=1
        continue
    fi

    age=$((CURRENT_TIME - handshake_time))

    if [ "$age" -gt "$MAX_HANDSHAKE_AGE" ]; then
        echo "⚠ Peer ${peer_key:0:16}... handshake too old (${age}s ago)"
        EXIT_CODE=1
    else
        echo "✓ Peer ${peer_key:0:16}... healthy (handshake ${age}s ago)"
        HEALTHY_PEERS=$((HEALTHY_PEERS + 1))
    fi
done < <(sudo wg show "$INTERFACE" latest-handshakes)

echo ""
echo "Summary: $HEALTHY_PEERS/$PEER_COUNT peers healthy"

exit $EXIT_CODE
```

## Performance Tuning

### Optimize UDP Buffer Sizes

```bash
# Increase UDP buffer sizes
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.rmem_default=26214400
sudo sysctl -w net.core.wmem_max=26214400
sudo sysctl -w net.core.wmem_default=26214400
sudo sysctl -w net.core.netdev_max_backlog=2000
```

### MTU Optimization

```bash
# Determine optimal MTU
# Standard Ethernet MTU: 1500
# WireGuard overhead: 60 bytes (IPv4) or 80 bytes (IPv6)
# Optimal MTU: 1420 (IPv4) or 1400 (IPv6)

# Set in configuration
# MTU = 1420

# Or manually
sudo ip link set mtu 1420 dev wg0
```

### CPU Affinity

```bash
# Pin WireGuard to specific CPU cores (for high-throughput scenarios)
# Find WireGuard kernel threads
ps -eLo psr,pid,comm | grep wg

# Set CPU affinity
sudo taskset -cp 0,1 <wg_pid>
```

## Integration Examples

### With Docker

```bash
# Allow Docker containers to use WireGuard
docker run -it --rm \
  --cap-add=NET_ADMIN \
  --device=/dev/net/tun \
  -v /path/to/wg0.conf:/etc/wireguard/wg0.conf \
  alpine sh -c "apk add wireguard-tools && wg-quick up wg0 && sh"
```

### With NetworkManager

```bash
# Import WireGuard connection
sudo nmcli connection import type wireguard file /etc/wireguard/wg0.conf

# Activate connection
sudo nmcli connection up wg0

# Deactivate connection
sudo nmcli connection down wg0
```

### With systemd-networkd

Create `/etc/systemd/network/99-wg0.netdev`:

```ini
[NetDev]
Name=wg0
Kind=wireguard
Description=WireGuard tunnel wg0

[WireGuard]
PrivateKey=<private_key>
ListenPort=51820

[WireGuardPeer]
PublicKey=<peer_public_key>
AllowedIPs=10.0.0.2/32
Endpoint=peer.example.com:51820
PersistentKeepalive=25
```

Create `/etc/systemd/network/99-wg0.network`:

```ini
[Match]
Name=wg0

[Network]
Address=10.0.0.1/24

[Route]
Gateway=10.0.0.1
Destination=10.0.0.0/24
```

Enable:

```bash
sudo systemctl enable systemd-networkd
sudo systemctl restart systemd-networkd
```

### Cloud Provider Integrations

#### AWS VPC Peering with WireGuard

Connect on-premises network to AWS VPC:

**AWS EC2 Instance Setup**:

```bash
# Launch EC2 instance (Amazon Linux 2 or Ubuntu)
# Security Group: Allow UDP 51820 inbound

# Install WireGuard
sudo amazon-linux-extras install epel -y
sudo yum install wireguard-tools -y

# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf

# Disable source/dest check on EC2 instance (AWS Console)
```

**AWS Configuration** (`/etc/wireguard/wg0.conf`):

```ini
[Interface]
PrivateKey = <aws_private_key>
Address = 10.0.0.1/24
ListenPort = 51820
# Route VPC traffic through tunnel
PostUp = ip route add 172.31.0.0/16 dev wg0
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT
PostUp = iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = <onprem_public_key>
AllowedIPs = 10.0.0.2/32, 192.168.1.0/24
PersistentKeepalive = 25
```

**VPC Route Table**:
- Destination: `192.168.1.0/24`
- Target: WireGuard EC2 instance ENI

**Terraform Example**:

```hcl
resource "aws_instance" "wireguard" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"
  key_name      = var.key_name
  subnet_id     = aws_subnet.public.id

  vpc_security_group_ids = [aws_security_group.wireguard.id]
  source_dest_check      = false

  user_data = <<-EOF
    #!/bin/bash
    apt update
    apt install -y wireguard
    # Configuration deployment...
  EOF

  tags = {
    Name = "wireguard-gateway"
  }
}

resource "aws_security_group" "wireguard" {
  name        = "wireguard-sg"
  description = "WireGuard VPN"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 51820
    to_port     = 51820
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

#### Google Cloud Platform

**GCP Compute Engine Setup**:

```bash
# Create firewall rule
gcloud compute firewall-rules create wireguard-allow \
  --allow=udp:51820 \
  --source-ranges=0.0.0.0/0 \
  --description="Allow WireGuard VPN"

# Create instance
gcloud compute instances create wireguard-gateway \
  --machine-type=e2-micro \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --can-ip-forward \
  --tags=wireguard

# SSH and install
gcloud compute ssh wireguard-gateway
sudo apt update && sudo apt install -y wireguard
```

#### Azure Virtual Network

**Azure VM Setup**:

```bash
# Create VM with Azure CLI
az vm create \
  --resource-group myResourceGroup \
  --name wireguard-vm \
  --image UbuntuLTS \
  --size Standard_B1s \
  --admin-username azureuser \
  --generate-ssh-keys

# Enable IP forwarding
az network nic update \
  --resource-group myResourceGroup \
  --name wireguard-vmVMNic \
  --ip-forwarding true

# Add NSG rule
az network nsg rule create \
  --resource-group myResourceGroup \
  --nsg-name wireguard-vmNSG \
  --name AllowWireGuard \
  --priority 1000 \
  --protocol Udp \
  --destination-port-range 51820 \
  --access Allow
```

### Kubernetes Integration

#### WireGuard as CNI Plugin

Use WireGuard for pod-to-pod encryption:

**Installation with Helm**:

```bash
# Add WireGuard CNI chart
helm repo add wiretrustee https://wiretrustee.github.io/helm-charts
helm repo update

# Install
helm install wireguard-cni wiretrustee/wireguard-cni \
  --namespace kube-system \
  --set subnet=10.244.0.0/16
```

#### DaemonSet Deployment

Deploy WireGuard on all nodes:

```yaml
# wireguard-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: wireguard
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: wireguard
  template:
    metadata:
      labels:
        app: wireguard
    spec:
      hostNetwork: true
      containers:
      - name: wireguard
        image: linuxserver/wireguard:latest
        securityContext:
          privileged: true
          capabilities:
            add:
              - NET_ADMIN
              - SYS_MODULE
        env:
        - name: PUID
          value: "1000"
        - name: PGID
          value: "1000"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: lib-modules
          mountPath: /lib/modules
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: wireguard-config
      - name: lib-modules
        hostPath:
          path: /lib/modules
```

**ConfigMap for WireGuard**:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: wireguard-config
  namespace: kube-system
data:
  wg0.conf: |
    [Interface]
    PrivateKey = <node_private_key>
    Address = 10.0.0.x/24
    ListenPort = 51820

    [Peer]
    PublicKey = <peer_public_key>
    AllowedIPs = 10.0.0.0/24
    Endpoint = peer.example.com:51820
    PersistentKeepalive = 25
```

#### Service Mesh Integration

Integrate with service meshes like Linkerd or Istio:

```yaml
# Pod annotation for WireGuard routing
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  annotations:
    wireguard.io/tunnel: "wg0"
spec:
  containers:
  - name: app
    image: myapp:latest
```

### Mobile Client Management

#### QR Code Generation

Generate QR codes for easy mobile client setup:

```bash
#!/bin/bash
# qr-gen.sh

CLIENT_NAME=$1
CONFIG_FILE="/etc/wireguard/clients/${CLIENT_NAME}.conf"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Install qrencode if needed
if ! command -v qrencode &> /dev/null; then
    sudo apt install -y qrencode
fi

# Generate QR code
qrencode -t ansiutf8 < "$CONFIG_FILE"

# Or save to file
qrencode -t PNG -o "${CLIENT_NAME}.png" < "$CONFIG_FILE"
echo "QR code saved to ${CLIENT_NAME}.png"
```

#### iOS Configuration Profile

Generate iOS configuration profile:

```bash
#!/bin/bash
# ios-profile-gen.sh

CLIENT_NAME=$1
PROFILE_UUID=$(uuidgen)

cat > "${CLIENT_NAME}.mobileconfig" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>PayloadContent</key>
    <array>
        <dict>
            <key>PayloadType</key>
            <string>com.wireguard.ios.config</string>
            <key>PayloadUUID</key>
            <string>${PROFILE_UUID}</string>
            <key>PayloadIdentifier</key>
            <string>com.example.wireguard.${CLIENT_NAME}</string>
            <key>PayloadVersion</key>
            <integer>1</integer>
        </dict>
    </array>
    <key>PayloadDisplayName</key>
    <string>WireGuard - ${CLIENT_NAME}</string>
    <key>PayloadIdentifier</key>
    <string>com.example.wireguard</string>
    <key>PayloadUUID</key>
    <string>${PROFILE_UUID}</string>
    <key>PayloadType</key>
    <string>Configuration</string>
    <key>PayloadVersion</key>
    <integer>1</integer>
</dict>
</plist>
EOF
```

#### Android Client Automation

Automate Android client distribution:

```bash
# Generate Tunnels.zip for WireGuard Android app
zip wireguard-configs.zip /etc/wireguard/clients/*.conf

# Or use wg-quick to generate configs
for client in client1 client2 client3; do
    # Generate config with wg-quick format
    cat > "${client}.conf" <<EOF
[Interface]
PrivateKey = $(wg genkey)
Address = 10.0.0.${i}/24
DNS = 1.1.1.1

[Peer]
PublicKey = ${SERVER_PUBLIC_KEY}
AllowedIPs = 0.0.0.0/0
Endpoint = vpn.example.com:51820
PersistentKeepalive = 25
EOF
done
```

### Automation and Orchestration

#### Ansible Playbook

Deploy WireGuard with Ansible:

```yaml
# wireguard-deploy.yml
---
- name: Deploy WireGuard VPN
  hosts: vpn_servers
  become: yes
  vars:
    wg_interface: wg0
    wg_port: 51820
    wg_network: 10.0.0.0/24

  tasks:
    - name: Install WireGuard
      apt:
        name:
          - wireguard
          - wireguard-tools
        state: present
        update_cache: yes

    - name: Enable IP forwarding
      sysctl:
        name: net.ipv4.ip_forward
        value: '1'
        sysctl_set: yes
        state: present
        reload: yes

    - name: Generate private key
      shell: wg genkey
      register: private_key
      changed_when: false
      no_log: true

    - name: Generate public key
      shell: echo "{{ private_key.stdout }}" | wg pubkey
      register: public_key
      changed_when: false

    - name: Create WireGuard config directory
      file:
        path: /etc/wireguard
        state: directory
        mode: '0700'

    - name: Deploy WireGuard configuration
      template:
        src: templates/wg0.conf.j2
        dest: /etc/wireguard/wg0.conf
        mode: '0600'
      notify: restart wireguard

    - name: Enable WireGuard service
      systemd:
        name: wg-quick@wg0
        enabled: yes
        state: started

  handlers:
    - name: restart wireguard
      systemd:
        name: wg-quick@wg0
        state: restarted
```

**Template** (`templates/wg0.conf.j2`):

```jinja2
[Interface]
PrivateKey = {{ private_key.stdout }}
Address = {{ ansible_default_ipv4.address }}/24
ListenPort = {{ wg_port }}
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

{% for peer in wg_peers %}
[Peer]
PublicKey = {{ peer.public_key }}
AllowedIPs = {{ peer.allowed_ips }}
{% if peer.endpoint is defined %}
Endpoint = {{ peer.endpoint }}
{% endif %}
PersistentKeepalive = 25
{% endfor %}
```

#### Terraform Module

Infrastructure as Code for WireGuard:

```hcl
# modules/wireguard/main.tf
terraform {
  required_providers {
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

resource "random_password" "wg_private_key" {
  length  = 32
  special = true
}

resource "null_resource" "wireguard_keys" {
  provisioner "local-exec" {
    command = <<-EOT
      wg genkey | tee privatekey | wg pubkey > publickey
    EOT
  }
}

output "public_key" {
  value = file("publickey")
}
```

## Migration Guides

### Migrating from OpenVPN

**Comparison and Strategy**:

| Aspect | OpenVPN | WireGuard | Migration Impact |
|--------|---------|-----------|------------------|
| Config Format | .ovpn files | .conf files | Manual conversion needed |
| Certificates | PKI/certs | Public keys | Simplified key management |
| Port | TCP/UDP customizable | UDP only | Firewall rules update |
| Routing | Complex routes | AllowedIPs | Routing logic change |

**Migration Steps**:

1. **Inventory existing OpenVPN setup**:

```bash
# List current OpenVPN clients
ls -l /etc/openvpn/clients/

# Document network topology
cat /etc/openvpn/server.conf | grep -E "server|route|push"
```

2. **Generate WireGuard equivalents**:

```bash
#!/bin/bash
# openvpn-to-wireguard.sh

OPENVPN_SERVER_CONF="/etc/openvpn/server.conf"
WG_CONFIG="/etc/wireguard/wg0.conf"

# Extract OpenVPN network
OVPN_NETWORK=$(grep "^server " "$OPENVPN_SERVER_CONF" | awk '{print $2}')
OVPN_NETMASK=$(grep "^server " "$OPENVPN_SERVER_CONF" | awk '{print $3}')

# Generate WireGuard server config
cat > "$WG_CONFIG" <<EOF
[Interface]
PrivateKey = $(wg genkey)
Address = ${OVPN_NETWORK}/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT
PostDown = iptables -D FORWARD -i %i -j ACCEPT
EOF

echo "WireGuard server config created at $WG_CONFIG"
echo "OpenVPN network: $OVPN_NETWORK/$OVPN_NETMASK"
```

3. **Parallel operation period**:

Run both OpenVPN and WireGuard simultaneously:

```bash
# Keep OpenVPN running
sudo systemctl status openvpn@server

# Start WireGuard on different port
sudo wg-quick up wg0

# Monitor both
watch -n 5 'echo "=== OpenVPN ==="; sudo systemctl status openvpn@server; echo "=== WireGuard ==="; sudo wg show'
```

4. **Gradual client migration**:

```bash
# Migrate one client at a time
# 1. Generate WireGuard config
# 2. Test connection
# 3. Remove from OpenVPN
# 4. Monitor for issues
```

5. **Decommission OpenVPN**:

```bash
# After all clients migrated
sudo systemctl stop openvpn@server
sudo systemctl disable openvpn@server
```

### Migrating from IPsec

**Key Differences**:

- IPsec uses IKE for key exchange; WireGuard uses static keys
- IPsec has multiple modes (transport/tunnel); WireGuard is tunnel-only
- IPsec configuration is complex; WireGuard is simple

**Conversion Example**:

IPsec (strongSwan) configuration:

```
conn site-to-site
    left=192.0.2.1
    leftsubnet=10.1.0.0/16
    right=198.51.100.1
    rightsubnet=10.2.0.0/16
    ike=aes256-sha2_256-modp2048!
    esp=aes256-sha2_256!
    keyexchange=ikev2
    auto=start
```

WireGuard equivalent:

```ini
# Site A
[Interface]
PrivateKey = <site_a_key>
Address = 10.0.0.1/24
ListenPort = 51820

[Peer]
PublicKey = <site_b_key>
AllowedIPs = 10.2.0.0/16, 10.0.0.2/32
Endpoint = 198.51.100.1:51820
```

## Advanced Debugging

### Packet Capture and Analysis

**Capture WireGuard Traffic**:

```bash
# Capture on physical interface (encrypted)
sudo tcpdump -i eth0 -n udp port 51820 -w wireguard-encrypted.pcap

# Capture on WireGuard interface (decrypted)
sudo tcpdump -i wg0 -n -w wireguard-decrypted.pcap

# Analyze with Wireshark
wireshark wireguard-encrypted.pcap
```

**Wireshark Filters**:

```
# WireGuard handshake packets
udp.port == 51820 && udp.length == 148

# WireGuard data packets
udp.port == 51820 && udp.length > 148

# Filter by endpoint
ip.addr == 203.0.113.1 && udp.port == 51820
```

### eBPF Tracing

Trace WireGuard kernel operations with eBPF:

```bash
# Install bpftrace
sudo apt install bpftrace

# Trace WireGuard packet processing
sudo bpftrace -e '
kprobe:wg_packet_receive {
    printf("RX packet on wg, size: %d\n", arg1);
}

kprobe:wg_packet_send {
    printf("TX packet on wg, size: %d\n", arg1);
}
'

# Trace handshakes
sudo bpftrace -e '
kprobe:wg_noise_handshake_create_initiation {
    printf("Initiating handshake\n");
}

kprobe:wg_noise_handshake_consume_response {
    printf("Consuming handshake response\n");
}
'
```

### Performance Profiling

**CPU Profiling**:

```bash
# Use perf to profile WireGuard
sudo perf record -g -p $(pgrep kworker/.*wg-crypt)
# Generate load, then:
sudo perf report

# Profile specific functions
sudo perf record -e cycles -g -- sleep 30
sudo perf report --sort comm,dso,symbol | grep wireguard
```

**Latency Measurement**:

```bash
#!/bin/bash
# wg-latency-test.sh

PEER_IP="10.0.0.2"
SAMPLES=1000

echo "Testing WireGuard latency ($SAMPLES samples)..."

ping -c "$SAMPLES" -i 0.01 "$PEER_IP" | tee ping-results.txt

# Calculate statistics
awk '/^rtt/ {
    split($4, values, "/");
    printf "Min: %.2f ms\n", values[1];
    printf "Avg: %.2f ms\n", values[2];
    printf "Max: %.2f ms\n", values[3];
    printf "StdDev: %.2f ms\n", values[4];
}' ping-results.txt
```

**Throughput Testing**:

```bash
# Server side (run iperf3 server through tunnel)
iperf3 -s -B 10.0.0.1

# Client side
iperf3 -c 10.0.0.1 -t 60 -i 1

# Bidirectional test
iperf3 -c 10.0.0.1 -t 60 -i 1 --bidir

# UDP throughput
iperf3 -c 10.0.0.1 -t 60 -u -b 1G
```

### Kernel Debugging

**Enable debug logging**:

```bash
# Check current debug level
cat /sys/kernel/debug/dynamic_debug/control | grep wireguard

# Enable all WireGuard debug messages
echo 'module wireguard +p' | sudo tee /sys/kernel/debug/dynamic_debug/control

# View logs
sudo dmesg -w | grep wireguard

# Disable when done
echo 'module wireguard -p' | sudo tee /sys/kernel/debug/dynamic_debug/control
```

**Trace route debugging**:

```bash
# Check routing for specific destination
ip route get 10.0.0.2

# Show all routes in WireGuard table
ip route show table all | grep wg0

# Trace packet path
sudo mtr --report 10.0.0.2

# Check policy routing
ip rule show
```

## Comparison with Other VPN Solutions

| Feature | WireGuard | OpenVPN | IPsec |
|---------|-----------|---------|-------|
| Lines of Code | ~4,000 | ~100,000 | ~400,000 |
| Performance | Excellent | Good | Good |
| Setup Complexity | Simple | Medium | Complex |
| Cryptography | Modern, fixed | Configurable | Configurable |
| Kernel Integration | Yes (Linux 5.6+) | No | Yes |
| Roaming | Seamless | Requires reconnect | Requires reconnect |
| NAT Traversal | Excellent | Good | Challenging |
| CPU Usage | Low | Medium | Medium-High |

## Quick Reference

### Essential Commands

```bash
# Key generation
wg genkey | tee privatekey | wg pubkey > publickey

# Interface management
sudo wg-quick up wg0
sudo wg-quick down wg0

# Show status
sudo wg show
sudo wg show wg0

# Add peer
sudo wg set wg0 peer <pubkey> allowed-ips 10.0.0.2/32

# Remove peer
sudo wg set wg0 peer <pubkey> remove

# Reload configuration
sudo wg syncconf wg0 <(wg-quick strip wg0)
```

### Configuration Template

```ini
[Interface]
PrivateKey = <base64_private_key>
Address = <interface_ip>/24
ListenPort = 51820
DNS = 1.1.1.1

[Peer]
PublicKey = <base64_public_key>
AllowedIPs = <allowed_cidr>
Endpoint = <hostname_or_ip>:51820
PersistentKeepalive = 25
```

## Useful Resources

### Official Documentation
- [WireGuard Official Site](https://www.wireguard.com/)
- [WireGuard Kernel Documentation](https://www.kernel.org/doc/html/latest/networking/wireguard.html)
- [WireGuard Whitepaper](https://www.wireguard.com/papers/wireguard.pdf)

### Tools and Utilities
- [wg-quick](https://git.zx2c4.com/wireguard-tools/about/src/man/wg-quick.8) - Configuration management tool
- [wg](https://git.zx2c4.com/wireguard-tools/about/src/man/wg.8) - Low-level configuration tool
- [wireguard-tools](https://git.zx2c4.com/wireguard-tools/) - Official command-line tools

### Community Resources
- [WireGuard Mailing List](https://lists.zx2c4.com/mailman/listinfo/wireguard)
- [WireGuard Subreddit](https://www.reddit.com/r/WireGuard/)
- [Awesome WireGuard](https://github.com/cedrickchee/awesome-wireguard) - Curated resources

## Troubleshooting Checklist

- [ ] Private/public keys correctly generated and configured
- [ ] Firewall allows UDP traffic on WireGuard port
- [ ] IP forwarding enabled on server
- [ ] AllowedIPs correctly configured for both directions
- [ ] Endpoint reachable and correct
- [ ] PersistentKeepalive set for clients behind NAT
- [ ] MTU configured appropriately
- [ ] No IP conflicts with existing networks
- [ ] WireGuard kernel module loaded
- [ ] Correct permissions on configuration files (600)
