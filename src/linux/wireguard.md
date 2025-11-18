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
