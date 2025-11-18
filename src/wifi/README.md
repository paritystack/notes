# WiFi (Wireless Networking)

Comprehensive documentation on WiFi technology, protocols, configuration, and troubleshooting.

## Overview

WiFi is a family of wireless network protocols based on the IEEE 802.11 standards. It enables devices to connect to networks and the internet wirelessly, forming the backbone of modern wireless communication.

## Table of Contents

### Core Topics

- **[WiFi Basics](basics.md)** - Fundamental concepts and how WiFi works
  - Radio frequencies and channels
  - Access points and clients
  - Network topologies (infrastructure vs ad-hoc)
  - WiFi architecture and components

- **[WiFi Standards](standards.md)** - Evolution of IEEE 802.11 protocols
  - **802.11a/b/g** - Legacy standards (5GHz, 2.4GHz, mixed)
  - **802.11n (WiFi 4)** - MIMO, 40MHz channels, up to 600 Mbps
  - **802.11ac (WiFi 5)** - MU-MIMO, 160MHz channels, up to 6.9 Gbps
  - **802.11ax (WiFi 6/6E)** - OFDMA, 1024-QAM, improved efficiency
  - **802.11be (WiFi 7)** - Next generation, up to 46 Gbps
  - Frequency bands, channel widths, and data rates
  - Backward compatibility considerations

- **[WiFi Security](security.md)** - Authentication and encryption protocols
  - **WEP** - Deprecated, insecure
  - **WPA/WPA2** - TKIP and CCMP/AES encryption
  - **WPA3** - Modern security with SAE, enhanced open
  - **Enterprise Security** - 802.1X, RADIUS, EAP methods
  - Best practices for secure WiFi deployment
  - Common vulnerabilities and mitigations

### Advanced Topics

- **[Scanning](scanning.md)** - Network discovery mechanisms
  - Passive scanning (beacon frames)
  - Active scanning (probe request/response)
  - Channel scanning strategies
  - Hidden SSID handling
  - Background vs foreground scanning

- **[Roaming](roaming.md)** - Seamless handoff between access points
  - Basic service set (BSS) transitions
  - Fast roaming (802.11r FT)
  - Opportunistic key caching (OKC)
  - 802.11k (neighbor reports)
  - 802.11v (BSS transition management)
  - Roaming decision algorithms

- **[QoS Management](qos_management.md)** - Quality of service prioritization
  - WMM (WiFi Multimedia)
  - Access categories (voice, video, best effort, background)
  - EDCA (Enhanced Distributed Channel Access)
  - Traffic prioritization and scheduling
  - Latency-sensitive application support

- **[OFDMA](ofdma.md)** - Orthogonal Frequency Division Multiple Access
  - Resource units (RUs) in WiFi 6/7
  - Multi-user efficiency improvements
  - Uplink and downlink OFDMA
  - Comparison with OFDM
  - Performance benefits in dense environments

## WiFi Architecture

### Network Components

```
┌─────────────────────────────────────────┐
│        WiFi Network Architecture        │
├─────────────────────────────────────────┤
│                                         │
│  Internet ◄──► Router/Gateway           │
│                    │                    │
│                    ▼                    │
│              Access Point(s)            │
│              ┌───┴───┐                  │
│              │       │                  │
│         ┌────▼─┐  ┌──▼───┐             │
│         │ STA  │  │ STA  │  (Clients)  │
│         └──────┘  └──────┘             │
│                                         │
└─────────────────────────────────────────┘
```

**Components:**
- **STA (Station)**: WiFi client device (laptop, phone, IoT device)
- **AP (Access Point)**: Bridge between wireless and wired networks
- **BSS (Basic Service Set)**: One AP and its associated clients
- **ESS (Extended Service Set)**: Multiple APs with the same SSID
- **Distribution System (DS)**: Backend network connecting APs

### Frequency Bands

| Band | Frequency | Channels | Range | Speed | Interference |
|------|-----------|----------|-------|-------|--------------|
| 2.4 GHz | 2.400-2.495 GHz | 1-13 (11 in US) | Longer | Lower | Higher |
| 5 GHz | 5.150-5.825 GHz | ~24 non-overlapping | Shorter | Higher | Lower |
| 6 GHz (WiFi 6E) | 5.925-7.125 GHz | 59 channels | Shortest | Highest | Lowest |

**2.4 GHz:**
- Better penetration through walls
- Longer range
- More crowded (Bluetooth, microwaves, other devices)
- 3 non-overlapping channels (1, 6, 11)

**5 GHz:**
- Less interference
- More available channels
- Higher speeds
- Shorter range

**6 GHz (WiFi 6E):**
- Clean spectrum, no legacy devices
- Very wide channels (160 MHz, 320 MHz in WiFi 7)
- Ultra-low latency
- Requires WiFi 6E compatible hardware

## Key Technologies

### MIMO (Multiple-Input Multiple-Output)

**Single-User MIMO (SU-MIMO):**
- Multiple antennas for spatial streams
- Increases throughput to single client
- WiFi 4 (802.11n) and later

**Multi-User MIMO (MU-MIMO):**
- Simultaneous transmission to multiple clients
- Downlink in WiFi 5, uplink + downlink in WiFi 6
- Up to 8 spatial streams

### Beamforming

- Focuses signal toward specific clients
- Improves SNR (Signal-to-Noise Ratio)
- Better range and reliability
- Explicit beamforming in 802.11ac+

### Channel Bonding

- Combines multiple channels for higher bandwidth
- 20 MHz, 40 MHz, 80 MHz, 160 MHz, 320 MHz (WiFi 7)
- Trade-off: Higher speed vs compatibility and range

## Configuration

### Common Tools

**Linux:**
- `iw` - Modern wireless configuration
- `iwconfig` - Legacy wireless tools
- `wpa_supplicant` - Client authentication daemon
- `hostapd` - Access point daemon
- `nmcli` - NetworkManager CLI

**Windows:**
- `netsh wlan` - Command-line configuration
- Windows Settings GUI
- WiFi analyzer tools

**macOS:**
- System Preferences
- `networksetup` - Command-line tool
- `airport` - Diagnostic utility

### Example: Connect to WiFi (Linux)

```bash
# Scan for networks
sudo iw dev wlan0 scan | grep SSID

# Connect using wpa_supplicant
wpa_passphrase "SSID" "password" | sudo tee -a /etc/wpa_supplicant/wpa_supplicant.conf
sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf

# Get IP address
sudo dhclient wlan0

# Or use NetworkManager
nmcli dev wifi connect "SSID" password "password"
```

## Troubleshooting

### Common Issues

**Weak Signal:**
- Move closer to access point
- Remove physical obstacles
- Switch to less congested channel
- Enable band steering (prefer 5GHz)
- Add additional access points

**Slow Speeds:**
- Check channel congestion
- Verify WiFi standard capabilities
- Update firmware and drivers
- Disable legacy 802.11b/g devices
- Use wider channels (if supported and clean)

**Connection Drops:**
- Check for interference sources
- Update access point firmware
- Adjust roaming thresholds
- Verify power management settings
- Check for IP conflicts

**Can't Connect:**
- Verify SSID and password
- Check security protocol compatibility
- Ensure MAC filtering is disabled (or device is allowed)
- Reset network settings
- Check DHCP availability

### Diagnostic Commands

```bash
# Check WiFi interface status
iw dev wlan0 info

# View link quality and signal strength
iw dev wlan0 link

# Scan for nearby networks
iw dev wlan0 scan

# Monitor signal quality
watch -n 1 iw dev wlan0 station dump

# Check channel utilization
iw dev wlan0 survey dump

# View connection logs
journalctl -u wpa_supplicant
```

## Performance Optimization

### Best Practices

1. **Channel Selection:**
   - Use WiFi analyzer to find least congested channels
   - 2.4 GHz: Use channels 1, 6, or 11
   - 5 GHz: Use DFS channels if available
   - 6 GHz: Leverage clean spectrum

2. **Channel Width:**
   - 2.4 GHz: 20 MHz only (avoid 40 MHz)
   - 5 GHz: 80 MHz or 160 MHz if supported
   - Balance between speed and compatibility

3. **Access Point Placement:**
   - Central location for coverage
   - Elevated position
   - Away from walls and metal objects
   - Minimize interference sources

4. **Security:**
   - Use WPA3 (or WPA2 minimum)
   - Strong passwords (>12 characters)
   - Disable WPS
   - Separate guest network
   - Regular firmware updates

5. **Network Management:**
   - Disable unused networks (2.4 GHz if not needed)
   - Enable band steering
   - Configure roaming thresholds
   - Monitor connected devices
   - Use QoS for latency-sensitive traffic

## WiFi Standards Comparison

| Standard | Name | Year | Band | Max Speed | Key Features |
|----------|------|------|------|-----------|--------------|
| 802.11a | - | 1999 | 5 GHz | 54 Mbps | OFDM |
| 802.11b | - | 1999 | 2.4 GHz | 11 Mbps | DSSS |
| 802.11g | - | 2003 | 2.4 GHz | 54 Mbps | OFDM |
| 802.11n | WiFi 4 | 2009 | 2.4/5 GHz | 600 Mbps | MIMO, 40 MHz |
| 802.11ac | WiFi 5 | 2014 | 5 GHz | 6.9 Gbps | MU-MIMO, 160 MHz |
| 802.11ax | WiFi 6 | 2019 | 2.4/5 GHz | 9.6 Gbps | OFDMA, TWT, BSS coloring |
| 802.11ax | WiFi 6E | 2020 | 6 GHz | 9.6 Gbps | 6 GHz spectrum |
| 802.11be | WiFi 7 | 2024 | 2.4/5/6 GHz | 46 Gbps | 320 MHz, 4096-QAM, MLO |

## Use Cases

### Home Networking
- Internet browsing and streaming
- Smart home devices (IoT)
- Gaming (prefer 5 GHz or wired)
- Video conferencing

### Enterprise
- High-density deployments
- Seamless roaming across buildings
- Guest access (isolated network)
- VoIP and video conferencing
- Location services

### Public WiFi
- Hotspots (cafes, airports)
- Captive portals for authentication
- Bandwidth management
- Security considerations

### Industrial IoT
- Sensor networks
- Real-time monitoring
- Low-latency requirements
- Mesh networking

## Security Considerations

### Threats
- **Eavesdropping**: Intercepting wireless traffic
- **Evil Twin**: Rogue access points mimicking legitimate ones
- **Man-in-the-Middle**: Intercepting and modifying traffic
- **Deauthentication Attacks**: Forcing clients to disconnect
- **WPS Brute Force**: Exploiting WPS PIN vulnerability

### Mitigations
- Use WPA3 encryption
- Disable WPS
- Strong, unique passwords
- Regular firmware updates
- MAC address filtering (defense in depth)
- Network segmentation (VLANs)
- Monitor for rogue access points
- Use VPN for sensitive traffic on public WiFi

## Related Topics

- **[Linux Networking](../linux/networking.md)** - Linux network configuration
- **[cfg80211 & mac80211](../linux/cfg80211_mac80211.md)** - Linux WiFi subsystem
- **[WireGuard](../linux/wireguard.md)** - VPN for secure connections
- **[Networking Protocols](../networking/)** - General networking concepts

## Resources

### Tools
- **WiFi Analyzers**: Wireshark, Kismet, inSSIDer
- **Speed Tests**: Ookla, Fast.com, iPerf
- **Site Survey**: Ekahau, NetSpot
- **Configuration**: wpa_supplicant, hostapd, NetworkManager

### Standards Organizations
- IEEE 802.11 Working Group
- Wi-Fi Alliance (certification)
- IETF (related protocols)

### Further Reading
- IEEE 802.11 specifications
- Wi-Fi Alliance whitepapers
- Wireless networking textbooks
- Security best practice guides

---

WiFi technology continues to evolve with each new standard, delivering faster speeds, lower latency, and better efficiency in crowded environments. Understanding these fundamentals helps optimize wireless networks for reliability and performance.
