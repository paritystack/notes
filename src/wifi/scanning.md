# Wi-Fi Scanning

Wi-Fi scanning is the process of identifying available wireless networks within range of a Wi-Fi-enabled device. This process is essential for connecting to Wi-Fi networks, troubleshooting connectivity issues, and optimizing network performance. Wi-Fi scanning can be performed using various tools and techniques, and it typically involves the following steps:

1. **Initiate Scan**:
   The Wi-Fi-enabled device sends out probe request frames to discover available networks. These frames are broadcasted on different channels to ensure that all nearby networks are detected.

2. **Receive Probe Responses**:
   Access points (APs) within range respond to the probe request frames with probe response frames. These frames contain information about the network, such as the Service Set Identifier (SSID), supported data rates, security protocols, and other capabilities.

3. **Analyze Beacon Frames**:
   In addition to probe responses, the device can also listen for beacon frames that are periodically broadcasted by APs. Beacon frames contain similar information to probe responses and help the device identify available networks.

4. **Compile Network List**:
   The device compiles a list of available networks based on the received probe responses and beacon frames. This list includes details such as the SSID, signal strength (RSSI), channel, and security type of each network.

5. **Select Network**:
   The user or device selects a network from the list to connect to. The selection can be based on various factors, such as signal strength, network name, or security requirements.

### Scanning Modes and Technical Details

#### Active vs. Passive Scanning

- **Active Scanning**: The device actively sends probe request frames on each channel and waits for probe responses from APs. This method is faster and more reliable but reveals the device's presence and can expose privacy information through probe requests.

- **Passive Scanning**: The device listens for beacon frames broadcasted by APs without sending any probe requests. This method is slower but more stealthy and preserves privacy. Passive scanning is commonly used in monitoring mode.

#### Frequency Bands and Channels

Modern Wi-Fi operates on multiple frequency bands:

- **2.4 GHz Band**: Channels 1-14 (channel availability varies by country). More crowded but better range and wall penetration.
- **5 GHz Band**: Multiple channels (36, 40, 44, 48, 149, 153, 157, 161, 165, etc.). Less congested with higher throughput but shorter range.
- **6 GHz Band (Wi-Fi 6E)**: Channels from 1-233. Newest band with the most available spectrum and least interference.

During scanning, the device typically hops through channels in sequence to discover all available networks across different bands.

#### Scan Types

- **Broadcast Probe Request**: Sent with an empty SSID field to discover all networks within range.
- **Directed Probe Request**: Sent with a specific SSID to discover a particular network. Used when connecting to known networks or hidden SSIDs.

### Tools for Wi-Fi Scanning

#### GUI Tools

- **Wireshark**: A network protocol analyzer that can capture and analyze Wi-Fi packets, including probe requests, probe responses, and beacon frames.
- **NetSpot**: A Wi-Fi survey and analysis tool that provides detailed information about available networks, including signal strength, channel usage, and security settings.
- **inSSIDer**: A Wi-Fi scanner that displays information about nearby networks, such as SSID, signal strength, channel, and security type.
- **Acrylic Wi-Fi**: A Wi-Fi scanner and analyzer that provides real-time information about available networks, including signal strength, channel usage, and network performance metrics.
- **WiFi Analyzer** (Mobile): Popular Android app for visualizing Wi-Fi networks, signal strength, and channel overlap.
- **Network Analyzer** (Mobile): iOS and Android app for scanning and analyzing Wi-Fi networks.

#### Command-Line Tools

- **iwlist**: Legacy Linux tool for scanning wireless networks and displaying detailed information about available APs.
- **iw**: Modern replacement for iwlist with more features and better support for newer Wi-Fi standards.
- **nmcli**: NetworkManager command-line interface that can scan and manage Wi-Fi connections on Linux systems.
- **wpa_cli**: Command-line interface for wpa_supplicant, useful for scanning and connecting to networks.
- **airodump-ng**: Part of the Aircrack-ng suite, used for packet capture and network analysis in monitor mode.
- **netsh** (Windows): Built-in Windows command for managing and scanning wireless networks.

#### Programming Libraries and Frameworks

- **Scapy** (Python): Powerful packet manipulation library that can be used to craft custom scanning tools and analyze Wi-Fi traffic.
- **pywifi** (Python): Simple Python library for controlling Wi-Fi interfaces and scanning networks.
- **CoreWLAN** (macOS): Native macOS framework for Wi-Fi scanning and management.
- **WlanAPI** (Windows): Native Windows API for wireless network scanning and management.

### Code Examples

#### Linux - Using iwlist

```bash
# Scan for available networks (requires root/sudo)
sudo iwlist wlan0 scan

# Scan and filter for specific information
sudo iwlist wlan0 scan | grep -E "ESSID|Quality|Encryption"

# Get brief scan results
sudo iwlist wlan0 scan | grep ESSID
```

#### Linux - Using iw

```bash
# Scan for networks (modern tool)
sudo iw dev wlan0 scan

# Scan and show only SSIDs and signal strength
sudo iw dev wlan0 scan | grep -E "SSID|signal"

# Scan specific frequency band (5GHz)
sudo iw dev wlan0 scan freq 5180 5240 5320
```

#### Linux - Using nmcli

```bash
# Scan and list all available networks
nmcli device wifi list

# Rescan for networks
nmcli device wifi rescan

# List with specific columns
nmcli -f SSID,SIGNAL,SECURITY device wifi list

# Connect to a network
nmcli device wifi connect "SSID_Name" password "your_password"
```

#### Windows - Using netsh

```cmd
# Display available wireless networks
netsh wlan show networks

# Show detailed network information including security type
netsh wlan show networks mode=bssid

# Show wireless interface information
netsh wlan show interfaces
```

#### Python - Using Scapy

```python
from scapy.all import *

def scan_wifi():
    """Scan for Wi-Fi networks using Scapy"""
    # Sniff for beacon frames
    packets = sniff(iface="wlan0", count=100,
                   filter="type mgt subtype beacon")

    networks = set()
    for packet in packets:
        if packet.haslayer(Dot11Beacon):
            # Extract SSID
            ssid = packet[Dot11Elt].info.decode('utf-8', errors='ignore')
            # Extract MAC address
            bssid = packet[Dot11].addr2
            # Extract channel
            channel = int(ord(packet[Dot11Elt:3].info))

            networks.add((ssid, bssid, channel))

    for ssid, bssid, channel in sorted(networks):
        print(f"SSID: {ssid:30} BSSID: {bssid} Channel: {channel}")

# Note: Requires monitor mode
# sudo ip link set wlan0 down
# sudo iw wlan0 set monitor none
# sudo ip link set wlan0 up
```

#### Shell Script - Automated Scanning

```bash
#!/bin/bash
# wifi_scanner.sh - Automated Wi-Fi scanning script

INTERFACE="wlan0"
OUTPUT_FILE="wifi_scan_$(date +%Y%m%d_%H%M%S).txt"

echo "Wi-Fi Scan Report - $(date)" > "$OUTPUT_FILE"
echo "================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Perform scan
nmcli -f SSID,SIGNAL,CHAN,SECURITY device wifi list | while read -r line; do
    echo "$line" >> "$OUTPUT_FILE"
done

echo "" >> "$OUTPUT_FILE"
echo "Scan completed. Results saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
```

### Security Considerations

Wi-Fi scanning involves several important security aspects that users and administrators should be aware of:

#### Privacy Concerns

- **Probe Request Leakage**: When devices perform active scanning, they broadcast probe requests that may contain previously connected SSIDs, potentially revealing location history and network associations.
- **MAC Address Tracking**: Each probe request contains the device's MAC address, which can be used to track device movement across different locations.
- **MAC Randomization**: Modern devices implement MAC address randomization during scanning to enhance privacy by using random MAC addresses instead of the device's true hardware address.

#### Security Protocols Detection

When scanning, it's important to identify the security protocols used by networks:

- **Open Networks**: No encryption - all traffic is sent in plaintext and can be easily intercepted.
- **WEP (Wired Equivalent Privacy)**: Deprecated and easily crackable - should never be used.
- **WPA/WPA2-PSK**: Common personal network security using pre-shared keys. WPA2 is secure with strong passwords.
- **WPA2-Enterprise**: Uses 802.1X authentication with RADIUS servers - more secure for organizational use.
- **WPA3**: Latest security protocol with improved encryption (SAE) and protection against offline dictionary attacks.
- **WPA3-Enterprise**: Enhanced security for enterprise networks with 192-bit encryption option.

#### Rogue Access Point Detection

Scanning can help identify rogue or malicious access points:

- **Evil Twin Attacks**: Malicious APs mimicking legitimate networks with the same SSID.
- **Unauthorized APs**: Devices connected to the network creating unauthorized wireless access points.
- **Detection Methods**: Regular scanning to maintain an inventory of authorized APs and identify unknown devices.

#### Best Practices for Secure Scanning

1. **Use Passive Scanning** when possible to minimize information disclosure.
2. **Enable MAC Randomization** on your devices to protect against tracking.
3. **Disable Auto-Connect** to prevent automatic connection to potentially malicious networks.
4. **Monitor for Rogue APs** regularly in enterprise environments.
5. **Verify Network Authenticity** before connecting, especially in public spaces.
6. **Use VPN** when connecting to untrusted networks, even after careful scanning.

### Importance of Wi-Fi Scanning

Wi-Fi scanning is crucial for several reasons:

- **Network Discovery**: It allows users to discover available networks and choose the best one to connect to.
- **Troubleshooting**: It helps identify connectivity issues, such as weak signals, interference, or misconfigured settings.
- **Optimization**: It provides insights into network performance and helps optimize the configuration, such as selecting the best channel to minimize interference.
- **Security Auditing**: It helps identify unauthorized or rogue access points, detect security protocol weaknesses, and ensure network infrastructure compliance.
- **Site Surveys**: Essential for planning Wi-Fi deployments, identifying coverage gaps, and optimizing AP placement.

By understanding and utilizing Wi-Fi scanning techniques, users and network administrators can ensure reliable, efficient, and secure wireless connectivity.

