# Wifi Basics


## Aggregation

Aggregation in Wi-Fi refers to the process of combining multiple data frames into a single transmission unit. This technique is used to improve the efficiency and throughput of wireless networks by reducing the overhead associated with each individual frame transmission. There are two main types of aggregation in Wi-Fi:

1. **A-MPDU (Aggregated MAC Protocol Data Unit)**:
   - Combines multiple MAC frames into a single PHY (Physical Layer) frame.
   - Reduces the inter-frame spacing and acknowledgment overhead.
   - Improves throughput by allowing multiple frames to be sent in a single transmission burst.

2. **A-MSDU (Aggregated MAC Service Data Unit)**:
   - Combines multiple MSDUs (MAC Service Data Units) into a single MPDU (MAC Protocol Data Unit).
   - Reduces the overhead by aggregating data at the MAC layer before it is passed to the PHY layer.
   - Increases efficiency by reducing the number of headers and acknowledgments required.

Both A-MPDU and A-MSDU are supported in 802.11n and later standards, such as 802.11ac and 802.11ax. These aggregation techniques are particularly beneficial in high-throughput and high-density environments, where they help to maximize the use of available bandwidth and improve overall network performance.

## Wifi Bands

### 2.4 GHz
- 802.11a
- 802.11b
- 802.11g

The 2.4 GHz band is one of the most commonly used frequency bands for Wi-Fi communication. It is known for its longer range and better penetration through obstacles such as walls and floors. However, it is also more susceptible to interference from other devices, such as microwaves, cordless phones, and Bluetooth devices, which operate in the same frequency range.

### Channels in 2.4 GHz Band
The 2.4 GHz band is divided into multiple channels, each with a specific frequency range. The channels are spaced 5 MHz apart, but due to the width of the channels (22 MHz), there is significant overlap between adjacent channels. This can lead to interference if multiple networks are operating on overlapping channels. The commonly used channels in the 2.4 GHz band are:

- **Channel 1**: 2.412 GHz
- **Channel 2**: 2.417 GHz
- **Channel 3**: 2.422 GHz
- **Channel 4**: 2.427 GHz
- **Channel 5**: 2.432 GHz
- **Channel 6**: 2.437 GHz
- **Channel 7**: 2.442 GHz
- **Channel 8**: 2.447 GHz
- **Channel 9**: 2.452 GHz
- **Channel 10**: 2.457 GHz
- **Channel 11**: 2.462 GHz

In some regions, additional channels are available:

- **Channel 12**: 2.467 GHz
- **Channel 13**: 2.472 GHz
- **Channel 14**: 2.484 GHz (only available in Japan)

To minimize interference, it is recommended to use non-overlapping channels. In the 2.4 GHz band, the non-overlapping channels are typically channels 1, 6, and 11. By configuring Wi-Fi networks to operate on these channels, interference can be reduced, leading to improved performance and reliability.


### 5 GHz
- 802.11a
- 802.11n
- 802.11ac
- 802.11ax

### Channels in 5 GHz Band
The 5 GHz band offers a larger number of channels compared to the 2.4 GHz band, which helps to reduce interference and congestion. The channels in the 5 GHz band are spaced 20 MHz apart, and there are several non-overlapping channels available. This band is divided into several sub-bands, each with its own set of channels:

- **UNII-1 (5150-5250 MHz)**:
  - Channel 36: 5.180 GHz
  - Channel 40: 5.200 GHz
  - Channel 44: 5.220 GHz
  - Channel 48: 5.240 GHz

- **UNII-2 (5250-5350 MHz)**:
  - Channel 52: 5.260 GHz
  - Channel 56: 5.280 GHz
  - Channel 60: 5.300 GHz
  - Channel 64: 5.320 GHz

- **UNII-2 Extended (5470-5725 MHz)**:
  - Channel 100: 5.500 GHz
  - Channel 104: 5.520 GHz
  - Channel 108: 5.540 GHz
  - Channel 112: 5.560 GHz
  - Channel 116: 5.580 GHz
  - Channel 120: 5.600 GHz
  - Channel 124: 5.620 GHz
  - Channel 128: 5.640 GHz
  - Channel 132: 5.660 GHz
  - Channel 136: 5.680 GHz
  - Channel 140: 5.700 GHz
  - Channel 144: 5.720 GHz

- **UNII-3 (5725-5850 MHz)**:
  - Channel 149: 5.745 GHz
  - Channel 153: 5.765 GHz
  - Channel 157: 5.785 GHz
  - Channel 161: 5.805 GHz
  - Channel 165: 5.825 GHz

The 5 GHz band is less crowded than the 2.4 GHz band and offers higher data rates and lower latency. However, it has a shorter range and less ability to penetrate obstacles such as walls and floors. The use of non-overlapping channels in the 5 GHz band helps to minimize interference and improve overall network performance. Additionally, Dynamic Frequency Selection (DFS) is used in some channels to avoid interference with radar systems.


### 6 GHz
- 802.11ax
- 802.11be

### Channels in 6 GHz Band
The 6 GHz band is a new addition to the Wi-Fi spectrum, providing even more channels and bandwidth for wireless communication. This band is divided into several sub-bands, each with its own set of channels. The channels in the 6 GHz band are spaced 20 MHz apart, similar to the 5 GHz band, and there are numerous non-overlapping channels available. The 6 GHz band offers higher data rates, lower latency, and reduced interference compared to the 2.4 GHz and 5 GHz bands.

- **UNII-5 (5925-6425 MHz)**:
  - Channel 1: 5.925 GHz
  - Channel 5: 5.945 GHz
  - Channel 9: 5.965 GHz
  - Channel 13: 5.985 GHz
  - Channel 17: 6.005 GHz
  - Channel 21: 6.025 GHz
  - Channel 25: 6.045 GHz
  - Channel 29: 6.065 GHz
  - Channel 33: 6.085 GHz
  - Channel 37: 6.105 GHz
  - Channel 41: 6.125 GHz
  - Channel 45: 6.145 GHz
  - Channel 49: 6.165 GHz
  - Channel 53: 6.185 GHz
  - Channel 57: 6.205 GHz
  - Channel 61: 6.225 GHz
  - Channel 65: 6.245 GHz
  - Channel 69: 6.265 GHz
  - Channel 73: 6.285 GHz
  - Channel 77: 6.305 GHz
  - Channel 81: 6.325 GHz
  - Channel 85: 6.345 GHz
  - Channel 89: 6.365 GHz
  - Channel 93: 6.385 GHz
  - Channel 97: 6.405 GHz
  - Channel 101: 6.425 GHz

- **UNII-6 (6425-6525 MHz)**:
  - Channel 105: 6.445 GHz
  - Channel 109: 6.465 GHz
  - Channel 113: 6.485 GHz
  - Channel 117: 6.505 GHz
  - Channel 121: 6.525 GHz

- **UNII-7 (6525-6875 MHz)**:
  - Channel 125: 6.545 GHz
  - Channel 129: 6.565 GHz
  - Channel 133: 6.585 GHz
  - Channel 137: 6.605 GHz
  - Channel 141: 6.625 GHz
  - Channel 145: 6.645 GHz
  - Channel 149: 6.665 GHz
  - Channel 153: 6.685 GHz
  - Channel 157: 6.705 GHz
  - Channel 161: 6.725 GHz
  - Channel 165: 6.745 GHz
  - Channel 169: 6.765 GHz
  - Channel 173: 6.785 GHz
  - Channel 177: 6.805 GHz
  - Channel 181: 6.825 GHz
  - Channel 185: 6.845 GHz
  - Channel 189: 6.865 GHz
  - Channel 193: 6.885 GHz
  - Channel 197: 6.905 GHz
  - Channel 201: 6.925 GHz
  - Channel 205: 6.945 GHz
  - Channel 209: 6.965 GHz
  - Channel 213: 6.985 GHz

- **UNII-8 (6875-7125 MHz)**:
  - Channel 217: 7.005 GHz
  - Channel 221: 7.025 GHz
  - Channel 225: 7.045 GHz
  - Channel 229: 7.065 GHz
  - Channel 233: 7.085 GHz
  - Channel 237: 7.105 GHz
  - Channel 241: 7.125 GHz

The 6 GHz band is expected to significantly enhance Wi-Fi performance, especially in dense environments, by providing more spectrum and reducing congestion. Devices that support the 6 GHz band can take advantage of these additional channels to achieve faster speeds and more reliable connections.


## Wifi channel width

Wi-Fi channel width refers to the size of the frequency band that a Wi-Fi signal occupies. The channel width determines the data rate and the amount of data that can be transmitted over the network. Wider channels can carry more data, but they are also more susceptible to interference and congestion. The most common channel widths in Wi-Fi are 20 MHz, 40 MHz, 80 MHz, and 160 MHz.

### 20 MHz Channels
20 MHz is the standard channel width for Wi-Fi and is widely used in both 2.4 GHz and 5 GHz bands. It provides a good balance between range and throughput. A 20 MHz channel is less likely to experience interference from other devices and networks, making it a reliable choice for most applications.

### 40 MHz Channels
40 MHz channels are used to increase the data rate by bonding two adjacent 20 MHz channels. This effectively doubles the bandwidth, allowing for higher throughput. However, 40 MHz channels are more prone to interference, especially in the crowded 2.4 GHz band. In the 5 GHz band, 40 MHz channels are more practical due to the availability of more non-overlapping channels.

### 80 MHz Channels
80 MHz channels further increase the data rate by bonding four adjacent 20 MHz channels. This provides even higher throughput, making it suitable for applications that require high data rates, such as HD video streaming and online gaming. However, 80 MHz channels are more susceptible to interference and are typically used in the 5 GHz and 6 GHz bands where more spectrum is available.

### 160 MHz Channels
160 MHz channels offer the highest data rates by bonding eight adjacent 20 MHz channels. This channel width is ideal for applications that demand extremely high throughput, such as virtual reality (VR) and large file transfers. However, 160 MHz channels are highly susceptible to interference and are only practical in the 5 GHz and 6 GHz bands with sufficient spectrum availability.

### Channel Width Selection
The choice of channel width depends on the specific requirements of the network and the environment. In dense environments with many Wi-Fi networks, narrower channels (20 MHz or 40 MHz) are preferred to minimize interference. In less congested environments, wider channels (80 MHz or 160 MHz) can be used to achieve higher data rates.

### Impact on Performance
Wider channels can significantly improve Wi-Fi performance by increasing the data rate and reducing latency. However, they also require more spectrum and are more vulnerable to interference. It is essential to balance the need for higher throughput with the potential for increased interference when selecting the appropriate channel width for a Wi-Fi network.

In summary, Wi-Fi channel width plays a crucial role in determining the performance and reliability of a wireless network. Understanding the trade-offs between different channel widths can help optimize the network for specific applications and environments.

## Identifying Channel Width from Beacon Frames

To identify the channel width from Wi-Fi beacon frames, you need to analyze the information elements (IEs) within the beacon frame. Beacon frames are periodically transmitted by access points (APs) to announce the presence of a Wi-Fi network. These frames contain various IEs that provide information about the network, including the channel width.

### Steps to Identify Channel Width

1. **Capture Beacon Frames**:
   Use a Wi-Fi packet capture tool (e.g., Wireshark) to capture beacon frames from the Wi-Fi network. Ensure that your capture device supports the frequency bands and channel widths used by the network.

2. **Locate the HT Capabilities IE**:
   In the captured beacon frame, locate the "HT Capabilities" information element. This IE is present in 802.11n and later standards and provides information about the supported channel widths.

3. **Check Supported Channel Widths**:
   Within the HT Capabilities IE, look for the "Supported Channel Width Set" field. This field indicates whether the AP supports 20 MHz, 40 MHz, or both channel widths. The field is typically represented as:
   - `0`: 20 MHz only
   - `1`: 20 MHz and 40 MHz

4. **Locate the VHT Capabilities IE**:
   For 802.11ac networks, locate the "VHT Capabilities" information element. This IE provides information about the supported channel widths for very high throughput (VHT) networks.

5. **Check VHT Supported Channel Widths**:
   Within the VHT Capabilities IE, look for the "Supported Channel Width Set" field. This field indicates whether the AP supports 20 MHz, 40 MHz, 80 MHz, or 160 MHz channel widths. The field is typically represented as:
   - `0`: 20 MHz and 40 MHz
   - `1`: 80 MHz
   - `2`: 160 MHz and 80+80 MHz

6. **Analyze HE Capabilities IE**:
   For 802.11ax (Wi-Fi 6) networks, locate the "HE Capabilities" information element. This IE provides information about the supported channel widths for high-efficiency (HE) networks.

7. **Check HE Supported Channel Widths**:
   Within the HE Capabilities IE, look for the "Supported Channel Width Set" field. This field indicates whether the AP supports 20 MHz, 40 MHz, 80 MHz, 160 MHz, or 80+80 MHz channel widths.

### Example

Here is an example of how to identify the channel width from a beacon frame using Wireshark:

1. Open Wireshark and start capturing packets on the desired Wi-Fi interface.
2. Filter the captured packets to display only beacon frames using the filter: `wlan.fc.type_subtype == 0x08`.
3. Select a beacon frame from the list and expand the "IEEE 802.11 wireless LAN management frame" section.
4. Locate the "HT Capabilities" IE and check the "Supported Channel Width Set" field.
5. If applicable, locate the "VHT Capabilities" IE and check the "Supported Channel Width Set" field.
6. If applicable, locate the "HE Capabilities" IE and check the "Supported Channel Width Set" field.

By following these steps, you can determine the channel width supported by the Wi-Fi network from the beacon frames.

### Tools

- **Wireshark**: A popular network protocol analyzer that can capture and analyze Wi-Fi packets, including beacon frames.
- **Aircrack-ng**: A suite of tools for capturing and analyzing Wi-Fi packets, including airodump-ng for capturing beacon frames.

Understanding the channel width from beacon frames can help optimize Wi-Fi network performance and troubleshoot connectivity issues. By analyzing the beacon frames, you can gain insights into the network's capabilities and configuration.


### Types of Frames in Wi-Fi

Wi-Fi communication relies on the exchange of various types of frames between devices. These frames are categorized into three main types: management frames, control frames, and data frames. Each type of frame serves a specific purpose in the operation and maintenance of the Wi-Fi network.

1. **Management Frames**:
   Management frames are used to establish and maintain connections between devices in a Wi-Fi network. They facilitate the discovery, authentication, and association processes. Common types of management frames include:
   - **Beacon Frames**: Broadcasted periodically by access points (APs) to announce the presence and capabilities of the network.
   - **Probe Request Frames**: Sent by clients to discover available networks.
   - **Probe Response Frames**: Sent by APs in response to probe requests, providing information about the network.
   - **Authentication Frames**: Used to initiate the authentication process between a client and an AP.
   - **Deauthentication Frames**: Used to terminate an existing authentication.
   - **Association Request Frames**: Sent by clients to request association with an AP.
   - **Association Response Frames**: Sent by APs in response to association requests, indicating acceptance or rejection.
   - **Disassociation Frames**: Used to terminate an existing association.

2. **Control Frames**:
   Control frames assist in the delivery of data frames and help manage access to the wireless medium. They ensure that data frames are transmitted efficiently and without collisions. Common types of control frames include:
   - **Request to Send (RTS) Frames**: Used to request permission to send data, helping to avoid collisions in a busy network.
   - **Clear to Send (CTS) Frames**: Sent in response to RTS frames, granting permission to send data.
   - **Acknowledgment (ACK) Frames**: Sent to confirm the successful receipt of data frames.
   - **Power Save Poll (PS-Poll) Frames**: Used by clients in power-saving mode to request buffered data from the AP.

3. **Data Frames**:
   Data frames carry the actual data payload between devices in a Wi-Fi network. They are used for the transmission of user data, such as web pages, emails, and file transfers. Data frames can also include additional information, such as quality of service (QoS) parameters, to prioritize certain types of traffic. Common types of data frames include:
   - **Data Frames**: Carry user data between devices.
   - **Null Data Frames**: Used for power management, indicating that a device is awake or entering sleep mode.
   - **QoS Data Frames**: Include QoS parameters to prioritize certain types of traffic, such as voice or video.

Understanding the different types of frames in Wi-Fi is essential for analyzing and troubleshooting wireless networks. Each frame type plays a crucial role in the overall operation and performance of the network, ensuring reliable and efficient communication between devices.
