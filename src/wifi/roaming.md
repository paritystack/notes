# Roaming

## 802.11r
- Also known as Fast BSS Transition (FT).
- Released: 2008.
- Purpose: Improves the speed of the handoff process between access points.
- Notes: Reduces the time required for re-authentication when a device moves from one AP to another.

### Technical Details of 802.11r

802.11r, also known as Fast BSS Transition (FT), is a standard that aims to improve the handoff process between access points (APs) in a wireless network. This is particularly important for applications that require seamless connectivity, such as VoIP (Voice over IP) and real-time video streaming. Here are some key technical details:

1. **Key Caching**:
   - 802.11r introduces the concept of key caching, which allows a client device to reuse the Pairwise Master Key (PMK) from a previous connection when roaming to a new AP. This reduces the time required for re-authentication.

2. **Fast Transition (FT) Protocol**:
   - The FT protocol defines two methods for fast transitions: over-the-air and over-the-DS (Distribution System).
     - **Over-the-Air**: The client communicates directly with the target AP to perform the handoff.
     - **Over-the-DS**: The client communicates with the target AP through the current AP, using the wired network (DS) as an intermediary.

3. **Reduced Latency**:
   - By minimizing the time required for re-authentication and key exchange, 802.11r significantly reduces the latency associated with roaming. This is crucial for maintaining the quality of real-time applications.

4. **FT Initial Mobility Domain Association**:
   - When a client first associates with an AP in an 802.11r-enabled network, it performs an FT Initial Mobility Domain Association. This process establishes the necessary security context and prepares the client for fast transitions within the mobility domain.

5. **Mobility Domain Information Element (MDIE)**:
   - The MDIE is included in the beacon frames and probe responses of 802.11r-enabled APs. It provides information about the mobility domain, allowing client devices to identify and connect to APs that support fast transitions.

6. **Fast BSS Transition Information Element (FTIE)**:
   - The FTIE is used during the authentication and reassociation processes to carry the necessary cryptographic information for fast transitions. It ensures that the security context is properly established and maintained during the handoff.

7. **Compatibility**:
   - 802.11r is designed to be backward compatible with non-802.11r devices. APs can support both 802.11r and non-802.11r clients simultaneously, ensuring a smooth transition for devices that do not support the standard.

By implementing these technical features, 802.11r enhances the efficiency and reliability of the roaming process, providing a better user experience in environments with multiple access points.


## 802.11k
- Also known as Radio Resource Management (RRM).
- Released: 2008.
- Purpose: Provides mechanisms for measuring and reporting the radio environment.
- Notes: Helps devices make better roaming decisions by providing information about neighboring APs.

### Technical Details of 802.11k

802.11k, also known as Radio Resource Management (RRM), is a standard that provides mechanisms for measuring and reporting the radio environment. This information helps client devices make better roaming decisions by providing data about neighboring access points (APs). Here are some key technical details:

1. **Neighbor Reports**:
   - 802.11k enables APs to provide neighbor reports to client devices. These reports contain information about nearby APs, including their signal strength, channel, and supported data rates. This helps clients identify the best AP to roam to.

2. **Beacon Reports**:
   - Client devices can request beacon reports from APs. These reports include details about the beacons received from neighboring APs, such as signal strength and channel utilization. This information assists clients in making informed roaming decisions.

3. **Channel Load Reports**:
   - APs can provide channel load reports, which indicate the level of traffic on a particular channel. This helps client devices avoid congested channels and select APs operating on less crowded frequencies.

4. **Noise Histogram Reports**:
   - Noise histogram reports provide information about the noise levels on different channels. By analyzing these reports, client devices can avoid channels with high levels of interference, improving overall network performance.

5. **Transmit Stream/Category Measurement Reports**:
   - These reports provide data on the performance of specific traffic streams or categories. This helps client devices assess the quality of service (QoS) provided by different APs and make better roaming decisions based on their specific needs.

6. **Location Tracking**:
   - 802.11k supports location tracking features, allowing APs to track the location of client devices within the network. This information can be used to optimize network performance and improve the accuracy of neighbor reports.

7. **Link Measurement Reports**:
   - Link measurement reports provide detailed information about the quality of the wireless link between the client device and the AP. This includes metrics such as signal-to-noise ratio (SNR) and packet error rate (PER), which help clients evaluate the performance of their current connection and potential target APs.

By implementing these technical features, 802.11k enhances the ability of client devices to make informed roaming decisions, leading to improved network performance and a better user experience in environments with multiple access points.


## 802.11v
- Also known as Wireless Network Management.
- Released: 2011.
- Purpose: Enhances network management by providing mechanisms for configuring client devices.
- Notes: Includes features like BSS Transition Management, which helps devices roam more efficiently.

### Technical Details of 802.11v

802.11v, also known as Wireless Network Management, is a standard that enhances network management by providing mechanisms for configuring client devices. This standard includes several features that improve the efficiency and performance of wireless networks. Here are some key technical details:

1. **BSS Transition Management**:
   - 802.11v provides BSS Transition Management, which helps client devices make better roaming decisions. APs can suggest the best APs for clients to roam to, based on factors like signal strength and load.

2. **Network Assisted Power Savings**:
   - This feature allows APs to provide information to client devices about the best times to enter power-saving modes. By coordinating power-saving activities, 802.11v helps extend battery life for client devices.

3. **Traffic Filtering Service (TFS)**:
   - TFS enables APs to filter traffic for client devices, reducing the amount of unnecessary data that clients need to process. This helps improve the efficiency of the network and reduces power consumption for client devices.

4. **Wireless Network Management (WNM) Sleep Mode**:
   - WNM Sleep Mode allows client devices to enter a low-power sleep state while remaining connected to the network. APs can buffer data for sleeping clients and deliver it when they wake up, improving power efficiency without sacrificing connectivity.

5. **Diagnostic and Reporting**:
   - 802.11v includes mechanisms for diagnostic and reporting, allowing APs and client devices to exchange information about network performance and issues. This helps network administrators identify and resolve problems more quickly.

6. **Location Services**:
   - The standard supports location services, enabling APs to provide location-based information to client devices. This can be used for applications like asset tracking and location-based services.

By implementing these technical features, 802.11v enhances the management and performance of wireless networks, leading to improved efficiency, better power management, and a more reliable user experience in environments with multiple access points.


## 802.11w
- Also known as Protected Management Frames (PMF).
- Released: 2009.
- Purpose: Enhances the security of management frames.
- Notes: Protects against certain types of attacks, such as deauthentication and disassociation attacks.

### Technical Details of 802.11w

802.11w, also known as Protected Management Frames (PMF), is a standard that enhances the security of management frames in wireless networks. This standard provides mechanisms to protect against certain types of attacks, such as deauthentication and disassociation attacks. Here are some key technical details:

1. **Management Frame Protection**:
   - 802.11w provides protection for management frames, which are used for network control and signaling. By securing these frames, the standard helps prevent attackers from disrupting network operations.

2. **Protected Management Frames (PMF)**:
   - PMF ensures that management frames are both encrypted and authenticated. This prevents unauthorized devices from injecting malicious management frames into the network.

3. **Robust Security Network (RSN) Associations**:
   - 802.11w requires the use of RSN associations, which provide a secure method for devices to join the network. This includes the use of cryptographic techniques to protect the integrity and confidentiality of management frames.

4. **Replay Protection**:
   - The standard includes mechanisms to protect against replay attacks, where an attacker captures and retransmits management frames to disrupt network operations. By using sequence numbers and timestamps, 802.11w ensures that management frames cannot be reused maliciously.

5. **Deauthentication and Disassociation Protection**:
   - 802.11w specifically addresses deauthentication and disassociation attacks, where an attacker forces a device to disconnect from the network. By securing these management frames, the standard helps maintain stable and reliable network connections.

By implementing these technical features, 802.11w enhances the security of wireless networks, protecting against various types of attacks and ensuring the integrity and reliability of network operations.
