# Ethernet

Ethernet is a widely used networking technology that enables devices to communicate over a local area network (LAN). It is a fundamental technology for connecting computers, printers, and other devices in homes and businesses. In embedded systems, Ethernet provides reliable, high-speed connectivity for industrial automation, IoT devices, and networked embedded applications.

## Key Concepts

- **Frames**: Ethernet transmits data in packets called frames. Each frame contains source and destination MAC addresses, as well as the data being transmitted.

- **MAC Address**: A Media Access Control (MAC) address is a unique identifier (48 bits / 6 bytes) assigned to network interfaces for communication on the physical network segment. Format: XX:XX:XX:XX:XX:XX (hexadecimal).

- **Switching**: Ethernet switches are devices that connect multiple devices on a LAN and use MAC addresses to forward frames to the correct destination.

## Ethernet Frame Structure

An Ethernet II frame consists of the following fields:

```
| Preamble | SFD | Dest MAC | Src MAC | EtherType | Payload | FCS |
| 7 bytes  | 1   | 6 bytes  | 6 bytes | 2 bytes   | 46-1500 | 4   |
```

- **Preamble**: 7 bytes of alternating 1s and 0s (10101010) for synchronization
- **Start Frame Delimiter (SFD)**: 1 byte (10101011) marks the start of the frame
- **Destination MAC Address**: 6 bytes - target device address
- **Source MAC Address**: 6 bytes - sender device address
- **EtherType**: 2 bytes - indicates the protocol in the payload (e.g., 0x0800 for IPv4, 0x0806 for ARP)
- **Payload**: 46-1500 bytes - actual data being transmitted
- **Frame Check Sequence (FCS)**: 4 bytes - CRC32 checksum for error detection

**Total Frame Size**: 64 to 1518 bytes (without preamble/SFD)

### IEEE 802.3 Frame (Alternative)

The 802.3 frame replaces EtherType with a Length field and uses LLC/SNAP headers in the payload.

## OSI Layers and Ethernet

Ethernet operates at two layers:

- **Physical Layer (Layer 1)**: Handles signal transmission, voltage levels, timing, and physical connectors
- **Data Link Layer (Layer 2)**: Divided into two sublayers:
  - **MAC (Media Access Control)**: Handles frame assembly, addressing, and channel access
  - **LLC (Logical Link Control)**: Provides interface to network layer

## Common Standards

### IEEE 802.3 Variants

| Standard | Speed | Name | Medium | Distance |
|----------|-------|------|--------|----------|
| 802.3 | 10 Mbps | 10BASE-T | Cat3/Cat5 UTP | 100m |
| 802.3u | 100 Mbps | 100BASE-TX | Cat5 UTP | 100m |
| 802.3ab | 1 Gbps | 1000BASE-T | Cat5e/Cat6 UTP | 100m |
| 802.3an | 10 Gbps | 10GBASE-T | Cat6a/Cat7 UTP | 100m |
| 802.3z | 1 Gbps | 1000BASE-X | Fiber optic | 550m-5km |
| 802.3ae | 10 Gbps | 10GBASE-SR/LR | Fiber optic | 300m-40km |

**Naming Convention**: `<Speed>BASE-<Signaling>`
- Speed: 10, 100, 1000 (Mbps) or 10G, 40G, 100G (Gbps)
- BASE: Baseband signaling
- Signaling: T (twisted pair), X (fiber), etc.

### Communication Modes

1. **Full Duplex**: Modern Ethernet supports full duplex communication, allowing devices to send and receive data simultaneously on separate wire pairs, which eliminates collisions and improves network efficiency. Most common in switched networks.

2. **Half Duplex**: Devices can either send or receive at any given time, but not both. Uses CSMA/CD (Carrier Sense Multiple Access with Collision Detection). Legacy mode, rare in modern networks.

### Advanced Features

- **VLANs (802.1Q)**: Virtual Local Area Networks allow network administrators to segment a single physical network into multiple logical networks for improved security and performance. VLAN tags add 4 bytes to the Ethernet frame.

- **Auto-Negotiation**: Automatically detects and configures the best common speed and duplex mode between connected devices.

- **Flow Control (802.3x)**: Pause frames allow receivers to signal transmitters to temporarily stop sending data when buffers are full.

## PHY and MAC Architecture

In embedded systems, Ethernet is typically implemented as two distinct components:

### MAC (Media Access Control) Controller

The MAC controller handles:
- Frame assembly and parsing
- MAC address filtering
- CRC generation and checking
- Buffer management (TX/RX FIFOs)
- DMA operations for efficient data transfer
- Collision detection (half-duplex)

Usually integrated into the microcontroller/SoC.

### PHY (Physical Layer) Transceiver

The PHY chip handles:
- Signal encoding/decoding (MLT-3, 4B5B, 8B10B, etc.)
- Line drivers and receivers
- Clock recovery
- Link status detection
- Auto-negotiation
- Analog signal processing

External IC connected to MAC via standard interfaces.

### MAC-PHY Interfaces

Common interfaces between MAC and PHY:

1. **MII (Media Independent Interface)**
   - Speed: 10/100 Mbps
   - Signals: 16 data/control lines
   - Clock: 25 MHz (100 Mbps) / 2.5 MHz (10 Mbps)
   - Parallel interface

2. **RMII (Reduced MII)**
   - Speed: 10/100 Mbps
   - Signals: 9 lines (reduced pin count)
   - Clock: 50 MHz (external reference clock required)
   - More common in embedded systems due to fewer pins

3. **GMII (Gigabit MII)**
   - Speed: 10/100/1000 Mbps
   - Signals: 24 data/control lines
   - Clock: 125 MHz at 1 Gbps
   - Parallel interface for Gigabit speeds

4. **RGMII (Reduced GMII)**
   - Speed: 10/100/1000 Mbps
   - Signals: 12 lines
   - Clock: 125 MHz with DDR (Double Data Rate)
   - Common in modern embedded systems

5. **SGMII (Serial GMII)**
   - Speed: 10/100/1000 Mbps
   - Signals: 4 differential pairs (TX+/-, RX+/-)
   - Serial interface, fewer pins
   - Common in SoCs and networking equipment

### MDIO/MDC Management Interface

The MAC communicates with the PHY for configuration and status monitoring via:
- **MDIO (Management Data Input/Output)**: Bidirectional data line
- **MDC (Management Data Clock)**: Clock signal (typically < 2.5 MHz)

This 2-wire serial interface allows:
- Reading/writing PHY registers
- Checking link status
- Configuring speed and duplex
- Reading PHY ID and capabilities

## Embedded Ethernet Controllers

### Common Integrated MAC Controllers

Many microcontrollers include built-in Ethernet MAC:
- **STM32F4/F7/H7**: ARM Cortex-M with 10/100 Mbps MAC
- **i.MX RT Series**: ARM Cortex-M7 with 10/100/1000 Mbps MAC
- **SAM E70/V71**: ARM Cortex-M7 with 10/100 Mbps MAC
- **ESP32**: Built-in MAC (requires external PHY)
- **Microchip PIC32**: MIPS-based with MAC
- **TI Sitara AM335x**: ARM Cortex-A with dual MACs

### Popular External PHY Chips

- **LAN8720A**: 10/100 Mbps, RMII, low cost, common in embedded
- **DP83848**: 10/100 Mbps, MII/RMII, TI
- **KSZ8081**: 10/100 Mbps, MII/RMII, Microchip
- **RTL8211F**: 10/100/1000 Mbps, RGMII, Realtek
- **KSZ9031**: 10/100/1000 Mbps, RGMII, Microchip

### Integrated Ethernet Solutions

For microcontrollers without MAC:
- **W5500**: SPI-to-Ethernet with hardwired TCP/IP stack
- **ENC28J60**: SPI-to-Ethernet, 10 Mbps
- **W5100S**: SPI/parallel, hardwired TCP/IP

These provide complete Ethernet solutions with MAC, PHY, and protocol handling.

## Applications

Ethernet is used in various applications, including:

- **Local Area Networking**: Connecting computers and devices within a limited geographical area, such as an office or home.

- **Data Centers**: Providing high-speed connections between servers and storage devices.

- **Industrial Automation**: Enabling communication between machines and control systems in manufacturing environments (EtherCAT, PROFINET, Ethernet/IP).

- **Embedded IoT Devices**: Network-connected sensors, cameras, and control systems.

- **Automotive Ethernet**: In-vehicle networking (100BASE-T1, 1000BASE-T1 standards).

## Physical Layer Signaling

### Twisted Pair Cable Configuration

Standard Ethernet uses 8-wire (4 pairs) twisted pair cables:

**10/100BASE-T (Fast Ethernet)**:
- Uses 2 pairs: Pairs 2 and 3 (pins 1,2,3,6)
- Pair 2 (Orange): Pins 1,2 - TX+/TX- or RX+/RX-
- Pair 3 (Green): Pins 3,6 - RX+/RX- or TX+/TX-
- Pairs 1 and 4 unused in standard implementation

**1000BASE-T (Gigabit Ethernet)**:
- Uses all 4 pairs simultaneously
- Each pair transmits and receives (full duplex)
- Bidirectional signaling on each pair

**T568A vs T568B Wiring Standards**:
- Both are valid standards with different color codes
- Straight-through cables: Same standard both ends
- Crossover cables: T568A one end, T568B other (legacy, auto-MDIX eliminates need)

### Voltage Levels and Signaling

**100BASE-TX Signaling**:
- Voltage: ±1V differential signaling
- Encoding: 4B5B (4 data bits encoded as 5 signal bits) + MLT-3
- Common mode: 0V ±25mV
- Transformer isolation: Magnetic coupling (1:1 transformer)

**1000BASE-T Signaling**:
- Voltage: ±1V peak differential
- Encoding: PAM-5 (5-level Pulse Amplitude Modulation)
- Symbol rate: 125 Mbaud per pair × 4 pairs = 1000 Mbps
- Hybrid circuits for simultaneous TX/RX

### Key Ethernet Signals

1. **Carrier Sense**: Detection of signal energy on the medium (half-duplex only)
   - CRS (Carrier Sense) signal in MII interface
   - Prevents transmission when medium is busy

2. **Collision Detection (CSMA/CD)**: Half-duplex mode
   - COL (Collision) signal in MII interface
   - Detected when simultaneous transmit/receive occurs
   - Random backoff algorithm after collision

3. **Link Integrity**:
   - **10BASE-T**: Link pulses (NLP - Normal Link Pulse) every 16ms
   - **100BASE-TX**: Idle symbols continuously transmitted
   - **Auto-negotiation**: FLP (Fast Link Pulse) bursts for capability exchange

4. **Preamble and SFD**:
   - Preamble: 7 bytes (0xAA) for clock synchronization
   - SFD: 1 byte (0xAB) marks frame start
   - Allows receiver to lock onto signal timing

5. **Inter-Frame Gap (IFG)**:
   - Minimum 96 bit-times between frames
   - 9.6µs at 10 Mbps, 960ns at 100 Mbps, 96ns at 1 Gbps
   - Ensures receivers can process previous frame

## Hardware Design Considerations

### Crystal/Oscillator Requirements

- **25 MHz**: Common for 10/100 Ethernet MAC
- **50 MHz**: For RMII interface (often external)
- **125 MHz**: For Gigabit Ethernet (RGMII)
- Accuracy: ±50 ppm typical requirement

### Magnetics (Transformers)

Purpose:
- Electrical isolation (safety)
- Common-mode noise rejection
- Impedance matching (75Ω to 100Ω differential)
- Protects against voltage transients

Types:
- Discrete transformers + common-mode chokes
- Integrated RJ45 connectors with built-in magnetics

### Power Supply

- PHY chips typically require multiple voltage rails:
  - 3.3V or 2.5V for digital I/O
  - 1.2V or 1.8V for core logic
  - Sometimes separate analog supply
- MAC usually powered from MCU supply (1.8V or 3.3V)

### PCB Layout Guidelines

- Maintain differential pair impedance (typically 100Ω)
- Keep TX and RX pairs separated
- Minimize distance between MAC and PHY
- Use ground plane
- Place magnetics close to RJ45 connector
- Add ESD protection on RJ45 pins

### Reset and Boot Configuration

- PHY chips often have:
  - Hardware reset pin (active low)
  - Bootstrap pins to configure address, mode
  - Configuration resistors/straps
- MAC reset via MCU reset system

## Network Protocols and Layers

### Common Layer 2 Protocols

- **ARP (Address Resolution Protocol)**: Maps IP addresses to MAC addresses
- **LLDP (Link Layer Discovery Protocol)**: Device discovery and capability advertisement
- **STP/RSTP**: Spanning Tree Protocol for loop prevention
- **802.1X**: Port-based network access control

### Layer 3 and Above

Ethernet carries higher-layer protocols:
- **IPv4/IPv6**: Internet Protocol
- **TCP/UDP**: Transport layer protocols
- **ICMP**: Internet Control Message Protocol
- **DHCP**: Dynamic Host Configuration Protocol

### Embedded TCP/IP Stacks

Software stacks for embedded Ethernet:
- **lwIP**: Lightweight IP stack, widely used, open source
- **uIP**: Micro IP, very small footprint
- **FreeRTOS+TCP**: Integrated with FreeRTOS
- **Zephyr networking**: Part of Zephyr RTOS
- **Embedded Wizard**: Commercial stack
- **Proprietary vendor stacks**: From STM32, NXP, etc.




## Power over Ethernet (PoE)

PoE allows electrical power to be transmitted over Ethernet cables along with data, eliminating the need for separate power cables.

### PoE Standards

| Standard | Power | Voltage | Max Current | Power Pairs |
|----------|-------|---------|-------------|-------------|
| 802.3af (PoE) | 15.4W (13W device) | 44-57V DC | 350 mA | 2 pairs |
| 802.3at (PoE+) | 30W (25.5W device) | 50-57V DC | 600 mA | 2 pairs |
| 802.3bt (PoE++) Type 3 | 60W (51W device) | 50-57V DC | 600 mA | 4 pairs |
| 802.3bt (PoE++) Type 4 | 100W (71W device) | 52-57V DC | 960 mA | 4 pairs |

### PoE Components

- **PSE (Power Sourcing Equipment)**: PoE switch or injector that provides power
- **PD (Powered Device)**: Device receiving power (IP camera, VoIP phone, embedded device)
- **PoE Controller**: IC that negotiates power and manages PD side (e.g., TI TPS2375, LTC4267)

### PoE Detection and Classification

1. **Detection**: PSE applies 2.7-10V to detect if PD is present (25kΩ signature resistance)
2. **Classification**: PSE determines power class needed (0-8)
3. **Power-up**: PSE applies full voltage if valid PD detected
4. **Operation**: Continuous power delivery with monitoring

### PoE in Embedded Systems

- Simplifies deployment (single cable for data + power)
- Common for IoT devices, sensors, IP cameras
- Reduces installation cost and complexity
- Enables remote power cycling via software

## Performance and Optimization

### Throughput Considerations

**Theoretical vs Actual**:
- 100 Mbps Fast Ethernet: ~94 Mbps actual (overhead from preamble, IFG, headers)
- 1 Gbps Gigabit: ~940 Mbps actual
- Jumbo frames (>1500 bytes MTU) can improve efficiency

### Latency

Typical latencies:
- Switch latency: 5-50 µs (cut-through) to 50-200 µs (store-and-forward)
- Cable delay: ~5 ns/m
- Processing in embedded system: depends on CPU, DMA, interrupt handling

### Buffer Management

- **TX buffers**: Hold outgoing frames before transmission
- **RX buffers**: Store received frames before processing
- **DMA**: Efficiently transfers data between memory and MAC
- Insufficient buffering leads to dropped packets

### Common Issues and Debugging

1. **No Link**:
   - Check cable connection
   - Verify PHY power and reset
   - Check MDIO communication
   - Verify clock signals (oscillator)
   - Check PHY address configuration

2. **Link Up but No Data**:
   - Verify MAC configuration (speed/duplex match)
   - Check TX/RX buffer setup
   - Verify DMA configuration
   - Check firewall/filtering rules
   - Inspect ARP resolution

3. **Packet Loss**:
   - Buffer overflow (increase buffer size)
   - CRC errors (cable quality, noise)
   - Collision in half-duplex (switch to full-duplex)
   - CPU not processing packets fast enough

4. **Performance Issues**:
   - Interrupt storm (use interrupt coalescing)
   - Inefficient buffer management
   - Memory bandwidth limitations
   - CPU bottleneck in packet processing

### Debug Tools

- **PHY registers**: Read via MDIO to check link status, speed, duplex
- **Wireshark/tcpdump**: Capture and analyze packets
- **Logic analyzer**: Observe MII/RMII signals
- **Oscilloscope**: Check signal quality, voltage levels
- **Built-in counters**: MAC statistics for TX/RX errors, collisions

## Industrial Ethernet Protocols

Specialized protocols for industrial automation:

- **EtherCAT**: Ethernet for Control Automation Technology, real-time, daisy-chain topology
- **PROFINET**: Industrial Ethernet by Siemens, real-time communication
- **Ethernet/IP**: Industrial Protocol, used with CIP (Common Industrial Protocol)
- **Modbus TCP**: Modbus protocol over TCP/IP
- **POWERLINK**: Real-time protocol by B&R Automation
- **EtherNet/IP**: Managed by ODVA

These protocols add deterministic, real-time capabilities on top of standard Ethernet.

## Conclusion

Ethernet remains a cornerstone of modern networking, providing reliable and high-speed communication for a wide range of applications. In embedded systems, understanding the hardware architecture (MAC/PHY separation), interface standards (MII, RMII, RGMII), and protocol details is essential for successful implementation. From selecting appropriate controllers and PHY chips to proper PCB layout and software stack integration, Ethernet design requires attention to both hardware and software aspects.

Key takeaways for embedded Ethernet design:
- Choose appropriate MAC-PHY interface based on speed and pin count requirements
- Follow PCB layout best practices for signal integrity
- Select suitable TCP/IP stack for your RTOS and application
- Implement proper error handling and buffer management
- Consider PoE for simplified deployment
- Use debugging tools effectively for troubleshooting

With proper design and implementation, Ethernet provides robust, high-performance networking for embedded applications ranging from simple IoT devices to complex industrial automation systems.
