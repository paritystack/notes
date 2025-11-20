# CAN (Controller Area Network)

Controller Area Network (CAN) is a robust vehicle bus standard designed to allow microcontrollers and devices to communicate with each other without a host computer. It is widely used in automotive and industrial applications due to its reliability and efficiency.

## Key Concepts

- **Frames**: CAN communication is based on frames, which are structured packets of data. Each frame contains an identifier, control bits, data, and error-checking information.

- **Identifiers**: Each frame has a unique identifier that determines the priority of the message. Lower identifier values have higher priority on the bus.

- **Bitwise Arbitration**: CAN uses a non-destructive bitwise arbitration method to control access to the bus. This ensures that the highest priority message is transmitted without collision.

## Common Standards

1. **CAN 2.0A**: This standard defines 11-bit identifiers for frames.
2. **CAN 2.0B**: This standard extends the identifier length to 29 bits, allowing for more unique message identifiers.
3. **CAN FD (Flexible Data-rate)**: This standard allows for higher data rates and larger data payloads compared to traditional CAN.


## Frame Formats and Bit Timing

### CAN 2.0A (Standard Frame - 11-bit Identifier)

The standard CAN frame consists of the following fields:

```
[SOF][Arbitration Field][Control Field][Data Field][CRC Field][ACK Field][EOF]
```

**Detailed Breakdown:**

| Field | Bits | Description |
|-------|------|-------------|
| **SOF** (Start of Frame) | 1 | Single dominant bit indicating frame start |
| **Identifier** | 11 | Message priority (lower = higher priority) |
| **RTR** (Remote Transmission Request) | 1 | 0=Data frame, 1=Remote frame |
| **IDE** (Identifier Extension) | 1 | 0=Standard frame, 1=Extended frame |
| **r0** (Reserved) | 1 | Reserved bit (must be dominant) |
| **DLC** (Data Length Code) | 4 | Number of data bytes (0-8) |
| **Data Field** | 0-64 | Actual payload data (0-8 bytes) |
| **CRC** (Cyclic Redundancy Check) | 15 | Error detection code |
| **CRC Delimiter** | 1 | Recessive bit separating CRC |
| **ACK Slot** | 1 | Receiver writes dominant if frame OK |
| **ACK Delimiter** | 1 | Recessive bit |
| **EOF** (End of Frame) | 7 | Seven recessive bits |
| **IFS** (Interframe Space) | 3 | Three recessive bits (minimum) |

**Total Standard Frame Size:**
- Minimum (0 data bytes): 47 bits
- Maximum (8 data bytes): 111 bits
- Plus bit stuffing overhead (approximately 20% max)

**Example Standard Frame (ID=0x123, 2 data bytes: 0xAB, 0xCD):**
```
SOF | 00100100011 | 0 | 0 | 0 | 0010 | 10101011 11001101 | CRC(15) | 1 | ACK | 1 | 1111111
     \_________/   RTR IDE r0  DLC    \_______________/
       ID=0x123                        Data=0xABCD
```

### CAN 2.0B (Extended Frame - 29-bit Identifier)

Extended frames support longer identifiers for more complex networks:

```
[SOF][Base ID(11)][SRR][IDE][Extended ID(18)][RTR][r1][r0][DLC][Data][CRC][ACK][EOF]
```

**Key Differences:**

| Field | Bits | Description |
|-------|------|-------------|
| **Base Identifier** | 11 | Most significant 11 bits of 29-bit ID |
| **SRR** (Substitute Remote Request) | 1 | Always recessive (replaces RTR) |
| **IDE** | 1 | 1=Extended frame |
| **Extended Identifier** | 18 | Least significant 18 bits of ID |
| **RTR** | 1 | 0=Data frame, 1=Remote frame |
| **r1, r0** | 2 | Reserved bits |
| **DLC** | 4 | Data length (0-8 bytes) |

**Total Extended Frame Size:**
- Minimum (0 data bytes): 67 bits
- Maximum (8 data bytes): 131 bits
- Plus bit stuffing overhead

**29-bit ID Format:**
```
Base ID (11 bits) | Extended ID (18 bits)
MSB           LSB | MSB              LSB
```

**Arbitration Priority:**
- Standard frames have higher priority than extended frames with same base ID
- During arbitration, IDE bit gives standard frame priority

### CAN FD (Flexible Data-rate)

CAN FD extends classical CAN with:

**Key Enhancements:**
1. **Larger Payload**: Up to 64 bytes (vs 8 bytes in classic CAN)
2. **Faster Data Phase**: Up to 5 Mbps for data (vs 1 Mbps max for classic)
3. **Improved CRC**: Better error detection with longer CRC sequences

**CAN FD Frame Structure:**
```
[Arbitration Phase - same speed] | [Data Phase - faster speed] | [ACK/EOF - same speed]
```

**Additional Control Bits:**

| Field | Description |
|-------|-------------|
| **FDF** (FD Format) | 1=CAN FD frame, 0=Classic CAN |
| **res** (Reserved) | Reserved bit (replaces r0) |
| **BRS** (Bit Rate Switch) | 1=Switch to faster bit rate for data phase |
| **ESI** (Error State Indicator) | Error state of transmitting node |

**DLC to Data Length Mapping (CAN FD):**

| DLC | Data Bytes | DLC | Data Bytes |
|-----|------------|-----|------------|
| 0-8 | 0-8 (same) | 12  | 48 |
| 9   | 12         | 13  | 64 |
| 10  | 16         | 14  | Reserved |
| 11  | 24         | 15  | Reserved |

**Example CAN FD Advantages:**
```
Classic CAN: 8 bytes @ 500 kbps = 134 μs transmission time
CAN FD:      64 bytes @ 2 Mbps data = 210 μs transmission time
             (8x more data in ~1.5x time)
```

### Bit Stuffing

CAN uses bit stuffing to ensure sufficient transitions for synchronization:

**Rules:**
- After 5 consecutive bits of the same polarity, insert complementary bit
- Applies from SOF to CRC (excluding CRC delimiter, ACK, EOF)
- Receiver automatically removes stuffed bits

**Example:**
```
Original data:    1 1 1 1 1 0 0 0 1
After stuffing:   1 1 1 1 1 0 0 0 0 0 1
                           ↑       ↑
                      Stuff bits added
```

**Implications:**
- Maximum overhead: ~20% (worst case)
- Ensures no more than 5 consecutive identical bits
- Bit stuffing error if violation detected

### Bit Timing and Synchronization

A single CAN bit is divided into four time segments:

```
|<-- Sync Seg -->|<-- Prop Seg -->|<-- Phase Seg 1 -->|<-- Phase Seg 2 -->|
|      1 TQ      |    1-8 TQ      |     1-8 TQ        |     1-8 TQ        |
                                   ^
                              Sample Point
```

**Time Segments:**

1. **Sync Segment (Sync_Seg)**: 1 Time Quantum (TQ)
   - Used to synchronize nodes on bus
   - Always 1 TQ

2. **Propagation Segment (Prop_Seg)**: 1-8 TQ
   - Compensates for physical delay on bus
   - Accounts for transceiver delays

3. **Phase Segment 1 (Phase_Seg1)**: 1-8 TQ
   - Can be lengthened during resynchronization

4. **Phase Segment 2 (Phase_Seg2)**: 1-8 TQ
   - Can be shortened during resynchronization

5. **Sample Point**: Between Phase_Seg1 and Phase_Seg2
   - Where bit value is read
   - Typically 75-87.5% through bit time

**Baud Rate Calculation:**

```
Bit_Time = Sync_Seg + Prop_Seg + Phase_Seg1 + Phase_Seg2
Bit_Time = 1 TQ + Prop_Seg + Phase_Seg1 + Phase_Seg2

Baud_Rate = 1 / Bit_Time
Baud_Rate = f_clk / (BRP × Bit_Time_in_TQ)
```

Where:
- `f_clk`: CAN controller clock frequency
- `BRP`: Baud Rate Prescaler (divider)
- `Bit_Time_in_TQ`: Total time quanta per bit

**Example Calculation (500 kbps with 8 MHz clock):**
```
Target: 500 kbps (2 μs bit time)

Choose:
- BRP = 2
- Sync_Seg = 1 TQ
- Prop_Seg = 2 TQ
- Phase_Seg1 = 3 TQ
- Phase_Seg2 = 2 TQ
- Total = 8 TQ

TQ = 2 × (1/8MHz) = 0.25 μs
Bit_Time = 8 × 0.25 μs = 2 μs
Baud_Rate = 1 / 2 μs = 500 kbps ✓

Sample Point = (1+2+3) / 8 = 75%
```

**Common CAN Baud Rates:**

| Baud Rate | Bit Time | Max Bus Length* |
|-----------|----------|-----------------|
| 1 Mbps    | 1 μs     | 40 m |
| 500 kbps  | 2 μs     | 100 m |
| 250 kbps  | 4 μs     | 250 m |
| 125 kbps  | 8 μs     | 500 m |
| 100 kbps  | 10 μs    | 600 m |
| 50 kbps   | 20 μs    | 1000 m |

*Approximate values, depends on cable quality and transceiver

**Synchronization Jump Width (SJW):**
- Maximum adjustment allowed during resynchronization
- Typically 1-4 TQ
- Used to adjust for phase errors
- Should be min(4, Phase_Seg1, Phase_Seg2)


## Applications

CAN is used in various applications, including:

- **Automotive**: Enabling communication between different electronic control units (ECUs) in vehicles, such as engine control, transmission, and braking systems.
- **Industrial Automation**: Facilitating communication between sensors, actuators, and controllers in manufacturing and process control systems.
- **Medical Equipment**: Ensuring reliable data exchange between different components of medical devices.


## Physical Layer and Electrical Characteristics

### Bus Topology

CAN uses a **linear bus topology** with termination resistors at both ends:

```
   120Ω                                                              120Ω
    ┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬
    │         │         │         │         │         │         │
  Node 1    Node 2    Node 3    Node 4    Node 5    Node 6    Node 7
```

**Characteristics:**
- Linear bus with stub connections
- Maximum stub length: 0.3 m at 1 Mbps (shorter is better)
- All nodes connected in parallel
- Both ends must have termination resistors

**Important Notes:**
- Star or ring topologies are NOT recommended
- Minimize stub lengths to reduce reflections
- Total bus length depends on baud rate

### Differential Signaling

CAN uses **differential signaling** with two wires:

**CAN_H (CAN High)** and **CAN_L (CAN Low)**

**Principle:**
- Signal is the voltage difference between CAN_H and CAN_L
- Provides excellent noise immunity
- Common-mode noise rejection

**Bus States:**

| State | CAN_H | CAN_L | Differential Voltage | Description |
|-------|-------|-------|---------------------|-------------|
| **Dominant (0)** | ~3.5V | ~1.5V | ~2.0V | Logical 0 |
| **Recessive (1)** | ~2.5V | ~2.5V | ~0V | Logical 1 |

**Voltage Levels (ISO 11898):**

**Dominant State:**
- CAN_H: 2.75V - 4.5V (typical 3.5V)
- CAN_L: 0.5V - 2.25V (typical 1.5V)
- Differential: 1.5V - 3.0V (typical 2.0V)

**Recessive State:**
- CAN_H: 2.0V - 3.0V (typical 2.5V)
- CAN_L: 2.0V - 3.0V (typical 2.5V)
- Differential: -0.5V to +0.5V (typical 0V)

**Wired-AND Logic:**
- Any node can drive the bus dominant
- Recessive state only when ALL nodes release the bus
- This enables non-destructive arbitration

```
Node A: Recessive (1) |------|_______|------|  Dominant wins
Node B: Dominant  (0) |______|_______|______|
Bus Result:           |______|_______|______|
```

### Termination Resistors

Termination resistors are **critical** for proper CAN operation:

**Purpose:**
- Prevent signal reflections
- Ensure proper bus biasing
- Reduce ringing and overshoot

**Standard Value:** 120Ω at each end of the bus

**Why 120Ω?**
- Matches characteristic impedance of twisted pair cable
- Two 120Ω resistors in parallel = 60Ω bus impedance
- Optimal for reflection-free transmission

**Verification:**
With bus powered off, measure resistance between CAN_H and CAN_L:
- Correct: ~60Ω (two 120Ω in parallel)
- One terminator only: ~120Ω
- No terminators: Open circuit (infinite resistance)

**Split Termination (Advanced):**
For improved EMC performance:
```
       CAN_H ──┬── 60Ω ──┬── 100nF ── GND
                         │
       CAN_L ──┴── 60Ω ──┘
```

### Cable Specifications

**Recommended Cable Types:**
- **Twisted pair cable** (essential for noise immunity)
- Characteristic impedance: 120Ω
- Common standards: DeviceNet, CANopen cables

**Cable Parameters:**

| Parameter | Specification |
|-----------|---------------|
| Characteristic Impedance | 120Ω ± 5Ω |
| Twist Rate | 10-50 twists/meter |
| Cross-sectional Area | 0.25 - 0.75 mm² (AWG 24-18) |
| Maximum Capacitance | 60 pF/m |
| Shield | Optional but recommended for EMI |

**Cable Length vs Baud Rate:**

| Baud Rate | Max Length | Signal Propagation Delay |
|-----------|------------|--------------------------|
| 1 Mbps    | 40 m       | 5 ns/m |
| 500 kbps  | 100 m      | 5 ns/m |
| 250 kbps  | 250 m      | 5 ns/m |
| 125 kbps  | 500 m      | 5 ns/m |
| 50 kbps   | 1000 m     | 5 ns/m |
| 10 kbps   | 6000 m     | 5 ns/m |

**Calculation Rule:**
```
Max_Length = (Bit_Time - 2 × Delays) / (2 × Propagation_Delay)

Where:
- Bit_Time: Time for one bit (1/Baud_Rate)
- Delays: Transceiver + Node delays (~200-250 ns total)
- Propagation_Delay: ~5 ns/m for typical cable
```

### CAN Transceivers

CAN transceivers convert logic-level signals to differential bus signals:

**Popular Transceiver ICs:**

| Part Number | Speed | Voltage | Features |
|-------------|-------|---------|----------|
| **MCP2551** | 1 Mbps | 5V | Industry standard, slope control |
| **TJA1050** | 1 Mbps | 5V | NXP, low EME |
| **SN65HVD230** | 1 Mbps | 3.3V | TI, low power, standby mode |
| **MCP2562** | 1 Mbps | 5V | Improved EMC over MCP2551 |
| **TCAN1051** | 5 Mbps | 3.3/5V | CAN FD capable |

**Typical Transceiver Connections:**

```
Microcontroller                Transceiver               CAN Bus
┌─────────────┐               ┌──────────┐
│             │               │          │
│ CAN_TX ─────┼───────────────┼→ TXD    │
│             │               │          │
│ CAN_RX ←────┼───────────────┼← RXD    │
│             │               │          │       120Ω
│ GND ────────┼───────────────┼─ GND    │   ┬─────/\/\/\/\─┬
│             │               │          │   │              │
│ VCC ────────┼───────────────┼─ VCC  CANH ─┼─────────────┼─ CAN_H
└─────────────┘               │        │    │              │
                              │      CANL ─┼─────────────┼─ CAN_L
                              │        │    │              │
                              └──────────┘   │              │
                                             ┴              ┴
                                            GND         120Ω + GND
```

**Transceiver Modes:**

1. **Normal Mode**: Transmit and receive enabled
2. **Silent/Listen-Only Mode**: Receive only, no ACK transmission
3. **Standby/Sleep Mode**: Low power consumption

**Silent Mode Use Cases:**
- Bus monitoring/sniffing
- Network analysis
- Hot-plugging new nodes
- Debugging without affecting bus

### Electrical Specifications (ISO 11898-2)

**Common-Mode Range:**
- -2V to +7V (transceiver must handle)
- Allows ground potential differences between nodes

**Maximum Propagation Delay:**
- Transceiver: 120 ns (typical)
- Cable: 5 ns/m
- Node: ~50 ns (controller + driver delays)

**Input Thresholds:**

| Parameter | Min | Typ | Max | Unit |
|-----------|-----|-----|-----|------|
| Dominant Threshold (VTH_DOM) | 0.9 | 1.2 | 1.4 | V |
| Recessive Threshold (VTH_REC) | 0.5 | 0.6 | 0.9 | V |

**Output Drive:**
- Dominant state: 40-70 mA drive capability
- Recessive state: High impedance (only pull-ups drive bus)

### Power Supply and Grounding

**Best Practices:**

1. **Decoupling Capacitors:**
   - 100nF ceramic close to each transceiver VCC
   - 10-100μF bulk capacitor per module

2. **Ground Connections:**
   - Connect all node grounds through shield or separate ground wire
   - Minimize ground loops
   - Keep ground impedance low

3. **Galvanic Isolation (optional):**
   - Use isolated DC-DC converter for power
   - Use digital isolators for CAN signals
   - Common in industrial and automotive applications
   - Protects from ground loops and high voltages

**Isolated CAN Interface:**
```
MCU Side         Isolation Barrier        Bus Side
┌──────┐         ┌───────────┐           ┌──────────┐
│ TX ──┼────────→│ ISO7221   │──────────→│ TXD      │
│      │         │ (Digital  │           │          │
│ RX ←─┼─────────│  Isolator)│←──────────│ RXD   MCP2551
│      │         └───────────┘           │          │
│ GND ─┤                                 │ GND CANH/L
└──────┘                                 └──────────┘
 Isolated                               Isolated
  Power                                  Ground
```

### Fault Protection

**Protection Features in Transceivers:**

1. **Thermal Shutdown**: Prevents overheating
2. **Short Circuit Protection**: CAN_H/CAN_L to GND or VCC
3. **ESD Protection**: Electrostatic discharge protection (±8kV typical)
4. **Undervoltage Lockout**: Prevents operation at low VCC

**External Protection (recommended):**
- TVS diodes on CAN_H and CAN_L
- Common-mode choke for additional EMI filtering
- Polyfuse or current-limiting resistor


## Error Detection and Fault Confinement

CAN has sophisticated error detection and handling mechanisms that ensure high reliability.

### Five Error Detection Mechanisms

CAN implements five independent error detection methods:

#### 1. Bit Monitoring

**Mechanism:**
- Each transmitter monitors the bus while transmitting
- Compares transmitted bit with actual bus state
- Exception: During arbitration and ACK slot (recessive allowed to become dominant)

**Error Condition:**
- Transmitted bit ≠ Observed bit (outside allowed exceptions)

**Example:**
```
Node transmits: 1 (Recessive)
Bus reads:      0 (Dominant) ← Another node pulling bus down
Result: Bit Error detected (unless during arbitration/ACK)
```

#### 2. Bit Stuffing

**Mechanism:**
- After 5 consecutive identical bits, a complementary bit is inserted
- Receiver expects and removes stuff bits
- Applies from SOF to CRC

**Error Condition:**
- Six consecutive identical bits detected by receiver

**Example:**
```
Received: 1 1 1 1 1 1 0 ← Six consecutive 1s
Result: Stuff Error detected
```

#### 3. Frame Check (Format Error)

**Mechanism:**
- Certain bit fields must have fixed values
- CRC Delimiter, ACK Delimiter, EOF must be recessive

**Error Condition:**
- Fixed-format bits have wrong value

**Example:**
```
Expected EOF: 1 1 1 1 1 1 1 (seven recessive bits)
Received:     1 1 0 1 1 1 1 ← Dominant bit in EOF
Result: Form Error detected
```

#### 4. ACK Error

**Mechanism:**
- Transmitter sends recessive bit in ACK slot
- At least one receiver must write dominant bit if frame correct
- Transmitter monitors ACK slot

**Error Condition:**
- ACK slot remains recessive (no receiver acknowledged)

**Causes:**
- No other nodes on bus
- All receivers detected errors
- Receiver hardware failure
- Bus disconnected

**Example:**
```
Transmitter sends ACK slot: 1 (Recessive)
Expected from receiver:     0 (Dominant) ← Acknowledgment
Bus remains:                1 (Recessive) ← No ACK!
Result: ACK Error detected
```

#### 5. CRC Error

**Mechanism:**
- Transmitter calculates 15-bit CRC over data
- Receiver performs same calculation
- Both must match

**Error Condition:**
- Calculated CRC ≠ Received CRC

**CRC Polynomial:**
```
CAN 2.0: x^15 + x^14 + x^10 + x^8 + x^7 + x^4 + x^3 + 1
```

### Error Frames

When a node detects an error, it transmits an **Error Frame** to notify all nodes:

**Error Frame Structure:**
```
[Error Flag] + [Error Delimiter]
```

**Error Flag Types:**

| Node State | Error Flag | Description |
|------------|------------|-------------|
| **Error-Active** | 6 dominant bits | Active Error Flag |
| **Error-Passive** | 6 recessive bits | Passive Error Flag |

**Error Frame Sequence:**

1. Node detects error
2. Node transmits Error Flag (violates bit stuffing rule)
3. All other nodes detect stuff error
4. Other nodes also send Error Flags (Error Flag Superposition)
5. Results in 6-12 dominant bits total
6. All nodes send Error Delimiter (8 recessive bits)
7. Original transmitter retransmits the frame

**Example Error Sequence:**
```
Normal frame: [SOF][ID]....[Data]...[CRC] ← Error detected here
                                     ↓
Error Frame:                    [000000][11111111]
                                 ↑        ↑
                            Error Flag  Delimiter

Retransmission: [SOF][ID]....[Data]...[CRC][ACK][EOF] ← Retry
```

### Error Counters and States

Each CAN node maintains two error counters:

**TEC (Transmit Error Counter):**
- Incremented when transmission errors occur
- Decremented when successful transmission

**REC (Receive Error Counter):**
- Incremented when reception errors occur
- Decremented when successful reception

**Counter Rules:**

| Event | TEC Change | REC Change |
|-------|------------|------------|
| Transmitter detects error | +8 | - |
| Receiver detects error | - | +1 |
| Successful transmission | -1 | - |
| Successful reception | - | -1* |
| Transmit dominant during error flag | +8 | - |

*REC decremented by 1 if between 1-127, otherwise set to 119-127 range

### Node States

CAN nodes operate in one of three states based on error counter values:

```
                  TEC or REC > 127           TEC > 255
Error-Active  ────────────────────→  Error-Passive  ─────────→  Bus-Off
              ←────────────────────
                 Both counters < 128
```

#### 1. Error-Active State

**Conditions:**
- TEC ≤ 127 AND REC ≤ 127

**Behavior:**
- Normal operation
- Can transmit and receive
- Sends **Active Error Flags** (6 dominant bits)
- Immediately interrupts faulty transmissions

**Characteristics:**
- Most common operational state
- Node actively participates in bus communication
- Can dominate the bus during error signaling

#### 2. Error-Passive State

**Conditions:**
- TEC > 127 OR REC > 127
- AND TEC ≤ 255

**Behavior:**
- Can still transmit and receive
- Sends **Passive Error Flags** (6 recessive bits)
- Must wait for **Suspend Transmission Time** (8 recessive bits) after error
- Cannot interrupt other nodes' transmissions

**Purpose:**
- Prevents faulty node from disrupting bus
- Node is "quarantined" but can still monitor and transmit
- Reduces impact of malfunctioning node

**Suspend Transmission:**
After transmitting error flag, error-passive node must wait:
```
[Passive Error Flag][Error Delimiter][Suspend Transmission]
     6 recessive         8 recessive        8 recessive
```

#### 3. Bus-Off State

**Conditions:**
- TEC > 255

**Behavior:**
- Node is **disconnected** from bus
- Cannot transmit or receive
- Does not send ACK bits
- Must wait for recovery

**Entry:**
- Only transmit errors can cause Bus-Off
- Indicates serious problem with node or connection

**Recovery Process:**

1. **TEC exceeds 255** → Node enters Bus-Off
2. **Wait for recovery**: Monitor bus for 128 × 11 recessive bits (128 idle frames)
3. **Automatic reset**: After recovery period, node can rejoin
4. **Software intervention**: Some systems require explicit reset

```
Bus-Off Recovery:
TEC = 256+ → Bus-Off State
             ↓
        Wait 128 × 11 recessive bits
             ↓
        TEC = 0, REC = 0
             ↓
        Error-Active State
```

### State Transition Diagram

```
┌─────────────────┐
│  Error-Active   │  Normal operation
│  TEC ≤ 127      │  Active Error Flags (6 dominant bits)
│  REC ≤ 127      │
└────────┬────────┘
         │
         │ TEC > 127 or REC > 127
         ↓
┌─────────────────┐
│ Error-Passive   │  Reduced participation
│  TEC ≤ 255      │  Passive Error Flags (6 recessive bits)
│  REC > 127      │  Suspend Transmission delay
└────────┬────────┘
         │
         │ TEC > 255
         ↓
┌─────────────────┐
│    Bus-Off      │  Disconnected from bus
│   TEC > 255     │  Cannot transmit/receive
│                 │  Requires recovery period
└────────┬────────┘
         │
         │ 128 × 11 recessive bits observed
         ↓
┌─────────────────┐
│  Error-Active   │  Return to normal (counters reset)
└─────────────────┘
```

### Fault Confinement Philosophy

**Goal:** Prevent faulty nodes from disrupting the entire network

**Mechanisms:**

1. **Error Counters**: Track node reliability
2. **State Transitions**: Progressive isolation of faulty nodes
3. **Exponential Penalty**: Repeated errors increase counter faster
4. **Automatic Recovery**: Nodes can rejoin after proving stability

**Benefits:**
- Self-healing network
- Faulty nodes automatically isolated
- Good nodes continue operating
- No external intervention needed (in most cases)

### Practical Error Scenarios

#### Scenario 1: Broken Cable Connection

```
Node A (disconnected) tries to transmit:
1. Sends frame, gets no ACK → ACK Error
2. TEC += 8
3. Retransmits, gets no ACK → ACK Error
4. TEC += 8
5. After ~32 failed transmissions: TEC > 255 → Bus-Off
6. Node A automatically disconnects from bus
```

#### Scenario 2: Noisy Environment

```
Receiver detects corrupted frame:
1. CRC Error detected
2. REC += 1
3. Sends Error Frame
4. Transmitter retransmits
5. If errors persist: REC grows
6. At REC > 127: Node becomes Error-Passive
7. Reduces its impact on bus while still receiving
```

#### Scenario 3: Incorrect Baud Rate

```
Node with wrong baud rate:
1. Detects constant bit errors
2. TEC/REC increment rapidly
3. Quickly enters Error-Passive
4. Eventually Bus-Off
5. Does not disrupt properly configured nodes
```

### Error Statistics Monitoring

**Best Practices:**

1. **Monitor Error Counters**: Read TEC and REC periodically
2. **Log Error Types**: Track which errors occur most
3. **Set Thresholds**: Alert when counters exceed limits
4. **Trend Analysis**: Increasing errors indicate problems

**Typical Monitoring:**
```c
// Pseudocode for error monitoring
if (TEC > 96 || REC > 96) {
    // Warning: Node approaching Error-Passive
    log_warning("High error count");
}

if (node_state == ERROR_PASSIVE) {
    // Error: Node in degraded state
    log_error("Node Error-Passive");
    // Investigate: wiring, termination, baud rate
}

if (node_state == BUS_OFF) {
    // Critical: Node offline
    log_critical("Node Bus-Off");
    // Check: physical connection, transceiver
}
```

### Common Causes of Errors

| Error Type | Common Causes |
|------------|---------------|
| **ACK Error** | No other nodes, all nodes off, disconnected bus |
| **Bit Error** | Bus contention, faulty transceiver, wrong termination |
| **Stuff Error** | Noise, incorrect baud rate, EMI |
| **CRC Error** | Noise, bit errors during transmission, EMI |
| **Form Error** | Incorrect baud rate, synchronization issues |

**Troubleshooting Checklist:**
- ✓ Verify 120Ω termination at both ends
- ✓ Check all nodes have same baud rate configuration
- ✓ Measure voltage levels on CAN_H and CAN_L
- ✓ Ensure twisted-pair cable used
- ✓ Check for EMI sources near cable
- ✓ Verify transceiver power supply stable
- ✓ Test cable continuity and resistance


## Programming Examples

### SocketCAN (Linux)

SocketCAN is the standard CAN interface for Linux systems.

**Setup CAN Interface:**

```bash
# Load kernel modules
sudo modprobe can
sudo modprobe can_raw
sudo modprobe vcan  # Virtual CAN for testing

# Create virtual CAN interface (for testing)
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# For real hardware (e.g., Raspberry Pi with MCP2515)
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# Verify interface is up
ip -details link show can0
```

**Using can-utils:**

```bash
# Install can-utils
sudo apt-get install can-utils

# Send a CAN frame (ID=0x123, data=0x11 0x22 0x33)
cansend can0 123#112233

# Send extended frame (ID=0x12345678)
cansend can0 12345678#DEADBEEF

# Receive and display CAN frames
candump can0

# Filter specific ID
candump can0,123:7FF  # Receive only ID 0x123

# Generate random traffic (testing)
cangen can0 -I 100 -L 8 -D r -g 100

# Display statistics
canfdtest can0 -v
```

**C Programming with SocketCAN:**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>

int main() {
    int s;
    struct sockaddr_can addr;
    struct ifreq ifr;
    struct can_frame frame;

    // Create socket
    if ((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
        perror("Socket");
        return 1;
    }

    // Specify can0 interface
    strcpy(ifr.ifr_name, "can0");
    ioctl(s, SIOCGIFINDEX, &ifr);

    // Bind socket to can0
    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Bind");
        return 1;
    }

    // Prepare frame
    frame.can_id = 0x123;       // Standard ID
    frame.can_dlc = 4;          // Data length
    frame.data[0] = 0x11;
    frame.data[1] = 0x22;
    frame.data[2] = 0x33;
    frame.data[3] = 0x44;

    // Send frame
    if (write(s, &frame, sizeof(struct can_frame)) != sizeof(struct can_frame)) {
        perror("Write");
        return 1;
    }

    printf("Sent CAN frame: ID=0x%X, Data=", frame.can_id);
    for (int i = 0; i < frame.can_dlc; i++) {
        printf("%02X ", frame.data[i]);
    }
    printf("\n");

    // Receive frame
    struct can_frame rx_frame;
    int nbytes = read(s, &rx_frame, sizeof(struct can_frame));

    if (nbytes < 0) {
        perror("Read");
        return 1;
    }

    printf("Received CAN frame: ID=0x%X, DLC=%d, Data=",
           rx_frame.can_id, rx_frame.can_dlc);
    for (int i = 0; i < rx_frame.can_dlc; i++) {
        printf("%02X ", rx_frame.data[i]);
    }
    printf("\n");

    close(s);
    return 0;
}
```

**Compile:**
```bash
gcc socketcan_example.c -o socketcan_example
./socketcan_example
```

**CAN Filtering:**

```c
// Filter example: only receive ID 0x100-0x1FF
struct can_filter rfilter[1];
rfilter[0].can_id   = 0x100;
rfilter[0].can_mask = 0x700;  // Mask bits 8-10

setsockopt(s, SOL_CAN_RAW, CAN_RAW_FILTER, &rfilter, sizeof(rfilter));
```

### Arduino with MCP2515

The MCP2515 is a popular SPI-based CAN controller for Arduino.

**Hardware Setup:**
```
Arduino Uno         MCP2515 Module
Pin 13 (SCK)   →    SCK
Pin 12 (MISO)  ←    SO
Pin 11 (MOSI)  →    SI
Pin 10 (SS)    →    CS
5V             →    VCC
GND            →    GND
                    CANH → CAN Bus High
                    CANL → CAN Bus Low
```

**Library Installation:**
```
Arduino IDE: Library Manager → Install "mcp2515" by autowp
```

**Basic Send Example:**

```cpp
#include <mcp2515.h>

MCP2515 mcp2515(10);  // CS pin 10

struct can_frame canMsg;

void setup() {
  Serial.begin(115200);

  // Initialize MCP2515 at 500kbps with 8MHz crystal
  mcp2515.reset();
  mcp2515.setBitrate(CAN_500KBPS, MCP_8MHZ);
  mcp2515.setNormalMode();

  Serial.println("MCP2515 Initialized");
}

void loop() {
  // Prepare message
  canMsg.can_id  = 0x123;
  canMsg.can_dlc = 4;
  canMsg.data[0] = 0xAA;
  canMsg.data[1] = 0xBB;
  canMsg.data[2] = 0xCC;
  canMsg.data[3] = 0xDD;

  // Send message
  mcp2515.sendMessage(&canMsg);

  Serial.println("Message sent: ID=0x123");
  delay(1000);
}
```

**Receive Example:**

```cpp
#include <mcp2515.h>

MCP2515 mcp2515(10);

void setup() {
  Serial.begin(115200);

  mcp2515.reset();
  mcp2515.setBitrate(CAN_500KBPS, MCP_8MHZ);
  mcp2515.setNormalMode();

  Serial.println("Waiting for CAN messages...");
}

void loop() {
  struct can_frame canMsg;

  // Check if message available
  if (mcp2515.readMessage(&canMsg) == MCP2515::ERROR_OK) {
    Serial.print("ID: 0x");
    Serial.print(canMsg.can_id, HEX);
    Serial.print(" DLC: ");
    Serial.print(canMsg.can_dlc);
    Serial.print(" Data: ");

    for (int i = 0; i < canMsg.can_dlc; i++) {
      Serial.print("0x");
      Serial.print(canMsg.data[i], HEX);
      Serial.print(" ");
    }
    Serial.println();
  }
}
```

**Using Filters (Receive only specific IDs):**

```cpp
void setup() {
  Serial.begin(115200);

  mcp2515.reset();
  mcp2515.setBitrate(CAN_500KBPS, MCP_8MHZ);

  // Filter to receive only ID 0x100-0x10F
  struct can_filter filter;
  filter.can_id = 0x100;
  filter.can_mask = 0x7F0;  // Mask lower 4 bits

  mcp2515.setFilter(filter);
  mcp2515.setNormalMode();

  Serial.println("Listening for IDs 0x100-0x10F");
}
```

### Raspberry Pi with MCP2515

**Hardware Setup:**

```
Raspberry Pi    MCP2515
BCM 11 (SCLK)  → SCK
BCM 10 (MOSI)  → SI
BCM 9  (MISO)  ← SO
BCM 8  (CE0)   → CS
BCM 25         → INT (optional)
3.3V           → VCC
GND            → GND
```

**Enable SPI and Configure:**

```bash
# Enable SPI interface
sudo raspi-config
# Interface Options → SPI → Enable

# Edit boot config
sudo nano /boot/config.txt

# Add these lines:
dtparam=spi=on
dtoverlay=mcp2515-can0,oscillator=8000000,interrupt=25
dtoverlay=spi0-hw-cs

# Reboot
sudo reboot
```

**Configure Interface:**

```bash
# Bring up CAN interface at 500 kbps
sudo ip link set can0 up type can bitrate 500000

# Auto-start on boot
sudo nano /etc/network/interfaces
# Add:
auto can0
iface can0 inet manual
    pre-up /sbin/ip link set can0 type can bitrate 500000
    up /sbin/ifconfig can0 up
    down /sbin/ifconfig can0 down
```

**Use can-utils or Python** (see SocketCAN and Python-CAN sections)

### STM32 HAL

STM32 microcontrollers have built-in CAN peripherals (bxCAN or FDCAN).

**CubeMX Configuration:**
- Enable CAN1 peripheral
- Set bit timing: Prescaler, BS1, BS2, SJW for desired baud rate
- Configure pins (e.g., PA11=CAN_RX, PA12=CAN_TX)
- Enable interrupts if needed

**Initialization Code:**

```c
#include "main.h"

CAN_HandleTypeDef hcan1;

// Configure CAN
void CAN_Config(void) {
    CAN_FilterTypeDef canFilterConfig;

    // Configure filter to accept all messages
    canFilterConfig.FilterBank = 0;
    canFilterConfig.FilterMode = CAN_FILTERMODE_IDMASK;
    canFilterConfig.FilterScale = CAN_FILTERSCALE_32BIT;
    canFilterConfig.FilterIdHigh = 0x0000;
    canFilterConfig.FilterIdLow = 0x0000;
    canFilterConfig.FilterMaskIdHigh = 0x0000;
    canFilterConfig.FilterMaskIdLow = 0x0000;
    canFilterConfig.FilterFIFOAssignment = CAN_RX_FIFO0;
    canFilterConfig.FilterActivation = ENABLE;

    HAL_CAN_ConfigFilter(&hcan1, &canFilterConfig);

    // Start CAN
    HAL_CAN_Start(&hcan1);

    // Enable RX interrupt
    HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING);
}

// Send CAN message
void CAN_Send(uint32_t id, uint8_t *data, uint8_t len) {
    CAN_TxHeaderTypeDef txHeader;
    uint32_t txMailbox;

    txHeader.StdId = id;              // Standard ID
    txHeader.ExtId = 0;
    txHeader.RTR = CAN_RTR_DATA;      // Data frame
    txHeader.IDE = CAN_ID_STD;        // Standard ID
    txHeader.DLC = len;               // Data length
    txHeader.TransmitGlobalTime = DISABLE;

    // Transmit message
    if (HAL_CAN_AddTxMessage(&hcan1, &txHeader, data, &txMailbox) != HAL_OK) {
        Error_Handler();
    }
}

// Receive interrupt callback
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan) {
    CAN_RxHeaderTypeDef rxHeader;
    uint8_t rxData[8];

    // Receive message
    if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &rxHeader, rxData) == HAL_OK) {
        // Process received message
        printf("Received ID: 0x%lX, DLC: %lu, Data: ",
               rxHeader.StdId, rxHeader.DLC);
        for (int i = 0; i < rxHeader.DLC; i++) {
            printf("%02X ", rxData[i]);
        }
        printf("\n");
    }
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_CAN1_Init();

    CAN_Config();

    uint8_t txData[4] = {0x11, 0x22, 0x33, 0x44};

    while (1) {
        CAN_Send(0x123, txData, 4);
        HAL_Delay(1000);
    }
}
```

**Message Filtering Example:**

```c
// Accept only ID 0x100-0x1FF
canFilterConfig.FilterIdHigh = 0x100 << 5;      // ID in upper bits
canFilterConfig.FilterIdLow = 0x0000;
canFilterConfig.FilterMaskIdHigh = 0x700 << 5;  // Mask
canFilterConfig.FilterMaskIdLow = 0x0000;
```

### Python-CAN

Python-CAN provides a high-level interface for CAN communication.

**Installation:**

```bash
pip install python-can
```

**Basic Send/Receive:**

```python
import can
import time

# Create bus instance (SocketCAN interface)
bus = can.interface.Bus(channel='can0', bustype='socketcan')

# Send a message
msg = can.Message(
    arbitration_id=0x123,
    data=[0x11, 0x22, 0x33, 0x44, 0x55],
    is_extended_id=False
)

try:
    bus.send(msg)
    print(f"Message sent on {bus.channel_info}")
except can.CanError:
    print("Message NOT sent")

# Receive messages
print("Waiting for messages...")
for message in bus:
    print(f"ID: 0x{message.arbitration_id:X} "
          f"DLC: {message.dlc} "
          f"Data: {message.data.hex()}")

    # Exit after 10 messages
    if message.arbitration_id == 0x200:
        break

bus.shutdown()
```

**Using Filters:**

```python
# Filter to receive only specific IDs
filters = [
    {"can_id": 0x100, "can_mask": 0x7F0, "extended": False},  # 0x100-0x10F
    {"can_id": 0x200, "can_mask": 0x7FF, "extended": False},  # Exact 0x200
]

bus = can.interface.Bus(channel='can0', bustype='socketcan',
                        can_filters=filters)
```

**Periodic Transmission:**

```python
from can import Message
from can.interfaces.socketcan import SocketcanBus

bus = SocketcanBus(channel='can0')

# Create periodic message (every 100ms)
msg = Message(arbitration_id=0x123,
              data=[0xAA, 0xBB, 0xCC, 0xDD],
              is_extended_id=False)

task = bus.send_periodic(msg, 0.1)  # 100ms period

time.sleep(5)  # Send for 5 seconds

task.stop()
bus.shutdown()
```

**Logging to File:**

```python
import can

bus = can.interface.Bus(channel='can0', bustype='socketcan')

# Log to ASC format
logger = can.ASCWriter('logfile.asc')

for message in bus:
    logger(message)

logger.stop()
bus.shutdown()
```

**Virtual CAN for Testing:**

```python
# Setup virtual CAN first (bash):
# sudo ip link add dev vcan0 type vcan
# sudo ip link set up vcan0

import can
import threading
import time

def sender():
    bus = can.interface.Bus(channel='vcan0', bustype='socketcan')
    msg = can.Message(arbitration_id=0x123, data=[1, 2, 3, 4])

    while True:
        bus.send(msg)
        time.sleep(1)

def receiver():
    bus = can.interface.Bus(channel='vcan0', bustype='socketcan')

    for message in bus:
        print(f"Received: {message}")

# Run sender and receiver in parallel
t1 = threading.Thread(target=sender, daemon=True)
t2 = threading.Thread(target=receiver, daemon=True)

t1.start()
t2.start()

time.sleep(10)  # Run for 10 seconds
```

### OBD-II Example (Automotive)

Reading vehicle data using CAN:

```python
import can
import time

# OBD-II uses CAN ID 0x7DF for requests, 0x7E8+ for responses
bus = can.interface.Bus(channel='can0', bustype='socketcan')

# Request engine RPM (PID 0x0C)
request = can.Message(
    arbitration_id=0x7DF,
    data=[0x02, 0x01, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00],
    is_extended_id=False
)

bus.send(request)

# Wait for response
response = bus.recv(timeout=1.0)

if response and response.arbitration_id == 0x7E8:
    # Parse RPM from response
    # Response format: [bytes_returned, 0x41, PID, data_A, data_B, ...]
    rpm = ((response.data[3] << 8) + response.data[4]) / 4
    print(f"Engine RPM: {rpm}")

bus.shutdown()
```

### Message Priority Example

Demonstrating arbitration (lower ID wins):

```python
import can
import threading

bus = can.interface.Bus(channel='vcan0', bustype='socketcan')

def send_high_priority():
    msg = can.Message(arbitration_id=0x100, data=[0xFF])
    bus.send(msg)
    print("High priority (0x100) sent")

def send_low_priority():
    msg = can.Message(arbitration_id=0x700, data=[0xAA])
    bus.send(msg)
    print("Low priority (0x700) sent")

# Send simultaneously - 0x100 will win arbitration
t1 = threading.Thread(target=send_high_priority)
t2 = threading.Thread(target=send_low_priority)

t1.start()
t2.start()

t1.join()
t2.join()

# Monitor order received
for i in range(2):
    msg = bus.recv()
    print(f"Received ID: 0x{msg.arbitration_id:X}")

bus.shutdown()
```


## Conclusion

CAN is a critical communication protocol in automotive and industrial systems, providing reliable and efficient data exchange. Understanding CAN's principles and standards is essential for engineers working in these fields.
