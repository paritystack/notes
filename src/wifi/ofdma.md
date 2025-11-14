# OFDMA (Orthogonal Frequency Division Multiple Access)

## Introduction

OFDMA (Orthogonal Frequency Division Multiple Access) is a key technology introduced in Wi-Fi 6 (802.11ax) and further enhanced in Wi-Fi 7 (802.11be) that enables multiple users to share the same channel simultaneously. OFDMA divides the available channel bandwidth into smaller frequency allocations called Resource Units (RUs), allowing an Access Point (AP) to communicate with multiple stations (STAs) concurrently, improving spectral efficiency and reducing latency, particularly in dense deployment scenarios.

### Key Benefits

- **Improved Efficiency**: Better utilization of available bandwidth by allowing multiple users to transmit simultaneously
- **Reduced Latency**: Lower wait times for small packet transmissions, crucial for IoT and real-time applications
- **Better Performance in Dense Environments**: Optimized for scenarios with many devices transmitting small amounts of data
- **Enhanced Power Efficiency**: Devices can sleep longer between transmissions, conserving battery life
- **Increased Capacity**: Supports more concurrent users on the same channel

## OFDMA vs OFDM

Understanding the difference between OFDM and OFDMA is fundamental to grasping the improvements introduced in Wi-Fi 6 and Wi-Fi 7.

### OFDM (Orthogonal Frequency Division Multiplexing)

OFDM, used in 802.11a/g/n/ac, divides the channel into multiple subcarriers:
- All subcarriers are allocated to a single user at any given time
- Time Division Multiple Access (TDMA) approach: users take turns
- Inefficient for small packets (common in IoT and real-time applications)
- Significant overhead for each transmission opportunity
- Ideal for bulk data transfers to a single user

### OFDMA (Orthogonal Frequency Division Multiple Access)

OFDMA extends OFDM by enabling multi-user access:
- Subcarriers are grouped into Resource Units (RUs)
- Multiple users can transmit/receive simultaneously on different RUs
- Frequency Division Multiple Access (FDMA) combined with time division
- Reduces overhead by serving multiple users in a single transmission opportunity
- Optimized for mixed traffic patterns with varying packet sizes
- Combines well with MU-MIMO for even greater efficiency

**Example Scenario**:
- OFDM: AP sends data to 4 users sequentially, each using the full channel for their turn
- OFDMA: AP sends data to 4 users simultaneously, each using 1/4 of the channel bandwidth

## Resource Units (RUs)

Resource Units are the fundamental building blocks of OFDMA. They represent specific allocations of subcarriers that can be assigned to different users.

### RU Sizes and Specifications

Different RU sizes provide flexibility in allocating bandwidth based on user needs and traffic patterns.

#### 26-Tone RU
- **Subcarriers**: 26 (24 data + 2 pilot subcarriers)
- **Bandwidth**: ~2 MHz
- **Use Case**: IoT devices, sensors, small control packets
- **Maximum RUs per 20 MHz**: 9 RUs
- **Typical Data Rate**: 0.3 - 8 Mbps (depending on MCS)
- **Best For**: Low-bandwidth devices with intermittent traffic

#### 52-Tone RU
- **Subcarriers**: 52 (48 data + 4 pilot subcarriers)
- **Bandwidth**: ~4 MHz
- **Use Case**: Smart home devices, moderate IoT traffic
- **Maximum RUs per 20 MHz**: 4 RUs
- **Typical Data Rate**: 0.7 - 17 Mbps (depending on MCS)
- **Best For**: Devices requiring moderate throughput

#### 106-Tone RU
- **Subcarriers**: 106 (102 data + 4 pilot subcarriers)
- **Bandwidth**: ~8 MHz
- **Use Case**: Standard Wi-Fi clients, streaming devices
- **Maximum RUs per 20 MHz**: 2 RUs
- **Typical Data Rate**: 1.4 - 34 Mbps (depending on MCS)
- **Best For**: Regular data traffic and streaming

#### 242-Tone RU
- **Subcarriers**: 242 (234 data + 8 pilot subcarriers)
- **Bandwidth**: ~20 MHz
- **Use Case**: High-throughput single user or mixed allocation
- **Maximum RUs per 20 MHz**: 1 RU
- **Typical Data Rate**: 3.5 - 86 Mbps (depending on MCS)
- **Best For**: Full 20 MHz channel allocation to one user

#### 484-Tone RU
- **Subcarriers**: 484 (468 data + 16 pilot subcarriers)
- **Bandwidth**: ~40 MHz
- **Use Case**: High-bandwidth applications
- **Maximum RUs per 40 MHz**: 1 RU
- **Typical Data Rate**: 7.3 - 172 Mbps (depending on MCS)
- **Best For**: Video streaming, large file transfers

#### 996-Tone RU
- **Subcarriers**: 996 (980 data + 16 pilot subcarriers)
- **Bandwidth**: ~80 MHz
- **Use Case**: Very high-throughput applications
- **Maximum RUs per 80 MHz**: 1 RU
- **Typical Data Rate**: 15 - 344 Mbps (depending on MCS)
- **Best For**: 4K video, VR applications, bulk transfers

#### 2x996-Tone RU (160 MHz)
- **Subcarriers**: 2 × 996 (1960 data + 32 pilot subcarriers)
- **Bandwidth**: ~160 MHz
- **Use Case**: Maximum throughput scenarios
- **Maximum RUs per 160 MHz**: 1 RU
- **Typical Data Rate**: 30 - 688 Mbps (depending on MCS)
- **Best For**: Extreme bandwidth requirements, 8K video

### Wi-Fi 7 (802.11be) RU Enhancements

Wi-Fi 7 introduces additional RU sizes for improved flexibility:

#### 52+26-Tone RU
- Combines a 52-tone and 26-tone RU
- Provides more granular allocation options
- Better packing efficiency

#### 106+26-Tone RU
- Combines a 106-tone and 26-tone RU
- Reduces waste in channel allocation
- Improved utilization in mixed-traffic scenarios

#### 484+242-Tone RU
- Allows asymmetric allocation in 80 MHz
- Better adaptation to varying user requirements

#### 996+484-Tone RU
- Optimized for 160 MHz channels
- Flexible allocation for mixed high/medium bandwidth users

#### 996+484+242-Tone RU
- Maximum flexibility in 160 MHz
- Supports complex traffic patterns

#### Multi-RU Allocation
- Wi-Fi 7 allows a single user to be assigned multiple non-contiguous RUs
- Dramatically improves channel utilization
- Reduces fragmentation waste

## RU Allocation Patterns

### 20 MHz Channel Allocation Patterns

#### Pattern 1: Maximum Granularity (9 × 26-Tone RUs)
```
|26|26|26|26|26|26|26|26|26|
```
- **Users**: Up to 9 simultaneous users
- **Use Case**: Dense IoT deployments, sensors, smart home networks
- **Efficiency**: Highest for small packets, high overhead for large packets
- **Latency**: Lowest for small transmissions

#### Pattern 2: Mixed Small/Medium (4 × 52-Tone RUs)
```
|  52  |  52  |  52  |  52  |
```
- **Users**: Up to 4 simultaneous users
- **Use Case**: Smart home devices, moderate traffic
- **Efficiency**: Good balance for small to medium packets
- **Typical Scenario**: 4 devices streaming audio or making VoIP calls

#### Pattern 3: Balanced (2 × 106-Tone RUs)
```
|     106     |     106     |
```
- **Users**: Up to 2 simultaneous users
- **Use Case**: Standard Wi-Fi clients with moderate bandwidth needs
- **Efficiency**: Good for typical web browsing, video streaming
- **Typical Scenario**: 2 users streaming HD video

#### Pattern 4: Single User (1 × 242-Tone RU)
```
|          242          |
```
- **Users**: 1 user
- **Use Case**: High-bandwidth single user
- **Efficiency**: Maximum throughput for single user
- **Typical Scenario**: Single user downloading large files

#### Pattern 5: Hybrid 1 (1 × 106 + 4 × 26-Tone RUs)
```
|     106     |26|26|26|26|
```
- **Users**: Up to 5 simultaneous users
- **Use Case**: Mixed traffic: one medium-bandwidth user + IoT devices
- **Efficiency**: Excellent for heterogeneous networks
- **Typical Scenario**: One laptop browsing + multiple sensors

### 40 MHz Channel Allocation Patterns

#### Pattern 1: Maximum Granularity (18 × 26-Tone RUs)
```
|26|26|26|26|26|26|26|26|26|26|26|26|26|26|26|26|26|26|
```
- **Users**: Up to 18 simultaneous users
- **Use Case**: Very dense IoT deployments
- **Efficiency**: Maximum concurrent users for small packets

#### Pattern 2: Medium Granularity (8 × 52-Tone RUs)
```
|  52  |  52  |  52  |  52  |  52  |  52  |  52  |  52  |
```
- **Users**: Up to 8 simultaneous users
- **Use Case**: Dense smart home or office environment
- **Efficiency**: Good for moderate concurrent traffic

#### Pattern 3: Balanced (4 × 106-Tone RUs)
```
|     106     |     106     |     106     |     106     |
```
- **Users**: Up to 4 simultaneous users
- **Use Case**: Multiple users with standard bandwidth needs
- **Typical Scenario**: 4 users each streaming HD video

#### Pattern 4: Mixed (2 × 242-Tone RUs)
```
|          242          |          242          |
```
- **Users**: Up to 2 simultaneous users
- **Use Case**: Two high-bandwidth users
- **Typical Scenario**: 2 users with high-throughput applications

#### Pattern 5: Single User (1 × 484-Tone RU)
```
|                    484                    |
```
- **Users**: 1 user
- **Use Case**: Maximum throughput for single user
- **Typical Scenario**: Large file transfer or 4K streaming

### 80 MHz Channel Allocation Patterns

#### Pattern 1: Maximum Users (36 × 26-Tone RUs)
- Up to 36 simultaneous low-bandwidth users
- Ideal for massive IoT deployments

#### Pattern 2: High-Density (16 × 52-Tone RUs)
- Up to 16 simultaneous medium-bandwidth users
- Good for dense office or residential environments

#### Pattern 3: Standard Density (8 × 106-Tone RUs)
- Up to 8 simultaneous users with moderate bandwidth
- Balanced for typical enterprise deployment

#### Pattern 4: Mixed High/Low (4 × 242-Tone RUs)
- Up to 4 high-bandwidth users
- Good for mixed office/streaming environment

#### Pattern 5: Dual High-Bandwidth (2 × 484-Tone RUs)
- Up to 2 very high-bandwidth users
- Ideal for 4K streaming or large transfers

#### Pattern 6: Single Maximum (1 × 996-Tone RU)
- Single user with maximum throughput
- Best for extreme bandwidth requirements

### 160 MHz Channel Allocation Patterns

160 MHz channels offer the most flexibility and are used in Wi-Fi 6E (6 GHz band) and Wi-Fi 7:

#### Pattern 1: Maximum Granularity (72 × 26-Tone RUs)
- Up to 72 simultaneous ultra-low-bandwidth users
- Extreme IoT scenarios

#### Pattern 2: Very High Density (32 × 52-Tone RUs)
- Up to 32 simultaneous users
- Dense deployments with moderate traffic

#### Pattern 3: High Density (16 × 106-Tone RUs)
- Up to 16 simultaneous users with standard bandwidth
- Large office or campus environments

#### Pattern 4: Mixed Allocation (8 × 242-Tone RUs)
- Up to 8 high-bandwidth users
- Balanced enterprise deployment

#### Pattern 5: Dual Maximum (2 × 996-Tone RUs)
- Up to 2 users with extreme throughput
- Specialized high-bandwidth scenarios

#### Pattern 6: Single Maximum (1 × 2×996-Tone RU)
- Single user maximum throughput
- 8K streaming, VR, or extreme data transfers

## OFDMA Operations

### Downlink OFDMA (DL OFDMA)

Downlink OFDMA allows the AP to transmit data to multiple stations simultaneously on different RUs.

#### Operation Flow
1. **AP Decision**: AP's scheduler decides which STAs need data and allocates RUs
2. **Transmission**: AP sends data to multiple STAs in a single PPDU (Physical Protocol Data Unit)
3. **STA Reception**: Each STA receives and decodes only its assigned RU
4. **Acknowledgment**: STAs send acknowledgments (can be uplink OFDMA)

#### Frame Structure
```
+---------+--------+--------+--------+--------+
| Preamble| SIG-A  | SIG-B  | Data 1 | Data 2 |
|         |        |        | (RU 1) | (RU 2) |
+---------+--------+--------+--------+--------+
                    |
                    +-- Contains RU allocation information
```

#### Benefits
- Single transmission opportunity serves multiple users
- Reduced medium contention overhead
- Improved airtime utilization
- Better latency for small packets

#### Use Cases
- IoT device management (sensor data collection)
- Mixed traffic scenarios (browsing + streaming + IoT)
- Dense deployments with many clients
- Real-time traffic mixed with bulk transfers

### Uplink OFDMA (UL OFDMA)

Uplink OFDMA enables multiple stations to transmit to the AP simultaneously, coordinated by trigger frames.

#### Operation Flow
1. **Trigger Frame**: AP sends a trigger frame specifying:
   - Which STAs should transmit
   - RU allocation for each STA
   - Transmission parameters (MCS, power, etc.)
2. **Synchronized Transmission**: STAs transmit simultaneously on their assigned RUs
3. **AP Reception**: AP receives and decodes all RUs simultaneously
4. **Acknowledgment**: AP sends multi-STA Block Ack

#### Trigger Frame Structure
```
+--------------+---------------+---------------+
| Common Info  | User Info 1   | User Info 2   |
+--------------+---------------+---------------+
       |               |               |
       |               |               +-- RU allocation, MCS, etc.
       |               +-- RU allocation, MCS, etc.
       +-- Trigger type, channel info, duration
```

#### Benefits
- Eliminates uplink contention overhead
- Perfect synchronization of transmissions
- Efficient use of bandwidth for small uplink packets
- Reduced power consumption (devices transmit on schedule)

#### Use Cases
- IoT device reporting (sensors, meters)
- VoIP and video conferencing (uplink voice/video)
- Acknowledgment and control traffic
- Mixed uplink traffic from multiple clients

### Trigger-Based Operation Details

Trigger frames are central to uplink OFDMA operation. They coordinate simultaneous transmissions from multiple stations.

#### Trigger Frame Types

##### Basic Trigger Frame
- **Purpose**: Schedule uplink data transmission
- **Allocation**: Specifies RU assignment for each STA
- **Parameters**: MCS, transmit power, spatial streams
- **Response**: STAs send data on assigned RUs

##### BSRP (Buffer Status Report Poll)
- **Purpose**: Query stations about their buffer status
- **Allocation**: Assigns RUs for buffer status reports
- **Response**: STAs report queue sizes for different access categories
- **Usage**: AP uses this information for efficient scheduling

##### MU-BAR (Multi-User Block Acknowledgment Request)
- **Purpose**: Request acknowledgments from multiple STAs
- **Allocation**: RUs for sending Block Ack responses
- **Response**: STAs send Block Ack frames on assigned RUs
- **Usage**: Efficient acknowledgment in multi-user scenarios

##### BQRP (Bandwidth Query Report Poll)
- **Purpose**: Query stations about bandwidth needs
- **Allocation**: RUs for bandwidth requirement reports
- **Response**: STAs report required bandwidth
- **Usage**: Dynamic bandwidth allocation

##### NFRP (NDP Feedback Report Poll)
- **Purpose**: Request channel state information
- **Allocation**: RUs for CSI feedback
- **Response**: STAs send channel feedback
- **Usage**: Beamforming and link adaptation

#### Common Info Field (in Trigger Frame)
```
+------------+--------+---------+----------+
| Trigger    | UL BW  | GI+LTF  | MU-MIMO  |
| Type       |        | Type    | LTF Mode |
+------------+--------+---------+----------+
| AP TX Pwr  | Pre-   | Doppler | Guard    |
| Info       | FEC    |         | Interval |
+------------+--------+---------+----------+
```

- **Trigger Type**: Identifies the specific trigger type (Basic, BSRP, MU-BAR, etc.)
- **UL BW**: Uplink bandwidth (20/40/80/160 MHz)
- **GI+LTF Type**: Guard interval and Long Training Field configuration
- **AP TX Power Info**: AP's transmit power for power control
- **Doppler**: Indicates if Doppler is expected (for mobility)

#### User Info Field (per STA in Trigger Frame)
```
+----------+----------+----------+----------+
| AID12    | RU       | UL FEC   | UL MCS   |
|          | Alloc    | Coding   |          |
+----------+----------+----------+----------+
| UL DCM   | SS       | UL Target| Trigger  |
|          | Alloc    | RSSI     | Dependent|
+----------+----------+----------+----------+
```

- **AID12**: Association ID of the target STA
- **RU Allocation**: Specific RU assigned (26/52/106/242/484/996)
- **UL FEC Coding**: LDPC or BCC coding
- **UL MCS**: Modulation and coding scheme for this STA
- **SS Allocation**: Spatial stream allocation
- **UL Target RSSI**: Target received signal strength
- **Trigger Dependent Info**: Varies based on trigger type

### Multi-User MIMO with OFDMA (MU-MIMO + OFDMA)

Combining MU-MIMO with OFDMA provides even greater capacity and flexibility.

#### How It Works
- **Frequency Domain**: OFDMA divides channel into RUs
- **Spatial Domain**: MU-MIMO uses multiple antennas for spatial multiplexing
- **Combined**: Each RU can support multiple spatial streams to different users

#### Example Scenario
```
80 MHz Channel:
+---------------+---------------+---------------+---------------+
|   242-tone    |   242-tone    |   242-tone    |   242-tone    |
|   RU (STA A)  |   RU (STA B)  | RU (STA C+D)  | RU (STA E+F)  |
|   2 streams   |   1 stream    | 2 STAs, 1 SS  | 2 STAs, 1 SS  |
|               |               | each (MIMO)   | each (MIMO)   |
+---------------+---------------+---------------+---------------+
```

- RU 1: Single user with 2 spatial streams
- RU 2: Single user with 1 spatial stream
- RU 3: Two users sharing via MU-MIMO (1 stream each)
- RU 4: Two users sharing via MU-MIMO (1 stream each)

#### Benefits
- Maximum spectral efficiency
- Supports diverse device capabilities
- Flexibility for different traffic types
- Optimal capacity in dense environments

#### Requirements
- AP with multiple antennas (typically 4×4 or 8×8)
- STAs with good spatial separation or orthogonal channels
- Sophisticated scheduling algorithms
- Accurate channel state information

## Scheduling and Access Patterns

Efficient OFDMA scheduling is critical for realizing performance benefits.

### Centralized Scheduling

The AP is responsible for all OFDMA scheduling decisions.

#### Scheduler Responsibilities
1. **Traffic Monitoring**: Track buffer states and QoS requirements
2. **RU Allocation**: Decide RU sizes and assignments
3. **User Selection**: Choose which STAs transmit/receive in each opportunity
4. **Parameter Selection**: Determine MCS, power, and other PHY parameters
5. **Trigger Generation**: Create and send trigger frames for uplink

#### Scheduling Algorithms

##### Round-Robin Scheduling
```python
# Pseudocode
for each transmission_opportunity:
    available_users = get_active_users()
    selected_users = []
    remaining_rus = get_available_rus()

    for user in available_users:
        if len(remaining_rus) > 0:
            ru = allocate_ru(user, remaining_rus)
            selected_users.append((user, ru))
            remaining_rus.remove(ru)

    transmit_to_users(selected_users)
```

- **Pros**: Fair, simple, predictable
- **Cons**: May not optimize throughput or latency
- **Use Case**: Uniform traffic patterns

##### Proportional Fair Scheduling
```python
# Pseudocode
for each transmission_opportunity:
    for each user:
        priority[user] = current_demand[user] / average_throughput[user]

    selected_users = []
    remaining_rus = get_available_rus()

    for user in sorted_by_priority(users):
        if len(remaining_rus) > 0:
            ru = allocate_optimal_ru(user, remaining_rus)
            selected_users.append((user, ru))
            remaining_rus.remove(ru)

    transmit_to_users(selected_users)
```

- **Pros**: Balances throughput and fairness
- **Cons**: More complex, requires tracking
- **Use Case**: Mixed traffic with varying demands

##### QoS-Aware Scheduling
```python
# Pseudocode
for each transmission_opportunity:
    high_priority_users = get_users_by_ac([AC_VO, AC_VI])
    low_priority_users = get_users_by_ac([AC_BE, AC_BK])

    selected_users = []
    remaining_rus = get_available_rus()

    # Allocate to high-priority traffic first
    for user in high_priority_users:
        if len(remaining_rus) > 0:
            ru = allocate_optimal_ru(user, remaining_rus)
            selected_users.append((user, ru))
            remaining_rus.remove(ru)

    # Fill remaining RUs with lower-priority traffic
    for user in low_priority_users:
        if len(remaining_rus) > 0:
            ru = allocate_optimal_ru(user, remaining_rus)
            selected_users.append((user, ru))
            remaining_rus.remove(ru)

    transmit_to_users(selected_users)
```

- **Pros**: Respects QoS requirements, low latency for priority traffic
- **Cons**: May starve low-priority users
- **Use Case**: Mixed traffic with strict QoS requirements (VoIP, video, data)

##### Buffer-State Driven Scheduling
```python
# Pseudocode
for each transmission_opportunity:
    # Periodically poll buffer status
    if time_to_poll:
        send_bsrp_trigger()
        collect_buffer_reports()

    selected_users = []
    remaining_rus = get_available_rus()

    # Prioritize users with larger buffers
    for user in sorted_by_buffer_size(users, descending=True):
        if buffer_size[user] > 0 and len(remaining_rus) > 0:
            ru = allocate_ru_based_on_buffer(user, remaining_rus)
            selected_users.append((user, ru))
            remaining_rus.remove(ru)

    transmit_to_users(selected_users)
```

- **Pros**: Efficient resource utilization, responsive to actual demand
- **Cons**: Overhead of buffer status polling
- **Use Case**: Dynamic, unpredictable traffic patterns

### RU Allocation Strategies

#### Static Allocation
- Fixed RU assignments per STA
- Simple, predictable
- Inefficient with varying traffic

#### Dynamic Allocation
- RU assignments change based on conditions
- Optimizes for current traffic
- Requires sophisticated scheduler

#### Hybrid Allocation
- Some RUs statically assigned (guaranteed bandwidth)
- Remaining RUs dynamically allocated
- Good balance of predictability and efficiency

### Access Categories and OFDMA

OFDMA integrates with 802.11e QoS access categories:

#### Access Categories (ACs)
- **AC_VO (Voice)**: Highest priority, low latency
- **AC_VI (Video)**: High priority, moderate latency tolerance
- **AC_BE (Best Effort)**: Default priority, no guarantees
- **AC_BK (Background)**: Lowest priority, bulk transfers

#### OFDMA QoS Integration
```
Trigger Frame Scheduling:
+----------------+----------------+----------------+----------------+
| RU 1: AC_VO    | RU 2: AC_VO    | RU 3: AC_VI    | RU 4: AC_VI    |
| (VoIP STA 1)   | (VoIP STA 2)   | (Video STA 1)  | (Video STA 2)  |
+----------------+----------------+----------------+----------------+
| RU 5: AC_BE    | RU 6: AC_BE    | RU 7: AC_BK    | RU 8: AC_BK    |
| (Web STA 1)    | (Web STA 2)    | (Download 1)   | (Download 2)   |
+----------------+----------------+----------------+----------------+
```

## Common Use Cases and Deployment Patterns

### Dense IoT Deployment

**Scenario**: Smart building with hundreds of sensors

#### Configuration
- **Channel**: 80 MHz
- **RU Pattern**: 36 × 26-tone RUs
- **Traffic**: Small, periodic sensor readings

#### Benefits
- Serve 36 sensors simultaneously
- Minimal latency for sensor reports
- Efficient power usage (sensors transmit quickly and sleep)
- Reduced contention overhead

#### Implementation Pattern
```python
# Pseudocode for IoT scheduler
def schedule_iot_uplink():
    sensors_ready = get_sensors_with_data()

    # Group sensors into batches of 36
    for batch in chunk(sensors_ready, 36):
        trigger = create_basic_trigger()

        for i, sensor in enumerate(batch):
            trigger.add_user_info(
                aid=sensor.aid,
                ru_allocation=RU_26_TONE[i],
                mcs=0,  # Robust modulation for sensors
                target_rssi=-70
            )

        send_trigger(trigger)
        receive_sensor_data(batch)
        send_multi_sta_block_ack(batch)
```

### Mixed Office Environment

**Scenario**: Office with laptops, phones, IoT devices, and video conferencing

#### Configuration
- **Channel**: 80 MHz
- **RU Pattern**: Mixed allocation based on traffic
- **Traffic**: Varied (VoIP, video, web, IoT)

#### Typical Allocation
```
80 MHz Channel:
+--------+--------+--------+--------+--------+--------+--------+--------+
| 106-RU | 106-RU | 52-RU  | 52-RU  | 26×4   | 106-RU | 106-RU | 242-RU |
| VoIP 1 | VoIP 2 |Video 1 |Video 2 |IoT×4   | Web 1  | Web 2  | DL user|
+--------+--------+--------+--------+--------+--------+--------+--------+
```

#### Benefits
- VoIP gets consistent, low-latency allocation
- Video streams receive adequate bandwidth
- IoT devices share small RUs
- Web browsing gets good throughput
- One user can get large RU for downloads

### High-Density Residential

**Scenario**: Apartment building with many overlapping networks

#### Configuration
- **Channel**: 40 or 80 MHz (depending on availability)
- **RU Pattern**: Adaptive based on active users
- **Traffic**: Streaming, gaming, browsing

#### Strategy
- Monitor active users continuously
- Allocate larger RUs during off-peak (few users)
- Switch to smaller RUs during peak (many users)
- Prioritize gaming traffic (low latency) with dedicated RUs

#### Peak Time Allocation (80 MHz, 8 active users)
```
+----------+----------+----------+----------+----------+----------+----------+----------+
| 106-RU   | 106-RU   | 106-RU   | 106-RU   | 106-RU   | 106-RU   | 106-RU   | 106-RU   |
| Stream 1 | Stream 2 | Gaming 1 | Gaming 2 | Browse 1 | Browse 2 | Stream 3 | Browse 3 |
+----------+----------+----------+----------+----------+----------+----------+----------+
```

### Public Wi-Fi / Stadium Deployment

**Scenario**: Stadium or conference venue with thousands of users

#### Configuration
- **Channel**: 160 MHz
- **RU Pattern**: Maximum granularity during peak
- **Traffic**: Social media, messaging, photo uploads

#### Ultra-Dense Mode
```
160 MHz Channel:
72 × 26-tone RUs for maximum concurrent users
```

#### Benefits
- Serve 72 users simultaneously
- Handle social media traffic efficiently (small packets)
- Reduce contention in ultra-dense environment
- Improve overall user experience

#### Scheduling Strategy
```python
# Pseudocode for stadium scheduler
def schedule_stadium_uplink():
    # Use BSRP to efficiently poll many users
    active_users = []

    # Poll in batches of 72
    for user_batch in chunk(all_associated_users, 72):
        bsrp = create_bsrp_trigger()
        for i, user in enumerate(user_batch):
            bsrp.add_user_info(
                aid=user.aid,
                ru_allocation=RU_26_TONE[i]
            )

        send_trigger(bsrp)
        buffer_reports = receive_buffer_reports()

        # Identify users with data to send
        active_users.extend([u for u in user_batch if buffer_reports[u] > 0])

    # Schedule actual data transmission for active users
    for user_batch in chunk(active_users, 72):
        trigger = create_basic_trigger()
        for i, user in enumerate(user_batch):
            trigger.add_user_info(
                aid=user.aid,
                ru_allocation=RU_26_TONE[i],
                mcs=select_mcs(user)
            )

        send_trigger(trigger)
        receive_uplink_data(user_batch)
```

### VoIP Optimization Pattern

**Scenario**: Enterprise environment with many VoIP users

#### Configuration
- **RU Size**: 52-tone or 106-tone RUs
- **Allocation**: Reserved RUs for active VoIP sessions
- **QoS**: AC_VO priority

#### Pattern
```
Dedicated VoIP UL Trigger (20ms interval):
40 MHz Channel:
+--------+--------+--------+--------+--------+--------+--------+--------+
| 52-RU  | 52-RU  | 52-RU  | 52-RU  | 52-RU  | 52-RU  | 52-RU  | 52-RU  |
| VoIP 1 | VoIP 2 | VoIP 3 | VoIP 4 | VoIP 5 | VoIP 6 | VoIP 7 | VoIP 8 |
+--------+--------+--------+--------+--------+--------+--------+--------+

Periodic (every 20ms) trigger ensures low latency for voice packets
```

#### Benefits
- Guaranteed low latency (< 20ms)
- Efficient bandwidth usage (VoIP packets are small)
- Supports many concurrent calls
- Reduces jitter through consistent scheduling

## Performance Considerations

### Efficiency Gains

#### Airtime Efficiency
Traditional OFDM (single user):
- DIFS (34 μs) + Backoff (avg 67.5 μs) + Preamble (40 μs) + Data + SIFS (16 μs) + ACK
- **Overhead per packet**: ~157.5 μs + preamble + ACK
- For 4 small packets: 4× overhead = ~630 μs

OFDMA (multi-user):
- DIFS (34 μs) + Backoff (avg 67.5 μs) + Preamble (40 μs) + Data (4 users) + SIFS (16 μs) + Multi-STA Block Ack
- **Overhead for 4 packets**: ~157.5 μs + preamble + ACK (once)
- **Efficiency gain**: ~4× for small packets

#### Throughput Improvement
- **Small Packets** (IoT, VoIP): 2-4× improvement
- **Mixed Traffic**: 1.5-2.5× improvement
- **Large Packets** (file transfers): Minimal improvement (OFDM already efficient)

### Latency Improvements

#### OFDM Latency
- Average wait time: DIFS + Average Backoff + Queue wait
- With contention: Can be 10-100ms or more in dense networks

#### OFDMA Latency
- Scheduled transmission: Deterministic, low latency
- Typical latency: 1-10ms (depending on schedule interval)
- **Improvement**: 10-100× better for small packets in dense networks

#### Real-World Example
```
VoIP Packet (100 bytes):

OFDM (dense network):
- Contention: 0-500ms (variable)
- Transmission: 0.5ms
- Total: 0.5-500ms (highly variable)

OFDMA (scheduled):
- Wait for trigger: 0-20ms (depends on schedule interval)
- Transmission: 0.3ms
- Total: 0.3-20ms (predictable)
```

### Overhead Considerations

#### Additional OFDMA Overhead
- **Trigger Frames**: Each UL OFDMA transmission requires a trigger frame
- **Preamble Overhead**: Still present for each transmission opportunity
- **Scheduler Complexity**: AP needs more processing power
- **Buffer Status Reports**: Periodic polling adds overhead

#### When OFDMA Adds Overhead
- Very low user density (1-2 users): OFDM may be more efficient
- Large packet sizes only: OFDM provides similar efficiency
- All users have poor channel conditions: Small RUs may not work well

### OFDMA vs MU-MIMO

#### When to Use OFDMA
- Many users with small packets
- Mixed packet sizes
- IoT and sensor networks
- High user density with varying QoS needs
- Users with single antenna or limited MIMO capability

#### When to Use MU-MIMO
- Few users (2-4) with large packets
- All users have multiple antennas
- Good spatial separation between users
- High SNR environment
- Maximum throughput to small number of users

#### When to Combine Both
- Very high density with mixed capabilities
- Maximum spectral efficiency required
- Some users have MIMO, others don't
- Complex traffic patterns (voice + video + data)

### Optimal OFDMA Configuration

#### General Guidelines

##### Low Density (1-4 users)
- Use larger RUs (242, 484, 996)
- Consider standard OFDM or MU-MIMO
- OFDMA benefits are minimal

##### Medium Density (5-10 users)
- Use medium RUs (52, 106, 242)
- Mix of OFDMA and MU-MIMO
- Good balance of efficiency and throughput

##### High Density (10-20 users)
- Use small to medium RUs (26, 52, 106)
- Primarily OFDMA
- Focus on latency and fairness

##### Ultra-High Density (20+ users)
- Use smallest RUs (26, 52)
- Pure OFDMA strategy
- Maximize concurrent users

## Implementation and Configuration

### Driver and Firmware Considerations

#### Capabilities Negotiation
During association, STAs and APs negotiate OFDMA capabilities through HE Capabilities element:

```
HE Capabilities Element:
+------------------+
| MAC Capabilities |  (indicates OFDMA support)
+------------------+
| PHY Capabilities |  (indicates supported RU sizes)
+------------------+
| Supported MCS    |
+------------------+
```

Key fields:
- **Triggered SU/MU Beamforming Feedback**: Support for OFDMA feedback
- **HE SU/MU PPDU with 4× HE-LTF**: Support for longer training fields
- **Max Number of Supported Users**: Maximum RUs in multi-user transmission
- **RU Allocation**: Bitmap of supported RU sizes

#### Driver Interfaces (Linux example)

##### Enable OFDMA in hostapd
```bash
# /etc/hostapd/hostapd.conf

# Enable Wi-Fi 6
ieee80211ax=1

# Enable OFDMA
he_su_beamformer=1
he_su_beamformee=1
he_mu_beamformer=1

# OFDMA specific
he_default_pe_duration=4
he_twt_required=0

# Multi-user settings
he_rts_threshold=1023
mu_edca_qos_info_param_count=0
mu_edca_qos_info_q_ack=0

# Per-AC EDCA parameters for MU
mu_edca_ac_be_aifsn=8
mu_edca_ac_be_aci=0
mu_edca_ac_be_ecwmin=9
mu_edca_ac_be_ecwmax=10
mu_edca_ac_be_timer=255
```

##### Query OFDMA Status
```bash
# Check if interface supports OFDMA
iw dev wlan0 info | grep -i "HE\|ax"

# View detailed HE capabilities
iw phy phy0 info | grep -A 20 "HE Iftypes"

# Check connected stations HE capabilities
iw dev wlan0 station dump | grep -i "HE\|rx\|tx"
```

##### Enable OFDMA in wpa_supplicant (Client)
```bash
# /etc/wpa_supplicant/wpa_supplicant.conf

network={
    ssid="MyWiFi6Network"
    psk="password"

    # Enable Wi-Fi 6 features
    ieee80211ax=1

    # OFDMA support
    he_su_beamformee=1
}
```

#### Firmware Parameters

Many vendors provide firmware-level tuning:

##### Example: Qualcomm firmware parameters
```bash
# Enable aggressive OFDMA scheduling
iwpriv ath0 he_ul_ofdma 1
iwpriv ath0 he_dl_ofdma 1

# OFDMA RU allocation mode
# 0 = disabled, 1 = auto, 2 = force
iwpriv ath0 he_ul_ofdma_mode 1

# Minimum users for OFDMA activation
iwpriv ath0 he_ofdma_min_users 2
```

##### Example: Intel firmware parameters
```bash
# Load iwlwifi with OFDMA enabled
modprobe iwlwifi enable_ax=1

# Check module parameters
cat /sys/module/iwlwifi/parameters/enable_ax
```

### Configuration Parameters

#### Key Tuning Parameters

##### RU Allocation Threshold
Minimum number of users before OFDMA is activated:
```
ofdma_min_users=2  # Don't use OFDMA for single user
```

##### RU Size Selection
Configure preferred RU sizes based on deployment:
```python
# Pseudocode configuration
config = {
    'iot_deployment': {
        'preferred_ru_sizes': [26, 52],
        'max_users_per_txop': 36
    },
    'enterprise': {
        'preferred_ru_sizes': [52, 106, 242],
        'max_users_per_txop': 16
    },
    'residential': {
        'preferred_ru_sizes': [106, 242, 484],
        'max_users_per_txop': 8
    }
}
```

##### Trigger Frame Interval
How often to send trigger frames for uplink:
```
trigger_interval_ms=10  # 10ms for VoIP
trigger_interval_ms=50  # 50ms for general traffic
trigger_interval_ms=100 # 100ms for background traffic
```

##### BSRP Polling Interval
How often to poll buffer status:
```
bsrp_interval_ms=500  # Poll every 500ms
bsrp_on_demand=true   # Also poll when needed
```

### Debugging and Monitoring

#### Monitor OFDMA Performance

##### Using iw (Linux)
```bash
# View detailed station statistics
iw dev wlan0 station get <MAC_ADDRESS>

# Look for HE-specific stats:
# - rx HE-MCS
# - tx HE-MCS
# - HE RU allocation histogram
```

##### Using Wireshark
1. Capture on monitor mode interface
2. Filter for HE frames: `wlan.fc.type == 0 && wlan.ext_tag.number == 35`
3. Examine HE PPDU format and RU allocations
4. Analyze trigger frames: `wlan.ext_tag.number == 35 && wlan.ext_tag.he.trigger_type`

#### Common Debug Filters
```
# Show all trigger frames
wlan.fc.type_subtype == 0x02 && wlan.ext_tag.he.trigger_type

# Show HE MU PPDUs (multi-user)
wlan.ext_tag.he.ppdu_format == 2

# Show RU allocations
wlan.ext_tag.he.ru_allocation
```

#### Performance Metrics

##### Key Metrics to Monitor
- **OFDMA Utilization**: Percentage of transmissions using OFDMA
- **Average RU Size**: Indicates typical allocation pattern
- **Users per TXOP**: How many users served simultaneously
- **Trigger Frame Overhead**: Trigger frames / data frames ratio
- **Latency Distribution**: Latency histogram for different traffic types
- **Throughput per User**: Individual user throughput
- **Airtime Efficiency**: Data transmitted / total airtime

##### Example Monitoring Script (Pseudocode)
```python
def monitor_ofdma_performance():
    stats = {
        'total_txop': 0,
        'ofdma_txop': 0,
        'ru_size_histogram': {},
        'users_per_txop': []
    }

    while monitoring:
        frame = capture_frame()

        if is_he_mu_ppdu(frame):
            stats['total_txop'] += 1
            stats['ofdma_txop'] += 1

            ru_allocations = parse_ru_allocations(frame)
            for ru in ru_allocations:
                stats['ru_size_histogram'][ru.size] = \
                    stats['ru_size_histogram'].get(ru.size, 0) + 1

            stats['users_per_txop'].append(len(ru_allocations))

        elif is_su_ppdu(frame):
            stats['total_txop'] += 1

    ofdma_utilization = stats['ofdma_txop'] / stats['total_txop']
    avg_users = mean(stats['users_per_txop'])

    print(f"OFDMA Utilization: {ofdma_utilization:.1%}")
    print(f"Average users per TXOP: {avg_users:.1f}")
    print(f"RU size distribution: {stats['ru_size_histogram']}")
```

### Troubleshooting Common Issues

#### Issue: OFDMA Not Activating

**Symptoms**: All transmissions use single-user OFDM

**Causes**:
- Insufficient number of users (below ofdma_min_users threshold)
- STAs don't support OFDMA (not Wi-Fi 6 capable)
- OFDMA disabled in configuration
- Channel width too narrow (20 MHz with few users)

**Solutions**:
```bash
# Verify AP supports OFDMA
iw phy phy0 info | grep "HE MAC Capabilities" -A 10

# Check connected stations
for sta in $(iw dev wlan0 station dump | grep Station | awk '{print $2}'); do
    echo "=== $sta ==="
    iw dev wlan0 station get $sta | grep -i "HE\|ax"
done

# Verify configuration
grep -i "he\|ax\|ofdma" /etc/hostapd/hostapd.conf

# Enable OFDMA explicitly
# Add to hostapd.conf:
# he_mu_beamformer=1
```

#### Issue: High Latency Despite OFDMA

**Symptoms**: Latency still high with OFDMA enabled

**Causes**:
- Trigger interval too long
- Poor scheduler algorithm
- Excessive BSRP polling overhead
- Channel contention from overlapping networks

**Solutions**:
```bash
# Reduce trigger interval for latency-sensitive traffic
# In hostapd or vendor-specific configuration:
trigger_interval_voip=10    # 10ms for voice
trigger_interval_video=20   # 20ms for video

# Adjust BSRP polling
bsrp_interval=250          # Poll every 250ms instead of 500ms
bsrp_on_trigger=true       # Combine with data triggers

# Optimize channel selection to avoid interference
iw dev wlan0 survey dump   # Check channel utilization
# Select least congested channel
```

#### Issue: Lower Throughput with OFDMA

**Symptoms**: Total throughput decreased after enabling OFDMA

**Causes**:
- Using OFDMA with very few users (overhead exceeds benefit)
- All traffic is large packets (OFDM more efficient)
- Poor RU allocation (too small RUs for high-throughput users)

**Solutions**:
```bash
# Adjust OFDMA activation threshold
ofdma_min_users=4  # Only use OFDMA with 4+ active users

# Use adaptive scheduling
scheduler_mode=adaptive  # Switch between OFDM/OFDMA based on traffic

# Configure RU size selection
min_ru_size_high_throughput=242  # Use larger RUs for bulk transfers
```

#### Issue: Frequent Trigger Frame Failures

**Symptoms**: Many trigger frames not resulting in uplink transmissions

**Causes**:
- Poor channel quality
- Incorrect target RSSI in trigger frames
- STAs in power save mode
- Buffer status reports stale

**Solutions**:
```bash
# Adjust target RSSI
ul_target_rssi=-75  # More conservative target

# Increase BSRP frequency
bsrp_interval=200   # Poll more frequently for accurate buffer status

# Enable TWT (Target Wake Time) for better power save coordination
he_twt_required=1

# Monitor and log trigger frame success rate
enable_trigger_logging=1
```

## Advanced Topics

### Multi-AP Coordination

In dense deployments, multiple APs can coordinate OFDMA:

#### Spatial Reuse with OFDMA
- **BSS Coloring**: Differentiate overlapping BSSs
- **OBSS PD (Overlapping BSS Preamble Detection)**: Adjust CCA thresholds
- **Coordinated Scheduling**: Multiple APs schedule non-interfering RUs

#### Example Configuration
```bash
# Enable BSS coloring
he_bss_color=5  # Color this BSS (1-63)

# OBSS PD parameters
he_obss_pd_min_threshold=-82
he_obss_pd_max_threshold=-62
```

### Wi-Fi 7 OFDMA Enhancements

#### Multi-RU to Single STA
Wi-Fi 7 allows allocating multiple RUs to a single user:
```
80 MHz Channel:
+----------+----------+----------+----------+
| 242-RU   | 242-RU   | 242-RU   | 242-RU   |
| STA A    | STA A    | STA B    | STA C    |
| (MRU)    | (MRU)    |          |          |
+----------+----------+----------+----------+
```

STA A receives two non-contiguous 242-RUs for 2× bandwidth

#### Preamble Puncturing
Wi-Fi 7 can puncture (skip) interfered 20 MHz sub-channels:
```
80 MHz Channel with interference on sub-channel 2:
+----------+----------+----------+----------+
| 20 MHz   | 20 MHz   | 20 MHz   | 20 MHz   |
| RUs      | PUNCTURE | RUs      | RUs      |
| Active   | (Interf) | Active   | Active   |
+----------+----------+----------+----------+
```

Still use 60 MHz effectively despite 20 MHz interference

### Future Directions

- **AI-Driven Scheduling**: Machine learning for optimal RU allocation
- **Predictive OFDMA**: Anticipate traffic patterns for proactive scheduling
- **Enhanced Multi-AP OFDMA**: Better coordination across APs
- **Dynamic RU Sizing**: Real-time RU size adjustment based on conditions

## Summary

OFDMA represents a significant evolution in Wi-Fi technology, enabling:

1. **Multi-user efficiency**: Serve many users simultaneously
2. **Reduced latency**: Especially for small packets
3. **Better resource utilization**: Adaptive allocation based on needs
4. **Improved dense deployment performance**: Handle more concurrent users
5. **Power efficiency**: Devices transmit quickly and sleep longer

**Key Takeaways**:
- OFDMA divides channels into Resource Units (RUs) of various sizes
- Downlink and uplink OFDMA enable multi-user concurrent transmission
- Trigger frames coordinate uplink OFDMA transmissions
- Scheduling algorithms determine RU allocation and user selection
- Best for dense deployments with mixed traffic patterns
- Combine with MU-MIMO for maximum spectral efficiency
- Proper configuration and monitoring essential for optimal performance

OFDMA is most beneficial when:
- Multiple users with varying bandwidth needs
- Small packet sizes (IoT, VoIP, messaging)
- High user density
- Mixed QoS requirements
- Latency-sensitive applications

For maximum Wi-Fi 6/7 performance, understanding and properly configuring OFDMA is essential.
