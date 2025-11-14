# RTP (Real-Time Transport Protocol)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [RTP vs Other Protocols](#rtp-vs-other-protocols)
- [RTP Packet Format](#rtp-packet-format)
- [How RTP Works](#how-rtp-works)
- [RTCP (RTP Control Protocol)](#rtcp-rtp-control-protocol)
- [Payload Types and Codecs](#payload-types-and-codecs)
- [Code Examples](#code-examples)
- [Jitter Buffer Management](#jitter-buffer-management)
- [Packet Loss Handling](#packet-loss-handling)
- [RTP Extensions](#rtp-extensions)
- [Security: SRTP](#security-srtp)
- [Integration with Other Protocols](#integration-with-other-protocols)
- [Common Use Cases](#common-use-cases)
- [Advanced Topics](#advanced-topics)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [RTP Libraries and Tools](#rtp-libraries-and-tools)
- [ELI10](#eli10)
- [Further Resources](#further-resources)

---

## Overview

**RTP (Real-Time Transport Protocol)** is a network protocol designed for delivering audio and video over IP networks in real-time. Defined in RFC 3550, RTP provides end-to-end delivery services for data with real-time characteristics, such as interactive audio and video.

### What is RTP?

RTP is **not** a complete transport protocol by itself. Instead, it's designed to work on top of UDP, providing:

- **Payload type identification** - Indicates the format of the data (codec)
- **Sequence numbering** - Allows detection of packet loss and out-of-order delivery
- **Timestamping** - Enables synchronization and jitter calculations
- **Source identification** - Identifies the sender of a stream

**Key Point**: RTP does NOT guarantee delivery, quality of service, or in-order delivery. It provides the mechanisms to detect and handle these issues at the application level.

### RTP and RTCP Relationship

RTP works together with **RTCP (RTP Control Protocol)**:

- **RTP**: Carries the actual media data (audio/video packets)
- **RTCP**: Provides out-of-band control information (quality feedback, participant information)

Think of them as partners:
- RTP = The delivery trucks carrying packages
- RTCP = The quality control reports and delivery confirmations

### Why RTP Exists

Before RTP, applications had to build custom solutions for real-time media. RTP provides:

1. **Standardization**: Common format for real-time media transport
2. **Codec Independence**: Works with any audio/video codec
3. **Synchronization**: Enables lip-sync between audio and video
4. **Quality Monitoring**: Via RTCP feedback
5. **Mixing/Translation**: Support for multiparty scenarios
6. **Scalability**: From peer-to-peer to large broadcasts

### Primary Use Cases

- **VoIP (Voice over IP)**: Telephone calls over the internet
- **Video Conferencing**: Zoom, Teams, Google Meet
- **Live Streaming**: Broadcast media delivery
- **WebRTC**: Browser-to-browser real-time communication
- **IPTV**: Television over IP networks
- **Gaming**: Voice chat in multiplayer games

---

## Key Features

### 1. Real-Time Delivery

RTP is optimized for real-time delivery, not reliability:
- Uses UDP (not TCP) for low latency
- No retransmissions by default (optional RTX extension)
- Prioritizes timeliness over completeness

### 2. Payload Flexibility

RTP can carry any codec:
- **Audio**: Opus, G.711, AAC, AMR
- **Video**: H.264, VP8, VP9, AV1
- **Other**: Text, application data

### 3. Timing Information

Each packet includes:
- **Timestamp**: When the data was sampled (not when sent)
- **Clock rate**: Specific to the codec (e.g., 48000 Hz for Opus)
- Enables jitter buffer and synchronization

### 4. Sequence Numbering

- Increments by 1 for each packet sent
- Detects packet loss (gaps in sequence)
- Detects out-of-order delivery
- Detects duplicate packets

### 5. Source Identification

- **SSRC (Synchronization Source)**: Unique identifier for each stream
- **CSRC (Contributing Source)**: Lists sources in mixed streams
- Enables multiple streams in one session

### 6. Quality Feedback (via RTCP)

- Packet loss statistics
- Jitter measurements
- Round-trip time
- Bandwidth usage

---

## RTP vs Other Protocols

| Feature | RTP | TCP | UDP | RTCP |
|---------|-----|-----|-----|------|
| **Purpose** | Real-time media transport | Reliable data transfer | Unreliable datagram | Control/feedback for RTP |
| **Reliability** | No (optional RTX) | Yes (guaranteed) | No | No |
| **Ordering** | Sequence numbers | Yes (guaranteed) | No | N/A |
| **Latency** | Low | Variable (retransmits) | Low | Low |
| **Use Case** | Audio/video streaming | File transfer, web | DNS, gaming | Quality monitoring |
| **Overhead** | 12+ bytes | 20+ bytes | 8 bytes | Variable |
| **Transport** | Over UDP | Direct IP | Direct IP | Over UDP |
| **Timing** | Timestamps | No | No | Yes (SR packets) |
| **Bandwidth** | Adaptive | Flow control | None | Reports usage |

### When to Use RTP

**Use RTP when:**
- Transporting real-time audio or video
- Latency is critical (< 200ms target)
- Some packet loss is acceptable (1-3%)
- Need synchronization between streams
- Interoperability with VoIP/video systems

**Don't use RTP when:**
- Transferring files (use TCP/HTTP)
- Every packet is critical (use TCP)
- Not time-sensitive data
- Simple request-response patterns

---

## RTP Packet Format

### Header Structure

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|X|  CC   |M|     PT      |       Sequence Number         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           Timestamp                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Synchronization Source (SSRC) Identifier            |
+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
|            Contributing Source (CSRC) Identifiers             |
|                             ....                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Header Extension (optional)                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                            Payload                            |
|                             ....                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Field Descriptions

#### Version (V) - 2 bits
- **Value**: Always 2 for current RTP
- Identifies the RTP version

#### Padding (P) - 1 bit
- **0**: No padding
- **1**: Packet contains padding bytes at the end
- Last byte indicates padding length
- Used for encryption block alignment

#### Extension (X) - 1 bit
- **0**: No header extension
- **1**: Header extension follows fixed header
- Allows custom additions (audio level, video orientation, etc.)

#### CSRC Count (CC) - 4 bits
- **Value**: 0-15
- Number of CSRC identifiers following the SSRC
- Used in mixed streams (conference servers)

#### Marker (M) - 1 bit
- **Meaning**: Codec-specific
- **Audio**: Typically marks start of talk burst
- **Video**: Marks end of video frame
- **Usage**: Application-defined boundary marker

#### Payload Type (PT) - 7 bits
- **Value**: 0-127
- Identifies the codec/format of payload
- **0-95**: Static assignments (e.g., 0=PCMU, 8=PCMA)
- **96-127**: Dynamic assignments (negotiated via SDP)

#### Sequence Number - 16 bits
- **Value**: 0-65535, wraps around
- Increments by 1 for each packet sent
- **Uses**:
  - Detect packet loss (gaps)
  - Detect duplicates
  - Restore packet order
  - Initial value is random (security)

#### Timestamp - 32 bits
- **Value**: Sampling instant of first byte in payload
- Increments based on **clock rate** (codec-specific)
- **Not** wall-clock time
- **Examples**:
  - Audio (48kHz): Increments by 960 for 20ms packet
  - Video (90kHz): Increments by 3000 for 33ms frame
- Used for jitter calculation and synchronization

#### SSRC (Synchronization Source) - 32 bits
- **Unique identifier** for the source of the stream
- Randomly chosen to avoid collisions
- Stays constant for duration of session
- Different streams (audio/video) have different SSRCs

#### CSRC (Contributing Source) - 0-15 items, 32 bits each
- Lists sources that contributed to mixed stream
- Example: Conference server mixing 3 participants
- Count specified in CC field
- Rarely used in peer-to-peer scenarios

### Header Extension Format

When X=1, extension follows SSRC/CSRC:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|      Defined by Profile       |           Length              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Extension Data                         |
|                             ....                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Example Packet Breakdown

```
Hex: 80 08 1a 2b 00 00 03 e8 ab cd ef 01 ...

Binary breakdown:
80 = 10000000
     10 = Version 2
       0 = No padding
        0 = No extension
         0000 = CC=0 (no CSRC)

08 = 00001000
     0 = Marker=0
      0001000 = PT=8 (PCMA/G.711 A-law)

1a 2b = Sequence number = 6699

00 00 03 e8 = Timestamp = 1000

ab cd ef 01 = SSRC = 2882400001
```

---

## How RTP Works

### Session Establishment

RTP itself doesn't establish sessions. That's done by signaling protocols:

1. **SDP (Session Description Protocol)**: Describes media parameters
2. **SIP/SDP**: VoIP call setup
3. **WebRTC**: ICE/DTLS/SDP negotiation

**Example SDP for Audio:**
```
v=0
o=- 123456 123456 IN IP4 192.168.1.100
s=Audio Call
c=IN IP4 192.168.1.100
t=0 0
m=audio 5004 RTP/AVP 111
a=rtpmap:111 opus/48000/2
a=fmtp:111 minptime=10;useinbandfec=1
```

**Breakdown:**
- `m=audio 5004 RTP/AVP 111`: Audio on port 5004, payload type 111
- `a=rtpmap:111 opus/48000/2`: PT 111 = Opus, 48kHz, stereo
- RTP on even port 5004, RTCP on odd port 5005 (convention)

### Port Allocation

**Convention:**
- RTP uses **even** port numbers (e.g., 5004, 16384)
- RTCP uses **odd** port numbers (e.g., 5005, 16385)
- RTCP port = RTP port + 1

**Modern approach (RTP/RTCP Multiplexing):**
- Both RTP and RTCP on same port (WebRTC)
- Distinguishes using packet type

### Packet Flow

```
Sender                                    Receiver
  |                                           |
  | [1] Capture audio/video frame             |
  |                                           |
  | [2] Encode with codec (Opus, H.264)       |
  |                                           |
  | [3] Packetize into RTP packets            |
  |     - Add RTP header                      |
  |     - Set timestamp, sequence, PT         |
  |                                           |
  | [4] Send RTP packet over UDP              |
  |------------------------------------------>|
  |                                           | [5] Receive UDP packet
  |                                           |
  |                                           | [6] Parse RTP header
  |                                           |     - Check sequence
  |                                           |     - Extract timestamp
  |                                           |
  |                                           | [7] Buffer in jitter buffer
  |                                           |     - Absorb network jitter
  |                                           |
  |                                           | [8] Decode payload
  |                                           |
  |                                           | [9] Play audio/video
  |                                           |
  | [10] Periodic RTCP reports                |
  |<----------------------------------------->|
  |   (quality feedback, statistics)          |
```

### Timestamps and Clock Rates

**Timestamp Calculation:**

```
timestamp = previous_timestamp + (samples_in_packet)
```

**Clock Rates by Codec:**

| Codec | Clock Rate | Typical Packet Duration | Timestamp Increment |
|-------|------------|------------------------|-------------------|
| Opus | 48000 Hz | 20ms | 960 |
| G.711 (PCMU/PCMA) | 8000 Hz | 20ms | 160 |
| AAC | 90000 Hz | Variable | Variable |
| H.264 (video) | 90000 Hz | 33ms (30fps) | 3000 |
| VP8/VP9 (video) | 90000 Hz | 33ms (30fps) | 3000 |

**Example (Opus audio at 48kHz):**
```
Packet 1: timestamp = 0
Packet 2: timestamp = 960     (20ms * 48000 Hz = 960)
Packet 3: timestamp = 1920
Packet 4: timestamp = 2880
...
```

**Key Points:**
- Timestamp is based on **sampling time**, not sending time
- Clock rate is **codec-specific**
- Video typically uses 90kHz (historical MPEG convention)
- Timestamps enable jitter calculation and synchronization

### Synchronization Between Streams

For lip-sync (audio-video synchronization):

1. Each stream has different SSRC
2. Both use same **NTP timeline** (via RTCP SR)
3. Receiver correlates timestamps to NTP time
4. Aligns playback based on NTP correlation

**Example:**
```
Audio SSRC: 0x12345678
Video SSRC: 0x87654321

RTCP SR (Audio):
  NTP time: 1234567890.500000
  RTP timestamp: 48000

RTCP SR (Video):
  NTP time: 1234567890.500000  (same wall time)
  RTP timestamp: 90000

Receiver can now sync both streams to same timeline
```

---

## RTCP (RTP Control Protocol)

RTCP is RTP's companion protocol for quality monitoring and control. While RTP carries media, RTCP carries statistics about the RTP session.

### Key Functions

1. **Quality Feedback**: Packet loss, jitter, delay
2. **Participant Identification**: Names, email, etc.
3. **Session Control**: Notify when leaving
4. **Feedback for Congestion Control**: Adapt bitrate based on reports

### RTCP Packet Types

| Type | Name | Purpose |
|------|------|---------|
| 200 | SR (Sender Report) | Statistics from active senders |
| 201 | RR (Receiver Report) | Statistics from receivers |
| 202 | SDES (Source Description) | Participant information (name, email) |
| 203 | BYE | Leaving session |
| 204 | APP | Application-specific messages |

### Sender Report (SR) - Type 200

Sent by active senders (those transmitting RTP):

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|    RC   |   PT=SR=200   |             Length            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         SSRC of Sender                        |
+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
|              NTP Timestamp (most significant word)            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              NTP Timestamp (least significant word)           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         RTP Timestamp                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Sender's Packet Count                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Sender's Octet Count                     |
+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
|                 Report Blocks (0 or more)                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Fields:**
- **NTP Timestamp**: Wall-clock time when report sent
- **RTP Timestamp**: Corresponds to NTP time (for sync)
- **Packet Count**: Total packets sent
- **Octet Count**: Total bytes sent
- **Report Blocks**: Reception quality from this sender

### Receiver Report (RR) - Type 201

Sent by receivers (not actively sending):

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|    RC   |   PT=RR=201   |             Length            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     SSRC of Packet Sender                     |
+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
|                 Report Blocks (0 or more)                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Report Block Format

Each SR/RR can contain multiple report blocks (one per source):

```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 SSRC of Source Being Reported                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Fraction Lost |       Cumulative Packets Lost                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Extended Highest Sequence Number Received           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Interarrival Jitter                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Last SR (LSR)                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Delay Since Last SR (DLSR)                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Key Metrics:**

1. **Fraction Lost** (8 bits): Packet loss since last report
   - Value: 0-255 (0 = 0%, 255 = 100%)
   - Formula: `(packets_lost / packets_expected) * 256`

2. **Cumulative Packets Lost** (24 bits): Total lost since start
   - Can be negative (duplicates exceed losses)

3. **Extended Highest Sequence** (32 bits):
   - Highest sequence number received
   - Plus cycle count (upper 16 bits)

4. **Interarrival Jitter** (32 bits):
   - Statistical variance of packet arrival times
   - Lower is better (smoother delivery)

5. **LSR/DLSR**: For calculating round-trip time
   - LSR = Middle 32 bits of NTP timestamp from last SR
   - DLSR = Delay since receiving that SR
   - RTT = (current_time - LSR - DLSR)

### Jitter Calculation

```
J(i) = J(i-1) + (|D(i-1, i)| - J(i-1)) / 16

Where:
D(i-1, i) = (R_i - R_{i-1}) - (S_i - S_{i-1})
R_i = Receive timestamp of packet i
S_i = Send timestamp of packet i (from RTP header)
```

**In plain English:**
Jitter measures how consistently packets arrive. High jitter = inconsistent timing.

### SDES (Source Description) - Type 202

Contains participant information:

```
Items:
- CNAME (Canonical Name): user@host.domain
- NAME: Full name
- EMAIL: Email address
- PHONE: Phone number
- LOC: Geographic location
- TOOL: Application/tool name
- NOTE: Transient messages
```

**Example:**
```
CNAME: alice@192.168.1.100
NAME: Alice Smith
TOOL: MyVoIPApp 1.0
```

### BYE Packet - Type 203

Indicates participant is leaving:

```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|    SC   |   PT=BYE=203  |             Length            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           SSRC/CSRC                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Length    |               Reason for leaving              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Usage:**
- Clean session termination
- Allows receivers to free resources quickly
- Optional reason string (e.g., "User disconnected")

### RTCP Bandwidth Management

**Rules** (to prevent RTCP from overwhelming network):

1. **RTCP bandwidth d 5% of RTP bandwidth**
2. **Senders get 25% of RTCP bandwidth**
3. **Receivers share remaining 75%**
4. **Minimum interval between reports: 5 seconds**

**Calculation:**

```python
def rtcp_interval(members, senders, rtcp_bw, we_sent):
    """Calculate RTCP report interval"""
    # Constants
    RTCP_MIN_TIME = 5.0  # seconds
    COMPENSATION = 2.71828  # e

    # Fraction for senders
    if we_sent:
        rtcp_fraction = 0.25
    else:
        rtcp_fraction = 0.75

    # Average packet size (assume 200 bytes for RTCP)
    avg_rtcp_size = 200

    # Calculate interval
    n = members
    t = (n * avg_rtcp_size) / (rtcp_fraction * rtcp_bw)
    t = max(t, RTCP_MIN_TIME)

    # Randomize to prevent synchronization
    # Actual interval: [0.5*t, 1.5*t]
    import random
    return t * (random.random() + 0.5)
```

**Example:**
```
10 participants, 256 kbps audio stream
RTP bandwidth = 256 kbps
RTCP bandwidth = 5% = 12.8 kbps

Average RTCP report interval H 5-10 seconds
```

---

## Payload Types and Codecs

RTP can carry any media format. The **Payload Type (PT)** field identifies the codec.

### Static Payload Types (0-95)

Defined in RFC 3551, permanently assigned:

| PT | Codec | Type | Clock Rate | Channels | Bitrate |
|----|-------|------|-----------|----------|---------|
| 0 | PCMU (G.711 μ-law) | Audio | 8000 Hz | 1 | 64 kbps |
| 3 | GSM | Audio | 8000 Hz | 1 | 13 kbps |
| 4 | G.723 | Audio | 8000 Hz | 1 | 5.3/6.3 kbps |
| 8 | PCMA (G.711 A-law) | Audio | 8000 Hz | 1 | 64 kbps |
| 9 | G.722 | Audio | 8000 Hz | 1 | 64 kbps |
| 18 | G.729 | Audio | 8000 Hz | 1 | 8 kbps |
| 26 | JPEG | Video | 90000 Hz | - | Variable |
| 31 | H.261 | Video | 90000 Hz | - | Variable |
| 32 | MPV (MPEG-1/2 Video) | Video | 90000 Hz | - | Variable |
| 34 | H.263 | Video | 90000 Hz | - | Variable |

### Dynamic Payload Types (96-127)

Negotiated via SDP for modern codecs:

| PT | Codec | Type | Clock Rate | Notes |
|----|-------|------|-----------|-------|
| 96-127 | Opus | Audio | 48000 Hz | Recommended for WebRTC |
| 96-127 | H.264 | Video | 90000 Hz | Most common video codec |
| 96-127 | VP8 | Video | 90000 Hz | WebRTC video |
| 96-127 | VP9 | Video | 90000 Hz | Better compression than VP8 |
| 96-127 | AV1 | Video | 90000 Hz | Next-gen codec |
| 96-127 | AAC | Audio | Variable | High-quality audio |

### SDP Payload Type Mapping

**Example SDP with multiple codecs:**

```
m=audio 5004 RTP/AVP 111 0 8
a=rtpmap:111 opus/48000/2
a=fmtp:111 minptime=10;useinbandfec=1
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000

m=video 5006 RTP/AVP 96 97
a=rtpmap:96 VP8/90000
a=rtpmap:97 H264/90000
a=fmtp:97 profile-level-id=42e01f;packetization-mode=1
```

**Breakdown:**
- Audio: PT 111=Opus, 0=PCMU, 8=PCMA
- Video: PT 96=VP8, 97=H.264
- `a=fmtp`: Format-specific parameters
- Endpoints negotiate which to use

### Audio Codec Comparison

| Codec | Bitrate | Latency | Quality | Complexity | Use Case |
|-------|---------|---------|---------|-----------|----------|
| **Opus** | 6-510 kbps | 5-66 ms | Excellent | Medium | Recommended for all |
| **G.711** | 64 kbps | 0.125 ms | Good | Very low | Legacy VoIP |
| **G.722** | 64 kbps | Low | Very good | Low | HD VoIP |
| **AAC** | 64-320 kbps | Medium | Excellent | High | Streaming, music |
| **AMR-WB** | 6.6-23.85 kbps | Low | Good | Low | Mobile networks |
| **iLBC** | 13.3/15.2 kbps | 20-30 ms | Fair | Low | Lossy networks |

**Recommendation**: Use **Opus** for new implementations (best quality/bitrate ratio, low latency).

### Video Codec Comparison

| Codec | Bitrate (1080p) | Compression | Complexity | Use Case |
|-------|----------------|-------------|-----------|----------|
| **H.264** | 2-5 Mbps | Good | Medium | Universal support |
| **VP8** | 2-6 Mbps | Good | Medium | WebRTC, open-source |
| **VP9** | 1-3 Mbps | Better | High | YouTube, streaming |
| **AV1** | 0.5-2 Mbps | Best | Very high | Future, streaming |
| **H.265/HEVC** | 1-3 Mbps | Better | High | 4K streaming |

**Recommendation**:
- **WebRTC**: H.264 or VP8 (best compatibility)
- **Streaming**: VP9 or AV1 (better compression)
- **Universal**: H.264 (widest support)

### Packetization Examples

#### Audio Packetization (Opus)

```
Audio frame: 20ms of audio at 48kHz
Samples: 20ms * 48000 Hz = 960 samples
Encoded size: ~40 bytes (at 16 kbps)

RTP Packet:
[12 byte RTP header][40 byte Opus payload]

Timestamp increment: 960 (for next packet)
```

#### Video Packetization (H.264)

```
Video frame: 1920x1080, encoded to 10 KB
Too large for single packet (MTU typically 1500 bytes)

Solution: Fragmentation (FU-A)
Packet 1: [RTP header][FU-A header][Fragment 1 (1400 bytes)]
Packet 2: [RTP header][FU-A header][Fragment 2 (1400 bytes)]
...
Packet 8: [RTP header][FU-A header][Fragment 8 (400 bytes)]

All packets have SAME timestamp (same frame)
Sequence numbers increment: 1, 2, 3, ...
Marker bit set on LAST packet of frame
```

---

## Code Examples

### Python: Basic RTP Sender

```python
import socket
import struct
import time

class RTPSender:
    def __init__(self, dest_ip, dest_port, payload_type=96, ssrc=None):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.payload_type = payload_type
        self.ssrc = ssrc or random.randint(0, 0xFFFFFFFF)

        # RTP state
        self.sequence = random.randint(0, 0xFFFF)
        self.timestamp = random.randint(0, 0xFFFFFFFF)

        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def create_rtp_packet(self, payload, marker=False):
        """Create RTP packet with given payload"""
        # RTP header (12 bytes)
        version = 2
        padding = 0
        extension = 0
        csrc_count = 0
        marker_bit = 1 if marker else 0

        # Byte 0: V(2), P(1), X(1), CC(4)
        byte0 = (version << 6) | (padding << 5) | (extension << 4) | csrc_count

        # Byte 1: M(1), PT(7)
        byte1 = (marker_bit << 7) | self.payload_type

        # Pack header
        header = struct.pack(
            '!BBHII',
            byte0,                    # V, P, X, CC
            byte1,                    # M, PT
            self.sequence,            # Sequence number
            self.timestamp,           # Timestamp
            self.ssrc                 # SSRC
        )

        return header + payload

    def send_packet(self, payload, marker=False, timestamp_increment=960):
        """Send RTP packet"""
        packet = self.create_rtp_packet(payload, marker)
        self.sock.sendto(packet, (self.dest_ip, self.dest_port))

        # Update state
        self.sequence = (self.sequence + 1) & 0xFFFF
        self.timestamp = (self.timestamp + timestamp_increment) & 0xFFFFFFFF

    def close(self):
        self.sock.close()


# Example usage: Send audio packets
if __name__ == '__main__':
    import random

    sender = RTPSender('127.0.0.1', 5004, payload_type=111)  # PT 111 = Opus

    # Simulate sending audio packets (20ms each, 48kHz)
    for i in range(100):
        # Generate dummy audio payload (40 bytes for 16kbps Opus)
        audio_data = bytes([random.randint(0, 255) for _ in range(40)])

        # Send packet with 960 timestamp increment (20ms at 48kHz)
        sender.send_packet(audio_data, marker=False, timestamp_increment=960)

        print(f"Sent packet {i+1}, seq={sender.sequence-1}, ts={sender.timestamp-960}")

        # Wait 20ms between packets
        time.sleep(0.020)

    sender.close()
```

### Python: Basic RTP Receiver

```python
import socket
import struct

class RTPReceiver:
    def __init__(self, listen_port):
        self.listen_port = listen_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', listen_port))

        # Statistics
        self.packets_received = 0
        self.last_sequence = None
        self.packets_lost = 0

    def parse_rtp_header(self, packet):
        """Parse RTP header from packet"""
        if len(packet) < 12:
            return None

        # Unpack fixed header
        byte0, byte1, seq, ts, ssrc = struct.unpack('!BBHII', packet[:12])

        # Extract fields
        version = (byte0 >> 6) & 0x03
        padding = (byte0 >> 5) & 0x01
        extension = (byte0 >> 4) & 0x01
        csrc_count = byte0 & 0x0F

        marker = (byte1 >> 7) & 0x01
        payload_type = byte1 & 0x7F

        # Calculate header length
        header_len = 12 + (csrc_count * 4)

        # TODO: Handle extension headers if present

        return {
            'version': version,
            'padding': padding,
            'extension': extension,
            'csrc_count': csrc_count,
            'marker': marker,
            'payload_type': payload_type,
            'sequence': seq,
            'timestamp': ts,
            'ssrc': ssrc,
            'header_length': header_len,
            'payload': packet[header_len:]
        }

    def receive_packet(self):
        """Receive and parse RTP packet"""
        data, addr = self.sock.recvfrom(2048)

        rtp = self.parse_rtp_header(data)
        if not rtp:
            return None

        # Update statistics
        self.packets_received += 1

        # Check for packet loss
        if self.last_sequence is not None:
            expected = (self.last_sequence + 1) & 0xFFFF
            if rtp['sequence'] != expected:
                loss = (rtp['sequence'] - expected) & 0xFFFF
                self.packets_lost += loss
                print(f"WARNING: Detected {loss} packet(s) lost!")

        self.last_sequence = rtp['sequence']

        return rtp

    def close(self):
        self.sock.close()


# Example usage: Receive and display packets
if __name__ == '__main__':
    receiver = RTPReceiver(5004)

    print("Listening for RTP packets on port 5004...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            rtp = receiver.receive_packet()
            if rtp:
                print(f"RTP: seq={rtp['sequence']:5d}, "
                      f"ts={rtp['timestamp']:10d}, "
                      f"PT={rtp['payload_type']:3d}, "
                      f"marker={rtp['marker']}, "
                      f"payload={len(rtp['payload'])} bytes")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print(f"\nStatistics:")
        print(f"  Packets received: {receiver.packets_received}")
        print(f"  Packets lost: {receiver.packets_lost}")
        if receiver.packets_received > 0:
            loss_rate = (receiver.packets_lost /
                        (receiver.packets_received + receiver.packets_lost)) * 100
            print(f"  Loss rate: {loss_rate:.2f}%")

        receiver.close()
```

### Python: Jitter Buffer Implementation

```python
import time
import heapq
from collections import deque

class JitterBuffer:
    """Adaptive jitter buffer for RTP packets"""

    def __init__(self, min_delay_ms=20, max_delay_ms=200, target_delay_ms=50):
        self.min_delay = min_delay_ms / 1000.0
        self.max_delay = max_delay_ms / 1000.0
        self.target_delay = target_delay_ms / 1000.0

        # Buffer storage (priority queue by timestamp)
        self.buffer = []

        # Statistics
        self.last_played_ts = None
        self.arrival_times = {}  # sequence -> arrival time
        self.jitter = 0.0

    def add_packet(self, rtp_packet):
        """Add packet to jitter buffer"""
        arrival_time = time.time()
        seq = rtp_packet['sequence']
        ts = rtp_packet['timestamp']

        # Store arrival time for jitter calculation
        self.arrival_times[seq] = arrival_time

        # Add to buffer (priority queue by timestamp)
        heapq.heappush(self.buffer, (ts, seq, rtp_packet))

        # Update jitter estimate
        self._update_jitter(rtp_packet)

    def _update_jitter(self, rtp_packet):
        """Update jitter estimate (RFC 3550 formula)"""
        if self.last_played_ts is None:
            self.last_played_ts = rtp_packet['timestamp']
            return

        seq = rtp_packet['sequence']
        ts = rtp_packet['timestamp']

        if seq in self.arrival_times and (seq - 1) in self.arrival_times:
            # Calculate interarrival jitter
            arrival_diff = self.arrival_times[seq] - self.arrival_times[seq - 1]
            ts_diff = (ts - self.last_played_ts) / 48000.0  # Assume 48kHz

            D = abs(arrival_diff - ts_diff)
            self.jitter = self.jitter + (D - self.jitter) / 16.0

        self.last_played_ts = ts

    def get_packet(self):
        """Get next packet to play (if ready)"""
        if not self.buffer:
            return None

        # Check if oldest packet is ready to play
        ts, seq, packet = self.buffer[0]

        if seq not in self.arrival_times:
            return None

        arrival_time = self.arrival_times[seq]
        current_time = time.time()
        buffered_time = current_time - arrival_time

        # Adaptive delay based on jitter
        required_delay = max(self.min_delay,
                            min(self.max_delay,
                                self.target_delay + self.jitter * 4))

        if buffered_time >= required_delay:
            # Ready to play
            heapq.heappop(self.buffer)
            del self.arrival_times[seq]
            return packet

        return None

    def get_stats(self):
        """Get buffer statistics"""
        return {
            'buffer_size': len(self.buffer),
            'jitter_ms': self.jitter * 1000,
            'current_delay_ms': self._get_current_delay() * 1000
        }

    def _get_current_delay(self):
        """Get current adaptive delay"""
        return max(self.min_delay,
                  min(self.max_delay,
                      self.target_delay + self.jitter * 4))


# Example usage
if __name__ == '__main__':
    jitter_buffer = JitterBuffer(min_delay_ms=20, max_delay_ms=200)

    # Simulate receiving packets with jitter
    import random

    for i in range(50):
        # Create dummy RTP packet
        packet = {
            'sequence': i,
            'timestamp': i * 960,  # 20ms at 48kHz
            'payload': b'audio_data'
        }

        jitter_buffer.add_packet(packet)

        # Simulate network jitter (0-50ms)
        time.sleep(0.020 + random.uniform(-0.010, 0.030))

        # Try to get packets ready for playout
        while True:
            ready_packet = jitter_buffer.get_packet()
            if ready_packet is None:
                break
            print(f"Playing packet seq={ready_packet['sequence']}")

        stats = jitter_buffer.get_stats()
        print(f"  Buffer: {stats['buffer_size']}, "
              f"Jitter: {stats['jitter_ms']:.1f}ms, "
              f"Delay: {stats['current_delay_ms']:.1f}ms")
```

### JavaScript: RTP in WebRTC (Browser)

```javascript
// WebRTC handles RTP automatically, but you can inspect it

async function startVideoCall() {
    const pc = new RTCPeerConnection({
        iceServers: [{urls: 'stun:stun.l.google.com:19302'}]
    });

    // Get local media
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            echoCancellation: true,
            noiseSuppression: true
        },
        video: {
            width: 1280,
            height: 720
        }
    });

    // Add tracks to peer connection
    stream.getTracks().forEach(track => {
        pc.addTrack(track, stream);
    });

    // Monitor RTP statistics
    setInterval(async () => {
        const stats = await pc.getStats();

        stats.forEach(report => {
            if (report.type === 'outbound-rtp') {
                console.log('Outbound RTP Stats:');
                console.log(`  SSRC: ${report.ssrc}`);
                console.log(`  Packets sent: ${report.packetsSent}`);
                console.log(`  Bytes sent: ${report.bytesSent}`);
                console.log(`  Codec: ${report.codecId}`);
            }

            if (report.type === 'inbound-rtp') {
                console.log('Inbound RTP Stats:');
                console.log(`  SSRC: ${report.ssrc}`);
                console.log(`  Packets received: ${report.packetsReceived}`);
                console.log(`  Packets lost: ${report.packetsLost}`);
                console.log(`  Jitter: ${report.jitter} seconds`);
                console.log(`  Loss rate: ${(report.packetsLost /
                    (report.packetsReceived + report.packetsLost) * 100).toFixed(2)}%`);
            }
        });
    }, 2000);

    // Create offer, exchange SDP, etc.
    // (simplified for brevity)
}

// Get RTP capabilities
const capabilities = RTCRtpReceiver.getCapabilities('video');
console.log('Supported video codecs:');
capabilities.codecs.forEach(codec => {
    console.log(`  ${codec.mimeType} (PT ${codec.clockRate})`);
});
```

### C: Low-Level RTP Packet Parsing

```c
#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>

typedef struct {
    uint8_t version;
    uint8_t padding;
    uint8_t extension;
    uint8_t csrc_count;
    uint8_t marker;
    uint8_t payload_type;
    uint16_t sequence;
    uint32_t timestamp;
    uint32_t ssrc;
} rtp_header_t;

int parse_rtp_header(const uint8_t *packet, size_t len, rtp_header_t *hdr) {
    if (len < 12) {
        return -1;  // Packet too short
    }

    // Byte 0: V(2), P(1), X(1), CC(4)
    uint8_t byte0 = packet[0];
    hdr->version = (byte0 >> 6) & 0x03;
    hdr->padding = (byte0 >> 5) & 0x01;
    hdr->extension = (byte0 >> 4) & 0x01;
    hdr->csrc_count = byte0 & 0x0F;

    // Byte 1: M(1), PT(7)
    uint8_t byte1 = packet[1];
    hdr->marker = (byte1 >> 7) & 0x01;
    hdr->payload_type = byte1 & 0x7F;

    // Sequence number (network byte order)
    hdr->sequence = ntohs(*(uint16_t*)(packet + 2));

    // Timestamp (network byte order)
    hdr->timestamp = ntohl(*(uint32_t*)(packet + 4));

    // SSRC (network byte order)
    hdr->ssrc = ntohl(*(uint32_t*)(packet + 8));

    return 0;
}

void print_rtp_header(const rtp_header_t *hdr) {
    printf("RTP Header:\n");
    printf("  Version: %u\n", hdr->version);
    printf("  Padding: %u\n", hdr->padding);
    printf("  Extension: %u\n", hdr->extension);
    printf("  CSRC count: %u\n", hdr->csrc_count);
    printf("  Marker: %u\n", hdr->marker);
    printf("  Payload type: %u\n", hdr->payload_type);
    printf("  Sequence: %u\n", hdr->sequence);
    printf("  Timestamp: %u\n", hdr->timestamp);
    printf("  SSRC: 0x%08X\n", hdr->ssrc);
}

int main() {
    // Example RTP packet (hex)
    uint8_t packet[] = {
        0x80,                          // V=2, P=0, X=0, CC=0
        0x60,                          // M=0, PT=96 (0x60)
        0x1A, 0x2B,                    // Sequence = 6699
        0x00, 0x00, 0x03, 0xE8,        // Timestamp = 1000
        0xAB, 0xCD, 0xEF, 0x01,        // SSRC = 0xABCDEF01
        // ... payload follows
    };

    rtp_header_t hdr;
    if (parse_rtp_header(packet, sizeof(packet), &hdr) == 0) {
        print_rtp_header(&hdr);
    }

    return 0;
}
```

---

## Jitter Buffer Management

**Jitter** is the variation in packet arrival times. Network jitter causes packets to arrive irregularly, even if sent at constant intervals.

### The Problem

```
Sender sends every 20ms:
  t=0ms:   Packet 1 sent
  t=20ms:  Packet 2 sent
  t=40ms:  Packet 3 sent
  t=60ms:  Packet 4 sent

Receiver arrival times (with jitter):
  t=25ms:  Packet 1 arrives (25ms delay)
  t=48ms:  Packet 2 arrives (28ms delay)
  t=61ms:  Packet 3 arrives (21ms delay)
  t=95ms:  Packet 4 arrives (35ms delay)

Without buffering  choppy audio/video
```

### The Solution: Jitter Buffer

A jitter buffer absorbs timing variations by:
1. **Buffering** incoming packets
2. **Delaying** playout to allow late packets to arrive
3. **Smoothing** output to constant rate

### Fixed Jitter Buffer

Simplest approach: constant delay

```python
class FixedJitterBuffer:
    def __init__(self, delay_ms=50):
        self.delay = delay_ms / 1000.0
        self.buffer = {}

    def add_packet(self, packet):
        arrival_time = time.time()
        playout_time = arrival_time + self.delay
        self.buffer[packet['sequence']] = (playout_time, packet)

    def get_packet_if_ready(self):
        current_time = time.time()

        for seq in sorted(self.buffer.keys()):
            playout_time, packet = self.buffer[seq]
            if current_time >= playout_time:
                del self.buffer[seq]
                return packet

        return None
```

**Pros**: Simple, predictable latency
**Cons**: Wastes delay when network is good, insufficient when network is bad

### Adaptive Jitter Buffer

Adjusts delay based on observed jitter:

```python
class AdaptiveJitterBuffer:
    def __init__(self):
        self.buffer = []
        self.jitter_estimate = 0.020  # Start with 20ms
        self.min_delay = 0.010         # 10ms minimum
        self.max_delay = 0.200         # 200ms maximum

        # Statistics
        self.last_arrival_time = None
        self.last_rtp_timestamp = None

    def add_packet(self, packet):
        arrival_time = time.time()

        # Update jitter estimate
        if self.last_arrival_time and self.last_rtp_timestamp:
            # Calculate interarrival jitter
            arrival_delta = arrival_time - self.last_arrival_time
            timestamp_delta = (packet['timestamp'] - self.last_rtp_timestamp) / 48000.0

            D = abs(arrival_delta - timestamp_delta)
            self.jitter_estimate = self.jitter_estimate + (D - self.jitter_estimate) / 16.0

        self.last_arrival_time = arrival_time
        self.last_rtp_timestamp = packet['timestamp']

        # Calculate playout time (arrival + adaptive delay)
        adaptive_delay = self._calculate_delay()
        playout_time = arrival_time + adaptive_delay

        # Store packet
        heapq.heappush(self.buffer, (playout_time, packet['sequence'], packet))

    def _calculate_delay(self):
        """Calculate adaptive delay based on jitter"""
        # Delay = base + (jitter * safety_factor)
        delay = 0.040 + (self.jitter_estimate * 4.0)

        # Clamp to min/max
        return max(self.min_delay, min(self.max_delay, delay))

    def get_packet(self):
        if not self.buffer:
            return None

        playout_time, seq, packet = self.buffer[0]

        if time.time() >= playout_time:
            heapq.heappop(self.buffer)
            return packet

        return None
```

**Pros**: Optimizes delay for current network conditions
**Cons**: More complex, can oscillate

### Playout Strategies

#### 1. Wait for First Packet

```python
# Simplest: play packets as they become ready
while True:
    packet = jitter_buffer.get_packet()
    if packet:
        play_audio(packet['payload'])
    else:
        time.sleep(0.001)  # Small sleep
```

#### 2. Timed Playout (Better)

```python
# Play at fixed intervals regardless of arrival
playout_interval = 0.020  # 20ms

while True:
    start_time = time.time()

    packet = jitter_buffer.get_packet()
    if packet:
        play_audio(packet['payload'])
    else:
        # Packet loss concealment
        play_silence_or_repeat_last()

    # Sleep until next playout time
    elapsed = time.time() - start_time
    if elapsed < playout_interval:
        time.sleep(playout_interval - elapsed)
```

### Packet Loss Concealment (PLC)

When packet is late or lost:

```python
def conceal_packet_loss(last_packet, codec_type):
    if codec_type == 'opus':
        # Opus has built-in PLC
        return opus_decoder.decode(None, fec=True)

    elif codec_type == 'pcm':
        # Simple: repeat last packet
        return last_packet['payload']

    elif codec_type == 'advanced':
        # Interpolation between last and next packet
        return interpolate(last_packet, next_packet)
```

### Buffer Underrun/Overrun Handling

```python
def monitor_buffer_health(jitter_buffer):
    buffer_size = len(jitter_buffer.buffer)

    if buffer_size == 0:
        # Underrun: buffer empty
        print("WARNING: Buffer underrun - increasing delay")
        jitter_buffer.target_delay += 0.010  # Add 10ms

    elif buffer_size > 20:
        # Overrun: too much buffered
        print("WARNING: Buffer overrun - decreasing delay")
        jitter_buffer.target_delay -= 0.010  # Remove 10ms
```

---

## Packet Loss Handling

RTP doesn't guarantee delivery. Handling packet loss is crucial for quality.

### Loss Detection

#### Via Sequence Numbers

```python
def detect_loss(current_seq, last_seq):
    """Detect packet loss from sequence numbers"""
    if last_seq is None:
        return 0

    expected = (last_seq + 1) & 0xFFFF

    if current_seq == expected:
        return 0  # No loss
    elif current_seq > expected:
        return current_seq - expected
    else:
        # Wraparound case
        return (0x10000 - expected) + current_seq
```

#### Statistics Tracking

```python
class LossStatistics:
    def __init__(self):
        self.packets_received = 0
        self.packets_expected = 0
        self.packets_lost = 0
        self.last_seq = None

    def update(self, seq):
        if self.last_seq is not None:
            expected = (self.last_seq + 1) & 0xFFFF
            gap = (seq - expected) & 0xFFFF

            if gap > 0:
                self.packets_lost += gap
                self.packets_expected += gap + 1
            else:
                self.packets_expected += 1

        self.packets_received += 1
        self.last_seq = seq

    def get_loss_rate(self):
        if self.packets_expected == 0:
            return 0.0
        return self.packets_lost / self.packets_expected
```

### Loss Concealment Techniques

#### 1. Packet Repetition (Simplest)

```python
def packet_repetition(last_good_packet):
    """Repeat last good packet"""
    return last_good_packet.copy()
```

**Pros**: Simple, works for all codecs
**Cons**: Noticeable for long losses, can cause "robotic" sound

#### 2. Silence Insertion

```python
def silence_insertion(packet_size):
    """Insert silence for lost packet"""
    return bytes([0] * packet_size)
```

**Pros**: Simple, no artifacts
**Cons**: Causes gaps in audio

#### 3. Interpolation

```python
def interpolate_audio(prev_packet, next_packet):
    """Linear interpolation between packets"""
    prev_samples = decode(prev_packet)
    next_samples = decode(next_packet)

    interpolated = []
    for i in range(len(prev_samples)):
        value = (prev_samples[i] + next_samples[i]) / 2
        interpolated.append(value)

    return encode(interpolated)
```

**Pros**: Smoother than repetition
**Cons**: Requires looking ahead (adds delay)

#### 4. Codec-Specific PLC

Many modern codecs have built-in PLC:

```python
# Opus example
import opuslib

decoder = opuslib.Decoder(48000, 2)  # 48kHz stereo

# Decode normal packet
audio = decoder.decode(rtp_packet.payload, frame_size=960)

# Packet lost - use PLC
audio = decoder.decode(None, frame_size=960, fec=False)
```

**Opus PLC**: Excellent, nearly transparent for 1-2% loss

### Forward Error Correction (FEC)

Send redundant data to reconstruct lost packets.

#### Simple XOR FEC

```python
def create_fec_packet(packet1, packet2):
    """Create FEC packet from XOR of two packets"""
    fec_payload = bytes([a ^ b for a, b in zip(packet1, packet2)])
    return fec_payload

def recover_lost_packet(good_packet, fec_packet):
    """Recover lost packet using FEC"""
    recovered = bytes([a ^ b for a, b in zip(good_packet, fec_packet)])
    return recovered
```

**Usage:**
```
Send:
  Packet 1 (data)
  Packet 2 (data)
  Packet 3 (FEC = P1 XOR P2)

Receive scenario:
   Packet 1 received
   Packet 2 lost
   Packet 3 (FEC) received
   Recover P2 = P1 XOR FEC
```

**Overhead**: 33% for this scheme (1 FEC per 2 data packets)

#### Opus In-Band FEC

```python
# Encode with FEC
encoder = opuslib.Encoder(48000, 2, opuslib.APPLICATION_VOIP)
encoder.enable_inband_fec()

# Current frame
encoded = encoder.encode(audio_frame, frame_size=960)

# If next packet is lost, decoder can use FEC from current frame
if packet_lost:
    # Decoder extracts FEC from previous packet
    recovered_audio = decoder.decode(previous_packet, frame_size=960, fec=True)
```

### RTP Retransmission (RTX)

RFC 4588 defines retransmission for RTP.

**How it works:**
1. Receiver detects loss (sequence gap)
2. Receiver sends RTCP NACK (Negative Acknowledgment)
3. Sender retransmits lost packet
4. RTX uses separate payload type and SSRC

**RTX Packet Format:**
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         RTP Header                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            OSN (Original Sequence Number)                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                  Original RTP Payload                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**SDP Negotiation:**
```
m=video 5006 RTP/AVP 96 97
a=rtpmap:96 VP8/90000
a=rtpmap:97 rtx/90000
a=fmtp:97 apt=96
```
- PT 96 = VP8 (primary)
- PT 97 = RTX for VP8
- `apt=96` means "associated payload type = 96"

**Trade-off**: Retransmission adds latency (round-trip time). Only useful for applications that can tolerate 50-100ms extra delay.

---

## RTP Extensions

RTP header extensions allow adding metadata without breaking compatibility.

### Extension Mechanism

When X=1 in RTP header, extension follows:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|      0xBEDE   |    length     | Extension data...             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### One-Byte Extension Format (RFC 5285)

```
 0                   1                   2
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  ID   |  len  |     data      |  ID   |  len  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

- **ID** (4 bits): Extension identifier (1-14)
- **len** (4 bits): Length in bytes minus 1
- **data**: Extension payload

### Common RTP Extensions

#### 1. Audio Level (RFC 6464)

Indicates audio level in packet:

```
 0                   1
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  ID   | len=0 |V| level       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

- **V**: Voice activity (1 = speech, 0 = silence)
- **level**: Audio level in -dBov (0-127)

**Usage**: UI indicators, voice activity detection

#### 2. Video Orientation (CVO)

Indicates camera rotation:

```
Extension data: 0 0 0 R R R 0 0
RRR = rotation (0=0°, 1=90°, 2=180°, 3=270°)
```

**Usage**: Correctly rotate video on receiver

#### 3. Transmission Time Offset

Difference between capture and transmission time:

```
Extension data: 24-bit signed offset
```

**Usage**: Improves jitter calculation, synchronization

#### 4. Absolute Send Time

Timestamp when packet was sent (NTP format):

```
Extension data: 24 bits of NTP timestamp
```

**Usage**: More accurate RTT measurements

### Example: Parsing Audio Level Extension

```python
def parse_audio_level_extension(extension_data):
    """Parse audio level extension (RFC 6464)"""
    if len(extension_data) < 1:
        return None

    byte = extension_data[0]
    voice_activity = (byte & 0x80) >> 7
    level_dbov = byte & 0x7F

    # Convert to human-readable
    level_db = -level_dbov  # Negative dBov

    return {
        'voice_activity': bool(voice_activity),
        'level_dbov': level_dbov,
        'level_db': level_db
    }

# Example
ext_data = bytes([0x85])  # V=1, level=5
result = parse_audio_level_extension(ext_data)
# {'voice_activity': True, 'level_dbov': 5, 'level_db': -5}
```

---

## Security: SRTP

**SRTP (Secure RTP)** adds encryption and authentication to RTP. Defined in RFC 3711.

### Why SRTP?

Plain RTP has no security:
- **Eavesdropping**: Anyone can capture and decode packets
- **Tampering**: Packets can be modified in transit
- **Replay**: Old packets can be re-sent
- **Injection**: Fake packets can be inserted

SRTP provides:
- **Confidentiality**: AES encryption
- **Authentication**: HMAC integrity check
- **Replay Protection**: Sequence/timestamp verification

### SRTP Packet Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     RTP Header (unencrypted)                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Encrypted Payload (AES)                       |
|                          ...                                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                Authentication Tag (HMAC)                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Key points:**
- RTP header remains **unencrypted** (needed for routing)
- Payload is **encrypted** with AES
- Authentication tag protects header + encrypted payload
- Typically adds 10-16 bytes overhead (auth tag)

### Encryption

**Algorithm**: AES in Counter Mode (AES-CTR)
- **AES-128**: 128-bit keys (default)
- **AES-256**: 256-bit keys (higher security)

**Why Counter Mode?**
- Stream cipher (can encrypt arbitrary lengths)
- No padding needed
- Parallel encryption/decryption
- Same encryption key for all packets (with unique IV)

### Authentication

**Algorithm**: HMAC-SHA1
- **Tag length**: 80 bits (default) or 32 bits
- Protects against tampering

**What's authenticated:**
- RTP header
- Encrypted payload
- Prevents modification without detection

### Key Derivation

SRTP doesn't use keys directly. Instead:

```
Master Key (128 or 256 bits)
Master Salt (112 bits)
    
Key Derivation Function (KDF)
    
Encryption Key, Auth Key, Salting Key
```

**Separate keys for:**
- RTP encryption
- RTP authentication
- RTCP encryption
- RTCP authentication

### Key Exchange: DTLS-SRTP (WebRTC)

**DTLS-SRTP** is the modern approach (used by WebRTC):

```
1. DTLS Handshake (over UDP)
   - Certificate exchange
   - Verify fingerprints (from SDP)

2. DTLS derives SRTP keys
   - Master key
   - Master salt

3. Switch to SRTP/SRTCP
   - Use derived keys
   - DTLS only for re-keying
```

**SDP Example:**
```
a=fingerprint:sha-256 AA:BB:CC:...
a=setup:actpass
a=ice-ufrag:abc123
a=ice-pwd:xyz789
```

### Alternative: SDES (SDP Security Descriptions)

**Older approach**: Keys in SDP

```
a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:WVNfX19zZ...
```

**Problems:**
- Keys in plaintext SDP (must secure signaling)
- No perfect forward secrecy
- Deprecated in WebRTC (use DTLS-SRTP instead)

### Python SRTP Example (Conceptual)

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hmac
import hashlib

class SRTPEncryptor:
    def __init__(self, master_key, master_salt):
        self.master_key = master_key
        self.master_salt = master_salt

        # Derive keys (simplified)
        self.enc_key = self._derive_key(0x00, 16)
        self.auth_key = self._derive_key(0x01, 20)
        self.salt_key = self._derive_key(0x02, 14)

    def _derive_key(self, label, length):
        """Simplified key derivation"""
        # Real implementation uses proper KDF (RFC 3711)
        data = self.master_key + bytes([label]) + self.master_salt
        return hashlib.sha256(data).digest()[:length]

    def encrypt_rtp(self, rtp_packet):
        """Encrypt RTP packet"""
        # Parse RTP header (first 12 bytes)
        header = rtp_packet[:12]
        payload = rtp_packet[12:]

        # Extract SSRC and sequence for IV
        ssrc = int.from_bytes(header[8:12], 'big')
        seq = int.from_bytes(header[2:4], 'big')

        # Construct IV (SSRC || packet index)
        iv = ssrc.to_bytes(4, 'big') + seq.to_bytes(8, 'big')
        iv = bytes([a ^ b for a, b in zip(iv, self.salt_key)])

        # Encrypt payload with AES-CTR
        cipher = Cipher(
            algorithms.AES(self.enc_key),
            modes.CTR(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted_payload = encryptor.update(payload) + encryptor.finalize()

        # Compute authentication tag
        auth_data = header + encrypted_payload
        tag = hmac.new(self.auth_key, auth_data, hashlib.sha1).digest()[:10]

        # Return SRTP packet
        return header + encrypted_payload + tag

# Usage
master_key = b'sixteen byte key'
master_salt = b'fourteen byte!!'
encryptor = SRTPEncryptor(master_key, master_salt)

# Encrypt RTP packet
srtp_packet = encryptor.encrypt_rtp(rtp_packet)
```

### Best Practices

1. **Always use SRTP** for real-world applications
2. **Use DTLS-SRTP** (not SDES) for key exchange
3. **Verify fingerprints** out-of-band if possible
4. **Re-key periodically** (after ~2^48 packets for AES-128)
5. **Use strong master keys** (cryptographically random)
6. **Protect signaling channel** (HTTPS for SDP exchange)

---

## Integration with Other Protocols

RTP rarely works alone. It integrates with signaling and transport protocols.

### SDP (Session Description Protocol)

**SDP** describes media sessions. Used with SIP, WebRTC, etc.

**Basic Structure:**
```
v=0                                    # Version
o=alice 123456 123456 IN IP4 192.168.1.100  # Origin
s=Audio/Video Call                     # Session name
c=IN IP4 192.168.1.100                 # Connection info
t=0 0                                   # Time (0 0 = permanent)

m=audio 5004 RTP/SAVPF 111 0           # Media description
a=rtpmap:111 opus/48000/2              # Payload mapping
a=fmtp:111 minptime=10;useinbandfec=1  # Format parameters
a=rtpmap:0 PCMU/8000                   # Fallback codec

m=video 5006 RTP/SAVPF 96 97           # Video media
a=rtpmap:96 VP8/90000                  # VP8 codec
a=rtpmap:97 H264/90000                 # H.264 codec
a=fmtp:97 profile-level-id=42e01f      # H.264 profile
```

**Key Fields:**
- `m=`: Media line (type, port, protocol, payload types)
- `a=rtpmap`: Maps PT to codec/clock rate
- `a=fmtp`: Format-specific parameters
- `RTP/SAVPF`: Secure RTP with feedback

### WebRTC

**WebRTC** is the biggest user of RTP today. Architecture:

```
Application (JavaScript)
        
   WebRTC API
        
                 
  Signaling       (SDP offer/answer)
                 $
  ICE             (NAT traversal)
                 $
  DTLS            (Key exchange)
                 $
  SRTP/SRTCP      (Media transport)  RTP here
                 $
  SCTP            (Data channels)
                 
        
      UDP
```

**RTP in WebRTC:**
- Always uses SRTP (encryption mandatory)
- DTLS-SRTP for key exchange
- ICE for NAT traversal
- Multiplexes RTP/RTCP on same port
- Bundle: audio + video on same port

**Example WebRTC Session Establishment:**

```javascript
// Create peer connection
const pc = new RTCPeerConnection({
    iceServers: [{urls: 'stun:stun.l.google.com:19302'}]
});

// Add media tracks
const stream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: true
});
stream.getTracks().forEach(track => pc.addTrack(track, stream));

// Create offer (generates SDP)
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

// Send offer SDP to remote peer via signaling
// (WebSocket, HTTP, etc.)
signalingChannel.send({type: 'offer', sdp: offer.sdp});

// Receive answer from remote
signalingChannel.on('answer', async (answer) => {
    await pc.setRemoteDescription(answer);
    // ICE negotiation, DTLS handshake, then RTP flows!
});
```

**Generated SDP (simplified):**
```
v=0
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=rtpmap:111 opus/48000/2
a=fmtp:111 minptime=10;useinbandfec=1
a=rtcp-mux                              # RTP and RTCP multiplexed
a=setup:actpass
a=fingerprint:sha-256 AA:BB:CC:...      # DTLS cert fingerprint
a=ice-ufrag:xyz
a=ice-pwd:abc123
a=ssrc:123456789 cname:user@host        # RTP SSRC

m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 VP8/90000
a=rtcp-fb:96 nack                       # NACK support
a=rtcp-fb:96 nack pli                   # Picture Loss Indication
a=rtcp-fb:96 goog-remb                  # Bandwidth estimation
```

### SIP (Session Initiation Protocol)

**SIP** is used for VoIP calls. SIP handles signaling, RTP carries media.

**Call Flow:**

```
Alice                    SIP Server                    Bob
  |                          |                          |
  |--- INVITE (SDP offer) -->|                          |
  |                          |--- INVITE (SDP) -------->|
  |                          |<-- 180 Ringing ----------|
  |<-- 180 Ringing ----------|                          |
  |                          |<-- 200 OK (SDP answer) --|
  |<-- 200 OK (SDP) ---------|                          |
  |--- ACK ----------------->|--- ACK ----------------->|
  |                          |                          |
  |<=============== RTP Audio Stream ==================>|
  |                          |                          |
  |--- BYE ----------------->|--- BYE ----------------->|
  |<-- 200 OK ---------------|<-- 200 OK ---------------|
```

**SIP INVITE with SDP:**
```
INVITE sip:bob@example.com SIP/2.0
Via: SIP/2.0/UDP alice-phone.example.com
From: Alice <sip:alice@example.com>
To: Bob <sip:bob@example.com>
Content-Type: application/sdp

v=0
o=alice 123456 123456 IN IP4 192.168.1.100
s=VoIP Call
c=IN IP4 192.168.1.100
t=0 0
m=audio 5004 RTP/AVP 0 8 111
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=rtpmap:111 opus/48000/2
```

After SIP negotiation, RTP flows directly peer-to-peer (or via media server).

### Multicast RTP

RTP supports IP multicast for efficient one-to-many delivery:

```
Sender
  |
  | RTP to 239.1.2.3:5004
  |
    > Receiver 1
    > Receiver 2
    > Receiver 3
    > Receiver N
```

**Challenges:**
- SSRC collision detection (multiple senders)
- Scalable RTCP (report interval increases with receivers)
- Network must support multicast (IGMP)

**RTCP in Multicast:**
- Report interval adapts to group size
- Prevents RTCP implosion
- BW_rtcp = 0.05 * BW_session / num_participants

---

## Common Use Cases

### 1. VoIP Phone Call

**Architecture:**
```
Phone A                               Phone B
  |                                     |
  |-- SIP INVITE (with SDP) ----------->|
  |<- SIP 200 OK (with SDP) ------------|
  |                                     |
  |<======= RTP Audio (G.711) =========>|
  |<======= RTCP Reports ===============|
  |                                     |
  |-- SIP BYE ------------------------->|
```

**Typical Setup:**
- **Codec**: G.711 (PCMU/PCMA) or Opus
- **Packet size**: 20ms audio (160 bytes for G.711)
- **Bandwidth**: ~64 kbps for G.711, ~32 kbps for Opus
- **Latency target**: < 150ms end-to-end
- **Loss tolerance**: Up to 3%

**Code Example:**
```python
# VoIP call parameters
SAMPLE_RATE = 8000  # 8kHz for G.711
PACKET_DURATION = 0.020  # 20ms
SAMPLES_PER_PACKET = int(SAMPLE_RATE * PACKET_DURATION)  # 160

# Send audio packets
def send_voip_audio(sender, audio_stream):
    for audio_chunk in audio_stream:
        # Encode with G.711 (μ-law)
        encoded = g711_ulaw_encode(audio_chunk)

        # Send RTP packet (PT=0 for PCMU)
        sender.send_packet(
            payload=encoded,
            marker=False,
            timestamp_increment=SAMPLES_PER_PACKET
        )

        time.sleep(PACKET_DURATION)
```

### 2. Video Streaming

**Architecture:**
```
Streamer                             Viewer
  Camera                               Display
                                        
  H.264 Encoder                    H.264 Decoder
                                        
  RTP Packetizer                  RTP Depacketizer
                                        
  |=========== RTP/UDP/IP =============|
```

**Typical Setup:**
- **Codec**: H.264 or VP8
- **Resolution**: 720p or 1080p
- **Frame rate**: 30 fps
- **Bitrate**: 2-5 Mbps (adaptive)
- **Latency target**: 200-500ms (buffering)
- **Loss tolerance**: 0.5-2% (FEC helps)

**Challenges:**
- **Large frames**: Need fragmentation (FU-A for H.264)
- **Keyframes**: Must arrive intact (or wait for next)
- **Bitrate adaptation**: Adjust to network conditions

**Example: H.264 Fragmentation**
```python
def fragment_h264_frame(frame_data, mtu=1400):
    """Fragment large H.264 frame into RTP packets"""
    max_payload = mtu - 12  # Account for RTP header

    if len(frame_data) <= max_payload:
        # Small frame - single NAL unit
        return [frame_data]

    # Large frame - use FU-A fragmentation
    fragments = []
    nal_header = frame_data[0]
    nal_payload = frame_data[1:]

    fu_indicator = (nal_header & 0xE0) | 28  # Type = FU-A

    offset = 0
    first = True
    while offset < len(nal_payload):
        chunk_size = min(max_payload - 2, len(nal_payload) - offset)
        chunk = nal_payload[offset:offset + chunk_size]

        # FU header
        fu_header = (nal_header & 0x1F)
        if first:
            fu_header |= 0x80  # Start bit
            first = False
        if offset + chunk_size >= len(nal_payload):
            fu_header |= 0x40  # End bit

        fragment = bytes([fu_indicator, fu_header]) + chunk
        fragments.append(fragment)

        offset += chunk_size

    return fragments
```

### 3. Video Conferencing

**Architecture:**
```
Participant A                 MCU/SFU                 Participant B
     |                          |                          |
     |-- RTP (audio+video) ---->|                          |
     |                          |<-- RTP (audio+video) ----|
     |<-- RTP (mixed) ----------|                          |
     |                          |-- RTP (mixed) ---------->|
```

**Two Approaches:**

#### MCU (Multipoint Control Unit)
- Mixes all streams into one
- Low bandwidth for participants
- Higher server load
- Transcoding required

#### SFU (Selective Forwarding Unit)
- Forwards streams without mixing
- Higher bandwidth for participants
- Lower server load
- No transcoding (just routing)

**Simulcast** (used in modern conferencing):
```
Sender encodes 3 versions:
  - 1080p high quality
  - 720p medium quality
  - 360p low quality

SFU selects appropriate version for each receiver
based on their bandwidth/screen size
```

**Code: Detect Active Speaker (via audio level extension)**
```python
def detect_active_speaker(participants):
    """Detect active speaker based on audio levels"""
    max_level = -127
    active_speaker = None

    for participant in participants:
        # Parse audio level extension from recent packets
        level = participant.get_average_audio_level()

        if level > max_level and level > -40:  # -40dBov threshold
            max_level = level
            active_speaker = participant

    return active_speaker
```

### 4. Screen Sharing

**Characteristics:**
- **High resolution**: 1920x1080 or higher
- **Variable frame rate**: 1-30 fps (based on activity)
- **Content type**: Text, images, video
- **Compression**: Screen content codecs (H.264 Screen Content Coding)

**Optimization:**
```python
def adaptive_screen_sharing(encoder, screen_capturer):
    """Adapt frame rate based on screen activity"""
    last_frame = None
    static_count = 0

    while True:
        frame = screen_capturer.capture()

        # Detect if screen changed
        if frame == last_frame:
            static_count += 1
        else:
            static_count = 0

        # Adaptive frame rate
        if static_count > 5:
            # Screen static - send at low rate (1 fps)
            time.sleep(1.0)
        else:
            # Screen changing - send at high rate (15 fps)
            time.sleep(1.0 / 15)

        # Encode and send
        encoded = encoder.encode(frame)
        send_rtp_video(encoded)

        last_frame = frame
```

### 5. Gaming Voice Chat

**Requirements:**
- **Ultra-low latency**: < 50ms target
- **Small packets**: 10-20ms audio
- **Opus codec**: Best quality/latency trade-off
- **Minimal jitter buffer**: 20-40ms

**Configuration:**
```python
# Gaming VoIP optimized settings
opus_encoder = OpusEncoder(
    sample_rate=48000,
    channels=1,  # Mono sufficient for voice
    application=OPUS_APPLICATION_VOIP,
    bitrate=24000,  # 24 kbps
    frame_duration=10  # 10ms frames for low latency
)

# Minimal jitter buffer
jitter_buffer = JitterBuffer(
    min_delay_ms=20,
    max_delay_ms=60,
    target_delay_ms=30
)
```

---

## Advanced Topics

### Simulcast

**Simulcast**: Sending multiple encodings of same source simultaneously.

**Use case**: Video conferencing where receivers have different bandwidth/screen sizes.

```
Encoder produces 3 streams:
  SSRC 1: 1080p @ 2.5 Mbps  (high)
  SSRC 2: 720p  @ 1.0 Mbps  (medium)
  SSRC 3: 360p  @ 0.3 Mbps  (low)

SFU routes appropriate stream to each receiver:
  Desktop with good connection  high
  Mobile with poor connection   low
```

**SDP Signaling:**
```
m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 VP8/90000
a=ssrc-group:SIM 11111111 22222222 33333333
a=ssrc:11111111 cname:user@host
a=ssrc:22222222 cname:user@host
a=ssrc:33333333 cname:user@host
```

### SVC (Scalable Video Coding)

**SVC**: Single encoded stream with multiple quality layers.

```
Base layer: 360p
Enhancement layer 1: +360p  720p
Enhancement layer 2: +720p  1080p

Receiver can decode:
  - Base only  360p
  - Base + EL1  720p
  - Base + EL1 + EL2  1080p
```

**Advantages over Simulcast:**
- Lower encoding complexity
- Bandwidth efficiency
- Smoother quality adaptation

**Disadvantages:**
- Less codec support
- More complex decoder

### RTP Mixer

**Mixer**: Combines multiple RTP streams into one.

```
Input:
  SSRC A: Audio from participant A
  SSRC B: Audio from participant B
  SSRC C: Audio from participant C

Mixer:
  1. Decode all streams
  2. Mix audio (add samples)
  3. Encode mixed audio
  4. Send as new stream

Output:
  SSRC M: Mixed audio
  CSRC list: [A, B, C]  (who contributed)
```

**Use case**: Audio conferencing with many participants.

### RTP Translator

**Translator**: Forwards RTP packets between networks.

```
Internal Network    Translator    External Network

Functions:
- NAT traversal
- Protocol conversion (RTP  RTP/RTCP mux)
- Transcoding (optional)
```

### Bandwidth Estimation

Modern RTP implementations adapt sending bitrate:

**Approaches:**

1. **RTCP Feedback** (Loss-based):
```python
def adjust_bitrate_on_loss(current_bitrate, loss_rate):
    if loss_rate > 0.05:  # > 5% loss
        return current_bitrate * 0.85  # Reduce 15%
    elif loss_rate < 0.01:  # < 1% loss
        return current_bitrate * 1.05  # Increase 5%
    return current_bitrate
```

2. **REMB (Receiver Estimated Maximum Bitrate)**:
- Receiver measures available bandwidth
- Sends RTCP REMB message
- Sender adjusts bitrate accordingly

3. **Transport-CC (Transport-Wide Congestion Control)**:
- Fine-grained feedback on every packet
- Uses receive timestamps
- ML-based bandwidth estimation

**SDP:**
```
a=rtcp-fb:96 goog-remb
a=rtcp-fb:96 transport-cc
```

### NTP Synchronization

For multi-stream sync (lip-sync):

```python
import ntplib
from time import time

def get_ntp_time():
    """Get current time in NTP format"""
    client = ntplib.NTPClient()
    response = client.request('pool.ntp.org')
    return response.tx_time  # NTP timestamp

def create_rtcp_sr(rtp_timestamp, ntp_time):
    """Create RTCP Sender Report with NTP correlation"""
    # NTP format: seconds since 1900-01-01
    # Split into 32-bit integer and fraction
    ntp_sec = int(ntp_time)
    ntp_frac = int((ntp_time - ntp_sec) * 2**32)

    sr_packet = struct.pack(
        '!HHIIIII',
        0x80C8,           # V=2, PT=SR(200)
        6,                # Length
        ssrc,             # SSRC
        ntp_sec,          # NTP timestamp (MSW)
        ntp_frac,         # NTP timestamp (LSW)
        rtp_timestamp,    # RTP timestamp
        packet_count,     # Sender's packet count
        octet_count       # Sender's octet count
    )
    return sr_packet
```

**Receiver uses NTP correlation:**
```
Audio SR: NTP=12345.500, RTP=48000
Video SR: NTP=12345.500, RTP=90000

Both streams aligned to same NTP time
 Perfect lip-sync
```

---

## Monitoring and Debugging

### Wireshark Analysis

**Capture RTP traffic:**
```bash
# Capture on specific port
tcpdump -i eth0 -w rtp_capture.pcap udp port 5004

# Open in Wireshark
wireshark rtp_capture.pcap
```

**Wireshark RTP Filters:**
```
rtp                          # All RTP packets
rtp.ssrc == 0x12345678       # Specific SSRC
rtp.p_type == 96             # Specific payload type
rtp.marker == 1              # Packets with marker bit
rtp.seq > 1000 && rtp.seq < 1100  # Sequence range
```

**RTP Stream Analysis:**
1. **Telephony**  **RTP**  **RTP Streams**
2. Select stream  **Analyze**

**Metrics shown:**
- Packet count
- Lost packets and percentage
- Maximum delta (jitter)
- Maximum jitter
- Mean jitter
- Clock drift

**Stream Player:**
1. **Telephony**  **RTP**  **RTP Streams**
2. Select audio stream  **Play Streams**
3. Listen to decoded audio

**Packet Details:**
```
Real-Time Transport Protocol
    Version: 2
    Padding: False
    Extension: False
    CSRC count: 0
    Marker: False
    Payload type: Opus (96)
    Sequence number: 1234
    Timestamp: 48000
    Synchronization Source identifier: 0xABCD1234 (2882400052)
    Payload: 40 bytes
```

### Command-Line Tools

#### tcpdump RTP Filtering

```bash
# Capture RTP on even ports (convention)
tcpdump -i eth0 'udp[1] & 1 == 0 && udp[8] & 0xC0 == 0x80'

# Explanation:
# udp[1] & 1 == 0    Even destination port
# udp[8] & 0xC0 == 0x80    RTP version 2
```

#### ffmpeg with RTP

**Send video via RTP:**
```bash
# Stream video file via RTP
ffmpeg -re -i input.mp4 \
  -c:v libvpx -b:v 1M \
  -f rtp rtp://192.168.1.100:5004

# Generate SDP file for receiver
ffmpeg -re -i input.mp4 \
  -c:v libvpx -b:v 1M \
  -f rtp rtp://192.168.1.100:5004 \
  > stream.sdp
```

**Receive video via RTP:**
```bash
# Receive using SDP file
ffplay -protocol_whitelist file,rtp,udp stream.sdp

# Or specify directly
ffplay -protocol_whitelist rtp,udp \
  -i rtp://0.0.0.0:5004
```

#### GStreamer RTP Pipelines

**Send audio:**
```bash
gst-launch-1.0 \
  audiotestsrc ! \
  opusenc ! \
  rtpopuspay ! \
  udpsink host=192.168.1.100 port=5004
```

**Receive audio:**
```bash
gst-launch-1.0 \
  udpsrc port=5004 caps="application/x-rtp" ! \
  rtpopusdepay ! \
  opusdec ! \
  autoaudiosink
```

**Send video:**
```bash
gst-launch-1.0 \
  videotestsrc ! \
  x264enc ! \
  rtph264pay ! \
  udpsink host=192.168.1.100 port=5006
```

### RTP Statistics Monitoring

```python
class RTPStatistics:
    def __init__(self):
        self.packets_received = 0
        self.packets_lost = 0
        self.bytes_received = 0
        self.last_seq = None
        self.highest_seq = 0

        # For jitter calculation
        self.jitter = 0.0
        self.last_arrival = None
        self.last_timestamp = None

    def update(self, rtp_packet):
        seq = rtp_packet['sequence']
        ts = rtp_packet['timestamp']
        arrival_time = time.time()

        # Packet count
        self.packets_received += 1
        self.bytes_received += len(rtp_packet['payload'])

        # Loss detection
        if self.last_seq is not None:
            expected = (self.last_seq + 1) & 0xFFFF
            if seq != expected:
                loss = (seq - expected) & 0xFFFF
                self.packets_lost += loss

        self.last_seq = seq
        self.highest_seq = max(self.highest_seq, seq)

        # Jitter calculation (RFC 3550)
        if self.last_arrival and self.last_timestamp:
            D = abs((arrival_time - self.last_arrival) -
                   ((ts - self.last_timestamp) / 48000.0))
            self.jitter = self.jitter + (D - self.jitter) / 16.0

        self.last_arrival = arrival_time
        self.last_timestamp = ts

    def get_report(self):
        total_expected = self.packets_received + self.packets_lost
        loss_rate = self.packets_lost / total_expected if total_expected > 0 else 0

        return {
            'packets_received': self.packets_received,
            'packets_lost': self.packets_lost,
            'loss_rate': loss_rate * 100,
            'bytes_received': self.bytes_received,
            'jitter_ms': self.jitter * 1000,
            'highest_seq': self.highest_seq
        }

# Usage
stats = RTPStatistics()
for packet in rtp_stream:
    stats.update(packet)

report = stats.get_report()
print(f"Loss: {report['loss_rate']:.2f}%, Jitter: {report['jitter_ms']:.1f}ms")
```

---

## Troubleshooting

### Common Issues

#### 1. No Audio/Video

**Symptoms:**
- Packets not arriving
- Silent audio, blank video

**Debugging:**
```bash
# Check if packets arriving
tcpdump -i eth0 -n udp port 5004

# Check firewall
sudo iptables -L -n -v | grep 5004

# Check listening processes
sudo netstat -ulnp | grep 5004
```

**Common causes:**
- Firewall blocking UDP ports
- Wrong IP address or port
- NAT issues (need STUN/TURN)
- Codec mismatch (sender/receiver disagree)

**Solutions:**
```python
# Test with simple sender/receiver
# Sender:
sender = RTPSender('192.168.1.100', 5004)
sender.send_packet(b'test_data')

# Receiver:
receiver = RTPReceiver(5004)
packet = receiver.receive_packet()
print(f"Received: {packet}")
```

#### 2. One-Way Audio

**Symptoms:**
- Alice hears Bob, but Bob doesn't hear Alice

**Common causes:**
- Asymmetric NAT traversal
- Firewall allows outbound but blocks inbound
- Wrong IP in SDP (private vs public)

**Debug with Wireshark:**
```
# Check if packets flowing both directions
rtp && ip.addr == 192.168.1.100
```

**Solutions:**
- Use STUN to discover public IP
- Use TURN relay if direct path blocked
- Check SDP has correct IP addresses

#### 3. Choppy/Garbled Audio

**Symptoms:**
- Audio cuts in and out
- Robotic/distorted sound

**Common causes:**
- High packet loss (> 5%)
- Excessive jitter
- Buffer underruns
- CPU overload

**Debugging:**
```python
# Monitor packet loss and jitter
stats = RTPStatistics()
while True:
    packet = receive_packet()
    stats.update(packet)

    if stats.packets_received % 100 == 0:
        report = stats.get_report()
        print(f"Loss: {report['loss_rate']:.1f}%, "
              f"Jitter: {report['jitter_ms']:.1f}ms")

        if report['loss_rate'] > 5:
            print("WARNING: High packet loss!")
        if report['jitter_ms'] > 50:
            print("WARNING: High jitter!")
```

**Solutions:**
- Increase jitter buffer size
- Use FEC (Opus in-band FEC)
- Reduce bitrate
- Use packet loss concealment
- Check network quality (QoS)

#### 4. Video Freezing

**Symptoms:**
- Video pauses/freezes
- Last frame stuck on screen

**Common causes:**
- Keyframe loss (I-frame didn't arrive)
- Bandwidth too low
- Packet reordering

**Debugging:**
```python
def detect_keyframe_loss(packets):
    """Detect if we lost a keyframe"""
    last_keyframe_seq = None

    for packet in packets:
        if is_keyframe(packet):
            if last_keyframe_seq is not None:
                gap = packet['sequence'] - last_keyframe_seq
                if gap > 300:  # > 10 seconds at 30fps
                    print(f"WARNING: Long gap between keyframes: {gap} packets")
            last_keyframe_seq = packet['sequence']
```

**Solutions:**
- Request keyframe (via RTCP PLI - Picture Loss Indication)
- Increase keyframe frequency
- Use RTX for keyframe retransmission
- Implement error concealment (freeze-frame vs skip-to-next)

#### 5. Audio/Video Out of Sync

**Symptoms:**
- Lips don't match speech
- Delay between audio and video

**Common causes:**
- Different jitter buffer delays
- Clock drift
- Missing NTP synchronization

**Debugging:**
```python
def check_av_sync(audio_stats, video_stats):
    """Check if A/V streams are synchronized"""
    # Compare playout times based on NTP correlation
    audio_ntp = audio_stats['ntp_time']
    video_ntp = video_stats['ntp_time']

    sync_diff_ms = abs(audio_ntp - video_ntp) * 1000

    if sync_diff_ms > 100:  # > 100ms out of sync
        print(f"WARNING: A/V sync off by {sync_diff_ms:.0f}ms")
        return False
    return True
```

**Solutions:**
- Use RTCP Sender Reports for NTP correlation
- Synchronize jitter buffer depths
- Implement drift compensation
- Use same clock source for both streams

### Diagnostic Commands

```bash
# Check RTP packet headers
tshark -i eth0 -Y rtp -T fields \
  -e rtp.ssrc -e rtp.seq -e rtp.timestamp -e rtp.p_type

# Calculate packet loss
tshark -i eth0 -Y rtp -T fields -e rtp.seq | \
  awk 'NR>1 {diff=$1-prev; if(diff>1) loss+=diff-1} {prev=$1} END {print "Lost:", loss}'

# Monitor jitter
tshark -i eth0 -Y rtcp -T fields -e rtcp.jitter

# Find SSRC collisions
tshark -i eth0 -Y rtp -T fields -e rtp.ssrc | sort | uniq -c
```

---

## Performance Optimization

### Codec Selection

Choose codec based on requirements:

| Requirement | Recommended Codec | Rationale |
|-------------|------------------|-----------|
| Voice quality | Opus @ 16-24 kbps | Best quality/bitrate |
| Low bandwidth | Opus @ 6-12 kbps | Efficient at low rates |
| Low latency | Opus @ 10ms frames | Lowest latency |
| Universal compat | G.711 (PCMU/PCMA) | Works everywhere |
| Music streaming | Opus @ 64-128 kbps | Excellent music quality |
| Video - universal | H.264 | Widest support |
| Video - efficiency | VP9 or AV1 | Better compression |
| Screen sharing | H.264 SCC | Optimized for text |

### Jitter Buffer Tuning

```python
# Latency-critical (gaming, live calls)
JitterBuffer(
    min_delay_ms=10,
    max_delay_ms=50,
    target_delay_ms=20
)

# Quality-critical (music streaming)
JitterBuffer(
    min_delay_ms=50,
    max_delay_ms=300,
    target_delay_ms=150
)

# Balanced (video conferencing)
JitterBuffer(
    min_delay_ms=20,
    max_delay_ms=200,
    target_delay_ms=60
)
```

### Packet Size Optimization

```python
def calculate_optimal_packet_size(codec, network_mtu):
    """Calculate optimal RTP packet size"""
    # Overhead: IP(20) + UDP(8) + RTP(12) = 40 bytes
    overhead = 40

    # Target: < 1200 bytes to avoid fragmentation
    max_payload = min(network_mtu - overhead, 1200)

    if codec == 'opus':
        # Opus: 20ms frames @ 24kbps = ~60 bytes
        # Can fit in single packet
        return 60

    elif codec == 'h264':
        # H.264: Use MTU - overhead
        return max_payload

    elif codec == 'g711':
        # G.711: 20ms @ 64kbps = 160 bytes
        return 160
```

### Bandwidth Management

```python
class BandwidthController:
    def __init__(self, target_bitrate_kbps):
        self.target_bitrate = target_bitrate_kbps * 1000
        self.current_bitrate = target_bitrate_kbps * 1000

    def adapt_to_loss(self, loss_rate):
        """Adapt bitrate based on packet loss"""
        if loss_rate > 0.05:  # > 5%
            self.current_bitrate *= 0.85  # Reduce 15%
        elif loss_rate < 0.01 and self.current_bitrate < self.target_bitrate:
            self.current_bitrate *= 1.05  # Increase 5%

        return int(self.current_bitrate)

    def adapt_to_rtt(self, rtt_ms):
        """Adapt to round-trip time"""
        if rtt_ms > 300:  # High latency
            # Reduce bitrate to lower queuing delay
            self.current_bitrate *= 0.90

        return int(self.current_bitrate)
```

### Network QoS

```bash
# Set DSCP for RTP packets (Linux)
# EF (Expedited Forwarding) for voice
iptables -t mangle -A OUTPUT -p udp --dport 5004 \
  -j DSCP --set-dscp 46

# AF41 for video
iptables -t mangle -A OUTPUT -p udp --dport 5006 \
  -j DSCP --set-dscp 34
```

**DSCP Values:**
- **EF (46)**: Expedited Forwarding - VoIP
- **AF41 (34)**: Assured Forwarding - Interactive video
- **AF31 (26)**: Streaming video
- **BE (0)**: Best effort - Default

### CPU Optimization

```python
# Use hardware encoding when available
def choose_encoder(codec):
    if codec == 'h264':
        # Try hardware encoders first
        encoders = [
            'h264_nvenc',    # NVIDIA
            'h264_qsv',      # Intel Quick Sync
            'h264_videotoolbox',  # Apple
            'libx264'        # Software fallback
        ]
        for enc in encoders:
            if is_available(enc):
                return enc

    return 'libx264'  # Fallback
```

---

## Best Practices

1. **Always use SRTP for security**
   - Encrypt all media in production
   - Use DTLS-SRTP for key exchange
   - Never send keys in plaintext

2. **Implement proper jitter buffer**
   - Use adaptive buffering
   - Monitor and tune delays
   - Handle underruns gracefully

3. **Handle packet loss gracefully**
   - Implement PLC (concealment)
   - Use FEC for important streams
   - Consider RTX for video keyframes

4. **Monitor quality with RTCP**
   - Send regular RTCP reports
   - Track loss, jitter, delay
   - Adapt bitrate based on feedback

5. **Use appropriate codecs**
   - **Audio**: Opus for new implementations
   - **Video**: H.264 for compatibility, VP9 for efficiency
   - Match codec to use case

6. **Set correct timestamp increments**
   - Based on codec clock rate
   - Consistent increments
   - Critical for synchronization

7. **Use even ports for RTP (convention)**
   - RTP on even ports (e.g., 5004)
   - RTCP on odd ports (e.g., 5005)
   - Or use RTP/RTCP multiplexing

8. **Implement proper session cleanup**
   - Send RTCP BYE when leaving
   - Close sockets properly
   - Free resources

9. **Validate incoming packets**
   - Check RTP version
   - Verify SSRC consistency
   - Detect duplicates

10. **Use NTP for cross-stream sync**
    - RTCP SR with NTP correlation
    - Essential for lip-sync
    - Use reliable NTP source

11. **Set appropriate DSCP/TOS**
    - QoS marking for prioritization
    - EF for voice, AF41 for video
    - Coordinate with network team

12. **Test with packet loss simulation**
    - Use `tc` or `netem` on Linux
    - Test 1%, 5%, 10% loss
    - Verify PLC and FEC work

13. **Profile and optimize**
    - Monitor CPU usage
    - Use hardware encoding
    - Optimize packet processing

14. **Log important events**
    - SSRC changes
    - High loss/jitter
    - Codec changes
    - Connection quality

15. **Implement adaptive bitrate**
    - Monitor network conditions
    - Adjust encoding bitrate
    - Smooth transitions

---

## RTP Libraries and Tools

### Python

**aiortc** - Async WebRTC and RTP
```python
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder

pc = RTCPeerConnection()
player = MediaPlayer('/dev/video0', format='v4l2')
pc.addTrack(player.video)
```

**pyRTP** - Basic RTP implementation
```python
import pyrtp
```

### JavaScript

**WebRTC API** - Built-in browser support
```javascript
const pc = new RTCPeerConnection();
const stream = await navigator.mediaDevices.getUserMedia({audio: true, video: true});
stream.getTracks().forEach(track => pc.addTrack(track, stream));
```

### C/C++

**Live555** - Streaming media library
```cpp
#include <liveMedia.hh>
// Full-featured RTSP/RTP server and client
```

**GStreamer** - Multimedia framework
```bash
gst-launch-1.0 videotestsrc ! x264enc ! rtph264pay ! udpsink
```

**FFmpeg** - Multimedia processing
```bash
ffmpeg -i input.mp4 -f rtp rtp://dest:port
```

### Go

**pion/rtp** - Pure Go RTP implementation
```go
import "github.com/pion/rtp"

packet := &rtp.Packet{
    Header: rtp.Header{
        Version: 2,
        PayloadType: 96,
        SequenceNumber: seq,
        Timestamp: ts,
        SSRC: ssrc,
    },
    Payload: payload,
}
```

### Testing Tools

**VLC** - Media player with RTP support
```bash
# Stream to RTP
vlc input.mp4 --sout '#rtp{dst=192.168.1.100,port=5004}'

# Receive RTP
vlc rtp://@:5004
```

**Wireshark** - Packet analysis
- Comprehensive RTP analysis
- Stream statistics
- Audio playback

**tcpdump** - Packet capture
```bash
tcpdump -i eth0 -w capture.pcap udp port 5004
```

**SIPp** - SIP/RTP testing tool
```bash
sipp -sn uac 192.168.1.100
```

---

## ELI10

Imagine you're watching a live sports game on TV.

**RTP is like the TV broadcast:**
- The game happens in real-time at the stadium
- The TV signal carries the video and sound to your home
- If the signal gets a bit fuzzy for a second, that's OK - the game keeps playing
- You'd rather see what's happening NOW, even if a tiny bit is missing, than wait for perfect quality

**How RTP works:**

1. **Packets = Delivery Trucks**
   - The video is split into small chunks (packets)
   - Each truck (packet) has a number on it (#1, #2, #3...)
   - Each truck has a timestamp (when it was recorded)

2. **Sequence Numbers = Package Tracking**
   - If truck #5 is missing, you know immediately
   - You can either wait a bit (maybe it's just late) or skip it

3. **Timestamps = Synchronization**
   - Makes sure the sound matches the video
   - Like making sure the announcer's voice matches the players' movements

4. **Jitter Buffer = DVR with Small Delay**
   - Buffers a few seconds to smooth out delays
   - If trucks arrive at irregular times, buffer evens them out
   - Trade-off: slight delay for smoother playback

5. **RTCP = Quality Reports**
   - Like a report card for the delivery service
   - "10% of trucks were late"  send trucks slower
   - "Everything arrived on time"  can send more trucks

6. **SRTP = Locked Trucks**
   - Regular RTP = open trucks (anyone can see inside)
   - SRTP = locked trucks with keys (encrypted)
   - Like putting the video in a safe box

**Why not just use regular file download?**
- File download waits for EVERYTHING before playing
- RTP starts playing immediately and keeps going
- Better for live events, calls, and real-time stuff

**Real-world examples:**
- **Zoom/Teams calls**: Your voice  RTP  Friend's computer
- **YouTube Live**: Streamer  RTP  YouTube  You
- **Online gaming voice chat**: Your mic  RTP  Other players

---

## Further Resources

### RFCs (Standards)

- **RFC 3550** - RTP: A Transport Protocol for Real-Time Applications
- **RFC 3551** - RTP Profile for Audio and Video Conferences
- **RFC 3711** - Secure Real-time Transport Protocol (SRTP)
- **RFC 4585** - Extended RTP Profile for RTCP-based Feedback
- **RFC 4588** - RTP Retransmission Payload Format
- **RFC 5285** - RTP Header Extensions
- **RFC 5761** - Multiplexing RTP and RTCP
- **RFC 6464** - Audio Level Extension
- **RFC 7742** - WebRTC Video Processing and Codec Requirements

### Books

- **"RTP: Audio and Video for the Internet"** by Colin Perkins
- **"Internet Multimedia Communications Using SIP"** by Rogelio Martinez Perea
- **"WebRTC: APIs and RTCWEB Protocols of the HTML5 Real-Time Web"** by Alan B. Johnston

### Online Resources

- **WebRTC Glossary**: https://webrtcglossary.com/
- **Pion WebRTC** (Go): https://github.com/pion/webrtc
- **aiortc** (Python): https://github.com/aiortc/aiortc
- **Jitsi Meet** (Open-source video conferencing): https://jitsi.org/

### Tutorials

- **MDN WebRTC**: https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API
- **WebRTC samples**: https://webrtc.github.io/samples/
- **GStreamer RTP**: https://gstreamer.freedesktop.org/documentation/rtp/

### Tools

- **Wireshark**: https://www.wireshark.org/
- **VLC**: https://www.videolan.org/
- **FFmpeg**: https://ffmpeg.org/
- **GStreamer**: https://gstreamer.freedesktop.org/

---

**Last Updated**: January 2025
