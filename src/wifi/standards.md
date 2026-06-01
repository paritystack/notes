# Wifi Standards

## Overview

Each 802.11 revision extends the previous with higher throughput, new bands, or new features. [Basics](basics.md) covers the channel and frame fundamentals shared by all revisions; [OFDMA](ofdma.md) is the key scheduler introduced in 802.11ax (Wi-Fi 6) enabling multi-user simultaneous transmission; [roaming](roaming.md) standards (802.11r/k/v/w) overlay any 802.11 PHY version; [security](security.md) tracks the parallel evolution from WEP to WPA3; [QoS management](qos_management.md) (WMM, SCS) applies from 802.11e onward.

## 802.11

### 802.11a
- Released: 1999
- Frequency: 5 GHz
- Maximum Speed: 54 Mbps
- Notes: First standard to use OFDM (Orthogonal Frequency Division Multiplexing).

### 802.11b
- Released: 1999
- Frequency: 2.4 GHz
- Maximum Speed: 11 Mbps
- Notes: Uses DSSS (Direct Sequence Spread Spectrum) modulation.

### 802.11g
- Released: 2003
- Frequency: 2.4 GHz
- Maximum Speed: 54 Mbps
- Notes: Backward compatible with 802.11b, uses OFDM.

### 802.11n
- Released: 2009
- Frequency: 2.4 GHz and 5 GHz
- Maximum Speed: 600 Mbps
- Notes: Introduced MIMO (Multiple Input Multiple Output) technology.

### 802.11ac
- Released: 2013
- Frequency: 5 GHz
- Maximum Speed: 1.3 Gbps
- Notes: Uses wider channels (80 or 160 MHz) and more spatial streams.

### 802.11ax
- Released: 2019
- Frequency: 2.4 GHz and 5 GHz
- Maximum Speed: 9.6 Gbps
- Notes: Also known as Wi-Fi 6, introduces OFDMA (Orthogonal Frequency Division Multiple Access) and improved efficiency in dense environments.

### 802.11be
- Released: 2024
- Frequency: 6 GHz
- Maximum Speed: 48 Gbps
- Notes: Also known as Wi-Fi 7, introduces EHT (Extremely High Throughput) technology.

## Where this connects

- [Basics](basics.md) — channels, frame types, and aggregation that apply across all 802.11 revisions
- [OFDMA](ofdma.md) — the multi-user scheduler introduced in 802.11ax (Wi-Fi 6); essential for dense deployments
- [Security](security.md) — WEP/WPA/WPA2/WPA3 track the parallel security evolution across standard versions
- [Roaming](roaming.md) — 802.11r/k/v/w are optional amendments layered on any 802.11 PHY version
- [QoS management](qos_management.md) — WMM (from 802.11e) and newer SCS/MSCS are QoS overlays