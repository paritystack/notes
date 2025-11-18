# TCP (Transmission Control Protocol)

TCP is a connection-oriented, reliable transport layer protocol that provides ordered delivery of data between applications running on hosts in an IP network. It is one of the core protocols of the Internet Protocol Suite.

## Key Features

- **Connection-Oriented**: Establishes connection before data transfer
- **Reliable**: Guarantees delivery of data in order
- **Error Checking**: Detects corrupted data with checksums
- **Flow Control**: Manages data transmission rate
- **Congestion Control**: Adjusts to network conditions
- **Full-Duplex**: Bidirectional communication

## TCP Packet Format

```
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |          Source Port          |       Destination Port        |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                        Sequence Number                        |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                    Acknowledgment Number                      |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |  Data |       |C|E|U|A|P|R|S|F|                               |
 | Offset| Rsrvd |W|C|R|C|S|S|Y|I|            Window             |
 |       |       |R|E|G|K|H|T|N|N|                               |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |           Checksum            |         Urgent Pointer        |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                    Options                    |    Padding    |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                             Data                              |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Header Fields

- **Source Port** (16 bits): Sending application port number
- **Destination Port** (16 bits): Receiving application port number
- **Sequence Number** (32 bits): Position of first data byte in segment
- **Acknowledgment Number** (32 bits): Next expected sequence number
- **Data Offset** (4 bits): Size of TCP header in 32-bit words
- **Reserved** (3 bits): Reserved for future use
- **Control Flags** (9 bits): Connection control flags
- **Window Size** (16 bits): Receive window size
- **Checksum** (16 bits): Error detection
- **Urgent Pointer** (16 bits): Offset of urgent data
- **Options** (variable): Optional header extensions
- **Padding**: Ensures header is multiple of 32 bits

### Control Flags

- **CWR** (Congestion Window Reduced): ECN-Echo flag received
- **ECE** (ECN-Echo): Congestion experienced
- **URG** (Urgent): Urgent pointer field is valid
- **ACK** (Acknowledgment): Acknowledgment number is valid
- **PSH** (Push): Push buffered data to application
- **RST** (Reset): Reset the connection
- **SYN** (Synchronize): Synchronize sequence numbers (connection setup)
- **FIN** (Finish): No more data from sender (connection termination)

## Three-Way Handshake

TCP uses a three-way handshake to establish a connection:

```
Client                                  Server
  |                                       |
  |  SYN (seq=x)                         |
  |-------------------------------------->|
  |                                       |
  |         SYN-ACK (seq=y, ack=x+1)    |
  |<--------------------------------------|
  |                                       |
  |  ACK (seq=x+1, ack=y+1)             |
  |-------------------------------------->|
  |                                       |
  |     Connection Established            |
  |                                       |
```

1. **SYN**: Client sends SYN packet with initial sequence number
2. **SYN-ACK**: Server responds with SYN-ACK, includes its own sequence number
3. **ACK**: Client sends ACK to confirm, connection established

### Python Example: TCP Client

```python
import socket

# Create TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to server (three-way handshake happens here)
server_address = ('localhost', 8080)
client_socket.connect(server_address)
print(f"Connected to {server_address}")

# Send data
message = "Hello, Server!"
client_socket.sendall(message.encode())

# Receive response
response = client_socket.recv(1024)
print(f"Received: {response.decode()}")

# Close connection
client_socket.close()
```

### Python Example: TCP Server

```python
import socket

# Create TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind to address and port
server_address = ('localhost', 8080)
server_socket.bind(server_address)

# Listen for connections
server_socket.listen(5)
print(f"Server listening on {server_address}")

while True:
    # Accept connection (completes three-way handshake)
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    try:
        # Receive data
        data = client_socket.recv(1024)
        print(f"Received: {data.decode()}")

        # Send response
        response = "Hello, Client!"
        client_socket.sendall(response.encode())
    finally:
        # Close connection
        client_socket.close()
```

## Connection Termination

TCP uses a four-way handshake to close a connection gracefully:

```
Client                                  Server
  |                                       |
  |  FIN (seq=x)                         |
  |-------------------------------------->|
  |                                       |
  |         ACK (ack=x+1)                |
  |<--------------------------------------|
  |                                       |
  |         FIN (seq=y)                  |
  |<--------------------------------------|
  |                                       |
  |  ACK (ack=y+1)                       |
  |-------------------------------------->|
  |                                       |
  |     Connection Closed                 |
```

1. **FIN**: Active closer sends FIN
2. **ACK**: Passive closer acknowledges FIN
3. **FIN**: Passive closer sends its FIN
4. **ACK**: Active closer acknowledges FIN

## TCP State Machine

```
         CLOSED
           |
           | (active open/SYN)
           v
       SYN-SENT
           |
           | (SYN received/SYN-ACK sent)
           v
    SYN-RECEIVED
           |
           | (ACK received)
           v
      ESTABLISHED
           |
           | (close/FIN sent)
           v
       FIN-WAIT-1
           |
           | (ACK received)
           v
       FIN-WAIT-2
           |
           | (FIN received/ACK sent)
           v
       TIME-WAIT
           |
           | (2*MSL timeout)
           v
         CLOSED
```

### TCP States

- **CLOSED**: No connection
- **LISTEN**: Server waiting for connection request
- **SYN-SENT**: Client sent SYN, waiting for SYN-ACK
- **SYN-RECEIVED**: Server received SYN, sent SYN-ACK
- **ESTABLISHED**: Connection established, data transfer
- **FIN-WAIT-1**: Sent FIN, waiting for ACK
- **FIN-WAIT-2**: Received ACK of FIN, waiting for peer FIN
- **CLOSE-WAIT**: Received FIN, waiting for close
- **CLOSING**: Both sides sent FIN simultaneously
- **LAST-ACK**: Waiting for final ACK
- **TIME-WAIT**: Waiting to ensure remote received ACK
- **CLOSED**: Connection fully terminated

### Check Connection States

```bash
# Linux - Show all TCP connections
netstat -tan

# Show listening ports
netstat -tln

# Show established connections
netstat -tan | grep ESTABLISHED

# Alternative: ss command (faster)
ss -tan
ss -tln
ss -tan state established

# Show connection state for specific port
ss -tan '( dport = :80 or sport = :80 )'
```

## TCP Internals

### Sequence Numbers and Acknowledgments

TCP uses sequence numbers to track every byte of data:

```
Sender                                    Receiver
  |                                          |
  |  SEQ=1000, LEN=100 (bytes 1000-1099)    |
  |----------------------------------------->|
  |                                          |
  |         ACK=1100 (expecting byte 1100)  |
  |<-----------------------------------------|
  |                                          |
  |  SEQ=1100, LEN=200 (bytes 1100-1299)    |
  |----------------------------------------->|
  |                                          |
  |         ACK=1300 (expecting byte 1300)  |
  |<-----------------------------------------|
```

**Initial Sequence Number (ISN)**:
- Randomly generated during connection establishment
- Protects against old duplicate segments
- Increments based on time and connection

```python
import socket
import struct

def get_tcp_sequence_info(sock):
    """
    Get TCP sequence number information (Linux)
    """
    # Get socket info
    TCP_INFO = 11  # Linux constant
    tcp_info = sock.getsockopt(socket.IPPROTO_TCP, TCP_INFO, 256)

    # Parse (simplified - actual struct is larger)
    # This is just for demonstration
    return {
        'state': tcp_info[0],
        'retransmits': tcp_info[5]
    }
```

### Sliding Window Mechanism

The sliding window protocol enables efficient data transfer:

```
Sender's View:
[Sent & ACKed][Sent, not ACKed][Ready to send][Cannot send yet]
              ^                               ^
              |<------ Window Size ---------->|
           Last ACK                      Window Edge

Receiver's View:
[Received & ACKed][Can receive][Cannot receive]
                  ^                          ^
                  |<---- Window Size ------->|
              Next expected            Window Edge
```

**Example with actual numbers**:
```
Initial state:
- Window size: 4000 bytes
- Last ACK: 1000
- Can send: bytes 1000-4999

After sending 2000 bytes (1000-2999):
- Waiting for ACK
- Can still send: bytes 3000-4999 (2000 bytes)

Receiver ACKs 3000:
- Window slides forward
- Can now send: bytes 3000-6999
```

```python
import socket

def demonstrate_sliding_window():
    """
    Demonstrate TCP sliding window behavior
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Get current window size
    window = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"Receive window: {window} bytes")

    # Set a specific window size
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)

    # This affects how much data sender can transmit
    # before waiting for acknowledgment
    return sock
```

### Delayed Acknowledgment

TCP delays ACKs to reduce overhead:

```
Without Delayed ACK:
Data1 -> <- ACK1
Data2 -> <- ACK2
Data3 -> <- ACK3

With Delayed ACK:
Data1 ->
Data2 -> <- ACK for both Data1 and Data2
Data3 -> <- ACK for Data3

Benefits:
- Reduces number of ACK packets by ~50%
- Allows ACKs to piggyback on response data
- Typical delay: 40-500ms (usually 200ms)
```

```bash
# Linux - Configure delayed ACK
# Disable delayed ACK (not recommended)
sudo sysctl -w net.ipv4.tcp_delack_seg=1

# Default behavior (ACK every 2nd segment or after timeout)
sudo sysctl -w net.ipv4.tcp_delack_seg=2
```

### Silly Window Syndrome

Problem when sender or receiver creates tiny segments:

```
Bad scenario:
App reads 1 byte -> Window opens 1 byte -> Sender sends 1 byte
Overhead: 40 bytes (IP+TCP headers) for 1 byte of data!

Solutions:
1. Sender-side: Nagle's algorithm
   - Wait to accumulate data before sending

2. Receiver-side: Window updates
   - Only advertise window when significant space available
```

```python
import socket

# Nagle's algorithm (enabled by default)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Keep Nagle enabled for bulk transfers (better efficiency)
# This automatically prevents silly window syndrome

# Disable only for interactive, low-latency apps
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
```

### Path MTU Discovery

TCP discovers maximum transmission unit along path:

```
Process:
1. Start with interface MTU (usually 1500 bytes)
2. Set "Don't Fragment" (DF) bit in IP header
3. If packet too large, router sends ICMP "Fragmentation Needed"
4. Reduce MSS and retry
5. Eventually finds optimal MTU for path

Common MTU values:
- Ethernet: 1500 bytes
- PPPoE: 1492 bytes
- VPN: 1400 bytes (varies)
- Jumbo frames: 9000 bytes

MSS = MTU - IP_header(20) - TCP_header(20)
Typical MSS = 1500 - 40 = 1460 bytes
```

```bash
# Linux - Configure PMTU discovery
sysctl net.ipv4.tcp_mtu_probing

# Values:
# 0 = Disabled
# 1 = Enabled when ICMP blackhole detected
# 2 = Always enabled

# Enable PMTU probing
sudo sysctl -w net.ipv4.tcp_mtu_probing=1

# Set base MSS for probing
sudo sysctl -w net.ipv4.tcp_base_mss=1024
```

```python
import socket

def get_path_mtu(host, port):
    """
    Attempt to determine path MTU
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    # Get effective MSS
    mss = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG)

    # MTU = MSS + TCP header (20) + IP header (20)
    estimated_mtu = mss + 40

    print(f"MSS: {mss}, Estimated MTU: {estimated_mtu}")
    sock.close()

    return estimated_mtu
```

## TCP Timers

TCP uses several timers to manage connections:

### Retransmission Timer (RTO)

Retransmits unacknowledged segments:

```
How RTO is calculated:
1. Measure RTT (Round Trip Time) for each segment
2. Calculate smoothed RTT (SRTT):
   SRTT = (1 - α) * SRTT + α * RTT
   where α = 0.125

3. Calculate RTT variation (RTTVAR):
   RTTVAR = (1 - β) * RTTVAR + β * |SRTT - RTT|
   where β = 0.25

4. Calculate RTO:
   RTO = SRTT + 4 * RTTVAR

Minimum RTO: 200ms (Linux default)
Maximum RTO: 120s
```

```bash
# View TCP timer statistics
netstat -s | grep timeout
ss -ti  # Show timer information

# Configure RTO parameters
sysctl net.ipv4.tcp_retries1  # 3 (early retransmit threshold)
sysctl net.ipv4.tcp_retries2  # 15 (max retries before reset)

# Set minimum RTO
sudo sysctl -w net.ipv4.tcp_rto_min=200  # milliseconds
```

### Persistence Timer

Handles zero window situations:

```
Scenario:
Receiver -> [Window=0] -> Sender (stops sending)

Problem: What if window update is lost?

Solution: Persistence timer
- Sender periodically sends 1-byte probe
- Forces receiver to send window update
- Prevents deadlock

Timer values: 5s, 10s, 20s, 40s... (exponential backoff)
Maximum: 60 seconds
```

```python
import socket
import time

def handle_zero_window():
    """
    Demonstrate persistence timer behavior
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 9000))
    server.listen(1)

    conn, addr = server.accept()

    # Set very small receive buffer (simulates zero window)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512)

    print("Receiver has small buffer, sender will hit zero window")
    print("Persistence timer will trigger periodic probes")

    # Don't read data immediately - let buffer fill
    time.sleep(10)

    # Now read data
    data = conn.recv(4096)
    print(f"Finally read {len(data)} bytes")

    conn.close()
    server.close()
```

### Keepalive Timer

Detects dead connections:

```
Purpose:
- Detect if peer has crashed
- Detect if connection is still alive
- Clean up half-open connections

How it works:
1. After idle period (tcp_keepalive_time), send probe
2. If no response, retry after interval (tcp_keepalive_intvl)
3. After max probes (tcp_keepalive_probes), close connection

Default Linux values:
- tcp_keepalive_time: 7200s (2 hours)
- tcp_keepalive_intvl: 75s
- tcp_keepalive_probes: 9
- Total time before reset: 2h + 75s * 9 ≈ 2h 11min
```

```bash
# Configure keepalive globally (Linux)
sudo sysctl -w net.ipv4.tcp_keepalive_time=600     # 10 minutes
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=60     # 60 seconds
sudo sysctl -w net.ipv4.tcp_keepalive_probes=5     # 5 probes

# View current settings
sysctl -a | grep tcp_keepalive
```

```python
import socket

def configure_keepalive(sock, idle=60, interval=10, count=3):
    """
    Configure TCP keepalive per-socket

    Args:
        idle: Seconds before first probe
        interval: Seconds between probes
        count: Number of failed probes before giving up
    """
    # Enable keepalive
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    # Platform-specific configuration
    try:
        # Linux
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, count)
        print(f"Keepalive: idle={idle}s, interval={interval}s, count={count}")
    except AttributeError:
        # macOS/BSD uses different constants
        TCP_KEEPALIVE = 0x10
        sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPALIVE, idle)
        print(f"Keepalive configured for macOS: idle={idle}s")

    return sock

# Example usage
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock = configure_keepalive(sock, idle=60, interval=10, count=3)
```

### TIME_WAIT Timer

Ensures clean connection termination:

```
Why TIME_WAIT exists:
1. Allow delayed packets to expire
2. Ensure remote received final ACK
3. Prevent old segments from new connection

Duration: 2 * MSL (Maximum Segment Lifetime)
- Linux default: 60 seconds (2 * 30s)
- Cannot be changed per-connection
- Ties up local port

TIME_WAIT state:
Client              Server
  |  FIN ->           |
  |      <- ACK       |
  |      <- FIN       |
  |  ACK ->           |
  |                   |
[TIME_WAIT]      [CLOSED]
(60 seconds)
  |
[CLOSED]
```

```bash
# View TIME_WAIT connections
netstat -tan | grep TIME_WAIT | wc -l
ss -tan state time-wait | wc -l

# Configure TIME_WAIT
sudo sysctl -w net.ipv4.tcp_fin_timeout=30  # Reduce from 60s

# Reuse TIME_WAIT sockets (safe for clients)
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# DO NOT use tcp_tw_recycle (removed in newer kernels)
# It causes problems with NAT

# Use SO_REUSEADDR to bind to TIME_WAIT port
```

```python
import socket

def reuse_address_example():
    """
    Demonstrates SO_REUSEADDR to handle TIME_WAIT
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Allow reuse of address in TIME_WAIT state
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Now can bind immediately after previous connection closes
    sock.bind(('localhost', 8080))
    sock.listen(5)

    print("Server can restart immediately despite TIME_WAIT")
    return sock
```

## Flow Control

TCP uses a sliding window protocol for flow control:

```python
import socket
import time

def tcp_receiver_with_flow_control():
    """
    Receiver controls flow using window size
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 8080))
    server.listen(1)

    conn, addr = server.accept()

    # Set receive buffer size (affects window size)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)

    total_received = 0
    while True:
        data = conn.recv(1024)
        if not data:
            break

        total_received += len(data)
        print(f"Received {len(data)} bytes, total: {total_received}")

        # Simulate slow processing
        time.sleep(0.1)

    conn.close()
    server.close()

def tcp_sender():
    """
    Sender adapts to receiver's window size
    """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 8080))

    # Set send buffer size
    client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8192)

    # Send large amount of data
    data = b'X' * 100000
    sent = 0

    while sent < len(data):
        chunk = data[sent:sent+1024]
        try:
            bytes_sent = client.send(chunk)
            sent += bytes_sent
            print(f"Sent {bytes_sent} bytes, total: {sent}")
        except socket.error as e:
            print(f"Send error: {e}")
            break

    client.close()
```

## Congestion Control

TCP implements congestion control algorithms:

### Algorithms

1. **Slow Start**: Exponentially increase congestion window
2. **Congestion Avoidance**: Linearly increase window
3. **Fast Retransmit**: Retransmit on 3 duplicate ACKs
4. **Fast Recovery**: Reduce window, avoid slow start

```
Window Size
    ^
    |     Slow Start  | Congestion Avoidance
    |                /|
    |              /  |
    |            /    |_______________
    |          /      |               \
    |        /        |                \
    |      /          |                 \ Fast Recovery
    |    /            |                  \_______________
    |  /              |
    |/________________|________________________> Time
         Threshold
```

### Check TCP Congestion Control

```bash
# Linux - Check current algorithm
sysctl net.ipv4.tcp_congestion_control

# Available algorithms
sysctl net.ipv4.tcp_available_congestion_control

# Set congestion control algorithm
sudo sysctl -w net.ipv4.tcp_congestion_control=cubic

# Common algorithms:
# - cubic (default on most Linux)
# - reno (traditional)
# - bbr (Google's BBR)
# - vegas
```

## Retransmission

TCP retransmits lost or corrupted packets:

```python
import socket
import time

def tcp_with_timeout():
    """
    TCP automatically handles retransmission
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Set timeout for operations
    sock.settimeout(5.0)

    try:
        sock.connect(('example.com', 80))

        # Send HTTP request
        request = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
        sock.sendall(request)

        # Receive response
        response = sock.recv(4096)
        print(f"Received {len(response)} bytes")

    except socket.timeout:
        print("Operation timed out - TCP retransmission may be occurring")
    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        sock.close()
```

### Retransmission Timeout (RTO)

```bash
# Linux - View TCP retransmission statistics
netstat -s | grep -i retrans

# Check retransmission timer settings
sysctl net.ipv4.tcp_retries1  # Threshold for alerting
sysctl net.ipv4.tcp_retries2  # Maximum retries before giving up

# Typical values:
# tcp_retries1 = 3  (alert after 3-6 seconds)
# tcp_retries2 = 15 (give up after ~13-30 minutes)
```

## TCP Options

Common TCP options in the header:

### Maximum Segment Size (MSS)

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Set TCP_MAXSEG option (MSS)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1400)

# Get current MSS
mss = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG)
print(f"TCP MSS: {mss}")
```

### Window Scaling

```bash
# Enable TCP window scaling (Linux)
sudo sysctl -w net.ipv4.tcp_window_scaling=1

# Check current setting
sysctl net.ipv4.tcp_window_scaling
```

### Selective Acknowledgment (SACK)

```bash
# Enable SACK (Linux)
sudo sysctl -w net.ipv4.tcp_sack=1

# Check current setting
sysctl net.ipv4.tcp_sack
```

### Timestamps

```bash
# Enable TCP timestamps
sudo sysctl -w net.ipv4.tcp_timestamps=1

# Check current setting
sysctl net.ipv4.tcp_timestamps
```

## TCP vs UDP

| Feature | TCP | UDP |
|---------|-----|-----|
| Connection | Connection-oriented | Connectionless |
| Reliability | Guaranteed delivery | No guarantee |
| Ordering | In-order delivery | No ordering |
| Speed | Slower (overhead) | Faster (minimal overhead) |
| Header Size | 20-60 bytes | 8 bytes |
| Error Checking | Yes (checksum) | Yes (checksum) |
| Flow Control | Yes | No |
| Congestion Control | Yes | No |
| Use Cases | HTTP, FTP, SSH, Email | DNS, VoIP, Streaming, Gaming |

### When to Use TCP

- File transfers
- Email
- Web browsing
- Remote shell (SSH)
- Any application requiring reliability

### When to Use UDP

- Real-time applications (VoIP, video streaming)
- DNS queries
- Online gaming
- IoT devices with small data
- Broadcasting/multicasting

## Performance Tuning

### Socket Buffer Sizes

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Increase buffer sizes for high-throughput applications
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB receive
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB send

# Get buffer sizes
rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
print(f"Receive buffer: {rcvbuf}, Send buffer: {sndbuf}")
```

### TCP Keepalive

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Enable keepalive
sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

# Set keepalive parameters (Linux)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)    # Start after 60s
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)   # Interval 10s
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)      # Retry 3 times
```

### Nagle's Algorithm

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Disable Nagle's algorithm for low-latency applications
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# Check status
nodelay = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
print(f"TCP_NODELAY: {nodelay}")
```

### Linux Kernel Tuning

```bash
# Increase maximum buffer sizes
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216

# Set TCP buffer sizes (min, default, max)
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"

# Increase backlog queue
sudo sysctl -w net.core.somaxconn=1024
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=2048

# Enable TCP Fast Open
sudo sysctl -w net.ipv4.tcp_fastopen=3

# Reuse TIME_WAIT sockets
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
```

## Troubleshooting

### Analyze TCP with tcpdump

```bash
# Capture TCP traffic on port 80
sudo tcpdump -i any tcp port 80 -n

# Capture SYN packets
sudo tcpdump 'tcp[tcpflags] & (tcp-syn) != 0' -n

# Capture RST packets
sudo tcpdump 'tcp[tcpflags] & (tcp-rst) != 0' -n

# Save to file for analysis
sudo tcpdump -i any tcp port 80 -w capture.pcap

# Read from file
tcpdump -r capture.pcap -n
```

### Analyze with Wireshark

```bash
# Start Wireshark
wireshark

# Useful display filters:
# tcp.port == 80
# tcp.flags.syn == 1
# tcp.flags.reset == 1
# tcp.analysis.retransmission
# tcp.analysis.duplicate_ack
# tcp.window_size_value < 1000
```

### Common Issues

**Connection Refused**
```bash
# Check if port is listening
netstat -tln | grep :80

# Check firewall
sudo iptables -L -n | grep 80
```

**Connection Timeout**
```bash
# Test connectivity
telnet example.com 80

# Check routing
traceroute example.com

# Test with timeout
timeout 5 telnet example.com 80
```

**Slow Connection**
```python
import socket
import time

def measure_tcp_performance():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Measure connection time
    start = time.time()
    sock.connect(('example.com', 80))
    connect_time = time.time() - start
    print(f"Connection time: {connect_time:.3f}s")

    # Send request
    request = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
    start = time.time()
    sock.sendall(request)
    send_time = time.time() - start
    print(f"Send time: {send_time:.3f}s")

    # Receive response
    start = time.time()
    data = sock.recv(4096)
    recv_time = time.time() - start
    print(f"Receive time: {recv_time:.3f}s")
    print(f"Received {len(data)} bytes")

    sock.close()

measure_tcp_performance()
```

### Monitoring TCP Connections

```bash
# Real-time connection monitoring
watch -n 1 'netstat -tan | grep ESTABLISHED | wc -l'

# Connection state distribution
netstat -tan | awk '{print $6}' | sort | uniq -c

# Show connections with process info
sudo netstat -tanp

# Alternative with ss
ss -tanp state established
```

## Advanced Topics

### TCP Fast Open (TFO)

Reduces latency by sending data in SYN packet:

```python
import socket

# Client with TFO
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Enable TFO (requires kernel support)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_FASTOPEN, 1)

# Send data during connection (SYN packet)
data = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
sock.sendto(data, socket.MSG_FASTOPEN, ('example.com', 80))
```

### TCP Multipath (MPTCP)

Allows connection over multiple paths:

```bash
# Check if MPTCP is available (Linux)
sysctl net.mptcp.enabled

# Enable MPTCP
sudo sysctl -w net.mptcp.enabled=1
```

### Zero Copy

Improve performance with zero-copy operations:

```python
import socket
import os

def sendfile_example(sock, filename):
    """
    Send file using zero-copy sendfile
    """
    with open(filename, 'rb') as f:
        # Get file size
        file_size = os.fstat(f.fileno()).st_size

        # Send file using sendfile (zero-copy)
        offset = 0
        while offset < file_size:
            sent = os.sendfile(sock.fileno(), f.fileno(), offset, file_size - offset)
            offset += sent
```

## Security Considerations

### SYN Flood Attack

Exploits three-way handshake to exhaust server resources:

```
Attack scenario:
Attacker sends many SYN packets with spoofed source IPs
Server allocates resources for each SYN-RECEIVED connection
Server's SYN queue fills up
Legitimate clients cannot connect

Defense mechanisms:
1. SYN Cookies
2. Increase SYN queue size
3. Reduce SYN-RECEIVED timeout
4. Firewall rate limiting
```

```bash
# Linux - Enable SYN cookies (recommended)
sudo sysctl -w net.ipv4.tcp_syncookies=1

# Increase SYN backlog
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=4096

# Reduce SYN-ACK retries
sudo sysctl -w net.ipv4.tcp_synack_retries=2

# View SYN attack statistics
netstat -s | grep -i syn
```

```python
import socket

def syn_flood_resistant_server():
    """
    Server configuration to resist SYN floods
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Reuse address
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Increase backlog (doesn't help much with SYN flood)
    # SYN cookies provide better protection
    server.bind(('0.0.0.0', 8080))
    server.listen(1024)  # Large backlog

    # Set accept timeout to prevent blocking
    server.settimeout(5.0)

    return server
```

### TCP Connection Hijacking

Attacker injects packets into existing connection:

```
Attack requirements:
1. Know source/destination IP addresses
2. Know source/destination port numbers
3. Predict sequence numbers (hardest part)

Prevention:
1. Use encrypted protocols (TLS/SSL)
2. Use random sequence numbers (ISN randomization)
3. Use authentication (IPsec, VPN)
4. Network isolation
```

```bash
# Linux - Ensure strong ISN randomization
sysctl net.ipv4.tcp_timestamps  # Should be 1 (helps with ISN)

# Use encrypted protocols
# HTTP -> HTTPS
# Telnet -> SSH
# FTP -> SFTP
```

### TCP Reset Attack

Attacker sends RST packet to terminate connection:

```
How it works:
1. Sniff packets to get connection details
2. Forge RST packet with correct sequence number
3. Send to either endpoint
4. Connection immediately terminated

Defense:
1. Use encrypted tunnels (VPN)
2. Network segmentation
3. Detect anomalous RST patterns
```

```python
import socket

def detect_unexpected_reset():
    """
    Detect unexpected connection resets
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('example.com', 80))

        # Send request
        sock.sendall(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")

        # Receive response
        response = sock.recv(4096)

    except ConnectionResetError:
        print("WARNING: Connection reset unexpectedly")
        print("Possible reset attack or network issue")
        # Log for security analysis

    except Exception as e:
        print(f"Error: {e}")

    finally:
        sock.close()
```

### Port Scanning

Attackers scan for open TCP ports:

```
Scan types:
1. SYN scan (stealth): Send SYN, check for SYN-ACK
2. Connect scan: Full three-way handshake
3. FIN scan: Send FIN to closed ports
4. XMAS scan: Set FIN, PSH, URG flags

Detection:
- Multiple connection attempts to different ports
- Incomplete handshakes
- Unusual flag combinations
```

```bash
# Linux - Detect port scans with iptables
sudo iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP  # Null scan
sudo iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP   # XMAS scan

# Log SYN packets (potential scanning)
sudo iptables -A INPUT -p tcp --syn -j LOG --log-prefix "SYN packet: "

# Rate limit new connections
sudo iptables -A INPUT -p tcp --syn -m limit --limit 1/s -j ACCEPT
```

### Slowloris Attack

Keeps many connections open with slow requests:

```
Attack method:
1. Open many TCP connections to server
2. Send partial HTTP requests very slowly
3. Server keeps connections open waiting for complete request
4. Exhaust server's connection pool

Defense:
1. Connection timeout limits
2. Request timeout limits
3. Limit connections per IP
4. Use reverse proxy with timeout handling
```

```python
import socket

def slowloris_resistant_server():
    """
    Server with defenses against slowloris
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 8080))
    server.listen(100)

    while True:
        client, addr = server.accept()

        # Set aggressive timeouts
        client.settimeout(10.0)  # 10 second timeout

        try:
            # Set deadline for receiving complete request
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = client.recv(1024)
                if not chunk:
                    break
                data += chunk

                # Limit request size
                if len(data) > 16384:  # 16KB max
                    client.close()
                    break

        except socket.timeout:
            print(f"Slow client from {addr} timed out")
            client.close()
```

### Man-in-the-Middle (MITM)

Attacker intercepts TCP traffic:

```
Attack scenario:
1. Attacker positions between client and server
2. Intercepts all TCP packets
3. Can read, modify, or drop packets
4. Appears transparent to both endpoints

Prevention:
1. Use TLS/SSL encryption
2. Certificate pinning
3. Mutual authentication
4. VPN or IPsec
```

```python
import socket
import ssl

def secure_tcp_connection(hostname, port):
    """
    Establish secure TLS connection to prevent MITM
    """
    # Create regular socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Wrap with TLS
    context = ssl.create_default_context()

    # Enable hostname checking
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED

    # Create secure connection
    secure_sock = context.wrap_socket(sock, server_hostname=hostname)
    secure_sock.connect((hostname, port))

    print(f"Secure connection established")
    print(f"Cipher: {secure_sock.cipher()}")
    print(f"Protocol: {secure_sock.version()}")

    return secure_sock

# Example usage
try:
    sock = secure_tcp_connection('example.com', 443)
    sock.sendall(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    response = sock.recv(4096)
    sock.close()
except ssl.SSLError as e:
    print(f"SSL Error: {e}")
    print("Possible MITM attack or certificate issue")
```

### TCP Sequence Prediction

Older attack exploiting predictable sequence numbers:

```
Attack (mostly historical):
1. Observe sequence number patterns
2. Predict next sequence number
3. Inject forged packets

Modern defenses:
- Random ISN generation (RFC 6528)
- Timestamp option for better randomization
- TCP MD5 signature option (BGP)
```

```bash
# Verify strong sequence number generation
sysctl net.ipv4.tcp_timestamps  # Should be 1

# For BGP and other critical protocols, use TCP MD5
# Configured per-connection with SO_TCP_MD5SIG
```

### Security Best Practices

```python
import socket
import ssl

def create_secure_tcp_client():
    """
    Example of security-hardened TCP client
    """
    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Set timeouts to prevent hanging
    sock.settimeout(30.0)

    # For TLS connections
    context = ssl.create_default_context()

    # Enforce TLS 1.2+
    context.minimum_version = ssl.TLSVersion.TLSv1_2

    # Disable compression (CRIME attack prevention)
    context.options |= ssl.OP_NO_COMPRESSION

    # Enable hostname verification
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED

    return sock, context

def create_secure_tcp_server():
    """
    Example of security-hardened TCP server
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Allow reuse
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind to specific interface (not 0.0.0.0 if possible)
    server.bind(('127.0.0.1', 8080))

    # Reasonable backlog
    server.listen(128)

    # Set timeout
    server.settimeout(5.0)

    return server
```

## TCP in Different Environments

### TCP over WAN (Long Fat Networks)

Wide Area Networks have high latency and bandwidth:

```
Challenges:
- High bandwidth × delay product (BDP)
- Large buffer requirements
- Window scaling essential
- Packet loss has severe impact

BDP Example:
- Bandwidth: 1 Gbps
- RTT: 100ms
- BDP = 1 Gbps × 0.1s = 100 Mb = 12.5 MB
- Need 12.5 MB window to fully utilize bandwidth!

Solutions:
1. Enable window scaling
2. Increase buffer sizes
3. Use CUBIC or BBR congestion control
4. Enable SACK
5. Consider TCP Fast Open
```

```bash
# Optimize TCP for WAN
sudo sysctl -w net.ipv4.tcp_window_scaling=1
sudo sysctl -w net.ipv4.tcp_sack=1
sudo sysctl -w net.core.rmem_max=134217728   # 128MB
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 67108864"  # 64MB
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 67108864"

# Use BBR congestion control (better for long-distance)
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
```

```python
import socket

def configure_for_wan():
    """
    Configure TCP socket for WAN transfer
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Large buffers for high BDP
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)

    # Disable Nagle for better latency
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Enable keepalive for long-lived connections
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    return sock
```

### TCP in Data Centers

Data center networks have different characteristics:

```
Characteristics:
- Very low latency (< 1ms)
- High bandwidth (10/25/100 Gbps)
- Low packet loss
- Many concurrent connections
- Incast problem

Incast problem:
Many senders -> Single receiver simultaneously
Causes buffer overflow and packet loss
Severely reduces throughput

Solutions:
1. Use DCTCP (Data Center TCP)
2. Reduce RTO minimum
3. ECN (Explicit Congestion Notification)
4. Priority queuing
```

```bash
# Optimize for data center
sudo sysctl -w net.ipv4.tcp_congestion_control=dctcp

# Enable ECN
sudo sysctl -w net.ipv4.tcp_ecn=1

# Reduce RTO min for fast retransmission
sudo sysctl -w net.ipv4.tcp_rto_min=10  # 10ms

# Increase connection tracking
sudo sysctl -w net.core.somaxconn=4096
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=8192
```

### TCP over Wireless

Wireless networks have unique challenges:

```
Challenges:
- Variable latency
- Higher packet loss (not always congestion)
- Handoff between access points
- Limited bandwidth
- Battery concerns

Problem:
TCP interprets wireless packet loss as congestion
Reduces congestion window unnecessarily
Performance suffers

Solutions:
1. Use loss differentiation
2. Link-layer retransmission
3. TCP Westwood (wireless-aware)
4. Explicit Loss Notification
```

```bash
# Optimize for wireless (Linux)
# Use Westwood or CUBIC
sudo sysctl -w net.ipv4.tcp_congestion_control=westwood

# More aggressive retransmission
sudo sysctl -w net.ipv4.tcp_retries1=2
sudo sysctl -w net.ipv4.tcp_retries2=8

# Enable timestamps for better RTT estimation
sudo sysctl -w net.ipv4.tcp_timestamps=1
```

### TCP with Satellite Links

Satellite has extreme latency:

```
Characteristics:
- Very high latency (500-700ms RTT)
- High bandwidth
- Occasional errors
- Asymmetric links

Issues:
- Huge BDP (bandwidth × delay)
- ACK packets delayed
- Window size limitations
- Timeout issues

Solutions:
1. Large TCP windows
2. SACK essential
3. ACK reduction
4. Header compression
5. Consider PEP (Performance Enhancing Proxies)
```

```bash
# Optimize for satellite
sudo sysctl -w net.ipv4.tcp_window_scaling=1  # Essential
sudo sysctl -w net.ipv4.tcp_sack=1            # Essential

# Very large buffers
sudo sysctl -w net.core.rmem_max=268435456    # 256MB
sudo sysctl -w net.core.wmem_max=268435456
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# Increase RTO max for high latency
sudo sysctl -w net.ipv4.tcp_retries2=8
```

## Advanced Programming Examples

### TCP Client in Different Languages

**Go Example:**
```go
package main

import (
    "fmt"
    "net"
    "time"
)

func main() {
    // Configure TCP dialer
    dialer := &net.Dialer{
        Timeout:   30 * time.Second,
        KeepAlive: 30 * time.Second,
    }

    // Connect to server
    conn, err := dialer.Dial("tcp", "example.com:80")
    if err != nil {
        fmt.Printf("Connection failed: %v\n", err)
        return
    }
    defer conn.close()

    // Set deadlines
    conn.SetDeadline(time.Now().Add(10 * time.Second))

    // Send data
    message := []byte("GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    _, err = conn.Write(message)
    if err != nil {
        fmt.Printf("Write failed: %v\n", err)
        return
    }

    // Receive response
    buffer := make([]byte, 4096)
    n, err := conn.Read(buffer)
    if err != nil {
        fmt.Printf("Read failed: %v\n", err)
        return
    }

    fmt.Printf("Received %d bytes\n", n)
}
```

**Node.js Example:**
```javascript
const net = require('net');

// TCP Client
const client = net.createConnection({ port: 80, host: 'example.com' }, () => {
    console.log('Connected to server');

    // Send data
    client.write('GET / HTTP/1.1\r\nHost: example.com\r\n\r\n');
});

// Handle data
client.on('data', (data) => {
    console.log(`Received: ${data.length} bytes`);
    client.end();
});

// Handle errors
client.on('error', (err) => {
    console.error(`Connection error: ${err.message}`);
});

// Handle close
client.on('end', () => {
    console.log('Disconnected from server');
});

// Set timeout
client.setTimeout(5000, () => {
    console.log('Connection timeout');
    client.destroy();
});
```

**Rust Example:**
```rust
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

fn main() -> std::io::Result<()> {
    // Connect to server
    let mut stream = TcpStream::connect("example.com:80")?;

    // Set timeouts
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    // Enable TCP_NODELAY
    stream.set_nodelay(true)?;

    // Send data
    let request = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
    stream.write_all(request)?;

    // Receive response
    let mut buffer = [0; 4096];
    let n = stream.read(&mut buffer)?;

    println!("Received {} bytes", n);

    Ok(())
}
```

### Async TCP Server (Python)

```python
import asyncio

async def handle_client(reader, writer):
    """
    Handle client connection asynchronously
    """
    addr = writer.get_extra_info('peername')
    print(f"Connection from {addr}")

    try:
        # Read data
        data = await asyncio.wait_for(reader.read(1024), timeout=10.0)
        message = data.decode()
        print(f"Received: {message}")

        # Process and respond
        response = f"Echo: {message}"
        writer.write(response.encode())
        await writer.drain()

    except asyncio.TimeoutError:
        print(f"Client {addr} timed out")

    except Exception as e:
        print(f"Error handling client {addr}: {e}")

    finally:
        # Close connection
        writer.close()
        await writer.wait_closed()

async def main():
    """
    Run async TCP server
    """
    server = await asyncio.start_server(
        handle_client,
        '0.0.0.0',
        8080,
        backlog=100
    )

    addr = server.sockets[0].getsockname()
    print(f"Serving on {addr}")

    async with server:
        await server.serve_forever()

# Run server
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped")
```

### Connection Pool Implementation

```python
import socket
import queue
import threading
import time

class TCPConnectionPool:
    """
    Simple TCP connection pool
    """
    def __init__(self, host, port, pool_size=5):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.pool = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """Create initial connections"""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            if conn:
                self.pool.put(conn)

    def _create_connection(self):
        """Create a new TCP connection"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30.0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.connect((self.host, self.port))
            return sock
        except Exception as e:
            print(f"Failed to create connection: {e}")
            return None

    def get_connection(self, timeout=5.0):
        """Get connection from pool"""
        try:
            conn = self.pool.get(timeout=timeout)

            # Verify connection is alive
            if not self._is_connection_alive(conn):
                conn.close()
                conn = self._create_connection()

            return conn

        except queue.Empty:
            # Pool exhausted, create new connection
            return self._create_connection()

    def return_connection(self, conn):
        """Return connection to pool"""
        if conn and self._is_connection_alive(conn):
            try:
                self.pool.put_nowait(conn)
            except queue.Full:
                # Pool full, close connection
                conn.close()
        elif conn:
            conn.close()

    def _is_connection_alive(self, conn):
        """Check if connection is still alive"""
        try:
            # Try to get socket error
            error = conn.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            return error == 0
        except:
            return False

    def close_all(self):
        """Close all connections in pool"""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except queue.Empty:
                break

# Example usage
pool = TCPConnectionPool('example.com', 80, pool_size=10)

def worker():
    """Worker thread using connection pool"""
    conn = pool.get_connection()
    if conn:
        try:
            conn.sendall(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
            response = conn.recv(4096)
            print(f"Received {len(response)} bytes")
        finally:
            pool.return_connection(conn)

# Create multiple workers
threads = []
for _ in range(20):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# Wait for completion
for t in threads:
    t.join()

pool.close_all()
```

## Best Practices

1. **Always close sockets**: Use try-finally or context managers
2. **Set appropriate timeouts**: Avoid hanging indefinitely
3. **Handle errors gracefully**: Network can fail at any time
4. **Use connection pooling**: Reuse connections for better performance
5. **Enable keepalive for long connections**: Detect dead connections
6. **Tune buffer sizes for workload**: Larger for throughput, smaller for latency
7. **Monitor connection states**: Watch for TIME_WAIT buildup
8. **Use TCP_NODELAY for interactive apps**: Reduce latency
9. **Enable window scaling for high-bandwidth**: Support larger windows
10. **Test under load**: Verify behavior under stress

## Real-World Troubleshooting Scenarios

### Scenario 1: High Latency Despite Good Bandwidth

**Symptoms:**
- Downloads are slow despite high bandwidth
- Small requests take long time
- Ping times are normal

**Diagnosis:**
```bash
# Check TCP statistics
ss -ti dst example.com

# Look for:
# - Small cwnd (congestion window)
# - High retransmissions
# - Small advertised window

# Check for bufferbloat
ping -c 100 example.com | tail -20

# Monitor real-time latency during transfer
while true; do ping -c 1 -W 1 example.com | grep time; sleep 0.5; done
```

**Possible causes and solutions:**
```python
import socket

# Solution 1: Enable TCP_NODELAY (disable Nagle)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# Solution 2: Increase buffer sizes
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)

# Solution 3: Check for bufferbloat in network equipment
# Use different congestion control (BBR is better for bufferbloat)
```

```bash
# System-wide fixes
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
sudo sysctl -w net.core.default_qdisc=fq
```

### Scenario 2: Connection Drops Frequently

**Symptoms:**
- Connections drop after period of inactivity
- "Connection reset by peer" errors
- Works fine with constant traffic

**Diagnosis:**
```bash
# Check if NAT or firewall has short timeout
# Monitor connection from both ends
watch -n 1 'ss -tan | grep ESTABLISHED'

# Check for middle boxes dropping idle connections
sudo tcpdump -i any -n 'tcp[tcpflags] & (tcp-rst) != 0'
```

**Solution:**
```python
import socket

def configure_keepalive_for_nat(sock):
    """
    Configure keepalive to prevent NAT timeout
    Most NATs timeout after 60-300 seconds
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    # Send probe after 30 seconds of idle time
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)

    # Send probes every 10 seconds
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)

    # Send 3 probes before giving up
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

    return sock

# Alternative: Application-level heartbeat
import time
import threading

def send_heartbeat(sock):
    """
    Application-level keepalive
    """
    while True:
        try:
            sock.sendall(b"PING\n")
            time.sleep(30)  # Every 30 seconds
        except:
            break

# Start heartbeat thread
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('example.com', 8080))
threading.Thread(target=send_heartbeat, args=(sock,), daemon=True).start()
```

### Scenario 3: Too Many TIME_WAIT Connections

**Symptoms:**
- Cannot create new outbound connections
- Error: "Cannot assign requested address"
- Many connections in TIME_WAIT state

**Diagnosis:**
```bash
# Count TIME_WAIT connections
ss -tan state time-wait | wc -l

# Show TIME_WAIT by remote host
ss -tan state time-wait | awk '{print $5}' | sort | uniq -c | sort -rn | head -10

# Check local port exhaustion
cat /proc/sys/net/ipv4/ip_local_port_range
```

**Solutions:**
```bash
# 1. Increase local port range
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# 2. Enable TIME_WAIT reuse (safe for clients)
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# 3. Reduce FIN timeout (careful!)
sudo sysctl -w net.ipv4.tcp_fin_timeout=30

# 4. Use connection pooling instead of opening/closing frequently
```

```python
# Application-level solution: Connection pooling
from urllib3 import PoolManager

# Reuse connections instead of creating new ones
http = PoolManager(
    maxsize=10,  # Pool size
    block=True,  # Block when pool is full
    timeout=30.0
)

# Make requests using pool
response = http.request('GET', 'http://example.com/')
```

### Scenario 4: Poor Performance Over VPN

**Symptoms:**
- Slow transfers over VPN
- High latency spikes
- Packet loss

**Diagnosis:**
```bash
# Check MTU issues
ping -M do -s 1472 vpn-host  # 1472 + 28 = 1500
ping -M do -s 1400 vpn-host  # Try smaller

# If 1472 fails but 1400 works, you have MTU issue

# Check current MTU
ip link show | grep mtu

# Measure path MTU
tracepath vpn-host
```

**Solutions:**
```bash
# 1. Reduce MTU on VPN interface
sudo ip link set dev tun0 mtu 1400

# 2. Enable TCP MSS clamping (on VPN server)
sudo iptables -t mangle -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu

# 3. Force MSS in application
```

```python
import socket

def configure_for_vpn(sock):
    """
    Configure socket for VPN connection
    """
    # Set smaller MSS for VPN
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1360)
    except:
        pass  # Not supported on all platforms

    # Use larger buffers to compensate for latency
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

    return sock
```

### Scenario 5: Retransmission Storms

**Symptoms:**
- Very high retransmission rate
- Degraded performance
- Network appears unstable

**Diagnosis:**
```bash
# Check retransmission statistics
netstat -s | grep retransmit

# Monitor in real-time
watch -n 1 'netstat -s | grep retransmit'

# Detailed per-connection retransmission info
ss -ti | grep -A 2 retrans

# Capture retransmissions with tcpdump
sudo tcpdump -i any -w retrans.pcap 'tcp[tcpflags] & tcp-syn != 0 or tcp[13] & 8 != 0'

# Analyze with Wireshark filter
# tcp.analysis.retransmission
```

**Solutions:**
```bash
# 1. Check for duplex mismatch (common cause)
ethtool eth0 | grep -i duplex
sudo ethtool -s eth0 speed 1000 duplex full autoneg on

# 2. Check for congestion
sar -n DEV 1 10  # Monitor interface utilization

# 3. Adjust congestion control
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr

# 4. Increase buffers if needed
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
```

### Scenario 6: Application Hangs on Connect

**Symptoms:**
- Connection attempts hang
- Eventually timeout
- No error, just slow

**Diagnosis:**
```python
import socket
import time

def diagnose_slow_connect(host, port):
    """
    Diagnose slow connection issues
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)

    try:
        print(f"Attempting connection to {host}:{port}")
        start = time.time()

        sock.connect((host, port))

        elapsed = time.time() - start
        print(f"Connected in {elapsed:.2f} seconds")

        if elapsed > 1.0:
            print("WARNING: Slow connection (> 1 second)")
            print("Possible causes:")
            print("- DNS resolution slow")
            print("- Firewall dropping SYN packets")
            print("- Server overloaded")
            print("- Network congestion")

    except socket.timeout:
        print(f"Connection timeout after {time.time() - start:.2f}s")
        print("Check if:")
        print("1. Host is reachable: ping", host)
        print("2. Port is open: telnet", host, port)
        print("3. Firewall blocking: check iptables/firewall rules")

    except ConnectionRefusedError:
        print("Connection refused - port is closed")

    except socket.gaierror as e:
        print(f"DNS resolution failed: {e}")

    finally:
        sock.close()

# Test
diagnose_slow_connect('example.com', 80)
```

**Solutions:**
```bash
# 1. Test DNS resolution
time nslookup example.com
# If slow, use different DNS or add to /etc/hosts

# 2. Test network path
traceroute example.com
mtr example.com  # Better tool

# 3. Test specific port
timeout 5 bash -c "</dev/tcp/example.com/80" && echo "Port open" || echo "Port closed"

# 4. Check firewall
sudo iptables -L -n | grep 80

# 5. Use shorter timeout in application
```

```python
# Better connection handling
import socket

def robust_connect(host, port, timeout=5):
    """
    Robust connection with proper error handling
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect((host, port))
        return sock
    except socket.timeout:
        print(f"Timeout connecting to {host}:{port}")
        raise
    except ConnectionRefusedError:
        print(f"Connection refused by {host}:{port}")
        raise
    except socket.gaierror as e:
        print(f"DNS error for {host}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

### Scenario 7: Inconsistent Performance

**Symptoms:**
- Performance varies wildly
- Sometimes fast, sometimes slow
- No clear pattern

**Diagnosis:**
```python
import socket
import time
import statistics

def benchmark_connection(host, port, iterations=10):
    """
    Benchmark TCP connection performance
    """
    connect_times = []
    transfer_times = []

    for i in range(iterations):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)

        # Measure connect time
        start = time.time()
        try:
            sock.connect((host, port))
            connect_time = time.time() - start
            connect_times.append(connect_time)

            # Measure transfer time
            request = b"GET / HTTP/1.1\r\nHost: " + host.encode() + b"\r\n\r\n"
            start = time.time()
            sock.sendall(request)
            data = sock.recv(1024)
            transfer_time = time.time() - start
            transfer_times.append(transfer_time)

        except Exception as e:
            print(f"Iteration {i+1} failed: {e}")

        finally:
            sock.close()

        time.sleep(0.5)  # Small delay between tests

    # Analyze results
    if connect_times:
        print(f"\nConnect time stats (n={len(connect_times)}):")
        print(f"  Mean: {statistics.mean(connect_times)*1000:.2f}ms")
        print(f"  Median: {statistics.median(connect_times)*1000:.2f}ms")
        print(f"  Stdev: {statistics.stdev(connect_times)*1000:.2f}ms")
        print(f"  Min: {min(connect_times)*1000:.2f}ms")
        print(f"  Max: {max(connect_times)*1000:.2f}ms")

    if transfer_times:
        print(f"\nTransfer time stats (n={len(transfer_times)}):")
        print(f"  Mean: {statistics.mean(transfer_times)*1000:.2f}ms")
        print(f"  Median: {statistics.median(transfer_times)*1000:.2f}ms")
        print(f"  Stdev: {statistics.stdev(transfer_times)*1000:.2f}ms")

# Run benchmark
benchmark_connection('example.com', 80, iterations=20)
```

### Debugging Tools Summary

```bash
# Essential TCP debugging tools

# 1. ss - Socket statistics (modern replacement for netstat)
ss -tan                    # All TCP connections
ss -tln                    # Listening TCP ports
ss -ti                     # Show TCP internals (timers, etc.)
ss -tm                     # Show socket memory usage

# 2. tcpdump - Packet capture
sudo tcpdump -i any port 80 -n -A           # Capture port 80, show ASCII
sudo tcpdump -i any -w capture.pcap         # Save to file
sudo tcpdump -r capture.pcap -n             # Read from file
sudo tcpdump 'tcp[tcpflags] & tcp-syn != 0' # Capture SYN packets

# 3. netstat - Network statistics
netstat -s                 # Protocol statistics
netstat -s | grep -i retrans   # Retransmission stats
netstat -i                 # Interface statistics

# 4. nstat - Network statistics delta
nstat -az                  # Show all stats since last reset
nstat TcpRetransSegs       # Monitor specific counter

# 5. iperf3 - Network performance testing
iperf3 -s                  # Server mode
iperf3 -c server_ip        # Client mode

# 6. mtr - Network path analysis
mtr example.com            # Interactive traceroute

# 7. socat - Advanced TCP tool
socat -v TCP-LISTEN:8080,fork EXEC:/bin/cat   # Debug server
socat - TCP:example.com:80                     # Simple client

# 8. lsof - List open files/sockets
sudo lsof -i TCP:80        # What's using port 80
sudo lsof -i -n -P         # All network connections

# 9. strace - System call tracing
strace -e trace=network nc example.com 80  # Trace network calls
strace -p <pid> -e trace=network           # Attach to process
```

## QUIC: Modern Alternative to TCP

QUIC (Quick UDP Internet Connections) is a modern transport protocol designed to address TCP's limitations.

### Why QUIC?

**TCP Limitations:**
```
1. Head-of-line blocking
   - One lost packet blocks entire stream
   - All data waits for retransmission

2. Slow connection establishment
   - TCP handshake: 1 RTT
   - TLS handshake: 1-2 RTTs
   - Total: 2-3 RTTs before sending data

3. Ossification
   - Middleboxes break TCP extensions
   - Hard to deploy improvements

4. No built-in encryption
   - TLS is separate layer
   - More complexity
```

### QUIC Advantages

**Key Features:**
```
1. Built on UDP
   - Avoids middlebox interference
   - Userspace implementation (faster updates)

2. Multiplexed streams
   - Multiple streams per connection
   - No head-of-line blocking between streams

3. 0-RTT connection establishment
   - Resume previous connections instantly
   - Send data in first packet

4. Built-in encryption (TLS 1.3)
   - Always encrypted
   - No plaintext handshake

5. Connection migration
   - Survives IP address changes
   - Mobile network switching

6. Improved congestion control
   - More accurate RTT measurement
   - Better loss detection
```

### QUIC vs TCP Comparison

| Feature | TCP | QUIC |
|---------|-----|------|
| Transport | Kernel space | Userspace |
| Connection setup | 1-3 RTTs | 0-1 RTT |
| Head-of-line blocking | Yes | No (per stream) |
| Encryption | Optional (TLS) | Built-in (TLS 1.3) |
| Stream multiplexing | No (HTTP/2 workaround) | Native |
| Connection migration | No | Yes |
| Ossification resistance | Low | High |
| CPU overhead | Lower | Higher |
| Deployment | Universal | Growing |

### QUIC Protocol Structure

```
QUIC Stack:
┌─────────────────────────┐
│   HTTP/3 Application    │
├─────────────────────────┤
│   QUIC Transport        │
│   - Streams             │
│   - Flow control        │
│   - Congestion control  │
├─────────────────────────┤
│   TLS 1.3 (built-in)    │
├─────────────────────────┤
│   UDP                   │
└─────────────────────────┘

vs

Traditional Stack:
┌─────────────────────────┐
│   HTTP/1.1 or HTTP/2    │
├─────────────────────────┤
│   TLS 1.2/1.3          │
├─────────────────────────┤
│   TCP                   │
├─────────────────────────┤
│   IP                    │
└─────────────────────────┘
```

### Connection Establishment

**TCP + TLS (2-3 RTTs):**
```
Client                          Server
  |                               |
  | TCP SYN ------------------>   |
  | <------------------ SYN-ACK   |
  | ACK ---------------------->   |  [1 RTT]
  |                               |
  | ClientHello -------------->   |
  | <-------- ServerHello, etc    |
  | Finished ----------------->   |  [1-2 RTTs]
  |                               |
  | HTTP Request ------------->   |
  | <------------ HTTP Response   |
```

**QUIC (0-1 RTT):**
```
Client                          Server
  |                               |
  | Initial (ClientHello) ---->   |
  | <-- Initial/Handshake/1-RTT  |  [1 RTT for new connection]
  | Handshake ---------------->   |
  |                               |
  | HTTP Request ------------->   |
  | <------------ HTTP Response   |

With 0-RTT resumption:
  |                               |
  | Initial + 0-RTT Data ------>  |  [0 RTT!]
  | <-- Initial/Handshake/1-RTT  |
```

### Stream Multiplexing

**TCP (with HTTP/2) - Head-of-line blocking:**
```
TCP Stream: [Stream1][Stream2][Stream3]
                ↓
If packet containing Stream2 data is lost:
    Stream1 data blocked ✗
    Stream2 data blocked ✗
    Stream3 data blocked ✗
All streams wait for retransmission!
```

**QUIC - No head-of-line blocking:**
```
QUIC Connection:
    Stream 1: [Data][Data][Data] ✓
    Stream 2: [Data][LOST][Data] ✗
    Stream 3: [Data][Data][Data] ✓

If packet containing Stream2 data is lost:
    Stream1 continues ✓
    Stream2 waits ✗
    Stream3 continues ✓
Only affected stream blocks!
```

### QUIC Implementation Example

**Python with aioquic:**
```python
import asyncio
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

async def quic_client():
    """
    QUIC client example using aioquic
    """
    # Configure QUIC
    configuration = QuicConfiguration(
        alpn_protocols=["h3"],  # HTTP/3
        is_client=True,
    )

    # Connect to server (0-RTT if resuming)
    async with connect(
        "quic.example.com",
        443,
        configuration=configuration,
    ) as client:
        # Send HTTP/3 request
        reader, writer = await client.create_stream()

        request = b"GET / HTTP/3\r\nHost: example.com\r\n\r\n"
        writer.write(request)
        await writer.drain()

        # Read response
        response = await reader.read()
        print(f"Received: {len(response)} bytes")

# Run
asyncio.run(quic_client())
```

**Node.js with node-quic:**
```javascript
const { createQuicSocket } = require('net');

// Create QUIC socket
const socket = createQuicSocket({ endpoint: { port: 0 } });

// Connect to server
const client = socket.connect({
  address: 'quic.example.com',
  port: 443,
  alpn: 'h3',  // HTTP/3
});

// Handle stream
client.on('stream', (stream) => {
  stream.on('data', (data) => {
    console.log(`Received: ${data.length} bytes`);
  });
});

// Create stream and send request
const stream = client.openStream();
stream.write('GET / HTTP/3\r\nHost: example.com\r\n\r\n');
```

### Connection Migration Example

```python
import asyncio
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration

async def demonstrate_migration():
    """
    QUIC survives network changes
    """
    configuration = QuicConfiguration(is_client=True)

    async with connect(
        "example.com",
        443,
        configuration=configuration,
    ) as client:
        # Create stream
        reader, writer = await client.create_stream()

        # Send data
        writer.write(b"Request 1")
        await writer.drain()

        # Network changes (WiFi -> 4G)
        # Client IP address changes
        # But connection continues!

        # QUIC automatically migrates using connection ID
        # No interruption to application

        # Continue using same stream
        writer.write(b"Request 2")
        await writer.drain()

        # Connection still works!
```

### When to Use QUIC vs TCP

**Use QUIC when:**
- Building web applications (HTTP/3)
- Need fast connection establishment
- Multiple parallel streams required
- Users on mobile networks (connection migration)
- Security is critical (built-in encryption)
- Latency is primary concern

**Use TCP when:**
- Maximum compatibility required
- IoT devices with limited resources
- Protocols that don't need multiplexing
- Environments that block UDP
- Lower CPU overhead required
- Existing TCP-optimized infrastructure

### QUIC Deployment Status

```bash
# Check if server supports QUIC/HTTP/3
curl -I --http3 https://www.google.com

# Major deployments:
# - Google (all services)
# - Facebook/Meta
# - Cloudflare
# - Fastly
# - LiteSpeed servers

# Browser support:
# - Chrome/Edge: Full support
# - Firefox: Full support
# - Safari: Full support (iOS 14.5+)

# Check QUIC support in browser:
# chrome://flags/#enable-quic
```

### QUIC Performance Testing

```bash
# Install quiche (Cloudflare's QUIC implementation)
git clone --recursive https://github.com/cloudflare/quiche
cd quiche

# Build HTTP/3 client
cargo build --release --examples

# Test QUIC connection
./target/release/examples/http3-client https://quic.tech:8443/

# Compare TCP vs QUIC
time curl https://example.com  # TCP
time curl --http3 https://example.com  # QUIC

# Use h2load for benchmarking
h2load -n 1000 -c 10 https://example.com  # HTTP/2 over TCP
h2load -n 1000 -c 10 --h3 https://example.com  # HTTP/3 over QUIC
```

### QUIC Congestion Control

QUIC uses pluggable congestion control:

```
Available algorithms:
1. CUBIC (default, similar to TCP)
2. BBR (Bottleneck Bandwidth and RTT)
3. Reno (classic TCP algorithm)
4. NewReno

Advantages over TCP:
- More accurate RTT measurement
- Better loss detection
- Faster convergence
- ACK frequency optimization
```

```python
from aioquic.quic.configuration import QuicConfiguration

config = QuicConfiguration()

# Use BBR congestion control
config.congestion_control_algorithm = "bbr"

# Or CUBIC
config.congestion_control_algorithm = "cubic"
```

### QUIC Security

**Built-in Security Features:**
```
1. Always encrypted (TLS 1.3)
   - No plaintext handshake
   - Forward secrecy by default

2. Connection ID
   - Prevents address spoofing
   - Enables connection migration

3. Packet protection
   - Header protection
   - Payload encryption

4. Version negotiation
   - Protected against downgrade attacks

5. Retry packets
   - DDoS mitigation
   - Similar to SYN cookies
```

### QUIC Limitations

**Current Challenges:**
```
1. UDP blocking
   - Some networks block UDP
   - Fallback to TCP still needed

2. CPU overhead
   - Userspace implementation
   - More processing required
   - Battery impact on mobile

3. Middlebox issues
   - Some firewalls drop QUIC
   - NAT traversal complexity

4. Maturity
   - Newer protocol
   - Fewer debugging tools
   - Less operational experience

5. OS support
   - Not kernel-integrated (yet)
   - Inconsistent across platforms
```

### Future of TCP and QUIC

```
TCP will remain important for:
- Legacy systems and protocols
- Environments blocking UDP
- Low-overhead requirements
- IoT and embedded systems

QUIC adoption growing for:
- Web browsing (HTTP/3)
- Video streaming
- Real-time communications
- Mobile applications
- API services

Convergence:
- TCP improvements inspired by QUIC
- QUIC learning from TCP experience
- Coexistence rather than replacement
```

## References

- [RFC 793](https://tools.ietf.org/html/rfc793) - TCP Specification
- [RFC 1323](https://tools.ietf.org/html/rfc1323) - TCP Extensions (Window Scaling, Timestamps)
- [RFC 2018](https://tools.ietf.org/html/rfc2018) - TCP Selective Acknowledgment
- [RFC 7413](https://tools.ietf.org/html/rfc7413) - TCP Fast Open
- [RFC 8684](https://tools.ietf.org/html/rfc8684) - Multipath TCP
- [RFC 9000](https://tools.ietf.org/html/rfc9000) - QUIC: A UDP-Based Multiplexed and Secure Transport
- [RFC 9114](https://tools.ietf.org/html/rfc9114) - HTTP/3
