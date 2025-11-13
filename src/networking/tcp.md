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

## References

- [RFC 793](https://tools.ietf.org/html/rfc793) - TCP Specification
- [RFC 1323](https://tools.ietf.org/html/rfc1323) - TCP Extensions (Window Scaling, Timestamps)
- [RFC 2018](https://tools.ietf.org/html/rfc2018) - TCP Selective Acknowledgment
- [RFC 7413](https://tools.ietf.org/html/rfc7413) - TCP Fast Open
- [RFC 8684](https://tools.ietf.org/html/rfc8684) - Multipath TCP
