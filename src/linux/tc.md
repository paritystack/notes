# tc (Traffic Control)

## Table of Contents
- [Overview](#overview)
- [Important Components](#important-components)
- [Command Syntax](#command-syntax)
- [Common Queuing Disciplines (qdiscs)](#common-queuing-disciplines-qdiscs)
- [Uses of tc](#uses-of-tc)
- [tc vs Netfilter (iptables/nftables)](#tc-vs-netfilter-iptablesnftables)
  - [Overview of Netfilter](#overview-of-netfilter)
  - [Key Differences](#key-differences)
  - [Working Together](#working-together)
  - [Choosing the Right Tool](#choosing-the-right-tool)
- [Practical Examples](#practical-examples)
  - [Basic Network Emulation](#basic-network-emulation)
  - [Bandwidth Limiting](#bandwidth-limiting)
  - [Traffic Shaping with HTB](#traffic-shaping-with-htb)
  - [QoS Prioritization](#qos-prioritization)
  - [Advanced Network Emulation](#advanced-network-emulation)
- [Viewing and Managing Configurations](#viewing-and-managing-configurations)
- [Real-World Scenarios](#real-world-scenarios)
- [Troubleshooting Tips](#troubleshooting-tips)

## Overview

`tc` (traffic control) is a powerful utility in the Linux kernel used to configure Traffic Control in the network stack. It allows administrators to configure the queuing discipline (qdisc), which determines how packets are enqueued and dequeued from network interfaces.

Traffic control enables you to:
- Control bandwidth usage
- Prioritize specific types of traffic
- Simulate various network conditions
- Implement Quality of Service (QoS) policies
- Test application performance under different network scenarios

## Important Components

1. **qdisc (Queuing Discipline)**: The core component of `tc`, which defines the algorithm used to manage the packet queue. Examples include `pfifo_fast`, `fq_codel`, `htb`, and `netem`.

2. **class**: A way to create a hierarchy within a qdisc, allowing for more granular control over traffic. Classes can be used to apply different rules to different types of traffic.

3. **filter**: Used to classify packets into different classes. Filters can match on various packet attributes, such as IP address, port number, or protocol.

4. **action**: Defines what to do with packets that match a filter. Actions can include marking, mirroring, or redirecting packets.

## Command Syntax

Basic `tc` command structure:

```bash
tc qdisc [ add | del | replace | change | show ] dev DEVICE [ parent QHANDLE ] [ handle QHANDLE ] [ QDISC ]
tc class [ add | del | change | show ] dev DEVICE parent QHANDLE [ classid CLASSID ] [ QDISC ]
tc filter [ add | del | change | show ] dev DEVICE [ parent QHANDLE ] protocol PROTOCOL prio PRIORITY filtertype [ filtertype-specific-parameters ]
```

Key components:
- **dev**: Network interface (e.g., eth0, wlan0)
- **parent**: Parent qdisc or class handle
- **handle**: Identifier for the qdisc or class
- **root**: Top-level qdisc (no parent)

## Common Queuing Disciplines (qdiscs)

### pfifo_fast (Default)
The default qdisc for most interfaces. It has three priority bands and uses a simple FIFO algorithm.

### HTB (Hierarchical Token Bucket)
Allows creating a hierarchy of rate-limited classes. Excellent for bandwidth shaping and guaranteeing rates.

### fq_codel (Fair Queuing with Controlled Delay)
Modern queue management algorithm that combines fair queuing with active queue management. Good for reducing bufferbloat.

### netem (Network Emulator)
Used to emulate network conditions like delay, packet loss, duplication, and reordering.

### TBF (Token Bucket Filter)
Simple qdisc for rate limiting. Packets are transmitted only when tokens are available.

### prio (Priority Scheduler)
Similar to pfifo_fast but allows more control over priority bands.

## Uses of tc

- **Traffic Shaping**: Control the rate of outgoing traffic to ensure that the network is not overwhelmed. This can be useful for managing bandwidth usage and ensuring fair distribution of network resources.

- **Traffic Policing**: Enforce limits on the rate of incoming traffic, dropping packets that exceed the specified rate. This can help protect against network abuse or attacks.

- **Network Emulation**: Simulate various network conditions, such as latency, packet loss, and jitter, to test the performance of applications under different scenarios.

- **Quality of Service (QoS)**: Prioritize certain types of traffic to ensure that critical applications receive the necessary bandwidth and low latency.

## tc vs Netfilter (iptables/nftables)

Understanding the differences between `tc` and netfilter (implemented via `iptables` or `nftables`) is crucial for effective network management in Linux. While both tools can manipulate network traffic, they serve different purposes and operate at different layers of the network stack.

### Overview of Netfilter

**Netfilter** is a packet filtering framework in the Linux kernel. It's primarily accessed through user-space tools like:
- **iptables**: Traditional tool for IPv4 packet filtering
- **ip6tables**: IPv4's counterpart for IPv6
- **nftables**: Modern replacement for iptables, offering improved performance and syntax

Netfilter operates at the network layer (Layer 3) and is primarily used for:
- Packet filtering (firewall rules)
- Network Address Translation (NAT)
- Port forwarding
- Packet mangling (modifying packet headers)
- Connection tracking

### Key Differences

| Aspect | tc (Traffic Control) | Netfilter (iptables/nftables) |
|--------|---------------------|-------------------------------|
| **Primary Purpose** | Traffic shaping and QoS | Packet filtering and firewalling |
| **Operating Layer** | Layer 2 (Link) and Layer 3 (Network) | Layer 3 (Network) and Layer 4 (Transport) |
| **Default Direction** | Egress (outgoing) traffic | Both ingress and egress |
| **Bandwidth Control** | Native and sophisticated | Limited, requires additional modules |
| **Packet Filtering** | Basic classification | Advanced and flexible |
| **Queue Management** | Extensive (multiple qdiscs) | Not applicable |
| **NAT/Port Forwarding** | Not supported | Native support |
| **Performance Impact** | Lower for shaping tasks | Lower for filtering tasks |
| **Configuration** | Complex syntax | Relatively straightforward rules |
| **State Tracking** | Limited | Connection tracking (conntrack) |

### Detailed Comparison

#### Operating in the Network Stack

**tc** operates primarily at the **egress** (outgoing) side of network interfaces:
- Processes packets as they leave the interface
- Controls queuing disciplines and scheduling
- For ingress control, requires IFB (Intermediate Functional Block) devices
- Works at the queueing layer

**Netfilter** operates at multiple **hook points** in the network stack:
- PREROUTING: Before routing decision
- INPUT: For packets destined to local system
- FORWARD: For packets being routed through the system
- OUTPUT: For locally generated packets
- POSTROUTING: After routing decision

#### Primary Use Cases

**Use tc when you need to:**
- Limit bandwidth for specific applications or interfaces
- Implement Quality of Service (QoS) policies
- Shape traffic to prevent network congestion
- Guarantee minimum bandwidth for critical services
- Simulate network conditions (latency, packet loss, jitter)
- Control bufferbloat
- Implement hierarchical bandwidth allocation
- Prioritize traffic based on complex criteria

**Use netfilter when you need to:**
- Filter packets based on IP, port, protocol, or state
- Implement firewall rules
- Perform Network Address Translation (NAT)
- Forward ports to different hosts
- Block or allow specific traffic
- Implement connection tracking
- Protect against network attacks
- Route packets differently based on criteria
- Log network traffic

### Working Together

`tc` and netfilter complement each other and can work together effectively. A common pattern is to use **netfilter to mark packets** and **tc to classify and shape** based on those marks.

#### Example: Mark packets with iptables, shape with tc

```bash
# 1. Mark SSH traffic with iptables
sudo iptables -t mangle -A OUTPUT -p tcp --sport 22 -j MARK --set-mark 1
sudo iptables -t mangle -A OUTPUT -p tcp --dport 22 -j MARK --set-mark 1

# 2. Mark HTTP traffic with iptables
sudo iptables -t mangle -A OUTPUT -p tcp --sport 80 -j MARK --set-mark 2
sudo iptables -t mangle -A OUTPUT -p tcp --dport 80 -j MARK --set-mark 2

# 3. Setup tc HTB qdisc
sudo tc qdisc add dev eth0 root handle 1: htb default 30

# 4. Create classes
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 10mbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 6mbit ceil 10mbit prio 1  # SSH
sudo tc class add dev eth0 parent 1:1 classid 1:20 htb rate 3mbit ceil 8mbit prio 2   # HTTP
sudo tc class add dev eth0 parent 1:1 classid 1:30 htb rate 1mbit ceil 5mbit prio 3   # Other

# 5. Create tc filters based on iptables marks
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 handle 1 fw flowid 1:10  # SSH
sudo tc filter add dev eth0 parent 1: protocol ip prio 2 handle 2 fw flowid 1:20  # HTTP
```

#### Example: Filter with iptables, then shape with tc

```bash
# 1. Allow only specific traffic through firewall
sudo iptables -A OUTPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A OUTPUT -j DROP

# 2. Shape the allowed traffic with tc
sudo tc qdisc add dev eth0 root handle 1: htb default 10
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 50mbit ceil 100mbit
```

#### Example: NAT with iptables, bandwidth limit with tc

```bash
# 1. Setup NAT for local network (iptables handles routing)
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# 2. Limit total bandwidth for NATed traffic (tc handles shaping)
sudo tc qdisc add dev eth0 root handle 1: htb default 10
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 50mbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 50mbit ceil 50mbit
```

### Choosing the Right Tool

**Use tc alone when:**
- You need sophisticated bandwidth management
- Implementing QoS is the primary goal
- Testing application performance under various network conditions
- Controlling buffer bloat
- No packet filtering is required

**Use netfilter alone when:**
- You need to filter packets (firewall)
- Implementing NAT or port forwarding
- Packet filtering based on connection state
- Logging network traffic
- No bandwidth control is needed

**Use both together when:**
- You need both firewalling and traffic shaping
- Complex QoS with packet classification based on multiple criteria
- Implementing enterprise-grade network policies
- Need to mark packets for classification in tc
- Building a router or gateway with QoS

### Performance Considerations

- **tc** is more efficient for bandwidth limiting and queuing operations
- **Netfilter** is more efficient for packet filtering and state tracking
- Using both together adds some overhead but provides maximum flexibility
- For simple rate limiting without complex rules, tc alone is often sufficient
- For complex packet filtering without bandwidth control, netfilter alone is appropriate

### Common Misconceptions

1. **"tc can replace iptables"**: False. tc cannot filter packets or provide firewall functionality.

2. **"iptables can do traffic shaping"**: Partially true. While iptables has some rate limiting capabilities (like `limit` and `hashlimit` modules), they are far less sophisticated than tc's QoS features.

3. **"tc only works on outgoing traffic"**: Mostly true. tc primarily controls egress traffic, but can be configured for ingress using IFB devices or ingress qdiscs.

4. **"Netfilter can control bandwidth as well as tc"**: False. Netfilter's rate limiting is packet-based and much simpler than tc's queue-based traffic control.

## Practical Examples

### Basic Network Emulation

#### Add delay to all traffic
```bash
sudo tc qdisc add dev eth0 root netem delay 100ms
```

#### Add delay with variation (jitter)
```bash
sudo tc qdisc add dev eth0 root netem delay 100ms 20ms
```

#### Simulate packet loss
```bash
# 10% packet loss
sudo tc qdisc add dev eth0 root netem loss 10%
```

#### Combine delay and packet loss
```bash
sudo tc qdisc add dev eth0 root netem delay 100ms loss 5%
```

### Bandwidth Limiting

#### Limit bandwidth using TBF
```bash
# Limit to 1mbit/s with burst of 32kbit and latency of 400ms
sudo tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms
```

#### Simple rate limiting
```bash
# Limit to 10mbit/s
sudo tc qdisc add dev eth0 root tbf rate 10mbit burst 32kbit latency 400ms
```

### Traffic Shaping with HTB

#### Create HTB qdisc with rate limits
```bash
# Add root HTB qdisc
sudo tc qdisc add dev eth0 root handle 1: htb default 30

# Create root class with 10mbit ceiling
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 10mbit

# Create child classes
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 5mbit ceil 10mbit
sudo tc class add dev eth0 parent 1:1 classid 1:20 htb rate 3mbit ceil 10mbit
sudo tc class add dev eth0 parent 1:1 classid 1:30 htb rate 2mbit ceil 10mbit
```

#### Add filters to classify traffic
```bash
# Send traffic to port 80 to class 1:10
sudo tc filter add dev eth0 protocol ip parent 1: prio 1 u32 \
    match ip dport 80 0xffff flowid 1:10

# Send traffic to port 22 to class 1:20 (prioritize SSH)
sudo tc filter add dev eth0 protocol ip parent 1: prio 1 u32 \
    match ip dport 22 0xffff flowid 1:20
```

### QoS Prioritization

#### Using prio qdisc for priority bands
```bash
# Create prio qdisc with 3 bands
sudo tc qdisc add dev eth0 root handle 1: prio bands 3

# Add filters to classify traffic
# High priority: SSH (port 22)
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
    match ip dport 22 0xffff flowid 1:1

# Medium priority: HTTP/HTTPS (ports 80, 443)
sudo tc filter add dev eth0 parent 1: protocol ip prio 2 u32 \
    match ip dport 80 0xffff flowid 1:2

# Low priority: everything else (default band 3)
```

### Advanced Network Emulation

#### Simulate mobile network conditions (3G)
```bash
# Typical 3G: 2mbit, 100ms latency, 1% loss
sudo tc qdisc add dev eth0 root netem rate 2mbit delay 100ms loss 1%
```

#### Simulate high-latency satellite connection
```bash
# Satellite: 500ms delay with 50ms variation
sudo tc qdisc add dev eth0 root netem delay 500ms 50ms
```

#### Packet reordering
```bash
# 25% of packets will be delayed by 10ms causing reordering
sudo tc qdisc add dev eth0 root netem delay 10ms reorder 25% 50%
```

#### Packet duplication
```bash
# Duplicate 1% of packets
sudo tc qdisc add dev eth0 root netem duplicate 1%
```

#### Packet corruption
```bash
# Corrupt 0.1% of packets
sudo tc qdisc add dev eth0 root netem corrupt 0.1%
```

#### Complex scenario combining multiple effects
```bash
# Simulate degraded network: 5mbit, 200ms delay, 5% loss, occasional duplicates
sudo tc qdisc add dev eth0 root netem rate 5mbit delay 200ms 50ms loss 5% duplicate 1%
```

## Viewing and Managing Configurations

### View current qdisc configuration
```bash
# Show all qdiscs
sudo tc qdisc show

# Show qdiscs for specific interface
sudo tc qdisc show dev eth0
```

### View class configuration
```bash
# Show all classes
sudo tc class show dev eth0

# Show with statistics
sudo tc -s class show dev eth0
```

### View filter configuration
```bash
sudo tc filter show dev eth0
```

### View detailed statistics
```bash
# Detailed qdisc statistics
sudo tc -s qdisc show dev eth0

# More detailed statistics with timestamps
sudo tc -s -d qdisc show dev eth0
```

### Remove qdisc
```bash
# Remove root qdisc (removes all classes and filters too)
sudo tc qdisc del dev eth0 root

# Remove specific qdisc
sudo tc qdisc del dev eth0 parent 1:1 handle 10:
```

### Replace existing qdisc
```bash
# Replace root qdisc
sudo tc qdisc replace dev eth0 root netem delay 50ms
```

### Change existing qdisc parameters
```bash
# Change delay from 100ms to 200ms
sudo tc qdisc change dev eth0 root netem delay 200ms
```

## Real-World Scenarios

### Scenario 1: Prioritize SSH over bulk downloads
```bash
# Setup HTB with two classes
sudo tc qdisc add dev eth0 root handle 1: htb default 20
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 10mbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 8mbit ceil 10mbit prio 0
sudo tc class add dev eth0 parent 1:1 classid 1:20 htb rate 2mbit ceil 10mbit prio 1

# Prioritize SSH
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
    match ip dport 22 0xffff flowid 1:10
```

### Scenario 2: Limit download speed from specific subnet
```bash
# Create HTB with rate limit
sudo tc qdisc add dev eth0 root handle 1: htb default 20
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 1mbit ceil 2mbit

# Match traffic from 192.168.1.0/24
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
    match ip src 192.168.1.0/24 flowid 1:10
```

### Scenario 3: Test application resilience to network issues
```bash
# Simulate unreliable network
sudo tc qdisc add dev eth0 root netem delay 150ms 50ms loss 3% corrupt 0.1% duplicate 0.5%

# Run your tests...

# Remove when done
sudo tc qdisc del dev eth0 root
```

### Scenario 4: Bandwidth allocation for web server
```bash
# Setup HTB for web server with guaranteed bandwidth
sudo tc qdisc add dev eth0 root handle 1: htb default 30

# Root class - total bandwidth
sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit

# HTTP traffic - guaranteed 50mbit, can use up to 80mbit
sudo tc class add dev eth0 parent 1:1 classid 1:10 htb rate 50mbit ceil 80mbit prio 1

# HTTPS traffic - guaranteed 40mbit, can use up to 80mbit
sudo tc class add dev eth0 parent 1:1 classid 1:20 htb rate 40mbit ceil 80mbit prio 1

# Other traffic - guaranteed 10mbit, can use up to 20mbit
sudo tc class add dev eth0 parent 1:1 classid 1:30 htb rate 10mbit ceil 20mbit prio 2

# Add filters
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
    match ip sport 80 0xffff flowid 1:10
sudo tc filter add dev eth0 parent 1: protocol ip prio 1 u32 \
    match ip sport 443 0xffff flowid 1:20
```

## Troubleshooting Tips

### Common Issues

1. **Permission Denied**: Most `tc` commands require root privileges. Use `sudo`.

2. **Device Not Found**: Ensure the network interface exists and is spelled correctly.
   ```bash
   ip link show  # List all interfaces
   ```

3. **Cannot add qdisc (File exists)**: A qdisc already exists. Delete it first or use `replace`.
   ```bash
   sudo tc qdisc del dev eth0 root
   ```

4. **Changes not taking effect**:
   - Verify the qdisc is actually applied: `tc -s qdisc show dev eth0`
   - Check filter rules: `tc filter show dev eth0`
   - Ensure you're testing with the correct interface

5. **HTB not working as expected**:
   - Verify class hierarchy is correct
   - Check that filters are properly directing traffic to classes
   - Use `tc -s class show dev eth0` to see which classes are receiving traffic

### Best Practices

- Always test traffic control rules in a non-production environment first
- Document your tc configurations as they don't persist across reboots
- Use `tc -s` to monitor actual traffic through classes and qdiscs
- Start with simple configurations and gradually add complexity
- Remember that tc only controls egress (outgoing) traffic by default
- For ingress (incoming) traffic control, use IFB (Intermediate Functional Block) devices

### Making tc Rules Persistent

Traffic control rules are not persistent across reboots. To make them persistent:

1. **Using systemd service**: Create a systemd service that runs your tc script at boot
2. **Using network manager scripts**: Add tc commands to network-up scripts
3. **Using rc.local**: Add commands to `/etc/rc.local` (on systems that support it)

Example systemd service:
```bash
# /etc/systemd/system/tc-setup.service
[Unit]
Description=Traffic Control Setup
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/tc-setup.sh

[Install]
WantedBy=multi-user.target
```

By using `tc`, administrators can fine-tune network performance, improve reliability, and ensure that critical applications have the necessary resources to function optimally.
