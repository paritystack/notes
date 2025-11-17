# Load Balancing

Load balancing is a critical component of distributed systems that distributes incoming network traffic across multiple servers to ensure no single server bears too much demand. By spreading the work evenly, load balancing improves application responsiveness, increases availability, and enables horizontal scaling.

## Table of Contents
- [Introduction](#introduction)
- [Load Balancing Fundamentals](#load-balancing-fundamentals)
- [OSI Layer Load Balancing](#osi-layer-load-balancing)
  - [Layer 4 (Transport Layer)](#layer-4-transport-layer)
  - [Layer 7 (Application Layer)](#layer-7-application-layer)
  - [L4 vs L7 Comparison](#l4-vs-l7-comparison)
- [Load Balancing Algorithms](#load-balancing-algorithms)
  - [Round Robin](#round-robin)
  - [Weighted Round Robin](#weighted-round-robin)
  - [Least Connections](#least-connections)
  - [Weighted Least Connections](#weighted-least-connections)
  - [IP Hash](#ip-hash)
  - [Consistent Hashing](#consistent-hashing)
  - [Least Response Time](#least-response-time)
  - [Random Selection](#random-selection)
  - [Resource-Based](#resource-based)
- [Health Checks and Monitoring](#health-checks-and-monitoring)
- [Session Persistence](#session-persistence)
- [SSL/TLS Termination](#ssltls-termination)
- [Cloud Load Balancers](#cloud-load-balancers)
  - [AWS Load Balancers](#aws-load-balancers)
  - [Google Cloud Load Balancing](#google-cloud-load-balancing)
  - [Azure Load Balancers](#azure-load-balancers)
- [Software Load Balancers](#software-load-balancers)
  - [NGINX](#nginx)
  - [HAProxy](#haproxy)
  - [Envoy](#envoy)
- [DNS-Based Load Balancing](#dns-based-load-balancing)
- [Global Server Load Balancing (GSLB)](#global-server-load-balancing-gslb)
- [Real-World Architectures](#real-world-architectures)
- [Performance Tuning](#performance-tuning)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Further Reading](#further-reading)

---

## Introduction

**What is Load Balancing?**

Load balancing distributes client requests or network load efficiently across multiple servers. It ensures that no single server becomes overwhelmed, which could lead to degraded performance or downtime.

**Why Load Balancing Matters:**

1. **High Availability**: If one server fails, traffic is automatically routed to healthy servers
2. **Scalability**: Add or remove servers based on demand without downtime
3. **Performance**: Distribute load to prevent bottlenecks and reduce response times
4. **Flexibility**: Perform maintenance on servers without affecting service availability
5. **Geographic Distribution**: Route users to the nearest data center for lower latency

**Basic Architecture:**

```
                    Internet
                       ↓
                 Load Balancer
                 /     |     \
                /      |      \
           Server1  Server2  Server3
              ↓        ↓        ↓
           Database Database Database
```

**Key Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| **Throughput** | Requests per second | Maximize |
| **Latency** | Response time | < 100ms |
| **Error Rate** | Failed requests | < 0.1% |
| **Availability** | Uptime percentage | > 99.9% |
| **Connection Count** | Active connections | Monitor capacity |

---

## Load Balancing Fundamentals

### Core Concepts

**1. Server Pool (Backend Pool)**
- Group of servers that receive distributed traffic
- Can be homogeneous or heterogeneous
- Dynamically adjusted based on demand

**2. Virtual IP (VIP)**
- Single IP address that clients connect to
- Load balancer listens on this address
- Hides complexity of backend infrastructure

**3. Backend Servers**
- Also called "real servers" or "pool members"
- Handle actual application logic
- Can be added/removed dynamically

**4. Health Monitoring**
- Continuous checking of server availability
- Automatic removal of unhealthy servers
- Automatic restoration when servers recover

### Load Balancing Flow

```
1. Client sends request to VIP (e.g., www.example.com)
   ↓
2. DNS resolves to load balancer IP
   ↓
3. Load balancer receives connection
   ↓
4. Algorithm selects backend server
   ↓
5. Load balancer forwards request
   ↓
6. Backend processes and responds
   ↓
7. Load balancer returns response to client
```

### Types of Load Balancers

**1. Hardware Load Balancers**
- Dedicated physical devices (F5, Citrix NetScaler)
- High performance and reliability
- Expensive and less flexible
- Used in enterprise data centers

**2. Software Load Balancers**
- Run on commodity hardware or VMs
- Cost-effective and flexible
- Examples: NGINX, HAProxy, Envoy
- Easy to scale and configure

**3. Cloud Load Balancers**
- Managed services from cloud providers
- Auto-scaling and high availability built-in
- Pay-per-use pricing
- Examples: AWS ALB, GCP Load Balancing

**4. DNS Load Balancers**
- Distribute traffic via DNS responses
- Geographic distribution
- Simple but with limitations (caching, TTL)

### Load Balancer Deployment Modes

**1. Inline (Proxy) Mode**
```
Client → Load Balancer → Server
         (modifies packets)
```
- Load balancer acts as proxy
- Can modify requests/responses
- Full visibility and control

**2. Direct Server Return (DSR)**
```
Request:  Client → Load Balancer → Server
Response: Server → Client (bypasses LB)
```
- Reduces load balancer bandwidth
- Faster response delivery
- Complex configuration

**3. Transparent Mode**
```
Client → Load Balancer → Server
         (Layer 2/3 only)
```
- No IP address changes
- Works at network layer
- Limited application awareness

---

## OSI Layer Load Balancing

Understanding the OSI model helps in choosing the right load balancing strategy.

```
Layer 7: Application (HTTP, HTTPS, gRPC)  ← L7 Load Balancing
Layer 6: Presentation (SSL/TLS)
Layer 5: Session
Layer 4: Transport (TCP, UDP)             ← L4 Load Balancing
Layer 3: Network (IP)
Layer 2: Data Link (MAC)
Layer 1: Physical
```

### Layer 4 (Transport Layer)

**How It Works:**
- Operates at TCP/UDP level
- Routes based on IP address and port
- No inspection of packet contents
- Fast and efficient

**Characteristics:**
- Protocol agnostic (works with any application protocol)
- Lower latency (minimal processing)
- Higher throughput
- Cannot make content-based decisions
- Simple session persistence (source IP)

**Use Cases:**
- High-performance applications
- Non-HTTP protocols (databases, game servers)
- When content inspection is unnecessary
- Maximum throughput requirements

**Example: L4 Decision Making**

```
Incoming Packet:
  Source IP: 192.168.1.100
  Source Port: 54321
  Dest IP: 10.0.0.1 (VIP)
  Dest Port: 80
  Protocol: TCP

Load Balancer Decision:
  Algorithm: Round Robin
  Selected Backend: 10.0.0.10:80

Forwarded Packet:
  Source IP: 10.0.0.1 (LB IP)
  Source Port: 12345
  Dest IP: 10.0.0.10
  Dest Port: 80
  Protocol: TCP
```

**L4 Configuration Example (HAProxy):**

```haproxy
frontend mysql_frontend
    bind *:3306
    mode tcp
    default_backend mysql_backend

backend mysql_backend
    mode tcp
    balance roundrobin
    server mysql1 10.0.1.10:3306 check
    server mysql2 10.0.1.11:3306 check
    server mysql3 10.0.1.12:3306 check
```

### Layer 7 (Application Layer)

**How It Works:**
- Operates at application protocol level
- Inspects HTTP headers, URLs, cookies
- Can modify requests and responses
- Content-based routing

**Characteristics:**
- Protocol-specific (HTTP, HTTPS, gRPC)
- Content-aware routing
- SSL termination
- Request/response modification
- Advanced session persistence
- Higher CPU overhead

**Use Cases:**
- Web applications
- Microservices routing
- API gateways
- Content-based routing
- SSL offloading

**Example: L7 Decision Making**

```
HTTP Request:
  GET /api/users/123 HTTP/1.1
  Host: api.example.com
  Cookie: session=abc123
  User-Agent: Mobile App
  X-API-Version: v2

Load Balancer Decisions:
  ✓ Route based on path (/api/users → User Service)
  ✓ Route based on header (X-API-Version: v2 → V2 Servers)
  ✓ Sticky session (session=abc123 → Server 2)
  ✓ Device routing (Mobile App → Mobile Optimized Servers)
```

**L7 Configuration Example (NGINX):**

```nginx
http {
    upstream api_servers {
        server 10.0.1.10:8080;
        server 10.0.1.11:8080;
        server 10.0.1.12:8080;
    }

    upstream static_servers {
        server 10.0.2.10:8080;
        server 10.0.2.11:8080;
    }

    server {
        listen 80;
        server_name api.example.com;

        # Route API requests
        location /api/ {
            proxy_pass http://api_servers;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Route static content
        location /static/ {
            proxy_pass http://static_servers;
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
        }
    }
}
```

### L4 vs L7 Comparison

| Feature | Layer 4 (L4) | Layer 7 (L7) |
|---------|--------------|--------------|
| **Speed** | Very fast | Moderate |
| **Resource Usage** | Low CPU/Memory | Higher CPU/Memory |
| **Protocol Support** | Any TCP/UDP | HTTP, HTTPS, gRPC, etc. |
| **Content Awareness** | No | Yes |
| **Routing Granularity** | IP:Port only | URL, headers, cookies |
| **SSL Termination** | No (passthrough) | Yes |
| **Session Persistence** | Source IP | Cookie, header-based |
| **DDoS Protection** | Basic | Advanced |
| **Caching** | No | Yes |
| **Compression** | No | Yes |
| **Cost** | Lower | Higher |
| **Use Case** | High throughput | Smart routing |

**When to Use L4:**
- ✓ Maximum performance needed
- ✓ Non-HTTP protocols
- ✓ Simple routing requirements
- ✓ End-to-end encryption required
- ✓ Database load balancing

**When to Use L7:**
- ✓ Web applications
- ✓ Microservices architecture
- ✓ Content-based routing
- ✓ SSL offloading
- ✓ API gateway functionality
- ✓ Rate limiting and WAF

**Hybrid Approach:**

```
            Internet
               ↓
    L7 Load Balancer (NGINX)
    /                    \
   /                      \
L4 LB (TCP)            L4 LB (TCP)
  ↓  ↓                   ↓  ↓
DB Servers           Cache Servers
```

---

## Load Balancing Algorithms

The algorithm determines how traffic is distributed across backend servers. Choosing the right algorithm is crucial for performance and reliability.

### Round Robin

**How It Works:**
Distributes requests sequentially to each server in the pool.

```
Request 1 → Server A
Request 2 → Server B
Request 3 → Server C
Request 4 → Server A (cycle repeats)
Request 5 → Server B
Request 6 → Server C
```

**Characteristics:**
- Simple and fair distribution
- No server state tracking required
- Works well with equal capacity servers
- May not account for server load

**Implementation:**

```python
class RoundRobinLoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0

    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# Usage
lb = RoundRobinLoadBalancer(['server1', 'server2', 'server3'])
print(lb.get_server())  # server1
print(lb.get_server())  # server2
print(lb.get_server())  # server3
print(lb.get_server())  # server1
```

**NGINX Configuration:**

```nginx
upstream backend {
    # Round robin is the default
    server backend1.example.com;
    server backend2.example.com;
    server backend3.example.com;
}
```

**Pros:**
- ✓ Simple to implement
- ✓ Equal distribution
- ✓ Low overhead
- ✓ Predictable behavior

**Cons:**
- ✗ Ignores server capacity
- ✗ Ignores current load
- ✗ Ignores server response time
- ✗ May overload slower servers

**Best For:**
- Homogeneous server pools
- Short-lived connections
- Stateless applications
- Similar request processing times

### Weighted Round Robin

**How It Works:**
Distributes requests based on server capacity weights.

```
Servers:
  Server A: weight = 5
  Server B: weight = 3
  Server C: weight = 2

Distribution (out of 10 requests):
  Server A: 5 requests (50%)
  Server B: 3 requests (30%)
  Server C: 2 requests (20%)

Sequence:
Request 1 → Server A
Request 2 → Server A
Request 3 → Server B
Request 4 → Server A
Request 5 → Server C
Request 6 → Server A
Request 7 → Server B
Request 8 → Server A
Request 9 → Server B
Request 10 → Server C
```

**Implementation:**

```python
class WeightedRoundRobinLoadBalancer:
    def __init__(self, servers):
        # servers = [('server1', 5), ('server2', 3), ('server3', 2)]
        self.servers = []
        for server, weight in servers:
            self.servers.extend([server] * weight)
        self.current = 0

    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# Advanced: Smooth Weighted Round Robin (Nginx algorithm)
class SmoothWeightedRoundRobin:
    def __init__(self, servers):
        # servers = [('server1', 5), ('server2', 1), ('server3', 1)]
        self.servers = [
            {'name': name, 'weight': weight, 'current_weight': 0}
            for name, weight in servers
        ]

    def get_server(self):
        total_weight = sum(s['weight'] for s in self.servers)

        # Increase current_weight by weight
        for server in self.servers:
            server['current_weight'] += server['weight']

        # Select server with highest current_weight
        selected = max(self.servers, key=lambda s: s['current_weight'])

        # Decrease selected server's current_weight by total
        selected['current_weight'] -= total_weight

        return selected['name']

# Usage
lb = SmoothWeightedRoundRobin([('server1', 5), ('server2', 1), ('server3', 1)])
for i in range(7):
    print(f"Request {i+1}: {lb.get_server()}")
# Output: server1, server1, server2, server1, server3, server1, server1
```

**HAProxy Configuration:**

```haproxy
backend app_backend
    balance roundrobin
    server app1 10.0.1.10:8080 weight 5 check
    server app2 10.0.1.11:8080 weight 3 check
    server app3 10.0.1.12:8080 weight 2 check
```

**Use Cases:**
- Heterogeneous server pools (different CPU/memory)
- Gradual rollout (new version gets low weight)
- Cost optimization (cheaper servers get less traffic)
- A/B testing (version A: 90%, version B: 10%)

**Example: Canary Deployment**

```nginx
upstream backend {
    server stable-v1.example.com:8080 weight=9;    # 90% traffic
    server canary-v2.example.com:8080 weight=1;    # 10% traffic
}
```

### Least Connections

**How It Works:**
Routes new requests to the server with the fewest active connections.

```
Current State:
  Server A: 5 active connections
  Server B: 3 active connections  ← Selected
  Server C: 8 active connections

New request → Server B (least connections)

After routing:
  Server A: 5 active connections
  Server B: 4 active connections
  Server C: 8 active connections
```

**Characteristics:**
- Tracks active connections per server
- Adapts to varying request durations
- Better for long-lived connections
- Requires state tracking

**Implementation:**

```python
class LeastConnectionsLoadBalancer:
    def __init__(self, servers):
        self.servers = {server: 0 for server in servers}

    def get_server(self):
        # Select server with minimum connections
        server = min(self.servers, key=self.servers.get)
        self.servers[server] += 1
        return server

    def release_connection(self, server):
        if server in self.servers:
            self.servers[server] = max(0, self.servers[server] - 1)

# Usage
lb = LeastConnectionsLoadBalancer(['server1', 'server2', 'server3'])

# Simulate requests
s1 = lb.get_server()  # server1 (all have 0, pick first)
print(f"Request 1: {s1}, Connections: {lb.servers}")

s2 = lb.get_server()  # server2 (least connections)
print(f"Request 2: {s2}, Connections: {lb.servers}")

lb.release_connection(s1)  # server1 completes
print(f"After release: {lb.servers}")
```

**NGINX Configuration:**

```nginx
upstream backend {
    least_conn;
    server backend1.example.com;
    server backend2.example.com;
    server backend3.example.com;
}
```

**HAProxy Configuration:**

```haproxy
backend app_backend
    balance leastconn
    server app1 10.0.1.10:8080 check
    server app2 10.0.1.11:8080 check
    server app3 10.0.1.12:8080 check
```

**Pros:**
- ✓ Adapts to server load
- ✓ Handles varying request duration
- ✓ Better resource utilization
- ✓ Prevents overload

**Cons:**
- ✗ Requires state tracking
- ✗ More complex than round robin
- ✗ Doesn't account for request weight

**Best For:**
- WebSocket connections
- Long-polling requests
- Streaming applications
- Variable request processing times

### Weighted Least Connections

**How It Works:**
Combines least connections with server capacity weights.

```
Servers:
  Server A: 10 connections, weight = 2  → ratio = 10/2 = 5.0
  Server B: 6 connections, weight = 1   → ratio = 6/1 = 6.0
  Server C: 4 connections, weight = 3   → ratio = 4/3 = 1.33 ← Selected

New request → Server C (lowest connections-to-weight ratio)
```

**Implementation:**

```python
class WeightedLeastConnectionsLoadBalancer:
    def __init__(self, servers):
        # servers = [('server1', 5), ('server2', 3), ('server3', 2)]
        self.servers = {
            server: {'weight': weight, 'connections': 0}
            for server, weight in servers
        }

    def get_server(self):
        # Calculate connection-to-weight ratio
        server = min(
            self.servers.items(),
            key=lambda x: x[1]['connections'] / x[1]['weight']
        )[0]

        self.servers[server]['connections'] += 1
        return server

    def release_connection(self, server):
        if server in self.servers:
            self.servers[server]['connections'] = max(
                0, self.servers[server]['connections'] - 1
            )

# Usage
lb = WeightedLeastConnectionsLoadBalancer([
    ('server1', 5),  # High capacity
    ('server2', 3),  # Medium capacity
    ('server3', 2)   # Low capacity
])
```

**HAProxy Configuration:**

```haproxy
backend app_backend
    balance leastconn
    server app1 10.0.1.10:8080 weight 5 check
    server app2 10.0.1.11:8080 weight 3 check
    server app3 10.0.1.12:8080 weight 2 check
```

### IP Hash

**How It Works:**
Routes requests based on client IP address hash. Same client always goes to the same server (unless server becomes unavailable).

```
Client IP: 192.168.1.100
Hash: hash('192.168.1.100') = 12345
Server: 12345 % 3 = 0 → Server A

Client always routes to Server A (unless it fails)
```

**Implementation:**

```python
import hashlib

class IPHashLoadBalancer:
    def __init__(self, servers):
        self.servers = servers

    def get_server(self, client_ip):
        # Hash the client IP
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        # Select server based on hash
        index = hash_value % len(self.servers)
        return self.servers[index]

# Usage
lb = IPHashLoadBalancer(['server1', 'server2', 'server3'])

print(lb.get_server('192.168.1.100'))  # Always same server
print(lb.get_server('192.168.1.100'))  # Same as above
print(lb.get_server('192.168.1.101'))  # Different server
```

**NGINX Configuration:**

```nginx
upstream backend {
    ip_hash;
    server backend1.example.com;
    server backend2.example.com;
    server backend3.example.com;
}
```

**HAProxy Configuration:**

```haproxy
backend app_backend
    balance source
    hash-type consistent
    server app1 10.0.1.10:8080 check
    server app2 10.0.1.11:8080 check
    server app3 10.0.1.12:8080 check
```

**Pros:**
- ✓ Simple session persistence
- ✓ No state tracking needed
- ✓ Deterministic routing
- ✓ Works at L4 and L7

**Cons:**
- ✗ Uneven distribution (NAT, proxies)
- ✗ Server changes affect many clients
- ✗ Poor for dynamic server pools
- ✗ Doesn't adapt to load

**Best For:**
- Session-based applications
- Caching scenarios
- Stateful connections
- Simple persistence needs

**Problem: Adding/Removing Servers**

```
Original: 3 servers
  Client A → hash % 3 = 1 → Server B

After adding 4th server:
  Client A → hash % 4 = 2 → Server C (CHANGED!)

Result: Many clients re-mapped, cache invalidated
```

### Consistent Hashing

**How It Works:**
Uses a hash ring to minimize remapping when servers are added or removed.

```
Hash Ring (0-360):
  Server A: position 45
  Server B: position 150
  Server C: position 270

Client IP: 192.168.1.100
  Hash: 200
  Assigned to: Server C (next clockwise: 270)

Client IP: 192.168.1.101
  Hash: 50
  Assigned to: Server B (next clockwise: 150)

If Server B fails:
  Previous Server B clients → Server C
  Server A and C clients: UNCHANGED ✓
```

**Implementation:**

```python
import hashlib
import bisect

class ConsistentHashLoadBalancer:
    def __init__(self, servers, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []

        for server in servers:
            self.add_server(server)

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_server(self, server):
        # Add virtual nodes for better distribution
        for i in range(self.replicas):
            virtual_key = f"{server}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = server
            bisect.insort(self.sorted_keys, hash_value)

    def remove_server(self, server):
        for i in range(self.replicas):
            virtual_key = f"{server}:{i}"
            hash_value = self._hash(virtual_key)
            del self.ring[hash_value]
            self.sorted_keys.remove(hash_value)

    def get_server(self, client_key):
        if not self.ring:
            return None

        hash_value = self._hash(client_key)

        # Find the first server clockwise
        index = bisect.bisect_right(self.sorted_keys, hash_value)
        if index == len(self.sorted_keys):
            index = 0

        return self.ring[self.sorted_keys[index]]

# Usage
lb = ConsistentHashLoadBalancer(['server1', 'server2', 'server3'], replicas=150)

# Test distribution
from collections import Counter
distribution = Counter()
for i in range(1000):
    server = lb.get_server(f'client_{i}')
    distribution[server] += 1

print("Distribution:", distribution)
# Output: Distribution: Counter({'server2': 339, 'server1': 334, 'server3': 327})

# Add a server - minimal remapping
lb.add_server('server4')
```

**Virtual Nodes Visualization:**

```
Hash Ring with Virtual Nodes (replicas=3):

   0° ─────────────────────────────────── 360°
   ↓                                       ↓
[S1:0] [S2:0] [S3:0] [S1:1] [S2:1] [S3:1] [S1:2] [S2:2] [S3:2]
  45°   80°   120°   180°   210°   240°   290°   320°   350°

Better distribution with more replicas (150+)
```

**NGINX with Consistent Hashing (Plus/Commercial):**

```nginx
upstream backend {
    hash $request_uri consistent;
    server backend1.example.com;
    server backend2.example.com;
    server backend3.example.com;
}
```

**Pros:**
- ✓ Minimal remapping on changes
- ✓ Better cache hit rates
- ✓ Scalable for large pools
- ✓ Even distribution with virtual nodes

**Cons:**
- ✗ More complex implementation
- ✗ Requires careful tuning
- ✗ Higher memory overhead

**Best For:**
- Distributed caching (Memcached, Redis)
- CDN edge selection
- Database sharding
- Large dynamic server pools

**Comparison: IP Hash vs Consistent Hashing**

```
Scenario: 3 servers → 4 servers

IP Hash:
  Remapped clients: ~75% (3/4 of all clients)

Consistent Hashing (150 replicas):
  Remapped clients: ~25% (only 1/4 of clients)
```

### Least Response Time

**How It Works:**
Routes requests to the server with the lowest average response time and fewest active connections.

```
Current Metrics:
  Server A: 50ms avg, 5 connections  → score = 50 * 5 = 250
  Server B: 30ms avg, 8 connections  → score = 30 * 8 = 240 ← Selected
  Server C: 40ms avg, 10 connections → score = 40 * 10 = 400

New request → Server B (lowest score)
```

**Implementation:**

```python
import time
from collections import deque

class LeastResponseTimeLoadBalancer:
    def __init__(self, servers, window_size=100):
        self.servers = {
            server: {
                'response_times': deque(maxlen=window_size),
                'connections': 0
            }
            for server in servers
        }

    def get_server(self):
        def calculate_score(server_data):
            avg_time = (
                sum(server_data['response_times']) / len(server_data['response_times'])
                if server_data['response_times']
                else 0
            )
            connections = server_data['connections']
            return avg_time * (connections + 1)  # +1 to avoid zero

        server = min(self.servers.items(), key=lambda x: calculate_score(x[1]))[0]
        self.servers[server]['connections'] += 1
        return server

    def record_response_time(self, server, response_time):
        if server in self.servers:
            self.servers[server]['response_times'].append(response_time)

    def release_connection(self, server):
        if server in self.servers:
            self.servers[server]['connections'] = max(
                0, self.servers[server]['connections'] - 1
            )

# Usage
lb = LeastResponseTimeLoadBalancer(['server1', 'server2', 'server3'])

# Simulate request handling
start = time.time()
server = lb.get_server()
# ... process request ...
response_time = time.time() - start
lb.record_response_time(server, response_time)
lb.release_connection(server)
```

**NGINX Plus Configuration:**

```nginx
upstream backend {
    least_time header;  # or 'last_byte' for full response time
    server backend1.example.com;
    server backend2.example.com;
    server backend3.example.com;
}
```

**Pros:**
- ✓ Optimal user experience
- ✓ Adapts to server performance
- ✓ Considers both load and speed
- ✓ Self-optimizing

**Cons:**
- ✗ Complex to implement
- ✗ Requires response time tracking
- ✗ Higher overhead
- ✗ May need tuning

**Best For:**
- Performance-critical applications
- Heterogeneous server pools
- Variable network conditions
- SLA-driven systems

### Random Selection

**How It Works:**
Randomly selects a server from the pool.

```python
import random

class RandomLoadBalancer:
    def __init__(self, servers):
        self.servers = servers

    def get_server(self):
        return random.choice(self.servers)

# Weighted random
class WeightedRandomLoadBalancer:
    def __init__(self, servers):
        # servers = [('server1', 5), ('server2', 3), ('server3', 2)]
        self.servers = []
        self.weights = []
        for server, weight in servers:
            self.servers.append(server)
            self.weights.append(weight)

    def get_server(self):
        return random.choices(self.servers, weights=self.weights, k=1)[0]

# Usage
lb = WeightedRandomLoadBalancer([
    ('server1', 5),
    ('server2', 3),
    ('server3', 2)
])
```

**Pros:**
- ✓ Simple implementation
- ✓ No state required
- ✓ Good distribution over time

**Cons:**
- ✗ Short-term imbalance
- ✗ No optimization
- ✗ Unpredictable

**Best For:**
- Simple setups
- Stateless applications
- Testing environments

### Resource-Based

**How It Works:**
Routes based on real-time server resource metrics (CPU, memory, disk I/O).

```python
class ResourceBasedLoadBalancer:
    def __init__(self, servers):
        self.servers = servers

    def get_server_metrics(self, server):
        # In real implementation, query server metrics
        # via monitoring system (Prometheus, CloudWatch, etc.)
        return {
            'cpu_usage': 45.2,      # percentage
            'memory_usage': 60.1,   # percentage
            'connections': 120,
            'disk_io': 30.5         # percentage
        }

    def calculate_load_score(self, metrics):
        # Lower score = better
        return (
            metrics['cpu_usage'] * 0.4 +
            metrics['memory_usage'] * 0.3 +
            metrics['disk_io'] * 0.2 +
            (metrics['connections'] / 1000) * 0.1
        )

    def get_server(self):
        server_scores = {}
        for server in self.servers:
            metrics = self.get_server_metrics(server)
            server_scores[server] = self.calculate_load_score(metrics)

        return min(server_scores, key=server_scores.get)
```

**Best For:**
- Cloud auto-scaling
- Heterogeneous environments
- Resource-intensive applications

---

## Health Checks and Monitoring

Health checks ensure traffic is only sent to healthy servers. A robust health checking system is critical for high availability.

### Types of Health Checks

**1. Active Health Checks**
Load balancer actively probes servers at regular intervals.

```
Load Balancer sends probes every 5 seconds:
  ↓
Server responds with health status
  ↓
LB marks server as healthy or unhealthy
```

**2. Passive Health Checks**
Load balancer monitors actual traffic and marks servers unhealthy based on errors.

```
Client request → Server
  ↓
Server returns 500 error
  ↓
LB increments error count
  ↓
If errors > threshold: mark unhealthy
```

### Health Check Methods

**1. TCP Connection Check**
```bash
# Simple TCP connection
nc -zv server1.example.com 8080
```

Most basic check - verifies port is open.

**2. HTTP/HTTPS Check**
```bash
# HTTP GET request
curl -f http://server1.example.com/health
```

Verifies application is responding.

**3. Custom Health Endpoint**
```python
# Flask example
from flask import Flask, jsonify
import psutil

app = Flask(__name__)

@app.route('/health')
def health_check():
    # Check database connection
    db_healthy = check_database_connection()

    # Check CPU usage
    cpu_usage = psutil.cpu_percent()

    # Check memory
    memory = psutil.virtual_memory()

    if db_healthy and cpu_usage < 90 and memory.percent < 90:
        return jsonify({
            'status': 'healthy',
            'cpu': cpu_usage,
            'memory': memory.percent,
            'database': 'connected'
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'cpu': cpu_usage,
            'memory': memory.percent,
            'database': 'connected' if db_healthy else 'disconnected'
        }), 503

def check_database_connection():
    try:
        # Check database
        db.execute('SELECT 1')
        return True
    except:
        return False
```

**4. Deep Health Check**
```python
@app.route('/health/deep')
def deep_health_check():
    checks = {
        'database': check_database(),
        'cache': check_cache(),
        'message_queue': check_message_queue(),
        'external_api': check_external_api(),
        'disk_space': check_disk_space(),
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return jsonify({
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks
    }), status_code
```

### Health Check Configuration

**NGINX:**

```nginx
upstream backend {
    server backend1.example.com:8080;
    server backend2.example.com:8080;
    server backend3.example.com:8080;
}

server {
    listen 80;

    location / {
        proxy_pass http://backend;

        # Passive health checks
        proxy_next_upstream error timeout http_500 http_502 http_503;
        proxy_connect_timeout 2s;
        proxy_read_timeout 5s;
    }
}

# NGINX Plus - Active health checks
upstream backend {
    zone backend 64k;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}

server {
    location / {
        proxy_pass http://backend;
        health_check interval=5s
                     fails=3
                     passes=2
                     uri=/health
                     match=server_ok;
    }
}

match server_ok {
    status 200;
    header Content-Type = application/json;
    body ~ "healthy";
}
```

**HAProxy:**

```haproxy
backend app_backend
    balance roundrobin

    # Health check options
    option httpchk GET /health
    http-check expect status 200

    # Server definitions with checks
    server app1 10.0.1.10:8080 check inter 5s fall 3 rise 2
    server app2 10.0.1.11:8080 check inter 5s fall 3 rise 2
    server app3 10.0.1.12:8080 check inter 5s fall 3 rise 2

    # Backup server (only used when all others fail)
    server app_backup 10.0.1.99:8080 check backup

# Advanced health check
backend api_backend
    option httpchk GET /health
    http-check expect status 200
    http-check expect string "healthy"

    # Custom headers
    http-check send-state

    server api1 10.0.2.10:8080 check
```

**Parameters:**
- `interval` (inter): Time between checks (default: 2s)
- `fails` (fall): Failed checks before marking unhealthy (default: 3)
- `passes` (rise): Successful checks before marking healthy (default: 2)
- `timeout`: Health check timeout (default: same as connect timeout)

### Health Check Best Practices

**1. Appropriate Intervals**
```
Too frequent (< 1s):  Unnecessary load
Good (2-5s):          Quick detection, low overhead
Too slow (> 30s):     Slow failure detection
```

**2. Layered Health Checks**
```
Frontend LB:  Simple TCP check (fast)
      ↓
Application:  HTTP /health endpoint
      ↓
Deep Check:   Database, dependencies (periodic)
```

**3. Circuit Breaker Pattern**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

**4. Gradual Restoration**
```
Server marked unhealthy:
  Wait 10s → First health check
  ↓ Pass
  Wait 10s → Second health check
  ↓ Pass
  Wait 10s → Third health check
  ↓ Pass
  Mark healthy, restore traffic gradually (10% → 50% → 100%)
```

### Monitoring Metrics

**Key Metrics to Track:**

```python
# Server-level metrics
metrics = {
    'health_status': 'healthy|unhealthy|unknown',
    'response_time_avg': 45.2,        # milliseconds
    'response_time_p95': 120.5,       # 95th percentile
    'response_time_p99': 250.8,       # 99th percentile
    'request_rate': 1250,             # requests/second
    'error_rate': 0.02,               # percentage
    'active_connections': 450,
    'total_connections': 125000,
    'bytes_in': 1024000000,          # bytes
    'bytes_out': 5120000000,         # bytes
    'cpu_usage': 65.5,               # percentage
    'memory_usage': 72.3,            # percentage
}

# Load balancer metrics
lb_metrics = {
    'total_requests': 500000,
    'requests_per_server': {
        'server1': 180000,
        'server2': 170000,
        'server3': 150000
    },
    'failed_requests': 100,
    'backend_response_time': 45.2,
    'lb_processing_time': 2.1,
    'active_backends': 3,
    'total_backends': 3,
}
```

**Prometheus Metrics Example:**

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter(
    'lb_requests_total',
    'Total requests',
    ['backend', 'method', 'status']
)

request_duration = Histogram(
    'lb_request_duration_seconds',
    'Request duration',
    ['backend']
)

# Backend health
backend_health = Gauge(
    'lb_backend_health',
    'Backend health status (1=healthy, 0=unhealthy)',
    ['backend']
)

active_connections = Gauge(
    'lb_active_connections',
    'Active connections',
    ['backend']
)

# Usage
request_count.labels(backend='server1', method='GET', status='200').inc()
request_duration.labels(backend='server1').observe(0.045)
backend_health.labels(backend='server1').set(1)
active_connections.labels(backend='server1').set(450)
```

### Failover Strategies

**1. Immediate Failover**
```
Server fails → Immediately remove from pool
Fast but may cause false positives
```

**2. Graceful Degradation**
```
Server slow → Reduce traffic gradually
Server error rate high → Remove from pool
```

**3. Active-Passive Failover**
```
Active Server (handles traffic)
   ↓ fails
Passive Server activated
```

**4. Active-Active Failover**
```
Server A (50% traffic) ← fails
    ↓
Server B (100% traffic)
Server C (added to pool)
```

**HAProxy Failover Configuration:**

```haproxy
backend app_backend
    balance roundrobin

    # Primary servers
    server app1 10.0.1.10:8080 check
    server app2 10.0.1.11:8080 check

    # Backup servers (only used when primaries fail)
    server backup1 10.0.2.10:8080 check backup
    server backup2 10.0.2.11:8080 check backup

    # Error recovery
    retries 3
    retry-on all-retryable-errors

    # Timeouts
    timeout connect 5s
    timeout server 30s
```

---

## Session Persistence

Session persistence (also called sticky sessions or session affinity) ensures that requests from the same client are routed to the same backend server.

### Why Session Persistence?

**Without Persistence:**
```
User login → Server A (session created)
User request → Server B (no session, user logged out!)
```

**With Persistence:**
```
User login → Server A (session created)
User request → Server A (session available ✓)
```

### Session Persistence Methods

**1. Cookie-Based Persistence**

Load balancer inserts a cookie to track which server handled the request.

```
Initial Request:
  Client → LB → Server A

Response:
  Server A → LB → Client
  Set-Cookie: SERVERID=server_a; Path=/

Subsequent Requests:
  Client → LB (reads cookie) → Server A
```

**NGINX Configuration:**

```nginx
upstream backend {
    server backend1.example.com:8080;
    server backend2.example.com:8080;
    server backend3.example.com:8080;

    # Cookie-based sticky sessions
    sticky cookie srv_id expires=1h domain=.example.com path=/;
}
```

**HAProxy Configuration:**

```haproxy
backend app_backend
    balance roundrobin

    # Insert cookie
    cookie SERVERID insert indirect nocache

    server app1 10.0.1.10:8080 check cookie app1
    server app2 10.0.1.11:8080 check cookie app2
    server app3 10.0.1.12:8080 check cookie app3
```

**2. Application Cookie Tracking**

Track existing application cookies (e.g., session ID).

```nginx
upstream backend {
    server backend1.example.com:8080;
    server backend2.example.com:8080;

    # Use existing session cookie
    sticky learn
        create=$upstream_cookie_PHPSESSID
        lookup=$cookie_PHPSESSID
        zone=client_sessions:1m;
}
```

**3. IP-Based Persistence (Source IP)**

Route based on client IP address.

```nginx
upstream backend {
    ip_hash;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}
```

```haproxy
backend app_backend
    balance source
    server app1 10.0.1.10:8080 check
    server app2 10.0.1.11:8080 check
```

**Problems with IP-based persistence:**
- Clients behind NAT share IP
- Mobile clients change IP
- Proxy servers aggregate many clients

**4. URL Parameter Persistence**

Route based on URL parameter.

```nginx
upstream backend {
    hash $arg_userid;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}

# Example URLs:
# /api/user?userid=123 → Always routes to same server
# /api/user?userid=456 → Routes to different server
```

**5. HTTP Header Persistence**

Route based on custom HTTP header.

```nginx
upstream backend {
    hash $http_x_user_id consistent;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}
```

### Session Persistence Duration

```haproxy
backend app_backend
    # Stick for 30 minutes of inactivity
    stick-table type string len 32 size 100k expire 30m
    stick on cookie(JSESSIONID)

    server app1 10.0.1.10:8080 check
    server app2 10.0.1.11:8080 check
```

### Alternatives to Sticky Sessions

Sticky sessions can cause uneven load distribution. Better alternatives:

**1. Centralized Session Store**

```
         Load Balancer
         /     |     \
    Server1 Server2 Server3
         \     |     /
        Redis Session Store

All servers share session data
```

```python
# Flask with Redis sessions
from flask import Flask, session
from flask_session import Session
import redis

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')
Session(app)

@app.route('/login')
def login():
    session['user_id'] = 123
    return "Logged in"

@app.route('/profile')
def profile():
    user_id = session.get('user_id')  # Available on any server
    return f"User {user_id}"
```

**2. JWT Tokens (Stateless)**

```python
# No server-side session needed
import jwt
from datetime import datetime, timedelta

def create_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, 'secret_key', algorithm='HS256')

@app.route('/login')
def login():
    token = create_token(123)
    return {'token': token}

@app.route('/profile')
def profile():
    token = request.headers.get('Authorization')
    payload = jwt.decode(token, 'secret_key', algorithms=['HS256'])
    user_id = payload['user_id']
    return f"User {user_id}"
```

**3. Client-Side Sessions**

```javascript
// Store session data in encrypted cookie
// No server-side storage needed
// Works with any backend server
```

### Best Practices

**1. Avoid sticky sessions when possible**
- Use stateless authentication (JWT)
- Use centralized session storage (Redis, Memcached)

**2. If you must use sticky sessions:**
- Use cookie-based (more reliable than IP)
- Set reasonable expiration
- Handle server failures gracefully

**3. Monitor session distribution:**
```python
# Check if sessions are balanced
session_distribution = {
    'server1': 1000,
    'server2': 950,
    'server3': 1050
}
# Good: relatively even distribution
```

---

## SSL/TLS Termination

SSL/TLS termination is the process of decrypting HTTPS traffic at the load balancer, then forwarding it to backend servers.

### Termination Options

**1. SSL Termination at Load Balancer**

```
Client (HTTPS) → Load Balancer (decrypt) → Backend (HTTP)
```

**Pros:**
- ✓ Reduced backend CPU load
- ✓ Centralized certificate management
- ✓ Content inspection possible
- ✓ Easier caching

**Cons:**
- ✗ Unencrypted internal traffic
- ✗ Compliance concerns

**2. SSL Passthrough**

```
Client (HTTPS) → Load Balancer (forward) → Backend (HTTPS)
```

**Pros:**
- ✓ End-to-end encryption
- ✓ Better compliance
- ✓ Backend controls certificates

**Cons:**
- ✗ Higher backend CPU usage
- ✗ No L7 routing
- ✗ No content inspection

**3. SSL Re-encryption**

```
Client (HTTPS) → LB (decrypt/encrypt) → Backend (HTTPS)
```

**Pros:**
- ✓ L7 routing available
- ✓ Encrypted internal traffic
- ✓ Content inspection

**Cons:**
- ✗ Highest CPU usage
- ✗ Complex configuration
- ✗ Certificate management overhead

### SSL Termination Configuration

**NGINX:**

```nginx
upstream backend {
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}

server {
    listen 443 ssl http2;
    server_name www.example.com;

    # SSL certificate
    ssl_certificate /etc/nginx/ssl/example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers off;

    # SSL session cache
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/nginx/ssl/chain.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name www.example.com;
    return 301 https://$server_name$request_uri;
}
```

**HAProxy:**

```haproxy
frontend https_frontend
    bind *:443 ssl crt /etc/haproxy/certs/example.com.pem

    # Security
    http-response set-header Strict-Transport-Security "max-age=63072000"

    # Route to backend
    default_backend app_backend

frontend http_frontend
    bind *:80
    # Redirect to HTTPS
    redirect scheme https code 301 if !{ ssl_fc }

backend app_backend
    balance roundrobin

    # Forward to HTTP backends
    server app1 10.0.1.10:8080 check
    server app2 10.0.1.11:8080 check
```

**SSL Re-encryption (NGINX):**

```nginx
upstream backend_ssl {
    server backend1.example.com:8443;
    server backend2.example.com:8443;
}

server {
    listen 443 ssl;
    server_name www.example.com;

    ssl_certificate /etc/nginx/ssl/frontend.crt;
    ssl_certificate_key /etc/nginx/ssl/frontend.key;

    location / {
        # Re-encrypt to backend
        proxy_pass https://backend_ssl;
        proxy_ssl_verify on;
        proxy_ssl_trusted_certificate /etc/nginx/ssl/backend-ca.crt;
        proxy_ssl_protocols TLSv1.2 TLSv1.3;
    }
}
```

### Certificate Management

**1. Let's Encrypt (Free Certificates)**

```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d www.example.com -d example.com

# Auto-renewal
certbot renew --dry-run

# Cron job for renewal
0 0 * * * /usr/bin/certbot renew --quiet
```

**2. Wildcard Certificates**

```bash
# Single certificate for *.example.com
certbot certonly --manual --preferred-challenges dns -d *.example.com
```

**3. Certificate Monitoring**

```bash
# Check certificate expiration
openssl x509 -in /etc/nginx/ssl/example.com.crt -noout -enddate

# Monitor with script
#!/bin/bash
CERT="/etc/nginx/ssl/example.com.crt"
EXPIRE_DATE=$(openssl x509 -in $CERT -noout -enddate | cut -d= -f2)
EXPIRE_EPOCH=$(date -d "$EXPIRE_DATE" +%s)
NOW_EPOCH=$(date +%s)
DAYS_REMAINING=$(( ($EXPIRE_EPOCH - $NOW_EPOCH) / 86400 ))

if [ $DAYS_REMAINING -lt 30 ]; then
    echo "WARNING: Certificate expires in $DAYS_REMAINING days"
fi
```

### Performance Optimization

**1. SSL Session Resumption**

```nginx
# Reduces SSL handshake overhead
ssl_session_cache shared:SSL:50m;
ssl_session_timeout 1d;
ssl_session_tickets off;
```

**2. OCSP Stapling**

```nginx
# Load balancer fetches OCSP response
# Reduces client latency
ssl_stapling on;
ssl_stapling_verify on;
```

**3. HTTP/2**

```nginx
listen 443 ssl http2;
# Multiplexing, header compression, server push
```

**Performance Impact:**

```
Operation                  CPU Cost
-------------------------- ----------
No SSL                     Baseline
SSL Termination            +15-25%
SSL Passthrough            Minimal
SSL Re-encryption          +30-40%
```

---

## Cloud Load Balancers

### AWS Load Balancers

AWS offers three types of load balancers, each optimized for different use cases.

**1. Application Load Balancer (ALB) - Layer 7**

**Features:**
- HTTP/HTTPS traffic
- Path-based routing
- Host-based routing
- WebSocket and HTTP/2 support
- Native WAF integration
- Fixed hostname (xxx.region.elb.amazonaws.com)

**Use Cases:**
- Web applications
- Microservices
- Container-based applications

**Routing Rules:**

```yaml
# Path-based routing
/api/* → API Target Group
/images/* → Image Server Target Group
/* → Default Web Server Target Group

# Host-based routing
api.example.com → API Target Group
www.example.com → Web Target Group

# Header-based routing
X-Client-Type: mobile → Mobile Target Group
X-Client-Type: desktop → Desktop Target Group

# Query string routing
?version=beta → Beta Target Group
?version=stable → Stable Target Group
```

**Terraform Configuration:**

```hcl
resource "aws_lb" "app_lb" {
  name               = "app-load-balancer"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb_sg.id]
  subnets            = aws_subnet.public.*.id

  enable_deletion_protection = true
  enable_http2              = true
  enable_cross_zone_load_balancing = true

  tags = {
    Environment = "production"
  }
}

resource "aws_lb_target_group" "app_tg" {
  name     = "app-target-group"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = true
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.app_lb.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.cert.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app_tg.arn
  }
}

# Path-based routing rule
resource "aws_lb_listener_rule" "api_routing" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api_tg.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}

# Header-based routing
resource "aws_lb_listener_rule" "mobile_routing" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 200

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mobile_tg.arn
  }

  condition {
    http_header {
      http_header_name = "User-Agent"
      values           = ["*Mobile*", "*Android*", "*iPhone*"]
    }
  }
}
```

**2. Network Load Balancer (NLB) - Layer 4**

**Features:**
- Ultra-high performance (millions of requests/second)
- Static IP addresses
- Elastic IP support
- TCP, UDP, TLS traffic
- Low latency (microseconds)
- Preserve source IP
- PrivateLink support

**Use Cases:**
- Extreme performance requirements
- Non-HTTP protocols
- Static IP requirements
- Volatile traffic patterns

**Terraform Configuration:**

```hcl
resource "aws_lb" "network_lb" {
  name               = "network-load-balancer"
  internal           = false
  load_balancer_type = "network"
  subnets            = aws_subnet.public.*.id

  enable_deletion_protection       = true
  enable_cross_zone_load_balancing = true

  tags = {
    Environment = "production"
  }
}

resource "aws_lb_target_group" "tcp_tg" {
  name        = "tcp-target-group"
  port        = 3306
  protocol    = "TCP"
  vpc_id      = aws_vpc.main.id
  target_type = "instance"

  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 10
    port                = 3306
    protocol            = "TCP"
    unhealthy_threshold = 3
  }

  stickiness {
    enabled = true
    type    = "source_ip"
  }
}

resource "aws_lb_listener" "tcp" {
  load_balancer_arn = aws_lb.network_lb.arn
  port              = "3306"
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.tcp_tg.arn
  }
}

# Associate Elastic IP
resource "aws_eip" "lb_eip" {
  count = 2
  vpc   = true

  tags = {
    Name = "nlb-eip-${count.index + 1}"
  }
}
```

**3. Classic Load Balancer (CLB) - Legacy**

**Features:**
- Layer 4 and Layer 7
- Legacy, not recommended for new applications
- Being phased out

**Migration to ALB/NLB recommended.**

**AWS Load Balancer Comparison:**

| Feature | ALB | NLB | CLB |
|---------|-----|-----|-----|
| **OSI Layer** | Layer 7 | Layer 4 | Layer 4 & 7 |
| **Protocol** | HTTP, HTTPS, gRPC | TCP, UDP, TLS | HTTP, HTTPS, TCP, SSL |
| **Performance** | Good | Excellent | Moderate |
| **Latency** | ~ms | ~μs | ~ms |
| **Static IP** | No | Yes | No |
| **Path-based Routing** | Yes | No | No |
| **Host-based Routing** | Yes | No | No |
| **WebSocket** | Yes | Yes | Yes |
| **Target Types** | Instance, IP, Lambda | Instance, IP | Instance |
| **Pricing** | Moderate | Higher | Lower |
| **Use Case** | Web apps | High perf | Legacy |

### Google Cloud Load Balancing

Google Cloud offers a unified load balancing solution with different types.

**1. Global HTTP(S) Load Balancer**

**Features:**
- Global anycast IP
- Cross-region load balancing
- URL map-based routing
- Cloud CDN integration
- Cloud Armor (DDoS protection)
- SSL certificates managed by Google

**Architecture:**

```
User (Asia) → Anycast IP → Asia Backend
User (US) → Anycast IP → US Backend
User (EU) → Anycast IP → EU Backend

Single IP, global distribution
```

**Terraform Configuration:**

```hcl
# Backend service
resource "google_compute_backend_service" "web_backend" {
  name                  = "web-backend-service"
  protocol              = "HTTP"
  port_name             = "http"
  timeout_sec           = 30
  enable_cdn            = true
  health_checks         = [google_compute_health_check.http_health.id]
  load_balancing_scheme = "EXTERNAL"

  backend {
    group           = google_compute_instance_group.web_ig_us.id
    balancing_mode  = "UTILIZATION"
    capacity_scaler = 1.0
  }

  backend {
    group           = google_compute_instance_group.web_ig_eu.id
    balancing_mode  = "UTILIZATION"
    capacity_scaler = 1.0
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# URL map
resource "google_compute_url_map" "web_url_map" {
  name            = "web-url-map"
  default_service = google_compute_backend_service.web_backend.id

  host_rule {
    hosts        = ["api.example.com"]
    path_matcher = "api"
  }

  path_matcher {
    name            = "api"
    default_service = google_compute_backend_service.api_backend.id

    path_rule {
      paths   = ["/v1/*"]
      service = google_compute_backend_service.api_v1_backend.id
    }

    path_rule {
      paths   = ["/v2/*"]
      service = google_compute_backend_service.api_v2_backend.id
    }
  }
}

# HTTPS proxy
resource "google_compute_target_https_proxy" "web_https_proxy" {
  name             = "web-https-proxy"
  url_map          = google_compute_url_map.web_url_map.id
  ssl_certificates = [google_compute_ssl_certificate.web_cert.id]
}

# Forwarding rule (global IP)
resource "google_compute_global_forwarding_rule" "web_https" {
  name       = "web-https-forwarding-rule"
  target     = google_compute_target_https_proxy.web_https_proxy.id
  port_range = "443"
  ip_address = google_compute_global_address.web_ip.address
}

# Health check
resource "google_compute_health_check" "http_health" {
  name               = "http-health-check"
  check_interval_sec = 5
  timeout_sec        = 5

  http_health_check {
    port         = 80
    request_path = "/health"
  }
}
```

**2. Regional Load Balancers**

```hcl
# Internal TCP/UDP Load Balancer
resource "google_compute_region_backend_service" "internal_tcp" {
  name                  = "internal-tcp-backend"
  region                = "us-central1"
  protocol              = "TCP"
  load_balancing_scheme = "INTERNAL"
  health_checks         = [google_compute_health_check.tcp_health.id]

  backend {
    group = google_compute_instance_group.app_ig.id
  }
}

# Network Load Balancer (External)
resource "google_compute_region_backend_service" "network_lb" {
  name                  = "network-lb-backend"
  region                = "us-central1"
  protocol              = "TCP"
  load_balancing_scheme = "EXTERNAL"

  backend {
    group = google_compute_instance_group.app_ig.id
  }
}
```

**GCP Load Balancer Types:**

| Type | Scope | Layer | Use Case |
|------|-------|-------|----------|
| **Global HTTP(S)** | Global | L7 | Web apps, APIs |
| **Global SSL Proxy** | Global | L4 (SSL) | Non-HTTP SSL |
| **Global TCP Proxy** | Global | L4 (TCP) | Non-HTTP TCP |
| **Regional Network** | Regional | L4 | High perf TCP/UDP |
| **Regional Internal** | Regional | L4 | Internal services |

### Azure Load Balancers

**1. Azure Load Balancer (Layer 4)**

**Features:**
- Layer 4 (TCP, UDP)
- High performance
- Availability zones support
- Outbound connectivity

**Types:**
- **Public**: Internet-facing
- **Internal**: Private networks

**Azure CLI:**

```bash
# Create load balancer
az network lb create \
  --resource-group myResourceGroup \
  --name myLoadBalancer \
  --sku Standard \
  --public-ip-address myPublicIP \
  --frontend-ip-name myFrontEnd \
  --backend-pool-name myBackEndPool

# Create health probe
az network lb probe create \
  --resource-group myResourceGroup \
  --lb-name myLoadBalancer \
  --name myHealthProbe \
  --protocol tcp \
  --port 80 \
  --interval 5 \
  --threshold 2

# Create LB rule
az network lb rule create \
  --resource-group myResourceGroup \
  --lb-name myLoadBalancer \
  --name myHTTPRule \
  --protocol tcp \
  --frontend-port 80 \
  --backend-port 80 \
  --frontend-ip-name myFrontEnd \
  --backend-pool-name myBackEndPool \
  --probe-name myHealthProbe
```

**2. Azure Application Gateway (Layer 7)**

**Features:**
- Layer 7 load balancing
- URL-based routing
- SSL termination
- Web Application Firewall (WAF)
- Auto-scaling
- Session affinity

**Terraform:**

```hcl
resource "azurerm_application_gateway" "app_gw" {
  name                = "app-gateway"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku {
    name     = "Standard_v2"
    tier     = "Standard_v2"
    capacity = 2
  }

  gateway_ip_configuration {
    name      = "gateway-ip-config"
    subnet_id = azurerm_subnet.frontend.id
  }

  frontend_port {
    name = "https-port"
    port = 443
  }

  frontend_ip_configuration {
    name                 = "frontend-ip-config"
    public_ip_address_id = azurerm_public_ip.app_gw_pip.id
  }

  backend_address_pool {
    name = "backend-pool"
    ip_addresses = ["10.0.1.10", "10.0.1.11", "10.0.1.12"]
  }

  backend_http_settings {
    name                  = "http-settings"
    cookie_based_affinity = "Enabled"
    port                  = 80
    protocol              = "Http"
    request_timeout       = 20
    probe_name            = "health-probe"
  }

  http_listener {
    name                           = "https-listener"
    frontend_ip_configuration_name = "frontend-ip-config"
    frontend_port_name             = "https-port"
    protocol                       = "Https"
    ssl_certificate_name           = "app-cert"
  }

  request_routing_rule {
    name                       = "routing-rule"
    rule_type                  = "Basic"
    http_listener_name         = "https-listener"
    backend_address_pool_name  = "backend-pool"
    backend_http_settings_name = "http-settings"
  }

  probe {
    name                = "health-probe"
    protocol            = "Http"
    path                = "/health"
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3
    host                = "127.0.0.1"
  }
}
```

**3. Azure Front Door (Global)**

**Features:**
- Global HTTP(S) load balancing
- CDN capabilities
- URL-based routing
- WAF integration
- SSL offloading

```hcl
resource "azurerm_frontdoor" "main" {
  name                = "my-front-door"
  resource_group_name = azurerm_resource_group.main.name

  routing_rule {
    name               = "routing-rule"
    accepted_protocols = ["Https"]
    patterns_to_match  = ["/*"]
    frontend_endpoints = ["frontend-endpoint"]

    forwarding_configuration {
      forwarding_protocol = "HttpsOnly"
      backend_pool_name   = "backend-pool"
    }
  }

  backend_pool_load_balancing {
    name = "load-balancing-settings"
  }

  backend_pool_health_probe {
    name = "health-probe"
    path = "/health"
  }

  backend_pool {
    name = "backend-pool"
    backend {
      host_header = "www.example.com"
      address     = "backend1.example.com"
      http_port   = 80
      https_port  = 443
    }
  }

  frontend_endpoint {
    name      = "frontend-endpoint"
    host_name = "my-front-door.azurefd.net"
  }
}
```

---

## Software Load Balancers

### NGINX

NGINX is one of the most popular open-source load balancers and web servers.

**Basic Load Balancing:**

```nginx
http {
    upstream backend {
        server backend1.example.com:8080;
        server backend2.example.com:8080;
        server backend3.example.com:8080;
    }

    server {
        listen 80;
        server_name www.example.com;

        location / {
            proxy_pass http://backend;
        }
    }
}
```

**Advanced Configuration:**

```nginx
http {
    # Connection pooling
    upstream backend {
        least_conn;  # Load balancing algorithm

        server backend1.example.com:8080 weight=3 max_fails=3 fail_timeout=30s;
        server backend2.example.com:8080 weight=2 max_fails=3 fail_timeout=30s;
        server backend3.example.com:8080 weight=1 max_fails=3 fail_timeout=30s backup;

        # Connection pool
        keepalive 32;
        keepalive_requests 100;
        keepalive_timeout 60s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;
    limit_conn_zone $binary_remote_addr zone=addr:10m;

    server {
        listen 80;
        server_name www.example.com;

        # Apply rate limits
        limit_req zone=one burst=20 nodelay;
        limit_conn addr 10;

        location / {
            proxy_pass http://backend;

            # Headers
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 10s;

            # Buffering
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;

            # Error handling
            proxy_next_upstream error timeout http_500 http_502 http_503;
            proxy_next_upstream_tries 3;
            proxy_next_upstream_timeout 10s;

            # HTTP version
            proxy_http_version 1.1;
            proxy_set_header Connection "";
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

**Dynamic Upstream with NGINX Plus:**

```nginx
upstream backend {
    zone backend 64k;
    server backend1.example.com:8080;
    server backend2.example.com:8080;
}

server {
    location / {
        proxy_pass http://backend;
        health_check interval=5s fails=3 passes=2;
    }

    # API for dynamic configuration
    location /api {
        api write=on;
        allow 10.0.0.0/8;
        deny all;
    }
}
```

**Caching:**

```nginx
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m max_size=10g
                 inactive=60m use_temp_path=off;

server {
    location / {
        proxy_cache my_cache;
        proxy_cache_key "$scheme$request_method$host$request_uri";
        proxy_cache_valid 200 60m;
        proxy_cache_valid 404 10m;
        proxy_cache_bypass $http_cache_control;
        add_header X-Cache-Status $upstream_cache_status;

        proxy_pass http://backend;
    }
}
```

### HAProxy

HAProxy is a high-performance TCP/HTTP load balancer.

**Basic Configuration:**

```haproxy
global
    log /dev/log local0
    log /dev/log local1 notice
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

    # Security
    ssl-default-bind-ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256
    ssl-default-bind-options ssl-min-ver TLSv1.2 no-tls-tickets

defaults
    log     global
    mode    http
    option  httplog
    option  dontlognull
    timeout connect 5000
    timeout client  50000
    timeout server  50000
    errorfile 400 /etc/haproxy/errors/400.http
    errorfile 403 /etc/haproxy/errors/403.http
    errorfile 408 /etc/haproxy/errors/408.http
    errorfile 500 /etc/haproxy/errors/500.http
    errorfile 502 /etc/haproxy/errors/502.http
    errorfile 503 /etc/haproxy/errors/503.http
    errorfile 504 /etc/haproxy/errors/504.http

frontend http_front
    bind *:80
    stats uri /haproxy?stats
    default_backend http_back

backend http_back
    balance roundrobin
    server server1 10.0.1.10:8080 check
    server server2 10.0.1.11:8080 check
    server server3 10.0.1.12:8080 check
```

**Advanced Configuration:**

```haproxy
frontend https_front
    bind *:443 ssl crt /etc/haproxy/certs/example.com.pem

    # Request headers
    http-request set-header X-Forwarded-Proto https if { ssl_fc }
    http-request add-header X-Forwarded-For %[src]

    # Security headers
    http-response set-header Strict-Transport-Security "max-age=31536000; includeSubDomains"
    http-response set-header X-Frame-Options "DENY"
    http-response set-header X-Content-Type-Options "nosniff"

    # ACLs (Access Control Lists)
    acl is_api path_beg /api
    acl is_static path_beg /static
    acl is_admin path_beg /admin

    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny deny_status 429 if { sc_http_req_rate(0) gt 100 }

    # Routing
    use_backend api_backend if is_api
    use_backend static_backend if is_static
    use_backend admin_backend if is_admin
    default_backend web_backend

backend api_backend
    balance leastconn
    option httpchk GET /health
    http-check expect status 200

    server api1 10.0.2.10:8080 check inter 5s fall 3 rise 2
    server api2 10.0.2.11:8080 check inter 5s fall 3 rise 2
    server api3 10.0.2.12:8080 check inter 5s fall 3 rise 2

backend web_backend
    balance roundrobin
    cookie SERVERID insert indirect nocache

    server web1 10.0.1.10:8080 check cookie web1
    server web2 10.0.1.11:8080 check cookie web2
    server web3 10.0.1.12:8080 check cookie web3

backend static_backend
    balance source
    hash-type consistent

    server static1 10.0.3.10:8080 check
    server static2 10.0.3.11:8080 check

# Statistics
listen stats
    bind *:8404
    stats enable
    stats uri /
    stats refresh 30s
    stats show-legends
    stats show-node
```

**TCP Load Balancing:**

```haproxy
frontend mysql_front
    mode tcp
    bind *:3306
    option tcplog
    default_backend mysql_back

backend mysql_back
    mode tcp
    balance leastconn
    option mysql-check user haproxy

    server mysql1 10.0.4.10:3306 check
    server mysql2 10.0.4.11:3306 check
    server mysql3 10.0.4.12:3306 check backup
```

**Blue-Green Deployment:**

```haproxy
backend app_backend
    # Blue environment (stable)
    server blue1 10.0.5.10:8080 check weight 100
    server blue2 10.0.5.11:8080 check weight 100

    # Green environment (new version, disabled initially)
    server green1 10.0.6.10:8080 check weight 0
    server green2 10.0.6.11:8080 check weight 0
```

Switch traffic using runtime API:
```bash
# Set weight to 0 (disable)
echo "set weight app_backend/blue1 0" | socat stdio /run/haproxy/admin.sock

# Set weight to 100 (enable)
echo "set weight app_backend/green1 100" | socat stdio /run/haproxy/admin.sock
```

### Envoy

Envoy is a modern, cloud-native proxy designed for microservices.

**Basic Configuration:**

```yaml
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 10000
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          access_log:
          - name: envoy.access_loggers.stdout
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
          http_filters:
          - name: envoy.filters.http.router
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  cluster: backend_cluster

  clusters:
  - name: backend_cluster
    connect_timeout: 0.25s
    type: STRICT_DNS
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: backend_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: backend1.example.com
                port_value: 8080
        - endpoint:
            address:
              socket_address:
                address: backend2.example.com
                port_value: 8080
```

**Advanced Configuration:**

```yaml
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 443
    filter_chains:
    - transport_socket:
        name: envoy.transport_sockets.tls
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
          common_tls_context:
            tls_certificates:
            - certificate_chain:
                filename: "/etc/envoy/certs/cert.pem"
              private_key:
                filename: "/etc/envoy/certs/key.pem"
      filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          codec_type: AUTO

          # Route configuration
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend
              domains: ["api.example.com"]
              routes:
              # API v1
              - match:
                  prefix: "/api/v1"
                route:
                  cluster: api_v1_cluster
                  retry_policy:
                    retry_on: "5xx"
                    num_retries: 3

              # API v2
              - match:
                  prefix: "/api/v2"
                route:
                  cluster: api_v2_cluster
                  timeout: 15s

              # Health check
              - match:
                  prefix: "/health"
                direct_response:
                  status: 200
                  body:
                    inline_string: "healthy"

          http_filters:
          # Rate limiting
          - name: envoy.filters.http.ratelimit
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.ratelimit.v3.RateLimit
              domain: backend
              request_type: both
              rate_limit_service:
                grpc_service:
                  envoy_grpc:
                    cluster_name: ratelimit

          # Router (must be last)
          - name: envoy.filters.http.router

  clusters:
  # API v1 cluster
  - name: api_v1_cluster
    connect_timeout: 0.25s
    type: STRICT_DNS
    lb_policy: LEAST_REQUEST

    # Health check
    health_checks:
    - timeout: 1s
      interval: 5s
      unhealthy_threshold: 2
      healthy_threshold: 2
      http_health_check:
        path: "/health"
        expected_statuses:
        - start: 200
          end: 200

    # Circuit breaking
    circuit_breakers:
      thresholds:
      - priority: DEFAULT
        max_connections: 1000
        max_pending_requests: 100
        max_requests: 1000
        max_retries: 3

    # Outlier detection
    outlier_detection:
      consecutive_5xx: 5
      interval: 30s
      base_ejection_time: 30s
      max_ejection_percent: 50

    load_assignment:
      cluster_name: api_v1_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.0.1.10
                port_value: 8080
          load_balancing_weight: 3
        - endpoint:
            address:
              socket_address:
                address: 10.0.1.11
                port_value: 8080
          load_balancing_weight: 2

  # API v2 cluster
  - name: api_v2_cluster
    connect_timeout: 0.25s
    type: STRICT_DNS
    lb_policy: RING_HASH  # Consistent hashing
    ring_hash_lb_config:
      minimum_ring_size: 1024

    load_assignment:
      cluster_name: api_v2_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.0.2.10
                port_value: 8080
        - endpoint:
            address:
              socket_address:
                address: 10.0.2.11
                port_value: 8080

admin:
  address:
    socket_address:
      address: 127.0.0.1
      port_value: 9901
```

**Service Mesh Integration (Envoy as Sidecar):**

```yaml
# Envoy sidecar configuration for Kubernetes
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 127.0.0.1
        port_value: 15001
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          tracing:
            provider:
              name: envoy.tracers.zipkin
              typed_config:
                "@type": type.googleapis.com/envoy.config.trace.v3.ZipkinConfig
                collector_cluster: zipkin
                collector_endpoint: "/api/v2/spans"
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match:
                  prefix: "/"
                route:
                  cluster: local_service

  clusters:
  - name: local_service
    connect_timeout: 0.25s
    type: STATIC
    load_assignment:
      cluster_name: local_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 127.0.0.1
                port_value: 8080
```

---

## DNS-Based Load Balancing

DNS load balancing distributes traffic by returning different IP addresses for the same domain name.

### How It Works

```
Client queries: www.example.com
    ↓
DNS server responds with one of:
  - 192.168.1.10 (Server 1)
  - 192.168.1.11 (Server 2)
  - 192.168.1.12 (Server 3)
    ↓
Client connects to returned IP
```

### DNS Load Balancing Methods

**1. Round Robin DNS**

```
; DNS Zone File
www.example.com.    IN  A   192.168.1.10
www.example.com.    IN  A   192.168.1.11
www.example.com.    IN  A   192.168.1.12

DNS server rotates order of IPs in response
```

**BIND Configuration:**

```bind
zone "example.com" {
    type master;
    file "/etc/bind/zones/example.com";

    # Enable round-robin
    rrset-order {
        order cyclic;
    };
};
```

**2. Weighted DNS**

Different weights for each server.

```
www.example.com.    IN  A   192.168.1.10  ; weight 50
www.example.com.    IN  A   192.168.1.11  ; weight 30
www.example.com.    IN  A   192.168.1.12  ; weight 20
```

**3. Geographic DNS (GeoDNS)**

Return IPs based on client location.

```
Client in US → US data center IP
Client in EU → EU data center IP
Client in Asia → Asia data center IP
```

**Route 53 Geolocation Routing:**

```hcl
resource "aws_route53_record" "www_us" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"
  ttl     = 300

  geolocation_routing_policy {
    continent = "NA"
  }

  records = ["192.168.1.10"]
}

resource "aws_route53_record" "www_eu" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"
  ttl     = 300

  geolocation_routing_policy {
    continent = "EU"
  }

  records = ["192.168.2.10"]
}

resource "aws_route53_record" "www_asia" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"
  ttl     = 300

  geolocation_routing_policy {
    continent = "AS"
  }

  records = ["192.168.3.10"]
}
```

**4. Latency-Based Routing**

```hcl
resource "aws_route53_record" "www_us_east" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"
  ttl     = 300

  latency_routing_policy {
    region = "us-east-1"
  }

  records = [aws_eip.us_east.public_ip]
}

resource "aws_route53_record" "www_eu_west" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"
  ttl     = 300

  latency_routing_policy {
    region = "eu-west-1"
  }

  records = [aws_eip.eu_west.public_ip]
}
```

**5. Failover Routing**

```hcl
resource "aws_route53_health_check" "primary" {
  ip_address        = "192.168.1.10"
  port              = 80
  type              = "HTTP"
  resource_path     = "/health"
  failure_threshold = 3
  request_interval  = 30
}

resource "aws_route53_record" "www_primary" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"
  ttl     = 60

  failover_routing_policy {
    type = "PRIMARY"
  }

  health_check_id = aws_route53_health_check.primary.id
  records         = ["192.168.1.10"]
}

resource "aws_route53_record" "www_secondary" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"
  ttl     = 60

  failover_routing_policy {
    type = "SECONDARY"
  }

  records = ["192.168.2.10"]
}
```

### DNS Load Balancing Limitations

**1. TTL Caching**
```
Problem: Clients cache DNS results
Impact: Can't instantly redirect traffic
Solution: Use low TTL (but increases DNS queries)
```

**2. No Health Checks (Traditional DNS)**
```
Problem: DNS returns IP even if server is down
Impact: Clients connect to failed servers
Solution: Use managed DNS (Route 53, CloudFlare) with health checks
```

**3. Uneven Distribution**
```
Problem: Client-side caching, recursive resolvers
Impact: Some servers get more traffic
Solution: Combine with application-level load balancing
```

**4. No Session Persistence**
```
Problem: Different IPs returned for same client
Impact: Session loss
Solution: Use sticky load balancers behind DNS
```

### Best Practices

**1. Use Low TTL for Critical Services**
```
; Quick failover (1 minute TTL)
www.example.com.  60  IN  A  192.168.1.10

; Less critical (5 minutes)
static.example.com.  300  IN  A  192.168.1.20
```

**2. Combine DNS with Application Load Balancers**
```
DNS → Multiple regions
  Each region → Load balancer
    Each load balancer → Multiple servers
```

**3. Health Check Integration**
```
Only return IPs of healthy endpoints
Automatic failover on health check failure
```

---

## Global Server Load Balancing (GSLB)

GSLB distributes traffic across geographically dispersed data centers for global availability and performance.

### GSLB Architecture

```
                        Internet
                           |
                    DNS/GSLB Layer
                    /      |      \
                   /       |       \
          US Data Center  EU Data Center  Asia Data Center
               |               |               |
          Regional LB      Regional LB     Regional LB
          /    |    \      /    |    \     /    |    \
        App1 App2 App3   App1 App2 App3  App1 App2 App3
```

### GSLB Algorithms

**1. Geographic Proximity**
Route users to nearest data center.

**2. Performance-Based**
Route based on measured latency/performance.

**3. Availability-Based**
Route to available data centers only.

**4. Load-Based**
Route based on current data center load.

**5. Cost-Based**
Optimize for infrastructure costs.

### Implementation Examples

**AWS Route 53 GSLB:**

```hcl
# Health checks for each region
resource "aws_route53_health_check" "us_east" {
  type              = "HTTPS"
  resource_path     = "/health"
  fqdn              = "us-east.example.com"
  port              = 443
  failure_threshold = 3
  request_interval  = 30

  tags = {
    Name = "US East Health Check"
  }
}

resource "aws_route53_health_check" "eu_west" {
  type              = "HTTPS"
  resource_path     = "/health"
  fqdn              = "eu-west.example.com"
  port              = 443
  failure_threshold = 3
  request_interval  = 30

  tags = {
    Name = "EU West Health Check"
  }
}

# Multi-region failover
resource "aws_route53_record" "www" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"

  # Geolocation + Latency + Failover
  set_identifier = "US-East-Primary"

  geolocation_routing_policy {
    continent = "NA"
  }

  alias {
    name                   = aws_lb.us_east.dns_name
    zone_id                = aws_lb.us_east.zone_id
    evaluate_target_health = true
  }

  health_check_id = aws_route53_health_check.us_east.id
}

resource "aws_route53_record" "www_eu" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"

  set_identifier = "EU-West-Primary"

  geolocation_routing_policy {
    continent = "EU"
  }

  alias {
    name                   = aws_lb.eu_west.dns_name
    zone_id                = aws_lb.eu_west.zone_id
    evaluate_target_health = true
  }

  health_check_id = aws_route53_health_check.eu_west.id
}

# Default/fallback
resource "aws_route53_record" "www_default" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "www.example.com"
  type    = "A"

  set_identifier = "Default"

  geolocation_routing_policy {
    continent = "*"
  }

  alias {
    name                   = aws_lb.us_east.dns_name
    zone_id                = aws_lb.us_east.zone_id
    evaluate_target_health = true
  }
}
```

**CloudFlare Load Balancing:**

```hcl
# Origin pools (data centers)
resource "cloudflare_load_balancer_pool" "us_east_pool" {
  name = "us-east-pool"

  origins {
    name    = "us-east-1"
    address = "192.168.1.10"
    enabled = true
  }

  origins {
    name    = "us-east-2"
    address = "192.168.1.11"
    enabled = true
  }

  latitude  = 39.0
  longitude = -77.5

  check_regions = ["WNAM", "ENAM"]

  monitor = cloudflare_load_balancer_monitor.http_monitor.id
}

resource "cloudflare_load_balancer_pool" "eu_west_pool" {
  name = "eu-west-pool"

  origins {
    name    = "eu-west-1"
    address = "192.168.2.10"
    enabled = true
  }

  latitude  = 51.5
  longitude = -0.1

  monitor = cloudflare_load_balancer_monitor.http_monitor.id
}

# Health monitor
resource "cloudflare_load_balancer_monitor" "http_monitor" {
  type        = "http"
  port        = 80
  method      = "GET"
  path        = "/health"
  interval    = 60
  timeout     = 5
  retries     = 2
  expected_codes = "200"
}

# Global load balancer
resource "cloudflare_load_balancer" "global_lb" {
  zone_id          = var.cloudflare_zone_id
  name             = "www.example.com"
  fallback_pool_id = cloudflare_load_balancer_pool.us_east_pool.id
  default_pool_ids = [
    cloudflare_load_balancer_pool.us_east_pool.id,
    cloudflare_load_balancer_pool.eu_west_pool.id
  ]

  # Geographic steering
  region_pools {
    region   = "WNAM"
    pool_ids = [cloudflare_load_balancer_pool.us_east_pool.id]
  }

  region_pools {
    region   = "WEUR"
    pool_ids = [cloudflare_load_balancer_pool.eu_west_pool.id]
  }

  # Steering policy
  steering_policy = "geo"  # or "dynamic_latency", "random", "off"

  # Session affinity
  session_affinity = "cookie"
  session_affinity_ttl = 3600
}
```

### GSLB Failover Scenarios

**1. Regional Failure**
```
Normal:
  US Users → US Data Center
  EU Users → EU Data Center

US Data Center Fails:
  US Users → EU Data Center (automatic failover)
  EU Users → EU Data Center
```

**2. Degraded Performance**
```
US Data Center High Latency:
  Some US Users → EU Data Center (dynamic routing)
  EU Users → EU Data Center
```

**3. Maintenance**
```
Planned US Maintenance:
  Gradually drain US traffic to EU
  Perform maintenance
  Restore US, gradually shift traffic back
```

---

## Real-World Architectures

### Architecture 1: Simple Web Application

```
              Internet
                 |
            CloudFlare
                 |
          AWS ALB (Layer 7)
           /    |    \
          /     |     \
     EC2-1   EC2-2   EC2-3
      \       |       /
       \      |      /
         RDS (Read Replicas)
              |
        RDS Primary
```

**Configuration:**
- CloudFlare: DDoS protection, CDN
- ALB: L7 routing, SSL termination
- EC2: Application servers (auto-scaling)
- RDS: Database (multi-AZ)

**Terraform:**

```hcl
# Application Load Balancer
resource "aws_lb" "app_lb" {
  name               = "app-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb_sg.id]
  subnets            = aws_subnet.public.*.id

  enable_deletion_protection = true
  enable_http2              = true
}

# Target group with health checks
resource "aws_lb_target_group" "app_tg" {
  name     = "app-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }

  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = false
  }
}

# Auto-scaling group
resource "aws_autoscaling_group" "app_asg" {
  name                = "app-asg"
  vpc_zone_identifier = aws_subnet.private.*.id
  target_group_arns   = [aws_lb_target_group.app_tg.arn]

  min_size         = 2
  max_size         = 10
  desired_capacity = 3

  launch_template {
    id      = aws_launch_template.app_lt.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "app-server"
    propagate_at_launch = true
  }
}
```

### Architecture 2: Microservices

```
                 Internet
                    |
              API Gateway
                    |
              Kubernetes
                    |
          Ingress Controller
           /       |        \
          /        |         \
    Service A  Service B  Service C
      |  |       |  |       |  |
    Pod Pod    Pod Pod    Pod Pod
```

**Kubernetes Ingress with NGINX:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/load-balance: "ewma"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: tls-secret
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /users
        pathType: Prefix
        backend:
          service:
            name: user-service
            port:
              number: 80
      - path: /orders
        pathType: Prefix
        backend:
          service:
            name: order-service
            port:
              number: 80
      - path: /products
        pathType: Prefix
        backend:
          service:
            name: product-service
            port:
              number: 80
```

**Service with Load Balancing:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: user-service
spec:
  type: ClusterIP
  selector:
    app: user-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
    spec:
      containers:
      - name: user-service
        image: user-service:v1.0
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Architecture 3: Global E-commerce Platform

```
                    Global Users
                         |
                  Route 53 (GSLB)
                   /      |      \
                  /       |       \
         US Region    EU Region   Asia Region
             |            |            |
         CloudFront   CloudFront   CloudFront
             |            |            |
           ALB          ALB          ALB
          /   \        /   \        /   \
       App1  App2   App1  App2   App1  App2
         \     /      \     /      \     /
         Aurora      Aurora        Aurora
           |            |            |
      (Read Replicas across regions)
             \          |          /
              \         |         /
               Global Aurora Cluster
```

**Features:**
- Multi-region deployment
- Local read replicas
- Global write to primary
- CloudFront for static assets
- Geo-routing for low latency

---

## Performance Tuning

### Load Balancer Tuning

**1. Connection Pooling**

```nginx
upstream backend {
    server backend1.example.com:8080;
    server backend2.example.com:8080;

    # Connection pool
    keepalive 64;           # Keep 64 idle connections
    keepalive_requests 100; # Max requests per connection
    keepalive_timeout 60s;  # Idle timeout
}

server {
    location / {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

**Benefits:**
- Reduced connection overhead
- Lower latency
- Better throughput

**2. Buffer Tuning**

```nginx
proxy_buffering on;
proxy_buffer_size 4k;
proxy_buffers 8 4k;
proxy_busy_buffers_size 8k;
proxy_max_temp_file_size 1024m;
```

**3. Timeout Optimization**

```nginx
proxy_connect_timeout 5s;   # Connection to backend
proxy_send_timeout 10s;     # Sending request
proxy_read_timeout 10s;     # Reading response

# Client timeouts
client_body_timeout 12s;
client_header_timeout 12s;
send_timeout 10s;
```

**4. Worker Process Tuning**

```nginx
# nginx.conf
user nginx;
worker_processes auto;  # One per CPU core
worker_rlimit_nofile 100000;

events {
    worker_connections 4096;  # Max connections per worker
    use epoll;                # Efficient I/O method
    multi_accept on;          # Accept multiple connections
}
```

**Calculate capacity:**
```
Max connections = worker_processes * worker_connections
Example: 8 cores * 4096 = 32,768 concurrent connections
```

### TCP Tuning

**Linux Kernel Tuning:**

```bash
# /etc/sysctl.conf

# TCP settings
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 8192

# Connection tracking
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_timestamps = 1

# Buffer sizes
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# Congestion control
net.ipv4.tcp_congestion_control = bbr

# File descriptors
fs.file-max = 2097152

# Apply settings
sysctl -p
```

### HAProxy Tuning

```haproxy
global
    maxconn 100000
    nbproc 8  # Number of processes
    cpu-map auto:1/1-8 0-7

    # Buffers
    tune.bufsize 32768
    tune.maxrewrite 1024

    # SSL
    ssl-default-bind-ciphers ECDHE-ECDSA-AES128-GCM-SHA256
    tune.ssl.default-dh-param 2048
    tune.ssl.cachesize 100000
    tune.ssl.lifetime 600

defaults
    maxconn 10000

    # Timeouts
    timeout connect 5s
    timeout client 50s
    timeout server 50s
    timeout http-keep-alive 10s
    timeout queue 30s

    # Performance
    option http-server-close
    option forwardfor
```

### Monitoring Performance

**Key Metrics:**

```python
performance_metrics = {
    # Throughput
    'requests_per_second': 1000,
    'bytes_per_second': 10_000_000,

    # Latency
    'avg_response_time': 50,      # ms
    'p50_response_time': 45,      # ms
    'p95_response_time': 120,     # ms
    'p99_response_time': 250,     # ms

    # Connection metrics
    'active_connections': 5000,
    'queued_connections': 10,
    'dropped_connections': 0,

    # Backend metrics
    'backend_response_time': 45,  # ms
    'lb_processing_time': 5,      # ms

    # Error rates
    'error_rate': 0.01,           # 0.01%
    'timeout_rate': 0.001,        # 0.001%

    # Resource utilization
    'cpu_usage': 60,              # %
    'memory_usage': 45,           # %
    'network_bandwidth': 800,     # Mbps
}
```

**Prometheus Queries:**

```promql
# Request rate
rate(http_requests_total[5m])

# Average response time
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Backend health
avg(lb_backend_health) by (backend)
```

### Load Testing

**Apache Bench:**

```bash
# Simple load test
ab -n 10000 -c 100 http://example.com/

# With keepalive
ab -n 10000 -c 100 -k http://example.com/

# POST requests
ab -n 1000 -c 10 -p data.json -T application/json http://example.com/api
```

**wrk:**

```bash
# Basic test
wrk -t12 -c400 -d30s http://example.com/

# With script
wrk -t12 -c400 -d30s -s script.lua http://example.com/
```

**Locust (Python):**

```python
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def index(self):
        self.client.get("/")

    @task(1)
    def api(self):
        self.client.get("/api/users")

    @task(1)
    def post_data(self):
        self.client.post("/api/data", json={"key": "value"})
```

Run:
```bash
locust -f loadtest.py --host=http://example.com --users 1000 --spawn-rate 10
```

---

## Best Practices

### 1. Design for Failure

**Assume components will fail:**
- Use health checks
- Implement automatic failover
- Use circuit breakers
- Set appropriate timeouts
- Implement retry logic with backoff

```python
from retrying import retry

@retry(
    stop_max_attempt_number=3,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000
)
def call_backend_service():
    # Will retry up to 3 times with exponential backoff
    return requests.get('http://backend/api')
```

### 2. Use Multiple Layers

```
Layer 1: DNS/GSLB (Geographic distribution)
Layer 2: CDN (Static content, DDoS protection)
Layer 3: L7 Load Balancer (Application routing)
Layer 4: L4 Load Balancer (High performance)
Layer 5: Service Mesh (Microservices)
```

### 3. Health Checks

**Implement comprehensive health checks:**
- TCP connection check (fast)
- HTTP endpoint check (application)
- Deep health check (dependencies)

**Frequency:**
```
Critical services: Every 5 seconds
Normal services: Every 10-30 seconds
Deep checks: Every 1-5 minutes
```

### 4. Monitoring and Alerting

**Monitor:**
- Request rate and latency
- Error rates (4xx, 5xx)
- Backend health status
- SSL certificate expiration
- Connection pool saturation

**Alert on:**
- Error rate > 1%
- Latency p95 > SLA
- All backends unhealthy
- SSL cert expires in < 30 days

### 5. Capacity Planning

**Calculate required capacity:**

```python
# Example calculation
monthly_users = 1_000_000
requests_per_user_per_day = 10
peak_multiplier = 3  # Peak is 3x average

average_rps = (monthly_users * requests_per_user_per_day) / (30 * 24 * 3600)
peak_rps = average_rps * peak_multiplier

# Add 50% headroom
required_capacity = peak_rps * 1.5

print(f"Required capacity: {required_capacity:.0f} RPS")
```

### 6. Security

**SSL/TLS:**
- Use TLS 1.2+ only
- Strong cipher suites
- Certificate monitoring
- HSTS headers

**Rate Limiting:**
```nginx
limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;
limit_req zone=one burst=20 nodelay;
```

**DDoS Protection:**
- Use CDN (CloudFlare, CloudFront)
- Connection limits
- SYN flood protection
- Application-level protection

### 7. Avoid Common Pitfalls

**Don't:**
- ✗ Use DNS round-robin alone for critical services
- ✗ Ignore health checks
- ✗ Set TTL too high
- ✗ Forget to monitor SSL certificates
- ✗ Use sticky sessions unnecessarily
- ✗ Ignore connection limits
- ✗ Skip load testing

**Do:**
- ✓ Use managed load balancers when possible
- ✓ Implement proper health checks
- ✓ Use centralized session storage
- ✓ Monitor all metrics
- ✓ Test failover scenarios
- ✓ Document runbooks
- ✓ Regular load testing

---

## Common Pitfalls

### 1. Single Point of Failure

**Problem:**
```
Single load balancer fails → Entire system down
```

**Solution:**
```
Active-Passive or Active-Active load balancers
Use managed load balancers with built-in redundancy
```

### 2. Inefficient Session Management

**Problem:**
```
Sticky sessions → Uneven load distribution
Session loss on server failure
```

**Solution:**
```
Use centralized session store (Redis)
Use stateless authentication (JWT)
```

### 3. Poor Health Check Design

**Problem:**
```
Health check always returns 200 → Unhealthy servers receive traffic
Health check too expensive → Overloads servers
```

**Solution:**
```python
@app.route('/health')
def health_check():
    # Check critical dependencies
    checks = {
        'database': quick_db_ping(),  # Simple query
        'cache': redis_ping(),
        'disk_space': check_disk_space() > 10  # 10% minimum
    }

    if all(checks.values()):
        return {'status': 'healthy'}, 200
    else:
        return {'status': 'unhealthy', 'checks': checks}, 503
```

### 4. Ignoring Connection Limits

**Problem:**
```
Backend has 1000 max connections
Load balancer sends 2000 connections
→ Backend overloaded
```

**Solution:**
```
Configure connection limits in load balancer
Monitor backend capacity
Implement backpressure
```

### 5. Cascading Failures

**Problem:**
```
One backend slow → Load balancer waits → Other requests queue → All backends slow
```

**Solution:**
```
Aggressive timeouts
Circuit breakers
Rate limiting
Request queuing limits
```

---

## Further Reading

### Books
- "The Art of Scalability" by Martin L. Abbott
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Site Reliability Engineering" by Google
- "High Performance Browser Networking" by Ilya Grigorik

### Documentation
- [NGINX Documentation](https://nginx.org/en/docs/)
- [HAProxy Documentation](https://www.haproxy.org/documentation.html)
- [Envoy Proxy Documentation](https://www.envoyproxy.io/docs)
- [AWS Load Balancing](https://aws.amazon.com/elasticloadbalancing/)
- [Google Cloud Load Balancing](https://cloud.google.com/load-balancing/docs)

### RFCs
- RFC 7540: HTTP/2
- RFC 8446: TLS 1.3
- RFC 7234: HTTP Caching

### Tools
- [Apache Bench](https://httpd.apache.org/docs/2.4/programs/ab.html)
- [wrk](https://github.com/wg/wrk)
- [Locust](https://locust.io/)
- [Vegeta](https://github.com/tsenart/vegeta)

### Blogs
- [Netflix Tech Blog](https://netflixtechblog.com/)
- [Cloudflare Blog](https://blog.cloudflare.com/)
- [HashiCorp Blog](https://www.hashicorp.com/blog)

---

## Summary

Load balancing is essential for building scalable, highly available distributed systems. Key takeaways:

1. **Choose the right layer**: L4 for performance, L7 for flexibility
2. **Select appropriate algorithm**: Match algorithm to use case
3. **Implement robust health checks**: Active and passive monitoring
4. **Plan for failure**: Automatic failover, circuit breakers
5. **Monitor everything**: Request rates, latency, errors, backend health
6. **Test regularly**: Load testing, failover testing
7. **Use managed services**: When possible, leverage cloud load balancers
8. **Design globally**: GSLB for global availability

Load balancing is not just about distributing traffic—it's about ensuring your application remains available, performant, and resilient under any conditions.
