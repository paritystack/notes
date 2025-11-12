# Scalability

## Overview

Scalability is the ability to handle increased load by adding more resources.

## Vertical Scaling

Add more power to existing machines:

```
1 machine: 8 cores, 32GB RAM
→ Upgrade to: 16 cores, 128GB RAM
```

**Pros**: Simple, less complexity
**Cons**: Hardware limits, single point of failure

## Horizontal Scaling

Add more machines:

```
Machine 1: Handle requests
Machine 2: Handle requests
Machine 3: Handle requests
↓ Load Balancer ↓
     Clients
```

**Pros**: Unlimited growth, fault tolerance
**Cons**: More complexity, state management

## Load Balancing

```
    Client Request
         ↓
   ┌─ Load Balancer ─┐
   ↓        ↓        ↓
Server1  Server2  Server3
```

**Algorithms**:
- **Round Robin**: Rotate servers
- **Least Connections**: Route to least busy
- **IP Hash**: Same client to same server
- **Weighted**: Distribute by capacity

## Database Scaling

### Replication
Master-slave setup:
- Master: Writes
- Slaves: Read copies

```
      Master (R/W)
      ↙    ↓    ↘
   Slave1 Slave2 Slave3 (R only)
```

### Sharding
Partition data across databases:

```
Shard 1: Users 1-1M
Shard 2: Users 1M-2M
Shard 3: Users 2M-3M

By User ID % 3 → Route to correct shard
```

## Caching

Store frequently accessed data:

```
Client Request
     ↓
Check Cache (fast)
  ↓ miss ↓ hit
Database → Client
(slow)
```

**Cache Invalidation**:
- **TTL**: Expire after time
- **Event-based**: Invalidate on update
- **LRU**: Remove least used items

## Common Patterns

### CDN (Content Delivery Network)
Distributed servers for static content:

```
User in Asia → Asia CDN Server (fast)
User in US → US CDN Server (fast)
```

### Queue Systems
Handle spikes asynchronously:

```
Request → Queue → Worker Pool → Database
(fast)              (slow processing)
```

### Read Replicas
Separate read and write:

```
Write (slow): Direct to master
Read (fast): From replicas
```

## Metrics

| Metric | Target |
|--------|--------|
| **Response Time** | <100ms |
| **Throughput** | >1000 req/s |
| **Uptime** | >99.9% |
| **Availability** | 5-9s |

## ELI10

Scalability is like growing a restaurant:
- **Vertical**: Make kitchen bigger (limited)
- **Horizontal**: Open more locations (unlimited)
- **Load balancer**: Customers split between locations
- **Caching**: Keep popular dishes ready
- **Queues**: Don't overwhelm kitchen

Design for growth from day one!

## Further Resources

- [Designing Data-Intensive Applications](https://dataintensive.app/)
- [System Design Interview](https://www.systemdesigninterview.com/)
