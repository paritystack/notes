# tc


`tc` (traffic control) is a utility in the Linux kernel used to configure Traffic Control in the network stack. It allows administrators to configure the queuing discipline (qdisc), which determines how packets are enqueued and dequeued from the network interface.

### Important Components of `tc`

1. **qdisc (Queuing Discipline)**: The core component of `tc`, which defines the algorithm used to manage the packet queue. Examples include `pfifo_fast`, `fq_codel`, and `netem`.
2. **class**: A way to create a hierarchy within a qdisc, allowing for more granular control over traffic. Classes can be used to apply different rules to different types of traffic.
3. **filter**: Used to classify packets into different classes. Filters can match on various packet attributes, such as IP address, port number, or protocol.
4. **action**: Defines what to do with packets that match a filter. Actions can include marking, mirroring, or redirecting packets.

### Uses of `tc`

- **Traffic Shaping**: Control the rate of outgoing traffic to ensure that the network is not overwhelmed. This can be useful for managing bandwidth usage and ensuring fair distribution of network resources.
- **Traffic Policing**: Enforce limits on the rate of incoming traffic, dropping packets that exceed the specified rate. This can help protect against network abuse or attacks.
- **Network Emulation**: Simulate various network conditions, such as latency, packet loss, and jitter, to test the performance of applications under different scenarios.
- **Quality of Service (QoS)**: Prioritize certain types of traffic to ensure that critical applications receive the necessary bandwidth and low latency.

By using `tc`, administrators can fine-tune network performance, improve reliability, and ensure that critical applications have the necessary resources to function optimally.




Add delay to all traffic on eth0

```bash
sudo tc qdisc add dev eth0 root netem delay 100ms
```