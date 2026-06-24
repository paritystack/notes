# Container Networking

## Overview

Containers (Docker, Kubernetes pods, systemd-nspawn, LXC) share a host kernel but get isolated network stacks via **Linux [network namespaces](../linux/namespace.md)**. Pieces like `veth` pairs, bridges, [iptables](../linux/iptables.md), and routing tables are composed to give each container the illusion of having its own NIC. The same primitives also power Kubernetes networking via **CNI** plugins (Calico, Cilium, Flannel, Weave, etc.), often using [overlay encapsulation](overlay_networks.md) and increasingly [eBPF](../linux/ebpf.md) instead of iptables.

This note covers what's actually happening underneath, so you can debug why your pod can't reach the database.

## Linux Network Namespaces

A network namespace is an isolated copy of the kernel network stack: its own interfaces, IP addresses, routing table, ARP cache, firewall rules, and sockets.

```bash
# Create a namespace
sudo ip netns add ns1
sudo ip netns list

# Execute a command inside
sudo ip netns exec ns1 ip link show
sudo ip netns exec ns1 ip addr
sudo ip netns exec ns1 ping 8.8.8.8

# Delete
sudo ip netns del ns1
```

Inside a fresh namespace you see only `lo` (loopback), which is **down**. You can't reach anything until you add an interface and bring lo up.

```bash
sudo ip netns exec ns1 ip link set lo up
```

### Inspecting Container Namespaces

```bash
# Docker
docker inspect -f '{{.State.Pid}}' my-container       # PID
sudo nsenter -t $PID -n ip addr                       # peek inside

# Or symlink it so ip netns works
sudo ln -s /proc/$PID/ns/net /var/run/netns/my-container
sudo ip netns exec my-container ip route

# Kubernetes
crictl ps                                             # list containers
crictl inspect <container-id> | jq .info.pid
sudo nsenter -t $PID -n ip addr
```

## veth Pairs

A `veth` (virtual Ethernet) pair is two interfaces wired together — packets in one come out the other. The standard way to connect a namespace to the host.

```bash
sudo ip link add veth-host type veth peer name veth-ns
sudo ip link set veth-ns netns ns1                    # move one end into ns1

# Configure host side
sudo ip addr add 10.0.0.1/24 dev veth-host
sudo ip link set veth-host up

# Configure namespace side
sudo ip netns exec ns1 ip addr add 10.0.0.2/24 dev veth-ns
sudo ip netns exec ns1 ip link set veth-ns up

# Default route in namespace
sudo ip netns exec ns1 ip route add default via 10.0.0.1

# Test
sudo ip netns exec ns1 ping 10.0.0.1
```

This gives one container/namespace external connectivity. For many containers, you usually attach them to a **bridge** instead.

## Linux Bridge (Software Switch)

A `bridge` is a kernel L2 switch — it learns MAC→port mappings and forwards frames between attached interfaces.

```bash
# Create bridge
sudo ip link add br0 type bridge
sudo ip addr add 10.0.0.1/24 dev br0
sudo ip link set br0 up

# Create a veth pair, plug one end into ns1, the other into br0
sudo ip link add veth1 type veth peer name veth1-ns
sudo ip link set veth1 master br0
sudo ip link set veth1 up
sudo ip link set veth1-ns netns ns1
sudo ip netns exec ns1 ip addr add 10.0.0.2/24 dev veth1-ns
sudo ip netns exec ns1 ip link set veth1-ns up
sudo ip netns exec ns1 ip link set lo up
sudo ip netns exec ns1 ip route add default via 10.0.0.1

# Repeat with veth2/ns2 for a second container
# Both containers can now talk to each other through br0 directly
```

This is essentially how **Docker bridge mode** works. `docker0` is just a Linux bridge with NAT'd egress.

## Docker Networking

### Default: bridge mode

```
docker0 (bridge, 172.17.0.1/16)
   ├── veth0 ──── eth0 in container1 (172.17.0.2)
   ├── veth1 ──── eth0 in container2 (172.17.0.3)
   └── veth2 ──── eth0 in container3 (172.17.0.4)

Outbound traffic:
  container → docker0 → host eth0 → MASQUERADE → internet
                                    (iptables NAT)
```

```bash
# Inspect
docker network ls
docker network inspect bridge
ip link show docker0
ip -j link | jq '.[] | select(.master=="docker0")'

# Add a custom bridge network
docker network create --driver bridge --subnet 192.168.50.0/24 mynet
docker run --network=mynet --name=web nginx
```

### Other Driver Modes

```
host       — container shares host's network namespace (no isolation)
none       — only lo, no external connectivity
overlay    — multi-host networking via VXLAN
macvlan    — container gets its own MAC on the physical network
ipvlan     — like macvlan but containers share host MAC, get unique IPs
container:X — share network namespace with another container (sidecar)
```

### Port Publishing

```
docker run -p 8080:80 nginx

Iptables DNAT rule:
  -A DOCKER -p tcp --dport 8080 -j DNAT --to 172.17.0.2:80
```

Inspect: `iptables -t nat -L DOCKER -n -v`.

### Container DNS

Docker runs an embedded DNS resolver at 127.0.0.11 inside containers. Custom networks let containers resolve each other by name:

```bash
docker network create app
docker run --network=app --name=db postgres
docker run --network=app --name=web nginx
# Inside web container: `ping db` → resolves to 172.x.x.x of postgres container
```

## Kubernetes Networking Model

Kubernetes mandates a few rules and lets a **CNI plugin** implement them:

```
1. Every pod gets its own IP address
2. Pods can communicate with all other pods without NAT
3. Nodes can communicate with all pods (and vice versa)
4. The pod sees the same IP as everyone else sees it as
```

This is intentionally simple — no port mapping between containers and the cluster. Pod IPs are first-class.

### Pod Networking Inside a Node

```
A pod is multiple containers sharing one network namespace.
The "pause" container holds the namespace; other containers join it.

┌────────────────────────────────────┐
│  Pod (network namespace)           │
│    eth0  (10.244.1.5/24)           │   ← veth peer
│    lo                              │
└──────────────┬─────────────────────┘
               │ veth pair
               │
       ┌───────┴───────────┐
       │  Host             │
       │  cni0 bridge      │   ← all pods on this node attach here
       │    └─ veth pods   │
       │    └─ veth pods   │
       │  eth0 (real NIC)  │
       └───────────────────┘
```

All containers in the same pod share `localhost` and ports — that's why two containers in the same pod cannot both listen on port 8080.

### Pod-to-Pod, Same Node

Trivial — just bridged through `cni0` (or equivalent).

### Pod-to-Pod, Different Nodes

The CNI plugin must route between nodes somehow. Several approaches:

```
1. Overlay (VXLAN, IP-in-IP)
   - Encapsulate inner pod packet in outer UDP/IP packet
   - Underlay only sees node-to-node traffic
   - Works on any underlay
   - Adds ~50 bytes overhead
   - Plugins: Flannel (vxlan), Calico (IP-in-IP), Cilium (vxlan/geneve), Weave

2. BGP / route distribution
   - Each node advertises its pod CIDR to the network
   - Underlay routers know which node owns which pod IP
   - No encapsulation, native speed
   - Requires control over underlay routing (or BGP-capable cloud)
   - Plugin: Calico in BGP mode, Cilium with BGP

3. Cloud-native (no overlay)
   - Cloud assigns pod IPs from VPC; routes built into VPC fabric
   - AWS VPC CNI, GKE alias IPs, Azure CNI
   - Best performance; tied to cloud
```

See [overlay_networks.md](overlay_networks.md) for VXLAN/GRE/Geneve details.

## CNI (Container Network Interface)

**CNI** is a spec for plugins that configure container networking. Kubernetes (and many other runtimes) shell out to CNI plugins on pod creation/deletion.

### Plugin Lifecycle

```
1. kubelet creates pod
2. kubelet creates network namespace
3. kubelet runs the configured CNI binary with:
   - command: ADD / DEL / CHECK
   - container ID
   - network namespace path
   - JSON config
4. Plugin:
   - allocates IP from IPAM
   - creates veth pair
   - moves one end into namespace
   - configures interface, routes, MTU
   - returns IP info to kubelet
5. kubelet starts containers in the namespace
```

### Plugins You'll Encounter

```
Flannel        simple overlay (VXLAN), no policies
Calico         L3 routing (BGP or IP-in-IP), NetworkPolicy
Cilium         eBPF-based, fast, observability, mesh-like features
Weave Net      mesh overlay
Antrea         OVS-based, NetworkPolicy
AWS VPC CNI    each pod gets ENI from VPC (no overlay)
GKE/Azure CNI  cloud-integrated
Multus         meta-plugin for multiple networks per pod
```

### Cilium / eBPF

Modern CNIs increasingly use **eBPF** instead of iptables — programmable kernel hooks that route packets without iptables overhead. Cilium has effectively become the default for many new K8s deployments.

```
Benefits:
  - O(1) lookups (vs iptables O(N) rule chains)
  - Per-pod policies enforced at L3/L4/L7
  - Bypasses some kernel layers entirely (XDP)
  - Observability built-in (Hubble)
```

## Services and kube-proxy

A Kubernetes **Service** is a stable virtual IP that load-balances to a set of pods.

```
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  selector: { app: web }
  ports:
    - port: 80
      targetPort: 8080
  clusterIP: 10.96.42.7
```

When traffic hits 10.96.42.7:80, kube-proxy (or the CNI) load-balances to the pod backends.

### kube-proxy Modes

```
iptables (default classic)
  - Rules: -A KUBE-SERVICES -d 10.96.42.7/32 -p tcp --dport 80 -j KUBE-SVC-...
  - DNAT to one of the endpoint IPs (probabilistic)
  - Works but rule count grows linearly with services × endpoints

ipvs
  - Kernel L4 load balancer
  - O(1) lookups; handles thousands of services well
  - Better for large clusters

eBPF (Cilium replaces kube-proxy)
  - No iptables / IPVS at all
  - Per-socket load-balancing decisions
```

### Service Types

```
ClusterIP        internal-only virtual IP (default)
NodePort         expose on every node's IP at a high port (30000-32767)
LoadBalancer     cloud-provider provisioned external LB
ExternalName     DNS CNAME, no proxy
Headless (None)  no cluster IP, DNS returns pod IPs directly
```

### EndpointSlices

Replaced legacy `Endpoints`. Splits big endpoint lists into smaller objects so the control plane doesn't choke when a service has thousands of pods.

## Ingress and Service Mesh

Beyond plain Services:

```
Ingress (L7 HTTP routing)
  - Reverse proxy on cluster edge: nginx, Traefik, HAProxy, Envoy
  - Routes by hostname/path
  - Replaced by Gateway API in newer versions

Service Mesh (sidecar or sidecarless)
  - Istio (Envoy sidecars)
  - Linkerd (lighter, Rust)
  - Cilium Service Mesh (sidecarless, eBPF)
  - mTLS, retries, traffic splitting, observability
```

## NetworkPolicy

Kubernetes-native L3/L4 firewall, enforced by the CNI:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-from-frontend-only
spec:
  podSelector:
    matchLabels: { app: api }
  policyTypes: [Ingress]
  ingress:
    - from:
        - podSelector:
            matchLabels: { app: frontend }
      ports:
        - protocol: TCP
          port: 8080
```

NetworkPolicy **only works if your CNI supports it** — Flannel doesn't, Calico/Cilium do.

Cilium adds `CiliumNetworkPolicy` for L7 (HTTP method/path) and FQDN policies.

## Common Diagnostic Commands

```bash
# What namespace does this container live in?
docker inspect -f '{{.State.Pid}}' <container>
ls -la /proc/<pid>/ns/net          # netns inode

# All veth interfaces and their peers
ip -d link show
ethtool -S vethXXXX

# Bridge port table
bridge fdb show
bridge link show

# Routing inside a pod
kubectl exec <pod> -- ip route
kubectl exec <pod> -- ip addr
kubectl exec <pod> -- ip -s link

# Service iptables rules
sudo iptables -t nat -L KUBE-SERVICES -n | grep <service-ip>
sudo iptables -t nat -L KUBE-SVC-XXX -n
sudo ipvsadm -Ln                    # if using IPVS mode

# Capture inside pod's namespace
PID=$(crictl inspect <id> | jq .info.pid)
sudo nsenter -t $PID -n tcpdump -i any -nn

# Or use `kubectl debug` / ephemeral containers / netshoot
kubectl debug -it <pod> --image=nicolaka/netshoot
```

## Common Issues

### "Pod cannot reach external internet"

```
Check on the node:
  sysctl net.ipv4.ip_forward          # must be 1
  iptables -t nat -L POSTROUTING -n | grep MASQUERADE
  ip route                            # default route exists?

Inside pod:
  cat /etc/resolv.conf                # cluster DNS reachable?
  nslookup kubernetes.default
```

### "Pods in same node cannot talk"

```
- Bridge interface up?
- Promiscuous mode on bridge?
- iptables FORWARD chain allow rules?
  iptables -P FORWARD ACCEPT
  (Docker sets DROP by default; CNI must add ACCEPT rules)
```

### "Pods on different nodes cannot talk"

```
- Underlay routing problem (BGP not converged, route missing)
- VXLAN port (4789) blocked between nodes
- MTU mismatch: pod MTU > underlay MTU − overhead
- Encryption (IPsec/WireGuard) not converged
```

### "Service IP unreachable"

```
- kube-proxy running on every node?
- iptables/IPVS rules present?
- Endpoints object has IPs? (kubectl get endpoints)
- Pod readiness probe failing → no endpoints
```

### "MTU pain"

This is the classic Kubernetes networking bug. Default pod MTU is often 1500, but VXLAN underlay only carries 1450. TCP works because PMTUD kicks in; UDP and large requests die mysteriously.

```bash
# Check pod MTU
kubectl exec <pod> -- ip link show eth0

# Check node MTU
ip link show eth0

# Should differ by overlay overhead:
#   VXLAN:  50 bytes
#   Geneve: 50 bytes
#   IP-in-IP: 20 bytes
#   WireGuard: 60-80 bytes
```

Fix in CNI config (`MTU` setting). See [mtu_pmtud.md](mtu_pmtud.md).

## eBPF and the Modern Stack

```
Traditional path:
  packet → iptables (PRE/INPUT/FORWARD/OUTPUT/POST) → routing → ...
  Many rule chains, every packet walks them

eBPF path:
  packet → XDP (NIC driver) → tc → socket
  Programmable bytecode runs at each hook
  Decisions in nanoseconds
```

Used by Cilium, Calico's newer modes, Falco, Tetragon. Worth learning if you're doing K8s networking at scale.

## A Minimal Two-Container Setup From Scratch

To cement the concepts:

```bash
# 1. Create bridge
sudo ip link add br0 type bridge
sudo ip addr add 192.168.99.1/24 dev br0
sudo ip link set br0 up

# 2. Create two namespaces
sudo ip netns add nsa
sudo ip netns add nsb

# 3. veth pairs
sudo ip link add veth-a type veth peer name veth-a-ns
sudo ip link add veth-b type veth peer name veth-b-ns

sudo ip link set veth-a master br0 && sudo ip link set veth-a up
sudo ip link set veth-b master br0 && sudo ip link set veth-b up

sudo ip link set veth-a-ns netns nsa
sudo ip link set veth-b-ns netns nsb

# 4. Configure namespace sides
sudo ip netns exec nsa ip addr add 192.168.99.10/24 dev veth-a-ns
sudo ip netns exec nsa ip link set veth-a-ns up
sudo ip netns exec nsa ip link set lo up
sudo ip netns exec nsa ip route add default via 192.168.99.1

sudo ip netns exec nsb ip addr add 192.168.99.20/24 dev veth-b-ns
sudo ip netns exec nsb ip link set veth-b-ns up
sudo ip netns exec nsb ip link set lo up
sudo ip netns exec nsb ip route add default via 192.168.99.1

# 5. Enable forwarding + NAT for external internet
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -s 192.168.99.0/24 -o eth0 -j MASQUERADE

# 6. Test
sudo ip netns exec nsa ping 192.168.99.20    # works (same bridge)
sudo ip netns exec nsa ping 8.8.8.8          # works (NAT'd out)
```

This is essentially Docker bridge mode in 25 commands.

## ELI10

Imagine your computer is an apartment building. A **network namespace** is one apartment with its own front door, mailbox, and address — even though it's in the same building.

A **veth pair** is a magic doorway: one door opens in the apartment, the other in the hallway. Anything walked through one comes out the other.

A **bridge** is the hallway with many doors leading off it — every apartment connected to that hallway can visit every other one.

In **Kubernetes**, each **pod** is one apartment, possibly with several roommates (containers) sharing the same mailbox and TV. A **service** is the building's front desk: any letter to "the engineering team" gets routed to whichever specific apartment of that team is least busy.

A **CNI plugin** is the building manager who decides how the doors and hallways are wired up. Different building managers (Flannel, Calico, Cilium) have very different opinions about whether to dig tunnels between buildings (overlays) or post signs at every street corner (BGP routes).

## Where this connects

- [Linux namespaces](../linux/namespace.md) — the network namespace primitive every container builds on
- [iptables](../linux/iptables.md), [eBPF](../linux/ebpf.md) — the two engines behind kube-proxy and CNI dataplanes
- [Overlay networks](overlay_networks.md) — VXLAN/Geneve/IP-in-IP for pod-to-pod across nodes
- [BGP & Anycast](bgp_anycast.md) — route-distribution mode (Calico/Cilium BGP) instead of overlays
- [Firewalls](firewalls.md) — NetworkPolicy is a CNI-enforced L3/L4 firewall
- [Microservices](../system_design/microservices.md), [Load balancing](../system_design/load_balancing.md) — Services/Ingress that ride on this plumbing

## Further Resources

- [Linux network namespaces (man 7 namespaces)](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [CNI specification](https://github.com/containernetworking/cni/blob/main/SPEC.md)
- [Cilium docs](https://docs.cilium.io/)
- [Calico docs](https://docs.tigera.io/calico/latest/about/)
- [Kubernetes networking model](https://kubernetes.io/docs/concepts/services-networking/)
- [Learn Kubernetes Networking — Murat Demirbas blog series](http://muratbuffalo.blogspot.com/)
- [iptables-tutorial.frozentux.net](https://www.frozentux.net/iptables-tutorial/iptables-tutorial.html)
