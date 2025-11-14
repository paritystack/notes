# Linux Namespaces

Linux namespaces are a kernel feature that partitions kernel resources so that one set of processes sees one set of resources while another set of processes sees a different set of resources. They are the fundamental building blocks for containerization technologies like Docker, LXC, and Kubernetes.

## Overview

Namespaces provide isolation by virtualizing system resources for processes. Each namespace type isolates a different aspect of the system, creating independent instances of global system resources.

**Key Benefits:**
- Process isolation and resource partitioning
- Foundation for container technologies
- Enhanced security through separation
- Resource management and control
- Support for multi-tenancy

**Namespace Types:**
- **PID**: Process ID isolation
- **NET**: Network stack isolation
- **MNT**: Filesystem mount points
- **UTS**: Hostname and domain name
- **IPC**: Inter-process communication
- **USER**: User and group ID mappings
- **CGROUP**: Control group isolation
- **TIME**: System time isolation (Linux 5.6+)

## Namespace Types in Detail

### PID Namespace

Isolates process IDs. Processes in a PID namespace only see processes within the same namespace. The first process becomes PID 1 and acts as init.

**Key Features:**
- Process tree isolation
- PID 1 is namespace init process
- Nested PID namespaces supported
- Orphaned processes reaped by namespace init

**Example:**
```bash
# Create new PID namespace
sudo unshare --pid --fork --mount-proc /bin/bash

# Inside namespace
ps aux  # Only shows processes in this namespace
echo $$  # Shows PID 1 or low PID number
```

### Network Namespace

Provides isolated network stack including interfaces, routing tables, firewall rules, and sockets.

**Key Features:**
- Independent network interfaces
- Separate routing tables
- Isolated iptables/nftables rules
- Unique IP addresses
- Separate port numbers

**Example:**
```bash
# Create network namespace
sudo ip netns add myns

# List namespaces
ip netns list

# Execute in namespace
sudo ip netns exec myns ip addr
sudo ip netns exec myns bash

# Create veth pair (virtual ethernet)
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth1 netns myns

# Configure interfaces
sudo ip addr add 10.0.0.1/24 dev veth0
sudo ip link set veth0 up

sudo ip netns exec myns ip addr add 10.0.0.2/24 dev veth1
sudo ip netns exec myns ip link set veth1 up
sudo ip netns exec myns ip link set lo up

# Test connectivity
ping -c 3 10.0.0.2
```

### Mount Namespace

Isolates filesystem mount points. Changes to mounts in one namespace don't affect others.

**Key Features:**
- Independent mount points
- Private filesystem hierarchy
- Propagation types (shared, private, slave, unbindable)
- Useful for chroot-like isolation

**Example:**
```bash
# Create mount namespace
sudo unshare --mount /bin/bash

# Mount is private to this namespace
mkdir /tmp/mydata
mount -t tmpfs tmpfs /tmp/mydata
df -h  # Shows the mount
exit

# Mount not visible in parent namespace
df -h  # No /tmp/mydata
```

**Mount Propagation:**
```bash
# View mount propagation
findmnt -o TARGET,PROPAGATION

# Make mount private
mount --make-private /mnt/shared

# Make mount shared
mount --make-shared /mnt/shared

# Make mount slave
mount --make-slave /mnt/shared

# Make mount unbindable
mount --make-unbindable /mnt/shared
```

### UTS Namespace

Isolates hostname and NIS domain name. Allows each container to have its own hostname.

**Key Features:**
- Independent hostname
- Independent domain name
- Useful for multi-tenant systems

**Example:**
```bash
# Create UTS namespace
sudo unshare --uts /bin/bash

# Change hostname (only in namespace)
hostname mycontainer
hostname  # Shows "mycontainer"
exit

# Original hostname unchanged
hostname  # Shows original hostname
```

### IPC Namespace

Isolates System V IPC objects and POSIX message queues.

**Key Features:**
- Separate message queues
- Isolated semaphores
- Private shared memory segments
- POSIX message queue isolation

**Example:**
```bash
# View IPC objects
ipcs -a

# Create IPC namespace
sudo unshare --ipc /bin/bash

# Create message queue
ipcmk -Q
ipcs -q  # Only visible in this namespace
exit

# Message queue not visible in parent
ipcs -q
```

### User Namespace

Maps user and group IDs between namespaces. Enables unprivileged containers.

**Key Features:**
- UID/GID mapping
- Capability isolation
- Non-root user can own namespaces
- Security boundary

**Example:**
```bash
# Create user namespace (no root required)
unshare --user --map-root-user /bin/bash

# Check UID
id  # Shows uid=0 (root) in namespace
cat /proc/self/uid_map  # Shows UID mapping

# Real UID outside is different
exit
id  # Shows actual UID
```

**UID Mapping:**
```bash
# Manual UID mapping
unshare --user /bin/bash
echo "0 1000 1" > /proc/self/uid_map
echo "0 1000 1" > /proc/self/gid_map

# Map range of UIDs
# Format: namespace_id host_id count
echo "0 100000 65536" > /proc/self/uid_map
echo "0 100000 65536" > /proc/self/gid_map
```

### Cgroup Namespace

Virtualizes the view of `/proc/self/cgroup` and cgroup mounts.

**Key Features:**
- Cgroup hierarchy isolation
- Prevents escape from cgroup
- Security boundary for containers

**Example:**
```bash
# Create cgroup namespace
sudo unshare --cgroup /bin/bash

# View cgroup
cat /proc/self/cgroup

# Root of cgroup tree appears as /
mount -t cgroup2 none /sys/fs/cgroup
```

### Time Namespace

Allows different processes to see different system times (Linux 5.6+).

**Key Features:**
- Offset CLOCK_MONOTONIC
- Offset CLOCK_BOOTTIME
- Useful for testing and migration

## Command-Line Tools

### unshare Command

Creates new namespaces and executes a program.

```bash
# Basic usage
unshare [options] [program [arguments]]

# Common options
unshare --pid --fork /bin/bash           # PID namespace
unshare --net /bin/bash                  # Network namespace
unshare --mount /bin/bash                # Mount namespace
unshare --uts /bin/bash                  # UTS namespace
unshare --ipc /bin/bash                  # IPC namespace
unshare --user /bin/bash                 # User namespace
unshare --cgroup /bin/bash               # Cgroup namespace

# Multiple namespaces
unshare --pid --net --mount --uts --ipc --fork /bin/bash

# All namespaces
unshare --pid --net --mount --uts --ipc --user --cgroup --fork /bin/bash

# User namespace with UID mapping
unshare --user --map-root-user /bin/bash

# Mount proc in PID namespace
unshare --pid --fork --mount-proc /bin/bash

# Propagation flags
unshare --mount --propagation private /bin/bash
unshare --mount --propagation shared /bin/bash
```

### nsenter Command

Enters existing namespaces of another process.

```bash
# Basic usage
nsenter [options] [program [arguments]]

# Enter specific namespace
nsenter --target PID --pid --net --mount /bin/bash

# Enter all namespaces
nsenter --target PID --all /bin/bash

# Common options
nsenter -t PID --pid /bin/bash           # PID namespace
nsenter -t PID --net /bin/bash           # Network namespace
nsenter -t PID --mount /bin/bash         # Mount namespace
nsenter -t PID --uts /bin/bash           # UTS namespace
nsenter -t PID --ipc /bin/bash           # IPC namespace
nsenter -t PID --user /bin/bash          # User namespace
nsenter -t PID --cgroup /bin/bash        # Cgroup namespace

# Enter Docker container namespace
docker inspect --format '{{.State.Pid}}' container_name
nsenter -t $(docker inspect --format '{{.State.Pid}}' container_name) -n -m -u -i -p /bin/bash

# Preserve effective UID
nsenter -t PID --all --preserve-credentials /bin/bash
```

### ip netns Command

Manages network namespaces.

```bash
# Create namespace
ip netns add namespace_name

# List namespaces
ip netns list
ip netns

# Delete namespace
ip netns delete namespace_name

# Execute command in namespace
ip netns exec namespace_name command
ip netns exec myns ip addr
ip netns exec myns ping 8.8.8.8

# Identify process namespace
ip netns identify PID

# Monitor namespace events
ip netns monitor

# Attach network namespace to name
ip netns attach NAME PID

# Set namespace for process
ip netns set PID NAME
```

### lsns Command

Lists namespaces.

```bash
# List all namespaces
lsns

# List specific type
lsns -t net     # Network namespaces
lsns -t pid     # PID namespaces
lsns -t mnt     # Mount namespaces
lsns -t uts     # UTS namespaces
lsns -t ipc     # IPC namespaces
lsns -t user    # User namespaces
lsns -t cgroup  # Cgroup namespaces

# Show namespace of specific process
lsns -p PID

# Output format
lsns -o NS,TYPE,NPROCS,PID,USER,COMMAND

# Show tree structure
lsns --tree

# JSON output
lsns -J
```

## Programming with Namespaces

### System Calls

**clone()** - Create new process with namespace flags:
```c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define STACK_SIZE (1024 * 1024)

static char child_stack[STACK_SIZE];

int child_fn(void *arg) {
    printf("Child PID: %d\n", getpid());
    printf("Child in new namespace\n");
    return 0;
}

int main() {
    pid_t pid;

    // Create child with new PID and UTS namespace
    pid = clone(child_fn, child_stack + STACK_SIZE,
                CLONE_NEWPID | CLONE_NEWUTS | SIGCHLD, NULL);

    if (pid == -1) {
        perror("clone");
        return 1;
    }

    printf("Parent PID: %d, Child PID: %d\n", getpid(), pid);
    waitpid(pid, NULL, 0);

    return 0;
}
```

**unshare()** - Disassociate parts of process execution context:
```c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    // Create new UTS namespace
    if (unshare(CLONE_NEWUTS) == -1) {
        perror("unshare");
        return 1;
    }

    // Change hostname in namespace
    if (sethostname("container", 9) == -1) {
        perror("sethostname");
        return 1;
    }

    printf("Hostname changed to: container\n");

    // Execute shell
    execlp("/bin/bash", "/bin/bash", NULL);

    return 0;
}
```

**setns()** - Join existing namespace:
```c
#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int fd;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <namespace-path>\n", argv[0]);
        return 1;
    }

    // Open namespace file
    fd = open(argv[1], O_RDONLY);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    // Join namespace
    if (setns(fd, 0) == -1) {
        perror("setns");
        close(fd);
        return 1;
    }

    close(fd);

    printf("Joined namespace\n");

    // Execute shell
    execlp("/bin/bash", "/bin/bash", NULL);

    return 0;
}
```

### Namespace Flags

```c
// Namespace type flags for clone() and unshare()
CLONE_NEWPID     // PID namespace
CLONE_NEWNET     // Network namespace
CLONE_NEWNS      // Mount namespace
CLONE_NEWUTS     // UTS namespace
CLONE_NEWIPC     // IPC namespace
CLONE_NEWUSER    // User namespace
CLONE_NEWCGROUP  // Cgroup namespace
CLONE_NEWTIME    // Time namespace (Linux 5.6+)

// Example: Create multiple namespaces
int flags = CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWNS |
            CLONE_NEWUTS | CLONE_NEWIPC;
clone(child_fn, stack, flags | SIGCHLD, NULL);
```

### Go Implementation

```go
package main

import (
    "fmt"
    "os"
    "os/exec"
    "syscall"
)

func main() {
    cmd := exec.Command("/bin/bash")

    // Set namespace flags
    cmd.SysProcAttr = &syscall.SysProcAttr{
        Cloneflags: syscall.CLONE_NEWUTS |
                    syscall.CLONE_NEWPID |
                    syscall.CLONE_NEWNS |
                    syscall.CLONE_NEWNET |
                    syscall.CLONE_NEWIPC,
    }

    cmd.Stdin = os.Stdin
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr

    if err := cmd.Run(); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
```

### Python Implementation

```python
import os
import subprocess

# Namespace constants
CLONE_NEWPID = 0x20000000
CLONE_NEWNET = 0x40000000
CLONE_NEWNS = 0x00020000
CLONE_NEWUTS = 0x04000000
CLONE_NEWIPC = 0x08000000
CLONE_NEWUSER = 0x10000000

def create_namespace():
    # Unshare to create new namespaces
    try:
        # Python doesn't have direct unshare binding
        # Use subprocess instead
        subprocess.run([
            'unshare',
            '--pid', '--net', '--mount', '--uts', '--ipc',
            '--fork',
            '/bin/bash'
        ])
    except Exception as e:
        print(f"Error: {e}")

# Using ctypes for direct syscall
import ctypes
import ctypes.util

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

def unshare(flags):
    if libc.unshare(flags) == -1:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))

# Create UTS namespace
unshare(CLONE_NEWUTS)
os.system('hostname mycontainer')
os.system('hostname')
```

## Common Patterns and Use Cases

### Container Creation Pattern

```bash
#!/bin/bash
# Simple container creation script

# Configuration
CONTAINER_NAME="mycontainer"
CONTAINER_ROOT="/var/lib/containers/$CONTAINER_NAME"
BRIDGE="br0"
VETH_HOST="veth0_$CONTAINER_NAME"
VETH_CONTAINER="veth1"

# Create container root
mkdir -p "$CONTAINER_ROOT"

# Create minimal rootfs (example with debootstrap)
# debootstrap --arch=amd64 stable "$CONTAINER_ROOT" http://deb.debian.org/debian/

# Create network namespace
ip netns add "$CONTAINER_NAME"

# Create veth pair
ip link add "$VETH_HOST" type veth peer name "$VETH_CONTAINER"

# Move veth to namespace
ip link set "$VETH_CONTAINER" netns "$CONTAINER_NAME"

# Configure host veth
ip link set "$VETH_HOST" up
ip link set "$VETH_HOST" master "$BRIDGE"

# Configure container veth
ip netns exec "$CONTAINER_NAME" ip link set "$VETH_CONTAINER" up
ip netns exec "$CONTAINER_NAME" ip link set lo up
ip netns exec "$CONTAINER_NAME" ip addr add 192.168.1.100/24 dev "$VETH_CONTAINER"
ip netns exec "$CONTAINER_NAME" ip route add default via 192.168.1.1

# Start container process with namespaces
unshare --pid --mount --uts --ipc --fork \
    --mount-proc="$CONTAINER_ROOT/proc" \
    chroot "$CONTAINER_ROOT" /bin/bash
```

### Network Namespace Bridge Setup

```bash
#!/bin/bash
# Setup bridge for container networking

BRIDGE="br0"
BRIDGE_IP="192.168.1.1/24"

# Create bridge
ip link add name "$BRIDGE" type bridge
ip link set "$BRIDGE" up
ip addr add "$BRIDGE_IP" dev "$BRIDGE"

# Enable IP forwarding
sysctl -w net.ipv4.ip_forward=1

# Setup NAT
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -j MASQUERADE
iptables -A FORWARD -i "$BRIDGE" -j ACCEPT
iptables -A FORWARD -o "$BRIDGE" -j ACCEPT

# Function to add container to bridge
add_container() {
    local ns_name=$1
    local container_ip=$2
    local veth_host="veth_${ns_name}_host"
    local veth_container="veth_${ns_name}_cont"

    # Create veth pair
    ip link add "$veth_host" type veth peer name "$veth_container"

    # Add host veth to bridge
    ip link set "$veth_host" up
    ip link set "$veth_host" master "$BRIDGE"

    # Move container veth to namespace
    ip link set "$veth_container" netns "$ns_name"

    # Configure container interface
    ip netns exec "$ns_name" ip link set "$veth_container" up
    ip netns exec "$ns_name" ip link set lo up
    ip netns exec "$ns_name" ip addr add "$container_ip" dev "$veth_container"
    ip netns exec "$ns_name" ip route add default via "${BRIDGE_IP%/*}"
}

# Example usage
# ip netns add container1
# add_container container1 192.168.1.10/24
```

### PID Namespace with Init Process

```bash
#!/bin/bash
# Container with proper init process

cleanup() {
    # Reap zombie processes
    while true; do
        wait -n 2>/dev/null || break
    done
}

trap cleanup SIGCHLD

# Become PID 1 in namespace
if [ $$ -eq 1 ]; then
    echo "Running as PID 1 in namespace"

    # Mount necessary filesystems
    mount -t proc proc /proc
    mount -t sysfs sys /sys
    mount -t tmpfs tmpfs /tmp

    # Start services here
    # ...

    # Keep running as init
    while true; do
        sleep 1
    done
else
    # Create namespace
    exec unshare --pid --fork --mount-proc "$0" "$@"
fi
```

### Rootless Containers with User Namespaces

```bash
#!/bin/bash
# Rootless container using user namespace

CONTAINER_ROOT="/tmp/rootless-container"

# Create rootfs
mkdir -p "$CONTAINER_ROOT"/{bin,lib,lib64,proc,sys,dev,etc}

# Copy minimal binaries
cp -v /bin/bash "$CONTAINER_ROOT/bin/"
cp -v /bin/ls "$CONTAINER_ROOT/bin/"
cp -v /bin/cat "$CONTAINER_ROOT/bin/"

# Copy required libraries
for lib in $(ldd /bin/bash | awk '{print $3}'); do
    if [ -f "$lib" ]; then
        cp -v "$lib" "$CONTAINER_ROOT/lib/"
    fi
done

# Create user namespace with UID mapping
unshare --user --map-root-user \
        --pid --fork --mount-proc \
        --mount --uts --ipc \
    bash -c "
        # Now running as 'root' in namespace
        hostname rootless-container

        # Setup minimal /dev
        mount -t tmpfs tmpfs '$CONTAINER_ROOT/dev'
        mknod -m 666 '$CONTAINER_ROOT/dev/null' c 1 3
        mknod -m 666 '$CONTAINER_ROOT/dev/zero' c 1 5
        mknod -m 666 '$CONTAINER_ROOT/dev/random' c 1 8
        mknod -m 666 '$CONTAINER_ROOT/dev/urandom' c 1 9

        # Chroot into container
        chroot '$CONTAINER_ROOT' /bin/bash
    "
```

### Isolated Build Environment

```bash
#!/bin/bash
# Isolated build environment using namespaces

PROJECT_DIR="$1"
BUILD_DIR="/tmp/build-$$"

if [ -z "$PROJECT_DIR" ]; then
    echo "Usage: $0 <project-directory>"
    exit 1
fi

# Create isolated environment
unshare --mount --pid --fork --uts --ipc --net \
    --mount-proc \
    bash -c "
        set -e

        # Set hostname
        hostname build-env

        # Create build directory
        mkdir -p '$BUILD_DIR'
        cd '$BUILD_DIR'

        # Mount project as read-only
        mount --bind -o ro '$PROJECT_DIR' '$BUILD_DIR/src'

        # Setup network (optional)
        ip link set lo up

        # Run build
        cd '$BUILD_DIR/src'
        make clean
        make all

        # Copy artifacts out before namespace cleanup
        cp -r build/ '$PROJECT_DIR/dist/'

        echo 'Build complete'
    "

# Cleanup
rm -rf "$BUILD_DIR"
```

### Testing Multiple Network Configurations

```bash
#!/bin/bash
# Test network configurations in isolated namespaces

test_network_config() {
    local test_name=$1
    local config_script=$2

    # Create temporary namespace
    local ns="test-$(date +%s)-$$"
    ip netns add "$ns"

    # Run test in namespace
    ip netns exec "$ns" bash -c "
        set -e

        # Setup loopback
        ip link set lo up

        # Run configuration
        $config_script

        # Run tests
        echo 'Testing configuration: $test_name'
        ip addr show
        ip route show

        # Test connectivity
        if ping -c 1 -W 1 8.8.8.8 &>/dev/null; then
            echo 'Internet connectivity: OK'
        else
            echo 'Internet connectivity: FAILED'
        fi
    "

    # Cleanup
    ip netns delete "$ns"
}

# Example test
test_network_config "Basic setup" "
    ip link add dummy0 type dummy
    ip link set dummy0 up
    ip addr add 10.0.0.1/24 dev dummy0
"
```

## Container Runtime Integration

### Docker and Namespaces

```bash
# Inspect Docker container namespaces
docker inspect --format '{{.State.Pid}}' container_name
PID=$(docker inspect --format '{{.State.Pid}}' container_name)

# View container namespaces
ls -la /proc/$PID/ns/

# Enter Docker container namespace
nsenter -t $PID --net --pid --mount --uts --ipc bash

# View namespace IDs
readlink /proc/$PID/ns/net
readlink /proc/$PID/ns/pid
readlink /proc/$PID/ns/mnt

# Share namespace between containers
docker run --net=container:container1 --name container2 image

# Use host namespace
docker run --pid=host --net=host image
```

### Understanding /proc/PID/ns

```bash
# List process namespaces
ls -la /proc/$$/ns/

# Output format
# lrwxrwxrwx 1 user user 0 Nov 14 12:00 net -> 'net:[4026531992]'

# Namespace types and their files
# cgroup -> 'cgroup:[inode]'
# ipc    -> 'ipc:[inode]'
# mnt    -> 'mnt:[inode]'
# net    -> 'net:[inode]'
# pid    -> 'pid:[inode]'
# pid_for_children -> 'pid:[inode]'
# time   -> 'time:[inode]'
# time_for_children -> 'time:[inode]'
# user   -> 'user:[inode]'
# uts    -> 'uts:[inode]'

# Keep namespace alive
touch /var/run/netns/myns
mount --bind /proc/$$/ns/net /var/run/netns/myns

# List persistent network namespaces
ip netns list

# Delete persistent namespace
umount /var/run/netns/myns
rm /var/run/netns/myns
```

## Security Considerations

### Namespace Security

```bash
# User namespace security
# Allows unprivileged users to create other namespace types
# Check if user namespaces are enabled
cat /proc/sys/kernel/unprivileged_userns_clone

# Disable user namespaces (security vs functionality tradeoff)
sudo sysctl -w kernel.unprivileged_userns_clone=0

# Limit number of user namespaces
sudo sysctl -w user.max_user_namespaces=0

# Security best practices
# 1. Use user namespaces for unprivileged containers
# 2. Combine with seccomp filters
# 3. Use AppArmor/SELinux profiles
# 4. Implement capability dropping
# 5. Use read-only root filesystems
```

### Capability Management

```c
// Drop capabilities in namespace
#define _GNU_SOURCE
#include <sys/capability.h>
#include <sys/prctl.h>

void drop_capabilities() {
    cap_t caps;

    // Get current capabilities
    caps = cap_get_proc();

    // Clear all capabilities
    cap_clear(caps);

    // Set capabilities
    cap_set_proc(caps);

    // Prevent gaining capabilities
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    cap_free(caps);
}
```

### Seccomp Integration

```c
// Restrict system calls with seccomp
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <sys/prctl.h>

void setup_seccomp() {
    struct sock_filter filter[] = {
        // Allow specific syscalls
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 offsetof(struct seccomp_data, nr)),
        // Add syscall filtering rules
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),
    };

    struct sock_fprog prog = {
        .len = sizeof(filter) / sizeof(filter[0]),
        .filter = filter,
    };

    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);
}
```

## Advanced Patterns

### Nested Namespaces

```bash
#!/bin/bash
# Create nested PID namespaces

echo "Level 0 (host): PID=$$"

unshare --pid --fork --mount-proc bash -c '
    echo "Level 1: PID=$$"
    ps aux

    unshare --pid --fork --mount-proc bash -c "
        echo \"Level 2: PID=\$\$\"
        ps aux

        # Each level has its own PID namespace
        # PID 1 at each level
    "
'
```

### Namespace Persistence

```bash
#!/bin/bash
# Persist namespace without running process

create_persistent_ns() {
    local ns_name=$1
    local ns_type=$2  # net, mnt, pid, etc.

    # Create directory for persistent namespaces
    mkdir -p /var/run/netns

    # Create and persist namespace
    case $ns_type in
        net)
            ip netns add "$ns_name"
            ;;
        *)
            # For non-network namespaces
            local ns_path="/var/run/netns/$ns_name"
            touch "$ns_path"

            # Create namespace and bind mount
            unshare --"$ns_type" /bin/bash -c "
                mount --bind /proc/self/ns/$ns_type '$ns_path'
            " &

            local pid=$!
            sleep 0.1  # Give time for namespace creation

            # Namespace persists even after process exits
            kill $pid 2>/dev/null
            ;;
    esac
}

# Example
create_persistent_ns my_net_ns net
```

### Inter-Namespace Communication

```bash
#!/bin/bash
# Setup communication between namespaces using Unix sockets

NS1="ns1"
NS2="ns2"
SOCK_DIR="/tmp/ns-comm"

mkdir -p "$SOCK_DIR"

# Create namespaces
ip netns add "$NS1"
ip netns add "$NS2"

# Setup network connection
ip link add veth1 type veth peer name veth2
ip link set veth1 netns "$NS1"
ip link set veth2 netns "$NS2"

ip netns exec "$NS1" ip addr add 10.0.0.1/24 dev veth1
ip netns exec "$NS1" ip link set veth1 up
ip netns exec "$NS1" ip link set lo up

ip netns exec "$NS2" ip addr add 10.0.0.2/24 dev veth2
ip netns exec "$NS2" ip link set veth2 up
ip netns exec "$NS2" ip link set lo up

# Start server in NS1
ip netns exec "$NS1" bash -c '
    nc -l 10.0.0.1 8080 &
    echo "Server started in NS1"
' &

sleep 1

# Connect from NS2
ip netns exec "$NS2" bash -c '
    echo "Hello from NS2" | nc 10.0.0.1 8080
'

# Cleanup
ip netns delete "$NS1"
ip netns delete "$NS2"
```

### Resource Monitoring in Namespaces

```bash
#!/bin/bash
# Monitor resource usage per namespace

monitor_namespace() {
    local ns_name=$1

    # Get processes in namespace
    local ns_inode=$(ip netns identify $ns_name)

    # Find all PIDs in namespace
    for pid in /proc/[0-9]*; do
        pid=${pid##*/}
        if [ -e "/proc/$pid/ns/net" ]; then
            local pid_ns=$(readlink "/proc/$pid/ns/net" 2>/dev/null)
            if [[ "$pid_ns" == *"$ns_inode"* ]]; then
                echo "PID $pid in namespace $ns_name"

                # Show CPU and memory
                ps -p "$pid" -o pid,ppid,cmd,%cpu,%mem,rss
            fi
        fi
    done
}

# Example
monitor_namespace myns
```

## Troubleshooting

### Common Issues

```bash
# Permission denied errors
# Solution: Use sudo or setup user namespace
sudo unshare --pid --fork --mount-proc bash
# Or
unshare --user --map-root-user --pid --fork bash

# Cannot open /proc/self/uid_map
# Solution: Write uid_map before gid_map, disable setgroups
echo "deny" > /proc/self/setgroups
echo "0 1000 1" > /proc/self/uid_map
echo "0 1000 1" > /proc/self/gid_map

# Network namespace cleanup
# Orphaned namespaces
ip netns delete namespace_name
# If that fails, find and kill processes
ip netns pids namespace_name
kill $(ip netns pids namespace_name)

# Mount namespace issues
# Can't unmount in namespace
mount --make-rprivate /
umount /mnt/point

# Device or resource busy
# Check for processes using mount
lsof | grep /mnt/point
fuser -m /mnt/point

# PID namespace - zombie processes
# Ensure PID 1 reaps children
trap 'wait' CHLD
```

### Debugging Commands

```bash
# Check namespace of process
ls -la /proc/$PID/ns/
lsns -p $PID

# Find processes in namespace
lsns -t net | grep namespace_id
ip netns pids namespace_name

# Compare namespaces
diff <(ls -la /proc/$PID1/ns/) <(ls -la /proc/$PID2/ns/)

# Verify namespace isolation
# In namespace
cat /proc/self/ns/net
readlink /proc/self/ns/pid

# Check UID mapping
cat /proc/self/uid_map
cat /proc/self/gid_map

# Network namespace debugging
ip netns exec ns1 ip addr
ip netns exec ns1 ip route
ip netns exec ns1 iptables -L
ip netns exec ns1 ss -tulpn

# Test namespace connectivity
ip netns exec ns1 ping ns2_ip
ip netns exec ns1 traceroute ns2_ip

# View cgroup namespace
cat /proc/self/cgroup
ls /sys/fs/cgroup/

# Kernel namespace limits
cat /proc/sys/user/max_user_namespaces
cat /proc/sys/user/max_pid_namespaces
cat /proc/sys/user/max_net_namespaces
```

## Best Practices

### Design Principles

```bash
# 1. Minimize privileges
# Use user namespaces and drop capabilities
unshare --user --map-root-user \
        --pid --net --mount --uts --ipc \
        --fork bash

# 2. Proper cleanup
# Always cleanup namespaces and resources
cleanup() {
    ip netns delete "$NS_NAME" 2>/dev/null
    umount "$MOUNT_POINT" 2>/dev/null
}
trap cleanup EXIT

# 3. Use appropriate namespace types
# Only use namespaces you need
# Example: web service might only need net and pid
unshare --net --pid --fork service

# 4. Implement proper init process
# PID 1 must reap zombie processes
if [ $$ -eq 1 ]; then
    trap 'wait' CHLD
fi

# 5. Set resource limits
# Combine with cgroups
# cgcreate -g memory,cpu:mycontainer
# cgset -r memory.limit_in_bytes=512M mycontainer
# cgset -r cpu.shares=512 mycontainer
# cgexec -g memory,cpu:mycontainer unshare --pid --fork bash

# 6. Secure mount propagation
# Use private propagation by default
mount --make-rprivate /

# 7. Implement health checks
# Monitor namespace processes
check_health() {
    ip netns pids "$NS_NAME" | wc -l
}

# 8. Log namespace events
# Track creation and deletion
logger -t namespace "Created namespace: $NS_NAME"

# 9. Use descriptive names
# Name namespaces logically
NS_NAME="web-app-prod-01"

# 10. Document dependencies
# Track which namespaces depend on others
```

### Performance Considerations

```bash
# Namespace creation overhead
# Reuse namespaces when possible
# Cache namespace references

# Network namespace performance
# Use veth pairs with minimal overhead
# Consider macvlan for better performance
ip link add macvlan0 link eth0 type macvlan mode bridge
ip link set macvlan0 netns myns

# Mount namespace efficiency
# Use shared subtrees for common mounts
mount --make-shared /media

# PID namespace optimization
# Minimize process count in namespace
# Use proper init to prevent zombie accumulation

# Benchmark namespace operations
time unshare --net --pid --fork true
time ip netns add test && ip netns delete test
```

## Quick Reference

### Namespace Types
| Type | Isolates | Common Use |
|------|----------|------------|
| PID | Process IDs | Process isolation |
| NET | Network stack | Network isolation |
| MNT | Mount points | Filesystem isolation |
| UTS | Hostname | Container identity |
| IPC | IPC objects | IPC isolation |
| USER | UIDs/GIDs | Privilege separation |
| CGROUP | Cgroup view | Resource limits |
| TIME | System time | Time virtualization |

### Common Commands
| Command | Description |
|---------|-------------|
| `unshare` | Create new namespaces |
| `nsenter` | Enter existing namespace |
| `ip netns` | Manage network namespaces |
| `lsns` | List namespaces |
| `clone()` | Create process with namespaces |
| `setns()` | Join namespace |
| `unshare()` | Leave namespace |

### System Call Flags
| Flag | Namespace Type |
|------|----------------|
| `CLONE_NEWPID` | PID namespace |
| `CLONE_NEWNET` | Network namespace |
| `CLONE_NEWNS` | Mount namespace |
| `CLONE_NEWUTS` | UTS namespace |
| `CLONE_NEWIPC` | IPC namespace |
| `CLONE_NEWUSER` | User namespace |
| `CLONE_NEWCGROUP` | Cgroup namespace |
| `CLONE_NEWTIME` | Time namespace |

Linux namespaces provide powerful isolation mechanisms that form the foundation of modern containerization, enabling secure multi-tenancy, resource partitioning, and lightweight virtualization for diverse use cases from development environments to production container orchestration.
