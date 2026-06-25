# Container Runtimes

## Overview

A "container" isn't a kernel object — it's a **composition of existing kernel primitives**:
[namespaces](namespace.md) for isolation, [cgroups](cgroups.md) for resource limits, capabilities
+ seccomp + [SELinux](selinux.md)/AppArmor for confinement, and an [overlayfs](filesystems.md)
union mount for the root filesystem. This page shows how the runtimes — **runc**, containerd,
Docker, and Podman — stack those primitives, and the **OCI** specs that standardise them. The CPU/
memory limits flow through the [CPU Scheduler](scheduler.md) and cgroup controllers; the ability
to ship a minimal "scratch" image ties back to static linking in
[ELF, Linking & Loading](elf_linking.md).

```
   ┌──────────── what a container actually is ────────────┐
   │ namespaces  → isolated view (pid, net, mnt, uts, ipc, │
   │               user, cgroup): "I am pid 1, my own net" │
   │ cgroups     → bounded resources (cpu.max, memory.max) │
   │ capabilities→ drop privileges (no CAP_SYS_ADMIN, …)   │
   │ seccomp     → restrict the syscall surface            │
   │ LSM         → SELinux/AppArmor mandatory access control│
   │ overlayfs   → layered, copy-on-write root filesystem  │
   └───────────────────────────────────────────────────────┘
```

## The OCI specifications

The Open Container Initiative decouples the pieces so tools interoperate:

- **OCI Image spec** — what an image *is*: a stack of filesystem layer tarballs + a JSON config
  (env, entrypoint, etc.), addressed by content digest in a registry.
- **OCI Runtime spec** — what a runtime consumes: a **bundle** = an unpacked root filesystem +
  `config.json` describing namespaces, cgroup limits, mounts, capabilities, and the seccomp
  profile.
- **CRI** (Container Runtime Interface) — the gRPC API Kubernetes uses to talk to a runtime
  (containerd, CRI-O).

```
   registry ──pull──► image (layers + config)
                         │ unpack + generate config.json
                         ▼
                    OCI bundle  ──►  runc  ──►  running container
```

## The runtime stack

```
   docker / podman CLI         (build images, manage networks/volumes, UX)
        │
   containerd / (podman: direct)   (image pull, snapshotting, lifecycle)
        │  creates an OCI bundle, invokes...
        ▼
   runc            (the low-level runtime: actually does the clone+setup)
        │  clone(CLONE_NEW*) + setns + cgroup setup + pivot_root +
        │  drop caps + apply seccomp + exec entrypoint
        ▼
   your process, now "in a container" (just a confined Linux process)
```

- **runc** — the reference low-level OCI runtime. Given a bundle, it does the actual
  `clone()`/`unshare()` into new namespaces, joins the cgroup, sets up the overlay root via
  `pivot_root`, drops capabilities, installs the seccomp filter, and `exec`s the entrypoint. Then
  it gets out of the way. (Alternatives: **crun** in C, **gVisor**/**Kata** for stronger
  isolation via a user-space kernel or a lightweight VM.)
- **containerd** — the daemon that manages images, snapshots, and the container lifecycle, calling
  runc per container. Used standalone and as the default Kubernetes CRI runtime.
- **Docker** — `dockerd` + CLI on top of containerd; adds image build (`Dockerfile`), networking,
  and volumes. Daemon-centric, root by default.
- **Podman** — daemonless, fork/exec model, **rootless** by default (uses **user namespaces** so a
  normal user maps to a fake root inside). Largely Docker-CLI-compatible.

## How isolation is built (the namespaces)

| Namespace | Isolates | In a container |
|-----------|----------|----------------|
| **mnt**   | mount table | private rootfs (overlay) + masked `/proc`, `/sys` |
| **pid**   | process IDs | entrypoint is PID 1; can't see host processes |
| **net**   | net devices/stack | own loopback + veth pair to a host bridge |
| **uts**   | hostname/domain | container hostname |
| **ipc**   | SysV IPC / mqueues | private IPC |
| **user**  | UID/GID mapping | rootless: host UID 1000 ↔ container UID 0 |
| **cgroup**| cgroup root view | container sees its own cgroup tree |

See [Namespace](namespace.md) for the underlying syscalls (`clone`, `unshare`, `setns`).

## The root filesystem: overlayfs

Images are layers; the runtime stacks them read-only and adds one writable layer on top with
**overlayfs**, so containers share base layers and only diverge on write (copy-up):

```
   upperdir   (rw)   ← container's writes land here
   ───────────────
   lowerdir   (ro)   ← image layer 3
   lowerdir   (ro)   ← image layer 2  (shared across containers)
   lowerdir   (ro)   ← base layer
   = merged mount → the container's /
```

This is why pulling a second image sharing a base is nearly free, and why container writes don't
persist unless mapped to a **volume** (a bind mount escaping the overlay).

## Build a container by hand

The primitives are directly usable — useful for understanding and debugging:

```bash
# new namespaces with a mapped root (rootless-style), in a chroot tree
unshare --user --map-root-user --pid --mount --net --uts --ipc --fork \
        chroot ./rootfs /bin/sh

# enter an existing container's namespaces (debugging)
nsenter -t <pid> -a /bin/sh

# bound resources via cgroup v2
mkdir /sys/fs/cgroup/demo
echo "100000 100000" > /sys/fs/cgroup/demo/cpu.max   # 1 CPU
echo $$ > /sys/fs/cgroup/demo/cgroup.procs
```

That `unshare`/`chroot`/cgroup trio *is* the skeleton of what runc automates.

## Confinement layers

Beyond namespaces, runtimes harden the container:

- **Capabilities** — drop the bounding set to a small default (no `CAP_SYS_ADMIN`,
  `CAP_NET_ADMIN`, etc.), so "root" inside is heavily declawed.
- **seccomp-bpf** — a syscall filter (the default Docker/Podman profile blocks ~tens of dangerous
  syscalls) that returns `EPERM`/kills on disallowed calls.
- **LSM** — [SELinux](selinux.md) (label-based, `container_t`) or AppArmor (path-based) enforce
  mandatory access control on top.
- **User namespaces** — map container-root to an unprivileged host UID so a container escape isn't
  host root; the foundation of rootless containers.

## Where this connects

- [Namespace](namespace.md) — the isolation primitives (`clone`/`unshare`/`setns`) every runtime
  is built on.
- [Control Groups (cgroups)](cgroups.md) — `cpu.max`/`memory.max`/`io.max` enforce container
  resource limits; throttling shows up here.
- [SELinux](selinux.md) — mandatory access control labels (`container_t`) confining containers.
- [Filesystems](filesystems.md) — overlayfs union mounts back the layered image rootfs.
- [CPU Scheduler](scheduler.md) — the cgroup cpu controller decides how container CPU limits are
  scheduled (and where throttling stalls come from).
- [ELF, Linking & Loading](elf_linking.md) — static linking is why minimal "scratch"/distroless
  images run without a full userland.

## Pitfalls

- **Running as root inside the container.** Container root + a kernel/runtime bug = host root.
  Drop capabilities, use a non-root `USER`, and prefer **user-namespace/rootless** containers.
- **`--privileged` as a fix.** It disables seccomp, grants all capabilities, and exposes host
  devices — effectively no isolation. Grant the specific capability/device instead.
- **Assuming containers are a security boundary like VMs.** They share the host kernel; a kernel
  vuln crosses the boundary. For hostile multi-tenancy use gVisor/Kata or VMs.
- **CPU/memory limits not behaving.** `memory.max` OOM-kills inside the container; `cpu.max`
  throttles (periodic latency spikes) — both are cgroup behaviour, debug via `memory.events` /
  `cpu.stat` (see [cgroups](cgroups.md)).
- **Writing data to the overlay upperdir.** It's ephemeral — lost when the container is removed.
  Persist with a volume/bind mount.
- **cgroup v1 vs v2 mismatch.** Older tooling expecting v1 hierarchies misbehaves on v2-only
  ("unified") hosts; check `stat -fc %T /sys/fs/cgroup` (`cgroup2fs`).
- **PID 1 signal/zombie handling.** The entrypoint is PID 1 and won't reap zombies or forward
  signals unless written to, or run under an init shim (`--init`/tini).
