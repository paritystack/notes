# Linux Virtualization

> **Domain:** Linux Kernel, Cloud Computing, Systems
> **Key Concepts:** Hypervisor, KVM, QEMU, VirtIO, Namespaces vs. VMs

**Virtualization** is the process of creating a software-based (or "virtual") representation of something, such as virtual applications, servers, storage, and networks. In the context of Linux, it typically refers to running multiple operating systems (Guests) on a single physical host.

---

## 1. Containers vs. Virtual Machines

*   **Containers (Docker/LXC):**
    *   *Virtualize:* The Userspace.
    *   *Kernel:* Shared with the Host.
    *   *Isolation:* Weak (Namespaces & Cgroups). A kernel panic crashes everything.
    *   *Speed:* Native.
*   **Virtual Machines (VMs):**
    *   *Virtualize:* The Hardware.
    *   *Kernel:* Each VM has its own Kernel.
    *   *Isolation:* Strong.
    *   *Speed:* Near-native (with hardware acceleration).

---

## 2. Hypervisor Types

The **Hypervisor** (or VMM - Virtual Machine Monitor) creates and runs VMs.

### 2.1. Type 1 (Bare Metal)
Runs directly on hardware. The "Host OS" is effectively the hypervisor itself.
*   *Examples:* VMware ESXi, Xen, Microsoft Hyper-V.
*   *Use Case:* Enterprise Datacenters.

### 2.2. Type 2 (Hosted)
Runs as an application inside a normal OS.
*   *Examples:* VirtualBox, VMware Workstation.
*   *Use Case:* Desktop testing.

### 2.3. The Linux Hybrid (KVM)
Linux blurs the line. **KVM (Kernel-based Virtual Machine)** turns the Linux Kernel *into* a Type 1 hypervisor.
*   When you load the `kvm.ko` module, Linux can execute Guest code directly on the CPU (using Intel VT-x or AMD-V extensions) while still running standard Linux processes (Firefox, Vim).

---

## 3. The KVM + QEMU Stack

KVM needs a userspace tool to emulate hardware (Disk, Network, Mouse). That tool is usually **QEMU**.

1.  **KVM (Kernel Space):**
    *   Handles CPU scheduling and Memory management for the Guest.
    *   Uses hardware extensions (VMX/SVM) to let the Guest CPU instructions run at native speed.
    *   Catches "privileged" instructions (like accessing hardware) and pauses the Guest.
2.  **QEMU (User Space):**
    *   When KVM pauses the Guest (VM Exit), control returns to QEMU.
    *   QEMU emulates the hardware request (e.g., "Write to Disk" -> Write to a file `disk.qcow2`).
    *   QEMU tells KVM to resume the Guest (VM Entry).

---

## 4. VirtIO (Paravirtualization)

Emulating a physical Realtek Network Card is slow. The Guest has to write to "registers," QEMU catches traps, etc.

**VirtIO** is a standard for "enlightened" drivers. The Guest OS *knows* it is virtualized.
*   Instead of talking to a fake network card, the Guest uses a `virtio-net` driver.
*   This driver writes data directly to a shared memory ring buffer that the Host can read.
*   **Result:** Performance is almost identical to bare metal.

---

## 5. Implementation Commands

**Check Support:**
```bash
grep -E 'vmx|svm' /proc/cpuinfo
# vmx = Intel, svm = AMD
```

**Creating a VM (Libvirt/Virsh):**
Most people don't run raw `qemu-system-x86`. They use **Libvirt**, which manages QEMU processes.
```bash
# Create a VM
virt-install \
  --name ubuntu-guest \
  --memory 2048 \
  --vcpus 2 \
  --disk size=10 \
  --cdrom ubuntu-22.04.iso \
  --os-variant ubuntu22.04
```

**Management:**
```bash
virsh list --all
virsh start ubuntu-guest
virsh console ubuntu-guest
```

---

## 6. Advanced Concepts

*   **PCI Passthrough (VFIO):** Give a Guest exclusive access to a physical PCI device (e.g., a GPU).
    *   *Use Case:* Gaming on a Windows VM inside Linux, or AI training inside a VM.
*   **Live Migration:** Moving a running VM from Host A to Host B with zero downtime.
    *   *Mechanism:* Copy RAM pages to Host B. If a page changes while copying, mark it "dirty" and copy again. Once dirty rate is low, pause VM, copy final state, resume on Host B.
