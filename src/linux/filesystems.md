# Linux Filesystems

Comprehensive guide to Linux filesystem types, operations, and patterns. Covers traditional filesystems (ext4, XFS, Btrfs, ZFS), special filesystems (procfs, sysfs), and modern container storage with OverlayFS.

## Table of Contents

- [Overview](#overview)
- [Filesystem Types](#filesystem-types)
  - [ext4](#ext4)
  - [XFS](#xfs)
  - [Btrfs](#btrfs)
  - [ZFS](#zfs)
  - [F2FS](#f2fs)
  - [tmpfs and ramfs](#tmpfs-and-ramfs)
  - [FAT/VFAT/exFAT](#fatvfatexfat)
  - [Other Filesystems](#other-filesystems)
- [Virtual and Special Filesystems](#virtual-and-special-filesystems)
  - [procfs](#procfs)
  - [sysfs](#sysfs)
  - [debugfs](#debugfs)
  - [devtmpfs](#devtmpfs)
  - [configfs](#configfs)
  - [cgroup Filesystems](#cgroup-filesystems)
  - [Other Special Filesystems](#other-special-filesystems)
- [OverlayFS Deep Dive](#overlayfs-deep-dive)
  - [Architecture and Concepts](#architecture-and-concepts)
  - [Copy-Up Mechanism](#copy-up-mechanism)
  - [Whiteouts and Opaque Directories](#whiteouts-and-opaque-directories)
  - [Multiple Lower Layers](#multiple-lower-layers)
  - [Container Integration](#container-integration)
  - [Advanced Features](#advanced-features)
  - [Performance and Limitations](#performance-and-limitations)
  - [OverlayFS Troubleshooting](#overlayfs-troubleshooting)
- [Mount Operations](#mount-operations)
  - [Basic Mounting](#basic-mounting)
  - [Mount Options](#mount-options)
  - [fstab Configuration](#fstab-configuration)
  - [Systemd Mount Units](#systemd-mount-units)
  - [Bind Mounts](#bind-mounts)
  - [Mount Propagation](#mount-propagation)
  - [Remounting and Unmounting](#remounting-and-unmounting)
  - [Mount Inspection](#mount-inspection)
- [Filesystem Management](#filesystem-management)
  - [Creating Filesystems](#creating-filesystems)
  - [Checking and Repairing](#checking-and-repairing)
  - [Resizing Filesystems](#resizing-filesystems)
  - [Tuning Parameters](#tuning-parameters)
  - [Labels and UUIDs](#labels-and-uuids)
  - [Monitoring](#monitoring)
- [Performance Considerations](#performance-considerations)
  - [I/O Schedulers](#io-schedulers)
  - [Mount Options for Performance](#mount-options-for-performance)
  - [SSD Optimizations](#ssd-optimizations)
  - [Block Size and Inode Sizing](#block-size-and-inode-sizing)
  - [Journal Tuning](#journal-tuning)
  - [Read-Ahead Tuning](#read-ahead-tuning)
  - [Filesystem-Specific Optimizations](#filesystem-specific-optimizations)
- [Common Patterns](#common-patterns)
  - [Container Storage with OverlayFS](#container-storage-with-overlayfs)
  - [RAM Disk Creation](#ram-disk-creation)
  - [Encrypted Filesystems](#encrypted-filesystems)
  - [LVM Integration](#lvm-integration)
  - [Snapshot Workflows](#snapshot-workflows)
  - [Quota Management](#quota-management)
  - [Extended Attributes and ACLs](#extended-attributes-and-acls)
  - [Network Filesystems](#network-filesystems)
  - [Loop Device Mounting](#loop-device-mounting)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

## Overview

The Linux Virtual Filesystem (VFS) provides a unified interface for interacting with different filesystem types. This abstraction allows applications to work with files and directories without knowing the underlying filesystem implementation.

### VFS Architecture

The VFS layer sits between user-space applications and filesystem implementations, providing a consistent API for file operations:

```
User Applications
       |
   System Calls (open, read, write, etc.)
       |
   Virtual Filesystem (VFS)
       |
   +---+---+---+---+---+---+
   |   |   |   |   |   |   |
  ext4 XFS Btrfs NFS tmpfs overlay
   |   |   |   |   |   |   |
  Block Layer / Network / Memory
```

### Key VFS Concepts

- **Superblock**: Contains filesystem metadata (size, block size, state)
- **Inode**: Represents a file or directory with metadata (permissions, timestamps, size)
- **Dentry**: Directory entry that maps names to inodes
- **File**: Represents an open file descriptor

## Filesystem Types

### ext4

The fourth extended filesystem, ext4, is the default filesystem for most Linux distributions. It's stable, well-tested, and performs well for general-purpose use.

#### Key Features

- **Extents**: More efficient large file storage (replaced block mapping)
- **Journaling**: Metadata and optional data journaling for crash recovery
- **Delayed allocation**: Improved performance and reduced fragmentation
- **Large filesystem support**: Up to 1 EiB volume, 16 TiB files
- **Online defragmentation**: e4defrag for reducing fragmentation
- **Backward compatibility**: Can mount ext2/ext3 filesystems

#### Creating ext4 Filesystems

```bash
# Create with default options
mkfs.ext4 /dev/sdb1

# Create with specific block size (4K is typical)
mkfs.ext4 -b 4096 /dev/sdb1

# Create with label
mkfs.ext4 -L mydata /dev/sdb1

# Create with more inodes (useful for many small files)
mkfs.ext4 -i 8192 /dev/sdb1  # One inode per 8KB

# Create with specific inode size (256 is default, 512 for extended attributes)
mkfs.ext4 -I 512 /dev/sdb1

# Create without reserved blocks (useful for data partitions)
mkfs.ext4 -m 0 /dev/sdb1

# Create with directory indexing (enabled by default, improves lookup)
mkfs.ext4 -O dir_index /dev/sdb1
```

#### Journaling Modes

ext4 supports three journaling modes, balancing safety and performance:

```bash
# Journal mode: data + metadata journaled (slowest, safest)
mount -o data=journal /dev/sdb1 /mnt

# Ordered mode: metadata journaled, data written before commit (default)
mount -o data=ordered /dev/sdb1 /mnt

# Writeback mode: metadata journaled only (fastest, least safe)
mount -o data=writeback /dev/sdb1 /mnt
```

#### Tuning ext4

```bash
# View current filesystem settings
tune2fs -l /dev/sdb1

# Set volume label
tune2fs -L mydata /dev/sdb1

# Adjust reserved block percentage (default 5%)
tune2fs -m 1 /dev/sdb1

# Set mount options in superblock
tune2fs -o journal_data_writeback /dev/sdb1

# Disable last-mounted time updates (reduces writes)
tune2fs -O ^has_journal /dev/sdb1  # Remove journal (convert to ext2)
tune2fs -O has_journal /dev/sdb1   # Add journal back

# Set filesystem check interval
tune2fs -c 30 /dev/sdb1            # Check every 30 mounts
tune2fs -i 6m /dev/sdb1            # Check every 6 months
tune2fs -c 0 -i 0 /dev/sdb1        # Disable periodic checks

# Enable or disable features
tune2fs -O extent /dev/sdb1        # Enable extents
tune2fs -O dir_index /dev/sdb1     # Enable directory indexing
tune2fs -O ^dir_index /dev/sdb1    # Disable feature (^ prefix)
```

#### ext4 Performance Tips

- Use noatime mount option to reduce writes
- Consider data=writeback for workloads where data consistency is less critical
- Use larger block sizes (4K) for large files
- Enable journal checksumming for better reliability

### XFS

XFS is a high-performance 64-bit journaling filesystem originally developed by SGI. It excels with large files and parallel I/O workloads.

#### Key Features

- **Excellent scalability**: Designed for large files and filesystems
- **Parallel I/O**: Multiple threads can perform I/O simultaneously
- **Allocation groups**: Divides filesystem for parallel operations
- **Delayed allocation**: Optimizes data placement
- **Online defragmentation**: Can defragment while mounted
- **Metadata journaling**: Fast crash recovery
- **Project quotas**: Quotas per directory tree
- **No ability to shrink**: Can only grow, not shrink

#### Creating XFS Filesystems

```bash
# Create with default options
mkfs.xfs /dev/sdb1

# Create with label
mkfs.xfs -L mydata /dev/sdb1

# Create with specific block size
mkfs.xfs -b size=4096 /dev/sdb1

# Create with specific allocation group size (for parallelism)
mkfs.xfs -d agcount=8 /dev/sdb1

# Create with specific inode size (512 for extended attributes)
mkfs.xfs -i size=512 /dev/sdb1

# Create optimized for SSDs
mkfs.xfs -d sunit=512,swidth=2048 /dev/sdb1

# Force creation (overwrite existing filesystem)
mkfs.xfs -f /dev/sdb1
```

#### XFS Management

```bash
# View filesystem information
xfs_info /dev/sdb1
# Or for mounted filesystem
xfs_info /mnt/data

# Grow filesystem (online, while mounted)
xfs_growfs /mnt/data

# Defragment filesystem
xfs_fsr /mnt/data               # Defragment entire filesystem
xfs_fsr -v /mnt/data/file.img   # Defragment specific file

# Check filesystem (must be unmounted)
xfs_repair /dev/sdb1

# Check in no-modify mode (read-only check)
xfs_repair -n /dev/sdb1

# Dump/restore metadata
xfs_metadump /dev/sdb1 dump.img
xfs_mdrestore dump.img /dev/sdc1

# Copy filesystem data
xfs_copy /dev/sdb1 /dev/sdc1
```

#### XFS Tuning

```bash
# Set filesystem label
xfs_admin -L mydata /dev/sdb1

# Set UUID
xfs_admin -U generate /dev/sdb1

# Modify parameters (rare, most set at creation)
xfs_admin -l /dev/sdb1  # View label
xfs_admin -u /dev/sdb1  # View UUID

# Mount options for performance
mount -o noatime,nodiratime,logbufs=8,logbsize=256k /dev/sdb1 /mnt
```

#### XFS Performance Tips

- Use allocation groups matching expected parallelism
- Consider larger log buffer size for write-heavy workloads
- Use nobarrier for battery-backed RAID controllers
- Project quotas are faster than user/group quotas

### Btrfs

Btrfs (B-tree Filesystem) is a modern copy-on-write filesystem with advanced features like snapshots, compression, and RAID support built-in.

#### Key Features

- **Copy-on-write (CoW)**: Data is never overwritten in place
- **Snapshots**: Instant, space-efficient snapshots
- **Subvolumes**: Independent file trees within the filesystem
- **Built-in RAID**: RAID 0, 1, 5, 6, 10 without mdadm
- **Compression**: Transparent compression (zlib, lzo, zstd)
- **Checksums**: Data and metadata checksumming for integrity
- **Send/receive**: Efficient incremental backups
- **Online resizing**: Grow and shrink while mounted
- **Deduplication**: Offline and online deduplication support

#### Creating Btrfs Filesystems

```bash
# Create simple filesystem
mkfs.btrfs /dev/sdb1

# Create with label
mkfs.btrfs -L mydata /dev/sdb1

# Create with specific node size (16K default, affects metadata)
mkfs.btrfs -n 32768 /dev/sdb1

# Create RAID1 across multiple devices (metadata and data)
mkfs.btrfs -m raid1 -d raid1 /dev/sdb1 /dev/sdc1

# Create RAID0 for data, RAID1 for metadata
mkfs.btrfs -m raid1 -d raid0 /dev/sdb1 /dev/sdc1

# Create RAID10
mkfs.btrfs -m raid10 -d raid10 /dev/sd[b-e]1

# Force creation
mkfs.btrfs -f /dev/sdb1
```

#### Subvolume Management

```bash
# Create subvolume
btrfs subvolume create /mnt/data/subvol1

# List subvolumes
btrfs subvolume list /mnt/data

# Show subvolume details
btrfs subvolume show /mnt/data/subvol1

# Delete subvolume
btrfs subvolume delete /mnt/data/subvol1

# Set default subvolume (mounted if no subvol= option)
btrfs subvolume set-default <id> /mnt/data

# Get default subvolume
btrfs subvolume get-default /mnt/data

# Mount specific subvolume
mount -o subvol=subvol1 /dev/sdb1 /mnt/subvol1
mount -o subvolid=256 /dev/sdb1 /mnt/subvol1
```

#### Snapshot Operations

```bash
# Create snapshot (read-write)
btrfs subvolume snapshot /mnt/data /mnt/data/snapshots/snap1

# Create read-only snapshot
btrfs subvolume snapshot -r /mnt/data /mnt/data/snapshots/snap1

# List snapshots (snapshots are subvolumes)
btrfs subvolume list -s /mnt/data

# Delete snapshot
btrfs subvolume delete /mnt/data/snapshots/snap1

# Rollback by changing default subvolume
btrfs subvolume set-default <snapshot-id> /mnt/data
# Then reboot or remount
```

#### Compression

```bash
# Enable compression on mount
mount -o compress=zstd /dev/sdb1 /mnt/data
mount -o compress=lzo /dev/sdb1 /mnt/data
mount -o compress=zlib /dev/sdb1 /mnt/data

# Set compression level (zstd supports 1-15)
mount -o compress=zstd:3 /dev/sdb1 /mnt/data

# Enable compression for existing data
btrfs filesystem defragment -r -czstd /mnt/data

# Set compression property on directory
btrfs property set /mnt/data/logs compression zstd

# Check compression property
btrfs property get /mnt/data/logs compression
```

#### Send/Receive for Backups

```bash
# Create initial snapshot
btrfs subvolume snapshot -r /mnt/data /mnt/data/snap1

# Send snapshot to another location
btrfs send /mnt/data/snap1 | btrfs receive /mnt/backup/

# Incremental backup
btrfs subvolume snapshot -r /mnt/data /mnt/data/snap2
btrfs send -p /mnt/data/snap1 /mnt/data/snap2 | btrfs receive /mnt/backup/

# Send over network
btrfs send /mnt/data/snap1 | ssh user@backup 'btrfs receive /backup/'

# Send with compression
btrfs send /mnt/data/snap1 | gzip | ssh user@backup 'gunzip | btrfs receive /backup/'
```

#### Btrfs Device Management

```bash
# Add device to filesystem
btrfs device add /dev/sdc1 /mnt/data

# Remove device
btrfs device remove /dev/sdc1 /mnt/data

# Balance filesystem (redistribute data across devices)
btrfs balance start /mnt/data

# Balance only data, convert to RAID1
btrfs balance start -dconvert=raid1 /mnt/data

# Balance only metadata
btrfs balance start -mconvert=raid1 /mnt/data

# Check balance status
btrfs balance status /mnt/data

# Replace failing device
btrfs replace start /dev/sdb1 /dev/sdd1 /mnt/data

# Scrub filesystem (verify checksums)
btrfs scrub start /mnt/data
btrfs scrub status /mnt/data
```

#### Btrfs Maintenance

```bash
# Check filesystem (unmounted)
btrfs check /dev/sdb1

# Check with repair (dangerous, backup first)
btrfs check --repair /dev/sdb1

# Show filesystem usage
btrfs filesystem usage /mnt/data

# Show device stats
btrfs device stats /mnt/data

# Resize filesystem
btrfs filesystem resize +10G /mnt/data    # Grow by 10GB
btrfs filesystem resize -5G /mnt/data     # Shrink by 5GB
btrfs filesystem resize max /mnt/data     # Grow to device size

# Defragment
btrfs filesystem defragment -r /mnt/data
```

### ZFS

ZFS is an advanced filesystem and volume manager originally developed by Sun Microsystems. On Linux, it's available through OpenZFS.

#### Key Features

- **Pooled storage**: Combines volume management and filesystem
- **Copy-on-write**: Data integrity and snapshots
- **Snapshots and clones**: Instant, space-efficient
- **RAID-Z**: Software RAID with better characteristics than traditional RAID5/6
- **Compression**: Built-in transparent compression
- **Deduplication**: Block-level deduplication
- **ARC/L2ARC**: Sophisticated caching with RAM and SSD
- **Send/receive**: Efficient replication and backups
- **Self-healing**: Automatic data corruption repair with redundancy

#### Installing ZFS on Linux

```bash
# Ubuntu/Debian
apt install zfsutils-linux

# RHEL/CentOS/Fedora
dnf install zfs

# Arch Linux
pacman -S zfs-linux

# Load kernel module
modprobe zfs
```

#### Creating ZFS Pools

```bash
# Create simple pool
zpool create mypool /dev/sdb

# Create mirror pool (RAID1)
zpool create mypool mirror /dev/sdb /dev/sdc

# Create RAID-Z pool (similar to RAID5, single parity)
zpool create mypool raidz /dev/sd[b-e]

# Create RAID-Z2 pool (similar to RAID6, double parity)
zpool create mypool raidz2 /dev/sd[b-f]

# Create RAID-Z3 pool (triple parity)
zpool create mypool raidz3 /dev/sd[b-g]

# Create with specific mount point
zpool create -m /data mypool /dev/sdb

# Create with specific ashift (sector size: 9=512B, 12=4K, 13=8K)
zpool create -o ashift=12 mypool /dev/sdb

# Add cache device (L2ARC)
zpool add mypool cache /dev/sdf

# Add log device (SLOG/ZIL)
zpool add mypool log mirror /dev/sdg /dev/sdh
```

#### ZFS Pool Management

```bash
# List pools
zpool list

# Show pool status
zpool status mypool

# Show detailed I/O statistics
zpool iostat mypool 1  # Update every second

# Add device to pool
zpool add mypool /dev/sdf

# Replace device
zpool replace mypool /dev/sdb /dev/sdg

# Remove device (only cache/log devices)
zpool remove mypool /dev/sdf

# Scrub pool (verify all data)
zpool scrub mypool

# Stop scrub
zpool scrub -s mypool

# Export pool (unmount, prepare for import elsewhere)
zpool export mypool

# Import pool
zpool import mypool

# Import pool with different name
zpool import oldname newname

# Import pool that was last used on another system
zpool import -f mypool

# Upgrade pool to latest features
zpool upgrade mypool
```

#### ZFS Dataset Management

```bash
# Create dataset (filesystem)
zfs create mypool/data

# Create dataset with specific mount point
zfs create -o mountpoint=/data mypool/data

# List datasets
zfs list

# Show detailed properties
zfs get all mypool/data

# Set properties
zfs set compression=lz4 mypool/data
zfs set recordsize=1M mypool/data           # Block size
zfs set atime=off mypool/data
zfs set quota=100G mypool/data
zfs set reservation=50G mypool/data

# Create dataset with properties
zfs create -o compression=lz4 -o mountpoint=/data mypool/data

# Destroy dataset
zfs destroy mypool/data

# Rename dataset
zfs rename mypool/data mypool/newdata
```

#### ZFS Snapshots

```bash
# Create snapshot
zfs snapshot mypool/data@snap1

# Create recursive snapshot (all child datasets)
zfs snapshot -r mypool/data@snap1

# List snapshots
zfs list -t snapshot

# Rollback to snapshot (destroys newer data)
zfs rollback mypool/data@snap1

# Clone snapshot (create writable copy)
zfs clone mypool/data@snap1 mypool/clone

# Promote clone (make it independent)
zfs promote mypool/clone

# Destroy snapshot
zfs destroy mypool/data@snap1

# Destroy all snapshots for dataset
zfs destroy -r mypool/data
```

#### ZFS Send/Receive

```bash
# Send full snapshot
zfs send mypool/data@snap1 > /backup/snap1.zfs

# Receive snapshot
zfs receive mypool/restore < /backup/snap1.zfs

# Send incremental snapshot
zfs send -i mypool/data@snap1 mypool/data@snap2 > /backup/incremental.zfs

# Send over network
zfs send mypool/data@snap1 | ssh user@backup 'zfs receive backuppool/data'

# Send with compression
zfs send mypool/data@snap1 | gzip | ssh user@backup 'gunzip | zfs receive backuppool/data'

# Send recursively (all child datasets)
zfs send -R mypool/data@snap1 | zfs receive backuppool/data
```

#### ZFS Performance Tuning

```bash
# Set recordsize for large sequential I/O
zfs set recordsize=1M mypool/data

# Set recordsize for small random I/O
zfs set recordsize=8K mypool/data

# Enable compression (lz4 is fast and efficient)
zfs set compression=lz4 mypool/data

# Disable atime updates
zfs set atime=off mypool/data

# Set ARC cache size (in /etc/modprobe.d/zfs.conf)
# options zfs zfs_arc_max=8589934592  # 8GB

# Enable deduplication (very RAM intensive, usually not recommended)
zfs set dedup=on mypool/data

# Set sync behavior
zfs set sync=standard mypool/data  # Default
zfs set sync=always mypool/data    # Slower, safer
zfs set sync=disabled mypool/data  # Faster, dangerous
```

### F2FS

F2FS (Flash-Friendly File System) is optimized for flash storage devices like SSDs, eMMC, and SD cards. It's designed with flash characteristics in mind, such as wear-leveling and write amplification.

#### Key Features

- **Flash-optimized**: Designed for NAND flash characteristics
- **Log-structured**: Reduces write amplification
- **Multi-head logging**: Multiple active logs for different data temperatures
- **Adaptive logging**: Switches between threaded and normal logging
- **Inline data**: Small files stored in inode
- **Data compression**: Transparent compression support

#### Creating F2FS

```bash
# Create F2FS filesystem
mkfs.f2fs /dev/sdb1

# Create with label
mkfs.f2fs -l mydata /dev/sdb1

# Create with specific overprovision ratio (extra space for GC)
mkfs.f2fs -o 5 /dev/sdb1  # 5% overprovision

# Create with specific segment size
mkfs.f2fs -s 4 /dev/sdb1  # 4MB segments
```

#### Mounting F2FS

```bash
# Mount with default options
mount -t f2fs /dev/sdb1 /mnt/data

# Mount with background GC
mount -t f2fs -o background_gc=on /dev/sdb1 /mnt/data

# Mount with inline data and inline dentry
mount -t f2fs -o inline_data,inline_dentry /dev/sdb1 /mnt/data

# Mount with compression
mount -t f2fs -o compress_algorithm=lz4 /dev/sdb1 /mnt/data
```

### tmpfs and ramfs

Memory-based filesystems store data in RAM, providing extremely fast access but volatile storage (data lost on reboot).

#### tmpfs

tmpfs uses both RAM and swap space, can be limited in size, and is the more commonly used option.

```bash
# Create tmpfs mount
mount -t tmpfs tmpfs /mnt/ramdisk

# Create with size limit
mount -t tmpfs -o size=1G tmpfs /mnt/ramdisk

# Create with size limit and specific permissions
mount -t tmpfs -o size=512M,mode=1777 tmpfs /tmp

# Create with uid/gid
mount -t tmpfs -o size=256M,uid=1000,gid=1000 tmpfs /mnt/ramdisk

# Create with inode limit
mount -t tmpfs -o size=1G,nr_inodes=10k tmpfs /mnt/ramdisk

# Show tmpfs usage
df -h /tmp
```

#### ramfs

ramfs is simpler than tmpfs, using only RAM (no swap), and cannot be limited in size.

```bash
# Create ramfs mount (no size limit, be careful!)
mount -t ramfs ramfs /mnt/ramdisk

# ramfs is rarely used; tmpfs is preferred in most cases
```

#### Common tmpfs Locations

```bash
# /tmp as tmpfs (common on modern systems)
tmpfs /tmp tmpfs defaults,noatime,mode=1777 0 0

# /run for runtime data
tmpfs /run tmpfs defaults,noatime,mode=0755 0 0

# /dev/shm for shared memory
tmpfs /dev/shm tmpfs defaults,noatime,mode=1777 0 0
```

### FAT/VFAT/exFAT

FAT filesystems are primarily used for compatibility with Windows and removable media.

#### FAT Variants

- **FAT16**: Legacy, max 2GB partition, max 2GB file
- **FAT32 (VFAT)**: Max 2TB partition, max 4GB file
- **exFAT**: Modern, large files and partitions, Windows/Mac compatible

#### Creating FAT Filesystems

```bash
# Create FAT32
mkfs.vfat /dev/sdb1

# Create FAT32 with label
mkfs.vfat -n MYUSB /dev/sdb1

# Create with specific cluster size
mkfs.vfat -s 8 /dev/sdb1  # 4KB clusters (8 * 512B)

# Create exFAT
mkfs.exfat /dev/sdb1

# Create exFAT with label
mkfs.exfat -n MYUSB /dev/sdb1
```

#### Mounting FAT Filesystems

```bash
# Mount with default options
mount -t vfat /dev/sdb1 /mnt/usb

# Mount with specific UID/GID (all files owned by user)
mount -t vfat -o uid=1000,gid=1000 /dev/sdb1 /mnt/usb

# Mount with UTF-8 encoding
mount -t vfat -o iocharset=utf8 /dev/sdb1 /mnt/usb

# Mount with specific umask (permissions)
mount -t vfat -o umask=022 /dev/sdb1 /mnt/usb

# Mount exFAT
mount -t exfat /dev/sdb1 /mnt/usb
```

### Other Filesystems

#### NTFS

```bash
# Install NTFS-3G driver
apt install ntfs-3g

# Mount NTFS
mount -t ntfs-3g /dev/sdb1 /mnt/windows

# Mount with full permissions for user
mount -t ntfs-3g -o uid=1000,gid=1000,dmask=022,fmask=133 /dev/sdb1 /mnt/windows
```

#### SquashFS

Read-only compressed filesystem, commonly used for live CDs and snap packages.

```bash
# Create SquashFS
mksquashfs /source/dir filesystem.squashfs

# Create with specific compression
mksquashfs /source/dir filesystem.squashfs -comp xz

# Mount SquashFS
mount -t squashfs filesystem.squashfs /mnt/squash
```

#### EROFS

Enhanced Read-Only File System, modern replacement for SquashFS in some distributions.

```bash
# Create EROFS
mkfs.erofs filesystem.erofs /source/dir

# Mount EROFS
mount -t erofs filesystem.erofs /mnt/erofs
```

#### ISO9660

```bash
# Create ISO
genisoimage -o image.iso /source/dir

# Mount ISO
mount -o loop image.iso /mnt/iso
```

## Virtual and Special Filesystems

Virtual filesystems don't store data on disk but provide interfaces to kernel data structures, device information, and debugging capabilities.

### procfs

The proc filesystem provides an interface to kernel data structures and process information.

#### Key Directories and Files

```bash
# Process information (one directory per PID)
/proc/[pid]/cmdline     # Command line
/proc/[pid]/environ     # Environment variables
/proc/[pid]/cwd         # Current working directory (symlink)
/proc/[pid]/exe         # Executable file (symlink)
/proc/[pid]/fd/         # Open file descriptors
/proc/[pid]/maps        # Memory mappings
/proc/[pid]/status      # Process status
/proc/[pid]/stat        # Process statistics
/proc/[pid]/io          # I/O statistics
/proc/[pid]/limits      # Resource limits

# System information
/proc/cpuinfo          # CPU information
/proc/meminfo          # Memory information
/proc/version          # Kernel version
/proc/uptime           # System uptime
/proc/loadavg          # Load average

# Kernel configuration
/proc/cmdline          # Kernel boot parameters
/proc/modules          # Loaded kernel modules
/proc/mounts           # Mounted filesystems
/proc/partitions       # Partition information
/proc/swaps            # Swap space information

# Network information
/proc/net/dev          # Network device statistics
/proc/net/route        # Routing table
/proc/net/tcp          # TCP socket information
/proc/net/udp          # UDP socket information

# System configuration (sysctl interface)
/proc/sys/kernel/      # Kernel parameters
/proc/sys/net/         # Network parameters
/proc/sys/vm/          # Virtual memory parameters
/proc/sys/fs/          # Filesystem parameters
```

#### Common procfs Operations

```bash
# View process command line
cat /proc/1234/cmdline | tr '\0' ' '

# View open files for process
ls -l /proc/1234/fd/

# Check process memory usage
cat /proc/1234/status | grep VmRSS

# View system memory
cat /proc/meminfo

# View CPU info
cat /proc/cpuinfo

# View kernel parameters
cat /proc/sys/net/ipv4/ip_forward

# Modify kernel parameter (temporary, lost on reboot)
echo 1 > /proc/sys/net/ipv4/ip_forward

# Better way: use sysctl
sysctl -w net.ipv4.ip_forward=1
```

### sysfs

The sys filesystem exposes kernel objects, their attributes, and relationships between them. It's structured around the kernel's device model.

#### Key Directories

```bash
/sys/block/           # Block devices
/sys/bus/             # Bus types (pci, usb, etc.)
/sys/class/           # Device classes (net, input, etc.)
/sys/devices/         # Device tree
/sys/firmware/        # Firmware information
/sys/fs/              # Filesystem information
/sys/kernel/          # Kernel information
/sys/module/          # Loaded modules
/sys/power/           # Power management
```

#### Common sysfs Operations

```bash
# View network device information
ls /sys/class/net/
cat /sys/class/net/eth0/address        # MAC address
cat /sys/class/net/eth0/statistics/rx_bytes
cat /sys/class/net/eth0/speed          # Link speed

# View block device information
ls /sys/block/
cat /sys/block/sda/size                # Size in sectors
cat /sys/block/sda/queue/scheduler     # I/O scheduler
cat /sys/block/sda/device/model        # Device model

# Change I/O scheduler
echo mq-deadline > /sys/block/sda/queue/scheduler

# View CPU information
ls /sys/devices/system/cpu/
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# View module parameters
ls /sys/module/zfs/parameters/
cat /sys/module/zfs/parameters/zfs_arc_max

# Modify module parameter
echo 8589934592 > /sys/module/zfs/parameters/zfs_arc_max

# View USB devices
ls /sys/bus/usb/devices/

# View PCI devices
ls /sys/bus/pci/devices/
```

### debugfs

A RAM-based filesystem for kernel debugging information. Not normally mounted on production systems.

```bash
# Mount debugfs
mount -t debugfs debugfs /sys/kernel/debug

# View tracing information
cat /sys/kernel/debug/tracing/available_tracers

# View block device trace
cat /sys/kernel/debug/block/sda/trace

# View filesystem-specific debug info
ls /sys/kernel/debug/ext4/sda1/
ls /sys/kernel/debug/btrfs/
```

### devtmpfs

Automatically manages device nodes in /dev. Modern systems use devtmpfs with udev.

```bash
# devtmpfs is typically mounted automatically at boot
mount -t devtmpfs devtmpfs /dev

# View mount
mount | grep devtmpfs
```

### configfs

Used for kernel object configuration, particularly in storage and network subsystems.

```bash
# Mount configfs
mount -t configfs configfs /sys/kernel/config

# Used by various kernel subsystems
ls /sys/kernel/config/
# target/       - SCSI target subsystem
# usb_gadget/   - USB gadget configuration
# nvmet/        - NVMe target
```

### cgroup Filesystems

Control groups provide resource limiting, prioritization, and accounting.

#### cgroup v1

```bash
# Mount cgroup v1 (typically at /sys/fs/cgroup/)
mount -t cgroup -o cpu cpu /sys/fs/cgroup/cpu
mount -t cgroup -o memory memory /sys/fs/cgroup/memory
mount -t cgroup -o blkio blkio /sys/fs/cgroup/blkio

# Create cgroup
mkdir /sys/fs/cgroup/cpu/mygroup

# Set CPU limit (50% of one CPU)
echo 50000 > /sys/fs/cgroup/cpu/mygroup/cpu.cfs_quota_us
echo 100000 > /sys/fs/cgroup/cpu/mygroup/cpu.cfs_period_us

# Add process to cgroup
echo $PID > /sys/fs/cgroup/cpu/mygroup/cgroup.procs
```

#### cgroup v2

```bash
# Mount cgroup v2 (unified hierarchy)
mount -t cgroup2 cgroup2 /sys/fs/cgroup

# Create cgroup
mkdir /sys/fs/cgroup/mygroup

# Enable controllers
echo "+cpu +memory +io" > /sys/fs/cgroup/cgroup.subtree_control

# Set CPU weight (100-10000, default 100)
echo 500 > /sys/fs/cgroup/mygroup/cpu.weight

# Set memory limit
echo 1G > /sys/fs/cgroup/mygroup/memory.max

# Add process
echo $PID > /sys/fs/cgroup/mygroup/cgroup.procs
```

For more information on cgroup interaction with namespaces, see [namespace.md](namespace.md).

### Other Special Filesystems

```bash
# securityfs - Security module information
mount -t securityfs securityfs /sys/kernel/security

# fusectl - FUSE control filesystem
mount -t fusectl fusectl /sys/fs/fuse/connections

# tracefs - Tracing infrastructure
mount -t tracefs tracefs /sys/kernel/tracing

# bpf - BPF filesystem for pinning BPF objects
mount -t bpf bpf /sys/fs/bpf

# pstore - Persistent storage for oops/panic logs
mount -t pstore pstore /sys/fs/pstore

# efivarfs - UEFI variable filesystem
mount -t efivarfs efivarfs /sys/firmware/efi/efivars
```

## OverlayFS Deep Dive

OverlayFS is a union mount filesystem that combines multiple directories into a single merged view. It's the default storage driver for Docker and widely used in containers.

### Architecture and Concepts

OverlayFS combines layers of directories with specific roles:

```
Merged View (appears to user)
         |
    +---------+
    |         |
  Upper     Lower
  (r/w)     (r/o)

Actual layout:
  /merged    <- Merged view (mount point)
  /upper     <- Read-write layer
  /work      <- Work directory (internal)
  /lower     <- Read-only base layer
```

#### Key Components

- **Lower Directory**: Read-only base layer(s). Can be multiple layers stacked.
- **Upper Directory**: Read-write layer where all changes are stored.
- **Work Directory**: Internal directory used by OverlayFS for atomic operations.
- **Merged Directory**: The unified view presented to users.

#### Basic OverlayFS Mount

```bash
# Create directories
mkdir -p /tmp/overlay/{lower,upper,work,merged}

# Add some content to lower layer
echo "Base content" > /tmp/overlay/lower/file.txt

# Mount overlay
mount -t overlay overlay \
  -o lowerdir=/tmp/overlay/lower,upperdir=/tmp/overlay/upper,workdir=/tmp/overlay/work \
  /tmp/overlay/merged

# View merged content
ls /tmp/overlay/merged  # Shows file.txt from lower

# Modify file (copy-up occurs)
echo "Modified content" > /tmp/overlay/merged/file.txt

# Original in lower is unchanged
cat /tmp/overlay/lower/file.txt    # "Base content"

# Modified version in upper
cat /tmp/overlay/upper/file.txt    # "Modified content"

# Merged view shows modified version
cat /tmp/overlay/merged/file.txt   # "Modified content"
```

### Copy-Up Mechanism

When a file from the lower layer is modified, OverlayFS copies it to the upper layer (copy-up). This is a key characteristic of copy-on-write behavior.

#### Copy-Up Behavior

```bash
# Initial state: file exists only in lower
stat /tmp/overlay/lower/file.txt   # Exists
stat /tmp/overlay/upper/file.txt   # Does not exist

# Read file (no copy-up)
cat /tmp/overlay/merged/file.txt

# Copy-up happens on first write
echo "new content" >> /tmp/overlay/merged/file.txt

# Now file exists in upper
stat /tmp/overlay/upper/file.txt   # Exists

# Metadata changes also trigger copy-up
chmod 755 /tmp/overlay/merged/script.sh  # Triggers copy-up
```

#### Copy-Up Performance Considerations

- Copy-up happens for the entire file, even for small changes
- Large files incur significant copy-up cost on first write
- Use volumes or bind mounts for database files and large mutable data
- Read-only workloads avoid copy-up entirely

```bash
# Check copy-up activity
# Files in upper directory show what has been copied up
find /tmp/overlay/upper -type f
```

### Whiteouts and Opaque Directories

OverlayFS uses special markers to represent deleted files and directories.

#### Whiteout Files

When a file is deleted from the merged view, OverlayFS creates a whiteout file in the upper layer.

```bash
# File exists in lower
echo "content" > /tmp/overlay/lower/file.txt
mount -t overlay ...

# Delete file from merged view
rm /tmp/overlay/merged/file.txt

# File still exists in lower (read-only)
ls /tmp/overlay/lower/file.txt

# Whiteout created in upper (character device 0:0)
ls -l /tmp/overlay/upper/
# c--------- 1 root root 0, 0 Jan 1 12:00 file.txt

# Whiteout hides lower file in merged view
ls /tmp/overlay/merged/file.txt  # No such file

# Check if file is a whiteout
stat /tmp/overlay/upper/file.txt
# Character Device (0:0)
```

#### Opaque Directories

When a directory is deleted and recreated, or when rmdir/mkdir happens, OverlayFS may create an opaque directory.

```bash
# Directory exists in lower with content
mkdir -p /tmp/overlay/lower/dir
echo "lower content" > /tmp/overlay/lower/dir/file.txt

# Remove directory in merged view
rm -rf /tmp/overlay/merged/dir

# Whiteout created for directory
ls -l /tmp/overlay/upper/
# c--------- 1 root root 0, 0 Jan 1 12:00 dir

# Recreate directory
mkdir /tmp/overlay/merged/dir
echo "new content" > /tmp/overlay/merged/dir/newfile.txt

# Directory becomes opaque (trusted.overlay.opaque=y xattr)
getfattr -n trusted.overlay.opaque /tmp/overlay/upper/dir
# trusted.overlay.opaque="y"

# Opaque directory hides all lower contents
ls /tmp/overlay/merged/dir/
# newfile.txt (file.txt from lower is hidden)
```

### Multiple Lower Layers

OverlayFS supports multiple lower layers, which is essential for container images with multiple layers.

```bash
# Create multiple lower layers
mkdir -p /tmp/overlay/{lower1,lower2,lower3,upper,work,merged}

# Add content to different layers
echo "Layer 1" > /tmp/overlay/lower1/file1.txt
echo "Layer 2" > /tmp/overlay/lower2/file2.txt
echo "Layer 3" > /tmp/overlay/lower3/file3.txt

# Same file in multiple layers (highest priority wins)
echo "From layer 1" > /tmp/overlay/lower1/common.txt
echo "From layer 2" > /tmp/overlay/lower2/common.txt

# Mount with multiple lower layers (left to right = high to low priority)
mount -t overlay overlay \
  -o lowerdir=/tmp/overlay/lower1:/tmp/overlay/lower2:/tmp/overlay/lower3,upperdir=/tmp/overlay/upper,workdir=/tmp/overlay/work \
  /tmp/overlay/merged

# Merged view shows all files
ls /tmp/overlay/merged/
# file1.txt file2.txt file3.txt common.txt

# common.txt comes from lower1 (highest priority)
cat /tmp/overlay/merged/common.txt
# From layer 1

# Layer ordering matters: lower1 > lower2 > lower3
```

#### Layer Limits

```bash
# Maximum number of lower layers varies by kernel version
# Kernel < 4.13: ~500 layers
# Kernel >= 4.13: ~500 layers (practical limit)
# Kernel >= 5.11: ~500 layers (with performance optimizations)

# Check overlay module info
modinfo overlay | grep -i layer
```

### Container Integration

OverlayFS is the default storage driver for Docker and commonly used in Kubernetes.

#### Docker OverlayFS Structure

```bash
# Docker overlay2 storage driver layout
/var/lib/docker/overlay2/
├── l/                    # Shortened layer identifiers (symlinks)
├── <layer-id>/
│   ├── diff/            # Layer content
│   ├── link             # Shortened identifier
│   ├── lower            # Parent layer references
│   └── work/            # Work directory
└── <layer-id>/
    └── merged/          # Container's merged filesystem (when running)

# Inspect Docker storage driver
docker info | grep -A5 "Storage Driver"

# View layer information for image
docker inspect nginx:latest | grep -A20 GraphDriver

# View overlay mounts for running containers
mount | grep overlay
docker ps -q | xargs -I {} docker inspect -f '{{.GraphDriver.Data}}' {}
```

#### Docker Overlay2 Mount Example

```bash
# When a container runs, Docker creates overlay mount
# Example mount command (simplified):
mount -t overlay overlay \
  -o lowerdir=/var/lib/docker/overlay2/l/ABC:/var/lib/docker/overlay2/l/DEF:/var/lib/docker/overlay2/l/GHI,\
     upperdir=/var/lib/docker/overlay2/xyz/diff,\
     workdir=/var/lib/docker/overlay2/xyz/work \
  /var/lib/docker/overlay2/xyz/merged

# Container's root filesystem is at merged/
# All changes go to upperdir (container layer)
# Image layers are in lowerdir (read-only)
```

#### Kubernetes and containerd

```bash
# containerd uses overlay snapshots
/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/

# View containerd snapshots
ctr -n k8s.io snapshots ls

# Kubernetes pod overlay mounts
mount | grep overlay | grep kube
```

### Advanced Features

#### redirect_dir

Allows directory rename/merge operations to be more efficient.

```bash
# Mount with redirect_dir (requires kernel >= 4.10)
mount -t overlay overlay \
  -o redirect_dir=on,lowerdir=/lower,upperdir=/upper,workdir=/work \
  /merged

# Directory renames don't require full copy
# Instead, xattr marks redirect: trusted.overlay.redirect
```

#### index

Improves performance and fixes hardlink issues.

```bash
# Mount with index feature (requires kernel >= 4.13)
mount -t overlay overlay \
  -o index=on,lowerdir=/lower,upperdir=/upper,workdir=/work \
  /merged

# Index directory created in work directory
ls /work/index/

# Hardlinks work correctly across layers
# Inode numbers are consistent
```

#### metacopy

Optimizes small metadata-only changes.

```bash
# Mount with metacopy (requires kernel >= 4.19)
mount -t overlay overlay \
  -o metacopy=on,lowerdir=/lower,upperdir=/upper,workdir=/work \
  /merged

# For metadata-only changes (chmod, chown), only metadata copied
# Data remains in lower layer (efficient)
```

#### xino

Provides unique inode numbers across layers.

```bash
# Mount with xino (requires kernel >= 4.17)
mount -t overlay overlay \
  -o xino=on,lowerdir=/lower,upperdir=/upper,workdir=/work \
  /merged

# Ensures unique inode numbers even with multiple layers
# Prevents inode collisions
```

### Performance and Limitations

#### Performance Characteristics

```bash
# Read performance:
# - Lower layer reads: Fast, direct from lower filesystem
# - Upper layer reads: Fast, direct from upper filesystem
# - Merged metadata operations: Slight overhead

# Write performance:
# - New files: Fast, written directly to upper
# - Modified files: First write incurs copy-up cost
# - Large files: Significant copy-up penalty

# Directory operations:
# - Readdir: Must merge entries from all layers (can be slow)
# - Directory rename: Complex, may require copying
```

#### Limitations

```bash
# 1. Copy-up of large files is expensive
# Workaround: Use volumes for large mutable files

# 2. Some operations not supported on lower layers
# - Extended attributes may not work correctly
# Workaround: Use specific mount options (xattr, noxattr)

# 3. Limited inode operations
# - Hardlinks across layers may not work without index=on

# 4. Nested overlays can cause issues
# Don't mount overlay on top of another overlay

# 5. File descriptor inconsistency
# After copy-up, existing file descriptors point to lower file
# New opens point to upper file

# 6. Rename limitations
# Some complex rename operations may fail or be inefficient
```

#### Best Practices

```bash
# 1. Use volumes for databases and large files
docker run -v /host/data:/container/data ...

# 2. Minimize layer count (merge layers in Dockerfile)
RUN apt-get update && apt-get install -y pkg1 pkg2 && rm -rf /var/lib/apt/lists/*

# 3. Enable modern features
mount -o index=on,xino=on,redirect_dir=on ...

# 4. Use appropriate filesystem for upper/lower
# - Upper: Fast filesystem with xattr support (ext4, xfs)
# - Lower: Can be anything, even compressed squashfs

# 5. Monitor overlay usage
du -sh /var/lib/docker/overlay2/
docker system df -v
```

### OverlayFS Troubleshooting

#### Error: upperdir is in-use

```bash
# Error message:
# overlayfs: upperdir is in-use as upperdir/workdir of another mount

# Cause: Directory already used by another overlay mount

# Solution 1: Unmount existing overlay
umount /merged

# Solution 2: Use different upper/work directories
mkdir /upper2 /work2
mount -t overlay overlay -o lowerdir=/lower,upperdir=/upper2,workdir=/work2 /merged

# Solution 3: Check for stale mounts
mount | grep overlay
findmnt -t overlay
```

#### Error: workdir and upperdir must be separate

```bash
# Error message:
# overlayfs: workdir and upperdir must be separate subtrees

# Cause: Work directory inside upper directory or vice versa

# Solution: Use separate directories
mkdir -p /overlay/{upper,work}  # Siblings, not nested
mount -t overlay overlay -o lowerdir=/lower,upperdir=/overlay/upper,workdir=/overlay/work /merged
```

#### Error: failed to resolve lowerdir

```bash
# Error message:
# overlayfs: failed to resolve 'lowerdir': -2

# Cause: Lower directory doesn't exist or path incorrect

# Solution: Verify all directories exist
mkdir -p /lower /upper /work /merged
ls -ld /lower /upper /work
```

#### Whiteouts Visible

```bash
# If you see character devices (0:0) in merged view, overlay is not working

# Check mount options
mount | grep overlay

# Verify proper mount
findmnt -t overlay

# Remount with correct options
umount /merged
mount -t overlay overlay -o lowerdir=/lower,upperdir=/upper,workdir=/work /merged
```

#### Permission Denied After Copy-Up

```bash
# Issue: File becomes inaccessible after modification

# Check filesystem xattr support
tune2fs -l /dev/sda1 | grep xattr

# Enable user_xattr if needed
mount -o remount,user_xattr /upper

# Or in fstab:
# /dev/sda1 /upper ext4 user_xattr 0 2
```

#### Debugging OverlayFS

```bash
# Enable overlay debugging (requires debugfs)
mount -t debugfs debugfs /sys/kernel/debug
echo 1 > /sys/kernel/debug/overlayfs

# View kernel logs
dmesg | grep overlay

# Check overlay mount details
cat /proc/mounts | grep overlay

# Inspect file origin (which layer it comes from)
getfattr -n trusted.overlay.origin /merged/file.txt

# Check if directory is opaque
getfattr -n trusted.overlay.opaque /merged/dir/
```

## Mount Operations

Mounting attaches a filesystem to the directory tree at a specific point (mount point).

### Basic Mounting

```bash
# Mount device to directory
mount /dev/sdb1 /mnt/data

# Mount with specific filesystem type
mount -t ext4 /dev/sdb1 /mnt/data

# Mount with options
mount -o ro,noatime /dev/sdb1 /mnt/data

# Mount all filesystems in /etc/fstab
mount -a

# Mount by label
mount LABEL=mydata /mnt/data

# Mount by UUID
mount UUID=12345678-1234-1234-1234-123456789abc /mnt/data

# Unmount
umount /mnt/data
# Or by device
umount /dev/sdb1
```

### Mount Options

Mount options control filesystem behavior. Options are specified with -o and comma-separated.

#### Generic Mount Options

```bash
# Read-only / Read-write
mount -o ro /dev/sdb1 /mnt/data    # Read-only
mount -o rw /dev/sdb1 /mnt/data    # Read-write (default)

# Access time updates
mount -o atime /dev/sdb1 /mnt/data      # Update access time (default)
mount -o noatime /dev/sdb1 /mnt/data    # Don't update access time (faster)
mount -o relatime /dev/sdb1 /mnt/data   # Update if older than mtime (default on modern systems)
mount -o nodiratime /dev/sdb1 /mnt/data # Don't update directory access times

# Synchronous I/O
mount -o sync /dev/sdb1 /mnt/data      # All I/O synchronous (slow, safe)
mount -o async /dev/sdb1 /mnt/data     # Asynchronous I/O (default)

# Execution and device files
mount -o exec /dev/sdb1 /mnt/data      # Allow execution (default)
mount -o noexec /dev/sdb1 /mnt/data    # Prevent execution
mount -o dev /dev/sdb1 /mnt/data       # Allow device files (default)
mount -o nodev /dev/sdb1 /mnt/data     # Ignore device files
mount -o suid /dev/sdb1 /mnt/data      # Allow setuid/setgid (default)
mount -o nosuid /dev/sdb1 /mnt/data    # Ignore setuid/setgid

# User mounts
mount -o user /dev/sdb1 /mnt/data      # Allow user to mount
mount -o users /dev/sdb1 /mnt/data     # Allow any user to mount
mount -o nouser /dev/sdb1 /mnt/data    # Only root can mount (default)

# Automatic mounting
mount -o auto /dev/sdb1 /mnt/data      # Can be mounted with -a (default)
mount -o noauto /dev/sdb1 /mnt/data    # Skip with -a

# Common combined options for security
mount -o noexec,nodev,nosuid /dev/sdb1 /mnt/data
```

#### Filesystem-Specific Options

```bash
# ext4 options
mount -o data=journal /dev/sdb1 /mnt      # Journal data and metadata
mount -o data=ordered /dev/sdb1 /mnt      # Journal metadata only (default)
mount -o data=writeback /dev/sdb1 /mnt    # No data ordering
mount -o barrier=1 /dev/sdb1 /mnt         # Enable write barriers
mount -o nobarrier /dev/sdb1 /mnt         # Disable write barriers
mount -o journal_checksum /dev/sdb1 /mnt  # Enable journal checksums
mount -o discard /dev/sdb1 /mnt           # Enable TRIM for SSDs
mount -o nodiscard /dev/sdb1 /mnt         # Disable TRIM

# XFS options
mount -o logbufs=8 /dev/sdb1 /mnt         # Number of log buffers
mount -o logbsize=256k /dev/sdb1 /mnt     # Log buffer size
mount -o nobarrier /dev/sdb1 /mnt         # Disable write barriers
mount -o discard /dev/sdb1 /mnt           # Enable TRIM

# Btrfs options
mount -o compress=zstd /dev/sdb1 /mnt     # Enable compression
mount -o compress-force=lzo /dev/sdb1 /mnt # Force compression
mount -o space_cache=v2 /dev/sdb1 /mnt    # Free space cache version
mount -o ssd /dev/sdb1 /mnt               # SSD optimizations
mount -o nossd /dev/sdb1 /mnt             # Disable SSD optimizations
mount -o subvol=name /dev/sdb1 /mnt       # Mount specific subvolume
mount -o subvolid=256 /dev/sdb1 /mnt      # Mount by subvolume ID

# tmpfs options
mount -t tmpfs -o size=1G tmpfs /mnt      # Size limit
mount -t tmpfs -o nr_inodes=10k tmpfs /mnt # Inode limit
mount -t tmpfs -o mode=1777 tmpfs /mnt    # Permissions
mount -t tmpfs -o uid=1000,gid=1000 tmpfs /mnt # Owner

# NTFS-3G options
mount -t ntfs-3g -o uid=1000,gid=1000 /dev/sdb1 /mnt  # User ownership
mount -t ntfs-3g -o permissions /dev/sdb1 /mnt        # Unix permissions
mount -t ntfs-3g -o windows_names /dev/sdb1 /mnt      # Windows filename rules

# NFS options
mount -t nfs -o vers=4.2 server:/export /mnt          # NFS version
mount -t nfs -o soft,timeo=30 server:/export /mnt     # Soft mount with timeout
mount -t nfs -o hard,intr server:/export /mnt         # Hard mount, interruptible
mount -t nfs -o tcp server:/export /mnt               # Use TCP (default for NFSv4)
mount -t nfs -o udp server:/export /mnt               # Use UDP
```

### fstab Configuration

The `/etc/fstab` file defines filesystems to be mounted at boot.

#### fstab Format

```bash
# /etc/fstab format:
# <device> <mount point> <type> <options> <dump> <pass>

# Example entries:
UUID=12345678-1234-1234-1234-123456789abc /               ext4    errors=remount-ro 0 1
UUID=abcdef12-3456-7890-abcd-ef1234567890 /home           ext4    defaults,noatime  0 2
UUID=11111111-2222-3333-4444-555555555555 none            swap    sw                0 0
/dev/sdb1                                   /mnt/data     ext4    defaults,nofail   0 2
LABEL=backup                                /mnt/backup   xfs     defaults,noauto   0 0
tmpfs                                       /tmp          tmpfs   defaults,size=2G  0 0
server.example.com:/export                  /mnt/nfs      nfs     defaults,_netdev  0 0

# Fields:
# 1. Device: /dev/sdX, UUID, LABEL, or remote path
# 2. Mount point: Where to mount
# 3. Filesystem type: ext4, xfs, btrfs, nfs, etc.
# 4. Options: Comma-separated mount options
# 5. Dump: Backup with dump command (0=no, 1=yes)
# 6. Pass: fsck order (0=skip, 1=root, 2=other)
```

#### Common fstab Patterns

```bash
# Root filesystem (errors=remount-ro protects system)
UUID=xxx / ext4 errors=remount-ro 0 1

# Home with noatime for performance
UUID=xxx /home ext4 defaults,noatime 0 2

# Swap space
UUID=xxx none swap sw 0 0

# Data partition (nofail allows boot if device missing)
UUID=xxx /mnt/data ext4 defaults,nofail 0 2

# External drive (noauto prevents automatic mount)
LABEL=backup /mnt/backup ext4 defaults,noauto 0 0

# Removable media with user mount permission
/dev/sdc1 /media/usb vfat defaults,noauto,user,uid=1000,gid=1000 0 0

# tmpfs for /tmp
tmpfs /tmp tmpfs defaults,noatime,mode=1777,size=2G 0 0

# NFS mount (_netdev waits for network)
server:/export /mnt/nfs nfs defaults,_netdev 0 0

# Bind mount
/home/user/docs /var/www/docs none bind 0 0

# OverlayFS (advanced)
overlay /merged overlay noauto,x-systemd.requires-mounts-for=/lower:/upper,lowerdir=/lower,upperdir=/upper,workdir=/work 0 0
```

#### UUID and Label Discovery

```bash
# Find UUID
blkid /dev/sdb1
lsblk -f /dev/sdb1
ls -l /dev/disk/by-uuid/

# Find label
blkid -s LABEL /dev/sdb1
e2label /dev/sdb1                # ext2/3/4
xfs_admin -l /dev/sdb1           # XFS
btrfs filesystem label /mnt/data # Btrfs

# Set label
e2label /dev/sdb1 newlabel       # ext2/3/4
xfs_admin -L newlabel /dev/sdb1  # XFS
tune2fs -L newlabel /dev/sdb1    # ext2/3/4
```

### Systemd Mount Units

Systemd can manage mounts with unit files, providing better control and dependencies.

```bash
# Create mount unit: /etc/systemd/system/mnt-data.mount
[Unit]
Description=Data partition
After=local-fs-pre.target

[Mount]
What=/dev/disk/by-uuid/12345678-1234-1234-1234-123456789abc
Where=/mnt/data
Type=ext4
Options=defaults,noatime

[Install]
WantedBy=multi-user.target

# Enable and start
systemctl daemon-reload
systemctl enable mnt-data.mount
systemctl start mnt-data.mount

# Status
systemctl status mnt-data.mount

# Note: Unit name must match mount path with dashes
# /mnt/data -> mnt-data.mount
# /mnt/my-data -> mnt-my\x2ddata.mount (escape dashes with \x2d)
```

### Bind Mounts

Bind mounts make a directory appear at another location.

```bash
# Create bind mount
mount --bind /source/dir /dest/dir

# Read-only bind mount
mount --bind /source/dir /dest/dir
mount -o remount,ro,bind /dest/dir

# Recursive bind mount (include submounts)
mount --rbind /source/dir /dest/dir

# Bind mount in fstab
/source/dir /dest/dir none bind 0 0
/source/dir /dest/dir none rbind 0 0

# Use cases:
# - Exposing directories in chroots
# - Container filesystem isolation
# - Sharing directories without symlinks
```

### Mount Propagation

Mount propagation controls how mount/unmount events propagate between mount namespaces. See [namespace.md](namespace.md) for detailed coverage.

```bash
# Make mount shared (propagates to/from peers)
mount --make-shared /mnt/data

# Make mount private (no propagation)
mount --make-private /mnt/data

# Make mount slave (receive but don't send propagation)
mount --make-slave /mnt/data

# Make mount unbindable (prevent bind mounts)
mount --make-unbindable /mnt/data

# Recursive versions
mount --make-rshared /mnt/data
mount --make-rprivate /mnt/data
mount --make-rslave /mnt/data
mount --make-runbindable /mnt/data

# View propagation
findmnt -o TARGET,PROPAGATION
cat /proc/self/mountinfo
```

### Remounting and Unmounting

```bash
# Remount with different options (no unmount required)
mount -o remount,ro /mnt/data       # Change to read-only
mount -o remount,rw /mnt/data       # Change to read-write
mount -o remount,noatime /mnt/data  # Add noatime option

# Normal unmount
umount /mnt/data

# Force unmount (kills processes using filesystem)
umount -f /mnt/data

# Lazy unmount (detach immediately, cleanup when no longer busy)
umount -l /mnt/data

# Unmount all filesystems of a type
umount -a -t nfs

# Find processes using a filesystem
lsof /mnt/data
fuser -m /mnt/data
fuser -km /mnt/data  # Kill processes
```

### Mount Inspection

```bash
# Show all mounts
mount

# Show specific filesystem types
mount -t ext4

# findmnt (modern, structured output)
findmnt                          # Tree view
findmnt -l                       # List view
findmnt /mnt/data                # Specific mount point
findmnt /dev/sdb1                # By device
findmnt -t ext4                  # By type
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS  # Custom columns

# /proc/mounts (kernel view)
cat /proc/mounts

# /etc/mtab (user-space view, usually symlink to /proc/self/mounts)
cat /etc/mtab

# lsblk (block device tree)
lsblk
lsblk -f                         # Include filesystem info
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,UUID

# df (disk usage)
df -h                            # Human-readable
df -T                            # Include filesystem type
df -i                            # Inode usage

# /proc/self/mountinfo (detailed propagation info)
cat /proc/self/mountinfo
```

## Filesystem Management

### Creating Filesystems

```bash
# Generic mkfs command (calls appropriate mkfs.*)
mkfs -t ext4 /dev/sdb1

# Specific mkfs commands
mkfs.ext4 /dev/sdb1
mkfs.xfs /dev/sdb1
mkfs.btrfs /dev/sdb1
mkfs.vfat /dev/sdb1
mkfs.exfat /dev/sdb1

# Warning: This destroys all data on the device!
# Always verify device name before running mkfs

# Common options:
# -L label      Set filesystem label
# -m percent    Reserved blocks percentage (ext4)
# -n            Dry run (ext4)
# -f            Force creation

# Verify device first
lsblk
fdisk -l /dev/sdb
```

### Checking and Repairing

Filesystem checking should be done on unmounted filesystems (or read-only mounts).

```bash
# ext2/ext3/ext4
fsck.ext4 /dev/sdb1               # Check and repair
fsck.ext4 -n /dev/sdb1            # Check only (no modifications)
fsck.ext4 -p /dev/sdb1            # Automatic repair (safe)
fsck.ext4 -y /dev/sdb1            # Answer yes to all questions
fsck.ext4 -f /dev/sdb1            # Force check even if clean

# XFS
xfs_repair /dev/sdb1              # Check and repair
xfs_repair -n /dev/sdb1           # Check only (no modifications)
xfs_repair -L /dev/sdb1           # Zero log (last resort)
xfs_repair -v /dev/sdb1           # Verbose output

# Btrfs
btrfs check /dev/sdb1             # Check filesystem
btrfs check --repair /dev/sdb1    # Dangerous! Backup first
btrfs scrub start /mnt/btrfs      # Online check (while mounted)
btrfs scrub status /mnt/btrfs

# FAT
fsck.vfat /dev/sdb1               # Check and repair
fsck.vfat -a /dev/sdb1            # Automatic repair
fsck.vfat -n /dev/sdb1            # Check only

# Generic fsck (detects type automatically)
fsck /dev/sdb1
fsck -A                            # Check all in /etc/fstab
fsck -AR                           # Check all except root
```

### Resizing Filesystems

Always backup before resizing!

```bash
# ext2/ext3/ext4
# Shrink requires unmount, grow can be online
e2fsck -f /dev/sdb1                # Must check first
resize2fs /dev/sdb1 50G            # Resize to 50GB
resize2fs /dev/sdb1                # Grow to partition size

# XFS (can only grow, not shrink)
xfs_growfs /mnt/data               # Grow to device size (must be mounted)
xfs_growfs -D 13107200 /mnt/data   # Grow to specific size (blocks)

# Btrfs (online resize)
btrfs filesystem resize +10G /mnt/data   # Grow by 10GB
btrfs filesystem resize -5G /mnt/data    # Shrink by 5GB
btrfs filesystem resize max /mnt/data    # Grow to device size
btrfs filesystem resize 1:+10G /mnt/data # Resize specific device in multi-device FS

# Typical workflow:
# 1. Resize partition (fdisk, parted, etc.)
# 2. Resize filesystem

# Example: Growing ext4
parted /dev/sdb resizepart 1 100%
resize2fs /dev/sdb1

# Example: Shrinking ext4
umount /mnt/data
e2fsck -f /dev/sdb1
resize2fs /dev/sdb1 50G
parted /dev/sdb resizepart 1 50GB
mount /dev/sdb1 /mnt/data
```

### Tuning Parameters

```bash
# ext2/ext3/ext4 (tune2fs)
tune2fs -l /dev/sdb1                    # List parameters
tune2fs -L newlabel /dev/sdb1           # Set label
tune2fs -m 1 /dev/sdb1                  # Reserved blocks (1%)
tune2fs -c 0 /dev/sdb1                  # Disable mount count check
tune2fs -i 0 /dev/sdb1                  # Disable time-based check
tune2fs -O ^has_journal /dev/sdb1       # Remove journal (ext4->ext2)
tune2fs -O has_journal /dev/sdb1        # Add journal
tune2fs -o journal_data_writeback /dev/sdb1  # Default mount options

# XFS (xfs_admin)
xfs_admin -l /dev/sdb1                  # Show label
xfs_admin -L newlabel /dev/sdb1         # Set label
xfs_admin -u /dev/sdb1                  # Show UUID
xfs_admin -U generate /dev/sdb1         # Generate new UUID

# Btrfs (btrfs property)
btrfs filesystem label /mnt/data newlabel  # Set label
btrfs property get /mnt/data            # Get properties
btrfs property set /mnt/data compression zstd  # Set property
```

### Labels and UUIDs

```bash
# View label and UUID
blkid /dev/sdb1
lsblk -f /dev/sdb1

# Set label
e2label /dev/sdb1 newlabel              # ext2/3/4
tune2fs -L newlabel /dev/sdb1           # ext2/3/4
xfs_admin -L newlabel /dev/sdb1         # XFS
btrfs filesystem label /mnt/data newlabel  # Btrfs
fatlabel /dev/sdb1 newlabel             # FAT
exfatlabel /dev/sdb1 newlabel           # exFAT

# Set UUID
tune2fs -U random /dev/sdb1             # ext2/3/4 (generate)
tune2fs -U 12345678-1234-1234-1234-123456789abc /dev/sdb1  # ext2/3/4 (specific)
xfs_admin -U generate /dev/sdb1         # XFS (generate)
xfs_admin -U 12345678-1234-1234-1234-123456789abc /dev/sdb1  # XFS (specific)
btrfstune -U 12345678-1234-1234-1234-123456789abc /dev/sdb1  # Btrfs
```

### Monitoring

```bash
# Disk space usage
df -h                                   # Human-readable
df -i                                   # Inodes
df -T                                   # Include filesystem type

# Directory usage
du -sh /path/to/dir                     # Summary
du -h --max-depth=1 /path/to/dir        # One level deep
du -ah /path/to/dir | sort -h           # All files, sorted

# Filesystem statistics
stat -f /mnt/data                       # Filesystem stats
stat /mnt/data/file                     # File stats

# ext4 specific
dumpe2fs /dev/sdb1                      # Detailed filesystem info
tune2fs -l /dev/sdb1                    # Superblock info

# XFS specific
xfs_info /mnt/data                      # Filesystem geometry
xfs_db -r -c "freesp -s" /dev/sdb1      # Free space analysis

# Btrfs specific
btrfs filesystem show                   # All Btrfs filesystems
btrfs filesystem usage /mnt/data        # Usage breakdown
btrfs device stats /mnt/data            # Device statistics

# ZFS specific
zpool list                              # Pool summary
zfs list                                # Dataset list
zpool iostat -v 1                       # I/O stats (1 sec interval)

# I/O statistics
iostat -x 1                             # Extended I/O stats (1 sec interval)
iotop                                   # Top-like I/O monitor
```

## Performance Considerations

### I/O Schedulers

The I/O scheduler determines how I/O requests are ordered and dispatched to block devices.

```bash
# View current scheduler
cat /sys/block/sda/queue/scheduler
# Output: [mq-deadline] none kyber bfq
# Brackets indicate active scheduler

# Available schedulers (modern multi-queue)
# none        - No scheduling (direct dispatch)
# mq-deadline - Deadline-based (good general purpose)
# kyber       - Token-based (low latency)
# bfq         - Budget Fair Queueing (desktop, interactive)

# Change scheduler (temporary)
echo mq-deadline > /sys/block/sda/queue/scheduler
echo bfq > /sys/block/sda/queue/scheduler

# Change scheduler permanently (kernel parameter)
# Add to GRUB_CMDLINE_LINUX in /etc/default/grub:
# elevator=mq-deadline

# Or with udev rule (/etc/udev/rules.d/60-scheduler.rules):
# ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/scheduler}="mq-deadline"
# ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/scheduler}="none"

# Recommendations:
# - SSDs/NVMe: none or mq-deadline
# - HDDs: mq-deadline or bfq
# - Virtual machines: none (let hypervisor handle)
# - Database servers: mq-deadline
# - Desktops: bfq
```

### Mount Options for Performance

```bash
# Reduce writes (improves SSD lifespan and performance)
mount -o noatime,nodiratime /dev/sdb1 /mnt/data
# noatime: Don't update access time on reads
# relatime: Update access time only if mtime is newer (default, good compromise)

# Commit interval (seconds between periodic syncs)
mount -o commit=60 /dev/sdb1 /mnt/data
# Default: 5 seconds
# Higher value = less frequent writes, more data loss potential

# Disable barriers (only if hardware has battery-backed cache)
mount -o nobarrier /dev/sdb1 /mnt/data
# Dangerous without protected cache!

# ext4 writeback mode (metadata only journaling)
mount -o data=writeback /dev/sdb1 /mnt/data
# Fastest, least safe

# Async mount (default, but explicit)
mount -o async /dev/sdb1 /mnt/data

# Optimal options for performance (SSD with integrity trade-off)
mount -o noatime,nodiratime,commit=60,data=writeback /dev/sdb1 /mnt/data
```

### SSD Optimizations

```bash
# Enable TRIM/discard
mount -o discard /dev/sdb1 /mnt/data
# Immediate TRIM on file deletion (may impact performance)

# Periodic TRIM (preferred for most SSDs)
# Enabled by fstrim.timer systemd unit
systemctl status fstrim.timer
systemctl enable fstrim.timer

# Manual TRIM
fstrim -v /mnt/data
fstrim -av                  # All mounted filesystems

# Check TRIM support
lsblk -D
# DISC-GRAN and DISC-MAX show TRIM granularity and max discard size

# XFS discard options
mount -o discard /dev/sdb1 /mnt/data           # Async discard

# Btrfs discard
mount -o discard=async /dev/sdb1 /mnt/data     # Async discard (recommended)
mount -o discard=sync /dev/sdb1 /mnt/data      # Sync discard

# F2FS (already SSD-optimized)
mount -o background_gc=on /dev/sdb1 /mnt/data

# Check SSD write amplification
# ZFS: zpool iostat -v
# Btrfs: btrfs device stats /mnt/data
```

### Block Size and Inode Sizing

```bash
# ext4 block size (set at creation)
mkfs.ext4 -b 4096 /dev/sdb1            # 4KB blocks (default, recommended)
mkfs.ext4 -b 1024 /dev/sdb1            # 1KB blocks (many small files)
# Larger blocks waste space with small files
# Smaller blocks add overhead for large files

# ext4 inode size
mkfs.ext4 -I 256 /dev/sdb1             # 256 bytes (default)
mkfs.ext4 -I 512 /dev/sdb1             # 512 bytes (more space for extended attributes)
# Larger inodes support more extended attributes and nanosecond timestamps

# ext4 inode ratio (bytes per inode)
mkfs.ext4 -i 16384 /dev/sdb1           # One inode per 16KB (default)
mkfs.ext4 -i 4096 /dev/sdb1            # One inode per 4KB (many small files)
# More inodes = less data space but supports more files

# XFS block size
mkfs.xfs -b size=4096 /dev/sdb1        # 4KB blocks

# Btrfs node size (metadata)
mkfs.btrfs -n 16384 /dev/sdb1          # 16KB (default)
mkfs.btrfs -n 32768 /dev/sdb1          # 32KB (large filesystems)

# ZFS recordsize (like block size)
zfs set recordsize=128K pool/dataset   # 128KB (large sequential I/O)
zfs set recordsize=8K pool/dataset     # 8KB (databases, random I/O)
zfs set recordsize=1M pool/dataset     # 1MB (video/large files)
```

### Journal Tuning

```bash
# ext4 journal size (set at creation)
mkfs.ext4 -J size=128 /dev/sdb1        # 128MB journal
# Larger journal = longer recovery, more write buffering

# ext4 journal options
mount -o journal_checksum /dev/sdb1 /mnt/data     # Enable checksums (safety)
mount -o journal_async_commit /dev/sdb1 /mnt/data # Async commit (performance)

# ext4 journal on separate device
mkfs.ext4 -J device=/dev/sdc1 /dev/sdb1

# Remove ext4 journal (convert to ext2)
tune2fs -O ^has_journal /dev/sdb1
# Only for read-heavy workloads on reliable systems

# XFS journal (log) size
mkfs.xfs -l size=128m /dev/sdb1        # 128MB log

# XFS log buffer tuning
mount -o logbufs=8,logbsize=256k /dev/sdb1 /mnt/data
# More buffers and larger size = better performance for write-heavy workloads
```

### Read-Ahead Tuning

Read-ahead prefetches data from disk to improve sequential read performance.

```bash
# View current read-ahead (in 512-byte sectors)
blockdev --getra /dev/sda
# Typical default: 256 (128 KB)

# Set read-ahead (in 512-byte sectors)
blockdev --setra 512 /dev/sda          # 256 KB
blockdev --setra 1024 /dev/sda         # 512 KB
blockdev --setra 4096 /dev/sda         # 2 MB

# Set permanently with udev rule (/etc/udev/rules.d/60-readahead.rules):
# ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{bdi/read_ahead_kb}="512"

# Recommendations:
# - Sequential workloads: Higher (2-4 MB)
# - Random workloads: Lower (128-256 KB)
# - SSDs: Moderate (256-512 KB)
# - HDDs: Higher (512 KB - 2 MB)
```

### Filesystem-Specific Optimizations

```bash
# ext4: Enable fast commits (kernel >= 5.10)
mount -o fast_commit /dev/sdb1 /mnt/data

# XFS: Allocation group count (parallelism)
mkfs.xfs -d agcount=16 /dev/sdb1
# More AGs = better parallel I/O

# Btrfs: Space cache v2 (better performance)
mount -o space_cache=v2 /dev/sdb1 /mnt/data

# Btrfs: Compression (can improve performance with fast CPU)
mount -o compress=lzo /dev/sdb1 /mnt/data      # Fast compression
mount -o compress=zstd:3 /dev/sdb1 /mnt/data   # Balanced

# ZFS: ARC size tuning
# Edit /etc/modprobe.d/zfs.conf:
# options zfs zfs_arc_max=8589934592  # 8GB
# options zfs zfs_arc_min=4294967296  # 4GB

# ZFS: Compression
zfs set compression=lz4 pool/dataset   # Fast, good ratio
zfs set compression=zstd pool/dataset  # Better ratio, slower

# F2FS: Background GC
mount -o background_gc=on /dev/sdb1 /mnt/data
```

## Common Patterns

### Container Storage with OverlayFS

```bash
# Manual container-like overlay setup
mkdir -p /var/lib/mycontainer/{lower,upper,work,merged}

# Lower layer: base image
mkdir -p /var/lib/mycontainer/lower/{bin,lib,etc}
# ... populate base system ...

# Mount overlay for container
mount -t overlay overlay \
  -o lowerdir=/var/lib/mycontainer/lower,\
     upperdir=/var/lib/mycontainer/upper,\
     workdir=/var/lib/mycontainer/work \
  /var/lib/mycontainer/merged

# Run process in container root
chroot /var/lib/mycontainer/merged /bin/bash

# Cleanup
umount /var/lib/mycontainer/merged
```

### RAM Disk Creation

```bash
# Create 1GB tmpfs RAM disk
mkdir /mnt/ramdisk
mount -t tmpfs -o size=1G tmpfs /mnt/ramdisk

# Use cases:
# - Temporary build directory
tmpfs /tmp/build tmpfs size=4G,mode=0755 0 0

# - Browser cache
tmpfs /home/user/.cache tmpfs size=2G,uid=1000,gid=1000 0 0

# - Application temporary files
mkdir /mnt/appcache
mount -t tmpfs -o size=512M,mode=0700 tmpfs /mnt/appcache
```

### Encrypted Filesystems

```bash
# Install cryptsetup (LUKS)
apt install cryptsetup

# Create encrypted device
cryptsetup luksFormat /dev/sdb1
# Enter passphrase

# Open encrypted device
cryptsetup luksOpen /dev/sdb1 encrypted_data
# Enter passphrase

# Create filesystem on encrypted device
mkfs.ext4 /dev/mapper/encrypted_data

# Mount
mount /dev/mapper/encrypted_data /mnt/encrypted

# Unmount and close
umount /mnt/encrypted
cryptsetup luksClose encrypted_data

# Add to /etc/crypttab for automatic mounting
# encrypted_data UUID=xxx none luks

# Add to /etc/fstab
# /dev/mapper/encrypted_data /mnt/encrypted ext4 defaults 0 2

# Key file instead of passphrase
dd if=/dev/urandom of=/root/keyfile bs=1024 count=4
chmod 0400 /root/keyfile
cryptsetup luksAddKey /dev/sdb1 /root/keyfile
# In /etc/crypttab:
# encrypted_data UUID=xxx /root/keyfile luks
```

### LVM Integration

```bash
# Create physical volume
pvcreate /dev/sdb1

# Create volume group
vgcreate myvg /dev/sdb1

# Create logical volume
lvcreate -L 10G -n mylv myvg

# Create filesystem
mkfs.ext4 /dev/myvg/mylv

# Mount
mount /dev/myvg/mylv /mnt/data

# LVM snapshots
lvcreate -L 2G -s -n mylv_snap /dev/myvg/mylv
mount /dev/myvg/mylv_snap /mnt/snapshot

# Extend logical volume
lvextend -L +5G /dev/myvg/mylv
resize2fs /dev/myvg/mylv

# Merge snapshot back (revert changes)
umount /mnt/data
lvconvert --merge /dev/myvg/mylv_snap
mount /dev/myvg/mylv /mnt/data
```

### Snapshot Workflows

```bash
# Btrfs snapshots for backups
btrfs subvolume snapshot -r /mnt/data /mnt/data/.snapshots/$(date +%Y%m%d)

# Automatic snapshot script
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
btrfs subvolume snapshot -r /mnt/data /mnt/data/.snapshots/$DATE
# Keep only last 7 days
find /mnt/data/.snapshots -maxdepth 1 -type d -mtime +7 -exec btrfs subvolume delete {} \;

# ZFS snapshots
zfs snapshot pool/data@$(date +%Y%m%d)

# Automatic ZFS snapshots (zfs-auto-snapshot package)
apt install zfs-auto-snapshot
# Creates frequent, hourly, daily, weekly, monthly snapshots

# List ZFS snapshots
zfs list -t snapshot

# Rollback to snapshot
zfs rollback pool/data@20250114
```

### Quota Management

```bash
# ext4 quotas
# Enable quotas at mount
mount -o usrquota,grpquota /dev/sdb1 /mnt/data

# Or in fstab
# /dev/sdb1 /mnt/data ext4 defaults,usrquota,grpquota 0 2

# Create quota files
quotacheck -cugm /mnt/data

# Enable quotas
quotaon /mnt/data

# Set quota for user
edquota -u username
# Or with command:
setquota -u username 10G 12G 0 0 /mnt/data
# soft=10GB hard=12GB, 0 inodes (unlimited)

# Set quota for group
setquota -g groupname 50G 55G 0 0 /mnt/data

# View quota
quota -u username
repquota -a

# XFS quotas (project quotas)
mount -o prjquota /dev/sdb1 /mnt/data

# Define project
echo "42:/mnt/data/project1" >> /etc/projects
echo "project1:42" >> /etc/projid

# Initialize project
xfs_quota -x -c 'project -s project1' /mnt/data

# Set project quota
xfs_quota -x -c 'limit -p bsoft=10g bhard=12g project1' /mnt/data

# View project quota
xfs_quota -x -c 'report -ph' /mnt/data

# ZFS quotas
zfs set quota=100G pool/data
zfs set refquota=50G pool/data              # Exclude snapshots
zfs set userquota@username=10G pool/data
zfs set groupquota@groupname=50G pool/data
```

### Extended Attributes and ACLs

```bash
# Extended attributes (xattr)
# Set attribute
setfattr -n user.comment -v "Important file" file.txt

# Get attribute
getfattr -n user.comment file.txt

# List all attributes
getfattr -d file.txt

# Remove attribute
setfattr -x user.comment file.txt

# ACLs (Access Control Lists)
# Set ACL
setfacl -m u:username:rw file.txt           # User permission
setfacl -m g:groupname:rx file.txt          # Group permission
setfacl -m o::r file.txt                    # Other permission

# View ACL
getfacl file.txt

# Remove specific ACL
setfacl -x u:username file.txt

# Remove all ACLs
setfacl -b file.txt

# Default ACLs (for directories)
setfacl -d -m u:username:rwx /mnt/data/dir  # New files inherit

# Copy ACLs
getfacl source.txt | setfacl --set-file=- dest.txt

# Enable ACLs at mount (usually enabled by default)
mount -o acl /dev/sdb1 /mnt/data
```

### Network Filesystems

```bash
# NFS mount
mount -t nfs server.example.com:/export /mnt/nfs

# NFS with options
mount -t nfs -o vers=4.2,soft,timeo=30,retrans=3 server:/export /mnt/nfs

# CIFS/SMB mount
mount -t cifs //server/share /mnt/smb -o username=user,password=pass

# CIFS with credentials file
# Create /root/.smbcredentials:
# username=user
# password=pass
chmod 0600 /root/.smbcredentials
mount -t cifs //server/share /mnt/smb -o credentials=/root/.smbcredentials,uid=1000,gid=1000

# SSHFS (FUSE)
sshfs user@server:/remote/path /mnt/sshfs
fusermount -u /mnt/sshfs  # Unmount
```

### Loop Device Mounting

```bash
# Mount disk image
mount -o loop disk.img /mnt/image

# Mount ISO
mount -o loop ubuntu.iso /mnt/iso

# Create disk image
dd if=/dev/zero of=disk.img bs=1M count=1024  # 1GB
mkfs.ext4 disk.img
mount -o loop disk.img /mnt/image

# Automatic loop device
losetup -f                          # Find free loop device
losetup /dev/loop0 disk.img         # Attach
mount /dev/loop0 /mnt/image
umount /mnt/image
losetup -d /dev/loop0               # Detach

# Multiple partitions in image
losetup -P /dev/loop0 disk.img      # Scan partitions
mount /dev/loop0p1 /mnt/part1
```

## Best Practices

### Filesystem Selection

```bash
# General purpose (root, home):
# - ext4: Stable, well-tested, good performance
# - XFS: Better for large files, cannot shrink

# Large files and databases:
# - XFS: Excellent performance
# - ext4: Also good

# Snapshots and advanced features:
# - Btrfs: Built-in snapshots, compression, RAID
# - ZFS: Most advanced features, requires more RAM

# SSDs and flash:
# - F2FS: Optimized for flash
# - ext4: Also works well with TRIM

# Containers:
# - OverlayFS: Standard for Docker
# - Btrfs: Alternative with native snapshots
# - ZFS: Alternative with native snapshots (requires setup)

# Temporary/cache:
# - tmpfs: RAM-based, very fast

# Removable media:
# - FAT32/exFAT: Windows/Mac compatibility
# - ext4: Linux-only, better features
```

### Security

```bash
# Mount options for security
# /tmp and /var/tmp should prevent execution
tmpfs /tmp tmpfs defaults,noexec,nodev,nosuid 0 0

# User-writable locations
/dev/sdb1 /mnt/usb ext4 defaults,noexec,nodev,nosuid 0 0

# Network filesystems
server:/export /mnt/nfs nfs defaults,nosuid,nodev,_netdev 0 0

# Filesystem encryption
# Use LUKS for block device encryption

# Secure deletion
# Some filesystems support secure deletion via extended attributes
chattr +s file.txt  # ext4 (may not be effective on SSDs)

# Immutable files (prevent deletion/modification)
chattr +i file.txt
chattr -i file.txt  # Remove immutable flag
```

### Partition Alignment

```bash
# Modern tools (parted, fdisk >= 2.26) align automatically to 1MiB
# This is optimal for most disks

# Check alignment
parted /dev/sdb align-check opt 1

# Create aligned partition with parted
parted /dev/sdb
(parted) mklabel gpt
(parted) mkpart primary 0% 100%
(parted) align-check opt 1

# Manual alignment (rarely needed)
# Start at 2048 sectors (1MiB) for traditional tools
fdisk /dev/sdb
# First sector: 2048
```

### Backup Strategies

```bash
# Filesystem snapshots
# - Instant, space-efficient
# - Btrfs: btrfs subvolume snapshot
# - ZFS: zfs snapshot
# - LVM: lvcreate -s

# File-level backups
# - rsync: Incremental, efficient
# - tar: Archives, compression
# - restic/borg: Deduplication, encryption

# Block-level backups
# - dd: Raw copy (slow, complete)
# - partclone: Filesystem-aware (faster)

# Remote backups
# - Btrfs send/receive
# - ZFS send/receive
# - rsync over SSH

# Example: Btrfs incremental backup
btrfs subvolume snapshot -r /data /data/.snap/$(date +%Y%m%d)
btrfs send -p /data/.snap/20250113 /data/.snap/20250114 | ssh backup 'btrfs receive /backup/'

# Example: ZFS incremental backup
zfs snapshot pool/data@$(date +%Y%m%d)
zfs send -i pool/data@20250113 pool/data@20250114 | ssh backup 'zfs receive pool/backup'
```

### Monitoring

```bash
# Regular checks
# - Disk space: df -h
# - Inode usage: df -i
# - Filesystem errors: dmesg | grep -i error
# - SMART status: smartctl -a /dev/sda

# Automated monitoring
# - Set up alerts for low disk space
# - Monitor filesystem errors in logs
# - Schedule regular scrubs (Btrfs, ZFS)

# Example: Disk space alert
df -h | awk '$5+0 > 90 {print "Warning: " $1 " is " $5 " full"}'

# Btrfs scrub schedule (monthly)
# Systemd timer: /etc/systemd/system/btrfs-scrub.timer
# [Timer]
# OnCalendar=monthly

# ZFS scrub schedule (monthly)
# Cron: 0 0 1 * * zpool scrub pool
```

## Troubleshooting

### Filesystem Corruption

```bash
# Symptoms:
# - I/O errors in dmesg
# - Mount failures
# - Read-only remount
# - File access errors

# Check dmesg for errors
dmesg | tail -50
dmesg | grep -i error

# Unmount filesystem
umount /mnt/data
# If busy:
lsof /mnt/data
fuser -km /mnt/data
umount /mnt/data

# Check and repair
fsck.ext4 -f /dev/sdb1     # ext4
xfs_repair /dev/sdb1       # XFS
btrfs check /dev/sdb1      # Btrfs (unmounted only)

# If automatic repair fails
fsck.ext4 -y /dev/sdb1     # Answer yes to all

# Last resort (ext4)
fsck.ext4 -b 32768 /dev/sdb1  # Use backup superblock

# Check SMART status for hardware issues
smartctl -a /dev/sda
```

### Read-Only Filesystem

```bash
# Causes:
# - Filesystem errors detected
# - Mount option explicitly ro
# - Write error triggering remount-ro

# Check mount options
mount | grep /mnt/data

# Check filesystem errors
dmesg | grep -i "read-only"

# Remount read-write
mount -o remount,rw /mnt/data

# If remount fails, filesystem check needed
umount /mnt/data
fsck /dev/sdb1
mount /dev/sdb1 /mnt/data
```

### Mount Failures

```bash
# Error: mount: unknown filesystem type 'xfs'
# Solution: Install filesystem tools
apt install xfsprogs     # XFS
apt install btrfs-progs  # Btrfs
apt install zfsutils-linux  # ZFS

# Error: mount: wrong fs type, bad option, bad superblock
# Solution 1: Specify filesystem type
mount -t ext4 /dev/sdb1 /mnt/data

# Solution 2: Check superblock
dumpe2fs /dev/sdb1 | grep superblock  # ext4
xfs_db -r -c "sb 0" -c "p" /dev/sdb1  # XFS

# Solution 3: Try backup superblock
mount -o sb=32768 /dev/sdb1 /mnt/data

# Error: device or resource busy
# Solution: Find and stop processes
lsof /mnt/data
fuser -m /mnt/data
fuser -km /mnt/data  # Kill processes

# Error: structure needs cleaning
# Solution: Run fsck
fsck /dev/sdb1
```

### No Space Left on Device

```bash
# Check disk space
df -h /mnt/data

# Check inode usage (can run out even with space available)
df -i /mnt/data

# If inodes exhausted:
# Find directories with many files
find /mnt/data -xdev -type d -exec sh -c 'echo "$(ls -a {} | wc -l) {}"' \; | sort -n | tail

# Solutions:
# - Delete unnecessary files
# - Recreate filesystem with more inodes
mkfs.ext4 -i 4096 /dev/sdb1  # One inode per 4KB

# Check for deleted but open files (still consuming space)
lsof +L1 /mnt/data
# Kill or restart processes holding deleted files
```

### Permission Denied with ACLs

```bash
# Check ACLs
getfacl /mnt/data/file.txt

# Check if filesystem supports ACLs
mount | grep /mnt/data

# Enable ACLs at mount
mount -o remount,acl /mnt/data

# Or in /etc/fstab
/dev/sdb1 /mnt/data ext4 defaults,acl 0 2

# Reset ACLs
setfacl -b /mnt/data/file.txt
```

### OverlayFS Issues

See [OverlayFS Troubleshooting](#overlayfs-troubleshooting) section above for detailed OverlayFS-specific issues.

### Performance Degradation

```bash
# Check I/O statistics
iostat -x 1 5                # 5 samples, 1 second apart

# Look for:
# - High %util: Device saturated
# - High await: I/O latency
# - High r_await/w_await: Read/write latency

# Check for fragmentation (ext4)
e4defrag -c /mnt/data        # Check fragmentation
e4defrag /mnt/data           # Defragment

# XFS fragmentation
xfs_db -r -c frag /dev/sdb1  # Check
xfs_fsr /mnt/data            # Defragment

# Btrfs defragmentation
btrfs filesystem defragment -r /mnt/data

# Check for failing disk
smartctl -a /dev/sda
smartctl -t short /dev/sda   # Run short test
smartctl -t long /dev/sda    # Run long test

# Check for swap thrashing
free -h
vmstat 1

# Check for inode exhaustion
df -i
```

## Quick Reference

### Filesystem Comparison

| Feature | ext4 | XFS | Btrfs | ZFS | F2FS |
|---------|------|-----|-------|-----|------|
| Stability | Excellent | Excellent | Good | Excellent | Good |
| Max File Size | 16 TiB | 8 EiB | 16 EiB | 16 EiB | 3.94 TiB |
| Max Volume Size | 1 EiB | 8 EiB | 16 EiB | 256 ZiB | 3.94 TiB |
| Journaling | Yes | Yes | CoW | CoW | Yes |
| Snapshots | No | No | Yes | Yes | No |
| Compression | No | No | Yes | Yes | Yes |
| Deduplication | No | No | Limited | Yes | No |
| Online Resize | Grow | Grow | Both | N/A | No |
| RAID Support | No | No | Yes | Yes | No |
| SSD Optimization | TRIM | TRIM | TRIM | TRIM | Native |
| Maturity | Mature | Mature | Maturing | Mature | Newer |

### Common Mount Options

| Option | Description |
|--------|-------------|
| `ro` | Mount read-only |
| `rw` | Mount read-write (default) |
| `noatime` | Don't update access times |
| `nodiratime` | Don't update directory access times |
| `relatime` | Update access time if older than modify time (default) |
| `noexec` | Prevent execution of binaries |
| `nodev` | Ignore device files |
| `nosuid` | Ignore setuid/setgid bits |
| `sync` | Synchronous I/O |
| `async` | Asynchronous I/O (default) |
| `user` | Allow user to mount |
| `noauto` | Don't mount with `mount -a` |
| `nofail` | Don't fail boot if device missing |
| `_netdev` | Network device (wait for network) |

### Command Cheat Sheet

```bash
# Filesystem creation
mkfs.ext4 /dev/sdb1
mkfs.xfs /dev/sdb1
mkfs.btrfs /dev/sdb1
mkfs.vfat /dev/sdb1

# Mounting
mount /dev/sdb1 /mnt/data
mount -t ext4 -o noatime /dev/sdb1 /mnt/data
umount /mnt/data

# Checking
fsck.ext4 /dev/sdb1
xfs_repair /dev/sdb1
btrfs check /dev/sdb1

# Resizing
resize2fs /dev/sdb1
xfs_growfs /mnt/data
btrfs filesystem resize max /mnt/data

# Information
df -h
df -i
lsblk -f
blkid
findmnt

# Tuning
tune2fs -l /dev/sdb1
tune2fs -L mylabel /dev/sdb1
xfs_admin -l /dev/sdb1

# Btrfs specific
btrfs subvolume create /mnt/data/subvol
btrfs subvolume snapshot /mnt/data /mnt/data/snap
btrfs filesystem usage /mnt/data

# ZFS specific
zpool create pool /dev/sdb
zfs create pool/dataset
zfs snapshot pool/dataset@snap
zfs send pool/dataset@snap | zfs receive backup/dataset

# OverlayFS
mount -t overlay overlay -o lowerdir=/lower,upperdir=/upper,workdir=/work /merged
```

### Filesystem Limits

| Filesystem | Max File Size | Max Volume Size | Max Filename | Max Path |
|------------|---------------|-----------------|--------------|----------|
| ext4 | 16 TiB | 1 EiB | 255 bytes | 4096 bytes |
| XFS | 8 EiB | 8 EiB | 255 bytes | 4096 bytes |
| Btrfs | 16 EiB | 16 EiB | 255 bytes | 4096 bytes |
| ZFS | 16 EiB | 256 ZiB | 255 bytes | No limit |
| FAT32 | 4 GiB | 2 TiB | 255 chars | No limit |
| exFAT | 16 EiB | 64 ZiB | 255 chars | No limit |
| NTFS | 16 EiB | 16 EiB | 255 chars | 32767 chars |

### Performance Characteristics

| Filesystem | Sequential Read | Sequential Write | Random Read | Random Write | Metadata |
|------------|-----------------|------------------|-------------|--------------|----------|
| ext4 | Excellent | Excellent | Very Good | Very Good | Good |
| XFS | Excellent | Excellent | Very Good | Very Good | Excellent |
| Btrfs | Very Good | Good | Good | Good | Good |
| ZFS | Excellent | Good | Very Good | Good | Very Good |
| F2FS | Very Good | Excellent | Very Good | Excellent | Good |
| tmpfs | Excellent | Excellent | Excellent | Excellent | Excellent |

Note: Performance varies greatly based on hardware, configuration, and workload. These are general characteristics.
