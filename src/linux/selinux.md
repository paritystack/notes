# SELinux (Security-Enhanced Linux)

SELinux is a mandatory access control (MAC) security mechanism implemented in the Linux kernel using the Linux Security Modules (LSM) framework. It provides a powerful and flexible security architecture that enforces access control policies on processes, files, ports, and other system resources beyond traditional discretionary access control (DAC).

## Table of Contents

- [Overview and Architecture](#overview-and-architecture)
  - [MAC vs DAC](#mac-vs-dac)
  - [SELinux Architecture](#selinux-architecture)
  - [Core Concepts](#core-concepts)
  - [Policy Types](#policy-types)
  - [Operating Modes](#operating-modes)
  - [Security Models](#security-models)
- [Basic Operations](#basic-operations)
  - [Status and Mode Management](#status-and-mode-management)
  - [Context Operations](#context-operations)
  - [File Context Management](#file-context-management)
  - [Port Context Management](#port-context-management)
  - [Boolean Management](#boolean-management)
  - [Module Management](#module-management)
  - [User and Role Management](#user-and-role-management)
- [Common Patterns](#common-patterns)
  - [Web Server Configuration](#web-server-configuration)
  - [Database Server Contexts](#database-server-contexts)
  - [Container Integration](#container-integration)
  - [Network File Sharing](#network-file-sharing)
  - [SSH Configuration](#ssh-configuration)
  - [systemd Service Contexts](#systemd-service-contexts)
  - [Custom Application Labeling](#custom-application-labeling)
  - [Home Directory Management](#home-directory-management)
- [Policy Development](#policy-development)
  - [Understanding Audit Logs](#understanding-audit-logs)
  - [Creating Custom Modules](#creating-custom-modules)
  - [Policy Module Structure](#policy-module-structure)
  - [Type Enforcement Rules](#type-enforcement-rules)
  - [Domain Transitions](#domain-transitions)
  - [File Context Specifications](#file-context-specifications)
  - [Policy Compilation and Loading](#policy-compilation-and-loading)
  - [Interface Development](#interface-development)
- [Troubleshooting](#troubleshooting)
  - [Understanding AVC Denials](#understanding-avc-denials)
  - [Troubleshooting Tools](#troubleshooting-tools)
  - [Common Denial Patterns](#common-denial-patterns)
  - [Debugging Workflows](#debugging-workflows)
  - [Performance Considerations](#performance-considerations)
- [Advanced Topics](#advanced-topics)
  - [Network Labeling](#network-labeling)
  - [Multi-Level Security (MLS/MCS)](#multi-level-security-mlsmcs)
  - [Confined Users](#confined-users)
  - [Sandbox Environments](#sandbox-environments)
  - [Policy Constraints](#policy-constraints)
  - [SELinux with Containers](#selinux-with-containers)
  - [Integration with Namespaces](#integration-with-namespaces)
- [Best Practices](#best-practices)
  - [Policy Development Workflow](#policy-development-workflow)
  - [Testing Strategies](#testing-strategies)
  - [Security Hardening](#security-hardening)
  - [Common Pitfalls](#common-pitfalls)
  - [Performance Optimization](#performance-optimization)
- [Resources](#resources)

---

## Overview and Architecture

### MAC vs DAC

SELinux implements Mandatory Access Control (MAC), which differs fundamentally from the traditional Discretionary Access Control (DAC) used by Unix permissions:

**Discretionary Access Control (DAC)**:
- Owner of a resource controls access permissions
- Users can grant access to their own resources
- Permissions: read, write, execute (rwx)
- Vulnerable to privilege escalation and compromised user accounts

**Mandatory Access Control (MAC)**:
- System-wide security policy enforced by the kernel
- Users cannot override or bypass security policies
- Fine-grained control over all system interactions
- Defense in depth: even if a process is compromised, its capabilities are limited

```
Traditional DAC:
User → File Permission Check → Access Granted/Denied

SELinux MAC:
User → DAC Check → SELinux Policy Check → Access Granted/Denied
         ↓              ↓
     Must Pass      Must Pass
```

### SELinux Architecture

SELinux is built on the Linux Security Modules (LSM) framework and consists of several key components:

```
┌──────────────────────────────────────────────────────┐
│                   User Space                         │
├──────────────────────────────────────────────────────┤
│  Applications  │  SELinux Tools  │  Policy Files    │
│  (httpd, sshd) │  (semanage,     │  (/etc/selinux/) │
│                │   restorecon)   │                  │
└────────┬───────┴────────┬────────┴──────────┬───────┘
         │                │                   │
         │ System Calls   │ Policy Queries    │ Policy Load
         ↓                ↓                   ↓
┌──────────────────────────────────────────────────────┐
│                    Kernel Space                      │
├──────────────────────────────────────────────────────┤
│                 LSM Hook Framework                   │
│                         ↓                            │
│  ┌──────────────────────────────────────────────┐   │
│  │           SELinux Security Server            │   │
│  │  ┌────────────┐  ┌──────────────────────┐   │   │
│  │  │    AVC     │  │   Policy Engine      │   │   │
│  │  │  (Access   │←→│  (Policy Database)   │   │   │
│  │  │   Vector   │  │                      │   │   │
│  │  │   Cache)   │  │  - Type Enforcement  │   │   │
│  │  └────────────┘  │  - RBAC Rules        │   │   │
│  │                  │  - MLS/MCS Rules     │   │   │
│  │                  └──────────────────────┘   │   │
│  └──────────────────────────────────────────────┘   │
│                         ↓                            │
│            Access Decision (Allow/Deny)             │
└──────────────────────────────────────────────────────┘
```

**Key Components**:

1. **LSM Hooks**: Kernel hooks that intercept security-relevant system calls
2. **Access Vector Cache (AVC)**: Caches access decisions for performance
3. **Security Server**: Makes access control decisions based on policy
4. **Policy Engine**: Evaluates the loaded security policy
5. **Security Context**: Labels (user:role:type:level) on all subjects and objects

### Core Concepts

#### Security Contexts

Every process (subject) and resource (object) in SELinux has a security context consisting of four fields:

```
user:role:type:level

Example: system_u:object_r:httpd_sys_content_t:s0
         ↑        ↑        ↑                    ↑
         |        |        |                    |
      SELinux   Role    Type/Domain          MLS Level
       User
```

**Fields**:
- **User**: SELinux user (not the same as Linux user) - constrains which roles can be entered
- **Role**: Intermediary between users and types - implements RBAC
- **Type**: The primary attribute used for Type Enforcement (TE) - defines what a process can access
- **Level**: Multi-Level Security (MLS) or Multi-Category Security (MCS) level

```bash
# View context of files
ls -Z /var/www/html/
# -rw-r--r--. root root unconfined_u:object_r:httpd_sys_content_t:s0 index.html

# View context of processes
ps -eZ | grep httpd
# system_u:system_r:httpd_t:s0    1234 ?        00:00:01 httpd

# View your own context
id -Z
# unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023
```

#### Subjects and Objects

**Subjects** (active entities):
- Processes and their domains
- Each process runs in a specific domain (type)
- Domain defines what the process can do

**Objects** (passive entities):
- Files, directories
- Sockets, pipes
- Network ports
- Devices

```bash
# Subject: httpd process running in httpd_t domain
ps -eZ | grep httpd
# system_u:system_r:httpd_t:s0    1234 ?        00:00:01 httpd

# Object: File with httpd_sys_content_t type
ls -Z /var/www/html/index.html
# unconfined_u:object_r:httpd_sys_content_t:s0 /var/www/html/index.html
```

#### Type Enforcement (TE)

Type Enforcement is the primary access control mechanism in SELinux:

- Each subject (process) has a **domain type**
- Each object (file, port) has a **type**
- Policy rules define which domains can access which types
- Default deny: only explicitly allowed operations are permitted

```bash
# Example: httpd_t domain can read httpd_sys_content_t files
# Policy rule (simplified):
# allow httpd_t httpd_sys_content_t:file { read open getattr };

# This works:
# httpd (httpd_t) → read → /var/www/html/index.html (httpd_sys_content_t) ✓

# This is denied:
# httpd (httpd_t) → read → /etc/shadow (shadow_t) ✗
```

### Policy Types

SELinux supports different policy types with varying levels of confinement:

#### Targeted Policy (Default on RHEL/CentOS/Fedora)

```bash
# Check current policy
sestatus | grep "Loaded policy"
# Loaded policy name:             targeted
```

**Characteristics**:
- **Targeted processes**: Only specific network-facing and privileged processes are confined
- **Unconfined processes**: Most user processes run in `unconfined_t` domain
- **Balance**: Security for critical services without restricting normal user activities
- **Examples of confined domains**: httpd_t, sshd_t, mysqld_t, named_t

```bash
# Confined process
ps -eZ | grep httpd
# system_u:system_r:httpd_t:s0    1234 ?        00:00:01 httpd

# Unconfined process
ps -eZ | grep bash
# unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023 2345 pts/0 00:00:00 bash
```

#### Strict Policy

**Characteristics**:
- **All processes** are confined (no unconfined_t domain)
- **Maximum security**: Every process has a specific domain with limited permissions
- **Complex**: Requires careful policy customization
- **Rarely used**: Too restrictive for general-purpose systems

#### MLS (Multi-Level Security) Policy

**Characteristics**:
- Implements **Bell-LaPadula** model for classified information
- **Sensitivity levels**: top-secret, secret, confidential, unclassified
- **Categories**: compartments for additional separation
- **Use case**: Government and military systems with classified data

```bash
# MLS context example
# user:role:type:sensitivity[:category,...]
# user_u:user_r:user_t:s1:c0,c1

# s0 = lowest level (unclassified)
# s15 = highest level (top secret)
# c0-c1023 = categories
```

### Operating Modes

SELinux operates in three modes:

#### Enforcing Mode

```bash
# Check current mode
getenforce
# Enforcing

# All SELinux denials are enforced
# Violations are logged and blocked
```

**Behavior**:
- SELinux policy is fully enforced
- Access violations are **denied** and **logged**
- System is protected according to policy

#### Permissive Mode

```bash
# Set to permissive mode temporarily
setenforce 0

getenforce
# Permissive
```

**Behavior**:
- SELinux policy is **NOT enforced**
- Violations are **allowed** but **logged**
- Useful for **debugging** and policy development
- System runs normally but logs what would be denied

**Use cases**:
- Testing new policies
- Troubleshooting access issues
- Identifying required permissions

#### Disabled Mode

```bash
# Check status
sestatus
# SELinux status: disabled
```

**Behavior**:
- SELinux is completely disabled
- No policy enforcement
- No logging of violations
- Requires reboot to enable/disable

**Warning**: Switching between disabled and enabled modes requires a **complete filesystem relabel**:

```bash
# Re-enable SELinux (requires editing config and reboot)
vi /etc/selinux/config
# SELINUX=enforcing

# This will trigger auto-relabel on next boot
touch /.autorelabel
reboot
```

### Security Models

SELinux implements multiple security models simultaneously:

#### Type Enforcement (TE)

The primary and most commonly used model:

```bash
# Policy rule structure
# allow <source_domain> <target_type>:<object_class> { <permissions> };

# Example: Allow httpd to read web content
allow httpd_t httpd_sys_content_t:file { read open getattr };

# Allow httpd to bind to HTTP port
allow httpd_t http_port_t:tcp_socket { bind };

# Allow httpd to connect to database
allow httpd_t mysqld_port_t:tcp_socket { name_connect };
```

**Key concepts**:
- **Domains**: Types for processes (httpd_t, sshd_t)
- **Types**: Labels for objects (httpd_sys_content_t, etc_t)
- **Allow rules**: Explicit permissions required for access
- **Default deny**: Everything not explicitly allowed is denied

#### Role-Based Access Control (RBAC)

Defines which roles users can assume and which domains those roles can enter:

```bash
# User → Role → Domain hierarchy

# Example mapping:
# user_u (SELinux user)
#   ├── user_r (role)
#   │     └── user_t (domain)
#   └── sysadm_r (role)
#         └── sysadm_t (domain)

# List user-role mappings
semanage user -l
# SELinux User    MLS/MCS Level    Roles
# root            s0-s0:c0.c1023   staff_r sysadm_r system_r unconfined_r
# staff_u         s0-s0:c0.c1023   staff_r sysadm_r
# user_u          s0               user_r
```

**Roles**:
- `object_r`: Default role for files and objects
- `system_r`: System processes and daemons
- `user_r`: Regular users with limited access
- `staff_r`: Staff users with some administrative capabilities
- `sysadm_r`: System administrators
- `unconfined_r`: Unconfined role (targeted policy)

#### Multi-Level Security (MLS)

Implements information flow control based on security clearances:

```bash
# MLS Levels (s0-s15)
# s0 = Unclassified
# s1 = Confidential
# s2 = Secret
# s3 = Top Secret
# ...

# MLS Categories (c0-c1023)
# Used for compartmentalization

# Example context with MLS:
# user_u:user_r:user_t:s1:c0,c1
#                        ↑  ↑
#                    Level Categories

# Bell-LaPadula Rules:
# - No read up: Process at s1 cannot read s2 files
# - No write down: Process at s2 cannot write to s1 files
```

#### Multi-Category Security (MCS)

A simplified version of MLS used in the targeted policy:

```bash
# Default on RHEL/CentOS
# Uses categories (c0-c1023) without sensitivity levels

# Example: Container isolation
# Container 1: system_u:system_r:svirt_lxc_net_t:s0:c123,c456
# Container 2: system_u:system_r:svirt_lxc_net_t:s0:c789,c012

# Containers cannot access each other's files due to different categories
```

---

## Basic Operations

### Status and Mode Management

#### Check SELinux Status

```bash
# Quick status check
getenforce
# Enforcing

# Detailed status
sestatus
# SELinux status:                 enabled
# SELinuxfs mount:                /sys/fs/selinux
# SELinux root directory:         /etc/selinux
# Loaded policy name:             targeted
# Current mode:                   enforcing
# Mode from config file:          enforcing
# Policy MLS status:              enabled
# Policy deny_unknown status:     allowed
# Memory protection checking:     actual (secure)
# Max kernel policy version:      31

# Check SELinux configuration
cat /etc/selinux/config
# SELINUX=enforcing
# SELINUXTYPE=targeted
```

#### Change SELinux Mode

```bash
# Set to permissive mode temporarily (until reboot)
setenforce 0
getenforce
# Permissive

# Set to enforcing mode temporarily
setenforce 1
getenforce
# Enforcing

# Permanent mode change (edit config and reboot)
vi /etc/selinux/config
# Change: SELINUX=enforcing
# or:     SELINUX=permissive
# or:     SELINUX=disabled

# Apply changes (reboot required)
reboot
```

#### Set Permissive Mode for Specific Domains

```bash
# Make only httpd_t domain permissive (everything else enforcing)
semanage permissive -a httpd_t

# List permissive domains
semanage permissive -l
# Customized Permissive Types
# httpd_t

# Remove permissive status
semanage permissive -d httpd_t

# This is useful for debugging specific services without disabling SELinux globally
```

### Context Operations

#### View Security Contexts

```bash
# View file contexts
ls -Z /var/www/html/
# -rw-r--r--. root root unconfined_u:object_r:httpd_sys_content_t:s0 index.html

ls -lZ /etc/passwd
# -rw-r--r--. root root system_u:object_r:passwd_file_t:s0 /etc/passwd

# View directory contexts recursively
ls -lZR /var/log/ | head -20

# View process contexts
ps -eZ
# LABEL                             PID TTY      TIME CMD
# system_u:system_r:init_t:s0         1 ?        00:00:02 systemd
# system_u:system_r:kernel_t:s0       2 ?        00:00:00 kthreadd

ps -eZ | grep httpd
# system_u:system_r:httpd_t:s0      1234 ?        00:00:01 httpd

# View your own context
id -Z
# unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023

# View context of current process
cat /proc/self/attr/current
# unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023

# View port contexts
semanage port -l | grep http
# http_cache_port_t              tcp      8080, 8118, 8123, 10001-10010
# http_port_t                    tcp      80, 81, 443, 488, 8008, 8009, 8443, 9000

# View network interface contexts
semanage interface -l

# View node contexts (for network labeling)
semanage node -l
```

#### Change Contexts Temporarily

```bash
# Change file context temporarily (lost on relabel)
chcon -t httpd_sys_content_t /var/www/html/index.html

# Change context with user:role:type
chcon -u system_u -r object_r -t httpd_sys_content_t /var/www/html/test.html

# Copy context from reference file
chcon --reference=/var/www/html/index.html /var/www/html/newfile.html

# Change context recursively
chcon -R -t httpd_sys_content_t /var/www/html/

# Warning: chcon changes are temporary and will be lost if you run restorecon
# or if the system performs an automatic relabel
```

#### Restore Default Contexts

```bash
# Restore context for a single file (based on policy)
restorecon -v /var/www/html/index.html
# Relabeled /var/www/html/index.html from unconfined_u:object_r:user_home_t:s0 to system_u:object_r:httpd_sys_content_t:s0

# Restore contexts recursively
restorecon -Rv /var/www/html/
# Relabeled /var/www/html/page1.html from user_home_t to httpd_sys_content_t
# Relabeled /var/www/html/page2.html from user_home_t to httpd_sys_content_t

# Check what would be changed without making changes
restorecon -Rvn /var/www/

# Restore context and show progress
restorecon -Rvp /var/

# Force relabel even if context appears correct
restorecon -F -Rv /var/www/html/
```

### File Context Management

File context specifications define the default contexts for files based on path patterns.

#### View File Context Rules

```bash
# Show file context specification for a path
semanage fcontext -l | grep '/var/www'
# /var/www(/.*)?                all files   system_u:object_r:httpd_sys_content_t:s0
# /var/www/html(/.*)?           all files   system_u:object_r:httpd_sys_content_t:s0
# /var/www/cgi-bin(/.*)?        all files   system_u:object_r:httpd_sys_script_exec_t:s0

# Check expected context for a path
matchpathcon /var/www/html/index.html
# /var/www/html/index.html  system_u:object_r:httpd_sys_content_t:s0

# Compare current vs expected context
matchpathcon -V /var/www/html/index.html
# /var/www/html/index.html verified.
# or
# /var/www/html/index.html has context unconfined_u:object_r:user_home_t:s0, should be system_u:object_r:httpd_sys_content_t:s0
```

#### Add File Context Rules

```bash
# Add context rule for custom web directory
semanage fcontext -a -t httpd_sys_content_t "/web(/.*)?"

# Add context for specific file type
semanage fcontext -a -t httpd_sys_script_exec_t "/web/cgi-bin(/.*)?"

# Add context with specific user and role
semanage fcontext -a -s system_u -r object_r -t httpd_sys_content_t "/custom/www(/.*)?"

# Apply the new context rule
restorecon -Rv /web/

# Add writable directory context for web server
semanage fcontext -a -t httpd_sys_rw_content_t "/web/uploads(/.*)?"
restorecon -Rv /web/uploads/
```

#### Modify File Context Rules

```bash
# Modify existing context rule
semanage fcontext -m -t httpd_sys_content_t "/data/website(/.*)?"

# Modify and apply
semanage fcontext -m -t httpd_sys_rw_content_t "/var/www/uploads(/.*)?"
restorecon -Rv /var/www/uploads/
```

#### Delete File Context Rules

```bash
# Delete custom context rule
semanage fcontext -d "/web(/.*)?"

# After deletion, restore to default
restorecon -Rv /web/

# List all customized file contexts (non-default)
semanage fcontext -l -C
```

#### Equivalence Rules

```bash
# Make /web equivalent to /var/www (inherit same contexts)
semanage fcontext -a -e /var/www /web

# Now /web automatically gets same contexts as /var/www
ls -Zd /web
# system_u:object_r:httpd_sys_content_t:s0 /web

# List all equivalence rules
semanage fcontext -l | grep "= "

# Delete equivalence
semanage fcontext -d -e /var/www /web
```

### Port Context Management

SELinux labels network ports to control which services can bind to which ports.

#### View Port Contexts

```bash
# List all port contexts
semanage port -l

# Show HTTP-related ports
semanage port -l | grep http
# http_cache_port_t              tcp      8080, 8118, 8123, 10001-10010
# http_port_t                    tcp      80, 81, 443, 488, 8008, 8009, 8443, 9000

# Show SSH ports
semanage port -l | grep ssh
# ssh_port_t                     tcp      22

# Show database ports
semanage port -l | grep -E '(mysql|postgresql)'
# mysqld_port_t                  tcp      1186, 3306, 63132-63164
# postgresql_port_t              tcp      5432, 9898
```

#### Add Port Labels

```bash
# Allow httpd to bind to port 8080 (if not already labeled)
semanage port -a -t http_port_t -p tcp 8080

# Add custom port for SSH
semanage port -a -t ssh_port_t -p tcp 2222

# Add port range
semanage port -a -t http_port_t -p tcp 8000-8010

# Now httpd can bind to these ports
systemctl restart httpd
```

#### Modify Port Labels

```bash
# Change port label type
semanage port -m -t http_port_t -p tcp 8080

# Modify port range
semanage port -m -t http_port_t -p tcp 8000-8100
```

#### Delete Port Labels

```bash
# Remove custom port label
semanage port -d -t http_port_t -p tcp 8080

# Remove port range
semanage port -d -t http_port_t -p tcp 8000-8010

# List customized ports only
semanage port -l -C
```

### Boolean Management

SELinux booleans are on/off switches that modify policy behavior without recompiling.

#### List Booleans

```bash
# List all booleans
getsebool -a
# abrt_anon_write --> off
# abrt_handle_event --> off
# httpd_can_network_connect --> off
# httpd_can_network_connect_db --> off

# List booleans with descriptions
semanage boolean -l
# httpd_can_network_connect      (off, off)  Allow httpd to can network connect
# httpd_enable_homedirs          (off, off)  Allow httpd to enable homedirs

# Search for specific booleans
getsebool -a | grep httpd
# httpd_anon_write --> off
# httpd_builtin_scripting --> on
# httpd_can_network_connect --> off
# httpd_can_network_connect_db --> off
# httpd_can_network_relay --> off
# httpd_can_sendmail --> off
# httpd_enable_cgi --> on
# httpd_enable_ftp_server --> off
# httpd_enable_homedirs --> off

# Get specific boolean value
getsebool httpd_can_network_connect
# httpd_can_network_connect --> off
```

#### Set Booleans

```bash
# Enable boolean temporarily (until reboot)
setsebool httpd_can_network_connect on

# Verify change
getsebool httpd_can_network_connect
# httpd_can_network_connect --> on

# Enable boolean permanently (persists across reboots)
setsebool -P httpd_can_network_connect on

# Disable boolean permanently
setsebool -P httpd_enable_homedirs off

# Set multiple booleans
setsebool -P httpd_can_network_connect on httpd_can_network_connect_db on
```

#### Common Boolean Use Cases

```bash
# Allow web server to connect to network (proxy, external APIs)
setsebool -P httpd_can_network_connect on

# Allow web server to connect to database
setsebool -P httpd_can_network_connect_db on

# Allow web server to send email
setsebool -P httpd_can_sendmail on

# Allow web server to serve user home directories
setsebool -P httpd_enable_homedirs on

# Allow NFS to export read/write directories
setsebool -P nfs_export_all_rw on

# Allow Samba to share home directories
setsebool -P samba_enable_home_dirs on

# Allow Samba to export all read/write
setsebool -P samba_export_all_rw on

# Allow ftpd to use NFS
setsebool -P ftpd_use_nfs on

# Allow containers to use NFS volumes
setsebool -P virt_use_nfs on

# Allow virtual machines to use USB devices
setsebool -P virt_use_usb on
```

### Module Management

SELinux policy is composed of modules that can be enabled, disabled, or removed.

#### List Modules

```bash
# List all installed modules
semodule -l
# abrt    1.4.1
# accountsd   1.1.0
# apache  2.6.8
# mysql   1.11.1
# ssh     2.5.2

# List modules with priority
semodule -l --full
# 100 abrt     1.4.1
# 100 apache   2.6.8
# 400 mycustom 1.0.0

# List enabled modules only
semodule --list-modules=full | grep -v disabled

# List disabled modules
semodule --list-modules=full | grep disabled
```

#### Install Modules

```bash
# Install a policy module
semodule -i myapp.pp

# Install with specific priority (higher = higher precedence)
semodule -X 400 -i myapp.pp

# Install and enable module
semodule -i myapp.pp -e myapp

# Install multiple modules
semodule -i module1.pp -i module2.pp
```

#### Enable/Disable Modules

```bash
# Disable a module (doesn't remove, just deactivates)
semodule -d apache

# Enable a module
semodule -e apache

# Disable multiple modules
semodule -d module1 -d module2

# Note: Disabling is preferred over removing for system modules
```

#### Remove Modules

```bash
# Remove a custom module
semodule -r myapp

# Remove with specific priority
semodule -X 400 -r myapp

# List before removing to confirm
semodule -l | grep myapp
semodule -r myapp
```

#### Module Information

```bash
# Extract a module for inspection
semodule -e apache --extract

# This creates: apache.pp

# Convert binary module to human-readable format
semodule -l | grep apache
# apache  2.6.8

# Get module details
semodule -l --full | grep apache
```

### User and Role Management

SELinux users are different from Linux users and map to roles.

#### List SELinux Users

```bash
# List SELinux users and their properties
semanage user -l
# Labeling   MLS/       MLS/
# SELinux User    Prefix     MCS Level  MCS Range       SELinux Roles
# guest_u         user       s0         s0              guest_r
# root            user       s0         s0-s0:c0.c1023  staff_r sysadm_r system_r unconfined_r
# staff_u         user       s0         s0-s0:c0.c1023  staff_r sysadm_r
# sysadm_u        user       s0         s0-s0:c0.c1023  sysadm_r
# system_u        user       s0         s0-s0:c0.c1023  system_r unconfined_r
# unconfined_u    user       s0         s0-s0:c0.c1023  system_r unconfined_r
# user_u          user       s0         s0              user_r
```

#### List Login Mappings

```bash
# Show mapping from Linux users to SELinux users
semanage login -l
# Login Name           SELinux User         MLS/MCS Range        Service
# __default__          unconfined_u         s0-s0:c0.c1023       *
# root                 unconfined_u         s0-s0:c0.c1023       *
# system_u             system_u             s0-s0:c0.c1023       *
```

#### Map Linux Users to SELinux Users

```bash
# Map a Linux user to SELinux user
semanage login -a -s user_u john

# Map with specific MLS range
semanage login -a -s staff_u -r s0-s0:c0.c1023 alice

# Modify existing mapping
semanage login -m -s staff_u -r s0-s0:c0.c1023 john

# Delete mapping (reverts to __default__)
semanage login -d john

# The mapped user will get the SELinux user context on next login
```

#### Create Custom SELinux Users

```bash
# Create SELinux user with specific roles
semanage user -a -R "staff_r sysadm_r" myuser_u

# Create user with MLS range
semanage user -a -R "user_r" -r s0 restricted_u

# Modify user roles
semanage user -m -R "staff_r sysadm_r system_r" myuser_u

# Delete SELinux user
semanage user -d myuser_u
```

---

## Common Patterns

### Web Server Configuration

#### Apache (httpd)

**Basic Apache SELinux Contexts**:

```bash
# Apache binary and libraries
# /usr/sbin/httpd → httpd_exec_t (entrypoint to httpd_t domain)
ls -Z /usr/sbin/httpd
# system_u:object_r:httpd_exec_t:s0 /usr/sbin/httpd

# Apache process runs in httpd_t domain
ps -eZ | grep httpd
# system_u:system_r:httpd_t:s0    1234 ?    00:00:01 httpd

# Default web content directory
ls -Zd /var/www/html/
# system_u:object_r:httpd_sys_content_t:s0 /var/www/html/

# CGI scripts directory
ls -Zd /var/www/cgi-bin/
# system_u:object_r:httpd_sys_script_exec_t:s0 /var/www/cgi-bin/
```

**Standard Content Types**:

```bash
# Read-only content (HTML, CSS, JS, images)
# Type: httpd_sys_content_t
ls -Z /var/www/html/
# -rw-r--r--. root root system_u:object_r:httpd_sys_content_t:s0 index.html
# -rw-r--r--. root root system_u:object_r:httpd_sys_content_t:s0 style.css

# Read-write content (upload directories, cache)
# Type: httpd_sys_rw_content_t
mkdir /var/www/html/uploads
semanage fcontext -a -t httpd_sys_rw_content_t "/var/www/html/uploads(/.*)?"
restorecon -Rv /var/www/html/uploads/

# CGI/script execution
# Type: httpd_sys_script_exec_t
ls -Z /var/www/cgi-bin/test.cgi
# -rwxr-xr-x. root root system_u:object_r:httpd_sys_script_exec_t:s0 test.cgi

# Script writable content (for CGI scripts)
# Type: httpd_sys_script_rw_t
mkdir /var/www/cgi-data
semanage fcontext -a -t httpd_sys_script_rw_t "/var/www/cgi-data(/.*)?"
restorecon -Rv /var/www/cgi-data/
```

**Custom DocumentRoot**:

```bash
# Set up custom web directory
mkdir -p /web/mysite
echo "Hello World" > /web/mysite/index.html

# Add file context rule
semanage fcontext -a -t httpd_sys_content_t "/web/mysite(/.*)?"

# Apply context
restorecon -Rv /web/mysite/

# Verify
ls -Zd /web/mysite/
# system_u:object_r:httpd_sys_content_t:s0 /web/mysite/

# Configure Apache
cat >> /etc/httpd/conf.d/mysite.conf <<EOF
<VirtualHost *:80>
    ServerName mysite.local
    DocumentRoot /web/mysite
    <Directory /web/mysite>
        Require all granted
    </Directory>
</VirtualHost>
EOF

systemctl restart httpd
```

**Common Apache Booleans**:

```bash
# Allow httpd to connect to network (for proxy, external APIs)
setsebool -P httpd_can_network_connect on

# Allow httpd to connect to databases
setsebool -P httpd_can_network_connect_db on

# Allow httpd to send email
setsebool -P httpd_can_sendmail on

# Allow httpd to serve content from user home directories
setsebool -P httpd_enable_homedirs on

# Allow httpd scripts and modules to connect to network
setsebool -P httpd_can_network_relay on

# Allow httpd to connect to LDAP
setsebool -P httpd_can_connect_ldap on

# Allow httpd to run as unified process
setsebool -P httpd_unified on

# Allow HTTPD scripts and modules to execute with user permissions
setsebool -P httpd_enable_cgi on

# Allow httpd to execute memory-mapped files
setsebool -P httpd_execmem on
```

**PHP Configuration**:

```bash
# PHP files should be httpd_sys_content_t or httpd_sys_script_exec_t
ls -Z /var/www/html/index.php
# system_u:object_r:httpd_sys_content_t:s0 /var/www/html/index.php

# PHP session directory
ls -Zd /var/lib/php/session/
# drwxrwx---. root apache system_u:object_r:httpd_var_lib_t:s0 /var/lib/php/session/

# If PHP needs to write to a directory
mkdir /var/www/html/data
semanage fcontext -a -t httpd_sys_rw_content_t "/var/www/html/data(/.*)?"
restorecon -Rv /var/www/html/data/
chmod 770 /var/www/html/data
chown apache:apache /var/www/html/data
```

#### Nginx

```bash
# Nginx binary
ls -Z /usr/sbin/nginx
# system_u:object_r:httpd_exec_t:s0 /usr/sbin/nginx

# Nginx runs in httpd_t domain (same as Apache)
ps -eZ | grep nginx
# system_u:system_r:httpd_t:s0    5678 ?    00:00:00 nginx

# Default content directory
ls -Zd /usr/share/nginx/html/
# drwxr-xr-x. root root system_u:object_r:httpd_sys_content_t:s0 /usr/share/nginx/html/

# Custom site configuration
mkdir -p /srv/www/example.com
echo "Test" > /srv/www/example.com/index.html

# Label the directory
semanage fcontext -a -t httpd_sys_content_t "/srv/www(/.*)?"
restorecon -Rv /srv/www/

# Nginx configuration
cat > /etc/nginx/conf.d/example.conf <<EOF
server {
    listen 80;
    server_name example.com;
    root /srv/www/example.com;

    location / {
        index index.html;
    }
}
EOF

# Test and reload
nginx -t
systemctl reload nginx

# Same booleans as Apache apply to Nginx
setsebool -P httpd_can_network_connect on
```

### Database Server Contexts

#### PostgreSQL

```bash
# PostgreSQL binary
ls -Z /usr/bin/postgres
# system_u:object_r:postgresql_exec_t:s0 /usr/bin/postgres

# PostgreSQL process domain
ps -eZ | grep postgres
# system_u:system_r:postgresql_t:s0  3456 ?  00:00:01 postgres

# Data directory
ls -Zd /var/lib/pgsql/
# drwx------. postgres postgres system_u:object_r:postgresql_db_t:s0 /var/lib/pgsql/

# Log directory
ls -Zd /var/log/postgresql/
# drwx------. postgres postgres system_u:object_r:postgresql_log_t:s0 /var/log/postgresql/

# Port labeling
semanage port -l | grep postgresql
# postgresql_port_t              tcp      5432, 9898

# Custom data directory
mkdir -p /data/postgresql
chown postgres:postgres /data/postgresql
chmod 700 /data/postgresql

# Add file context
semanage fcontext -a -t postgresql_db_t "/data/postgresql(/.*)?"
restorecon -Rv /data/postgresql/

# Initialize database
sudo -u postgres /usr/bin/initdb -D /data/postgresql

# Custom port
semanage port -a -t postgresql_port_t -p tcp 5433

# Allow PostgreSQL to connect to network (for replication)
setsebool -P postgresql_can_network_connect on
```

#### MySQL/MariaDB

```bash
# MySQL binary
ls -Z /usr/bin/mysqld
# system_u:object_r:mysqld_exec_t:s0 /usr/bin/mysqld

# MySQL process domain
ps -eZ | grep mysqld
# system_u:system_r:mysqld_t:s0     2345 ?  00:00:02 mysqld

# Data directory
ls -Zd /var/lib/mysql/
# drwxr-x---. mysql mysql system_u:object_r:mysqld_db_t:s0 /var/lib/mysql/

# Log file
ls -Z /var/log/mysqld.log
# -rw-r-----. mysql mysql system_u:object_r:mysqld_log_t:s0 /var/log/mysqld.log

# Port labeling
semanage port -l | grep mysql
# mysqld_port_t                  tcp      1186, 3306, 63132-63164

# Custom data directory
mkdir -p /data/mysql
chown mysql:mysql /data/mysql
chmod 750 /data/mysql

# Add file context
semanage fcontext -a -t mysqld_db_t "/data/mysql(/.*)?"
restorecon -Rv /data/mysql/

# Custom port
semanage port -a -t mysqld_port_t -p tcp 3307

# Common booleans
# Allow httpd to connect to MySQL
setsebool -P httpd_can_network_connect_db on

# Allow MySQL to connect to network (for replication)
setsebool -P mysql_connect_any on
```

### Container Integration

#### Docker/Podman with SELinux

Docker and Podman use SELinux to provide process and file isolation between containers.

**Container Process Labels**:

```bash
# Container processes run in svirt_lxc_net_t domain (or container_t)
docker run -d --name web nginx
ps -eZ | grep nginx
# system_u:system_r:svirt_lxc_net_t:s0:c123,c456  7890 ? 00:00:00 nginx

# Each container gets unique MCS labels (c123,c456)
# This prevents containers from accessing each other's files
```

**Volume Mounting**:

```bash
# Create a directory for container data
mkdir /data/web-content
echo "Hello from host" > /data/web-content/index.html

# Without :Z or :z, SELinux may block access
docker run -d --name web1 -v /data/web-content:/usr/share/nginx/html:ro nginx
# May fail with permission denied in container

# Check container logs
docker logs web1
# Permission denied errors

# Option 1: Use :z for shared volume (multiple containers)
docker run -d --name web1 -v /data/web-content:/usr/share/nginx/html:z nginx

# :z relabels with svirt_sandbox_file_t (shared among all containers)
ls -Zd /data/web-content/
# system_u:object_r:svirt_sandbox_file_t:s0 /data/web-content/

# Option 2: Use :Z for private volume (single container)
docker run -d --name web2 -v /data/web-private:/data:Z nginx

# :Z relabels with unique MCS label for this container only
ls -Zd /data/web-private/
# system_u:object_r:svirt_sandbox_file_t:s0:c789,c012 /data/web-private/

# Option 3: Manual labeling
mkdir /data/web-manual
semanage fcontext -a -t svirt_sandbox_file_t "/data/web-manual(/.*)?"
restorecon -Rv /data/web-manual/
docker run -d -v /data/web-manual:/data nginx
```

**SELinux Modes for Containers**:

```bash
# Disable SELinux for a specific container (not recommended)
docker run --security-opt label=disable nginx

# Run container in permissive mode (for debugging)
docker run --security-opt label=type:svirt_lxc_net_t --security-opt label=level:s0 nginx

# Check container's SELinux context
docker inspect --format='{{.ProcessLabel}}' web1
# system_u:system_r:svirt_lxc_net_t:s0:c123,c456
docker inspect --format='{{.MountLabel}}' web1
# system_u:object_r:svirt_sandbox_file_t:s0:c123,c456
```

**Podman SELinux Integration**:

```bash
# Podman has similar SELinux integration
podman run -d --name web -v /data/www:/usr/share/nginx/html:z nginx

# Rootless containers get user-specific labels
podman run --rm -it alpine id -Z
# system_u:system_r:container_t:s0:c123,c456

# Check labels
podman inspect --format='{{.ProcessLabel}}' web
```

**Common Container Booleans**:

```bash
# Allow containers to use NFS volumes
setsebool -P virt_use_nfs on

# Allow containers to use CIFS/Samba volumes
setsebool -P virt_use_samba on

# Allow containers to use FUSE filesystems
setsebool -P virt_use_fusefs on

# Allow containers to connect to sandbox network
setsebool -P virt_sandbox_use_all_caps on
```

### Network File Sharing

#### NFS Server

```bash
# NFS exports file
ls -Z /etc/exports
# system_u:object_r:exports_t:s0 /etc/exports

# NFS daemon
ps -eZ | grep nfsd
# system_u:system_r:kernel_t:s0  0 ?   00:00:00 nfsd

# Exported directory
mkdir /srv/nfs/share
chmod 755 /srv/nfs/share

# Label for read-only NFS export
semanage fcontext -a -t public_content_t "/srv/nfs/share(/.*)?"
restorecon -Rv /srv/nfs/share/

# Label for read-write NFS export
semanage fcontext -a -t public_content_rw_t "/srv/nfs/writable(/.*)?"
restorecon -Rv /srv/nfs/writable/

# Configure export
echo "/srv/nfs/share 192.168.1.0/24(ro,sync)" >> /etc/exports
echo "/srv/nfs/writable 192.168.1.0/24(rw,sync)" >> /etc/exports

# Enable booleans
setsebool -P nfs_export_all_ro on
setsebool -P nfs_export_all_rw on

# Export shares
exportfs -ra

# Start NFS
systemctl enable --now nfs-server
```

#### NFS Client

```bash
# Mount point
mkdir /mnt/nfs

# Mount NFS share
mount -t nfs 192.168.1.100:/srv/nfs/share /mnt/nfs

# Check context (inherited from server or use_nfs_home_dirs)
ls -Zd /mnt/nfs/
# system_u:object_r:nfs_t:s0 /mnt/nfs/

# Allow services to use NFS
# Allow httpd to use NFS mounted content
setsebool -P httpd_use_nfs on

# Allow Samba to export NFS
setsebool -P samba_share_nfs on

# Permanent mount
echo "192.168.1.100:/srv/nfs/share /mnt/nfs nfs defaults 0 0" >> /etc/fstab
```

#### Samba

```bash
# Samba daemon
ps -eZ | grep smbd
# system_u:system_r:smbd_t:s0  4567 ?  00:00:01 smbd

# Samba configuration
ls -Z /etc/samba/smb.conf
# system_u:object_r:samba_etc_t:s0 /etc/samba/smb.conf

# Samba share directory
mkdir /srv/samba/public
chmod 755 /srv/samba/public

# Label for Samba export
semanage fcontext -a -t samba_share_t "/srv/samba/public(/.*)?"
restorecon -Rv /srv/samba/public/

# For writable share
semanage fcontext -a -t samba_share_t "/srv/samba/writable(/.*)?"
chmod 775 /srv/samba/writable
restorecon -Rv /srv/samba/writable/

# Configure Samba
cat >> /etc/samba/smb.conf <<EOF
[public]
    path = /srv/samba/public
    read only = yes
    guest ok = yes

[writable]
    path = /srv/samba/writable
    read only = no
    valid users = @users
EOF

# Common Samba booleans
setsebool -P samba_enable_home_dirs on    # Share home directories
setsebool -P samba_export_all_ro on       # Share all files read-only
setsebool -P samba_export_all_rw on       # Share all files read-write
setsebool -P samba_share_nfs on           # Share NFS mounts

# Restart Samba
systemctl restart smb nmb
```

### SSH Configuration

#### Custom SSH Port

```bash
# Default SSH port
semanage port -l | grep ssh
# ssh_port_t                     tcp      22

# Configure SSH to use port 2222
vi /etc/ssh/sshd_config
# Port 2222

# Add SELinux label for port 2222
semanage port -a -t ssh_port_t -p tcp 2222

# Verify
semanage port -l | grep ssh
# ssh_port_t                     tcp      22, 2222

# Restart SSH
systemctl restart sshd

# Now SSH can bind to port 2222
```

#### SSH Key Files

```bash
# SSH daemon keys
ls -Z /etc/ssh/ssh_host_*_key
# system_u:object_r:sshd_key_t:s0 /etc/ssh/ssh_host_rsa_key

# User SSH directory
ls -Zd ~/.ssh/
# unconfined_u:object_r:ssh_home_t:s0 /home/user/.ssh/

# Authorized keys
ls -Z ~/.ssh/authorized_keys
# unconfined_u:object_r:ssh_home_t:s0 /home/user/.ssh/authorized_keys

# Private keys
ls -Z ~/.ssh/id_rsa
# unconfined_u:object_r:ssh_home_t:s0 /home/user/.ssh/id_rsa

# If context is wrong, restore it
restorecon -Rv ~/.ssh/
```

#### SFTP Chroot

```bash
# Create SFTP chroot directory
mkdir -p /var/sftp/uploads

# Directory must be owned by root for chroot
chown root:root /var/sftp
chmod 755 /var/sftp

# User writable directory
chown sftpuser:sftpuser /var/sftp/uploads
chmod 755 /var/sftp/uploads

# Set SELinux context
semanage fcontext -a -t ssh_home_t "/var/sftp(/.*)?"
restorecon -Rv /var/sftp/

# Configure sshd
cat >> /etc/ssh/sshd_config <<EOF
Match User sftpuser
    ChrootDirectory /var/sftp
    ForceCommand internal-sftp
    AllowTcpForwarding no
    X11Forwarding no
EOF

systemctl restart sshd
```

### systemd Service Contexts

#### Creating a Custom systemd Service

```bash
# Create application
mkdir -p /opt/myapp
cat > /opt/myapp/myapp.sh <<'EOF'
#!/bin/bash
while true; do
    echo "MyApp is running..."
    sleep 60
done
EOF
chmod +x /opt/myapp/myapp.sh

# Label the application
semanage fcontext -a -t bin_t "/opt/myapp/myapp.sh"
restorecon -v /opt/myapp/myapp.sh

# Create systemd service
cat > /etc/systemd/system/myapp.service <<EOF
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
ExecStart=/opt/myapp/myapp.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Service file context
restorecon -v /etc/systemd/system/myapp.service

# Enable and start
systemctl daemon-reload
systemctl enable --now myapp

# Check process context
ps -eZ | grep myapp
# system_u:system_r:initrc_t:s0  8901 ?  00:00:00 myapp.sh
# Note: Runs in initrc_t (default for systemd services)
```

#### Custom Domain for Service

For better isolation, create a custom SELinux domain:

```bash
# This requires policy development (see Policy Development section)
# Quick example using existing domain

# If app needs network access, use a suitable domain
# For example, to run in unconfined_service_t:

cat > /etc/systemd/system/myapp.service <<EOF
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
ExecStart=/opt/myapp/myapp.sh
Restart=on-failure
SELinuxContext=system_u:system_r:unconfined_service_t:s0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl restart myapp

ps -eZ | grep myapp
# system_u:system_r:unconfined_service_t:s0  9012 ?  00:00:00 myapp.sh
```

### Custom Application Labeling

#### Simple Application Setup

```bash
# Application in /opt
mkdir -p /opt/customapp/bin
mkdir -p /opt/customapp/data
mkdir -p /opt/customapp/logs

# Copy or create application files
cp myapp /opt/customapp/bin/

# Label executable
semanage fcontext -a -t bin_t "/opt/customapp/bin(/.*)?"
restorecon -Rv /opt/customapp/bin/

# Label data directory
semanage fcontext -a -t var_lib_t "/opt/customapp/data(/.*)?"
restorecon -Rv /opt/customapp/data/

# Label logs directory
semanage fcontext -a -t var_log_t "/opt/customapp/logs(/.*)?"
restorecon -Rv /opt/customapp/logs/

# Verify contexts
ls -ZR /opt/customapp/
```

#### Application with Custom Domain

For production applications, create a custom policy module (see Policy Development section).

### Home Directory Management

#### User Home Directories

```bash
# Default home directory context
ls -Zd /home/user/
# unconfined_u:object_r:user_home_dir_t:s0 /home/user/

# Files in home directory
ls -Z /home/user/
# unconfined_u:object_r:user_home_t:s0 file.txt
# unconfined_u:object_r:user_home_t:s0 documents/

# Allow httpd to read user home directories
setsebool -P httpd_enable_homedirs on

# Label specific directory for web access
mkdir /home/user/public_html
semanage fcontext -a -t httpd_user_content_t "/home/user/public_html(/.*)?"
restorecon -Rv /home/user/public_html/
```

#### Custom Home Directory Location

```bash
# Create custom home location
mkdir /data/home
mkdir /data/home/newuser

# Add file context equivalence
semanage fcontext -a -e /home /data/home
restorecon -Rv /data/home/

# Or manually add contexts
semanage fcontext -a -t user_home_dir_t "/data/home/[^/]+"
semanage fcontext -a -t user_home_t "/data/home/[^/]+/(.*)?'
restorecon -Rv /data/home/

# Create user with custom home
useradd -d /data/home/newuser -m newuser

# Verify
ls -Zd /data/home/newuser/
# unconfined_u:object_r:user_home_dir_t:s0 /data/home/newuser/
```

---

## Policy Development

### Understanding Audit Logs

SELinux denials are logged to the audit log, which is essential for troubleshooting and policy development.

#### Audit Log Location

```bash
# Primary audit log
tail -f /var/log/audit/audit.log

# SELinux-specific messages (AVC = Access Vector Cache)
ausearch -m avc -ts recent

# Last 10 minutes
ausearch -m avc -ts recent -i
# -i flag converts numeric IDs to names for readability

# Specific time range
ausearch -m avc -ts today
ausearch -m avc -ts 14:00:00 -te 14:30:00

# For specific process
ausearch -m avc -c httpd

# For specific path
ausearch -m avc -f /var/www/html/
```

#### Reading AVC Denial Messages

```bash
# Example AVC denial
ausearch -m avc --start recent -i
# ----
# type=AVC msg=audit(1700000000.123:456): avc:  denied  { read } for  pid=1234 comm="httpd" name="index.html" dev="dm-0" ino=5678 scontext=system_u:system_r:httpd_t:s0 tcontext=unconfined_u:object_r:user_home_t:s0 tclass=file permissive=0

# Breaking down the message:
# - denied { read }         : Action that was denied
# - pid=1234                : Process ID
# - comm="httpd"            : Command/process name
# - name="index.html"       : File being accessed
# - scontext=httpd_t        : Source context (process domain)
# - tcontext=user_home_t    : Target context (file type)
# - tclass=file             : Object class (file, dir, tcp_socket, etc.)
# - permissive=0            : 0=enforcing, 1=permissive
```

**Common AVC Fields**:
- `scontext`: Source (subject) - usually a process domain
- `tcontext`: Target (object) - file, port, socket, etc.
- `tclass`: Target class - file, dir, tcp_socket, process, capability, etc.
- `denied { ... }`: Permissions that were denied
- `comm`: Command name
- `pid`: Process ID
- `permissive`: Whether SELinux was in permissive mode

### Creating Custom Modules

#### Using audit2allow

The `audit2allow` tool generates policy modules from audit denials:

```bash
# Install required tools
yum install -y policycoreutils-python-utils

# Generate policy from recent denials
ausearch -m avc -ts recent | audit2allow

# Output shows what rules would allow the denied actions:
###### BEGIN ####
#
#module myapp 1.0;
#
#require {
#       type httpd_t;
#       type user_home_t;
#       class file { read open getattr };
#}
#
##============= httpd_t ==============
#allow httpd_t user_home_t:file { read open getattr };
#
####### END #####

# Generate and compile a policy module
ausearch -m avc -ts recent | audit2allow -M myapp
# ******************** IMPORTANT ***********************
# To make this policy package active, execute:
# semodule -i myapp.pp

# This creates two files:
# myapp.te - Type Enforcement file (source)
# myapp.pp - Policy Package (compiled)

# Install the module
semodule -i myapp.pp

# Verify installation
semodule -l | grep myapp
```

#### audit2why - Understanding Denials

```bash
# Explain why access was denied
ausearch -m avc -ts recent | audit2why

# Example output:
# type=AVC msg=audit(1700000000.123:456): avc:  denied  { read } for  pid=1234 comm="httpd" ...
#
#       Was caused by:
#               Missing type enforcement (TE) allow rule.
#
#               You can use audit2allow to generate a loadable module to allow this access.

# Or it might suggest a boolean:
# type=AVC msg=audit(...): avc:  denied  { name_connect } for  pid=1234 comm="httpd" ...
#
#       Was caused by:
#       The boolean httpd_can_network_connect was set incorrectly.
#       Description:
#       Allow httpd to can network connect
#
#       Allow access by executing:
#       # setsebool -P httpd_can_network_connect 1
```

### Policy Module Structure

A complete policy module consists of three files:

#### Type Enforcement (.te) File

Contains the actual policy rules:

```bash
# myapp.te
policy_module(myapp, 1.0.0)

########################################
#
# Declarations
#

# Define a new type for the application
type myapp_t;
type myapp_exec_t;
init_daemon_domain(myapp_t, myapp_exec_t)

# Define types for application files
type myapp_data_t;
files_type(myapp_data_t)

type myapp_log_t;
logging_log_file(myapp_log_t)

########################################
#
# myapp local policy
#

# Allow myapp to execute its binary
allow myapp_t myapp_exec_t:file { execute execute_no_trans };

# Allow reading data files
allow myapp_t myapp_data_t:dir list_dir_perms;
allow myapp_t myapp_data_t:file read_file_perms;

# Allow writing to log files
allow myapp_t myapp_log_t:dir { add_name write };
allow myapp_t myapp_log_t:file { create write append setattr };

# Allow network access
corenet_tcp_bind_generic_node(myapp_t)
corenet_tcp_bind_http_port(myapp_t)
corenet_tcp_connect_http_port(myapp_t)

# Allow reading /etc files
files_read_etc_files(myapp_t)

# Standard permissions
logging_send_syslog_msg(myapp_t)
miscfiles_read_localization(myapp_t)
```

#### Interface (.if) File

Defines interfaces for other modules to interact with your policy:

```bash
# myapp.if

## <summary>My Application policy</summary>

########################################
## <summary>
##      Execute myapp in the myapp domain.
## </summary>
## <param name="domain">
##      <summary>
##      Domain allowed to transition.
##      </summary>
## </param>
#
interface(`myapp_domtrans',`
        gen_require(`
                type myapp_t, myapp_exec_t;
        ')

        corecmd_search_bin($1)
        domtrans_pattern($1, myapp_exec_t, myapp_t)
')

########################################
## <summary>
##      Read myapp data files.
## </summary>
## <param name="domain">
##      <summary>
##      Domain allowed access.
##      </summary>
## </param>
#
interface(`myapp_read_data',`
        gen_require(`
                type myapp_data_t;
        ')

        files_search_var_lib($1)
        read_files_pattern($1, myapp_data_t, myapp_data_t)
')
```

#### File Context (.fc) File

Maps file paths to SELinux types:

```bash
# myapp.fc

# Executable
/opt/myapp/bin(/.*)?         gen_context(system_u:object_r:myapp_exec_t,s0)
/usr/sbin/myapp              gen_context(system_u:object_r:myapp_exec_t,s0)

# Data files
/opt/myapp/data(/.*)?        gen_context(system_u:object_r:myapp_data_t,s0)
/var/lib/myapp(/.*)?         gen_context(system_u:object_r:myapp_data_t,s0)

# Log files
/opt/myapp/logs(/.*)?        gen_context(system_u:object_r:myapp_log_t,s0)
/var/log/myapp(/.*)?         gen_context(system_u:object_r:myapp_log_t,s0)

# Configuration
/etc/myapp(/.*)?             gen_context(system_u:object_r:etc_t,s0)

# PID file
/var/run/myapp\.pid          gen_context(system_u:object_r:var_run_t,s0)
```

### Type Enforcement Rules

#### Allow Rules

```bash
# Basic allow rule syntax:
# allow <source_domain> <target_type>:<object_class> { <permissions> };

# Allow httpd to read files of type httpd_sys_content_t
allow httpd_t httpd_sys_content_t:file { read open getattr };

# Allow httpd to write to files of type httpd_sys_rw_content_t
allow httpd_t httpd_sys_rw_content_t:file { write create unlink };

# Allow httpd to list directories
allow httpd_t httpd_sys_content_t:dir { read getattr search open };

# Allow httpd to bind to HTTP port
allow httpd_t http_port_t:tcp_socket { bind };

# Allow httpd to connect to database port
allow httpd_t mysqld_port_t:tcp_socket { name_connect };

# Allow process to fork
allow myapp_t self:process { fork };

# Allow reading /proc filesystem
allow myapp_t proc_t:file { read open getattr };
```

#### Permission Macros

SELinux provides macros for common permission sets:

```bash
# Read file permissions
# Expands to: { read open getattr }
read_file_perms

# Write file permissions
# Expands to: { write append }
write_file_perms

# Create file permissions
# Expands to: { create write open getattr setattr }
create_file_perms

# List directory permissions
# Expands to: { read getattr search open }
list_dir_perms

# Add directory entry permissions
# Expands to: { add_name write }
add_entry_dir_perms

# Example usage:
allow httpd_t httpd_sys_content_t:file read_file_perms;
allow httpd_t httpd_sys_rw_content_t:file create_file_perms;
allow httpd_t httpd_sys_content_t:dir list_dir_perms;
```

#### Attribute-Based Rules

```bash
# Types can have attributes for grouping
# Example: All domain types have the "domain" attribute

# Allow all domains to read /etc/passwd
allow domain passwd_file_t:file read_file_perms;

# Common attributes:
# - domain: All process domains
# - file_type: All file types
# - port_type: All port types
# - domain_type: Synonym for domain
```

### Domain Transitions

Domain transitions allow a process to change from one security domain to another when executing a file.

#### Automatic Domain Transition

```bash
# Three rules required for automatic domain transition:
# 1. Source domain can execute the file
# 2. File is an entrypoint to target domain
# 3. Source domain can transition to target domain

# Example: init_t → httpd_t transition when executing /usr/sbin/httpd

# 1. Allow init_t to execute httpd_exec_t
allow init_t httpd_exec_t:file { execute read getattr open };

# 2. Allow httpd_exec_t as entrypoint to httpd_t domain
allow httpd_t httpd_exec_t:file entrypoint;

# 3. Allow init_t to transition to httpd_t
allow init_t httpd_t:process transition;

# Macro that does all three:
domtrans_pattern(init_t, httpd_exec_t, httpd_t)
```

#### Type Transition Rules

```bash
# Syntax: type_transition <source> <target>:<class> <default_type>;

# When httpd_t creates a file in tmp_t directory, label it httpd_tmp_t
type_transition httpd_t tmp_t:file httpd_tmp_t;

# When httpd_t creates a directory in tmp_t, label it httpd_tmp_t
type_transition httpd_t tmp_t:dir httpd_tmp_t;

# When init_t executes httpd_exec_t, transition to httpd_t
type_transition init_t httpd_exec_t:process httpd_t;

# Example in policy module:
type_transition myapp_t var_log_t:file myapp_log_t;
# Now when myapp_t creates a file in /var/log/, it's automatically labeled myapp_log_t
```

#### Named Type Transition

```bash
# Type transition based on filename
# type_transition <source> <target>:<class> <default_type> "<filename>";

# Example: systemd creating /run/myapp.pid
type_transition init_t var_run_t:file myapp_var_run_t "myapp.pid";

# This only applies when the filename matches exactly
```

### File Context Specifications

File context specifications in the .fc file use regular expressions:

```bash
# Exact match
/usr/sbin/httpd              gen_context(system_u:object_r:httpd_exec_t,s0)

# Match directory and all contents recursively
/var/www(/.*)?               gen_context(system_u:object_r:httpd_sys_content_t,s0)

# Match only the directory itself
/var/www                     gen_context(system_u:object_r:httpd_sys_content_t,s0)

# Match specific file types
/var/log/myapp/.*\.log       gen_context(system_u:object_r:myapp_log_t,s0)

# Match with character class
/etc/myapp/[^/]+\.conf       gen_context(system_u:object_r:myapp_etc_t,s0)

# Multiple paths
/usr/bin/myapp      --       gen_context(system_u:object_r:myapp_exec_t,s0)
/usr/sbin/myapp     --       gen_context(system_u:object_r:myapp_exec_t,s0)

# Specify file type (-- = regular file, -d = directory, -l = symlink, etc.)
/var/run/myapp      -d       gen_context(system_u:object_r:myapp_var_run_t,s0)
/var/run/myapp\.pid --       gen_context(system_u:object_r:myapp_var_run_t,s0)

# <<none>> means no context (removes labeling)
/proc/.*                     <<none>>
```

### Policy Compilation and Loading

#### Compile from Source

```bash
# Method 1: Using audit2allow (simple)
audit2allow -M myapp < denials.txt
semodule -i myapp.pp

# Method 2: Manual compilation (full control)
# You have: myapp.te, myapp.if, myapp.fc

# Compile type enforcement file
checkmodule -M -m -o myapp.mod myapp.te

# Create policy package
semodule_package -o myapp.pp -m myapp.mod

# If you have file contexts
semodule_package -o myapp.pp -m myapp.mod -fc myapp.fc

# If you have interface file (requires more complex build)
# Use reference policy build system or:
checkmodule -M -m -o myapp.mod myapp.te
semodule_package -o myapp.pp -m myapp.mod -fc myapp.fc

# Install the module
semodule -i myapp.pp

# Apply file contexts
restorecon -Rv /opt/myapp/
```

#### Managing Policy Modules

```bash
# Install module
semodule -i myapp.pp

# Install with priority (higher priority = higher precedence)
semodule -X 400 -i myapp.pp

# Update existing module
semodule -u myapp.pp

# Remove module
semodule -r myapp

# Enable module
semodule -e myapp

# Disable module (keeps it installed but inactive)
semodule -d myapp

# List all modules
semodule -l

# List module with priority
semodule --list-modules=full | grep myapp

# Extract installed module
semodule -e myapp --extract
# Creates: myapp.pp

# Rebuild policy after manual changes
semodule -B
```

#### Building with Reference Policy

For complex modules, use the SELinux Reference Policy build system:

```bash
# Clone reference policy
git clone https://github.com/SELinuxProject/refpolicy.git
cd refpolicy

# Create module directory
mkdir policy/modules/services/myapp

# Add your files
cp ~/myapp.te policy/modules/services/myapp/
cp ~/myapp.if policy/modules/services/myapp/
cp ~/myapp.fc policy/modules/services/myapp/

# Edit modules.conf to enable your module
echo "myapp = module" >> policy/modules.conf

# Build
make bare
make conf
make

# Install
make install

# Or build just your module
make myapp.pp
semodule -i myapp.pp
```

### Interface Development

Interfaces allow other modules to interact with your policy safely:

```bash
# myapp.if

## <summary>Policy for My Application</summary>

########################################
## <summary>
##      Execute myapp in the myapp domain.
## </summary>
## <param name="domain">
##      <summary>
##      Domain allowed to transition.
##      </summary>
## </param>
#
interface(`myapp_domtrans',`
        gen_require(`
                type myapp_t, myapp_exec_t;
        ')

        corecmd_search_bin($1)
        domtrans_pattern($1, myapp_exec_t, myapp_t)
')

########################################
## <summary>
##      Execute myapp in the myapp domain, and
##      allow the specified role the myapp domain.
## </summary>
## <param name="domain">
##      <summary>
##      Domain allowed to transition.
##      </summary>
## </param>
## <param name="role">
##      <summary>
##      Role allowed access.
##      </summary>
## </param>
#
interface(`myapp_run',`
        gen_require(`
                type myapp_t;
        ')

        myapp_domtrans($1)
        role $2 types myapp_t;
')

########################################
## <summary>
##      Read myapp configuration files.
## </summary>
## <param name="domain">
##      <summary>
##      Domain allowed access.
##      </summary>
## </param>
#
interface(`myapp_read_config',`
        gen_require(`
                type myapp_etc_t;
        ')

        files_search_etc($1)
        allow $1 myapp_etc_t:dir list_dir_perms;
        allow $1 myapp_etc_t:file read_file_perms;
')

########################################
## <summary>
##      Manage myapp data files.
## </summary>
## <param name="domain">
##      <summary>
##      Domain allowed access.
##      </summary>
## </param>
#
interface(`myapp_manage_data',`
        gen_require(`
                type myapp_data_t;
        ')

        files_search_var_lib($1)
        allow $1 myapp_data_t:dir manage_dir_perms;
        allow $1 myapp_data_t:file manage_file_perms;
')

########################################
## <summary>
##      Connect to myapp over a TCP socket.
## </summary>
## <param name="domain">
##      <summary>
##      Domain allowed access.
##      </summary>
## </param>
#
interface(`myapp_tcp_connect',`
        gen_require(`
                type myapp_t, myapp_port_t;
        ')

        corenet_tcp_recvfrom_labeled($1, myapp_t)
        corenet_tcp_sendrecv_myapp_port($1)
        corenet_tcp_connect_myapp_port($1)
')

########################################
## <summary>
##      All of the rules required to administrate
##      a myapp environment.
## </summary>
## <param name="domain">
##      <summary>
##      Domain allowed access.
##      </summary>
## </param>
## <param name="role">
##      <summary>
##      Role allowed access.
##      </summary>
## </param>
#
interface(`myapp_admin',`
        gen_require(`
                type myapp_t, myapp_data_t;
                type myapp_log_t, myapp_etc_t;
        ')

        allow $1 myapp_t:process { ptrace signal_perms };
        ps_process_pattern($1, myapp_t)

        myapp_manage_data($1)
        myapp_manage_log($1)
        myapp_manage_config($1)

        myapp_run($1, $2)
')
```

#### Using Custom Interfaces

```bash
# In another policy module (e.g., custom_httpd.te):
policy_module(custom_httpd, 1.0.0)

# Use myapp interface to allow httpd to connect to myapp
myapp_tcp_connect(httpd_t)

# This expands to all the rules defined in the interface
```

---

## Troubleshooting

### Understanding AVC Denials

#### Denial Message Anatomy

```bash
# Sample AVC denial
type=AVC msg=audit(1700000000.123:456): avc:  denied  { write } for  pid=1234 comm="httpd" name="upload.txt" dev="dm-0" ino=5678 scontext=system_u:system_r:httpd_t:s0 tcontext=unconfined_u:object_r:user_home_t:s0 tclass=file permissive=0

# Breakdown:
# - type=AVC: Access Vector Cache message
# - msg=audit(...): Timestamp and serial number
# - denied { write }: Permission(s) denied
# - pid=1234: Process ID
# - comm="httpd": Command name
# - name="upload.txt": Object name (file, port, etc.)
# - dev="dm-0": Device
# - ino=5678: Inode number
# - scontext=system_u:system_r:httpd_t:s0: Source context (process)
# - tcontext=unconfined_u:object_r:user_home_t:s0: Target context (object)
# - tclass=file: Object class
# - permissive=0: 0=enforcing, 1=permissive

# What this means:
# The httpd process (running in httpd_t domain) tried to write to upload.txt
# (labeled user_home_t), and was denied. This is expected because httpd
# shouldn't write to user home directories.
```

#### Object Classes and Permissions

Common object classes and their permissions:

```bash
# File
# Permissions: read, write, execute, append, create, unlink, rename, setattr, getattr, etc.
tclass=file

# Directory
# Permissions: read, write, add_name, remove_name, search, rmdir, etc.
tclass=dir

# TCP Socket
# Permissions: bind, connect, listen, accept, send_msg, recv_msg, name_bind, name_connect, etc.
tclass=tcp_socket

# Process
# Permissions: fork, signal, ptrace, setpgid, transition, etc.
tclass=process

# Capability
# Permissions: dac_override, dac_read_search, net_admin, sys_admin, etc.
tclass=capability

# Unix stream socket
tclass=unix_stream_socket

# Netlink socket
tclass=netlink_route_socket
```

### Troubleshooting Tools

#### sealert (setroubleshoot)

The user-friendly SELinux troubleshooting tool:

```bash
# Install setroubleshoot
yum install -y setroubleshoot setroubleshoot-server

# Restart auditd to enable sealert
systemctl restart auditd

# Run sealert on audit log
sealert -a /var/log/audit/audit.log

# Sample output:
# SELinux is preventing /usr/sbin/httpd from write access on the file upload.txt.
#
# *****  Plugin catchall_boolean (47.5 confidence) suggests   ******************
#
# If you want to allow httpd to unified
# Then you must tell SELinux about this by enabling the 'httpd_unified' boolean.
#
# Do
# setsebool -P httpd_unified 1
#
# *****  Plugin catchall_labels (36.2 confidence) suggests   *******************
#
# If you want to allow httpd to have write access on the upload.txt file
# Then you need to change the label on upload.txt
# Do
# # semanage fcontext -a -t FILE_TYPE 'upload.txt'
# where FILE_TYPE is one of the following: httpd_sys_rw_content_t, ...
# Then execute:
# restorecon -v 'upload.txt'

# Monitor in real-time
sealert -b

# Get alert by ID
sealert -l "*"

# sealert provides:
# - Human-readable explanations
# - Suggested fixes (booleans, relabeling, policy modules)
# - Confidence ratings for each suggestion
```

#### audit2why

```bash
# Analyze recent denials
ausearch -m avc -ts recent | audit2why

# Output explains the cause:
# type=AVC msg=audit(...): avc:  denied  { name_connect } ...
#
#       Was caused by:
#       The boolean httpd_can_network_connect was set incorrectly.
#       Description:
#       Allow httpd to can network connect
#
#       Allow access by executing:
#       # setsebool -P httpd_can_network_connect 1

# Or if no boolean exists:
# type=AVC msg=audit(...): avc:  denied  { read } ...
#
#       Was caused by:
#               Missing type enforcement (TE) allow rule.
#
#               You can use audit2allow to generate a loadable module to allow this access.
```

#### sesearch - Query Policy

```bash
# Search for allow rules
sesearch --allow -s httpd_t -t httpd_sys_content_t -c file -p read

# Output:
# allow httpd_t httpd_sys_content_t:file { read open getattr };

# Find all rules for a source domain
sesearch --allow -s httpd_t

# Find rules for a target type
sesearch --allow -t passwd_file_t

# Find domain transitions
sesearch --type_trans -s init_t -t httpd_exec_t

# Find boolean-controlled rules
sesearch --allow -s httpd_t -c tcp_socket -p name_connect -C

# seinfo - List policy components
seinfo -t | grep http  # List all types matching "http"
seinfo -r              # List all roles
seinfo -u              # List all users
seinfo -b | grep httpd # List booleans matching "httpd"
seinfo -x -t httpd_t   # Show attributes for httpd_t type
```

#### Other Useful Commands

```bash
# Check if a boolean exists
getsebool httpd_can_network_connect

# Find file context rules
semanage fcontext -l | grep /var/www

# Check expected vs actual context
matchpathcon -V /var/www/html/index.html

# List all customizations
semanage export

# Show port labels
semanage port -l | grep 8080

# Check if module is loaded
semodule -l | grep myapp

# View dontaudit rules (suppressed denials)
semodule -DB  # Disable dontaudit
# Generate denials...
semodule -B   # Re-enable dontaudit
```

### Common Denial Patterns

#### File Access Denials

**Problem**: Process can't read/write files

```bash
# AVC denial example
avc: denied { read } for comm="httpd" name="data.txt" scontext=system_u:system_r:httpd_t:s0 tcontext=unconfined_u:object_r:user_home_t:s0 tclass=file

# Solutions:
# 1. Fix file labeling (if file is in wrong location)
ls -Z /var/www/html/data.txt
restorecon -v /var/www/html/data.txt

# 2. Add file context rule (if file is in custom location)
semanage fcontext -a -t httpd_sys_content_t "/web/data.txt"
restorecon -v /web/data.txt

# 3. Check for boolean
audit2why < denial.log
# Might suggest: setsebool -P httpd_read_user_content 1

# 4. Create custom policy (last resort)
ausearch -m avc -ts recent | audit2allow -M myhttpd
semodule -i myhttpd.pp
```

#### Port Binding Denials

**Problem**: Service can't bind to custom port

```bash
# AVC denial
avc: denied { name_bind } for comm="httpd" src=8080 scontext=system_u:system_r:httpd_t:s0 tcontext=system_u:object_r:unreserved_port_t:s0 tclass=tcp_socket

# Solution: Add port label
semanage port -a -t http_port_t -p tcp 8080

# Verify
semanage port -l | grep 8080

# Restart service
systemctl restart httpd
```

#### Network Connection Denials

**Problem**: Process can't connect to network

```bash
# AVC denial
avc: denied { name_connect } for comm="httpd" dest=3306 scontext=system_u:system_r:httpd_t:s0 tcontext=system_u:object_r:mysqld_port_t:s0 tclass=tcp_socket

# Check boolean suggestion
ausearch -m avc -ts recent | audit2why
# Suggests: setsebool -P httpd_can_network_connect_db 1

# Enable boolean
setsebool -P httpd_can_network_connect_db 1

# For general network access
setsebool -P httpd_can_network_connect 1
```

#### Capability Denials

**Problem**: Process needs special capabilities

```bash
# AVC denial
avc: denied { dac_override } for comm="myapp" capability=1 scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:system_r:myapp_t:s0 tclass=capability

# This means myapp_t needs dac_override capability (bypass file permissions)

# Create policy module
cat > myapp_cap.te <<EOF
module myapp_cap 1.0;

require {
    type myapp_t;
    class capability dac_override;
}

allow myapp_t self:capability dac_override;
EOF

checkmodule -M -m -o myapp_cap.mod myapp_cap.te
semodule_package -o myapp_cap.pp -m myapp_cap.mod
semodule -i myapp_cap.pp
```

#### Domain Transition Denials

**Problem**: Process can't transition to new domain

```bash
# AVC denials (usually multiple)
avc: denied { execute } for comm="init" name="myapp" scontext=system_u:system_r:init_t:s0 tcontext=system_u:object_r:bin_t:s0 tclass=file
avc: denied { transition } for comm="init" exe="/usr/sbin/myapp" scontext=system_u:system_r:init_t:s0 tcontext=system_u:system_r:myapp_t:s0 tclass=process
avc: denied { entrypoint } for comm="myapp" path="/usr/sbin/myapp" scontext=system_u:system_r:myapp_t:s0 tcontext=system_u:object_r:bin_t:s0 tclass=file

# Solution: Create domain transition rules
cat > myapp_trans.te <<EOF
module myapp_trans 1.0;

require {
    type init_t, myapp_t, bin_t;
    class file { execute entrypoint };
    class process transition;
}

# Allow transition
domtrans_pattern(init_t, bin_t, myapp_t)
EOF

checkmodule -M -m -o myapp_trans.mod myapp_trans.te
semodule_package -o myapp_trans.pp -m myapp_trans.mod
semodule -i myapp_trans.pp
```

### Debugging Workflows

#### Standard Troubleshooting Workflow

```bash
# 1. Reproduce the issue
systemctl restart myapp
# Error occurs

# 2. Check recent AVC denials
ausearch -m avc -ts recent -i

# 3. Analyze denials with audit2why
ausearch -m avc -ts recent | audit2why

# 4. Check for boolean suggestions
# If audit2why suggests a boolean:
setsebool -P suggested_boolean 1

# 5. If no boolean exists, check file contexts
ls -Z /path/to/file
matchpathcon /path/to/file

# 6. Fix file context if wrong
restorecon -Rv /path/to/directory

# 7. If problem persists, use sealert
sealert -a /var/log/audit/audit.log

# 8. If still not resolved, create custom policy
ausearch -m avc -ts recent | audit2allow -M myapp_fix
semodule -i myapp_fix.pp

# 9. Test
systemctl restart myapp

# 10. Monitor for new denials
tail -f /var/log/audit/audit.log | grep AVC
```

#### Permissive Domain Debugging

```bash
# Make specific domain permissive for debugging
semanage permissive -a myapp_t

# Now myapp_t denials are logged but not enforced
# Perform all operations to generate complete denial log

# Collect all denials
ausearch -m avc -c myapp | audit2allow -M myapp_complete

# Review the generated policy
cat myapp_complete.te

# Install if appropriate
semodule -i myapp_complete.pp

# Remove permissive status
semanage permissive -d myapp_t

# Test in enforcing mode
```

#### Debugging Script

```bash
#!/bin/bash
# selinux_debug.sh - Quick SELinux debugging

COMMAND="$1"
shift
ARGS="$@"

if [ -z "$COMMAND" ]; then
    echo "Usage: $0 <command> [args]"
    exit 1
fi

# Clear previous audit logs marker
MARKER=$(date +%s)
logger "SELinux Debug: Starting $COMMAND at $MARKER"

# Run command
echo "Running: $COMMAND $ARGS"
$COMMAND $ARGS
RETVAL=$?

# Wait a moment for audit log
sleep 1

# Show denials
echo -e "\n=== AVC Denials ==="
ausearch -m avc -ts $MARKER -i 2>/dev/null

# Analyze
echo -e "\n=== Analysis ==="
ausearch -m avc -ts $MARKER 2>/dev/null | audit2why

# Suggest fix
echo -e "\n=== Suggested Fix ==="
ausearch -m avc -ts $MARKER 2>/dev/null | audit2allow -M ${COMMAND}_fix
if [ -f ${COMMAND}_fix.pp ]; then
    echo "Policy module created: ${COMMAND}_fix.pp"
    echo "To install: semodule -i ${COMMAND}_fix.pp"
fi

exit $RETVAL
```

### Performance Considerations

#### AVC Cache

SELinux uses an Access Vector Cache (AVC) to cache access decisions:

```bash
# View AVC statistics
seinfo --stats

# AVC cache stats in /proc
cat /proc/sys/fs/selinux/avc/cache_stats
# lookups hits misses allocations reclaims frees

# Increase AVC cache size if needed (default is usually sufficient)
# Edit /etc/selinux/semanage.conf or kernel parameters
```

#### Audit Performance Impact

```bash
# Disable audit logging temporarily (for performance testing)
auditctl -e 0

# Re-enable
auditctl -e 1

# Check audit status
auditctl -s

# Reduce audit log verbosity
# Add dontaudit rules to suppress unnecessary denials
# Example in policy:
# dontaudit httpd_t user_home_t:file read;

# Enable dontaudit rules
semodule -B

# Disable dontaudit rules (for debugging)
semodule -DB
```

#### Policy Size

```bash
# Check policy size
seinfo
# Statistics for policy file: /sys/fs/selinux/policy
# Policy Version & Type: v.31 (binary, mls)
#
#    Classes:           134    Permissions:       456
#    Types:            4972    Attributes:        256
#    Users:               9    Roles:              14
#    Booleans:          345    Cond. Expr.:       367
#    Allow:          109326    Neverallow:          0
#    Auditallow:        160    Dontaudit:       10234

# Large policies can impact performance
# Use targeted policy instead of strict
# Remove unused modules
semodule -r unused_module
```

---

## Advanced Topics

### Network Labeling

SELinux can label network packets using SECMARK (for packet filtering) and NetLabel (for CIPSO/CALIPSO).

#### SECMARK Integration with iptables

SECMARK allows labeling packets with SELinux contexts for fine-grained network access control.

```bash
# Requires iptables and SELinux integration
# See netfilter.md for iptables details

# Example: Label incoming HTTP traffic
iptables -t mangle -A INPUT -p tcp --dport 80 -j SECMARK --selctx system_u:object_r:http_packet_t:s0

# Save the mark for the connection
iptables -t mangle -A INPUT -j CONNSECMARK --save

# Restore mark for packets in existing connections
iptables -t mangle -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j CONNSECMARK --restore

# SELinux policy rule to allow httpd to receive http_packet_t
# In httpd policy:
allow httpd_t http_packet_t:packet { recv };

# Label outgoing packets
iptables -t mangle -A OUTPUT -p tcp --sport 80 -j SECMARK --selctx system_u:object_r:http_packet_t:s0
iptables -t mangle -A OUTPUT -j CONNSECMARK --save

# Allow httpd to send
allow httpd_t http_packet_t:packet { send };

# View packet labels
iptables -t mangle -L -n -v
```

#### NetLabel for Network Labeling

NetLabel provides CIPSO (Common IP Security Option) labeling for MLS networks:

```bash
# Install netlabel tools
yum install -y netlabel_tools

# Configure netlabel
# Example: Unlabeled network (most common)
netlabelctl map add default address:0.0.0.0/0 protocol:unlbl

# CIPSO for labeled network
netlabelctl cipsov4 add pass doi:1 tags:1,2,5,6
netlabelctl map add default address:192.168.1.0/24 protocol:cipsov4,1

# View netlabel configuration
netlabelctl -p map list

# SELinux policy for network labeling
# Allow domain to send/receive on labeled network
allow myapp_t netlabel_peer_t:peer recv;
allow myapp_t netlabel_peer_t:peer send;
```

### Multi-Level Security (MLS/MCS)

#### MLS Concepts

MLS implements the Bell-LaPadula model for classified information:

- **Sensitivity Levels**: s0 (unclassified) to s15 (top secret)
- **Categories**: c0 to c1023 (compartments)
- **Dominance**: s2 dominates s1; s1:c0,c1 dominates s1:c0

**Security Rules**:
- **No read up**: Process at s1 cannot read s2 files
- **No write down**: Process at s2 cannot write to s1 files

```bash
# MLS context format:
# user:role:type:sensitivity[:categories]
# user_u:user_r:user_t:s1:c0,c1

# View MLS range
id -Z
# unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023
#                                          ↑       ↑
#                                      clearance  categories

# MLS range notation:
# s0-s0:c0.c1023  means:
#   - Current level: s0
#   - Clearance: s0
#   - Categories: c0 through c1023
```

#### Configuring MLS

```bash
# Install MLS policy
yum install -y selinux-policy-mls

# Switch to MLS policy
vi /etc/selinux/config
# SELINUX=enforcing
# SELINUXTYPE=mls

# Relabel filesystem
fixfiles -F onboot
reboot

# After reboot, verify MLS is active
sestatus
# SELinux status:                 enabled
# Loaded policy name:             mls
# Policy MLS status:              enabled

# Create MLS users
semanage user -a -R "user_r" -r s0-s2:c0.c1023 mlsuser_u

# Map Linux user to MLS user
semanage login -a -s mlsuser_u -r s0-s2:c0.c1023 alice

# User logs in and gets MLS context
# alice logs in...
id -Z
# mlsuser_u:user_r:user_t:s0-s2:c0.c1023
```

#### MLS File Labeling

```bash
# Create files at different sensitivity levels
# As user with s0-s2 clearance:

# Create file at s0 (unclassified)
runcon -l s0 touch /tmp/unclassified.txt
ls -Z /tmp/unclassified.txt
# mlsuser_u:object_r:user_tmp_t:s0 /tmp/unclassified.txt

# Create file at s1 (confidential)
runcon -l s1 touch /tmp/confidential.txt
ls -Z /tmp/confidential.txt
# mlsuser_u:object_r:user_tmp_t:s1 /tmp/confidential.txt

# Create file at s2 (secret)
runcon -l s2 touch /tmp/secret.txt
ls -Z /tmp/secret.txt
# mlsuser_u:object_r:user_tmp_t:s2 /tmp/secret.txt

# Reading files:
# Process at s0 can read s0 files only
runcon -l s0 cat /tmp/unclassified.txt  # OK
runcon -l s0 cat /tmp/confidential.txt  # DENIED (read up)

# Process at s2 can read s0, s1, s2 files
runcon -l s2 cat /tmp/unclassified.txt  # OK (read down allowed)
runcon -l s2 cat /tmp/secret.txt        # OK

# Writing files:
# Process at s2 cannot write to s0 files (write down)
runcon -l s2 sh -c 'echo data > /tmp/unclassified.txt'  # DENIED

# Process at s0 cannot write to s2 files (write up)
runcon -l s0 sh -c 'echo data > /tmp/secret.txt'  # DENIED
```

#### MLS Categories

```bash
# Categories provide compartmentalization
# Example: c0 = Project A, c1 = Project B

# Create file in category c0
runcon -l s1:c0 touch /tmp/project_a.txt

# Create file in category c1
runcon -l s1:c1 touch /tmp/project_b.txt

# Process with only c0 cannot access c1 files
runcon -l s1:c0 cat /tmp/project_a.txt  # OK
runcon -l s1:c0 cat /tmp/project_b.txt  # DENIED

# Process with c0,c1 can access both
runcon -l s1:c0,c1 cat /tmp/project_a.txt  # OK
runcon -l s1:c0,c1 cat /tmp/project_b.txt  # OK
```

### Confined Users

Confined users have restricted capabilities compared to unconfined users:

```bash
# SELinux user types:
# - unconfined_u: No restrictions (default in targeted policy)
# - user_u: Restricted user, cannot su or sudo
# - staff_u: Can sudo to staff_t, limited admin tasks
# - sysadm_u: Can sudo to sysadm_t, full admin capabilities
# - guest_u: Very restricted, no network access, no X11

# View user mappings
semanage login -l
# Login Name           SELinux User         MLS/MCS Range        Service
# __default__          unconfined_u         s0-s0:c0.c1023       *
# root                 unconfined_u         s0-s0:c0.c1023       *

# Create restricted user
useradd -m restricteduser

# Map to user_u (restricted)
semanage login -a -s user_u restricteduser

# User logs in
# restricteduser logs in...
id -Z
# user_u:user_r:user_t:s0

# Restrictions:
# - Cannot su or sudo
# - Cannot execute files in /tmp (noexec)
# - Limited access to system resources

# Create staff user (can perform some admin tasks)
useradd -m staffuser
semanage login -a -s staff_u staffuser

# staffuser can sudo to staff_t
# As staffuser:
sudo -i
id -Z
# staff_u:staff_r:staff_t:s0-s0:c0.c1023

# Create sysadm user (full admin)
useradd -m adminuser
semanage login -a -s sysadm_u adminuser

# adminuser can sudo to sysadm_t (equivalent to root)
# As adminuser:
sudo -i
id -Z
# sysadm_u:sysadm_r:sysadm_t:s0-s0:c0.c1023

# Guest user (very restricted)
useradd -m guestuser
semanage login -a -s guest_u guestuser
# guest_u cannot:
# - Access network
# - Run programs in home directory
# - Use X11
# - Execute sudo/su
```

### Sandbox Environments

SELinux provides sandboxing capabilities for running untrusted code:

```bash
# Install sandbox tools
yum install -y policycoreutils-sandbox

# Run command in sandbox
sandbox firefox

# The sandboxed process:
# - Runs in sandbox_t domain
# - Has limited access to system
# - Cannot access network (by default)
# - Has temporary home directory

# Sandbox with network access
sandbox -M firefox

# Sandbox with access to specific directory
sandbox -M -H /tmp/sandbox_home firefox

# Custom sandbox options
sandbox -M -t sandbox_web_t -l s0:c100,c200 /usr/bin/myapp

# Check sandbox process
ps -eZ | grep sandbox
# user_u:user_r:sandbox_t:s0:c123,c456  5678 pts/0  00:00:01 firefox

# Sandbox configuration
cat /etc/sysconfig/sandbox
```

### Policy Constraints

Constraints add additional restrictions beyond type enforcement:

```bash
# Constraints in policy (usually in constraints file)

# Example: Users can only create files with their own user context
constrain file { create relabelto }
    (u1 == u2);

# Example: Only certain roles can transition to sysadm_r
constrain process { transition }
    (r1 == sysadm_r and r2 == sysadm_r) or
    (r1 == staff_r and r2 == sysadm_r);

# MLS constraints (enforcing Bell-LaPadula)
mlsconstrain file { read }
    (l1 dom l2);  # Process level must dominate file level

mlsconstrain file { write }
    (l1 eq l2);   # Write only at same level

# View active constraints
seinfo --constrain
```

### SELinux with Containers

#### Docker and SELinux

```bash
# Docker SELinux integration
# Containers run in svirt_lxc_net_t or container_t

# Check Docker SELinux status
docker info | grep -i security
# Security Options: selinux

# Run container with SELinux enabled
docker run -d --name web nginx

# Check process label
ps -eZ | grep nginx
# system_u:system_r:svirt_lxc_net_t:s0:c123,c456  7890 ? 00:00:00 nginx

# Volume labeling (private to container)
docker run -v /data:/data:Z nginx

# Volume labeling (shared across containers)
docker run -v /data:/data:z nginx

# Disable SELinux for specific container (not recommended)
docker run --security-opt label=disable nginx

# Custom SELinux label
docker run --security-opt label=type:svirt_apache_t nginx
```

#### Podman and SELinux

```bash
# Podman has better SELinux integration than Docker

# Run rootless container
podman run -d --name web nginx

# Check label
podman top web label
# system_u:system_r:container_t:s0:c123,c456

# Volume with :Z (private)
podman run -v /data:/data:Z nginx

# Volume with :z (shared)
podman run -v /data:/data:z nginx

# Check container labels
podman inspect --format='{{.ProcessLabel}}' web
podman inspect --format='{{.MountLabel}}' web
```

#### Kubernetes and SELinux

```bash
# Kubernetes pod security context with SELinux

apiVersion: v1
kind: Pod
metadata:
  name: selinux-pod
spec:
  securityContext:
    seLinuxOptions:
      level: "s0:c123,c456"
      type: "svirt_lxc_net_t"
  containers:
  - name: nginx
    image: nginx
    securityContext:
      seLinuxOptions:
        level: "s0:c123,c456"

# Volume with SELinux context
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-selinux
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /data
  seLinux:
    type: "svirt_sandbox_file_t"
    level: "s0:c123,c456"
```

### Integration with Namespaces

SELinux and Linux namespaces provide complementary isolation:

```bash
# Namespaces provide resource isolation (PID, network, mount, etc.)
# SELinux provides mandatory access control

# Example: Combining user namespace with SELinux
unshare --user --pid --fork --mount-proc bash

# Process still has SELinux context
id -Z
# unconfined_u:unconfined_r:unconfined_t:s0-s0:c0.c1023

# In containers, both are used:
# - Namespaces isolate resources (filesystem, network, PIDs)
# - SELinux (MCS) prevents cross-container access

# Example: Two containers with same namespaces but different MCS labels
# Container 1: s0:c100,c200
# Container 2: s0:c300,c400
# Even if they could see each other's files via namespace escape,
# SELinux would block access due to different categories
```

---

## Best Practices

### Policy Development Workflow

#### 1. Start with Permissive Mode for New Services

```bash
# Install and configure application first
systemctl start myapp

# Make domain permissive
semanage permissive -a myapp_t

# Exercise all functionality
# - Normal operations
# - Error conditions
# - Edge cases

# Collect denials
ausearch -m avc -c myapp > myapp_denials.log

# Generate comprehensive policy
audit2allow -M myapp < myapp_denials.log

# Review policy
cat myapp.te

# Install if appropriate
semodule -i myapp.pp

# Remove permissive status
semanage permissive -d myapp_t

# Test in enforcing mode
systemctl restart myapp
```

#### 2. Incremental Policy Development

```bash
# Don't create one giant policy module
# Instead, create focused modules:

# Base module: Core application permissions
# myapp_base.te

# Network module: Network-related permissions
# myapp_net.te

# Database module: Database access
# myapp_db.te

# This allows:
# - Easier maintenance
# - Selective enabling/disabling
# - Better organization
```

#### 3. Use Reference Policy Interfaces

```bash
# Don't reinvent the wheel
# Use existing reference policy interfaces

# Bad:
allow myapp_t etc_t:file { read open getattr };

# Good:
files_read_etc_files(myapp_t)

# Benefits:
# - Cleaner code
# - Consistent with system policy
# - Automatically updated with policy updates
```

#### 4. Document Your Policy

```bash
# Add comments to .te files
## <summary>
##   My Application - Web Service
## </summary>
## <desc>
##   <p>
##     This policy allows myapp to function as a web service,
##     including database access and log file writing.
##   </p>
## </desc>

# Document interfaces
## <summary>
##   Connect to myapp over TCP socket
## </summary>
## <param name="domain">
##   <summary>
##   Domain allowed access
##   </summary>
## </param>
```

### Testing Strategies

#### 1. Test in Virtual Machines

```bash
# Always test policy changes in VMs first
# - Easy to snapshot and restore
# - Safe to break
# - Can test bootup after changes

# Snapshot before changes
virsh snapshot-create-as test-vm before-selinux-change

# Make changes
semodule -i new_policy.pp

# Test
# If broken, restore snapshot
virsh snapshot-revert test-vm before-selinux-change
```

#### 2. Use Permissive Domains in Production

```bash
# Instead of disabling SELinux globally, use permissive domains

# Make specific domain permissive
semanage permissive -a myapp_t

# This allows:
# - SELinux remains enforcing for everything else
# - Denials are logged (for debugging)
# - myapp continues to work

# Fix policy based on logs
ausearch -m avc -c myapp | audit2allow -M myapp_fix
semodule -i myapp_fix.pp

# Remove permissive status
semanage permissive -d myapp_t
```

#### 3. Automated Testing

```bash
#!/bin/bash
# test_selinux_policy.sh

# Install policy
semodule -i myapp.pp

# Apply file contexts
restorecon -Rv /opt/myapp/

# Start service
systemctl start myapp

# Wait for startup
sleep 5

# Test functionality
curl http://localhost:8080/health
if [ $? -ne 0 ]; then
    echo "Health check failed"
    exit 1
fi

# Check for denials
DENIALS=$(ausearch -m avc -ts recent -c myapp | wc -l)
if [ $DENIALS -gt 0 ]; then
    echo "Found $DENIALS SELinux denials"
    ausearch -m avc -ts recent -c myapp | audit2why
    exit 1
fi

echo "Tests passed"
exit 0
```

### Security Hardening

#### 1. Principle of Least Privilege

```bash
# Grant only necessary permissions

# Bad:
allow myapp_t file_type:file { read write execute };  # Too broad!

# Good:
allow myapp_t myapp_data_t:file { read write };       # Specific types
allow myapp_t myapp_exec_t:file { execute };          # Only what's needed
```

#### 2. Use Booleans for Optional Features

```bash
# Don't grant permissions unconditionally
# Use booleans for features that may not be needed

# In policy:
gen_tunable(myapp_can_network, false)

if (myapp_can_network) {
    corenet_tcp_connect_all_ports(myapp_t)
}

# Administrators can enable as needed:
setsebool -P myapp_can_network on
```

#### 3. Confine All Network-Facing Services

```bash
# Never run network-facing services in unconfined_t
# Always create specific domains

# Check for unconfined network services
ps -eZ | grep unconfined_t | grep -E ':(httpd|sshd|mysqld|postfix)'

# If found, create or assign proper domain
```

#### 4. Regular Policy Audits

```bash
# Periodically review policy

# Find overly permissive rules
sesearch --allow -s myapp_t -p write | grep -v myapp_

# Find capabilities
sesearch --allow -s myapp_t -c capability

# Review custom policies
semodule -l -C  # List only custom modules

# Review each custom module
semodule -e myapp --extract
# Review myapp.te
```

### Common Pitfalls

#### 1. Disabling SELinux Instead of Fixing Issues

```bash
# Bad:
setenforce 0  # Gives up on SELinux

# Good:
ausearch -m avc -ts recent | audit2why  # Find root cause
# Fix with boolean, relabeling, or policy module
```

#### 2. Using chcon Instead of semanage

```bash
# Bad (temporary change):
chcon -t httpd_sys_content_t /web/index.html
# Lost on restorecon or relabel!

# Good (permanent change):
semanage fcontext -a -t httpd_sys_content_t "/web(/.*)?"
restorecon -Rv /web/
# Persists across relabels
```

#### 3. Overly Broad Policy Modules

```bash
# Bad:
audit2allow -M myapp < /var/log/audit/audit.log
# This includes ALL denials, possibly from other services!

# Good:
ausearch -m avc -c myapp -ts recent | audit2allow -M myapp
# Only denials from myapp
```

#### 4. Not Testing After Policy Changes

```bash
# Always test after changes!

# Install policy
semodule -i myapp.pp

# Restart service
systemctl restart myapp

# Verify no denials
ausearch -m avc -ts recent -c myapp

# Test functionality
curl http://localhost:8080/test
```

#### 5. Ignoring File Context Rules

```bash
# Creating files in wrong locations

# Bad:
mkdir /opt/web
cp index.html /opt/web/
# Default label: usr_t (not accessible by httpd)

# Good:
mkdir /opt/web
semanage fcontext -a -t httpd_sys_content_t "/opt/web(/.*)?"
restorecon -Rv /opt/web/
cp index.html /opt/web/
# Correct label: httpd_sys_content_t
```

### Performance Optimization

#### 1. Use dontaudit Rules

```bash
# Suppress harmless denials

# Many programs probe for optional features
# Example: httpd checking if user home dirs exist
# This generates denials even though it's not critical

# In policy:
dontaudit httpd_t user_home_dir_t:dir { search };

# This suppresses the denial from logs
# Reduces log noise and audit overhead
```

#### 2. Optimize AVC Cache

```bash
# Check AVC statistics
cat /proc/self/attr/current

# If high miss rate, consider kernel tuning
# (Usually not necessary on modern systems)
```

#### 3. Use Targeted Policy

```bash
# Targeted policy has better performance than strict/MLS
# Only confines necessary services

# Check current policy
sestatus | grep "policy name"

# For most use cases, targeted is sufficient
```

#### 4. Remove Unused Modules

```bash
# List all modules
semodule -l

# Remove unused modules
semodule -r unused_module1 unused_module2

# This reduces policy size and lookup time
```

---

## Resources

### Official Documentation

- **Red Hat SELinux Documentation**: https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/using_selinux/
- **SELinux Project**: https://github.com/SELinuxProject
- **SELinux Wiki**: https://selinuxproject.org/
- **NSA SELinux**: https://www.nsa.gov/what-we-do/research/selinux/

### Reference Policy

- **Reference Policy GitHub**: https://github.com/SELinuxProject/refpolicy
- **Reference Policy Documentation**: https://selinuxproject.org/page/ReferencePolicy

### Tools

- **audit2allow**: Generate policy modules from audit logs
- **audit2why**: Explain why SELinux denied access
- **sesearch**: Query SELinux policy
- **seinfo**: List policy components
- **semanage**: SELinux policy management tool
- **restorecon**: Restore file contexts
- **sealert**: User-friendly SELinux troubleshooting

### Books

- "SELinux System Administration" by Sven Vermeulen
- "SELinux by Example" by Frank Mayer, Karl MacMillan, David Caplan
- "The SELinux Notebook" (free): https://github.com/SELinuxProject/selinux-notebook

### Community

- **SELinux Mailing List**: selinux@vger.kernel.org
- **Fedora SELinux**: https://fedoraproject.org/wiki/SELinux
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/selinux

### Related Documentation in This Repository

- [Netfilter Patterns](netfilter.md) - Network filtering and SECMARK integration
- [Linux Namespaces](namespace.md) - Container isolation complementing SELinux
- [Kernel Patterns](kernel.md) - Linux kernel architecture including LSM

### Policy Examples

```bash
# Example policies from reference policy
cd /usr/share/selinux/devel/
# Contains example policies and Makefiles

# System policy source
cd /etc/selinux/targeted/
# Contains active policy files

# Custom policy development
mkdir ~/sepolicy
cd ~/sepolicy
# Create .te, .if, .fc files here
```

### Debugging Cheat Sheet

```bash
# Quick reference for common debugging tasks

# 1. Check SELinux status
getenforce
sestatus

# 2. View recent denials
ausearch -m avc -ts recent -i

# 3. Explain denials
ausearch -m avc -ts recent | audit2why

# 4. User-friendly analysis
sealert -a /var/log/audit/audit.log

# 5. Check file context
ls -Z /path/to/file
matchpathcon /path/to/file

# 6. Fix file context
restorecon -Rv /path/

# 7. Add permanent file context rule
semanage fcontext -a -t <type> "/path(/.*)?"
restorecon -Rv /path/

# 8. Add port label
semanage port -a -t <type> -p tcp <port>

# 9. Enable boolean
setsebool -P <boolean> on

# 10. Create policy module
ausearch -m avc -ts recent -c <program> | audit2allow -M <name>
semodule -i <name>.pp

# 11. Make domain permissive (debugging)
semanage permissive -a <domain>

# 12. Check for boolean-controlled rules
sesearch --allow -s <domain> -C

# 13. Query policy
sesearch --allow -s <source> -t <target>
seinfo -t | grep <pattern>

# 14. List customizations
semanage export
semanage fcontext -l -C
semanage port -l -C
semanage boolean -l -C
```
