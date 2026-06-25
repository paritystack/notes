# journald & Logging

## Overview

This page covers how log messages flow on a Linux system: the kernel's `printk` ring buffer and
`dmesg`, the classic `/dev/log` syslog socket, **systemd-journald**'s structured binary journal
(`journalctl`), forwarding to **syslog/rsyslog**, and rotation with **logrotate**. It builds on
[systemd](systemd.md) (journald is a systemd component, and units' stdout/stderr are captured by
it), references the `printk` log-level tuning exposed via [sysctl](sysctl.md), and the kernel ring
buffer originates in [Kernel Architecture](kernel.md).

There are two origins — the **kernel** and **userspace** — that converge in the journal, which can
then fan out to syslog and remote collectors.

```
   kernel  printk ──► kernel ring buffer ──► dmesg / journald (kmsg)
                                                   │
   userspace  app stdout/stderr (systemd unit) ───►│
              syslog(3) / /dev/log ───────────────►│  systemd-journald
              sd_journal_send (structured fields) ─►│   (binary, indexed,
                                                     │    structured journal)
                                                     ├──► journalctl (query)
                                                     └──► (Forward) rsyslog ──► /var/log/*.log
                                                                                └─► remote (TCP/RELP)
```

## Kernel logging: printk and the ring buffer

The kernel logs via `printk()` into a fixed-size **ring buffer** (`/dev/kmsg`). Each message has a
**log level** (`KERN_EMERG`=0 … `KERN_DEBUG`=7). The console only shows messages below a
threshold:

```bash
dmesg                       # dump the ring buffer
dmesg -w                    # follow (like tail -f)
dmesg -l err,warn           # filter by level
dmesg -H                    # human times, paged
cat /proc/sys/kernel/printk # console loglevel: current default minimum_boot
sysctl -w kernel.printk='3 4 1 7'   # quiet the console (see sysctl.md)
```

Because the buffer is fixed-size it **wraps** — old kernel messages are lost unless something
(journald) persists them. journald reads `/dev/kmsg` so kernel messages also land in the journal.

## systemd-journald

journald is the modern collector. It captures: the kernel ring buffer, the `/dev/log` syslog
socket, the native structured API (`sd_journal_send`), audit messages, and **stdout/stderr of
every systemd unit**. It stores them in an **indexed binary journal** keyed by rich metadata
fields (`_SYSTEMD_UNIT`, `_PID`, `_UID`, `PRIORITY`, `_BOOT_ID`, …), which is what makes filtering
fast and precise:

```bash
journalctl -u nginx.service          # one unit
journalctl -b                        # this boot   (-b -1 = previous boot)
journalctl -p err                    # priority ≤ err
journalctl -f                        # follow
journalctl --since "10 min ago" --until now
journalctl _PID=1234                 # by structured field
journalctl -k                        # kernel messages only (= dmesg, persisted)
journalctl -o json-pretty            # raw structured output
journalctl --disk-usage              # journal size
```

### Persistence and rotation

By default many distros keep the journal in **volatile** `/run/log/journal` (RAM, lost on reboot).
To persist across reboots, make `/var/log/journal` exist (`Storage=persistent` in
`/etc/systemd/journald.conf`). journald rotates *itself* by size/time — no logrotate needed for the
journal:

```ini
# /etc/systemd/journald.conf
Storage=persistent
SystemMaxUse=1G          # cap total journal size
MaxRetentionSec=1month
RateLimitIntervalSec=30s # drop bursts beyond...
RateLimitBurst=10000     # ...this many messages per interval
```

```bash
journalctl --vacuum-size=500M        # prune now by size
journalctl --vacuum-time=2weeks      # prune by age
```

## syslog, rsyslog and the classic /var/log

The traditional model uses the **syslog protocol**: messages tagged with a **facility**
(`auth`, `cron`, `mail`, `kern`, `local0`–`7`) and a **severity** (`emerg`…`debug`), written to a
socket and sorted into text files. **rsyslog** (or syslog-ng) is the daemon; on systemd hosts it
typically reads from journald (`omjournal`/imjournal) rather than `/dev/log` directly.

```
   facility.severity  ──►  destination          (rsyslog rule)
   authpriv.*              /var/log/auth.log
   mail.*                  /var/log/mail.log
   *.emerg                 :omusrmsg:*           (wall to all users)
   *.*                     @@logserver:514       (forward over TCP)
```

Why keep rsyslog alongside journald? Plain-text files for legacy tooling, and **remote
forwarding/aggregation** (UDP 514, TCP, or reliable **RELP**) to a central log server or SIEM —
journald's own remote shipping (`systemd-journal-remote`) is less widely deployed.

## logrotate

Plain-text logs (`/var/log/*.log`, app logs) grow unbounded, so **logrotate** (a cron/timer job)
rotates, compresses, and prunes them. It does *not* manage the journal (journald self-rotates).

```
# /etc/logrotate.d/myapp
/var/log/myapp/*.log {
    daily
    rotate 14            # keep 14 old copies
    compress             # gzip rotated files
    delaycompress        # don't compress the most recent rotation
    missingok
    notifempty
    copytruncate         # rotate without the app reopening (or use postrotate + reload)
}
```

`copytruncate` vs a `postrotate` reload signal is the key choice: `copytruncate` risks losing a
few in-flight lines but needs no app cooperation; signalling the app to reopen its file is cleaner
if it supports `SIGHUP`.

## Choosing where logs go

| You want | Use |
|----------|-----|
| Query a service's logs with metadata | `journalctl -u …` (journald) |
| Kernel messages (live or persisted) | `dmesg` / `journalctl -k` |
| Plain-text files for legacy tools | rsyslog reading from journald |
| Central aggregation / SIEM | rsyslog forward (TCP/RELP) or journal-remote |
| Cap growth of app text logs | logrotate |
| Structured app logging | `sd_journal_send` / log JSON to stdout (captured by journald) |

## Where this connects

- [systemd](systemd.md) — journald is a systemd service; a unit's stdout/stderr is captured
  automatically and queried with `journalctl -u`.
- [sysctl](sysctl.md) — `kernel.printk` tunes the console log level for the kernel ring buffer.
- [Kernel Architecture](kernel.md) — `printk` and the ring buffer are where kernel diagnostics
  originate before journald reads `/dev/kmsg`.
- [Container Runtimes](container_runtimes.md) — container stdout/stderr is collected by the runtime
  (json-file/journald driver), the userspace analogue of unit logging.

## Pitfalls

- **Journal not persisting across reboots.** Default volatile storage drops logs at reboot; create
  `/var/log/journal` (or set `Storage=persistent`) to investigate past boots.
- **Disk filled by an unbounded journal.** Without `SystemMaxUse`/retention caps a chatty service
  fills `/var/log`; set limits and `--vacuum-*` to recover.
- **Double-logging.** Running journald *and* rsyslog both capturing everything duplicates storage;
  pick one as the system of record and have the other read from it.
- **logrotate on a file the app holds open.** Without `copytruncate` or a reload signal, the app
  keeps writing to the (now-renamed) old inode and the new file stays empty.
- **Reading `dmesg` for old events.** The kernel ring buffer wraps and is volatile; use
  `journalctl -k -b -1` for kernel messages from a previous boot.
- **`kernel.printk` too verbose on serial console.** Flooding a slow serial console with debug
  printk can itself stall the system; quiet the console level and rely on the journal.
- **Trusting log timestamps across boots.** Pre-NTP early-boot timestamps can be wrong; correlate
  with `_BOOT_ID` and monotonic time, not just wall-clock.
