# Flash Filesystems

## Overview

Embedded devices that need to store config, logs, calibration data, or downloaded firmware reach for a flash filesystem. But MCU flash is **not a disk** — you can't write arbitrary bytes whenever you want. The flash filesystem's job is to hide the awkward physics of flash behind a normal `open/read/write/close` API.

Three flash physics constraints drive every design decision:

1. **Erase before write.** You can only flip bits 1→0 by writing. To flip 0→1 you must erase the entire sector (4 KB to 256 KB typically).
2. **Limited erase cycles.** ~10,000–100,000 per sector for NOR, ~1,000–10,000 for NAND. Hot spots wear out.
3. **Power-fail in the middle.** An interrupted write/erase leaves the sector in an unpredictable partial state.

```
Naïve FS                       Flash FS
                              
  fopen("a") → write           fopen("a") → wear-leveling layer
       │                              │
       │ "just write 4 bytes"          │ pick least-worn block
       │                              │ append log entry
  flash sector                  flash sector
  (erased entirely              (append-only, GC reclaims
   on every overwrite)           when full)
```

## Flash Types

| Type | Density | Erase block | Write granularity | Use |
|------|---------|-------------|-------------------|-----|
| **NOR (internal)** | KB–MB | 1–256 KB sector | bytes/words | MCU on-chip flash, config storage |
| **NOR (serial / SPI)** | KB–256 MB | 4–64 KB sector | bytes (often 256 B page) | Code XIP, config, FW images |
| **NAND (raw)** | 128 MB–GBs | 128 KB block | 2 KB page (whole pages) | Bulk storage; needs ECC and bad-block management |
| **NAND (eMMC/SD)** | GB–TB | Hidden behind FTL | sector (512 B) | "Just looks like a disk" |
| **FRAM / MRAM** | KB–MB | None | Byte-level, no erase | High-endurance, low-density alternative |

For MCU flash filesystems we mostly care about **internal NOR** and **SPI NOR**. eMMC/SD already has an FTL inside and you put FatFs / ext4 / similar on top.

## Wear Leveling

Without it, frequently-updated regions die. Two strategies:

- **Dynamic wear leveling**: spread *new* writes across less-worn blocks. Static blocks (rarely touched) keep their original locations.
- **Static wear leveling**: periodically *move* cold data into worn blocks so all blocks age evenly. Higher write amplification but better worst-case lifetime.

Most MCU filesystems do dynamic, and that's fine for typical workloads (config + occasional logs).

## Power-Fail Safety

Two pieces:

- **Atomic operations**: the FS's data structures move from one consistent state to another. Crash mid-write → on reboot, FS either sees the old state or the new state, never half-way.
- **Append-only journals**: instead of overwriting in place (which requires erase-then-write), append new versions and mark old as garbage. GC reclaims old space later.

**Rule of thumb**: if your "flash FS" can lose data on a power cut, it's not really a flash FS. It's a binary blob you wrote to flash.

## LittleFS

Made by ARM (now part of Arm Mbed). Power-fail-safe, wear-leveling, tiny (~30 KB code), used in production by Pebble, Particle, many IoT devices. **Current default choice** for new MCU projects.

### Properties

- **COW (copy-on-write) data structures.** Every metadata update produces new blocks; old blocks freed only after the new pointer is durable.
- **Two-list directory format.** Directories are linked lists of metadata pairs; updates atomically commit a new pair, garbage-collected later.
- **Bounded RAM.** RAM usage scales with cache size you configure, not with filesystem size.
- **No "mount table".** Just call `lfs_mount()` with your config struct.

### Minimal Setup

```c
#include "lfs.h"

lfs_t lfs;
lfs_file_t file;

// User-provided block device interface
int block_read(const struct lfs_config* c, lfs_block_t block,
               lfs_off_t off, void* buffer, lfs_size_t size) {
    return flash_read(block * c->block_size + off, buffer, size);
}
int block_prog(const struct lfs_config* c, lfs_block_t block,
               lfs_off_t off, const void* buffer, lfs_size_t size) {
    return flash_program(block * c->block_size + off, buffer, size);
}
int block_erase(const struct lfs_config* c, lfs_block_t block) {
    return flash_erase(block * c->block_size, c->block_size);
}
int block_sync(const struct lfs_config* c) { return 0; }

const struct lfs_config cfg = {
    .read  = block_read,
    .prog  = block_prog,
    .erase = block_erase,
    .sync  = block_sync,
    .read_size      = 16,
    .prog_size      = 16,
    .block_size     = 4096,
    .block_count    = 256,         // 1 MB
    .cache_size     = 64,
    .lookahead_size = 16,
    .block_cycles   = 500,         // wear-leveling threshold
};

void boot(void) {
    int err = lfs_mount(&lfs, &cfg);
    if (err) {
        lfs_format(&lfs, &cfg);   // first boot
        lfs_mount(&lfs, &cfg);
    }

    uint32_t boot_count = 0;
    lfs_file_open(&lfs, &file, "boot_count", LFS_O_RDWR | LFS_O_CREAT);
    lfs_file_read(&lfs, &file, &boot_count, sizeof(boot_count));
    boot_count++;
    lfs_file_rewind(&lfs, &file);
    lfs_file_write(&lfs, &file, &boot_count, sizeof(boot_count));
    lfs_file_close(&lfs, &file);

    lfs_unmount(&lfs);
}
```

### When LittleFS Hurts

- **Many small writes to one file.** Each `lfs_file_write` + close creates a new metadata block. Better: cache writes in RAM, flush in chunks.
- **No directory listing at boot speed.** Mount is fast, but enumerating a directory with 10,000 files involves walking the linked list.
- **Not a database.** Don't use it as one. SQLite or a custom record store sits on top.

## SPIFFS

Older. SPI Flash File System. Was the default for ESP8266 / early ESP32. Replaced by LittleFS in most projects.

- **Flat namespace** (no directories).
- **Log-structured.** All writes append.
- **Designed for SPI NOR specifically.**
- **Known bug-prone**: there's a long list of "SPIFFS corrupted" reports online.
- **GC is slow** when the filesystem gets close to full (>80%).

Use SPIFFS only if you're maintaining legacy code. New projects: LittleFS.

## FatFs

Generic FAT12/FAT16/FAT32/exFAT implementation by ChaN. Designed for SD cards but works on any block device.

- **Not power-fail-safe** out of the box. A power cut during write can corrupt the FAT itself.
- **No wear leveling.** The FAT is rewritten constantly on the same sectors.
- **Compatible with desktop OS.** Pull the SD card out, plug into a PC, files are there.

```c
FATFS fs;
FIL file;
UINT bw;

f_mount(&fs, "", 0);
f_open(&file, "log.txt", FA_WRITE | FA_OPEN_APPEND);
f_write(&file, "hello\n", 6, &bw);
f_close(&file);
```

Use FatFs only when:
- You're on SD card / eMMC (which has its own FTL handling wear leveling and power-fail).
- You need files readable on a PC.
- You're never writing — read-only resource access.

## eMMC / SD Filesystems

eMMC and SD cards contain their own controller and **flash translation layer (FTL)** that:
- Maps logical sectors to physical NAND pages.
- Handles wear leveling internally.
- Manages bad blocks.
- Implements (some) power-fail tolerance.

So you treat them like a normal disk: put FAT, exFAT, or even ext2/ext4 on top. The FS doesn't need wear-leveling logic itself.

But: **cheap SD cards have terrible FTLs.** Power-cut during a write can corrupt anything on the card, even data you weren't writing to. Industrial SD cards are designed for power-cut safety; consumer SD cards aren't. For reliable storage, use industrial/automotive-grade cards or eMMC.

## NVS / Key-Value Stores

When all you need is "save 50 bytes of config across reboots", a full FS is overkill. Use a key-value store:

- **ESP-IDF NVS** (Non-Volatile Storage): K/V over NVS partition. Power-fail safe, wear-leveled.
- **Zephyr NVS** (`nvs_*` API): similar.
- **Custom append-only log**: write `{key, value}` records to a flash sector. On read, scan and use latest. On reaching the end, compact to the next sector. ~200 lines of code.

```c
// ESP-IDF NVS example
nvs_handle_t h;
nvs_open("storage", NVS_READWRITE, &h);
nvs_set_i32(h, "boot_count", count);
nvs_commit(h);
nvs_close(h);
```

NVS is the right tool for config. LittleFS is overkill for "save 4 bytes when the user presses save".

## Choosing Block Size and Layout

For LittleFS on internal NOR (e.g., STM32):
- `block_size` = sector erase size (varies per chip — STM32F4 has 16K/64K/128K mixed sectors; **use the smaller ones for FS**, big ones for code).
- `block_count` = number of sectors dedicated to FS.

Keep the FS region **physically separate** in the linker script:

```ld
MEMORY {
    FLASH    (rx)  : ORIGIN = 0x08000000, LENGTH = 768K   /* code */
    FS_FLASH (r)   : ORIGIN = 0x080C0000, LENGTH = 256K   /* littlefs */
    RAM      (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}
```

Then in code, point the LittleFS block interface at addresses `0x080C0000..0x08100000`.

For external SPI NOR (e.g., 16 MB W25Q128): block_size 4 KB, block_count 4096. Reserve regions for fonts, OTA image, FS partitions, etc., in a partition table.

## Performance Notes

- **Reads are fast.** Random read is ~100s of ns on internal flash, ~tens of µs on SPI NOR.
- **Writes are slow.** 100 µs to few ms per page on internal flash; SPI NOR depends on bus speed.
- **Erases are very slow.** 50-500 ms per sector. Largest contributor to perceived latency.
- **GC happens during writes.** A write that triggers GC can stall for hundreds of ms. Plan for this — don't run GC-triggering writes inside ISRs or time-critical paths.

For latency-sensitive systems, **double-buffer** logs in RAM and flush in a low-priority task.

## Encryption

Three approaches:
- **Filesystem-level**: ESP32 NVS Encryption, Zephyr NVS with custom crypto.
- **Block-device-level**: encrypt block_prog / decrypt block_read in the LittleFS callbacks. AES-CTR or XTS, key in eFuse / RTC backup.
- **Whole-chip encryption**: ESP32 flash encryption mode; STM32 OTFDEC (on-the-fly decryption); NXP CSF.

For OTA-distributed firmware, encryption is usually whole-image, not FS-level.

## Common Pitfalls

### Pitfall 1: Writing to Sector That Code Runs From

Erasing a sector stalls the AHB bus. If the CPU is fetching instructions from that sector, it hangs. Put the FS in a different sector than code, or move flash routines to RAM.

### Pitfall 2: Wrong Erase Size

Misconfiguring `block_size` smaller than the actual sector erase size → LittleFS thinks it cleared a block but neighboring data was also wiped. Data corruption that looks random.

### Pitfall 3: Power-Cut During Mount

Sometimes a crash leaves the FS in a state that takes long to scan/recover on the next mount. Plan for slow first-boot or use Watchdog with a generous timeout for the mount call.

### Pitfall 4: Filling to 100%

LittleFS keeps working until ~95% full; SPIFFS degrades badly above 80%. Don't fill the FS — leave headroom for GC.

### Pitfall 5: Treating FS Like a Database

Constant tiny updates to a single file thrashes metadata. Either batch writes or use NVS for small structured data.

### Pitfall 6: Ignoring `block_cycles`

LittleFS's wear-leveling kicks in when a block exceeds `block_cycles` erases. Default is 500 — fine for most. Set to -1 to disable (only if you don't care about wear). Setting it too low causes excessive metadata churn.

### Pitfall 7: Using `printf` to Log to FS

`printf` plus FS write per log line plus FS-FLUSH = slow + flash wear. Use a circular RAM log + periodic batched flush instead.

### Pitfall 8: SD Card Power Cut

Yanking a SD card while writing corrupts arbitrary sectors due to in-card FTL behavior. Hot-plug-safe means: explicit unmount, then remove.

## Quick Comparison

| FS | Power-fail safe? | Wear leveling | Footprint | Best for |
|----|------------------|---------------|-----------|----------|
| **LittleFS** | Yes | Yes | ~30 KB | MCU NOR flash, default choice |
| **SPIFFS** | Sort of | Dynamic | ~20 KB | Legacy ESP code |
| **FatFs** | No | No | ~10-30 KB | SD cards, PC interop |
| **NVS (Zephyr/ESP)** | Yes | Yes | ~10 KB | Config / small K/V |
| **ext4/F2FS** | Yes | No (FTL does it) | Linux-only | eMMC on Linux MCUs |

## Summary

1. **Flash filesystems exist because flash is not a disk** — erase-before-write, limited cycles, power-fail.
2. **LittleFS** is the modern default for MCU flash.
3. **SPIFFS** is legacy; avoid for new work.
4. **FatFs** when you need PC compatibility or are on a real FTL (SD/eMMC).
5. **NVS** for small config-only use cases.
6. **Wear leveling** spreads erases; **power-fail safety** uses append + COW.
7. **Place FS in its own flash region** distinct from code sectors.
8. **Don't fill to 100%.** Leave headroom for GC.
9. **Batch writes** rather than tiny frequent updates.
10. **Encrypt at block-device layer** if needed.

## See Also

- [Linker Scripts](linker_scripts.md) — carving out an FS region
- [Bootloaders](bootloaders.md) — storing OTA images
- [OTA Updates](ota_updates.md) — downloading firmware to flash
- [SDIO](sdio.md) — SD card hardware interface
