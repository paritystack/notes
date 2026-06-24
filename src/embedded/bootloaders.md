# Bootloaders

## Overview

A bootloader is the first piece of code that runs on a microcontroller after reset. Its job is to bring up enough of the system to load and start the **application** (the user firmware). On PCs the bootloader is BIOS/UEFI → GRUB → kernel. On MCUs the chain is shorter but the same idea applies:

```
Power-on / Reset
       │
       ▼
┌──────────────┐    immutable, mask-programmed at the factory
│   BootROM    │    selects boot source (flash / UART / USB / SD)
└──────┬───────┘
       ▼
┌──────────────┐    OPTIONAL: your code, in protected flash region
│  Bootloader  │    verifies + loads the application image
└──────┬───────┘
       ▼
┌──────────────┐
│ Application  │    main firmware
└──────────────┘
```

On many MCUs (STM32F4, Nordic nRF52, ESP32) the BootROM is enough to flash code via UART/USB/JTAG, so a **custom bootloader is only needed if you need OTA, secure boot, dual-bank, or recovery**.

## Why You Need a Custom Bootloader

| Need | Why BootROM Isn't Enough |
|------|--------------------------|
| **OTA / field updates** | BootROM expects a host with a programmer attached |
| **Image verification** | BootROM doesn't check signatures |
| **A/B redundancy** | BootROM only knows one image location |
| **Fallback to recovery** | BootROM can't decide between two images |
| **Custom transport** | BootROM speaks UART/USB, not your LoRa link |
| **Encrypted firmware** | Plain BootROM loads cleartext only |

If you only need to flash via JTAG once and never update — skip the custom bootloader.

## Boot Flow on ARM Cortex-M

After reset the CPU does almost nothing magical:

1. Read **0x00000000** (or wherever VTOR points after BootROM remap) → load into MSP (main stack pointer)
2. Read **0x00000004** → jump to that address (Reset_Handler)

That's it. The "vector table" is literally an array starting with `{stack_top, reset_handler, NMI_handler, ...}`.

```c
// Linker places this at the start of flash
__attribute__((section(".isr_vector")))
const uint32_t vector_table[] = {
    (uint32_t)&_estack,            // 0x00: Initial MSP
    (uint32_t)Reset_Handler,       // 0x04: PC after reset
    (uint32_t)NMI_Handler,
    (uint32_t)HardFault_Handler,
    // ...
};
```

## Memory Layout for a Bootloader + App

The bootloader and app occupy non-overlapping flash regions. Typical 512KB STM32F4:

```
0x08000000  ┌──────────────────┐
            │   Bootloader     │  32 KB  (.text + vectors)
0x08008000  ├──────────────────┤
            │   App slot       │  240 KB (vectors + .text + .data init)
0x08044000  ├──────────────────┤
            │   App backup     │  240 KB (A/B redundancy, optional)
0x08080000  └──────────────────┘
```

**Critical**: the bootloader must be **smaller than the smallest flash sector it occupies alone**, or you can't update the app without risking the bootloader (sector erase is the smallest write unit on most MCUs).

### Linker Script Snippet

```ld
/* bootloader.ld */
MEMORY {
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 32K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

/* app.ld */
MEMORY {
    FLASH (rx)  : ORIGIN = 0x08008000, LENGTH = 240K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}
```

The app needs to know its flash offset because **the vector table location must match**.

## Vector Table Relocation (VTOR)

When the bootloader jumps to the app, the CPU still uses the vector table at 0x08000000 (bootloader's). Before jumping, the app — or the bootloader on the app's behalf — must set the **VTOR** (Vector Table Offset Register) so interrupts dispatch to the app's handlers.

```c
// Run from bootloader, just before jumping to app
#define APP_BASE  0x08008000

void jump_to_app(void) {
    uint32_t app_sp     = *(volatile uint32_t*)(APP_BASE + 0);
    uint32_t app_reset  = *(volatile uint32_t*)(APP_BASE + 4);

    // Sanity check: stack pointer should be in RAM
    if ((app_sp & 0x2FFE0000) != 0x20000000) {
        return;  // No valid app, stay in bootloader
    }

    // Disable interrupts and peripherals the bootloader enabled
    __disable_irq();
    SysTick->CTRL = 0;
    SysTick->LOAD = 0;
    SysTick->VAL  = 0;

    // Relocate vector table
    SCB->VTOR = APP_BASE;

    // Set stack pointer and jump
    __set_MSP(app_sp);
    ((void(*)(void))app_reset)();
}
```

On Cortex-M0/M0+ which lack VTOR, you must either:
- Place the app at flash origin and put the bootloader at the **end** (some chips have a remap feature), or
- Copy the app's vector table to RAM at known address and use SYSCFG_MEMRMP to remap RAM to 0x00000000.

## A/B (Dual-Bank) Updates

To survive a failed update, keep two app slots. The bootloader picks the valid one.

```
┌─────────────┐
│ Bootloader  │
├─────────────┤
│  Slot A     │ ◄─── currently running
├─────────────┤
│  Slot B     │ ◄─── new image written here during OTA
├─────────────┤
│  Metadata   │ ◄─── version, CRC, "boot from" flag
└─────────────┘
```

**Update flow:**
1. App downloads new image, writes to inactive slot (B)
2. App verifies CRC/signature of slot B, sets metadata → "boot B next"
3. App reboots
4. Bootloader reads metadata, loads B, increments a "boot attempts" counter
5. New app, on successful boot, sets metadata → "B confirmed", resets counter
6. If counter exceeds threshold (e.g., 3 reboots without confirmation), bootloader rolls back to A

```c
typedef struct {
    uint32_t magic;          // 0xB007C0DE
    uint8_t  active_slot;    // 0 or 1
    uint8_t  pending_slot;   // slot to try on next boot
    uint8_t  boot_attempts;  // incremented by bootloader, cleared by app
    uint8_t  rollback_armed;
    uint32_t slot_crc[2];
    uint32_t slot_version[2];
} boot_metadata_t;
```

Metadata lives in a dedicated flash sector or EEPROM. Some chips have **dual-bank flash** that lets you read from one bank while writing the other — STM32F4xx (-2 MB variants) and STM32H7 support this and it makes A/B trivial.

## Image Verification

A bootloader that trusts whatever is in slot B is a brick generator. Verify before booting.

| Method | Tamper-resistant? | Cost |
|--------|-------------------|------|
| **CRC32** | No (catches corruption only) | ~20 LOC, microseconds |
| **SHA-256** | No (still need a known-good hash) | Few KB code, ms |
| **HMAC** | Yes (against attacker without the shared key) | Requires key in MCU |
| **ECDSA / RSA signature** | Yes (asymmetric, only public key on device) | ~10 KB code, 100+ ms |

For consumer products that need to be field-updatable safely, **signed images with ECDSA** are now the de-facto standard. mbedTLS, MicroECC, or vendor crypto libraries (STM32 X-CUBE-CRYPTOLIB) provide the verify routine in a few KB.

```c
// Pseudo-code for a verified boot
boot_metadata_t* meta = (void*)META_ADDR;
uint8_t slot = meta->pending_slot;
uint32_t img_addr = (slot == 0) ? SLOT_A_ADDR : SLOT_B_ADDR;

if (verify_signature(img_addr, &meta->signature[slot], &public_key) != OK) {
    // Fall back to the other slot
    slot ^= 1;
    img_addr = (slot == 0) ? SLOT_A_ADDR : SLOT_B_ADDR;
    if (verify_signature(img_addr, &meta->signature[slot], &public_key) != OK) {
        enter_recovery_mode();   // both slots bad
    }
}
meta->boot_attempts++;
jump_to_app(img_addr);
```

See [secure_boot.md](secure_boot.md) for the full chain-of-trust story.

## Communicating With the Bootloader

Common ways the app (or a host) signals the bootloader to enter update mode:

| Mechanism | How |
|-----------|-----|
| **Magic value in RAM** | App writes `0xDEADBEEF` to a noinit RAM region, soft-resets. Bootloader checks on entry. |
| **GPIO at boot** | Hold a button at power-on. Bootloader samples pin. |
| **Backup register** | RTC backup registers survive reset, used as flag. |
| **Always-update protocol** | Bootloader waits N ms for a host "hello" on UART/USB, then proceeds. |

```c
// In .noinit section so it survives a soft reset
__attribute__((section(".noinit"))) volatile uint32_t boot_flag;

// In app: request bootloader mode
boot_flag = 0xB007L0AD;
NVIC_SystemReset();

// In bootloader Reset_Handler (before zeroing BSS!)
if (boot_flag == 0xB007L0AD) {
    boot_flag = 0;
    enter_dfu_mode();
}
```

## DFU and Standard Protocols

Rather than inventing a transport, use a standard:

- **USB DFU** (Device Firmware Upgrade, USB-IF class) — `dfu-util` on the host. Built into STM32 BootROM.
- **MCUboot** — open-source bootloader for Cortex-M, supports A/B, signed images, encrypted images. Used by Zephyr, Mbed, Pebble.
- **U-Boot / Coreboot** — heavier, mostly Linux-targeted; not common on bare-metal MCUs.
- **Vendor:** STM32 supports UART/USB/CAN/I2C DFU via BootROM; ESP32 has esptool over UART; Nordic has nrfutil over UART/BLE.

**MCUboot** is worth knowing — it implements 95% of what a serious bootloader needs (A/B with rollback, image confirmation, signatures, encrypted images) and the design is well-documented.

## ESP32 Boot Flow (As a Contrast)

ESP32 is two-stage and very different from Cortex-M:

```
ROM bootloader (in chip mask ROM)
   │ reads bootloader from flash at 0x1000
   ▼
Second-stage bootloader (your build, ~30 KB)
   │ reads partition table at 0x8000
   │ loads app from chosen OTA partition
   ▼
Application
```

The partition table is a tiny CSV describing flash regions (factory, ota_0, ota_1, nvs, ...). ESP-IDF's `esp_ota_*` APIs handle A/B for you. Boot selection lives in `otadata` partition.

## Common Pitfalls

### Pitfall 1: Forgetting to Disable Interrupts Before Jumping

```c
// BAD: jumping with SysTick still enabled
SCB->VTOR = APP_BASE;
__set_MSP(app_sp);
((void(*)(void))app_reset)();  // SysTick fires → jumps into app's
                               // SysTick_Handler before app inits clocks → fault
```

Always quiesce the system: `__disable_irq()`, stop SysTick, disable peripheral clocks the bootloader enabled.

### Pitfall 2: Stack Pointer Validation

If the app slot is blank (`0xFFFFFFFF` everywhere), `app_sp = 0xFFFFFFFF` and `app_reset = 0xFFFFFFFF`. Jumping there hardfaults. Always sanity-check.

```c
if (app_sp == 0xFFFFFFFF || app_reset == 0xFFFFFFFF) {
    enter_recovery();
}
```

### Pitfall 3: Bootloader and App With Different Compiler Settings

Different `-O` levels, different libc, different startup code → subtle ABI mismatches. Keep both projects in the same workspace with shared toolchain config.

### Pitfall 4: Erasing the Sector You're Running From

The flash controller stalls the bus during erase/write. If the bootloader is in the same sector as code that's erasing → CPU hangs or jumps to garbage. Either:
- Put flash routines in RAM (`__attribute__((section(".ramfunc")))`), or
- Use dual-bank flash, or
- Ensure the bootloader never writes to its own sector.

### Pitfall 5: Forgetting App's Vector Table Offset

App's linker script keeps `ORIGIN = 0x08000000` instead of `0x08008000`. App "works" because flash mirrors at 0x00000000, but VTOR is wrong and interrupts dispatch to bootloader handlers.

### Pitfall 6: Power Loss Mid-Update

Without dual-bank, a power cut during app erase = brick. Always have A/B or have the bootloader self-sufficient enough to re-download via its own DFU mode.

## Anti-Brick Recovery

For a "never bricks" device:

1. Bootloader is **never updated** in the field (or only via JTAG, or only by writing to the *other* of two bootloader slots, then atomically switching via option bytes).
2. Bootloader **always** has a self-contained update path (USB-DFU, UART) that doesn't depend on app code.
3. App images are **always** signed and verified.
4. **Watchdog confirmation**: new app must explicitly confirm "I'm healthy" within N seconds, or bootloader rolls back.

## Summary

1. **BootROM runs first** — many projects don't need anything else.
2. **Custom bootloader is needed for OTA, A/B, secure boot, recovery.**
3. **Vector table relocation (VTOR)** is the critical bit when jumping bootloader → app.
4. **A/B with metadata + rollback counter** is the standard pattern for safe updates.
5. **Verify images** with signatures, not just CRC, if attackers are in the threat model.
6. **MCUboot** is the de-facto open-source bootloader — read its source even if you write your own.
7. **Disable interrupts and peripherals** before jumping to app.
8. **Never put flash-write code in the sector being written.**

## See Also

- [OTA Updates](ota_updates.md)
- [Linker Scripts](linker_scripts.md)
- [Interrupts](interrupts.md) — vector table mechanics
- [Watchdog](watchdog.md) — rollback trigger on hung apps

## Where this connects

- [Secure boot](secure_boot.md) — adds signature verification to the boot chain
- [OTA updates](ota_updates.md) — how new images reach the bootloader
- [Startup code](startup_code.md), [Linker scripts](linker_scripts.md) — vector tables and memory layout it manages
- [Flash filesystems](flash_filesystems.md) — image storage and A/B slots
- [USB](usb.md) — DFU as a bootloader transport
