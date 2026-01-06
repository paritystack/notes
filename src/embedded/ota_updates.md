# Over-the-Air (OTA) Updates

> **Domain:** Embedded Systems, IoT, Firmware
> **Key Concepts:** Bootloader, Dual Bank Partitioning, Image Signing, Rollback

**Over-the-Air (OTA) Updates** allow embedded devices to upgrade their firmware remotely. In the IoT era, this is a strict requirement for patching security vulnerabilities and adding features post-deployment.

---

## 1. The Core Challenge

Updating a microcontroller is dangerous. If power fails during the write process, or if the new firmware is buggy (e.g., infinite loop on boot), the device becomes a "brick." A truck roll to fix it costs $500+.

**The Golden Rule:** The device must **never** be unbootable.

---

## 2. Partitioning Strategies

### 2.1. Dual Bank (A/B Partitioning)
The industry standard for reliability. Flash memory is divided into two large slots (banks).

*   **Bank A (Active):** Running the current firmware (v1.0).
*   **Bank B (Passive):** Empty or holding old firmware.

**The Flow:**
1.  Device downloads v1.1 into **Bank B** while v1.0 keeps running.
2.  Download verifies (Checksum/Signature).
3.  Device sets a flag in non-volatile memory: `BOOT_FROM_BANK_B`.
4.  Reboot.
5.  **Bootloader** sees the flag, maps Bank B to the execution address, and jumps to it.

**Pros:** Zero downtime during download. Instant rollback (just switch the flag back).
**Cons:** Requires 2x Flash memory.

### 2.2. Single Bank (Compression / Swap)
Used on constrained devices.
1.  Download update to an external SPI Flash or a temporary "Swap" partition.
2.  Reboot into Bootloader.
3.  Bootloader erases the App partition and copies the new image from the Swap partition.
4.  Boot.

**Cons:** High risk. If copy fails, no valid app exists (must stay in Bootloader).

---

## 3. The Bootloader Logic

The Bootloader is the immutable code at the start of Flash (e.g., `0x08000000`). It runs before the App.

```c
void main() {
    Status status = read_boot_status();
    
    if (status == UPDATE_PENDING) {
        if (verify_signature(BANK_B)) {
            swap_banks(); // or just update the mapping
            commit_update(); // Mark as "Tentative"
        }
    }
    
    // Watchdog Logic
    if (status == TENTATIVE && crashed_last_time()) {
        rollback(); // Switch back to Bank A
    }
    
    jump_to_app();
}
```

---

## 4. Security: The Chain of Trust

An attacker who can push their own firmware owns the device.

### 4.1. Image Signing
1.  **Build Server:** Has a **Private Key**. Hashes the binary (SHA-256) and encrypts the hash with the Private Key (Signature). Appends signature to the binary.
2.  **Device:** Has the matching **Public Key** burned into read-only memory (or eFuse).
3.  **Verification:** Bootloader decrypts the signature with Public Key. Hashes the downloaded binary. If `Hash_Calc == Hash_Decrypted`, the image is authentic.

### 4.2. Anti-Rollback
Attacker pushes a valid, signed *old* firmware (v1.0) that had a known vulnerability.
*   **Fix:** Firmware header contains a monotonic version counter. Bootloader rejects any image where `New_Ver < Current_Ver`.

---

## 5. The Update Cycle (State Machine)

1.  **Idle:** Normal operation.
2.  **Downloading:** Storing chunks to flash.
3.  **Verifying:** CRC/Hash check.
4.  **Pending Reboot:** Waiting for user or safe time.
5.  **Testing (Trial Boot):** The most critical phase.
    *   After update, the bootloader marks the state as "Testing".
    *   The App *must* call a function `confirm_update_success()` after running successfully for 2 minutes (and connecting to cloud).
    *   If the Watchdog resets the device *before* confirmation, the Bootloader assumes a crash loop and reverts to the old bank.

---

## 6. Tools & Libraries
*   **MCUBoot:** Open source secure bootloader (Zephyr, generic).
*   **ESP-IDF OTA:** Native implementation for ESP32.
*   **Mender / Balena:** Full fleet management platforms for Linux-based embedded (Raspberry Pi, i.MX).
