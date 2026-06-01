# Over-the-Air (OTA) Updates

## 1. Introduction to OTA Updates

Over-the-Air (OTA) updates are the process of remotely delivering new software, firmware, or configuration data to an embedded device or IoT node. A robust OTA system depends on the [bootloader](bootloaders.md) for partition management, the [watchdog timer](watchdog.md) for trial-boot rollback, [secure boot](secure_boot.md) for chain-of-trust, and [flash filesystem](flash_filesystems.md) layout for partition storage. The delivery pipeline mirrors concepts from [CI/CD](../devops/cicd.md). In the modern era of connected devices, an OTA mechanism is not an optional feature; it is a critical requirement for the entire product lifecycle.

### Why OTA is Mandatory
1.  **Security Vulnerability Patching:** IoT devices are constantly under attack. When a vulnerability in a networking stack (like a TCP/IP or TLS bug) is discovered, it must be patched across the entire fleet immediately. Without OTA, the device remains forever vulnerable.
2.  **Feature Expansion:** Product requirements evolve. OTA allows companies to ship a Minimum Viable Product (MVP) and add features over time, generating ongoing value for the customer.
3.  **Bug Fixes:** No firmware is perfect upon release. Software bugs that cause crashes, memory leaks, or sensor inaccuracies can be silently fixed without user intervention.
4.  **Cost Reduction:** The alternative to OTA is a physical product recall or sending a technician to the site ("truck roll"), which can cost hundreds of dollars per device.

### The Core Challenge: Avoiding the "Brick"
Updating a microcontroller or Linux SoC in the field is inherently dangerous. 
*   What happens if the power fails halfway through erasing the flash memory?
*   What happens if the network connection drops during the download?
*   What happens if the new firmware has a critical bug (like an infinite loop or a hard fault) immediately upon boot?

If any of these scenarios occur and the device cannot recover automatically, it becomes completely non-functional—a "brick." The golden rule of OTA architecture is: **The device must never, under any circumstances, become unbootable or permanently disconnected from the cloud due to a failed update.**

---

## 2. Bootloaders: The Foundation of OTA

The bootloader is a small, specialized piece of software that runs immediately when the microcontroller powers on or resets. Its primary jobs are to initialize basic hardware (clocks, memory controllers), verify the integrity of the main application firmware, and then jump to that application.

In an OTA-enabled system, the bootloader takes on the massive responsibility of managing the update process.

### Primary vs. Secondary Bootloaders
In complex systems, the boot process is often staged:
1.  **Primary Bootloader (ROM Bootloader):** Burned into silicon by the chip manufacturer (e.g., STMicroelectronics, Espressif). It is immutable. It usually checks specific GPIO pins on boot to decide whether to load code from internal Flash, USB, or UART.
2.  **Secondary Bootloader (The OTA Bootloader):** Written by the device developer or an open-source project (like MCUboot). It sits at the very beginning of the writable Flash memory. This is the software that understands partitions, cryptography, and OTA state machines.

### Bootloader Responsibilities in OTA
1.  **Image Verification:** Before executing any application code, the bootloader must calculate a cryptographic hash of the firmware image and verify its digital signature to ensure it is authentic and uncorrupted.
2.  **Partition Management:** The bootloader decides which memory partition contains the "active" firmware and maps that partition into the CPU's execution space.
3.  **Applying Updates:** If a new firmware image has been downloaded to a temporary location, the bootloader is responsible for safely copying or swapping it into the active execution slot.
4.  **Watchdog and Rollback:** If the bootloader detects that the active firmware is continuously crashing, it must automatically roll back to the previously known good firmware.

---

## 3. Partitioning Strategies

To safely update a device, the non-volatile memory (Flash) must be logically divided into distinct regions called partitions. The layout of these partitions dictates the update strategy.

### 3.1. Dual Bank (A/B) Partitioning (The Industry Standard)
This is the most robust and common approach for devices with sufficient memory. The flash memory is divided into two identical, executable slots: Bank A and Bank B.

*   **State 1:** The device is running firmware v1.0 from **Bank A**.
*   **Download:** The device downloads firmware v1.1 and writes it directly into **Bank B** in the background. The device continues to operate normally during this entire process.
*   **Verification:** Once downloaded, the application verifies the cryptographic signature of Bank B.
*   **Switch:** The application writes a flag to a dedicated, small persistent storage area (e.g., an EEPROM, a specific flash sector, or an RTC backup register) indicating: `BOOT_FROM = BANK_B`.
*   **Reboot:** The device reboots.
*   **Bootloader Action:** The bootloader reads the flag, verifies Bank B's integrity one last time, and then jumps to the entry point of Bank B.

**Pros:** 
*   Zero downtime during the download phase. 
*   Instant, safe rollback (if Bank B fails, the bootloader just switches the flag back to Bank A).
*   Power-fail safe at all times.

**Cons:** 
*   Requires exactly 2x the flash memory of the application size, which increases hardware costs.

### 3.2. Single Bank with External Swap
Used on highly constrained microcontrollers where the internal executable Flash is too small to hold two copies of the firmware, but cheap external SPI Flash is available.

*   **State 1:** Firmware runs from Internal Flash.
*   **Download:** The new firmware is downloaded to the External SPI Flash (which cannot be executed from directly).
*   **Reboot:** The device reboots into the Secondary Bootloader.
*   **Swap Phase (Danger Zone):** The Bootloader erases the Internal Flash and copies the new firmware from the External SPI Flash into the Internal Flash.
*   **Boot:** The Bootloader jumps to the newly copied application.

**Pros:** Saves internal MCU flash space.
**Cons:** The "Swap Phase" is dangerous. If power fails while the bootloader is erasing the Internal Flash, the device has no valid application. The bootloader must be extremely robust to detect this partial state on the next boot and resume the copy from the External Flash.

### 3.3. Advanced Swap (Scratch Partition)
Used by bootloaders like MCUboot. It involves a primary slot (executable), a secondary slot (storage), and a tiny "scratch" sector.

Instead of just erasing the primary slot, the bootloader uses the scratch sector to swap the images block by block. 
1. Copy Block 1 of Primary to Scratch.
2. Copy Block 1 of Secondary to Primary.
3. Copy Scratch to Secondary.
*(Repeat for all blocks)*

If a power failure occurs, the bootloader analyzes the state of the scratch partition to figure out exactly which block it was swapping and resumes the process. This ensures that the old firmware is never destroyed; it simply ends up in the secondary slot, allowing for a safe rollback.

### 3.4. Golden Image / Recovery Partition
For critical infrastructure, devices may have three partitions:
1.  **Active Partition:** Normal firmware.
2.  **Update Partition:** Where new firmware is downloaded.
3.  **Golden / Recovery Partition:** A factory-programmed, highly minimal firmware whose *only* job is to connect to the network and download an update. It is never overwritten. If the active partition gets totally corrupted, the bootloader falls back to the Golden Image.

---

## 4. Security: The Chain of Trust

If an attacker can push their own firmware to your devices, they own your entire fleet. They can turn IoT devices into a botnet, steal user data, or permanently brick the hardware. OTA security relies on cryptography to establish a **Chain of Trust**.

### 4.1. Cryptographic Hashing (Integrity)
When downloading a multi-megabyte file over a noisy network (like Cellular or Wi-Fi), bits will flip. The device must verify the integrity of the file.
*   The build server calculates a SHA-256 hash of the compiled firmware binary.
*   The device calculates the SHA-256 hash of the downloaded file.
*   If they match, the file was not corrupted in transit. *However, a hash alone provides no security against a malicious attacker, who can simply calculate a new hash for their malicious file.*

### 4.2. Asymmetric Cryptography (Authenticity)
To prove the file came from the true manufacturer, Digital Signatures are used (typically RSA-2048 or ECDSA secp256r1).

**The Build Server (Private Key):**
1.  Compiles the firmware.
2.  Calculates the SHA-256 hash.
3.  Encrypts that hash using a closely guarded **Private Key**. This encrypted hash is the **Signature**.
4.  Appends the Signature to the firmware binary.

**The Device (Public Key):**
1.  Has the corresponding **Public Key** embedded in its bootloader or burned into hardware eFuses.
2.  Downloads the firmware and the Signature.
3.  Calculates its own SHA-256 hash of the firmware.
4.  Decrypts the Signature using the Public Key to reveal the server's hash.
5.  If `Calculated_Hash == Decrypted_Hash`, the device guarantees two things:
    *   The file is not corrupted (Integrity).
    *   The file could *only* have been created by someone possessing the Private Key (Authenticity).

### 4.3. Hardware Root of Trust and Secure Boot
If the bootloader itself can be modified by an attacker, the entire signature check can be bypassed.
Modern microcontrollers utilize **Secure Boot**.
1.  The manufacturer's Public Key is burned into One-Time-Programmable (OTP) memory (eFuses) inside the silicon during factory production.
2.  When the chip powers on, the immutable ROM bootloader reads the eFuse Public Key and uses it to verify the signature of the Secondary Bootloader.
3.  If valid, it runs the Secondary Bootloader.
4.  The Secondary Bootloader uses its own embedded Public Key to verify the Application firmware.

This creates an unbreakable chain from the silicon itself up to the application.

### 4.4. Anti-Rollback Protection
Imagine a scenario where firmware v1.0 had a critical buffer overflow vulnerability. The manufacturer releases v1.1 to fix it. The devices update successfully.

An attacker intercepts the network traffic and pushes the old, validly signed v1.0 firmware back to the devices. The devices accept it because the signature is mathematically valid. The attacker then exploits the known vulnerability. This is a **Rollback Attack**.

**The Fix: Hardware Monotonic Counters.**
1.  The firmware header contains a Security Version Number (e.g., SVN = 2).
2.  The microcontroller has a hardware-backed monotonic counter (eFuses that can only be blown from 0 to 1, but never reset from 1 to 0).
3.  When the bootloader installs v1.1 (SVN=2), it blows an eFuse so the hardware counter equals 2.
4.  If the attacker tries to flash v1.0 (SVN=1), the bootloader checks the hardware counter. Because `1 < 2`, the bootloader immediately rejects the image, even if the cryptographic signature is perfectly valid.

---

## 5. The OTA State Machine and Trial Boots

The most dangerous moment in an OTA update is the first boot of the new firmware. It might pass all cryptographic checks, but contain a logical bug (e.g., it connects to the wrong Wi-Fi SSID, or a NULL pointer dereference causes an instant hard fault).

If the device boots this bad firmware and stays on it, it will never connect to the cloud again to receive a fix. It is bricked.

To prevent this, OTA relies on a "Trial Boot" or "Test" state machine.

### The Standard OTA State Machine

1.  **State: IDLE / ACTIVE:** The device is running normally from Bank A.
2.  **State: DOWNLOADING:** A new update is detected. The device downloads chunks and writes them to Bank B.
3.  **State: VERIFIED:** The download finishes. The application verifies the signature of Bank B.
4.  **State: PENDING_REBOOT:** The application sets a flag (`STATE = TESTING_BANK_B`) in persistent storage and issues a system reset.
5.  **State: TRIAL BOOT (The Danger Zone):**
    *   The Bootloader sees the `TESTING` flag. It boots into Bank B.
    *   At the exact same time, it starts an **Independent Hardware Watchdog Timer (WDT)**.
    *   Firmware v1.1 starts running. It must initialize hardware, connect to the network, and connect to the cloud server.
    *   Once the firmware successfully communicates with the cloud, the application code explicitly calls a function: `ota_commit_update()`.
6.  **State: COMMITTED:** The `ota_commit_update()` function overwrites the persistent flag to `STATE = ACTIVE_BANK_B`. The Watchdog is fed/disabled. The update is permanent.

### The Rollback Scenario (Trial Boot Failure)
What if firmware v1.1 crashes immediately upon boot?
1.  Because it crashes, it never reaches the `ota_commit_update()` function.
2.  Because it crashed, the CPU halts, or a hard fault handler spins infinitely.
3.  The Independent Hardware Watchdog Timer (started by the bootloader) eventually times out (e.g., after 60 seconds).
4.  The Watchdog forces a hard reset of the microcontroller.
5.  The Bootloader wakes up. It reads the persistent flag. It sees `STATE = TESTING_BANK_B`.
6.  The Bootloader realizes that a reboot occurred *while* in the testing phase. This implies a catastrophic crash.
7.  The Bootloader automatically rewrites the flag to `STATE = ACTIVE_BANK_A` and boots the old, known-good firmware. The device is saved.

---

## 6. Transport Protocols and Delivery

How does a 2MB binary actually get from the cloud to the device?

### 6.1. HTTPS (HTTP over TLS)
The most common method for Wi-Fi and Ethernet devices.
*   **Mechanism:** The device makes a standard HTTP `GET` request to a secure URL (e.g., an AWS S3 bucket).
*   **Chunking:** Embedded devices don't have 2MB of RAM to hold the whole file. They use the HTTP `Range` header (e.g., `Range: bytes=0-4095`) to download the file in 4KB chunks. They write one chunk to flash, then request the next.
*   **Security:** TLS prevents Man-in-the-Middle (MitM) attacks from intercepting or modifying the binary in transit.

### 6.2. MQTT
Commonly used in highly constrained IoT systems where the device is already using MQTT for telemetry.
*   **Mechanism:** The server publishes the binary data as payloads to a specific MQTT topic.
*   **Pros:** Keeps the device architecture simple (only one networking stack required).
*   **Cons:** MQTT is not optimized for large file transfers. Handing massive binaries through an MQTT broker can cause memory pressure on the broker.

### 6.3. BLE (Bluetooth Low Energy)
Used for wearables, smart home sensors, and devices without direct internet access.
*   **Mechanism:** A smartphone app downloads the firmware from the internet. The app then connects to the device via BLE and streams the binary over a custom GATT characteristic.
*   **Challenges:** BLE throughput is incredibly slow (often 10-50 kbps). A 1MB update can take several minutes, during which the user must keep their phone close to the device.

### 6.4. CoAP / LwM2M
Used in cellular IoT (NB-IoT, LTE-M) where bandwidth is expensive and UDP is preferred over TCP.
*   Lightweight M2M (LwM2M) has dedicated, standardized objects precisely for managing firmware updates over UDP.

---

## 7. Differential Updates (Delta OTA)

For devices on cellular networks (where data costs money per megabyte) or devices with very slow connections (BLE, LoRaWAN), downloading a full 2MB firmware image for a tiny bug fix is unacceptable.

**Delta OTA** solves this by only sending the *differences* between the old firmware and the new firmware.

### The Delta Process
1.  **Server Side:** The cloud server takes Firmware v1.0 and Firmware v1.1 and runs a diffing algorithm (like `bsdiff`, `courgette`, or `JojoDiff`). This generates a small "Patch" file.
    *   *Example:* A 2MB firmware might only have a 15KB patch file if only a few lines of code changed.
2.  **Delivery:** The server sends the 15KB patch to the device.
3.  **Device Side:** The device contains a patching engine in its bootloader or application.
4.  **Application:** The patching algorithm reads the old firmware (v1.0) block by block from Bank A, applies the mathematical transformations defined in the 15KB patch file, and writes the resulting new firmware (v1.1) into Bank B.

### Challenges of Delta Updates
*   **RAM/CPU Constraints:** Algorithms like `bsdiff` are designed for desktop computers and require massive amounts of RAM to apply a patch. Embedded systems require specialized, streaming patch algorithms that operate with just a few kilobytes of RAM.
*   **Code Shifting:** If you add one line of C code at the top of your program, the compiler will shift the memory address of *every single function* below it. A standard binary diff will see this as a 100% completely different file, resulting in a massive patch. Modern embedded delta algorithms (like Courgette) disassemble the binary, abstract the memory addresses, diff the underlying logic, and then re-assemble, resulting in tiny patches even when memory shifts.

---

## 8. Real-World Implementations

### 8.1. ESP-IDF OTA (Espressif ESP32)
Espressif provides one of the most robust, out-of-the-box OTA frameworks in the industry for the ESP32.
*   **Partition Table:** Defined via a CSV file. Typically includes a `factory` app partition, `ota_0`, `ota_1`, and an `otadata` partition.
*   **Mechanism:** The `esp_https_ota` component manages the entire process. It connects to the URL, reads the image header, finds the passive OTA slot, streams the chunks directly to flash, and verifies the SHA-256 and ECDSA signatures.
*   **State Machine:** Uses the `otadata` partition (which is just two small flash sectors) to maintain the state (`ESP_OTA_IMG_NEW`, `ESP_OTA_IMG_PENDING_VERIFY`, `ESP_OTA_IMG_VALID`). The hardware RTC watchdog is deeply integrated for the trial boot rollback.

### 8.2. MCUboot (Zephyr / RTOS)
MCUboot is a highly secure, OS-agnostic open-source bootloader originally developed for the Apache Mynewt project and now heavily used by Zephyr RTOS, nRF Connect SDK, and others.
*   **Architecture:** It relies heavily on image signing (RSA/ECDSA/ed25519) and supports the "Scratch" partition swap mechanism to allow updates on devices with only a single executable flash bank.
*   **Features:** It supports multiple independent executable images (e.g., updating a main CPU and a dedicated network co-processor simultaneously), hardware cryptographic offloading, and strict anti-rollback counters.

### 8.3. Linux OTA (Mender, SWUpdate, RAUC)
For embedded Linux devices (Raspberry Pi, i.MX6, BeagleBone), updating single files via `apt-get` is extremely dangerous. If power fails during an `apt upgrade`, the filesystem will be corrupted and the Linux kernel will kernel panic on boot.

Embedded Linux uses **Image-Based OTA**.
*   The eMMC or SD Card is partitioned with two identical Root Filesystems (RootFS A and RootFS B).
*   Tools like **Mender** or **RAUC** run as daemons in Linux.
*   They download a complete, compressed, read-only `ext4` or `squashfs` filesystem image.
*   They `dd` this image block-by-block directly into the passive RootFS partition.
*   They modify the U-Boot bootloader environment variables to switch the active boot partition.
*   They integrate with systemd to perform health checks on the next boot before committing the update in U-Boot.

---

## 9. Best Practices for Designing OTA Systems

If you are engineering a connected device, adhere to these rules:

1.  **OTA is Feature #1:** Do not write a single line of application code until your bootloader, partitioning, and OTA update mechanisms are fully implemented and tested. It is the foundation of the product.
2.  **Test the Failure Paths:** Don't just test successful updates. Write automated scripts that cut the power to the device randomly during the download phase, the flash write phase, and the trial boot phase. The device *must* recover every time.
3.  **Use Hardware Watchdogs:** Never rely on a software watchdog. If the CPU hard faults or deadlocks, software watchdogs stop running. The watchdog must be an independent hardware timer that physically pulls the reset pin if it isn't fed.
4.  **Version Everything:** The hardware revision, the bootloader version, and the application version must be strictly tracked. Ensure an update meant for Hardware Rev B cannot be installed on Hardware Rev A.
5.  **Bandwidth Management:** Randomize the time devices check for updates. If you have 100,000 devices and push an update, and they all make an HTTPS GET request to your server at the exact same second, you will accidentally DDoS your own infrastructure.
6.  **Secure Your Keys:** The Private Key used to sign firmware is the keys to the kingdom. It should never touch a developer's laptop. It should be generated and stored inside a Hardware Security Module (HSM) on AWS or Azure, and the CI/CD pipeline should only be able to request signatures from the HSM.

## 10. Deep Dive: The MCUboot Trailer and Swap Mechanism

To truly understand how a safe "Swap" partition works on constrained devices (where you don't have enough space for two full executable banks), we must examine the **MCUboot Trailer**.

When MCUboot performs a swap, it doesn't just blindly copy memory. It manages state across power failures using a cryptographic trailer appended to the end of every image slot.

### The Trailer Structure
The end of the image slot contains:
1.  **Swap Status:** A record of exactly which sectors have been swapped so far. If power fails, MCUboot reads this to resume the operation.
2.  **Copy Done Flag:** A single byte that indicates if the image copy was completely successful.
3.  **Image OK Flag:** A single byte that indicates if the image has been confirmed as working (the Trial Boot succeeded).
4.  **MAGIC Number:** A constant byte sequence (`0x96f3b83d...`) that tells the bootloader a valid trailer exists.

### The Swap State Machine
1.  **Testing State:** The user downloads an update to the Secondary Slot. A script writes the MAGIC number to the Secondary Slot trailer. On reboot, MCUboot sees the MAGIC number, sees that `Image OK` is NOT set, and begins swapping the Primary and Secondary slots block by block. It then boots the Primary slot.
2.  **Trial Boot:** The new application runs. If it works, it writes the `Image OK` flag to the Primary Slot trailer.
3.  **Revert State:** If the application crashes and the watchdog reboots the device, MCUboot checks the Primary Slot trailer. It sees the MAGIC number, but `Image OK` is STILL unset. It concludes the update failed, and it automatically swaps the Primary and Secondary slots back, restoring the original firmware.
4.  **Permanent State:** If the update was confirmed (`Image OK` is set), MCUboot considers the update permanent.

## 11. Deep Dive: ESP-IDF OTA API and Configuration

Espressif's ESP-IDF provides a high-level API for OTA.

### 11.1. The Partition Table (`partitions.csv`)
You must define the flash layout explicitly.
```csv
# Name,   Type, SubType, Offset,  Size, Flags
nvs,      data, nvs,     0x9000,  0x4000,
otadata,  data, ota,     0xd000,  0x2000,
phy_init, data, phy,     0xf000,  0x1000,
factory,  app,  factory, 0x10000, 1M,
ota_0,    app,  ota_0,   ,        1M,
ota_1,    app,  ota_1,   ,        1M,
```
*   `otadata`: Two small sectors that the ESP-IDF bootloader uses to track whether `ota_0` or `ota_1` is the active partition, and the state of the trial boot.

### 11.2. The C API
A minimal implementation using the raw OTA APIs:
```c
#include "esp_ota_ops.h"

void do_ota_update(const uint8_t *image_data, size_t image_size) {
    esp_ota_handle_t update_handle = 0;
    const esp_partition_t *update_partition = esp_ota_get_next_update_partition(NULL);

    // 1. Begin OTA: Erases the passive partition
    esp_ota_begin(update_partition, OTA_WITH_SEQUENTIAL_WRITES, &update_handle);

    // 2. Write Data (usually in a loop as chunks arrive over Wi-Fi)
    esp_ota_write(update_handle, image_data, image_size);

    // 3. End OTA: Verifies SHA-256 and ECDSA signature
    if (esp_ota_end(update_handle) == ESP_OK) {
        // 4. Set boot partition: Modifies the otadata sector
        esp_ota_set_boot_partition(update_partition);
        
        // 5. Restart into the trial boot
        esp_restart();
    }
}
```

## 12. Data Migration During OTA (NVS/EEPROM)

Updating the code is only half the battle. Often, firmware v1.1 requires a different configuration structure in Non-Volatile Storage (NVS) than v1.0.

*Example:* v1.0 stored a `struct` with Wi-Fi credentials. v1.1 adds an MQTT broker IP address to that `struct`.

If v1.1 boots and blindly casts the old NVS memory to the new `struct`, it will read garbage data, potentially crashing the device or causing a network disconnect.

### Migration Strategies
1.  **Schema Versioning:** Every data blob stored in NVS must include a schema version integer.
    ```c
    typedef struct {
        uint8_t schema_version; // Must be 1
        char ssid[32];
        char pass[64];
    } config_v1_t;
    
    typedef struct {
        uint8_t schema_version; // Must be 2
        char ssid[32];
        char pass[64];
        char mqtt_ip[16];
    } config_v2_t;
    ```
2.  **Upgrading:** On boot, the application reads the schema version. If it is `1`, it reads the data into `config_v1_t`, maps those fields into `config_v2_t`, sets defaults for the new fields, writes the new `config_v2_t` back to NVS, and increments the version to `2`.
3.  **Downgrading (Rollback Danger):** If the trial boot fails and the bootloader reverts to v1.0, v1.0 will wake up and see schema version `2`. It doesn't know how to read version `2`.
    *   *Solution:* Always write new configurations to a *new* NVS key/file. Only delete the old v1.0 configuration after the v1.1 update has been permanently committed.

## 13. Network Resilience and Error Handling

Embedded devices often operate on the edge of connectivity (weak Wi-Fi, low-signal Cellular). The OTA engine must be highly resilient.

*   **HTTP Range Requests:** If a 2MB download fails at 1.9MB, starting over wastes bandwidth and battery. The device should request the remaining bytes using the HTTP header: `Range: bytes=1900000-2000000`.
*   **Chunk-Level Hashing:** Instead of just hashing the entire 2MB file at the end, the server can provide a manifest with a hash for every 4KB chunk. The device verifies each chunk as it arrives. If a chunk is corrupted, it re-requests just that chunk.
*   **Battery Level Checks:** Never start an OTA if the battery is below 20%. The high current draw of erasing and writing to Flash memory causes the battery voltage to dip. If it drops below the Brown-Out Detect (BOD) threshold, the device will hard reset mid-update.

## 14. OTA in the Automotive Industry (UNECE WP.29)

Automotive OTA is heavily regulated. The UN Economic Commission for Europe (UNECE) WP.29 regulation enforces strict cybersecurity and software update management systems (CSMS and SUMS).

### Automotive OTA Architecture
A modern car contains over 100 ECUs (Electronic Control Units).
1.  **The OTA Gateway:** Usually the Telematics Control Unit (TCU). It acts as the central orchestrator. It connects to the OEM cloud over LTE/5G.
2.  **The In-Vehicle Network:** The Gateway distributes the firmware updates over internal networks like Automotive Ethernet, CAN-FD, or LIN to the target ECUs.
3.  **UDS over DoIP:** Updates are typically flashed using Unified Diagnostic Services (UDS) over Diagnostics over Internet Protocol (DoIP).
4.  **Campaign Management:** Cars can only be updated when in a "Safe State" (engine off, parking brake engaged, doors locked). The OTA Campaign Manager orchestrates this state machine.
5.  **A/B Clusters:** Unlike IoT devices, cars often A/B update entire clusters of ECUs simultaneously to ensure strict compatibility between software versions (e.g., the Engine Control Unit and the Transmission Control Unit must run paired software versions).

## 15. Factory Provisioning: Getting the Keys on the Device

An OTA system is only secure if the cryptographic keys are securely injected into the device during manufacturing.

1.  **The Hardware Security Module (HSM):** On the factory floor, a highly secure server (HSM) generates a unique Private/Public keypair (or Device Certificate) for every single microcontroller.
2.  **JTAG/SWD Injection:** Over a physical programming cable, the factory flasher writes the primary bootloader, the initial application, and the Public Key into the device.
3.  **Blowing the eFuses:** The factory flasher sends a command to blow the hardware eFuses on the microcontroller.
    *   This permanently write-protects the sector containing the Public Key.
    *   It completely disables the JTAG/SWD debugging interface, preventing physical attackers from reading out the firmware or injecting malicious code later.
    *   It enables the "Secure Boot" hardware feature.
4.  **Cloud Registration:** The factory HSM securely transmits the device's unique ID and Public Certificate to the OEM's cloud database, allowing the cloud to recognize and trust the device when it connects for its first OTA update.

## Where this connects

- [Bootloaders](bootloaders.md) — the bootloader manages partition selection and applies the update during reboot
- [Watchdog](watchdog.md) — the trial-boot state machine relies entirely on the hardware watchdog for rollback safety
- [Secure Boot](secure_boot.md) — chain-of-trust from ROM bootloader through application prevents unsigned firmware
- [Flash Filesystems](flash_filesystems.md) — NVS / EEPROM stores OTA state flags across reboots
- [CI/CD](../devops/cicd.md) — the firmware build, sign, and release pipeline feeds the OTA update server
