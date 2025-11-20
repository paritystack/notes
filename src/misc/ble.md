# Bluetooth Low Energy (BLE)

Bluetooth Low Energy (BLE), also known as Bluetooth Smart or Bluetooth 4.0+, is a wireless personal area network technology designed for short-range communication with significantly reduced power consumption compared to Classic Bluetooth. BLE is optimized for applications requiring periodic or burst data transfers, making it ideal for IoT devices, wearables, health monitors, beacons, and smart home applications.

## Table of Contents

- [Overview](#overview)
- [BLE vs Classic Bluetooth](#ble-vs-classic-bluetooth)
- [BLE Protocol Architecture](#ble-protocol-architecture)
- [Core Concepts](#core-concepts)
- [Linux/BlueZ Implementation](#linuxbluez-implementation)
- [Development & Programming](#development--programming)
- [Security & Pairing](#security--pairing)
- [Practical Examples](#practical-examples)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

BLE was introduced in the Bluetooth 4.0 specification in 2010. Unlike Classic Bluetooth, which is designed for continuous streaming applications like audio, BLE focuses on:

- **Ultra-low power consumption**: Devices can run for months or years on coin cell batteries
- **Fast connections**: Connection setup in milliseconds
- **Small data transfers**: Optimized for periodic small bursts of data
- **Simple architecture**: Reduced complexity for easier implementation
- **Wide platform support**: Native support on iOS, Android, Windows, Linux, and macOS

**Common Use Cases:**
- Fitness trackers and health monitors
- Smart watches and wearables
- Proximity sensors and beacons
- Smart home devices (lights, locks, thermostats)
- Asset tracking and location services
- Wireless sensors (temperature, humidity, motion)
- Keyboard, mice, and game controllers

---

## BLE vs Classic Bluetooth

| Feature | Classic Bluetooth (BR/EDR) | Bluetooth Low Energy (BLE) |
|---------|---------------------------|----------------------------|
| **Power Consumption** | Higher (continuous use) | Very low (intermittent use) |
| **Data Rate** | 1-3 Mbps | 125 Kbps - 2 Mbps |
| **Range** | ~10-100m | ~50-150m (up to 400m with BLE 5.0) |
| **Connection Time** | ~6 seconds | ~6 milliseconds |
| **Voice Capable** | Yes | No (until LE Audio in BT 5.2) |
| **Network Topology** | Point-to-point, piconet | Star, mesh, broadcast |
| **Primary Use** | Audio streaming, file transfer | Periodic data, sensors, IoT |
| **Security** | Secure Simple Pairing | LE Secure Connections |
| **Protocol Stack** | Complex, many profiles | Simplified, GATT-based |

**Key Architectural Differences:**
- BLE uses a simplified protocol stack
- Different radio modulation (Classic uses FHSS, BLE uses simpler frequency hopping)
- BLE devices can advertise their presence without pairing
- BLE supports connectionless data broadcast (advertising)
- Different profiles and services (Classic uses Bluetooth profiles, BLE uses GATT services)

---

## BLE Protocol Architecture

### Protocol Stack Layers

The BLE protocol stack consists of several layers, from physical radio to application:

```
┌─────────────────────────────────┐
│   Application Layer             │
├─────────────────────────────────┤
│   GAP (Generic Access Profile)  │
│   GATT (Generic Attribute)      │
├─────────────────────────────────┤
│   ATT (Attribute Protocol)      │
├─────────────────────────────────┤
│   L2CAP (Logical Link Control)  │
├─────────────────────────────────┤
│   HCI (Host Controller Interface│
├─────────────────────────────────┤
│   Link Layer                    │
├─────────────────────────────────┤
│   Physical Layer (PHY)          │
└─────────────────────────────────┘
```

#### 1. Physical Layer (PHY)

- **Frequency Band**: 2.4 GHz ISM band (2400-2483.5 MHz)
- **Channels**: 40 channels, each 2 MHz wide
  - 3 advertising channels (37, 38, 39)
  - 37 data channels (0-36)
- **Modulation**: GFSK (Gaussian Frequency Shift Keying)
- **Data Rates**:
  - BLE 4.x: 1 Mbps
  - BLE 5.0: 1 Mbps, 2 Mbps, 125 Kbps, 500 Kbps (coded PHY)

#### 2. Link Layer

Responsible for:
- Advertising and scanning
- Connection establishment and maintenance
- Channel hopping
- Packet acknowledgment and retransmission
- Encryption at the link level

**Link Layer States:**
```
Standby → Advertising → Connected
       ↓             ↗
       → Scanning →
       ↓
       → Initiating → Connected
```

#### 3. HCI (Host Controller Interface)

- Standardized interface between host (CPU running application) and controller (radio chip)
- Allows different host/controller combinations
- Communication via UART, USB, SPI, or shared memory
- Commonly used for debugging and low-level access

#### 4. L2CAP (Logical Link Control and Adaptation Protocol)

- Protocol multiplexing
- Packet segmentation and reassembly
- Flow control
- In BLE, provides Connection-Oriented Channels and credit-based flow control (BLE 4.1+)

#### 5. ATT (Attribute Protocol)

- Defines how data is organized and exchanged
- Client-server architecture
- **Attributes** are the fundamental data entities:
  - **Handle**: 16-bit unique identifier
  - **Type**: UUID defining the attribute type
  - **Value**: The actual data
  - **Permissions**: Read, write, notify, etc.

**ATT Operations:**
- **Read**: Client reads attribute value from server
- **Write**: Client writes value to server
- **Notify**: Server pushes data to client (no acknowledgment)
- **Indicate**: Server pushes data to client (with acknowledgment)

#### 6. GATT (Generic Attribute Profile)

Built on top of ATT, GATT defines the structure for organizing attributes:

```
Profile
  └── Service (UUID)
       ├── Characteristic (UUID)
       │    ├── Value
       │    └── Descriptors
       │         ├── Client Characteristic Configuration (0x2902)
       │         ├── Characteristic User Description (0x2901)
       │         └── ...
       └── Characteristic (UUID)
            └── ...
```

**Key Concepts:**
- **Service**: Collection of related characteristics (e.g., Heart Rate Service)
- **Characteristic**: Single data point with properties and value (e.g., Heart Rate Measurement)
- **Descriptor**: Metadata about a characteristic
- **UUID**:
  - 16-bit for Bluetooth SIG defined services/characteristics
  - 128-bit for custom/vendor-specific services

**Common Services:**
- Heart Rate Service (0x180D)
- Battery Service (0x180F)
- Device Information Service (0x180A)
- Nordic UART Service (custom)

#### 7. GAP (Generic Access Profile)

Defines device roles, modes, and procedures for:
- Device discovery
- Connection establishment
- Security
- Privacy

**GAP Roles:**
- **Broadcaster**: Only advertises (e.g., beacon)
- **Observer**: Only scans, doesn't connect
- **Peripheral**: Advertises and accepts connections (e.g., fitness tracker)
- **Central**: Scans and initiates connections (e.g., smartphone)

**GAP Modes:**
- **Discoverable**: Device can be discovered by others
- **Connectable**: Device accepts connection requests
- **Bondable**: Device can pair and bond

---

## Core Concepts

### Advertising

Advertising allows devices to broadcast their presence and data without establishing a connection.

**Advertising Packet Structure:**
- **Header**: PDU type, flags
- **MAC Address**: Device identifier
- **Payload**: Up to 31 bytes of data (extended advertising in BLE 5.0+ allows up to 255 bytes)

**Advertising Types:**
1. **ADV_IND**: Connectable and scannable undirected advertising
2. **ADV_DIRECT_IND**: Connectable directed advertising (fast reconnection)
3. **ADV_NONCONN_IND**: Non-connectable undirected advertising (beacons)
4. **ADV_SCAN_IND**: Scannable undirected advertising

**Advertising Interval:**
- Range: 20ms to 10.24 seconds
- Shorter interval = faster discovery but higher power consumption
- Recommended: 100ms - 1 second for balance

**Advertising Data Format:**
```
[Length][Type][Data][Length][Type][Data]...
```

Common AD Types:
- 0x01: Flags
- 0x02/0x03: Incomplete/Complete list of 16-bit UUIDs
- 0x09: Complete Local Name
- 0xFF: Manufacturer Specific Data

**Example Advertising Data:**
```
02 01 06  // Flags: General Discoverable, BR/EDR not supported
09 09 4D 79 44 65 76 69 63 65  // Complete Local Name: "MyDevice"
03 03 0F 18  // Complete list of 16-bit UUIDs: 0x180F (Battery Service)
```

### Scanning

Scanning is the process of listening for advertising packets.

**Scan Types:**
1. **Passive Scanning**: Just listens to advertising packets
2. **Active Scanning**: Sends scan requests to get additional scan response data

**Scan Parameters:**
- **Scan Interval**: How often to scan (e.g., every 100ms)
- **Scan Window**: How long to scan during each interval (e.g., 50ms)
- **Duty Cycle**: Scan Window / Scan Interval (e.g., 50%)

### Connections

Once a central discovers a peripheral, it can initiate a connection.

**Connection Parameters:**
- **Connection Interval**: Time between connection events (7.5ms - 4s)
  - Shorter = lower latency, higher power
  - Longer = higher latency, lower power
- **Slave Latency**: Number of events peripheral can skip (0-499)
  - Allows peripheral to sleep to save power
- **Supervision Timeout**: Max time before connection is considered lost (100ms - 32s)

**Connection Process:**
```
Central                    Peripheral
  |                             |
  |------ SCAN_REQ ----------->|
  |<----- SCAN_RSP ------------|
  |                             |
  |------ CONNECT_REQ -------->|
  |<----- Connection Event --->|
  |<----- Connection Event --->|
```

**MTU (Maximum Transmission Unit):**
- Minimum: 23 bytes (default in BLE 4.0/4.1)
- Negotiable up to 512 bytes (BLE 4.2+)
- Larger MTU = more efficient data transfer
- Must be negotiated after connection

### Data Transfer

**Methods:**
1. **Read**: Client requests data from server
2. **Write**: Client sends data to server
   - **Write Command**: No response, faster
   - **Write Request**: With response, reliable
3. **Notify**: Server pushes data to client (no ack)
4. **Indicate**: Server pushes data to client (with ack)

**Throughput Considerations:**
```
Theoretical Max = (MTU - 3) / Connection_Interval
Practical: 5-20 kB/s typical, up to 100+ kB/s with BLE 5.0 2M PHY
```

---

## Linux/BlueZ Implementation

BlueZ is the official Linux Bluetooth protocol stack, supporting both Classic Bluetooth and BLE.

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install bluez bluez-tools
```

**Fedora/RHEL:**
```bash
sudo dnf install bluez bluez-tools
```

**Arch Linux:**
```bash
sudo pacman -S bluez bluez-utils
```

### BlueZ Architecture

```
┌─────────────────────────────────┐
│   Applications (gatttool, etc)  │
├─────────────────────────────────┤
│   D-Bus API                     │
├─────────────────────────────────┤
│   bluetoothd (daemon)           │
├─────────────────────────────────┤
│   Kernel Bluetooth Subsystem    │
├─────────────────────────────────┤
│   HCI Driver                    │
├─────────────────────────────────┤
│   Bluetooth Hardware            │
└─────────────────────────────────┘
```

### Essential Tools

#### bluetoothctl

Interactive command-line tool for managing Bluetooth devices.

**Basic Usage:**
```bash
# Start bluetoothctl
bluetoothctl

# Show controller information
[bluetooth]# show

# Power on the controller
[bluetooth]# power on

# Enable scanning
[bluetooth]# scan on

# List discovered devices
[bluetooth]# devices

# Connect to a device
[bluetooth]# connect AA:BB:CC:DD:EE:FF

# Pair with a device
[bluetooth]# pair AA:BB:CC:DD:EE:FF

# Trust a device (auto-connect)
[bluetooth]# trust AA:BB:CC:DD:EE:FF

# Show device info
[bluetooth]# info AA:BB:CC:DD:EE:FF

# Disconnect
[bluetooth]# disconnect AA:BB:CC:DD:EE:FF

# Remove device
[bluetooth]# remove AA:BB:CC:DD:EE:FF
```

**GATT Operations:**
```bash
# List services
[bluetooth]# menu gatt
[bluetooth]# list-attributes AA:BB:CC:DD:EE:FF

# Select a characteristic
[bluetooth]# select-attribute /org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF/service0010/char0011

# Read characteristic
[bluetooth]# read

# Write characteristic
[bluetooth]# write 0x01 0x02 0x03

# Enable notifications
[bluetooth]# notify on
```

#### hcitool

Low-level tool for HCI operations (deprecated but still useful).

```bash
# List Bluetooth controllers
hciconfig

# Scan for BLE devices (requires root)
sudo hcitool lescan

# Scan with RSSI values
sudo hcitool lescan --duplicates

# Get device information
hcitool info AA:BB:CC:DD:EE:FF

# Connection info
hcitool conn
```

#### hciconfig

Configure Bluetooth devices.

```bash
# Show all controllers
hciconfig

# Bring interface up
sudo hciconfig hci0 up

# Bring interface down
sudo hciconfig hci0 down

# Reset device
sudo hciconfig hci0 reset

# Change device name
sudo hciconfig hci0 name "MyDevice"

# Enable/disable advertising
sudo hciconfig hci0 leadv 0  # Disable
sudo hciconfig hci0 leadv 3  # Enable non-connectable advertising
```

#### gatttool

GATT client tool (deprecated in favor of bluetoothctl, but still widely used).

```bash
# Interactive mode
gatttool -b AA:BB:CC:DD:EE:FF -I

# Connect
[AA:BB:CC:DD:EE:FF][LE]> connect

# Discover primary services
[AA:BB:CC:DD:EE:FF][LE]> primary

# Discover all characteristics
[AA:BB:CC:DD:EE:FF][LE]> characteristics

# Read characteristic by handle
[AA:BB:CC:DD:EE:FF][LE]> char-read-hnd 0x0011

# Read characteristic by UUID
[AA:BB:CC:DD:EE:FF][LE]> char-read-uuid 00002a00-0000-1000-8000-00805f9b34fb

# Write characteristic
[AA:BB:CC:DD:EE:FF][LE]> char-write-req 0x0011 0102

# Write without response
[AA:BB:CC:DD:EE:FF][LE]> char-write-cmd 0x0011 0102

# Listen for notifications
[AA:BB:CC:DD:EE:FF][LE]> char-write-req 0x0012 0100  # Enable notifications (write 0x0001 to CCCD)
```

**Non-interactive mode:**
```bash
# Read characteristic
gatttool -b AA:BB:CC:DD:EE:FF --char-read --handle=0x0011

# Write characteristic
gatttool -b AA:BB:CC:DD:EE:FF --char-write-req --handle=0x0011 --value=0102

# Listen for notifications
gatttool -b AA:BB:CC:DD:EE:FF --char-write-req --handle=0x0012 --value=0100 --listen
```

#### btmon

Bluetooth monitor - captures and displays HCI traffic in real-time.

```bash
# Monitor all Bluetooth traffic
sudo btmon

# Save to file
sudo btmon -w capture.btsnoop

# Filter by HCI index
sudo btmon -i hci0
```

**Example Output:**
```
< HCI Command: LE Set Scan Parameters (0x08|0x000b) plen 7
        Type: Passive (0x00)
        Interval: 10.000 msec (0x0010)
        Window: 10.000 msec (0x0010)
        Own address type: Public (0x00)
        Filter policy: Accept all advertisement (0x00)
> HCI Event: Command Complete (0x0e) plen 4
        LE Set Scan Parameters (0x08|0x000b) ncmd 1
        Status: Success (0x00)
```

#### bluetoothd

Main Bluetooth daemon.

```bash
# Check status
sudo systemctl status bluetooth

# Start/stop/restart
sudo systemctl start bluetooth
sudo systemctl stop bluetooth
sudo systemctl restart bluetooth

# Enable at boot
sudo systemctl enable bluetooth

# Run in foreground with debug
sudo bluetoothd -n -d
```

**Configuration File:** `/etc/bluetooth/main.conf`

```ini
[General]
# Device name
Name = MyDevice

# Discoverable timeout (0 = always discoverable)
DiscoverableTimeout = 0

# Pairable timeout (0 = always pairable)
PairableTimeout = 0

# Privacy (rotate MAC address)
Privacy = device

[Policy]
# Auto-enable controllers
AutoEnable = true

[GATT]
# ATT/GATT cache
Cache = always

# Key size for GATT (7-16)
KeySize = 16
```

### D-Bus API

BlueZ exposes its functionality via D-Bus, allowing programmatic access.

**List adapters:**
```bash
dbus-send --system --print-reply --dest=org.bluez / org.freedesktop.DBus.ObjectManager.GetManagedObjects
```

**Start discovery:**
```bash
dbus-send --system --print-reply --dest=org.bluez /org/bluez/hci0 org.bluez.Adapter1.StartDiscovery
```

**Python example using pydbus:**
```python
from pydbus import SystemBus

bus = SystemBus()
adapter = bus.get('org.bluez', '/org/bluez/hci0')

# Start scanning
adapter.StartDiscovery()

# Get properties
props = adapter.GetAll('org.bluez.Adapter1')
print(f"Address: {props['Address']}")
print(f"Name: {props['Name']}")
```

---

## Development & Programming

### Python Development

#### Using Bleak (Recommended)

Bleak is a cross-platform Python BLE library.

**Installation:**
```bash
pip install bleak
```

**Scanning for Devices:**
```python
import asyncio
from bleak import BleakScanner

async def scan():
    devices = await BleakScanner.discover(timeout=5.0)
    for device in devices:
        print(f"{device.address} - {device.name} - RSSI: {device.rssi}")

asyncio.run(scan())
```

**Scanning with Callback:**
```python
import asyncio
from bleak import BleakScanner

def detection_callback(device, advertisement_data):
    print(f"Found: {device.address} - {device.name}")
    print(f"  RSSI: {advertisement_data.rssi}")
    print(f"  Service UUIDs: {advertisement_data.service_uuids}")
    print(f"  Manufacturer Data: {advertisement_data.manufacturer_data}")

async def scan():
    scanner = BleakScanner(detection_callback)
    await scanner.start()
    await asyncio.sleep(5.0)
    await scanner.stop()

asyncio.run(scan())
```

**Connecting and Reading:**
```python
import asyncio
from bleak import BleakClient

DEVICE_ADDRESS = "AA:BB:CC:DD:EE:FF"
CHARACTERISTIC_UUID = "00002a00-0000-1000-8000-00805f9b34fb"  # Device Name

async def connect_and_read():
    async with BleakClient(DEVICE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        # Read characteristic
        value = await client.read_gatt_char(CHARACTERISTIC_UUID)
        print(f"Device Name: {value.decode()}")

        # List all services and characteristics
        for service in client.services:
            print(f"Service: {service.uuid}")
            for char in service.characteristics:
                print(f"  Characteristic: {char.uuid}")
                print(f"    Properties: {char.properties}")

asyncio.run(connect_and_read())
```

**Writing to Characteristic:**
```python
import asyncio
from bleak import BleakClient

DEVICE_ADDRESS = "AA:BB:CC:DD:EE:FF"
CHAR_UUID = "00002a06-0000-1000-8000-00805f9b34fb"

async def write_data():
    async with BleakClient(DEVICE_ADDRESS) as client:
        # Write with response
        await client.write_gatt_char(CHAR_UUID, b"\x01\x02\x03")

        # Write without response (faster)
        await client.write_gatt_char(CHAR_UUID, b"\x01\x02\x03", response=False)

asyncio.run(write_data())
```

**Receiving Notifications:**
```python
import asyncio
from bleak import BleakClient

DEVICE_ADDRESS = "AA:BB:CC:DD:EE:FF"
NOTIFY_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"  # Heart Rate Measurement

def notification_handler(sender, data):
    """Callback for notifications"""
    print(f"Notification from {sender}: {data.hex()}")

async def receive_notifications():
    async with BleakClient(DEVICE_ADDRESS) as client:
        # Start notifications
        await client.start_notify(NOTIFY_CHAR_UUID, notification_handler)

        # Listen for 30 seconds
        await asyncio.sleep(30.0)

        # Stop notifications
        await client.stop_notify(NOTIFY_CHAR_UUID)

asyncio.run(receive_notifications())
```

**Complete Example - Heart Rate Monitor:**
```python
import asyncio
from bleak import BleakClient, BleakScanner

HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

def parse_heart_rate(data):
    """Parse heart rate measurement data"""
    flags = data[0]
    hr_format = flags & 0x01

    if hr_format == 0:
        # uint8
        heart_rate = data[1]
    else:
        # uint16
        heart_rate = int.from_bytes(data[1:3], byteorder='little')

    return heart_rate

def notification_handler(sender, data):
    heart_rate = parse_heart_rate(data)
    print(f"Heart Rate: {heart_rate} bpm")

async def main():
    # Find heart rate monitor
    print("Scanning for heart rate monitors...")
    devices = await BleakScanner.discover(
        timeout=5.0,
        service_uuids=[HEART_RATE_SERVICE_UUID]
    )

    if not devices:
        print("No heart rate monitor found")
        return

    device = devices[0]
    print(f"Connecting to {device.name} ({device.address})...")

    async with BleakClient(device.address) as client:
        print("Connected!")

        # Start receiving heart rate notifications
        await client.start_notify(HEART_RATE_MEASUREMENT_UUID, notification_handler)

        print("Monitoring heart rate... (Press Ctrl+C to stop)")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")

        await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)

asyncio.run(main())
```

#### Creating a GATT Server with Bleak

Note: Bleak primarily focuses on central/client role. For peripheral/server role on Linux, use BlueZ D-Bus API directly.

**Simple GATT Server using D-Bus (Python):**
```python
#!/usr/bin/env python3
import dbus
import dbus.mainloop.glib
from gi.repository import GLib
from dbus.service import Object, method
import array

# Define UUIDs
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

class Characteristic(Object):
    def __init__(self, bus, index, uuid, flags, service):
        self.path = service.path + '/char' + str(index)
        self.uuid = uuid
        self.flags = flags
        self.service = service
        self.value = []

        Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            'org.bluez.GattCharacteristic1': {
                'Service': self.service.path,
                'UUID': self.uuid,
                'Flags': self.flags
            }
        }

    @method('org.bluez.GattCharacteristic1', out_signature='ay')
    def ReadValue(self, options):
        print('ReadValue called')
        return self.value

    @method('org.bluez.GattCharacteristic1', in_signature='ay')
    def WriteValue(self, value, options):
        print(f'WriteValue called: {bytes(value)}')
        self.value = value

class Service(Object):
    def __init__(self, bus, index, uuid, primary):
        self.path = '/org/bluez/example/service' + str(index)
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []

        Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            'org.bluez.GattService1': {
                'UUID': self.uuid,
                'Primary': self.primary,
                'Characteristics': [char.path for char in self.characteristics]
            }
        }

    def add_characteristic(self, char):
        self.characteristics.append(char)

class Application(Object):
    def __init__(self, bus):
        self.path = '/org/bluez/example'
        self.services = []
        Object.__init__(self, bus, self.path)

    @method('org.freedesktop.DBus.ObjectManager', out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}

        for service in self.services:
            response[service.path] = service.get_properties()
            for char in service.characteristics:
                response[char.path] = char.get_properties()

        return response

    def add_service(self, service):
        self.services.append(service)

def main():
    bus = dbus.SystemBus()

    # Create application
    app = Application(bus)

    # Create service
    service = Service(bus, 0, SERVICE_UUID, True)
    app.add_service(service)

    # Create characteristic
    char = Characteristic(bus, 0, CHAR_UUID, ['read', 'write'], service)
    char.value = array.array('B', b'Hello BLE').tolist()
    service.add_characteristic(char)

    # Register application
    adapter = bus.get_object('org.bluez', '/org/bluez/hci0')
    gatt_manager = dbus.Interface(adapter, 'org.bluez.GattManager1')

    gatt_manager.RegisterApplication(app.path, {})
    print('GATT application registered')

    # Start advertising
    # (Advertising setup code omitted for brevity - use BlueZ advertising API)

    mainloop = GLib.MainLoop()
    mainloop.run()

if __name__ == '__main__':
    main()
```

### C/C++ Development

#### Using BlueZ C API

**Basic Scanner (C):**
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/hci.h>
#include <bluetooth/hci_lib.h>

int main(int argc, char **argv) {
    int dev_id, sock, len, flags;
    int i, num_rsp;
    char addr[19] = {0};
    char name[248] = {0};
    inquiry_info *info = NULL;

    // Get default Bluetooth adapter
    dev_id = hci_get_route(NULL);
    if (dev_id < 0) {
        perror("No Bluetooth adapter found");
        exit(1);
    }

    // Open HCI socket
    sock = hci_open_dev(dev_id);
    if (sock < 0) {
        perror("Failed to open HCI socket");
        exit(1);
    }

    // Perform inquiry
    len = 8;  // Inquiry length (1.28 * len seconds)
    num_rsp = 255;  // Max number of responses
    flags = IREQ_CACHE_FLUSH;

    info = (inquiry_info*)malloc(num_rsp * sizeof(inquiry_info));

    printf("Scanning for devices...\n");
    num_rsp = hci_inquiry(dev_id, len, num_rsp, NULL, &info, flags);
    if (num_rsp < 0) {
        perror("Inquiry failed");
        exit(1);
    }

    printf("Found %d device(s)\n", num_rsp);

    // Get device info
    for (i = 0; i < num_rsp; i++) {
        ba2str(&(info + i)->bdaddr, addr);
        memset(name, 0, sizeof(name));

        if (hci_read_remote_name(sock, &(info + i)->bdaddr, sizeof(name), name, 0) < 0)
            strcpy(name, "[unknown]");

        printf("%s  %s\n", addr, name);
    }

    free(info);
    close(sock);

    return 0;
}
```

**Compile:**
```bash
gcc -o scanner scanner.c -lbluetooth
```

**BLE Scanner (C):**
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/hci.h>
#include <bluetooth/hci_lib.h>

int main() {
    int dev_id, sock;
    uint8_t scan_type = 0x01;  // Active scanning
    uint16_t interval = htobs(0x0010);  // 10ms
    uint16_t window = htobs(0x0010);    // 10ms
    uint8_t own_type = 0x00;  // Public address
    uint8_t filter_policy = 0x00;  // Accept all

    // Get default adapter
    dev_id = hci_get_route(NULL);
    if (dev_id < 0) {
        perror("No adapter found");
        return 1;
    }

    // Open HCI socket
    sock = hci_open_dev(dev_id);
    if (sock < 0) {
        perror("Could not open device");
        return 1;
    }

    // Set scan parameters
    if (hci_le_set_scan_parameters(sock, scan_type, interval, window,
                                   own_type, filter_policy, 1000) < 0) {
        perror("Set scan parameters failed");
        return 1;
    }

    // Enable scanning
    if (hci_le_set_scan_enable(sock, 0x01, 1, 1000) < 0) {
        perror("Enable scan failed");
        return 1;
    }

    printf("Scanning for BLE devices...\n");
    sleep(10);

    // Disable scanning
    hci_le_set_scan_enable(sock, 0x00, 1, 1000);

    hci_close_dev(sock);
    return 0;
}
```

---

## Security & Pairing

### BLE Security Levels

BLE supports four security levels:

| Level | Encryption | Authentication | Name |
|-------|-----------|----------------|------|
| **Level 1** | No | No | No Security |
| **Level 2** | Yes | No | Unauthenticated pairing with encryption |
| **Level 3** | Yes | Yes | Authenticated pairing with encryption |
| **Level 4** | Yes | Yes | LE Secure Connections (BLE 4.2+) |

### Security Modes

**Security Mode 1:**
- Level 1: No security (no encryption, no authentication)
- Level 2: Unauthenticated pairing with encryption
- Level 3: Authenticated pairing with encryption
- Level 4: LE Secure Connections with 128-bit strength

**Security Mode 2:**
- Data signing (authentication without encryption)
- Less commonly used

### Pairing Methods

#### 1. Just Works

- No user interaction required
- No MITM (Man-in-the-Middle) protection
- Used when neither device has display or keyboard
- **Security**: Vulnerable to passive eavesdropping and MITM attacks

**Process:**
```
Device A          Device B
   |                  |
   |--- Pairing Req ->|
   |<-- Pairing Rsp --|
   |                  |
   | (Exchange random values)
   |                  |
   |<-- Encrypted -->|
```

#### 2. Passkey Entry

- User enters same 6-digit PIN on both devices
- MITM protection
- Requires at least one device with keyboard/display

**Use Cases:**
- Display + Keyboard: User sees PIN on one, enters on other
- Keyboard + Keyboard: User enters same PIN on both
- Display + Display: Devices show same PIN, user confirms

**Process:**
1. Devices exchange capabilities
2. One device displays 6-digit passkey
3. User enters passkey on other device
4. Devices verify and establish encrypted connection

#### 3. Numeric Comparison (LE Secure Connections)

- Both devices display 6-digit number
- User confirms if numbers match
- MITM protection
- Requires both devices to have displays

#### 4. Out of Band (OOB)

- Pairing data exchanged via alternative channel (NFC, QR code, etc.)
- Highest security
- Requires additional hardware/technology

**Example:** Scan QR code to get pairing information

### Bonding

**Bonding** is the process of storing pairing information for future reconnections.

**Bonded Information:**
- Long Term Key (LTK)
- Identity Resolving Key (IRK)
- Connection Signature Resolving Key (CSRK)
- Device addresses

**BlueZ Bonding:**
```bash
bluetoothctl

# Trust device (allows auto-reconnect)
[bluetooth]# trust AA:BB:CC:DD:EE:FF

# List paired devices
[bluetooth]# paired-devices

# Remove bonding
[bluetooth]# remove AA:BB:CC:DD:EE:FF
```

### Privacy Features

#### Address Resolution

BLE supports private addresses that change periodically to prevent tracking.

**Address Types:**
1. **Public**: Fixed, similar to MAC address
2. **Random Static**: Random but doesn't change
3. **Private Resolvable**: Changes periodically, can be resolved by bonded devices
4. **Private Non-Resolvable**: Changes periodically, cannot be resolved

**Enable Privacy in BlueZ:**
Edit `/etc/bluetooth/main.conf`:
```ini
[General]
Privacy = device
```

#### LE Secure Connections (BLE 4.2+)

Improvements over legacy pairing:
- ECDH (Elliptic Curve Diffie-Hellman) key exchange
- Stronger MITM protection
- Numeric Comparison pairing method
- Mandatory for Security Level 4

### Best Practices

1. **Always Use Encryption**: Require Security Level 2 or higher for sensitive data
2. **Implement Bonding**: Store keys for trusted devices
3. **Use LE Secure Connections**: When both devices support BLE 4.2+
4. **Implement Application-Level Security**: Don't rely solely on BLE security
   - Use TLS/DTLS for critical data
   - Implement authentication tokens
5. **Validate Data**: Always validate received data
6. **Use Appropriate Pairing Method**:
   - Just Works: Only for non-sensitive applications
   - Passkey Entry or Numeric Comparison: For sensitive applications
7. **Enable Privacy**: Use resolvable private addresses to prevent tracking
8. **Update Firmware**: Keep BLE devices updated to patch vulnerabilities
9. **Implement Timeouts**: Disconnect idle connections
10. **Limit Permissions**: Only allow necessary operations (read vs write)

**Setting Permissions in GATT Characteristic:**
```python
# Example characteristic permissions
permissions = [
    'read',           # Allow read
    'write',          # Allow write (requires response)
    'encrypt-read',   # Require encryption for read
    'encrypt-write',  # Require encryption for write
    'secure-read',    # Require authenticated encryption
    'secure-write',   # Require authenticated encryption
]
```

---

## Practical Examples

### Example 1: BLE Thermometer Reader

```python
#!/usr/bin/env python3
"""
Read temperature from a BLE thermometer
Assumes Health Thermometer Service (0x1809)
"""
import asyncio
import struct
from bleak import BleakClient, BleakScanner

HEALTH_THERMOMETER_SERVICE = "00001809-0000-1000-8000-00805f9b34fb"
TEMPERATURE_MEASUREMENT_CHAR = "00002a1c-0000-1000-8000-00805f9b34fb"

def parse_temperature(data):
    """Parse temperature measurement characteristic"""
    flags = data[0]

    # Check if Fahrenheit (bit 0)
    unit = "°F" if flags & 0x01 else "°C"

    # Temperature is IEEE-11073 32-bit float
    temp_bytes = data[1:5]

    # IEEE-11073 format: mantissa (24-bit) + exponent (8-bit)
    mantissa = int.from_bytes(temp_bytes[0:3], byteorder='little', signed=True)
    exponent = struct.unpack('b', bytes([temp_bytes[3]]))[0]  # signed 8-bit

    temperature = mantissa * (10 ** exponent)

    return temperature, unit

def notification_handler(sender, data):
    temp, unit = parse_temperature(data)
    print(f"Temperature: {temp:.1f} {unit}")

async def main():
    print("Scanning for thermometers...")

    devices = await BleakScanner.discover(
        timeout=5.0,
        service_uuids=[HEALTH_THERMOMETER_SERVICE]
    )

    if not devices:
        print("No thermometer found")
        return

    device = devices[0]
    print(f"Found thermometer: {device.name} ({device.address})")

    async with BleakClient(device.address) as client:
        print("Connected! Waiting for temperature measurements...")

        await client.start_notify(TEMPERATURE_MEASUREMENT_CHAR, notification_handler)

        # Listen for 60 seconds
        await asyncio.sleep(60)

        await client.stop_notify(TEMPERATURE_MEASUREMENT_CHAR)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: iBeacon Scanner

```python
#!/usr/bin/env python3
"""
Scan for iBeacons and parse their data
"""
import asyncio
from bleak import BleakScanner

def parse_ibeacon(manufacturer_data):
    """Parse iBeacon advertisement data"""
    # iBeacon: Company ID (0x004C = Apple) + iBeacon prefix (0x02 0x15)
    for company_id, data in manufacturer_data.items():
        if company_id == 0x004C and len(data) >= 23:
            if data[0:2] == bytes([0x02, 0x15]):
                # iBeacon format:
                # 0-15: UUID (16 bytes)
                # 16-17: Major (2 bytes)
                # 18-19: Minor (2 bytes)
                # 20: TX Power (1 byte, signed)

                uuid = data[2:18].hex()
                uuid_formatted = f"{uuid[0:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:32]}"
                major = int.from_bytes(data[18:20], byteorder='big')
                minor = int.from_bytes(data[20:22], byteorder='big')
                tx_power = struct.unpack('b', bytes([data[22]]))[0]

                return {
                    'uuid': uuid_formatted,
                    'major': major,
                    'minor': minor,
                    'tx_power': tx_power
                }
    return None

def detection_callback(device, advertisement_data):
    beacon_data = parse_ibeacon(advertisement_data.manufacturer_data)

    if beacon_data:
        print(f"\niBeacon detected:")
        print(f"  Address: {device.address}")
        print(f"  UUID: {beacon_data['uuid']}")
        print(f"  Major: {beacon_data['major']}")
        print(f"  Minor: {beacon_data['minor']}")
        print(f"  TX Power: {beacon_data['tx_power']} dBm")
        print(f"  RSSI: {advertisement_data.rssi} dBm")

        # Estimate distance (very rough)
        if advertisement_data.rssi:
            ratio = advertisement_data.rssi / beacon_data['tx_power']
            if ratio < 1.0:
                distance = ratio ** 10
            else:
                distance = (0.89976) * (ratio ** 7.7095) + 0.111
            print(f"  Estimated distance: {distance:.2f} m")

async def main():
    print("Scanning for iBeacons... (Press Ctrl+C to stop)")

    scanner = BleakScanner(detection_callback)
    await scanner.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scan...")

    await scanner.stop()

if __name__ == "__main__":
    import struct
    asyncio.run(main())
```

### Example 3: Nordic UART Service (NUS)

Nordic UART Service provides simple serial-like communication over BLE.

```python
#!/usr/bin/env python3
"""
Nordic UART Service (NUS) example
Allows bidirectional serial communication over BLE
"""
import asyncio
from bleak import BleakClient, BleakScanner

# Nordic UART Service UUIDs
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Write to this
NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Receive from this

def notification_handler(sender, data):
    """Handle incoming data"""
    try:
        message = data.decode('utf-8')
        print(f"Received: {message}")
    except:
        print(f"Received (hex): {data.hex()}")

async def main():
    print("Scanning for devices with Nordic UART Service...")

    devices = await BleakScanner.discover(
        timeout=5.0,
        service_uuids=[NUS_SERVICE_UUID]
    )

    if not devices:
        print("No device with NUS found")
        return

    device = devices[0]
    print(f"Connecting to {device.name} ({device.address})...")

    async with BleakClient(device.address) as client:
        print("Connected!")

        # Enable notifications for RX
        await client.start_notify(NUS_TX_CHAR_UUID, notification_handler)

        # Send data
        message = "Hello from Python!\n"
        await client.write_gatt_char(NUS_RX_CHAR_UUID, message.encode('utf-8'))
        print(f"Sent: {message}")

        # Listen for responses
        await asyncio.sleep(10)

        await client.stop_notify(NUS_TX_CHAR_UUID)

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 4: Battery Level Monitor

```python
#!/usr/bin/env python3
"""
Monitor battery level from BLE device
"""
import asyncio
from bleak import BleakClient

BATTERY_SERVICE_UUID = "0000180f-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL_CHAR_UUID = "00002a19-0000-1000-8000-00805f9b34fb"

async def read_battery(address):
    async with BleakClient(address) as client:
        # Check if battery service exists
        services = client.services

        battery_service = services.get_service(BATTERY_SERVICE_UUID)
        if not battery_service:
            print("Device does not have Battery Service")
            return

        # Read battery level
        battery_level = await client.read_gatt_char(BATTERY_LEVEL_CHAR_UUID)
        level = int.from_bytes(battery_level, byteorder='little')

        print(f"Battery Level: {level}%")

        # Check if notifications are supported
        char = battery_service.get_characteristic(BATTERY_LEVEL_CHAR_UUID)
        if 'notify' in char.properties:
            print("Battery notifications supported")

            def battery_notification_handler(sender, data):
                level = int.from_bytes(data, byteorder='little')
                print(f"Battery Level Updated: {level}%")

            await client.start_notify(BATTERY_LEVEL_CHAR_UUID, battery_notification_handler)
            print("Listening for battery updates... (30 seconds)")
            await asyncio.sleep(30)
            await client.stop_notify(BATTERY_LEVEL_CHAR_UUID)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python battery_monitor.py <device_address>")
        sys.exit(1)

    address = sys.argv[1]
    asyncio.run(read_battery(address))
```

---

## Troubleshooting

### Common Issues

#### 1. Device Not Found During Scanning

**Symptoms:** `hcitool lescan` or BleakScanner returns no devices

**Possible Causes & Solutions:**

- **Bluetooth is off:**
  ```bash
  sudo hciconfig hci0 up
  # or
  bluetoothctl power on
  ```

- **Insufficient permissions:**
  ```bash
  # Run with sudo
  sudo hcitool lescan

  # Or add capabilities to Python
  sudo setcap cap_net_raw,cap_net_admin+eip $(which python3)
  ```

- **Device is not advertising:**
  - Check if device is in pairing/discoverable mode
  - Verify device battery level
  - Check device is within range

- **RF interference:**
  - Move away from WiFi routers, microwaves
  - Try different physical location
  - Check for other Bluetooth devices

#### 2. Connection Failures

**Symptoms:** Cannot establish connection to device

**Solutions:**

- **Reset Bluetooth adapter:**
  ```bash
  sudo hciconfig hci0 reset
  sudo systemctl restart bluetooth
  ```

- **Remove old pairing:**
  ```bash
  bluetoothctl remove AA:BB:CC:DD:EE:FF
  ```

- **Check connection parameters:**
  - Some devices require specific connection intervals
  - Try increasing supervision timeout

- **Verify device is connectable:**
  - Some beacons only advertise, don't accept connections

#### 3. GATT Operations Fail

**Symptoms:** Cannot read/write characteristics

**Solutions:**

- **Check permissions:**
  ```bash
  # In bluetoothctl, check characteristic flags
  bluetoothctl
  [bluetooth]# menu gatt
  [bluetooth]# list-attributes <device>
  ```

- **Enable notifications on CCCD:**
  ```python
  # For notifications, ensure CCCD is written
  CCCD_UUID = "00002902-0000-1000-8000-00805f9b34fb"
  await client.write_gatt_char(CCCD_UUID, b"\x01\x00")
  ```

- **Increase MTU:**
  ```python
  # Request larger MTU for better throughput
  await client.exchange_mtu(512)
  ```

#### 4. Pairing Issues

**Symptoms:** Pairing fails or doesn't complete

**Solutions:**

- **Use bluetoothctl for pairing:**
  ```bash
  bluetoothctl
  [bluetooth]# agent on
  [bluetooth]# default-agent
  [bluetooth]# pair AA:BB:CC:DD:EE:FF
  ```

- **Check agent is running:**
  ```bash
  # Ensure bluetooth-agent or bluetoothctl agent is active
  ps aux | grep agent
  ```

- **Clear previous pairing:**
  ```bash
  bluetoothctl remove AA:BB:CC:DD:EE:FF
  # Then pair again
  ```

#### 5. Disconnections

**Symptoms:** Device disconnects frequently

**Solutions:**

- **Check signal strength:**
  ```bash
  # In bluetoothctl
  [bluetooth]# info AA:BB:CC:DD:EE:FF
  # Look for RSSI value
  ```

- **Adjust connection parameters:**
  - Increase supervision timeout
  - Reduce connection interval
  - Some devices need specific parameters

- **Check for interference:**
  - Verify no physical obstructions
  - Check for WiFi on same 2.4 GHz band
  - Try different channels

- **Update firmware:**
  - Check for Bluetooth adapter firmware updates
  - Update device firmware

#### 6. High Latency

**Symptoms:** Slow response to commands

**Solutions:**

- **Reduce connection interval:**
  - Shorter interval = lower latency, higher power
  - Some devices allow negotiating connection parameters

- **Use write without response:**
  ```python
  await client.write_gatt_char(uuid, data, response=False)
  ```

- **Increase MTU:**
  ```python
  await client.exchange_mtu(512)
  ```

### Debugging Tools

#### btmon - Packet Capture

```bash
# Capture all Bluetooth traffic
sudo btmon

# Save to file for analysis
sudo btmon -w capture.btsnoop

# Analyze with Wireshark
wireshark capture.btsnoop
```

#### hcidump

```bash
# Dump HCI data
sudo hcidump -X
```

#### Check BlueZ Version

```bash
bluetoothctl --version
```

#### Check Kernel Module

```bash
# Check if Bluetooth modules loaded
lsmod | grep bluetooth

# Reload modules
sudo modprobe -r btusb
sudo modprobe btusb
```

#### dmesg Logs

```bash
# Check for Bluetooth errors
dmesg | grep -i bluetooth
```

### Performance Optimization

#### Maximize Throughput

1. **Use largest MTU possible:**
   ```python
   await client.exchange_mtu(512)
   ```

2. **Use write without response:**
   ```python
   await client.write_gatt_char(uuid, data, response=False)
   ```

3. **Minimize connection interval:**
   - Negotiate shortest interval device supports
   - Typically 7.5ms minimum

4. **Use BLE 5.0 2M PHY (if supported):**
   - Doubles data rate to 2 Mbps
   - Requires BLE 5.0 hardware on both sides

#### Minimize Power Consumption

1. **Increase connection interval:**
   - Longer interval = less power
   - Trade-off with latency

2. **Use slave latency:**
   - Allows peripheral to skip connection events
   - Peripheral can sleep more

3. **Reduce advertising frequency:**
   - Longer advertising interval
   - Only advertise when needed

4. **Use non-connectable advertising:**
   - For broadcast-only applications (beacons)

---

## References

### Official Specifications

- [Bluetooth Core Specification](https://www.bluetooth.com/specifications/bluetooth-core-specification/)
- [GATT Specifications](https://www.bluetooth.com/specifications/specs/gatt-specification-supplement/)
- [Assigned Numbers (UUIDs, Company IDs)](https://www.bluetooth.com/specifications/assigned-numbers/)

### BlueZ Documentation

- [BlueZ Official Site](http://www.bluez.org/)
- [BlueZ Git Repository](https://github.com/bluez/bluez)
- [BlueZ D-Bus API](https://git.kernel.org/pub/scm/bluetooth/bluez.git/tree/doc)

### Tools & Libraries

- [Bleak (Python)](https://github.com/hbldh/bleak)
- [PyBluez (Python)](https://github.com/pybluez/pybluez)
- [noble (Node.js)](https://github.com/abandonware/noble)
- [Web Bluetooth API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Bluetooth_API)

### Learning Resources

- [Bluetooth Low Energy: The Developer's Handbook](https://www.bluetooth.com/blog/bluetooth-low-energy-the-developers-handbook/)
- [Introduction to BLE](https://learn.adafruit.com/introduction-to-bluetooth-low-energy/introduction)
- [Nordic Semiconductor Developer Zone](https://devzone.nordicsemi.com/)

### Related Topics

- [Bluetooth Classic](./bluetooth.md)
- [Linux Networking](../linux/networking.md)
- [IoT Protocols](./iot.md)
- [Wireless Communication](./wireless.md)
