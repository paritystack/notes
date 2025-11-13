# USB Protocol

Comprehensive guide to USB protocol, device classes, and embedded implementation.

## Table of Contents

1. [Introduction](#introduction)
2. [USB Basics](#usb-basics)
3. [USB Protocol](#usb-protocol)
4. [Device Classes](#device-classes)
5. [Descriptors](#descriptors)
6. [Arduino USB](#arduino-usb)
7. [STM32 USB](#stm32-usb)
8. [USB CDC (Virtual Serial)](#usb-cdc-virtual-serial)

## Introduction

USB (Universal Serial Bus) is a standard for connecting devices to a host computer. It provides both power and data communication in a single cable.

### USB Versions

| Version | Name | Speed | Release | Connector |
|---------|------|-------|---------|-----------|
| **USB 1.0** | Low Speed | 1.5 Mbps | 1996 | Type A/B |
| **USB 1.1** | Full Speed | 12 Mbps | 1998 | Type A/B |
| **USB 2.0** | High Speed | 480 Mbps | 2000 | Type A/B, Mini, Micro |
| **USB 3.0** | SuperSpeed | 5 Gbps | 2008 | Type A/B, Micro B SS |
| **USB 3.1** | SuperSpeed+ | 10 Gbps | 2013 | Type C |
| **USB 3.2** | - | 20 Gbps | 2017 | Type C |
| **USB 4.0** | - | 40 Gbps | 2019 | Type C |

### USB Connectors

```
USB Type A (Host):
┌─────────────┐
│ ┌─┐ ┌─┐ ┌─┐ │
│ │1│ │2│ │3│ │4│
│ └─┘ └─┘ └─┘ │
└─────────────┘
1: VBUS (+5V)
2: D- (Data -)
3: D+ (Data +)
4: GND

USB Type B (Device):
    ┌───┐
  ┌─┘   └─┐
  │ 1   2 │
  │ 3   4 │
  └───────┘

USB Micro B (Common on embedded):
  ┌─────────┐
  │1 2 3 4 5│
  └─────────┘
1: VBUS (+5V)
2: D-
3: D+
4: ID (OTG)
5: GND

USB Type C (Modern):
  ┌───────────┐
  │A1 A2...A12│
  │B1 B2...B12│
  └───────────┘
Reversible, 24 pins
```

### USB Topology

```
Host (PC/Hub)
    │
    ├─── Device 1 (Address 1)
    │
    ├─── Hub (Address 2)
    │     │
    │     ├─── Device 2 (Address 3)
    │     └─── Device 3 (Address 4)
    │
    └─── Device 4 (Address 5)

Maximum:
- 127 devices per host
- 5 meter cable length per segment
- 7 tiers (including hub)
```

## USB Basics

### Signal Levels

- **Low Speed (1.5 Mbps)**: D+ pulled down, D- pulled up
- **Full Speed (12 Mbps)**: D+ pulled up, D- pulled down
- **High Speed (480 Mbps)**: Differential signaling

### Power

```
USB 2.0: 5V, 500 mA max
USB 3.0: 5V, 900 mA max
USB-C PD: 5V, 9V, 15V, 20V up to 100W
```

### Data Encoding

USB uses **NRZI** (Non-Return-to-Zero Inverted) encoding with bit stuffing:
- `0` bit: Transition
- `1` bit: No transition
- Bit stuffing: After six consecutive `1`s, insert a `0`

### Packet Types

```
Token Packets:
- SETUP: Initialize control transfer
- IN: Request data from device
- OUT: Send data to device
- SOF: Start of Frame (every 1 ms)

Data Packets:
- DATA0: Even data packet
- DATA1: Odd data packet
- DATA2: High-speed data
- MDATA: Multi-data

Handshake Packets:
- ACK: Acknowledge success
- NAK: Not ready
- STALL: Endpoint halted

Special Packets:
- PRE: Preamble for low-speed
- ERR: Error detected
- SPLIT: High-speed split transaction
```

## USB Protocol

### Enumeration Process

```
1. Device Plugged In
   │
   ├─ USB Reset (SE0 for 10ms)
   │
2. Host Assigns Address 0 (default)
   │
   ├─ Get Device Descriptor
   │  Response: VID, PID, max packet size
   │
3. Host Assigns Unique Address (1-127)
   │
   ├─ Set Address
   │
4. Host Requests Configuration
   │
   ├─ Get Configuration Descriptor
   │  Response: Interfaces, endpoints, class info
   │
   ├─ Get String Descriptors (optional)
   │  Response: Manufacturer, product, serial
   │
5. Host Configures Device
   │
   ├─ Set Configuration
   │
6. Device Ready for Use
```

### Transfer Types

| Transfer Type | Speed | Error Correction | Use Case |
|---------------|-------|------------------|----------|
| **Control** | Any | Yes | Device enumeration, configuration |
| **Bulk** | Full/High | Yes | Large data transfers (storage, printers) |
| **Interrupt** | Any | Yes | Small, periodic data (HID, mice) |
| **Isochronous** | Full/High | No | Real-time audio/video |

### Control Transfer Structure

```
Setup Stage:
  Host → Device: SETUP token + DATA0 packet

Data Stage (optional):
  IN:  Device → Host: DATA packets
  OUT: Host → Device: DATA packets

Status Stage:
  IN:  Device → Host: Zero-length DATA1 + ACK
  OUT: Host → Device: Zero-length DATA1 + ACK
```

### Standard Requests

```c
// bmRequestType: Direction | Type | Recipient
#define USB_DIR_OUT     0x00
#define USB_DIR_IN      0x80

#define USB_TYPE_STANDARD   0x00
#define USB_TYPE_CLASS      0x20
#define USB_TYPE_VENDOR     0x40

#define USB_RECIP_DEVICE    0x00
#define USB_RECIP_INTERFACE 0x01
#define USB_RECIP_ENDPOINT  0x02

// bRequest codes
#define USB_REQ_GET_STATUS        0
#define USB_REQ_CLEAR_FEATURE     1
#define USB_REQ_SET_FEATURE       3
#define USB_REQ_SET_ADDRESS       5
#define USB_REQ_GET_DESCRIPTOR    6
#define USB_REQ_SET_DESCRIPTOR    7
#define USB_REQ_GET_CONFIGURATION 8
#define USB_REQ_SET_CONFIGURATION 9
#define USB_REQ_GET_INTERFACE     10
#define USB_REQ_SET_INTERFACE     11
```

## Device Classes

### USB Class Codes

| Class | Code | Description | Examples |
|-------|------|-------------|----------|
| **CDC** | 0x02 | Communications Device | Virtual COM port, modems |
| **HID** | 0x03 | Human Interface Device | Keyboards, mice, game controllers |
| **Mass Storage** | 0x08 | Storage Device | USB flash drives, external HDDs |
| **Hub** | 0x09 | USB Hub | - |
| **Audio** | 0x01 | Audio Device | Speakers, microphones |
| **Video** | 0x0E | Video Device | Webcams |
| **Printer** | 0x07 | Printer | - |
| **Vendor Specific** | 0xFF | Custom | - |

### HID (Human Interface Device)

```c
// HID Descriptor
struct HID_Descriptor {
    uint8_t  bLength;           // Size of descriptor
    uint8_t  bDescriptorType;   // HID descriptor type (0x21)
    uint16_t bcdHID;            // HID specification release
    uint8_t  bCountryCode;      // Country code
    uint8_t  bNumDescriptors;   // Number of class descriptors
    uint8_t  bDescriptorType2;  // Report descriptor type (0x22)
    uint16_t wDescriptorLength; // Length of report descriptor
};

// HID Report Descriptor (Mouse example)
const uint8_t mouse_report_descriptor[] = {
    0x05, 0x01,        // Usage Page (Generic Desktop)
    0x09, 0x02,        // Usage (Mouse)
    0xA1, 0x01,        // Collection (Application)
    0x09, 0x01,        //   Usage (Pointer)
    0xA1, 0x00,        //   Collection (Physical)
    0x05, 0x09,        //     Usage Page (Buttons)
    0x19, 0x01,        //     Usage Minimum (Button 1)
    0x29, 0x03,        //     Usage Maximum (Button 3)
    0x15, 0x00,        //     Logical Minimum (0)
    0x25, 0x01,        //     Logical Maximum (1)
    0x95, 0x03,        //     Report Count (3)
    0x75, 0x01,        //     Report Size (1)
    0x81, 0x02,        //     Input (Data, Variable, Absolute)
    0x95, 0x01,        //     Report Count (1)
    0x75, 0x05,        //     Report Size (5)
    0x81, 0x01,        //     Input (Constant) - Padding
    0x05, 0x01,        //     Usage Page (Generic Desktop)
    0x09, 0x30,        //     Usage (X)
    0x09, 0x31,        //     Usage (Y)
    0x15, 0x81,        //     Logical Minimum (-127)
    0x25, 0x7F,        //     Logical Maximum (127)
    0x75, 0x08,        //     Report Size (8)
    0x95, 0x02,        //     Report Count (2)
    0x81, 0x06,        //     Input (Data, Variable, Relative)
    0xC0,              //   End Collection
    0xC0               // End Collection
};
```

### CDC (Communication Device Class)

Used for virtual serial ports (USB to UART).

```c
// CDC ACM (Abstract Control Model) Interface

// CDC Header Functional Descriptor
struct CDC_Header_Descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;
    uint8_t  bDescriptorSubtype;  // Header (0x00)
    uint16_t bcdCDC;
};

// CDC Call Management Descriptor
struct CDC_CallManagement_Descriptor {
    uint8_t bLength;
    uint8_t bDescriptorType;
    uint8_t bDescriptorSubtype;  // Call Management (0x01)
    uint8_t bmCapabilities;
    uint8_t bDataInterface;
};

// CDC Line Coding (115200 8N1 example)
struct CDC_LineCoding {
    uint32_t dwDTERate;      // Baud rate: 115200
    uint8_t  bCharFormat;    // Stop bits: 1
    uint8_t  bParityType;    // Parity: None (0)
    uint8_t  bDataBits;      // Data bits: 8
};
```

## Descriptors

### Device Descriptor

```c
struct USB_Device_Descriptor {
    uint8_t  bLength;            // Size: 18 bytes
    uint8_t  bDescriptorType;    // DEVICE (0x01)
    uint16_t bcdUSB;             // USB version (0x0200 for USB 2.0)
    uint8_t  bDeviceClass;       // Class code
    uint8_t  bDeviceSubClass;    // Subclass code
    uint8_t  bDeviceProtocol;    // Protocol code
    uint8_t  bMaxPacketSize0;    // Max packet size for EP0
    uint16_t idVendor;           // Vendor ID (VID)
    uint16_t idProduct;          // Product ID (PID)
    uint16_t bcdDevice;          // Device release number
    uint8_t  iManufacturer;      // Manufacturer string index
    uint8_t  iProduct;           // Product string index
    uint8_t  iSerialNumber;      // Serial number string index
    uint8_t  bNumConfigurations; // Number of configurations
};

// Example
const uint8_t device_descriptor[] = {
    18,         // bLength
    0x01,       // bDescriptorType (DEVICE)
    0x00, 0x02, // bcdUSB (USB 2.0)
    0x00,       // bDeviceClass (defined in interface)
    0x00,       // bDeviceSubClass
    0x00,       // bDeviceProtocol
    64,         // bMaxPacketSize0
    0x83, 0x04, // idVendor (0x0483 - STMicroelectronics)
    0x40, 0x57, // idProduct (0x5740)
    0x00, 0x02, // bcdDevice (2.0)
    1,          // iManufacturer
    2,          // iProduct
    3,          // iSerialNumber
    1           // bNumConfigurations
};
```

### Configuration Descriptor

```c
struct USB_Configuration_Descriptor {
    uint8_t  bLength;             // Size: 9 bytes
    uint8_t  bDescriptorType;     // CONFIGURATION (0x02)
    uint16_t wTotalLength;        // Total length of data
    uint8_t  bNumInterfaces;      // Number of interfaces
    uint8_t  bConfigurationValue; // Configuration index
    uint8_t  iConfiguration;      // Configuration string index
    uint8_t  bmAttributes;        // Attributes (self/bus powered)
    uint8_t  bMaxPower;           // Max power in 2mA units
};
```

### Interface Descriptor

```c
struct USB_Interface_Descriptor {
    uint8_t bLength;            // Size: 9 bytes
    uint8_t bDescriptorType;    // INTERFACE (0x04)
    uint8_t bInterfaceNumber;   // Interface index
    uint8_t bAlternateSetting;  // Alternate setting
    uint8_t bNumEndpoints;      // Number of endpoints
    uint8_t bInterfaceClass;    // Class code
    uint8_t bInterfaceSubClass; // Subclass code
    uint8_t bInterfaceProtocol; // Protocol code
    uint8_t iInterface;         // Interface string index
};
```

### Endpoint Descriptor

```c
struct USB_Endpoint_Descriptor {
    uint8_t  bLength;          // Size: 7 bytes
    uint8_t  bDescriptorType;  // ENDPOINT (0x05)
    uint8_t  bEndpointAddress; // Address (bit 7: direction)
    uint8_t  bmAttributes;     // Transfer type
    uint16_t wMaxPacketSize;   // Max packet size
    uint8_t  bInterval;        // Polling interval (ms)
};

// Endpoint address format:
// Bit 7: Direction (0 = OUT, 1 = IN)
// Bits 3-0: Endpoint number (0-15)
#define USB_EP_IN(n)  (0x80 | (n))
#define USB_EP_OUT(n) (n)

// Transfer types
#define USB_EP_TYPE_CONTROL     0x00
#define USB_EP_TYPE_ISOCHRONOUS 0x01
#define USB_EP_TYPE_BULK        0x02
#define USB_EP_TYPE_INTERRUPT   0x03
```

### String Descriptor

```c
struct USB_String_Descriptor {
    uint8_t bLength;
    uint8_t bDescriptorType;  // STRING (0x03)
    uint16_t wString[];       // Unicode string
};

// String 0 (Language ID)
const uint8_t string0[] = {
    4,      // bLength
    0x03,   // bDescriptorType
    0x09, 0x04  // wLANGID[0]: 0x0409 (English - US)
};

// String 1 (Manufacturer)
const uint8_t string1[] = {
    28,     // bLength
    0x03,   // bDescriptorType
    'M',0, 'a',0, 'n',0, 'u',0, 'f',0, 'a',0, 'c',0, 
    't',0, 'u',0, 'r',0, 'e',0, 'r',0, 0,0
};
```

## Arduino USB

### Arduino Leonardo/Micro (ATmega32u4)

The ATmega32u4 has native USB support.

#### USB Mouse

```cpp
#include <Mouse.h>

void setup() {
    Mouse.begin();
}

void loop() {
    // Move mouse in a square
    Mouse.move(10, 0);   // Right
    delay(500);
    Mouse.move(0, 10);   // Down
    delay(500);
    Mouse.move(-10, 0);  // Left
    delay(500);
    Mouse.move(0, -10);  // Up
    delay(500);
}
```

#### USB Keyboard

```cpp
#include <Keyboard.h>

const int BUTTON_PIN = 2;

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    Keyboard.begin();
}

void loop() {
    if (digitalRead(BUTTON_PIN) == LOW) {
        Keyboard.print("Hello, World!");
        delay(500);
    }
}
```

#### USB HID Custom

```cpp
#include <HID.h>

// Custom HID report descriptor
static const uint8_t _hidReportDescriptor[] PROGMEM = {
    0x06, 0x00, 0xFF,  // Usage Page (Vendor Defined)
    0x09, 0x01,        // Usage (Vendor Usage 1)
    0xA1, 0x01,        // Collection (Application)
    0x15, 0x00,        //   Logical Minimum (0)
    0x26, 0xFF, 0x00,  //   Logical Maximum (255)
    0x75, 0x08,        //   Report Size (8 bits)
    0x95, 0x40,        //   Report Count (64)
    0x09, 0x01,        //   Usage (Vendor Usage 1)
    0x81, 0x02,        //   Input (Data, Variable, Absolute)
    0x09, 0x01,        //   Usage (Vendor Usage 1)
    0x91, 0x02,        //   Output (Data, Variable, Absolute)
    0xC0               // End Collection
};

void setup() {
    static HIDSubDescriptor node(_hidReportDescriptor, sizeof(_hidReportDescriptor));
    HID().AppendDescriptor(&node);
}

void loop() {
    uint8_t data[64] = {1, 2, 3, 4};
    HID().SendReport(1, data, 64);
    delay(100);
}
```

## STM32 USB

### USB CDC Virtual COM Port (CubeMX)

```c
/* Generated by CubeMX with USB Device middleware */
#include "usbd_cdc_if.h"

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_USB_DEVICE_Init();
    
    uint8_t buffer[64];
    sprintf((char*)buffer, "Hello from STM32!\r\n");
    
    while (1) {
        CDC_Transmit_FS(buffer, strlen((char*)buffer));
        HAL_Delay(1000);
    }
}

/* In usbd_cdc_if.c */
static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len) {
    // Echo back received data
    CDC_Transmit_FS(Buf, *Len);
    return USBD_OK;
}
```

### USB HID Keyboard

```c
/* Configure USB Device as HID in CubeMX */
#include "usbd_hid.h"

extern USBD_HandleTypeDef hUsbDeviceFS;

// HID keyboard report
typedef struct {
    uint8_t modifiers;  // Ctrl, Shift, Alt, GUI
    uint8_t reserved;
    uint8_t keys[6];    // Up to 6 simultaneous keys
} KeyboardReport;

void send_key(uint8_t key) {
    KeyboardReport report = {0};
    
    // Press key
    report.keys[0] = key;
    USBD_HID_SendReport(&hUsbDeviceFS, (uint8_t*)&report, sizeof(report));
    HAL_Delay(10);
    
    // Release key
    memset(&report, 0, sizeof(report));
    USBD_HID_SendReport(&hUsbDeviceFS, (uint8_t*)&report, sizeof(report));
    HAL_Delay(10);
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_USB_DEVICE_Init();
    
    HAL_Delay(1000);  // Wait for enumeration
    
    while (1) {
        // Send 'A' key
        send_key(0x04);  // HID usage code for 'A'
        HAL_Delay(1000);
    }
}
```

### USB Mass Storage

```c
/* Configure USB Device as MSC in CubeMX */
#include "usbd_storage_if.h"

// Implement SCSI commands
int8_t STORAGE_Read_FS(uint8_t lun, uint8_t *buf, uint32_t blk_addr, uint16_t blk_len) {
    // Read from SD card or internal flash
    for (uint16_t i = 0; i < blk_len; i++) {
        // Read block at (blk_addr + i) to (buf + i * BLOCK_SIZE)
    }
    return USBD_OK;
}

int8_t STORAGE_Write_FS(uint8_t lun, uint8_t *buf, uint32_t blk_addr, uint16_t blk_len) {
    // Write to SD card or internal flash
    for (uint16_t i = 0; i < blk_len; i++) {
        // Write block at (blk_addr + i) from (buf + i * BLOCK_SIZE)
    }
    return USBD_OK;
}
```

## USB CDC (Virtual Serial)

### PC Side (Python)

```python
import serial
import time

# Open serial port
ser = serial.Serial('COM3', 115200, timeout=1)  # Windows
# ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Linux

# Write data
ser.write(b'Hello, Device!\n')

# Read data
while True:
    if ser.in_waiting > 0:
        data = ser.readline()
        print(f"Received: {data.decode()}")
    
    time.sleep(0.1)

ser.close()
```

### PC Side (C++)

```cpp
// Linux example
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

int main() {
    int fd = open("/dev/ttyACM0", O_RDWR);
    
    // Configure serial port
    struct termios tty;
    tcgetattr(fd, &tty);
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tcsetattr(fd, TCSANOW, &tty);
    
    // Write
    char msg[] = "Hello, Device!\n";
    write(fd, msg, sizeof(msg));
    
    // Read
    char buffer[256];
    int n = read(fd, buffer, sizeof(buffer));
    buffer[n] = '\0';
    printf("Received: %s\n", buffer);
    
    close(fd);
    return 0;
}
```

## Best Practices

1. **VID/PID**: Use unique Vendor ID and Product ID (or get your own)
2. **Descriptors**: Ensure correct descriptor chain
3. **Enumeration**: Handle USB reset and enumeration properly
4. **Power**: Declare correct power consumption
5. **String Descriptors**: Provide manufacturer, product, serial number
6. **Error Handling**: Handle NAK, STALL conditions
7. **Buffer Management**: Use DMA for better performance
8. **Compliance**: Test with USB-IF tools for certification

## Debugging Tools

### Linux
```bash
# List USB devices
lsusb

# Detailed info
lsusb -v

# Monitor USB traffic
sudo cat /sys/kernel/debug/usb/usbmon/0u

# Install usbutils
sudo apt install usbutils
```

### Windows
- **USBView**: Microsoft USB device viewer
- **USBDeview**: NirSoft utility
- **Wireshark**: With USB capture support

### Hardware
- **USB Protocol Analyzer**: Beagle USB, Total Phase
- **Logic Analyzer**: Can decode USB signals

## Troubleshooting

### Common Issues

**Device Not Recognized:**
- Check USB cable (data lines)
- Verify correct descriptors
- Check VID/PID not conflicting
- Ensure proper enumeration handling

**Intermittent Disconnects:**
- Power supply insufficient
- Check USB cable quality
- Verify proper suspend/resume handling

**Data Corruption:**
- Check buffer sizes
- Verify DMA configuration
- Ensure proper synchronization

**Slow Transfer Speed:**
- Use bulk transfers for large data
- Enable DMA
- Optimize buffer sizes
- Check USB 2.0 High Speed mode

## Resources

- **USB Specification**: USB.org
- **USB Made Simple**: https://www.usbmadesimple.co.uk/
- **STM32 USB Training**: ST's USB training materials
- **Jan Axelson's USB**: Classic USB development book
- **Linux USB**: https://www.kernel.org/doc/html/latest/driver-api/usb/

## See Also

- [STM32 USB](stm32.md)
- [Arduino USB HID](arduino.md)
- [Communication Protocols](../protocols/)
- [Embedded Systems Overview](README.md)
