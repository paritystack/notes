# ESP32

Comprehensive guide to ESP32 microcontroller development with WiFi and Bluetooth capabilities.

## Table of Contents

1. [Introduction](#introduction)
2. [Hardware Overview](#hardware-overview)
3. [Development Setup](#development-setup)
4. [Basic Programming](#basic-programming)
5. [WiFi Connectivity](#wifi-connectivity)
6. [Bluetooth](#bluetooth)
7. [Advanced Features](#advanced-features)
8. [Projects](#projects)

## Introduction

The ESP32 is a powerful, low-cost microcontroller with integrated WiFi and Bluetooth. Developed by Espressif Systems, it's ideal for IoT projects and wireless applications.

### Key Features

- **Dual-core Xtensa LX6** (or single-core RISC-V in ESP32-C3)
- **Clock Speed**: 160-240 MHz
- **Memory**: 520 KB SRAM, 4 MB Flash (typical)
- **WiFi**: 802.11 b/g/n (2.4 GHz)
- **Bluetooth**: BLE 4.2 and Classic Bluetooth
- **GPIO**: Up to 34 programmable pins
- **Peripherals**: ADC, DAC, SPI, I2C, UART, PWM, I2S
- **Low Power**: Multiple sleep modes
- **Price**: $2-$10 depending on variant

### ESP32 Variants

| Variant | Cores | WiFi | BLE | Classic BT | USB | Special Features |
|---------|-------|------|-----|------------|-----|------------------|
| **ESP32** | 2 | Yes | Yes | Yes | No | Original, most common |
| **ESP32-S2** | 1 | Yes | No | No | Native | USB OTG, lower power |
| **ESP32-S3** | 2 | Yes | Yes | No | Native | Vector instructions |
| **ESP32-C3** | 1 (RISC-V) | Yes | Yes | No | Native | RISC-V architecture |
| **ESP32-C6** | 1 (RISC-V) | Yes | Yes | No | Native | WiFi 6, Zigbee |

## Hardware Overview

### ESP32 DevKit Pinout

```
                    ESP32 DevKit
                ┌─────────────────┐
                │      USB        │
                ├─────────────────┤
    3V3    [ ]──┤3V3           D23├──[ ] MOSI
    EN     [ ]──┤EN            D22├──[ ] SCL (I2C)
    VP/36  [ ]──┤VP/A0         TX0├──[ ] TX
    VN/39  [ ]──┤VN/A3         RX0├──[ ] RX
    D34    [ ]──┤34/A6         D21├──[ ] SDA (I2C)
    D35    [ ]──┤35/A7         GND├──[ ] GND
    D32    [ ]──┤32/A4         D19├──[ ] MISO
    D33    [ ]──┤33/A5         D18├──[ ] SCK
    D25    [ ]──┤25/A18        D5 ├──[ ] SS
    D26    [ ]──┤26/A19        D17├──[ ] TX2
    D27    [ ]──┤27/A17        D16├──[ ] RX2
    D14    [ ]──┤14/A16        D4 ├──[ ]
    D12    [ ]──┤12/A15        D0 ├──[ ] (Boot)
    GND    [ ]──┤GND           D2 ├──[ ] (LED)
    D13    [ ]──┤13/A14        D15├──[ ]
    D9     [ ]──┤9/SD2          D8├──[ ] SD1
    D10    [ ]──┤10/SD3         D7├──[ ] SD0
    D11    [ ]──┤11/CMD         D6├──[ ] SCK
    VIN    [ ]──┤VIN            5V├──[ ] 5V
                └─────────────────┘

Note: Pins 6-11 connected to flash (avoid using)
      Pins with boot/strapping modes: 0, 2, 5, 12, 15
```

### Important Notes

- **Input Only Pins**: GPIO 34-39 (no pull-up/pull-down)
- **Strapping Pins**: 0, 2, 5, 12, 15 (affect boot mode)
- **Boot Mode**: GPIO 0 LOW = download mode
- **Built-in LED**: Usually GPIO 2
- **ADC2**: Cannot use while WiFi active (GPIO 0, 2, 4, 12-15, 25-27)

## Development Setup

### Arduino IDE Setup

```bash
# 1. Install Arduino IDE from arduino.cc

# 2. Add ESP32 Board Manager URL:
# File > Preferences > Additional Board Manager URLs
# Add: https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json

# 3. Install ESP32 boards:
# Tools > Board > Boards Manager > Search "ESP32" > Install

# 4. Select your board:
# Tools > Board > ESP32 Arduino > ESP32 Dev Module
```

### ESP-IDF Setup (Official Framework)

```bash
# Clone ESP-IDF
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf

# Install (Linux/Mac)
./install.sh

# Set up environment (run in each terminal session)
. ./export.sh

# Or add to ~/.bashrc:
alias get_idf='. $HOME/esp/esp-idf/export.sh'

# Create new project
idf.py create-project myproject
cd myproject

# Configure
idf.py menuconfig

# Build
idf.py build

# Flash
idf.py -p /dev/ttyUSB0 flash

# Monitor serial output
idf.py -p /dev/ttyUSB0 monitor
```

### PlatformIO Setup

```bash
# Install PlatformIO
pip install platformio

# Create ESP32 project
pio init --board esp32dev

# Build and upload
pio run --target upload

# Serial monitor
pio device monitor
```

## Basic Programming

### Blink LED (Arduino Framework)

```cpp
#define LED_PIN 2

void setup() {
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    digitalWrite(LED_PIN, HIGH);
    delay(1000);
    digitalWrite(LED_PIN, LOW);
    delay(1000);
}
```

### Dual Core Programming

```cpp
TaskHandle_t Task1;
TaskHandle_t Task2;

void setup() {
    Serial.begin(115200);
    
    // Create task for core 0
    xTaskCreatePinnedToCore(
        Task1code,   // Function
        "Task1",     // Name
        10000,       // Stack size
        NULL,        // Parameters
        1,           // Priority
        &Task1,      // Task handle
        0            // Core ID
    );
    
    // Create task for core 1
    xTaskCreatePinnedToCore(
        Task2code,
        "Task2",
        10000,
        NULL,
        1,
        &Task2,
        1
    );
}

void Task1code(void * parameter) {
    while(1) {
        Serial.print("Task 1 running on core ");
        Serial.println(xPortGetCoreID());
        delay(1000);
    }
}

void Task2code(void * parameter) {
    while(1) {
        Serial.print("Task 2 running on core ");
        Serial.println(xPortGetCoreID());
        delay(500);
    }
}

void loop() {
    // Empty - tasks handle everything
}
```

### Touch Sensor

```cpp
const int TOUCH_PIN = 4;  // T0
const int THRESHOLD = 40;

void setup() {
    Serial.begin(115200);
}

void loop() {
    int touchValue = touchRead(TOUCH_PIN);
    Serial.println(touchValue);
    
    if (touchValue < THRESHOLD) {
        Serial.println("Touch detected!");
    }
    
    delay(500);
}
```

### Hall Effect Sensor (Built-in)

```cpp
void setup() {
    Serial.begin(115200);
}

void loop() {
    // Read built-in Hall effect sensor
    int hallValue = hallRead();
    Serial.print("Hall Sensor: ");
    Serial.println(hallValue);
    
    delay(500);
}
```

## WiFi Connectivity

### WiFi Station Mode (Connect to Router)

```cpp
#include <WiFi.h>

const char* ssid = "YourSSID";
const char* password = "YourPassword";

void setup() {
    Serial.begin(115200);
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("\nConnected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("MAC Address: ");
    Serial.println(WiFi.macAddress());
    Serial.print("Signal Strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
}

void loop() {
    // Check connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi disconnected!");
        WiFi.reconnect();
    }
    
    delay(10000);
}
```

### WiFi Access Point Mode

```cpp
#include <WiFi.h>

const char* ssid = "ESP32-AP";
const char* password = "12345678";  // Minimum 8 characters

void setup() {
    Serial.begin(115200);
    
    // Start Access Point
    WiFi.softAP(ssid, password);
    
    IPAddress IP = WiFi.softAPIP();
    Serial.print("AP IP address: ");
    Serial.println(IP);
}

void loop() {
    // Print number of connected stations
    Serial.print("Stations connected: ");
    Serial.println(WiFi.softAPgetStationNum());
    
    delay(5000);
}
```

### Web Server

```cpp
#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "YourSSID";
const char* password = "YourPassword";

WebServer server(80);

const int LED_PIN = 2;
bool ledState = false;

void handleRoot() {
    String html = "<html><body>";
    html += "<h1>ESP32 Web Server</h1>";
    html += "<p>LED is: " + String(ledState ? "ON" : "OFF") + "</p>";
    html += "<p><a href=\"/led/on\"><button>Turn ON</button></a></p>";
    html += "<p><a href=\"/led/off\"><button>Turn OFF</button></a></p>";
    html += "</body></html>";
    
    server.send(200, "text/html", html);
}

void handleLEDOn() {
    ledState = true;
    digitalWrite(LED_PIN, HIGH);
    server.sendHeader("Location", "/");
    server.send(303);
}

void handleLEDOff() {
    ledState = false;
    digitalWrite(LED_PIN, LOW);
    server.sendHeader("Location", "/");
    server.send(303);
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("\nConnected!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    
    // Setup server routes
    server.on("/", handleRoot);
    server.on("/led/on", handleLEDOn);
    server.on("/led/off", handleLEDOff);
    
    server.begin();
    Serial.println("Web server started");
}

void loop() {
    server.handleClient();
}
```

### HTTP Client (GET Request)

```cpp
#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "YourSSID";
const char* password = "YourPassword";

void setup() {
    Serial.begin(115200);
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected!");
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        
        http.begin("http://api.github.com/users/octocat");
        int httpCode = http.GET();
        
        if (httpCode > 0) {
            Serial.printf("HTTP Code: %d\n", httpCode);
            
            if (httpCode == HTTP_CODE_OK) {
                String payload = http.getString();
                Serial.println(payload);
            }
        } else {
            Serial.printf("Error: %s\n", http.errorToString(httpCode).c_str());
        }
        
        http.end();
    }
    
    delay(10000);
}
```

### WiFi Scan

```cpp
#include <WiFi.h>

void setup() {
    Serial.begin(115200);
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    delay(100);
}

void loop() {
    Serial.println("Scanning WiFi networks...");
    int n = WiFi.scanNetworks();
    
    if (n == 0) {
        Serial.println("No networks found");
    } else {
        Serial.printf("%d networks found:\n", n);
        
        for (int i = 0; i < n; i++) {
            Serial.printf("%d: %s (%d dBm) %s\n",
                i + 1,
                WiFi.SSID(i).c_str(),
                WiFi.RSSI(i),
                WiFi.encryptionType(i) == WIFI_AUTH_OPEN ? "Open" : "Encrypted"
            );
        }
    }
    
    delay(5000);
}
```

### MQTT Client

```cpp
#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "YourSSID";
const char* password = "YourPassword";
const char* mqtt_server = "broker.hivemq.com";

WiFiClient espClient;
PubSubClient client(espClient);

void callback(char* topic, byte* payload, unsigned int length) {
    Serial.print("Message arrived [");
    Serial.print(topic);
    Serial.print("]: ");
    
    for (int i = 0; i < length; i++) {
        Serial.print((char)payload[i]);
    }
    Serial.println();
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("Attempting MQTT connection...");
        
        if (client.connect("ESP32Client")) {
            Serial.println("connected");
            client.subscribe("esp32/test");
        } else {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            Serial.println(" retrying in 5 seconds");
            delay(5000);
        }
    }
}

void setup() {
    Serial.begin(115200);
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");
    
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
}

void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();
    
    // Publish message every 10 seconds
    static unsigned long lastMsg = 0;
    unsigned long now = millis();
    
    if (now - lastMsg > 10000) {
        lastMsg = now;
        
        char msg[50];
        snprintf(msg, 50, "Hello from ESP32 #%lu", millis());
        client.publish("esp32/test", msg);
    }
}
```

## Bluetooth

### Bluetooth Classic - Serial

```cpp
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

void setup() {
    Serial.begin(115200);
    SerialBT.begin("ESP32-BT");  // Bluetooth device name
    Serial.println("Bluetooth Started! Pair with 'ESP32-BT'");
}

void loop() {
    // Forward from Serial to Bluetooth
    if (Serial.available()) {
        SerialBT.write(Serial.read());
    }
    
    // Forward from Bluetooth to Serial
    if (SerialBT.available()) {
        Serial.write(SerialBT.read());
    }
}
```

### BLE Server

```cpp
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
uint32_t value = 0;

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        Serial.println("Device connected");
    }
    
    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        Serial.println("Device disconnected");
    }
};

void setup() {
    Serial.begin(115200);
    
    // Create BLE Device
    BLEDevice::init("ESP32-BLE");
    
    // Create BLE Server
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    
    // Create BLE Service
    BLEService *pService = pServer->createService(SERVICE_UUID);
    
    // Create BLE Characteristic
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_READ |
        BLECharacteristic::PROPERTY_WRITE |
        BLECharacteristic::PROPERTY_NOTIFY
    );
    
    pCharacteristic->addDescriptor(new BLE2902());
    
    // Start service
    pService->start();
    
    // Start advertising
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->start();
    
    Serial.println("BLE Server started. Waiting for connections...");
}

void loop() {
    if (deviceConnected) {
        // Update and notify characteristic
        pCharacteristic->setValue((uint8_t*)&value, 4);
        pCharacteristic->notify();
        value++;
        delay(1000);
    }
}
```

### BLE Client (Scanner)

```cpp
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

BLEScan* pBLEScan;

class MyAdvertisedDeviceCallbacks: public BLEAdvertisedDeviceCallbacks {
    void onResult(BLEAdvertisedDevice advertisedDevice) {
        Serial.printf("Found device: %s\n", advertisedDevice.toString().c_str());
    }
};

void setup() {
    Serial.begin(115200);
    Serial.println("Starting BLE scan...");
    
    BLEDevice::init("");
    pBLEScan = BLEDevice::getScan();
    pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
    pBLEScan->setActiveScan(true);
}

void loop() {
    BLEScanResults foundDevices = pBLEScan->start(5, false);
    Serial.printf("Devices found: %d\n", foundDevices.getCount());
    pBLEScan->clearResults();
    delay(2000);
}
```

## Advanced Features

### Deep Sleep Mode

```cpp
#define uS_TO_S_FACTOR 1000000  // Conversion factor for microseconds to seconds
#define TIME_TO_SLEEP  30       // Sleep for 30 seconds

RTC_DATA_ATTR int bootCount = 0;  // Preserved in RTC memory

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    bootCount++;
    Serial.println("Boot number: " + String(bootCount));
    
    // Configure wake-up sources
    esp_sleep_enable_timer_wakeup(TIME_TO_SLEEP * uS_TO_S_FACTOR);
    
    // GPIO wake-up
    esp_sleep_enable_ext0_wakeup(GPIO_NUM_33, 1);  // Wake on HIGH
    
    Serial.println("Going to sleep for " + String(TIME_TO_SLEEP) + " seconds");
    Serial.flush();
    
    esp_deep_sleep_start();
}

void loop() {
    // Never reached
}
```

### Touch Wake-up

```cpp
#define THRESHOLD 40

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    // Configure touch wake-up
    touchAttachInterrupt(T0, callback, THRESHOLD);
    
    esp_sleep_enable_touchpad_wakeup();
    
    Serial.println("Going to sleep. Touch GPIO 4 to wake up.");
    delay(1000);
    
    esp_deep_sleep_start();
}

void callback() {
    // Empty
}

void loop() {
    // Never reached
}
```

### ADC with Calibration

```cpp
#include <esp_adc_cal.h>

#define ADC_PIN 34
#define DEFAULT_VREF 1100

esp_adc_cal_characteristics_t *adc_chars;

void setup() {
    Serial.begin(115200);
    
    // Configure ADC
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_11);
    
    // Characterize ADC
    adc_chars = (esp_adc_cal_characteristics_t*)calloc(1, sizeof(esp_adc_cal_characteristics_t));
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, DEFAULT_VREF, adc_chars);
}

void loop() {
    uint32_t adc_reading = 0;
    
    // Multisampling
    for (int i = 0; i < 64; i++) {
        adc_reading += adc1_get_raw(ADC1_CHANNEL_6);
    }
    adc_reading /= 64;
    
    // Convert to voltage
    uint32_t voltage = esp_adc_cal_raw_to_voltage(adc_reading, adc_chars);
    
    Serial.printf("Raw: %d, Voltage: %d mV\n", adc_reading, voltage);
    
    delay(1000);
}
```

### Over-The-Air (OTA) Updates

```cpp
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>

const char* ssid = "YourSSID";
const char* password = "YourPassword";

void setup() {
    Serial.begin(115200);
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("\nWiFi connected");
    Serial.println(WiFi.localIP());
    
    // Setup OTA
    ArduinoOTA.setHostname("esp32");
    ArduinoOTA.setPassword("admin");
    
    ArduinoOTA.onStart([]() {
        String type = (ArduinoOTA.getCommand() == U_FLASH) ? "sketch" : "filesystem";
        Serial.println("Start updating " + type);
    });
    
    ArduinoOTA.onEnd([]() {
        Serial.println("\nEnd");
    });
    
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
    });
    
    ArduinoOTA.onError([](ota_error_t error) {
        Serial.printf("Error[%u]: ", error);
        if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
        else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
        else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
        else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
        else if (error == OTA_END_ERROR) Serial.println("End Failed");
    });
    
    ArduinoOTA.begin();
    Serial.println("OTA Ready");
}

void loop() {
    ArduinoOTA.handle();
}
```

## Projects

### Project 1: WiFi Weather Station

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

const char* ssid = "YourSSID";
const char* password = "YourPassword";
const char* server = "http://api.thingspeak.com/update";
const char* apiKey = "YOUR_API_KEY";

void setup() {
    Serial.begin(115200);
    dht.begin();
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected!");
}

void loop() {
    float temp = dht.readTemperature();
    float humidity = dht.readHumidity();
    
    if (isnan(temp) || isnan(humidity)) {
        Serial.println("Failed to read sensor!");
        delay(2000);
        return;
    }
    
    Serial.printf("Temp: %.1f°C, Humidity: %.1f%%\n", temp, humidity);
    
    // Send to ThingSpeak
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        
        String url = String(server) + "?api_key=" + apiKey +
                     "&field1=" + String(temp) +
                     "&field2=" + String(humidity);
        
        http.begin(url);
        int httpCode = http.GET();
        
        if (httpCode > 0) {
            Serial.println("Data sent successfully");
        } else {
            Serial.println("Error sending data");
        }
        
        http.end();
    }
    
    delay(20000);  // ThingSpeak requires 15 second minimum
}
```

### Project 2: Bluetooth LED Controller

```cpp
#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

const int RED_PIN = 25;
const int GREEN_PIN = 26;
const int BLUE_PIN = 27;

void setup() {
    Serial.begin(115200);
    SerialBT.begin("ESP32-RGB");
    
    pinMode(RED_PIN, OUTPUT);
    pinMode(GREEN_PIN, OUTPUT);
    pinMode(BLUE_PIN, OUTPUT);
    
    Serial.println("Bluetooth RGB Controller Ready");
}

void loop() {
    if (SerialBT.available()) {
        String command = SerialBT.readStringUntil('\n');
        command.trim();
        
        if (command.startsWith("RGB")) {
            // Format: RGB,255,128,64
            int comma1 = command.indexOf(',');
            int comma2 = command.indexOf(',', comma1 + 1);
            int comma3 = command.indexOf(',', comma2 + 1);
            
            int r = command.substring(comma1 + 1, comma2).toInt();
            int g = command.substring(comma2 + 1, comma3).toInt();
            int b = command.substring(comma3 + 1).toInt();
            
            analogWrite(RED_PIN, r);
            analogWrite(GREEN_PIN, g);
            analogWrite(BLUE_PIN, b);
            
            SerialBT.printf("Set RGB to %d,%d,%d\n", r, g, b);
        } else if (command == "OFF") {
            analogWrite(RED_PIN, 0);
            analogWrite(GREEN_PIN, 0);
            analogWrite(BLUE_PIN, 0);
            SerialBT.println("LEDs OFF");
        }
    }
}
```

### Project 3: WiFi Smart Thermostat

```cpp
#include <WiFi.h>
#include <WebServer.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT11
#define RELAY_PIN 26

DHT dht(DHTPIN, DHTTYPE);
WebServer server(80);

const char* ssid = "YourSSID";
const char* password = "YourPassword";

float targetTemp = 25.0;
bool heatingOn = false;

void handleRoot() {
    float temp = dht.readTemperature();
    float humidity = dht.readHumidity();
    
    String html = "<!DOCTYPE html><html><head>";
    html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<style>body{font-family:Arial;text-align:center;margin:20px;}";
    html += ".button{padding:15px;margin:10px;font-size:20px;}</style></head>";
    html += "<body><h1>Smart Thermostat</h1>";
    html += "<p>Current: " + String(temp, 1) + "°C</p>";
    html += "<p>Humidity: " + String(humidity, 1) + "%</p>";
    html += "<p>Target: " + String(targetTemp, 1) + "°C</p>";
    html += "<p>Heating: " + String(heatingOn ? "ON" : "OFF") + "</p>";
    html += "<a href='/increase'><button class='button'>+1°C</button></a>";
    html += "<a href='/decrease'><button class='button'>-1°C</button></a>";
    html += "</body></html>";
    
    server.send(200, "text/html", html);
}

void handleIncrease() {
    targetTemp += 1.0;
    server.sendHeader("Location", "/");
    server.send(303);
}

void handleDecrease() {
    targetTemp -= 1.0;
    server.sendHeader("Location", "/");
    server.send(303);
}

void setup() {
    Serial.begin(115200);
    dht.begin();
    pinMode(RELAY_PIN, OUTPUT);
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }
    
    Serial.println("\nConnected!");
    Serial.println(WiFi.localIP());
    
    server.on("/", handleRoot);
    server.on("/increase", handleIncrease);
    server.on("/decrease", handleDecrease);
    server.begin();
}

void loop() {
    server.handleClient();
    
    static unsigned long lastCheck = 0;
    if (millis() - lastCheck > 5000) {
        lastCheck = millis();
        
        float temp = dht.readTemperature();
        
        if (!isnan(temp)) {
            if (temp < targetTemp - 0.5) {
                heatingOn = true;
                digitalWrite(RELAY_PIN, HIGH);
            } else if (temp > targetTemp + 0.5) {
                heatingOn = false;
                digitalWrite(RELAY_PIN, LOW);
            }
        }
    }
}
```

## Best Practices

1. **Power Management**: Use deep sleep for battery-powered projects
2. **WiFi**: Disconnect when not needed to save power
3. **Watchdog**: Enable watchdog timer for reliability
4. **OTA Updates**: Implement for remote firmware updates
5. **Error Handling**: Always check WiFi connection status
6. **Security**: Use HTTPS and encrypted connections
7. **Memory**: Monitor heap usage with `ESP.getFreeHeap()`

## Troubleshooting

### Common Issues

**Boot Loop:**
- Check strapping pins (0, 2, 5, 12, 15)
- Ensure stable power supply (500mA minimum)
- Add 10µF capacitor on EN pin

**WiFi Not Connecting:**
- Check SSID and password
- Verify 2.4GHz network (ESP32 doesn't support 5GHz)
- Move closer to router

**Upload Failed:**
- Hold BOOT button during upload
- Check correct COM port selected
- Try lower baud rate (115200)

**Brown-out Detector:**
- Use external 5V power supply
- Add bulk capacitor (100-1000µF)

## Resources

- **Espressif Documentation**: https://docs.espressif.com/
- **ESP32 Arduino Core**: https://github.com/espressif/arduino-esp32
- **ESP-IDF Programming Guide**: https://docs.espressif.com/projects/esp-idf/
- **Forum**: https://www.esp32.com/

## See Also

- [Arduino Programming](arduino.md)
- [WiFi Projects](../networking/wifi.md)
- [MQTT Protocol](../protocols/mqtt.md)
- [BLE Communication](bluetooth.md)
