# IoT Protocols: MQTT & CoAP

## Overview

Two dominant application-layer protocols built for **constrained devices** (low CPU, low memory, intermittent battery-powered radios):

- **MQTT** — broker-mediated publish/subscribe over TCP. Best for telemetry from many sensors to a central cloud, asymmetric command/control.
- **CoAP** — REST-like request/response over UDP. Best for direct device-to-device interaction and HTTP-style APIs that need to fit in tiny packets.

Both are designed for situations where HTTP+TLS+JSON is too heavyweight.

---

# MQTT (Message Queuing Telemetry Transport)

## What MQTT Is

A pub/sub messaging protocol. Clients (devices) don't talk to each other directly — they **publish** messages to **topics** on a **broker**, and other clients **subscribe** to topics to receive matching messages.

```
       publish("home/livingroom/temp", "22.5")
   ┌─────────────────────────────────────────┐
   │                                         ▼
[Sensor]                                  [Broker]
                                             │  forwards to subscribers
                                             │
                                             ▼
                                         [Dashboard]
                                         [Logger]
                                         [Alerting]
```

Designed in 1999 by IBM for SCADA over satellite links. Now an OASIS standard (MQTT 3.1.1, MQTT 5).

## Why MQTT

```
✓ Tiny protocol overhead (2-byte fixed header minimum)
✓ Persistent TCP connection → no repeated handshakes
✓ Broker handles fan-out (sensor doesn't know subscribers)
✓ Last Will and Testament for crash detection
✓ Retained messages for "current state" semantics
✓ QoS levels for reliability vs throughput tradeoff
✓ Wide ecosystem (Mosquitto, EMQX, HiveMQ, AWS IoT, etc.)
```

## Connection Lifecycle

```
Client                                Broker
  |                                     |
  |--- TCP SYN ------------------------>|
  |<-- SYN-ACK -------------------------|
  |                                     |
  |--- CONNECT (clientId, will, auth) ->|
  |<-- CONNACK (session present, rc) ---|
  |                                     |
  |--- SUBSCRIBE (topic, qos) --------->|
  |<-- SUBACK -------------------------|
  |                                     |
  |--- PUBLISH (topic, payload) ------->|
  |                                     |
  |<-- PUBLISH (from another client) ---|
  |                                     |
  |--- PINGREQ (keepalive) ----------->|
  |<-- PINGRESP -----------------------|
  |                                     |
  |--- DISCONNECT --------------------->|
```

### CONNECT Packet

```
Client identifier (unique per connection)
Username / password (optional)
Keep-Alive interval (seconds, 0 = disabled)
Clean session flag
  - true:  fresh session, no persisted subscriptions
  - false: broker resumes previous session
Last Will and Testament (optional):
  topic, payload, QoS, retain
  → broker publishes this if client disconnects unexpectedly
```

## Topics

Topics are hierarchical, slash-separated:

```
home/livingroom/temperature
home/kitchen/light/state
factory/line1/conveyor/motor/rpm
vehicle/abc123/telemetry/speed
```

### Wildcards (for subscriptions only)

```
+   single-level wildcard
    home/+/temperature
    matches: home/livingroom/temperature, home/kitchen/temperature
    not:     home/livingroom/sensor/temperature

#   multi-level wildcard (must be last)
    home/#
    matches: home/livingroom/temperature, home/kitchen/light/state
```

### Best Practices

```
✓ Use hierarchy: device-type / device-id / measurement
✓ Lowercase
✓ Avoid leading slash
✓ Use $ for system topics (broker internal)
✗ Don't put dynamic IDs in subscription wildcards if you can avoid it
✗ Don't make subscribers parse topic strings to extract data — use payload
```

## QoS Levels

QoS is **per-message**, negotiated as min(publisher QoS, subscriber QoS).

### QoS 0 — At most once ("fire and forget")

```
Client    Broker
  |--PUBLISH-->|     no ack, may be lost on link failure
```

Use: high-frequency telemetry where occasional loss is fine.

### QoS 1 — At least once

```
Client    Broker
  |--PUBLISH-->|     stored at broker
  |<--PUBACK---|     ack
```

If PUBACK lost, client retransmits → broker may deliver twice. Use idempotent payloads.

### QoS 2 — Exactly once

```
Client    Broker
  |--PUBLISH--->|
  |<--PUBREC----|
  |--PUBREL--->|
  |<--PUBCOMP---|
```

Four-message handshake guarantees one and only one delivery. Highest reliability, highest overhead.

### Choosing QoS

```
QoS 0: sensor data at 10 Hz (loss tolerable)
QoS 1: home automation commands (must arrive, dedup is easy)
QoS 2: billing events, irreversible actions
```

## Retained Messages

A message published with `retain=true` is stored by the broker. Any future subscriber to that topic immediately receives the retained message.

```
[Device boots]
  publish("home/livingroom/light/state", "on", retain=true)

[Dashboard later connects]
  subscribe("home/livingroom/light/state")
  → immediately receives "on" (the retained value)
```

Used for "current state" semantics — perfect for IoT dashboards that need to know the latest value without waiting for the next sensor update.

Only **one** retained message per topic (latest wins). Empty payload + retain=true → delete retained message.

## Last Will and Testament (LWT)

The broker publishes a pre-arranged message when a client disconnects ungracefully (TCP RST, keepalive timeout).

```
On CONNECT:
  Will topic: "home/livingroom/sensor/status"
  Will payload: "offline"
  Will QoS: 1
  Will retain: true

If client disconnects unexpectedly:
  broker publishes "offline" on behalf of the client

If client disconnects gracefully (sends DISCONNECT):
  will is NOT published
```

Pattern: device publishes `"online"` (retained) on connect, sets will to `"offline"` (retained).

## Keep-Alive

```
Client → CONNECT (keepalive = 60s)

Within 60s, if no other traffic:
  client → PINGREQ
  broker → PINGRESP

If broker sees no traffic for 1.5 × keepalive:
  closes connection, fires will
```

## Sample Code

### Python (paho-mqtt)

```python
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("connected", rc)
    client.subscribe("home/+/temperature", qos=1)

def on_message(client, userdata, msg):
    print(msg.topic, msg.payload.decode())

client = mqtt.Client(client_id="dashboard-1", clean_session=False)
client.username_pw_set("user", "pass")
client.will_set("dashboard/status", "offline", qos=1, retain=True)
client.tls_set("ca.crt")
client.on_connect = on_connect
client.on_message = on_message

client.connect("broker.example.com", 8883, keepalive=60)
client.publish("dashboard/status", "online", qos=1, retain=True)
client.loop_forever()
```

### Publishing telemetry

```python
import time, json
while True:
    payload = json.dumps({"temp": 22.5, "ts": time.time()})
    client.publish("home/livingroom/temperature", payload, qos=0)
    time.sleep(1)
```

## MQTT 5 Improvements (over 3.1.1)

```
✓ Reason codes (specific error info on every ack)
✓ User properties (key-value metadata in any packet)
✓ Topic aliases (numeric shorthand to save bandwidth)
✓ Message expiry intervals
✓ Shared subscriptions ($share/group/topic — load balance among subscribers)
✓ Request/response patterns (responseTopic + correlationData)
✓ Server-side disconnect with reason
✓ Flow control (receive maximum)
```

## Brokers

| Broker | Notes |
|--------|-------|
| **Mosquitto** | Reference open-source, simple |
| **EMQX** | Massive scale (millions of connections), clustering |
| **HiveMQ** | Enterprise, MQTT 5, distributed |
| **AWS IoT Core** | Managed, integrated with AWS |
| **Azure IoT Hub** | Managed Microsoft |
| **VerneMQ** | Erlang-based, distributed |

## Tooling

```bash
# Subscribe
mosquitto_sub -h broker -t "home/#" -v
mosquitto_sub -h broker -t "home/+/temperature" -q 1

# Publish
mosquitto_pub -h broker -t home/livingroom/temp -m "22.5"
mosquitto_pub -h broker -t home/livingroom/light -m "on" --retain

# TLS
mosquitto_sub -h broker -p 8883 --cafile ca.crt -t '#'
```

GUI clients: **MQTT Explorer** (highly recommended for debugging), MQTTX.

## Security

```
✓ TLS (port 8883 instead of 1883)
✓ Username/password authentication
✓ Client certificates (mTLS)
✓ ACLs per topic (read/write per user)
✓ Rate limiting at broker
✗ Never expose unauthenticated 1883 to internet (Shodan has bots scanning constantly)
✗ Don't put secrets in topic names (visible in metrics, logs)
```

## MQTT Ports

| Port | Use |
|------|-----|
| 1883 | MQTT over TCP (plaintext) |
| 8883 | MQTT over TLS |
| 80 / 443 | MQTT over WebSocket (for browsers) |
| 8884 | MQTT over TLS with client cert |

---

# CoAP (Constrained Application Protocol)

## What CoAP Is

A REST-like protocol over **UDP** designed for devices that may have only a few KB of RAM and run on batteries. Looks like HTTP semantically (GET, POST, PUT, DELETE) but encoded as 4-byte binary headers instead of text. RFC 7252.

```
Client                              Server
  |--- GET coap://device/sensor --->|
  |<--- 2.05 Content "22.5" --------|
```

## Why CoAP

```
✓ Tiny: 4-byte header + URI option
✓ UDP: no connection state, no SYN cost
✓ Multicast support (CoRE Resource Directory, group requests)
✓ Asynchronous (observable resources push updates)
✓ Maps cleanly to HTTP via proxies
✓ Discovery via /.well-known/core
```

## Message Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Ver| T |  TKL  |      Code     |          Message ID           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Token (if any, TKL bytes) ...
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Options (if any) ...
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|1 1 1 1 1 1 1 1|    Payload (if any) ...
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Field | Bits | Notes |
|-------|------|-------|
| Ver | 2 | 01 |
| T (type) | 2 | CON, NON, ACK, RST |
| TKL | 4 | Token length (0-8) |
| Code | 8 | 0.01-0.04 methods / 2.xx-5.xx responses |
| Message ID | 16 | Match request to response |
| Token | 0-8B | Match request to response across multiple exchanges |
| Options | varies | URI, content-format, etc. |
| Payload | varies | After 0xFF marker |

## Message Types

```
CON  Confirmable    requires ACK or retransmit
NON  Non-confirmable fire and forget
ACK  Acknowledgement (often piggybacks response)
RST  Reset (error, can't process)
```

### Reliable CON Exchange

```
Client → CON GET (MID=42, Token=T1)
Server → ACK 2.05 Content (MID=42, Token=T1, payload)
```

If ACK lost, client retransmits CON after timeout (exponential backoff). Server uses MID dedup cache.

### Separate Response (long processing)

```
Client → CON GET (MID=42, Token=T1)
Server → ACK empty (MID=42)                  ← "got it, working on it"
... later ...
Server → CON 2.05 Content (MID=99, Token=T1) ← carries response
Client → ACK empty (MID=99)
```

## Methods & Codes

### Methods

```
0.01 GET
0.02 POST
0.03 PUT
0.04 DELETE
0.05 FETCH (RFC 8132)
0.06 PATCH
0.07 iPATCH
```

### Response Codes (CoAP style "class.detail")

```
2.01 Created
2.02 Deleted
2.03 Valid
2.04 Changed
2.05 Content

4.00 Bad Request
4.01 Unauthorized
4.04 Not Found
4.05 Method Not Allowed
4.15 Unsupported Content-Format

5.00 Internal Server Error
5.02 Bad Gateway
5.04 Gateway Timeout
```

## URIs

```
coap://device.local/sensors/temperature
coaps://device.local:5684/sensors/temperature       (DTLS)

Standard port: 5683
DTLS port:     5684
```

## Observe (RFC 7641)

Subscribe to a resource — server pushes updates when it changes. CoAP's "websocket equivalent" for IoT.

```
Client → GET /temp + Observe option
Server → 2.05 Content + Observe (sequence 1) "22.5"
                                              (later)
Server → 2.05 Content + Observe (sequence 2) "22.7"
                                              (later)
Server → 2.05 Content + Observe (sequence 3) "22.9"

Client cancels: GET /temp + Observe=1 (deregister)
```

## Block-wise Transfer

CoAP payloads should fit in a single UDP datagram (~1024 bytes). For larger transfers, use block options to fragment:

```
GET /firmware → 2.05 Content Block2(0/1/256)    (256-byte chunks)
GET /firmware Block2(1/1/256) → 2.05 Block2(1/1/256)
GET /firmware Block2(2/0/256) → 2.05 Block2(2/0/256)  ← last block
```

Used for OTA firmware updates of microcontrollers.

## Resource Discovery

Every CoAP server exposes `/.well-known/core`:

```
GET /.well-known/core
→ </sensors/temp>;rt="temperature";ct=0,
  </sensors/humid>;rt="humidity";ct=0,
  </actuators/relay>;rt="switch"
```

The CoRE Resource Directory (RFC 9176) is a registry where devices announce their resources to a central catalog.

## Sample Code

### Python (aiocoap)

```python
import asyncio
from aiocoap import *

async def main():
    context = await Context.create_client_context()
    request = Message(code=GET, uri='coap://device.local/sensors/temp')
    response = await context.request(request).response
    print(response.code, response.payload.decode())

asyncio.run(main())
```

### CoAP Server (aiocoap)

```python
import aiocoap.resource as resource
import aiocoap
import asyncio

class TempResource(resource.Resource):
    async def render_get(self, request):
        return aiocoap.Message(payload=b'22.5', content_format=0)

async def main():
    root = resource.Site()
    root.add_resource(['sensors', 'temp'], TempResource())
    await aiocoap.Context.create_server_context(root, bind=('::', 5683))
    await asyncio.get_running_loop().create_future()

asyncio.run(main())
```

## DTLS Security

CoAP over UDP uses **DTLS** (TLS adapted for UDP). PSK (pre-shared key) is common because device certificates are expensive to manage.

```
Cipher: TLS_PSK_WITH_AES_128_CCM_8 is the IoT default
        (small footprint, hardware AES support)

PSK identity → PSK lookup → mutual auth + encryption
```

OSCORE (RFC 8613) is an alternative: end-to-end CoAP encryption that works across proxies.

## CoAP-HTTP Proxying

A proxy can translate between CoAP and HTTP:

```
HTTP client → GET http://proxy/coap?uri=coap://device/temp
Proxy → GET coap://device/temp
Proxy ← 2.05 Content "22.5"
HTTP client ← 200 OK "22.5"
```

Lets cloud services talk to constrained devices without learning a new protocol.

---

# MQTT vs CoAP vs HTTP

| Feature | MQTT | CoAP | HTTP/1.1 |
|---------|------|------|----------|
| **Transport** | TCP | UDP | TCP |
| **Model** | Pub/Sub (broker) | REST (client/server) | REST (client/server) |
| **Header overhead** | 2-byte minimum | 4-byte minimum | ~100s of bytes |
| **Reliability** | TCP + QoS levels | Confirmable messages | TCP |
| **Push** | Native pub/sub | Observe extension | Polling / SSE |
| **Discovery** | Topic conventions | /.well-known/core | OpenAPI / out of band |
| **Encryption** | TLS | DTLS | TLS |
| **NAT-friendly** | Persistent TCP works through NAT | UDP needs hole punching | Yes |
| **Multicast** | No | Yes | No |
| **Use case** | Many devices → cloud | Direct device interaction | Web/mobile clients |

## When to Use Which

```
MQTT — fan-out from many devices to many subscribers
       e.g., factory floor telemetry, home automation, mobile push

CoAP — direct request/response to constrained devices on a local network
       e.g., asking a thermostat for its current temperature

HTTP — when devices are powerful enough and you want simplicity
       e.g., REST APIs on a Raspberry Pi
```

Often combined: a fleet of CoAP sensors → CoAP-MQTT bridge → MQTT broker → cloud subscribers.

## Other IoT Protocols (Brief Mentions)

```
AMQP        — heavyweight enterprise messaging (RabbitMQ); too big for sensors
DDS         — high-performance pub/sub for robotics and defense
LwM2M       — device management built on CoAP (firmware, monitoring, provisioning)
Zigbee/Z-Wave — mesh radio protocols, not IP
LoRaWAN     — long-range low-power radio (not really application-layer)
Matter      — modern IP-based smart-home standard, uses MDNS + UDP + TLS
SigFox      — ultra-low-power proprietary
NB-IoT      — cellular IoT
```

## ELI10

**MQTT** is like a school bulletin board (the broker). Sensors are kids who pin notes ("temperature is 22.5°") onto specific cork-boards (topics like `home/livingroom/temp`). Other kids who care about that topic visit and read whatever's there. The bulletin board keeps **retained** notes pinned so new kids see the current value. If a kid stops showing up, the principal posts their pre-written "I'm offline" note (Last Will).

**CoAP** is like walking up to a friend's house and **knocking** (GET). They open the door and hand you something (2.05 Content). It's HTTP-shaped, but everything is squeezed into postcards instead of envelopes so it fits on a tiny mailman robot's back (UDP, 4-byte headers). If you want updates whenever they change something, you say "Observe" and they keep tossing postcards out the window whenever the answer changes.

The big difference: **MQTT needs a building (broker)** in the middle; **CoAP devices talk directly** to each other.

## Further Resources

- [MQTT 5.0 OASIS Standard](https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html)
- [Eclipse Mosquitto](https://mosquitto.org/)
- [HiveMQ MQTT Essentials (great free series)](https://www.hivemq.com/mqtt-essentials/)
- [RFC 7252 - CoAP](https://tools.ietf.org/html/rfc7252)
- [RFC 7641 - CoAP Observe](https://tools.ietf.org/html/rfc7641)
- [aiocoap (Python CoAP)](https://aiocoap.readthedocs.io/)
- [libcoap (C CoAP)](https://libcoap.net/)
