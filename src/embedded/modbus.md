# Modbus & RS-485

## Overview

RS-485 and Modbus are the workhorse pairing of industrial communication: **RS-485** is the *physical layer* — a rugged, differential, multidrop bus that runs reliably over hundreds of metres of noisy factory cable — and **Modbus** is the simple *application protocol* that usually rides on top of it. Where [UART](uart.md) gives you point-to-point single-ended TX/RX good for a metre, RS-485 turns the same UART byte stream into a 32+ device shared bus with strong noise immunity, much like [CAN](can.md) but simpler and master/slave instead of multi-master. If you are reading a PLC, a VFD, an energy meter, or an industrial sensor, it is almost certainly Modbus over RS-485.

```
   Master (MCU)                          Slaves (PLC, sensor, VFD...)
   ┌─────────┐  A ───────┳───────────┳───────────┳───── A
   │ UART +  │           │           │           │
   │ RS-485  │  B ───────┻───────────┻───────────┻───── B
   │ xceiver │  twisted pair, 120Ω termination at both ends
   └─────────┘     +bias resistors to define idle state
   DE/RE ── direction control (half-duplex: one talker at a time)
```

## RS-485 the Physical Layer

RS-485 signals a bit as the **voltage difference between two wires (A and B)**, not relative to ground. A noise spike couples into both wires equally and cancels at the differential receiver — this common-mode rejection is why it survives industrial environments that destroy single-ended [UART](uart.md).

| Property | RS-485 | RS-232 | TTL UART |
|----------|--------|--------|----------|
| Signaling | Differential | Single-ended ±12 V | Single-ended 3.3/5 V |
| Distance | ~1200 m | ~15 m | ~1 m |
| Nodes | 32+ (up to 256 w/ low-load xceivers) | 1:1 | 1:1 |
| Topology | Multidrop bus | Point-to-point | Point-to-point |
| Duplex | Usually half (2-wire) | Full | Full |

Key wiring rules:
- **120 Ω termination at *both* physical ends** of the bus (matched to cable impedance) to kill reflections — see [Signal Integrity](signal_integrity.md).
- **Bias (fail-safe) resistors** pull A/B to a defined difference when no one is driving, so idle reads as a clean "1" instead of floating noise.
- **Daisy-chain, not star** — stubs cause reflections.

### The Direction-Control Gotcha

2-wire RS-485 is **half-duplex**: only one transceiver may drive the bus at a time. The MCU controls a **DE (Driver Enable)** pin:

```
   DE high → MCU drives bus (transmit)
   DE low  → MCU releases bus, listens (receive)
```

You must assert DE *before* the first bit and de-assert it *only after the last stop bit has physically left the wire* — not when the UART data register empties. De-asserting early truncates the last byte; late, you collide with the slave's reply. The robust trick is to drive DE from the UART's **Transmission Complete (TC)** interrupt, not the TX-empty one. Many modern MCUs have a hardware "RS-485 DE" mode that toggles the pin automatically with a programmable guard time.

## Modbus the Protocol

Modbus is deliberately tiny: **one master, many slaves, request/response, no spontaneous slave traffic**. Each slave has a 1-byte address (1–247); the master polls.

### Data Model

Everything is one of four tables, addressed by a 16-bit index:

| Table | Access | Size | Typical use |
|-------|--------|------|-------------|
| **Coils** | R/W | 1 bit | Outputs, relays |
| **Discrete Inputs** | R | 1 bit | Digital inputs |
| **Input Registers** | R | 16 bit | Sensor readings |
| **Holding Registers** | R/W | 16 bit | Config, setpoints |

### Common Function Codes

| FC | Operation |
|----|-----------|
| 0x01 | Read Coils |
| 0x02 | Read Discrete Inputs |
| 0x03 | Read Holding Registers |
| 0x04 | Read Input Registers |
| 0x05 | Write Single Coil |
| 0x06 | Write Single Register |
| 0x0F | Write Multiple Coils |
| 0x10 | Write Multiple Registers |

An error reply sets the high bit of the function code (e.g. 0x83) and returns an **exception code** (Illegal Function, Illegal Data Address, etc.).

### Transports: RTU vs ASCII vs TCP

```
Modbus RTU (binary, over RS-485 — the common case):
┌──────┬──────┬─────────────────┬──────────┐
│ Addr │  FC  │      Data       │  CRC-16  │
│ 1 B  │ 1 B  │      N bytes    │   2 B    │
└──────┴──────┴─────────────────┴──────────┘
  frame delimited by ≥3.5 character-times of silence

Modbus ASCII: hex-encoded, ':' start, CRLF end, LRC checksum (legacy)

Modbus TCP: drop CRC, prepend 7-byte MBAP header (transaction/proto/len/unit),
            run over TCP/502 — see ../networking/tcp.md
```

The RTU framing rule is the subtle one: frames are separated by **silence of at least 3.5 character times**, and a gap of more than 1.5 character times *within* a frame is an error. At 9600 baud that's ~4 ms of idle to mark a frame boundary — typically detected with the UART's idle-line interrupt or a [timer](timers.md) rather than parsing lengths.

```c
// RTU frame end detection via idle timeout
void UART_IRQHandler(void) {
    if (UART->ISR & IDLE) {        // line idle = inter-frame gap
        frame_len = rx_idx;
        if (modbus_crc16(rx_buf, frame_len - 2) == get_crc(rx_buf))
            process_request(rx_buf, frame_len);
        rx_idx = 0;
    }
}
```

## Where this connects

- [UART](uart.md) — Modbus RTU is a framed UART byte stream; RS-485 is the UART's physical layer.
- [CAN](can.md) — the other dominant industrial bus; multi-master and message-oriented vs Modbus's master/slave polling.
- [Signal Integrity](signal_integrity.md) — termination, biasing, and ground-potential differences are RS-485's main failure modes.
- [Timers](timers.md) / [Interrupts](interrupts.md) — the 3.5-char inter-frame gap is measured with an idle-line IRQ or timer.
- [Ethernet](ethernet.md) / [TCP](../networking/tcp.md) — Modbus TCP carries the same data model over IP on port 502.

## Pitfalls

1. **DE de-asserted too early.** Dropping Driver Enable when the UART data register empties (not when transmission *completes*) chops the last byte. Drive DE from the TC interrupt or hardware RS-485 mode.
2. **Missing or wrong termination.** No 120 Ω at both ends → reflections and corrupted bytes at higher baud/longer cable. One resistor isn't enough; you need both ends.
3. **No fail-safe bias.** Floating idle bus reads as random framing errors. Add bias resistors (or use a fail-safe transceiver).
4. **A/B (D+/D−) swapped.** Extremely common; the bus simply doesn't work. Vendors disagree on A/B labeling — try swapping.
5. **Inter-frame gap too short.** Parsing RTU by fixed length instead of the 3.5-char silence merges two frames or splits one. Use idle detection.
6. **Register addressing off-by-one.** Modbus docs often use 1-based "4xxxx" PLC notation while the wire uses 0-based addresses — holding register 40001 is wire address 0.
7. **Byte/word order of 32-bit values.** Modbus registers are 16-bit; a 32-bit float spans two registers and vendors disagree on word order (big vs little-endian, swapped). Confirm against the device.
8. **Stub/star topology.** RS-485 wants a daisy chain; long stubs reflect.

## See Also

- [UART](uart.md) — the underlying serial layer
- [CAN](can.md) — alternative industrial bus
- [Signal Integrity](signal_integrity.md) — termination and biasing
- [Ethernet](ethernet.md) — for Modbus TCP
