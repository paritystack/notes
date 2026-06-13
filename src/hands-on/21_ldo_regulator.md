# 21 · LDO regulator — measure dropout & heat

## Overview

Everything so far ran off the Nano's regulated 5 V or USB. A real
[portable device](README.md) starts from a messy source (a battery that sags from 4.2 V to
3.0 V) and must produce a *stable* rail. A **linear regulator** (LDO — low-dropout) is the
simplest way: feed it a higher voltage in, get a fixed voltage out. You'll build one, load
it, and measure its two defining costs — **dropout** (how much headroom it needs) and
**heat** (wasted as `(Vin−Vout)×I`). This is the power front-end of your capstone.

```
   Vin ──┤IN   OUT├── Vout (fixed, e.g. 3.3V) ── load
         │  LDO    │
   Cin ===  GND  === Cout
         │         │
   GND ──┴────┴────┘   decoupling caps per the datasheet
```

## What you'll need

From **Stage B/C**: an LDO (MCP1700-3302 for low quiescent current, or an AMS1117-3.3 to
start), its input/output [decoupling caps](../electronics/capacitors.md) (~1 µF, check the
[datasheet](20_read_a_datasheet.md)), a load (resistor or the Nano), a bench supply or
battery, and the multimeter.

## The build

1. Wire **Vin → IN**, **OUT → Vout**, **GND** common, with the datasheet's
   [caps](../electronics/capacitors.md) on input and output.
2. Feed, say, **5 V in**, measure **Vout** → ~3.3 V.
3. **Find dropout:** slowly lower Vin toward 3.3 V while watching Vout. At some point Vout
   starts following Vin down — that gap is the **dropout voltage**. An AMS1117 needs ~1.1 V
   headroom (so it can't make 3.3 V from a 3.5 V battery!); an MCP1700 needs ~0.2 V.
4. **Feel the heat:** load it (e.g. 100 mA) with a big Vin–Vout gap and the regulator warms;
   compute `P = (Vin−Vout) × I` and confirm.

```
   Efficiency of a linear reg = Vout / Vin.
   5V → 3.3V at 100 mA: wastes (5−3.3)×0.1 = 0.17 W as heat (66% efficient).
   That heat (and dropout) is why switchers exist — but LDOs are simpler & quiet.
```

## It works when…

- [ ] Vout holds steady at the rated voltage across a range of Vin above dropout.
- [ ] You measured the dropout voltage and saw Vout collapse below it.
- [ ] You computed the dissipated power and felt the corresponding heat.

## What's happening

A linear regulator is essentially a [transistor](../electronics/transistors_mosfet.md) acting
as a variable resistor in series with the load, continuously adjusted to hold the output
constant — so it *burns* the voltage difference as heat. **Dropout** is the minimum Vin−Vout
at which it can still regulate; choosing a *low*-dropout part matters enormously when running
from a [LiPo](22_lipo_charging.md) that sags to 3.3 V. Efficiency is just Vout/Vin, which is
poor for big gaps — the trade-off you accept for simplicity and clean, noise-free output.
For your battery node you'll pick an LDO whose dropout and quiescent current suit a sagging
cell. See [Power Supplies](../electronics/power_supplies.md) for the switcher alternative.

## Pitfalls

- **Vin below dropout** — feeding 3.6 V into an AMS1117 (1.1 V dropout) can't make 3.3 V; output droops. Pick an LDO whose dropout fits your battery's *lowest* voltage.
- **Missing decoupling caps** — LDOs can oscillate without the datasheet's input/output caps. Always fit them, the specified type (some need low-ESR).
- **Ignoring quiescent current** — the regulator's own idle draw (Iq) drains the battery 24/7; an AMS1117 wastes ~5 mA, an MCP1700 ~1.6 µA. Huge for [low-power](24_deep_sleep.md) designs.
- **Underestimating heat** — a big Vin−Vout × I can exceed the package's rating; check thermals per [rung 20](20_read_a_datasheet.md).

## Where this connects

- [Power Supplies](../electronics/power_supplies.md) — linear vs switching regulators, regulation in depth
- [Capacitors](../electronics/capacitors.md) — why input/output decoupling is mandatory
- [22 · LiPo charging](22_lipo_charging.md) — the battery this regulator will run from
- **Previous:** [20 · Read a datasheet](20_read_a_datasheet.md) · **Next:** [22 · LiPo charging](22_lipo_charging.md)
