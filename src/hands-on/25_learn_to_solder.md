# 25 · Learn to solder (practice kit)

## Overview

Everything so far lived on a [breadboard](../electronics/prototyping.md) — fast to build,
but loose, unreliable, and temporary. To make anything permanent (and to populate your
[PCB](27_kicad_schematic_capstone.md) later) you must solder. This rung is pure skill-
building: melt metal, make a few dozen joints, and learn what a *good* one looks and feels
like — before you risk your real circuit on it. Soldering is muscle memory; do it on scrap
first.

```
   Anatomy of a good joint:

      │ lead
      │
     ╱█╲   ← shiny, smooth, concave "volcano" cone
    ╱███╲     wetting both the pad and the lead
   ──pad──
```

## What you'll need

From **Stage C**: a temperature-controlled iron (Pinecil V2 or similar) + USB-C PD supply,
flux-core solder, a brass tip-cleaner, a cheap **soldering practice kit** (a small PCB with
LEDs/resistors to assemble), helping hands, and good ventilation.

## The build

1. **Tin the tip:** heat the iron to ~350 °C, wipe on the brass wool, melt a little solder on
   the tip so it's shiny. Re-tin often — a dull tip won't transfer heat.
2. **Make a joint** (the [datasheet/prototyping](../electronics/prototyping.md) procedure):
   - Heat the **pad and the lead together** for 2–3 s.
   - Feed solder into the *joint* (not the iron tip) — it flows toward heat.
   - Remove solder, then iron. Don't move the joint for ~3 s while it solidifies.
3. **Inspect every joint:**

```
   GOOD: shiny, smooth, concave cone, wets pad + lead
   COLD: dull, grainy, blobby     → reheat (joint moved while cooling)
   BLOB/BRIDGE: too much, touches  → remove with wick/braid
   BALL: solder didn't flow        → pad/lead not hot enough; add flux, more heat
```

4. Build the whole practice kit. By the last few joints they should look consistent. Use
   **solder wick** to practise *removing* solder too — you'll need it to fix mistakes.

## It works when…

- [ ] Your joints are consistently shiny, smooth cones (not dull blobs).
- [ ] You can clear a solder bridge between two pads with wick.
- [ ] The practice kit works (its LEDs light) — proving your joints conduct.

## What's happening

Solder bonds by **wetting** — it flows onto and alloys with clean, hot copper, which is why
you heat the *joint*, not the solder, and why **flux** (in the solder core) is essential: it
strips oxide so the metal can wet. A **cold joint** happens when the joint moved or wasn't hot
enough, leaving a weak, high-resistance, dull connection — the source of countless
intermittent faults. Temperature control and a clean, tinned tip do most of the work; the
rest is the rhythm of heat-feed-remove. These are the exact joints you'll make on your
[capstone PCB](29_order_assemble.md), where a single cold joint can mean a dead board.

## Pitfalls

- **Heating the solder, not the joint** — gives a blob sitting on a cold pad (a cold joint). Heat the pad+lead first.
- **Dirty or dry tip** — a dull, oxidised tip won't transfer heat; keep it tinned and wiped.
- **Too much solder / bridges** — especially on close pads; less is more, and wick fixes excess.
- **No ventilation** — flux fumes are an irritant; work in fresh air or with a fan. Wash hands (leaded solder) afterward.
- **Moving the joint while cooling** — produces a cold joint. Hold steady until it sets.

## Where this connects

- [Prototyping & Test Equipment](../electronics/prototyping.md) — soldering technique, tools, and joint inspection
- [26 · Perfboard build](26_perfboard_build.md) — your first *real* soldered circuit
- [29 · Order & assemble](29_order_assemble.md) — soldering the capstone PCB
- **Previous:** [24 · Deep-sleep the AVR](24_deep_sleep.md) · **Next:** [26 · Perfboard build](26_perfboard_build.md)
