# Electronics

Electronics is the study of how to control the flow of electric charge to make useful things happen — from a blinking LED to a smartphone processor. Every digital device you use runs on the principles here.

This section starts from absolute zero. No prior knowledge assumed. Every concept builds on the previous one, and every explanation starts with an intuition or analogy before going near a formula.

## Learning path

Follow the pages in this order — each one uses concepts from those above it.

| Step | Topic | What you learn |
|------|-------|---------------|
| 1 | [Charge & Current](charge_current.md) | What electricity actually *is* |
| 2 | [Voltage](voltage.md) | What *pushes* the electricity |
| 3 | [Resistance & Ohm's Law](resistance.md) | What *fights* the flow, and V=IR |
| 4 | [Power & Energy](power.md) | How much work the flow does |
| 5 | [Circuits](circuits.md) | Series/parallel paths, Kirchhoff's laws |
| 6 | [AC Signals & Impedance](ac_signals.md) | Sine waves, RMS, reactance, impedance |
| 7 | [Capacitors](capacitors.md) | Storing charge like a tiny battery |
| 8 | [Inductors](inductors.md) | Storing energy in a magnetic field |
| 9 | [Transformers](transformers.md) | Changing AC voltage and isolating circuits |
| 10 | [Diodes](diodes.md) | One-way valves for current |
| 11 | [BJT Transistors](transistors_bjt.md) | Current-controlled switches and amplifiers |
| 12 | [MOSFET Transistors](transistors_mosfet.md) | Voltage-controlled switches |
| 13 | [Switches, Relays & Electromechanical](switches_relays.md) | Making and breaking connections; driving coils |
| 14 | [Op-Amps](op_amps.md) | Amplifying differences |
| 15 | [Logic Gates](logic_gates.md) | The building blocks of digital logic |
| 16 | [Filters](filters.md) | Frequency-selective circuits |
| 17 | [Oscillators & the 555 Timer](oscillators.md) | Circuits that generate their own signal |
| 18 | [Power Supplies](power_supplies.md) | Converting and regulating voltage |
| 19 | [Sensors & Transducers](sensors.md) | Turning the physical world into a voltage |
| 20 | [Prototyping & Test Equipment](prototyping.md) | Building and measuring real circuits |
| 21 | [Reading a Datasheet](datasheets.md) | Decoding a component's ratings and limits |

## PCB Design

Once you can build and measure circuits on a breadboard, the next step is designing a permanent PCB.

| Topic | What you learn |
|-------|---------------|
| [Circuit Design](circuit_design.md) | Schematics, PCB layers, trace rules, design-for-manufacture — tool-agnostic concepts |
| [KiCad Schematic](kicad_schematic.md) | Capturing your circuit in KiCad's schematic editor (eeschema) |
| [KiCad PCB](kicad_pcb.md) | Laying out the board, routing traces, Gerber export, and ordering from JLCPCB |
| [Schematic Symbol Reference](symbols.md) | A one-page lookup of every component symbol used in this section |

## How this section connects to the rest of the book

Once you understand these fundamentals, the [Embedded](../embedded/README.md) section will make far more sense — concepts like [ADC](../embedded/adc.md), [PWM](../embedded/pwm.md), [Power Management](../embedded/power_management.md), and [Signal Integrity](../embedded/signal_integrity.md) all build directly on what is here.
