# Processor Design (Microarchitecture)

## 1. Introduction to Microarchitecture

If the [Instruction Set Architecture (ISA)](isa.md) is the blueprint of a building, the **microarchitecture** (or processor design) is the actual construction: the materials used, the layout of the plumbing, and the speed of the elevators. It defines how the logical instructions of the ISA are executed by physical transistors. The cache hierarchy is covered in [Cache & TCM](cache_tcm.md), and interrupt latency is determined by the [interrupt controller](interrupts.md) together with pipeline depth.

The holy grail of processor design is optimizing the "PPA" metrics:
1.  **Performance:** Usually measured in IPC (Instructions Per Cycle) multiplied by the Clock Frequency.
2.  **Power:** Both dynamic power (energy consumed when transistors switch state) and static power (leakage current when transistors are idle).
3.  **Area:** The physical silicon real estate required. Larger chips cost significantly more to manufacture and have lower yields.

Processor performance is defined by the **Iron Law of Performance**:
`Time/Program = (Instructions/Program) × (Cycles/Instruction) × (Time/Cycle)`

*   *Instructions/Program* is determined by the compiler and the ISA.
*   *Cycles/Instruction (CPI)* is determined by the microarchitecture (how much parallelism can be extracted, how often the pipeline stalls).
*   *Time/Cycle (Clock Period)* is determined by the physical design and how deeply the pipeline is divided.

## 2. Digital Logic Foundations

Microarchitectures are built from two fundamental types of digital logic:
*   **Combinational Logic:** The output is a pure function of the current inputs. It has no memory. Examples include ALUs (Adders, Multipliers), multiplexers, and decoders.
*   **Sequential Logic:** Contains memory elements (flip-flops, registers). The output depends on the current inputs *and* the previous state. Sequential logic is driven by a global **Clock**.

The clock signal dictates the rhythm of the processor. During a single clock cycle, data leaves a register, propagates through a cloud of combinational logic, and arrives at the input of the next register.
*   **Propagation Delay:** The time it takes for electrical signals to settle through the combinational logic.
*   **Setup Time:** Data must arrive at the destination register slightly *before* the clock edge strikes.
*   **Hold Time:** Data must remain stable for a brief period *after* the clock edge.
If the combinational logic path is too long, the data won't arrive before the setup time, causing catastrophic errors. This "Critical Path" defines the maximum possible clock frequency of the processor.

## 3. The Single-Cycle and Multi-Cycle Datapath

The simplest way to build a CPU is the **Single-Cycle Datapath**. Every instruction executes entirely within one massive clock cycle.
*   **Flow:** Fetch Instruction -> Decode -> Read Registers -> ALU Math -> Memory Read/Write -> Write Register.
*   **Flaw:** The clock cycle must be long enough to accommodate the absolute slowest instruction (usually a Memory Load). This means simple instructions (like `ADD`) finish quickly and sit idle, wasting huge amounts of time. The clock speed will be measured in MHz, not GHz.

The historical stepping stone was the **Multi-Cycle Datapath**. The instruction execution is broken into steps. A complex instruction might take 5 clock cycles, while a simple one takes 3. The clock frequency can be much higher, but only one instruction is ever processed at a time.

## 4. Pipelining

Pipelining is the foundational technique of modern processor performance. It works like an automobile assembly line. Instead of one station building an entire car, the work is divided into discrete stages.

### The Classic 5-Stage RISC Pipeline
1.  **IF (Instruction Fetch):** The PC provides the address. The instruction is fetched from the L1 Instruction Cache. The PC is incremented.
2.  **ID (Instruction Decode & Register Read):** The control logic parses the opcode. The source registers are read from the Register File.
3.  **EX (Execute):** The ALU performs arithmetic or calculates a memory address for a load/store.
4.  **MEM (Memory Access):** If the instruction is a Load or Store, the L1 Data Cache is accessed. Otherwise, this stage does nothing.
5.  **WB (Writeback):** The result from the ALU or the Memory is written back into the destination register.

Between each stage are **Pipeline Registers** that hold the intermediate data and control signals.

*   **Impact:** Pipelining does not make a single instruction faster (latency is still 5 cycles). However, it massively increases **throughput**. Ideally, one instruction finishes every single clock cycle (CPI = 1.0).

### Deep Pipelining (Superpipelining)
To achieve multi-gigahertz clock speeds, modern processors chop these 5 stages into many smaller stages. Less logic per stage means a shorter critical path, allowing a faster clock.
*   The infamous Intel Pentium 4 ("NetBurst" architecture) had a 31-stage pipeline to hit high clock speeds.
*   Modern Intel Core and AMD Ryzen architectures typically have 14 to 20 pipeline stages.

## 5. Pipeline Hazards

The ideal CPI of 1.0 is ruined by "hazards" — situations that prevent the next instruction from executing in the correct cycle.

### 5.1 Structural Hazards
Hardware lacks the resources to support all active instructions simultaneously.
*   *Example:* A unified memory for both code and data. The IF stage tries to fetch code while the MEM stage tries to read data. They collide.
*   *Solution:* **Harvard Architecture** at the cache level (separate L1i and L1d caches).

### 5.2 Data Hazards
An instruction depends on the result of a previous instruction that hasn't finished yet.
*   *Read-After-Write (RAW):*
    `ADD R1, R2, R3` (R1 is calculated in EX, but not written to the register file until WB).
    `SUB R4, R1, R5` (Needs R1 in ID, which is too early).
*   *Solution 1: Stalls (Bubbles).* Halt the pipeline for 3 cycles. Destroys performance.
*   *Solution 2: Data Forwarding (Bypassing).* Add massive multiplexers and wiring to the CPU. As soon as the ALU calculates R1 in the EX stage, forward that result directly back to the input of the ALU for the `SUB` instruction in the very next cycle, bypassing the register file entirely.

### 5.3 Control Hazards
Caused by conditional branches (`IF` statements, loops).
*   When a `BEQ` (Branch if Equal) is in the IF stage, the CPU doesn't know if the branch is taken, nor the target address, until the EX stage. It doesn't know what instruction to fetch next.
*   If the CPU stalls waiting for the branch to resolve, a 20-stage pipeline will waste 20 cycles on every loop iteration.

## 6. Advanced Branch Prediction

To solve Control Hazards, CPUs guess. They predict the branch outcome and begin fetching and executing instructions along the guessed path. This is **Speculative Execution**.
*   If the guess is right, the pipeline stays full. Zero penalty.
*   If the guess is wrong (a **Mispredict**), the CPU must flush all speculatively executed instructions from the pipeline and fetch the correct path. In a deep pipeline, a mispredict costs 15-20 cycles.

### Predictor Architectures
1.  **Static:** Hardwired rules (e.g., backward branches are taken, forward are not).
2.  **Branch Target Buffer (BTB):** A cache that maps the PC of a branch to its target address, allowing the CPU to know *where* to jump immediately.
3.  **Bimodal Predictor (2-bit counters):** A table of 2-bit state machines (Strongly Taken, Weakly Taken, Weakly Not Taken, Strongly Not Taken). It requires two consecutive mispredictions to flip its prediction. Good for standard loops.
4.  **Two-Level Adaptive (Global History):** Maintains a Shift Register recording the outcome (T/NT) of the last N branches executed anywhere in the program. This history is used to index into a table of 2-bit counters. This allows the CPU to learn complex correlations (e.g., `if(a) {..} if(b) {..} if(a && b) {..}`).
5.  **Perceptrons:** Modern CPUs (like AMD Zen) use hardware neural networks (perceptrons). They use weights and biases to weigh long branch histories, providing incredibly high accuracy (>95%) even on chaotic code.
6.  **Return Address Stack (RAS):** A small, hardware-managed stack used exclusively to perfectly predict the target of `RET` (Return) instructions, which otherwise jump to variable locations depending on who called the function.

## 7. Instruction-Level Parallelism (ILP)

Pipelining overlaps instructions over time. ILP executes multiple instructions simultaneously in space.

### Superscalar Architecture
Instead of one ALU, provide four. Instead of fetching one instruction per cycle, fetch four. A "4-wide Superscalar" processor has parallel decode and execution lanes.
*   Hardware must dynamically analyze the fetched instructions. If it finds `ADD R1, R2, R3` and `SUB R4, R5, R6`, they are totally independent. The CPU dispatches them to different ALUs in the exact same clock cycle.
*   If it finds dependencies, the hardware must serialize them.
*   Superscalar execution allows the CPI to drop below 1.0 (e.g., executing 3 instructions per cycle yields a CPI of 0.33).

### VLIW (Very Long Instruction Word)
An alternative to superscalar. Instead of complex hardware figuring out dependencies at runtime, force the compiler to do it. The compiler bundles independent operations into one massive 128-bit instruction (e.g., [ALU op, Memory op, Branch op]).
*   *Pros:* Radically simpler hardware. Low power.
*   *Cons:* Compilers struggle to extract enough ILP. If the compiler can't find independent instructions, it must insert NOPs, bloating code size. Binary compatibility across different chip generations is impossible. (e.g., Intel Itanium / EPIC architecture failed largely due to these issues).

## 8. Out-of-Order Execution (OoOE)

In strictly "In-Order" processors, if an instruction stalls (e.g., waiting 200 cycles for data from main memory), the entire pipeline behind it halts.

**Out-of-Order Execution (OoOE)** breaks the sequential chain. It allows younger, independent instructions to bypass older, stalled instructions. Almost all modern high-performance CPUs are Superscalar OoO designs.

### The OoOE Pipeline (Based on Tomasulo's Algorithm)

1.  **Fetch & Decode (In-Order):** Fetch wide blocks of instructions.
2.  **Register Renaming:** Solves "False Dependencies" (Write-After-Write and Write-After-Read). The ISA might only have 16 Architectural Registers (like x86), but the hardware contains hundreds of hidden **Physical Registers**. The **Register Alias Table (RAT)** dynamically remaps architectural registers to physical registers on the fly.
3.  **Dispatch to Reservation Stations (In-Order):** Instructions are placed into waiting areas (Issue Queues or Reservation Stations) located right in front of the execution units (ALUs).
4.  **Execute (Out-of-Order):** Instructions sit in the reservation stations monitoring a common data bus. As soon as an instruction sees that its required source operands have been computed by other units, and an execution unit is free, it fires. It executes regardless of its original program order.
5.  **Commit / Retire (In-Order):** This is critical. To maintain the illusion of sequential execution (for debugging and handling exceptions/interrupts), finished instructions are placed into a **Reorder Buffer (ROB)**. The ROB sorts the out-of-order results back into the original program order. An instruction is only "retired" (allowed to permanently update the architectural register file or memory) when it reaches the head of the ROB and all older instructions have safely retired.

### Memory Disambiguation
OoOE for arithmetic is easy. OoOE for memory is hard. If a Store writes to address X, and a subsequent Load reads from address X, the Load must wait. But the CPU often doesn't know the addresses until the EX stage.
*   **Load/Store Queues (LSQ):** Hardware structures that track all inflight memory operations. They perform "Memory Disambiguation," aggressively forwarding data from pending stores to younger loads if addresses match, or stalling loads if an older store's address is still unknown.

## 9. The Memory Hierarchy

The CPU is blindingly fast; DRAM is agonizingly slow. The memory hierarchy bridges this gap using the principles of Temporal and Spatial Locality.

### Cache Architecture
Caches are built from fast, power-hungry SRAM.
*   **Cache Line (Block):** Caches don't move single bytes; they move blocks (usually 64 bytes). Fetching an integer brings the next 15 integers into the cache for free (exploiting spatial locality).
*   **Direct-Mapped Cache:** Each memory address maps to exactly one specific line in the cache. Simple, fast, but suffers from "Conflict Misses" if two heavily used variables map to the same line.
*   **Fully Associative Cache:** A memory block can be stored anywhere in the cache. Maximizes hit rate, but searching the entire cache simultaneously requires massive, slow, power-hungry comparator logic.
*   **Set-Associative Cache:** The sweet spot. The cache is divided into "Sets". A memory address maps to a specific set, but can be placed in any of the N "Ways" within that set (e.g., an 8-way set-associative cache).

### Multicore Cache Coherence
In a multicore processor, each core has its own private L1 and L2 caches, sharing a large L3 cache.
*   *The Problem:* Core 0 reads variable X into its L1. Core 1 reads X into its L1. Core 0 modifies X. Core 1 now has "stale" (incorrect) data.
*   *The Solution:* **Cache Coherence Protocols**. Hardware monitors (snoops) all memory traffic.
*   **MESI Protocol:** Every cache line is tagged with a state:
    *   **M (Modified):** This core has modified the data. It is the only valid copy in the system.
    *   **E (Exclusive):** This core has a clean copy, and no other core has it. It can modify it without asking permission.
    *   **S (Shared):** Multiple cores have clean copies. If a core wants to write, it must broadcast an "Invalidate" signal to all other cores.
    *   **I (Invalid):** The data is stale and cannot be used.

## 10. Thread-Level Parallelism (TLP)

When extracting ILP yields diminishing returns, architects turn to TLP.

### Simultaneous Multithreading (SMT / Hyper-Threading)
A superscalar OoOE core has massive resources (e.g., 6 ALUs), but a single thread rarely has enough independent instructions to keep them all busy (due to cache misses or serial dependencies).
*   **SMT:** The hardware duplicates the architectural state (the Registers and the PC). To the Operating System, it looks like two separate logical cores.
*   The hardware interleaves instructions from Thread A and Thread B into the *same* out-of-order execution engine in the exact same clock cycle. Thread B utilizes the ALUs that Thread A left idle.

### Multicore (SMP - Symmetric Multiprocessing)
Duplicating the entire core (pipelines, L1/L2 caches, execution units) on a single silicon die. Modern desktop chips have 8 to 24 cores; server chips have over 100.
*   They communicate via an **Interconnect** (a Ring Bus, a Mesh Network, or a Crossbar Switch) and share access to the L3 cache and memory controllers.

## 11. Power, Thermals, and Modern Challenges

For decades, Moore's Law gave architects more transistors, and Dennard Scaling ensured those transistors consumed less power, allowing higher clock speeds. Dennard Scaling ended around 2005.

*   **Dark Silicon:** Modern chips have so many transistors that if they were all powered on at maximum frequency simultaneously, the chip would melt. Significant portions of the chip must remain "dark" (powered off or heavily throttled) at any given time.
*   **Dynamic Voltage and Frequency Scaling (DVFS):** The CPU micro-controller constantly monitors current, temperature, and workload. It dynamically alters the clock frequency and operating voltage millisecond by millisecond (e.g., Intel Turbo Boost).
*   **Heterogeneous Architecture (big.LITTLE):** Mixing massive, power-hungry, OoO "Performance Cores" with small, highly efficient "Efficiency Cores" on the same die. The OS schedules heavy tasks on the P-cores and background tasks on the E-cores to maximize battery life (used heavily in ARM smartphones and Intel's newer architectures).
*   **Chiplets:** Monolithic silicon dies are becoming too large to manufacture profitably (yield defects). The future is "Chiplets" — manufacturing CPU cores, IO controllers, and caches as separate, smaller pieces of silicon and packaging them tightly together on an interposer (e.g., AMD Zen architecture).


## 12. Advanced Execution Units

The "Execute" stage of the pipeline is not a monolith. It contains multiple distinct functional units, each optimized for different mathematical or logical operations.

### 12.1 The Arithmetic Logic Unit (ALU)
The heart of the CPU. Handles integer arithmetic, bitwise logic, and address calculation.
*   **Adders:** The most critical component. Simple Ripple-Carry Adders are too slow (the carry bit propagates sequentially through every bit). Modern CPUs use **Carry-Lookahead Adders** or **Kogge-Stone Adders**, which use parallel logic gates to compute the carry bits for all positions simultaneously, trading massive silicon area for speed.
*   **Multipliers:** Integer multiplication is complex. Hardware multipliers use structures like the **Wallace Tree**, which adds partial products in parallel using a tree of half-adders and full-adders, reducing the latency from O(N) to O(log N).
*   **Dividers:** The slowest integer operation. Often implemented iteratively using algorithms like SRT division (similar to long division), which can take 10 to 40 clock cycles and cannot be fully pipelined.

### 12.2 Floating Point Units (FPU)
Handles IEEE 754 floating-point math. FPUs are massively complex because they must handle sign, exponent, and mantissa calculations independently.
*   *Addition/Subtraction:* Requires aligning the decimal points (shifting the mantissa of the smaller number based on the exponent difference), performing the addition, and then normalizing the result.
*   *Multiplication:* Multiply the mantissas, add the exponents, and normalize.
*   *Fused Multiply-Add (FMA):* A highly optimized unit that performs `A * B + C` in a single operation, with only one rounding step at the very end. This is crucial for digital signal processing, matrices, and neural networks.

### 12.3 Vector Units (SIMD)
Single Instruction, Multiple Data. Instead of adding two 32-bit numbers, a vector unit might add two 512-bit registers, effectively performing sixteen 32-bit additions simultaneously.
*   Vector units contain massive datapathes. To keep them fed with data, the memory subsystem must support extremely wide load/store operations (fetching 64 bytes per cycle from the L1 cache).
*   These units are massive power hogs. When a CPU executes dense AVX-512 instructions, it often must dynamically lower its clock frequency (DVFS) to prevent thermal meltdown.

### 12.4 Matrix / Tensor Cores
The newest addition to high-performance microarchitectures (e.g., Apple Neural Engine, Google TPU, NVIDIA Tensor Cores, Intel AMX).
*   Instead of vectors (1D arrays), these units operate natively on 2D matrices.
*   They perform hardware-accelerated matrix multiplication and convolution operations using systolic arrays (a grid of processing elements where data flows through rhythmically), offering orders of magnitude better performance for Machine Learning inference.

## 13. System-on-Chip (SoC) and Interconnects

A modern processor is rarely just a CPU. It is a System-on-Chip (SoC) containing multiple CPU cores, a GPU, memory controllers, PCIe controllers, and dedicated accelerators. How these blocks communicate is defined by the interconnect architecture.

### 13.1 Bus Architecture
The oldest and simplest. All components share a common set of wires (address, data, control).
*   *Pros:* Simple to implement.
*   *Cons:* Does not scale. Only one device can talk at a time. If Core 1 is talking to RAM, Core 2 cannot talk to the GPU. Severe electrical capacitance issues at high speeds.

### 13.2 Crossbar Switch
Every component is connected to every other component via a grid of switches.
*   *Pros:* Massive bandwidth. Core 1 can talk to RAM while Core 2 talks to the GPU simultaneously (non-blocking).
*   *Cons:* Area scales quadratically (N^2). A 16x16 crossbar requires 256 switches. Impractical for chips with dozens of components.

### 13.3 Ring Bus
Components are arranged in a circle. Data packets hop from one component to the next in a ring.
*   Used heavily by Intel in their Core architectures.
*   *Pros:* Excellent scaling up to about 10-12 cores. Predictable latency.
*   *Cons:* As the number of cores grows, the time it takes for a packet to travel halfway around the ring becomes a severe bottleneck.

### 13.4 Network-on-Chip (NoC) / Mesh Architecture
The modern standard for massive many-core processors (e.g., AMD EPYC, Intel Xeon Scalable).
*   Components are laid out in a 2D grid (Mesh). Each component has a router connected to its North, South, East, and West neighbors.
*   Data is packetized and routed across the chip much like the internet.
*   *Pros:* Scales almost infinitely. Extremely high aggregate bandwidth.
*   *Cons:* Complex routing logic required at every node. Latency is variable depending on the physical distance between nodes (NUMA - Non-Uniform Memory Access effects within the chip itself).

### 13.5 The AMBA Standard (AXI / AHB)
In the embedded space (especially ARM), the interconnect protocols are standardized by the Advanced Microcontroller Bus Architecture (AMBA).
*   **AHB (Advanced High-performance Bus):** Used for connecting the CPU to high-speed memory and DMA.
*   **APB (Advanced Peripheral Bus):** A slower, simpler bus connected via a bridge, used for low-speed peripherals (UART, Timers, GPIO).
*   **AXI (Advanced eXtensible Interface):** The modern standard for high-performance SoCs. It features separate read/write channels, out-of-order transaction completion, and burst transfers, acting more like a network protocol than a traditional bus.

## 14. Modern Manufacturing and Packaging

Microarchitecture is intimately tied to the physical realities of semiconductor manufacturing.

### 14.1 Lithography Nodes
Transistor sizes are defined by the "node" (e.g., 7nm, 5nm, 3nm), though these numbers are now purely marketing terms and do not reflect physical gate lengths.
*   Smaller nodes allow packing more transistors into the same area, reducing the distance electrons must travel, thus lowering capacitance, decreasing switching power, and increasing maximum clock speed.
*   However, smaller nodes suffer massively from **Quantum Tunneling / Leakage Current**, meaning transistors consume power even when turned off.

### 14.2 Transistor Designs
*   **Planar MOSFETs:** Traditional 2D transistors. Hit physical limits around 20nm due to leakage.
*   **FinFET:** A 3D transistor where the gate wraps around a raised silicon "fin" on three sides, providing vastly superior electrical control to prevent leakage. Dominant from 14nm down to 5nm.
*   **GAAFET (Gate-All-Around / RibbonFET):** The next generation (3nm and below). The gate completely surrounds horizontal silicon nanosheets on all four sides, offering the ultimate electrostatic control required for extreme miniaturization.

### 14.3 Advanced Packaging (Chiplets)
Building a monolithic (single piece of silicon) 600mm^2 chip is incredibly expensive because a single microscopic dust particle during manufacturing ruins the entire chip (low yield).

*   **Chiplets:** AMD revolutionized modern architecture with "Zen" by splitting the processor into smaller, cheaper chiplets. Multiple CPU "Core Complex" (CCX) chiplets (made on an expensive 5nm node) are glued together with a central I/O chiplet (made on a cheaper 12nm node) on an organic substrate.
*   **2.5D Packaging (Silicon Interposer):** Connecting chiplets using a silicon base layer with microscopic wiring, allowing massive bandwidth between chiplets (e.g., integrating High Bandwidth Memory - HBM directly next to a GPU).
*   **3D Stacking:** Using Through-Silicon Vias (TSVs) to stack silicon dies directly on top of each other. (e.g., AMD 3D V-Cache, which glues an extra 64MB of SRAM directly on top of the CPU cores to massively expand the L3 cache footprint without increasing the 2D footprint).

## Where this connects

- [ISA](isa.md) — the ISA is the contract the microarchitecture must implement; same ISA, many microarchitectures
- [Cache & TCM](cache_tcm.md) — the cache hierarchy is the largest single factor affecting real-world IPC
- [Interrupts](interrupts.md) — pipeline depth and out-of-order execution affect interrupt entry/exit latency
- [Linker Scripts](linker_scripts.md) — placing hot code in TCM (Tightly Coupled Memory) is a microarchitecture-aware optimization
- [Power Management](power_management.md) — DVFS, power gating, and clock gating are microarchitecture-level power techniques
