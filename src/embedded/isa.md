# Instruction Set Architecture (ISA)

## 1. Introduction to Instruction Set Architecture

An Instruction Set Architecture (ISA) defines the abstract model of a computer. It is the critical interface between the hardware (the physical silicon, logic gates, and microarchitecture) and the software (the operating system, compilers, and application code). A well-defined ISA allows hardware engineers to design faster, more efficient processors while guaranteeing that existing software will continue to run without modification.

The ISA specifies everything a machine language programmer or compiler writer needs to know to operate the machine:
- The set of available instructions.
- The data types supported natively by the hardware.
- The registers available for computation and state management.
- The memory model, including addressing modes, virtual memory semantics, and memory ordering.
- The mechanisms for handling interrupts, exceptions, and privilege levels.

It is crucial to distinguish the **ISA** from the **Microarchitecture**:
*   **ISA:** The specification. It answers *what* the processor can do. (e.g., x86-64, ARMv8-A, RISC-V RV64GC).
*   **Microarchitecture:** The implementation. It answers *how* the processor does it. (e.g., Intel "Alder Lake", AMD "Zen 4", ARM "Cortex-A78"). Multiple completely different microarchitectures can implement the exact same ISA.

## 2. Historical Evolution of the ISA

The evolution of ISAs reflects the changing constraints of computing, moving from eras where hardware was astronomically expensive to modern times where power consumption and thermal dissipation are the primary limits.

### 2.1 Accumulator Architectures
In the earliest computers, logic gates were extremely expensive. The simplest way to build a CPU was to have a single, central register called the **Accumulator**. All arithmetic and logical operations implicitly read one operand from the accumulator, read the other from memory, and wrote the result back to the accumulator.
*   **Instruction Format:** `ADD <memory_address>`
*   **Semantics:** `Accumulator = Accumulator + Memory[memory_address]`
*   **Pros:** Minimal internal CPU state, short instruction encoding (only one operand address needed).
*   **Cons:** Extremely high memory traffic. Complex expressions like `A = (B + C) * (D + E)` required constant swapping of the accumulator's contents to and from memory.

### 2.2 Stack Architectures
To alleviate the single-register bottleneck without adding many registers, stack architectures were developed. The CPU operates on a pushdown stack. Operands are pushed onto the stack, and ALU operations implicitly consume the top elements of the stack and push the result.
*   **Instruction Format:** `ADD`
*   **Semantics:** `Push(Pop() + Pop())`
*   **Pros:** Incredible code density (many instructions have zero operands and are only 1 byte long). Compilers can easily generate code by traversing Abstract Syntax Trees (ASTs) in post-order.
*   **Cons:** The stack acts as a severe bottleneck. Operands cannot be accessed randomly; if a value deep in the stack is needed, multiple operations must be performed to retrieve it. Instruction-Level Parallelism (ILP) is difficult to extract.

### 2.3 General-Purpose Register (GPR) Architectures
As transistors became cheaper, CPUs incorporated arrays of interchangeable registers. GPR architectures dominate the modern computing landscape. They are subdivided into three categories based on how they access memory:
1.  **Memory-Memory:** All operands for an ALU instruction can be memory addresses. (e.g., VAX). Highly complex and slow to decode.
2.  **Register-Memory:** One operand can be a memory address, the other must be a register. The result usually overwrites the register. (e.g., x86, IBM System/360).
3.  **Register-Register (Load-Store):** ALU operations can *only* access registers. Memory is only accessed via dedicated `LOAD` and `STORE` instructions. This decouples memory latency from arithmetic execution. (e.g., ARM, RISC-V, MIPS).

### 2.4 CISC vs. RISC: The Great Debate
During the 1980s and 1990s, the computer architecture world was divided between two philosophies.

**CISC (Complex Instruction Set Computer)**
*   **Context:** In the 1970s, memory was extremely expensive and slow, and compilers were primitive.
*   **Philosophy:** Move the semantic gap closer to high-level languages. Provide complex instructions that perform multiple operations (e.g., an instruction to copy an entire string, or an instruction that does `A = B + C` where all three are memory addresses).
*   **Characteristics:** Variable-length instructions, complex addressing modes, heavy use of microcode (internal ROM that breaks complex instructions down into simpler steps).
*   **Example:** Intel x86.

**RISC (Reduced Instruction Set Computer)**
*   **Context:** In the 1980s, researchers noticed that compilers rarely used the complex CISC instructions. Furthermore, complex decoding logic prevented fast clock speeds.
*   **Philosophy:** Optimize the hardware to execute the most common, simple instructions extremely fast. Let the compiler synthesize complex operations from simple ones.
*   **Characteristics:** Fixed-length instructions (e.g., 32 bits), Load-Store architecture, many GPRs, hardwired control logic (no microcode), designed specifically for pipelining.
*   **Example:** ARM, RISC-V.

**The Modern Synthesis:** Today, the distinction is largely academic. Modern CISC processors (like x86-64) have complex hardware decoders that translate CISC instructions into RISC-like "micro-ops" before execution. Modern RISC processors have added complex extensions (like vector math and cryptography) to accelerate specific workloads.

## 3. Data Types and Endianness

An ISA defines the fundamental data types the hardware can manipulate directly.

### 3.1 Integer Types
*   **Bytes (8-bit):** Used for characters (ASCII) and small flags.
*   **Halfwords (16-bit):** Historical artifacts or used for compact data (Unicode UTF-16).
*   **Words (32-bit):** Standard integer size for most modern 32-bit and many 64-bit systems.
*   **Doublewords (64-bit):** Native integer size for modern 64-bit systems, necessary for addressing more than 4GB of RAM.
*   **Signed vs. Unsigned:** The ISA defines instructions (or flags) that treat the Most Significant Bit (MSB) as a sign bit (using Two's Complement representation) or as part of the magnitude. For example, `ADD` is the same for both, but `MULTIPLY` and `BRANCH_IF_GREATER` require distinct signed and unsigned variants.

### 3.2 Floating-Point Types
Almost all modern ISAs adhere to the IEEE 754 standard for floating-point arithmetic.
*   **Single Precision (32-bit):** 1 sign bit, 8 exponent bits, 23 fraction bits.
*   **Double Precision (64-bit):** 1 sign bit, 11 exponent bits, 52 fraction bits.
*   **Half Precision (16-bit) and Bfloat16:** Increasingly common in ISAs due to the rise of Machine Learning, where precision can be sacrificed for bandwidth and compute density.

### 3.3 Endianness
Endianness describes the order in which bytes within a multi-byte word are stored in computer memory.
*   **Little-Endian:** The Least Significant Byte (LSB) is stored at the lowest memory address. (Used by x86, RISC-V by default).
    *   Example: 32-bit hex value `0x1A2B3C4D` at address `0x100`
    *   `0x100: 0x4D`
    *   `0x101: 0x3C`
    *   `0x102: 0x2B`
    *   `0x103: 0x1A`
*   **Big-Endian:** The Most Significant Byte (MSB) is stored at the lowest memory address. (Used by traditional network protocols, older PowerPC).
    *   `0x100: 0x1A`
    *   `0x101: 0x2B` ... etc.
*   **Bi-Endian:** Some architectures (like ARM) can be configured at boot time or via a control register to operate in either mode, though Little-Endian is the overwhelming default for modern operating systems.

### 3.4 Alignment
Data alignment refers to how data is arranged in memory relative to its size. A 32-bit (4-byte) integer is "naturally aligned" if its memory address is a multiple of 4.
*   Strict alignment ISAs (many older RISC chips) will trigger a hardware exception (Alignment Fault) if an unaligned access is attempted.
*   Permissive alignment ISAs (x86) handle unaligned accesses transparently in hardware, though usually with a performance penalty because the cache must fetch two separate cache lines.

## 4. Architectural State and Registers

The architectural state is the "context" of a running thread. When an OS switches context, it must save and restore this state.

### 4.1 General Purpose Registers (GPRs)
Used for data manipulation and address calculation. More GPRs mean fewer memory spills, but require more bits in the instruction encoding.
*   **x86 (32-bit):** 8 GPRs (`EAX`, `EBX`, `ECX`, `EDX`, `ESI`, `EDI`, `EBP`, `ESP`). Highly restrictive.
*   **x86-64:** 16 GPRs (`RAX` through `R15`).
*   **ARMv8 (AArch64):** 31 GPRs (`X0` through `X30`), plus a zero register (`XZR`).
*   **RISC-V:** 32 GPRs (`x0` through `x31`). `x0` is hardwired to zero (writes are ignored, reads return 0).

### 4.2 Special Purpose Registers
*   **Program Counter (PC) / Instruction Pointer (RIP):** Holds the memory address of the next instruction.
*   **Stack Pointer (SP):** Points to the top of the currently active stack.
*   **Link Register (LR):** In RISC ISAs (ARM, RISC-V), the `CALL` or `Branch-and-Link` instruction automatically saves the return PC into the LR. This avoids hitting the memory stack for leaf functions.
*   **Status Register (EFLAGS, CPSR):** Contains flags updated by the ALU.
    *   `Z` (Zero), `N` (Negative), `C` (Carry), `V` (Overflow).
    *   Also contains system state flags like Interrupt Enable and Privilege Level.

## 5. Addressing Modes

Addressing modes define how instructions identify operands. A rich set of addressing modes makes compiler generation easier but complicates hardware.

1.  **Implicit / Implied:** The operand is hardcoded into the instruction definition. (e.g., x86 `CLI` - Clear Interrupt flag; the target is implicitly the status register).
2.  **Immediate:** The operand is a constant encoded within the instruction bits. (e.g., `ADD R1, R1, #42`).
3.  **Register Direct:** The operand is the value in a register. (e.g., `MOV R1, R2`).
4.  **Memory Direct (Absolute):** The instruction contains the full 32-bit or 64-bit memory address. Very rare in modern RISC due to fixed instruction lengths. (e.g., `LOAD R1, [0x12345678]`).
5.  **Register Indirect:** A register holds the memory address. (e.g., `LOAD R1, [R2]`).
6.  **Base + Displacement (Indexed):** Address = Register + Constant Offset. Crucial for accessing struct members and stack variables. (e.g., `LOAD R1, [R2 + 16]`).
7.  **Base + Index:** Address = Register1 + Register2. Used for array access. (e.g., `LOAD R1, [R2 + R3]`).
8.  **Scaled Index:** Address = Base + (Index * Scale) + Offset. Very common in x86. Scale is usually 1, 2, 4, or 8. (e.g., `MOV EAX, [EBX + ECX*4]`).
9.  **PC-Relative:** Address = Program Counter + Constant Offset. Essential for Position-Independent Code (PIC) used in shared libraries. It allows code to be loaded at any memory address without rewriting pointers.
10. **Auto-Increment / Auto-Decrement:** The base register is updated automatically as part of the load/store.
    *   *Pre-increment:* Update register, then access memory.
    *   *Post-increment:* Access memory, then update register (e.g., ARM `LDR R1, [R2], #4`).

## 6. Instruction Encoding Formats

The layout of the bits within an instruction word.

### Fixed-Length Encoding (e.g., RISC-V 32-bit Base ISA)
RISC-V instructions are exactly 32 bits long. They use a highly regular format to make the hardware decoder as fast and small as possible.
*   **R-Type (Register):** `[funct7: 7 bits] [rs2: 5 bits] [rs1: 5 bits] [funct3: 3 bits] [rd: 5 bits] [opcode: 7 bits]`
    *   Used for ALU ops like `ADD rd, rs1, rs2`.
*   **I-Type (Immediate):** `[imm[11:0]: 12 bits] [rs1: 5 bits] [funct3: 3 bits] [rd: 5 bits] [opcode: 7 bits]`
    *   Used for loads and immediate arithmetic like `ADDI rd, rs1, imm`.
*   Notice that the destination register `rd`, source register `rs1`, and the `opcode` are always in the exact same bit positions across different formats, allowing the hardware to extract them simultaneously without waiting to figure out the instruction type.

### Variable-Length Encoding (e.g., x86-64)
x86 instructions can be anywhere from 1 to 15 bytes long.
*   **Format:** `[Prefixes: 0-4 bytes] [Opcode: 1-3 bytes] [ModR/M: 1 byte] [SIB: 1 byte] [Displacement: 1, 2, 4 bytes] [Immediate: 1, 2, 4 bytes]`
*   The `ModR/M` (Mode-Register/Memory) byte defines the addressing mode and registers used.
*   The `SIB` (Scale-Index-Base) byte is used for complex array addressing.
*   Decoding this is a sequential nightmare for hardware, requiring massively complex prediction and pre-decoding logic.

## 7. Control Flow

### Branching and Jumps
*   **Unconditional Jumps:** Always change the PC to a new location.
*   **Conditional Branches:** Change the PC only if a condition is met.
    *   *Condition Code Based:* Compare instructions set flags in the Status Register. A subsequent branch checks those flags (e.g., x86 `CMP EAX, EBX` followed by `JE target`).
    *   *Register Based:* The branch instruction itself compares two registers (e.g., RISC-V `BEQ rs1, rs2, target`). This eliminates the need for a global Status Register, which helps Out-of-Order execution engines.

### Function Calls
Calling a subroutine requires saving the return address.
*   **x86:** The `CALL` instruction implicitly pushes the PC onto the memory stack and jumps. `RET` pops it off.
*   **ARM/RISC-V:** `BL` (Branch with Link) or `JAL` (Jump and Link) saves the current PC into a dedicated Link Register (LR). The called function must manually push the LR to the stack only if it intends to call another function (i.e., it is not a "leaf" function). This makes leaf functions incredibly fast.

### Predication
Predication is an alternative to branching. Instead of jumping around an instruction, the instruction executes conditionally.
*   **ARMv7 IT Blocks (If-Then):** `CMP R1, R2` followed by `ADDEQ R3, R4, R5`. The `ADD` is only committed to architectural state if the Zero flag was set by the `CMP`. If not, it acts as a NOP. This eliminates branch misprediction penalties for short IF statements.

## 8. Exceptions, Interrupts, and System Calls

An ISA must handle events outside the normal flow of the program.

### Types of Exceptions
1.  **Interrupts (Asynchronous):** Generated by external hardware (e.g., a network packet arrives, a timer expires). The CPU halts the current program between instructions.
2.  **Traps (Synchronous):** Intentional exceptions generated by the program, usually via a special instruction (`SYSCALL`, `SVC`, `INT 0x80`). This is how user-space programs request services from the OS kernel (like reading a file or allocating memory).
3.  **Faults (Synchronous):** Unintentional errors during instruction execution (e.g., Page Fault, Divide by Zero, Illegal Instruction). The OS might resolve it (e.g., loading a swapped page from disk) and retry the instruction, or it might kill the program (Segmentation Fault).
4.  **Aborts:** Severe hardware errors (e.g., memory parity error) from which recovery is impossible.

### The Exception Handling Process
1.  Hardware saves the current PC (to a special `Exception PC` register or the kernel stack).
2.  Hardware saves the current Status Register.
3.  Hardware elevates the privilege level to Supervisor/Kernel mode.
4.  Hardware looks up the exception number in an **Interrupt Vector Table (IVT)**, which contains the memory addresses of the OS's handler functions.
5.  Hardware loads the PC with the handler's address and resumes execution.
6.  When the OS is done, it executes a special "Return from Exception" instruction (`IRET`, `ERET`, `MRET`) which restores the user PC, restores the user Status Register, and drops the privilege level back to User mode.

## 9. Privilege Levels and Virtualization

Modern ISAs provide hardware protection rings to prevent user applications from crashing the system.

### Privilege Modes (e.g., ARMv8 Exception Levels)
*   **EL0 (User):** Unprivileged. Runs application code. Cannot access hardware directly or change the MMU.
*   **EL1 (Kernel):** Privileged. Runs the Operating System. Can configure hardware, interrupts, and page tables.
*   **EL2 (Hypervisor):** Used for hardware virtualization. Manages multiple guest OSs running in EL1.
*   **EL3 (Secure Monitor):** The highest privilege level. Used by ARM TrustZone to manage the transition between the "Normal World" (Linux/Android) and the "Secure World" (a trusted OS handling DRM or cryptography).

### Hardware Virtualization
Originally, x86 was not "strictly virtualizable" because certain privileged instructions failed silently when executed in user mode rather than trapping.
*   **Intel VT-x / AMD-V:** Added new CPU modes (VMX Root and VMX Non-Root) and hardware structures (VMCS - Virtual Machine Control Structure) to allow a hypervisor to safely and efficiently run unmodified guest operating systems. The hardware automatically handles the translation of Guest Physical Addresses to Host Physical Addresses using Extended Page Tables (EPT).

## 10. Memory Management Unit (MMU)

The ISA defines the architecture of the MMU, which translates Virtual Addresses to Physical Addresses.
*   **Paging:** Memory is divided into fixed-size chunks (usually 4KB).
*   **Page Tables:** The ISA defines the format of the page table entries (PTEs) in memory. A PTE contains the physical frame number and permissions (Read/Write/Execute, User/Supervisor).
*   **Security:** The NX (No-eXecute) or DEP bit in the page table prevents the CPU from fetching instructions from memory pages marked for data. This is a crucial defense against buffer overflow exploits.

## 11. Advanced Architectures in Detail

### 11.1 The x86-64 Architecture
*   **History:** Evolved from the 16-bit 8086 (1978) to the 32-bit 80386 (1985) to AMD's 64-bit extension (2003). It maintains backwards compatibility with all of them.
*   **Modes of Operation:**
    *   *Real Mode:* 16-bit, no memory protection, 1MB address space. How the CPU boots.
    *   *Protected Mode:* 32-bit, segmentation and paging, ring-based security.
    *   *Long Mode:* 64-bit, flat memory model, 16 GPRs.
*   **SIMD:** x86 relies heavily on extensions. SSE (128-bit), AVX/AVX2 (256-bit), and AVX-512 (512-bit registers) allow massive parallel data processing.
*   **Complexity:** The instruction set has grown to thousands of instructions.

### 11.2 ARMv8-A / AArch64
*   **Design:** A clean-slate 64-bit design that learned from the legacy of 32-bit ARM.
*   **Registers:** 31 general-purpose 64-bit registers.
*   **SIMD (NEON):** Advanced SIMD architecture with 32 128-bit registers.
*   **SVE (Scalable Vector Extension):** A revolutionary SIMD approach where the vector length is not defined by the ISA, but by the hardware implementation (from 128 to 2048 bits). The exact same binary code scales automatically to the hardware's capabilities without recompilation.

### 11.3 RISC-V
*   **Philosophy:** Open standard, highly modular, clean design without legacy baggage.
*   **Base ISA:** The bare minimum needed to build a compiler (e.g., `RV64I` provides 40 integer instructions).
*   **Standard Extensions:**
    *   `M`: Integer Multiply/Divide
    *   `A`: Atomic instructions for concurrency
    *   `F`/`D`: Single/Double precision floats
    *   `C`: Compressed 16-bit instructions (increases code density, reducing instruction cache misses)
    *   `V`: Vector extension (similar philosophy to ARM SVE)
*   **Custom Extensions:** The ISA explicitly reserves opcode space for companies to add proprietary, domain-specific instructions (e.g., custom AI accelerators) without breaking standard software.


## 12. Memory Consistency and Ordering Models

When discussing ISAs, especially in the context of multicore processors, the Memory Consistency Model is as important as the instruction set itself. It defines the rules about the order in which memory operations (loads and stores) appear to execute across different threads or cores.

### 12.1 The Need for Memory Models
In a single-threaded program running on a single core, memory operations appear to happen sequentially in the order dictated by the program (program order). However, to improve performance, modern microarchitectures aggressively reorder instructions using out-of-order execution, and they use complex memory hierarchies (store buffers, L1/L2 caches).

When multiple cores interact with shared memory, these optimizations become visible. If Core A writes to variable X and then to variable Y, Core B might see the write to Y *before* the write to X because of how store buffers flush to the shared L2 cache. The ISA must define what reorderings are legally allowed.

### 12.2 Sequential Consistency (SC)
The most intuitive model, defined by Leslie Lamport.
*   **Definition:** The result of any execution is the same as if the operations of all processors were executed in some sequential order, and the operations of each individual processor appear in this sequence in the order specified by its program.
*   **Implication:** No reordering of memory operations is allowed to be visible to other cores. Every memory access must globally synchronize.
*   **Performance:** Terrible. SC prohibits the use of store buffers and out-of-order memory execution. No modern high-performance ISA uses pure Sequential Consistency as its default hardware model.

### 12.3 Total Store Ordering (TSO) - The x86 Model
x86 and x86-64 use a relatively strong memory model known as TSO.
*   **Rules:**
    *   Reads are not reordered with other reads.
    *   Writes are not reordered with other writes (all cores see writes in the same order).
    *   Writes are not reordered with older reads.
    *   **Crucial Exception:** A read *can* be reordered ahead of an older, independent write.
*   **Hardware Implementation:** This maps perfectly to a processor that uses a FIFO **Store Buffer**. When a core executes a store, it puts it in the store buffer and continues executing. A subsequent load can fetch data immediately from the cache (or the store buffer), effectively bypassing the pending store.
*   **Pros:** Very easy for programmers to reason about. Most multi-threaded algorithms (like Peterson's algorithm) work correctly without explicit memory barriers.
*   **Cons:** Limits hardware optimization. The strict ordering prevents the CPU from merging stores or reordering cache misses aggressively.

### 12.4 Weak Ordering (WO) - The ARM and RISC-V Model
ARM, PowerPC, and RISC-V use a relaxed or weak memory model.
*   **Rules:** The hardware is allowed to reorder memory operations almost arbitrarily (Read-Read, Read-Write, Write-Read, Write-Write) as long as data dependencies within a single thread are respected.
*   **Implication:** If Core A writes X=1 then Y=1, Core B might read Y=1 but X=0.
*   **Memory Barriers (Fences):** To enforce ordering when it actually matters (e.g., when acquiring a lock or publishing a pointer to a data structure), the ISA provides explicit barrier instructions.
    *   *ARM:* `DMB` (Data Memory Barrier), `DSB` (Data Synchronization Barrier).
    *   *RISC-V:* `FENCE` (with specific bits to order Read-Read, Read-Write, etc.).
*   **Pros:** Hardware designers have maximum freedom to optimize the memory subsystem. Allows massive performance gains and power savings.
*   **Cons:** Extremely difficult to write lock-free concurrent software. Programmers must manually insert fences in exactly the right places.

## 13. Advanced Hardware Security Features in the ISA

As software vulnerabilities have grown more sophisticated, ISAs have evolved to provide hardware-level mitigations against exploitation.

### 13.1 Executable Space Protection (NX/DEP)
Historically, buffer overflows allowed attackers to write malicious machine code into data structures on the stack or heap, and then hijack the PC to execute it.
*   **The Fix:** The ISA added a "No-eXecute" (NX) bit to the page table entries (PTEs).
*   **Mechanism:** When the OS allocates memory for the stack or heap, it sets the NX bit. If the CPU attempts to fetch an instruction from a physical page with the NX bit set, the MMU triggers a hard Page Fault exception, immediately terminating the program.

### 13.2 Return-Oriented Programming (ROP) Defenses
With NX preventing the execution of injected code, attackers pivoted to ROP. They hijack the stack to string together existing snippets of executable code (called "gadgets") ending in a `RET` instruction to perform malicious actions.

*   **Pointer Authentication Codes (PAC) - ARMv8.3+:**
    *   Since 64-bit architectures don't use all 64 bits for addressing (usually only 48 bits are wired up), the upper 16 bits are "free".
    *   PAC uses a hardware cryptographic unit to sign a pointer (like a return address on the stack) using a secret key and a context value (like the SP), storing the signature in the upper bits.
    *   Before the pointer is used (e.g., before returning), an authentication instruction verifies the signature. If an attacker modifies the return address, the signature becomes invalid, and the CPU throws an exception.
*   **Branch Target Identification (BTI) - ARMv8.5+ & Intel CET:**
    *   Prevents attackers from jumping into the middle of a function to execute a gadget.
    *   When enabled, every valid indirect jump target (like the start of a function) must begin with a special `BTI` (or Intel `ENDBR`) instruction.
    *   If an indirect jump lands on an instruction that is *not* a BTI instruction, the CPU traps.

### 13.3 Trusted Execution Environments (TEEs)
To protect sensitive code (like DRM decryption or biometric validation) even if the entire OS kernel is compromised.
*   **ARM TrustZone:** Partitions the CPU into a "Secure World" and a "Normal World". The Secure World has its own memory, isolated by hardware. The Normal World cannot access Secure memory, but the Secure World can access Normal memory. Transitioning between them requires a hypervisor-like exception (SMC - Secure Monitor Call).
*   **Intel SGX (Software Guard Extensions):** Allows user-level applications to create encrypted enclaves in memory. The memory is decrypted by the CPU package only when it is pulled into the cache. Even a malicious OS kernel reading the physical RAM will only see ciphertext.

## 14. Real-World Assembly Paradigms

To truly understand an ISA, one must look at its assembly language. Consider a simple loop calculating the sum of an array of integers.

### 14.1 x86-64 Assembly (CISC)
```nasm
; RDI = pointer to array
; ECX = number of elements (counter)
; EAX = accumulator (sum)

    xor eax, eax        ; Zero out EAX (sum = 0)
    test ecx, ecx       ; Check if count is 0
    jz .done            ; If 0, jump to end
.loop:
    add eax, [rdi]      ; Add the integer at address [RDI] to EAX
    add rdi, 4          ; Move pointer to the next 32-bit integer
    dec ecx             ; Decrement counter
    jnz .loop           ; If counter != 0, jump back to .loop
.done:
```
*Note the memory operand in `ADD EAX, [RDI]` - a hallmark of CISC Register-Memory architecture.*

### 14.2 ARMv8 AArch64 Assembly (RISC)
```nasm
; X0 = pointer to array
; W1 = number of elements (counter)
; W2 = accumulator (sum)

    mov w2, #0          ; Zero out W2
    cbz w1, .done       ; Compare Branch Zero: if W1 == 0, go to .done
.loop:
    ldr w3, [x0], #4    ; Load 32-bit int from [X0] into W3, then post-increment X0 by 4
    add w2, w2, w3      ; Add W3 into our sum W2
    sub w1, w1, #1      ; Decrement counter
    cbnz w1, .loop      ; Compare Branch Not Zero: if W1 != 0, loop
.done:
```
*Note the Load-Store nature (must use `LDR` to get memory) and the powerful post-increment addressing mode `[X0], #4`.*

### 14.3 RISC-V RV32I Assembly (RISC)
```nasm
; a0 = pointer to array
; a1 = number of elements
; a2 = accumulator (sum)

    li a2, 0            ; Load immediate 0 into a2
    beqz a1, .done      ; Branch if equal to zero
.loop:
    lw t0, 0(a0)        ; Load Word from address a0+0 into temp register t0
    add a2, a2, t0      ; Add to sum
    addi a0, a0, 4      ; Increment pointer by 4 bytes
    addi a1, a1, -1     ; Decrement counter
    bnez a1, .loop      ; Branch if not equal to zero
.done:
```
*Very clean, simple, orthagonal instructions. No auto-increment addressing mode means explicit `addi` instructions are required.*
