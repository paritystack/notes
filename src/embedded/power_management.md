# Power Management

Power management refers to the process of managing the power consumption of a device or system to optimize energy efficiency and prolong battery life. It is crucial in various applications, especially in portable devices like smartphones, laptops, and IoT devices. Modern power management involves a sophisticated combination of hardware mechanisms, software policies, and system-level optimizations that work together to minimize energy consumption while maintaining performance requirements.

## Key Concepts

### Sleep Modes

Sleep modes are low-power states that processors and peripherals can enter when idle. These modes trade off power savings against wake-up latency and state retention.

#### Processor-Specific Sleep States

**ARM Cortex-M Sleep Modes:**
- **Sleep Mode**: CPU clock stopped, peripherals continue running. Wake on any interrupt.
- **Deep Sleep**: CPU and most peripherals stopped, only wake-up peripherals active. Requires reconfiguration on wake.
- **Standby/Shutdown**: Minimal power consumption, only RTC and wake pins active. Full system reset on wake.

```c
// ARM Cortex-M sleep mode entry
void enter_sleep_mode(void) {
    __WFI();  // Wait For Interrupt - enters sleep mode
}

void enter_deep_sleep(void) {
    SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;  // Set SLEEPDEEP bit
    __WFI();  // Enter deep sleep
}
```

**ARM Cortex-A (Linux) C-States:**
- **C0**: Active/running state
- **C1**: Halt/WFI - lowest latency (~1-10μs), CPU execution stopped
- **C2**: Stop clock - medium latency (~50-100μs), CPU and cache clocks gated
- **C3**: Sleep - higher latency (~200-500μs), cache flushed, voltage may be reduced

**x86 Processor C-States:**
- **C0**: Active
- **C1**: Auto HALT - ~1μs wake latency
- **C3**: Deep Sleep - ~100μs wake latency, L1/L2 cache flushed
- **C6**: Deep Power Down - ~200μs latency, core voltage reduced
- **C7/C8**: Deeper states with progressively lower power and higher latency

```c
// Linux kernel - entering idle state
void cpu_idle(void) {
    while (!need_resched()) {
        if (cpuidle_idle_call())
            pm_idle();  // Architecture-specific idle function
    }
}
```

#### Wake-up Latency vs. Power Savings

| Sleep Mode | Power Savings | Wake Latency | State Retention |
|------------|---------------|--------------|-----------------|
| C1/WFI     | 10-20%        | 1-10 μs      | Full           |
| C2/C3      | 50-70%        | 50-500 μs    | Partial        |
| Deep Sleep | 90-95%        | 1-10 ms      | Minimal        |
| Shutdown   | 99%+          | 100+ ms      | None           |

### Dynamic Voltage and Frequency Scaling (DVFS)

DVFS adjusts processor voltage and frequency dynamically based on workload, exploiting the relationship: **Power ∝ V² × f**

#### Voltage-Frequency Relationship

Processors have defined operating points (OPPs) that pair voltage levels with maximum safe frequencies:

```
Frequency (MHz)  Voltage (V)  Power (mW)
1500            1.20         1800
1200            1.10         1210
800             1.00         640
400             0.90         324
```

Reducing frequency from 1500 MHz to 800 MHz saves ~64% power.

#### DVFS Governors

**Performance Governor**: Always runs at maximum frequency
```bash
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

**Powersave Governor**: Always runs at minimum frequency

**Ondemand Governor**: Rapidly scales to maximum frequency on load, gradually decreases when idle
- Fast response to workload increases
- Check interval typically 10-100ms
- Threshold-based scaling

**Conservative Governor**: Gradual frequency changes
- Slower ramp-up than ondemand
- Better for steady workloads

**Schedutil Governor**: Uses scheduler utilization data
- Integrated with task scheduler
- More responsive to transient loads

#### Implementation Example

```c
// Setting CPU frequency on Linux
#include <cpufreq.h>

int set_cpu_frequency(unsigned int cpu, unsigned long target_freq) {
    unsigned long min, max;
    cpufreq_get_hardware_limits(cpu, &min, &max);

    if (target_freq < min || target_freq > max)
        return -EINVAL;

    return cpufreq_set_frequency(cpu, target_freq);
}
```

Device Tree DVFS configuration:
```dts
cpu0_opp_table: opp-table {
    compatible = "operating-points-v2";

    opp-400000000 {
        opp-hz = /bits/ 64 <400000000>;
        opp-microvolt = <900000>;
        clock-latency-ns = <300000>;
    };
    opp-800000000 {
        opp-hz = /bits/ 64 <800000000>;
        opp-microvolt = <1000000>;
        clock-latency-ns = <300000>;
    };
    opp-1200000000 {
        opp-hz = /bits/ 64 <1200000000>;
        opp-microvolt = <1100000>;
        clock-latency-ns = <300000>;
    };
};
```

### Power Gating

Power gating completely shuts off power to unused circuit blocks, eliminating both static and dynamic power consumption.

#### Technical Implementation

**Key Components:**
- **Isolation Cells**: Prevent floating signals from powered-down blocks from affecting active circuits
- **Retention Registers**: Special flip-flops that maintain state during power-down (with backup power)
- **Power Switches**: Large transistors (typically PMOS) that control power domains
- **Power State Machine**: Controls sequencing of isolation, power switch control, and retention

#### Power Domain Control

```c
// Power domain control example (vendor-specific)
struct power_domain {
    void __iomem *pmu_base;
    u32 domain_id;
};

int power_domain_on(struct power_domain *pd) {
    // 1. Turn on retention
    writel(RETENTION_ON, pd->pmu_base + RETENTION_CTRL);

    // 2. Enable power switch
    writel(POWER_ON, pd->pmu_base + POWER_CTRL);
    udelay(10);  // Wait for power stabilization

    // 3. Disable isolation
    writel(ISOLATION_OFF, pd->pmu_base + ISO_CTRL);

    // 4. Restore clocks
    writel(CLOCK_ON, pd->pmu_base + CLOCK_CTRL);

    return 0;
}

int power_domain_off(struct power_domain *pd) {
    // Reverse sequence
    writel(CLOCK_OFF, pd->pmu_base + CLOCK_CTRL);
    writel(ISOLATION_ON, pd->pmu_base + ISO_CTRL);
    writel(POWER_OFF, pd->pmu_base + POWER_CTRL);

    return 0;
}
```

#### Power Domain Design Considerations

- **Granularity**: Fine-grained domains offer more savings but increase complexity
- **Isolation overhead**: Isolation cells add area and delay
- **Power-up sequencing**: Incorrect ordering can cause latch-up or data corruption
- **Retention power**: Retention registers still consume power

### Clock Gating

Clock gating disables clock signals to idle circuit blocks, eliminating dynamic power from clock tree switching.

#### Types of Clock Gating

**Coarse-Grained**: Module-level gating
```c
// Peripheral clock control
#define RCC_APB1ENR (*(volatile uint32_t*)0x40023840)

void uart_clock_enable(void) {
    RCC_APB1ENR |= (1 << 17);  // Enable USART2 clock
}

void uart_clock_disable(void) {
    RCC_APB1ENR &= ~(1 << 17);  // Disable USART2 clock
}
```

**Fine-Grained**: Register-level gating
- Automatically inserted by synthesis tools
- Uses enable signal to gate clock to registers
- Typical power savings: 20-40% of dynamic power

```verilog
// Hardware clock gating cell
module clock_gate (
    input clk,
    input enable,
    output gated_clk
);
    reg enable_latch;

    always @(clk or enable)
        if (!clk)
            enable_latch <= enable;

    assign gated_clk = clk & enable_latch;
endmodule
```

#### Linux Runtime PM Framework

```c
#include <linux/pm_runtime.h>

// Enable runtime PM for a device
static int my_driver_probe(struct platform_device *pdev) {
    pm_runtime_enable(&pdev->dev);
    pm_runtime_get_sync(&pdev->dev);  // Increment usage count, power on

    // Initialize device...

    pm_runtime_put(&pdev->dev);  // Decrement usage count, may power down
    return 0;
}

// Runtime suspend callback
static int my_driver_runtime_suspend(struct device *dev) {
    struct my_device *mydev = dev_get_drvdata(dev);

    // Disable clocks
    clk_disable_unprepare(mydev->clk);

    return 0;
}

// Runtime resume callback
static int my_driver_runtime_resume(struct device *dev) {
    struct my_device *mydev = dev_get_drvdata(dev);

    // Enable clocks
    return clk_prepare_enable(mydev->clk);
}
```

## Power Measurement and Profiling

### Measurement Techniques

#### Hardware Measurement Methods

**Shunt Resistor Method:**
- Insert low-value resistor (0.01-0.1Ω) in power path
- Measure voltage drop across resistor: I = V/R
- High accuracy but requires hardware modification

**Current Probe:**
- Clamp-on or inline current probes
- Non-invasive for clamp-on type
- Bandwidth considerations for dynamic measurements

**Power Monitoring ICs:**
- Dedicated ICs like INA219, INA3221, PAC1934
- Integrated shunt resistor and ADC
- I2C/SPI interface for software monitoring

```c
// INA219 power monitor example
#include "ina219.h"

float measure_system_power(void) {
    float voltage = ina219_read_bus_voltage();  // V
    float current = ina219_read_current();      // mA
    return voltage * current / 1000.0;          // mW
}
```

#### Software Profiling Tools

**Linux PowerTOP:**
```bash
sudo powertop --html=power_report.html
```
- Identifies power-hungry processes and devices
- Provides tuning suggestions
- Estimates power consumption by component

**perf power events:**
```bash
perf stat -e power/energy-cores/,power/energy-pkg/ ./my_app
```

**ARM Streamline:**
- CPU frequency and voltage tracking
- Per-core power estimation
- GPU and memory power profiling

**Vendor-Specific Tools:**
- Intel VTune (x86 energy analysis)
- NVIDIA Nsight (GPU power profiling)
- Qualcomm Trepn (mobile power profiling)

### Power Budgeting

Power budget allocation for a typical embedded system:

| Component      | Active (mW) | Idle (mW) | Sleep (μW) |
|---------------|-------------|-----------|------------|
| CPU Core      | 300-1000    | 50-100    | 10-50      |
| RAM           | 50-200      | 20-50     | 5-10       |
| Flash         | 20-50       | 1-5       | 0.1-1      |
| Wireless      | 100-300     | 10-20     | 1-5        |
| Sensors       | 10-50       | 1-5       | 0.1-1      |
| Display       | 200-800     | 0         | 0          |
| **Total**     | **680-2400**| **82-180**| **16-67**  |

## Thermal Management

Thermal management is intrinsically linked to power management: **Power = Heat**

### Thermal Throttling

When junction temperature (Tj) exceeds threshold, reduce performance to lower power:

```c
// Thermal throttling implementation
#define TEMP_THRESHOLD_HIGH  85.0  // °C
#define TEMP_THRESHOLD_LOW   75.0  // °C

void thermal_monitor_task(void) {
    float temp = read_junction_temperature();

    if (temp > TEMP_THRESHOLD_HIGH) {
        // Reduce to 50% frequency
        set_cpu_frequency(get_max_frequency() / 2);
        throttled = true;
    }
    else if (throttled && temp < TEMP_THRESHOLD_LOW) {
        // Restore full frequency
        set_cpu_frequency(get_max_frequency());
        throttled = false;
    }
}
```

### Thermal Design Power (TDP)

TDP represents the maximum sustained power dissipation under realistic workloads:
- Used for cooling system design
- Typically lower than absolute maximum power
- Example: Intel i7 TDP = 65W, max turbo = 90W+

### Cooling Strategies

**Passive Cooling:**
- Heat sinks with thermal interface material (TIM)
- Thermal spreading planes in PCB
- Natural convection

**Active Cooling:**
- Fans with PWM speed control
- Liquid cooling for high-power systems
- Thermoelectric coolers (Peltier devices)

## Battery Management

### Battery Chemistry Characteristics

| Chemistry | Voltage | Energy Density | Cycle Life | Self-Discharge |
|-----------|---------|----------------|------------|----------------|
| Li-Ion    | 3.7V    | 150-200 Wh/kg  | 500-1000   | ~5% /month     |
| Li-Po     | 3.7V    | 180-250 Wh/kg  | 300-500    | ~5% /month     |
| LiFePO4   | 3.2V    | 90-120 Wh/kg   | 2000-5000  | ~3% /month     |
| NiMH      | 1.2V    | 60-100 Wh/kg   | 500-1000   | ~30% /month    |

### Battery Management Systems (BMS)

**Key Functions:**
- **Voltage monitoring**: Per-cell voltage measurement
- **Current monitoring**: Charge/discharge current tracking
- **State of Charge (SOC)**: Remaining capacity estimation
- **State of Health (SOH)**: Battery degradation tracking
- **Cell balancing**: Equalize voltages across series cells
- **Protection**: Overvoltage, undervoltage, overcurrent, overtemperature

#### Fuel Gauging

**Coulomb Counting:**
```c
// Simple coulomb counter
float battery_capacity_mah = 2500.0;
float current_charge_mah = 2500.0;

void update_battery_charge(float current_ma, float time_s) {
    // Integrate current over time
    current_charge_mah -= (current_ma * time_s / 3600.0);

    // Estimate SOC
    float soc = (current_charge_mah / battery_capacity_mah) * 100.0;
}
```

**Voltage-based Estimation:**
- Uses discharge curve lookup table
- Less accurate, affected by load and temperature
- Simple, no current measurement needed

**Impedance Tracking:**
- Measures internal resistance changes
- Better SOC accuracy under varying loads
- Used in advanced fuel gauge ICs (e.g., Texas Instruments bq series)

### Charging Algorithms

**Constant Current / Constant Voltage (CC/CV):**
1. **CC Phase**: Charge at constant current (typically 0.5-1C) until voltage reaches 4.2V
2. **CV Phase**: Hold voltage at 4.2V, current decreases exponentially
3. **Termination**: Stop when current drops below 0.05-0.1C

```c
// CC/CV charging state machine
typedef enum {
    CHARGE_IDLE,
    CHARGE_CC,
    CHARGE_CV,
    CHARGE_COMPLETE
} charge_state_t;

charge_state_t charging_update(void) {
    float voltage = read_battery_voltage();
    float current = read_charge_current();

    switch (charge_state) {
        case CHARGE_CC:
            set_charge_current(CHARGE_CURRENT_MAX);
            if (voltage >= CHARGE_VOLTAGE_MAX)
                charge_state = CHARGE_CV;
            break;

        case CHARGE_CV:
            set_charge_voltage(CHARGE_VOLTAGE_MAX);
            if (current < CHARGE_CURRENT_TERMINATION)
                charge_state = CHARGE_COMPLETE;
            break;
    }

    return charge_state;
}
```

## Best Practices and Design Patterns

### Power-Aware Software Architecture

**Event-Driven vs. Polling:**
```c
// Bad: Polling wastes power
while (1) {
    if (button_pressed()) {
        handle_button();
    }
    // CPU stays active checking button
}

// Good: Event-driven
void button_irq_handler(void) {
    handle_button();
}

while (1) {
    enter_sleep_mode();  // Sleep until interrupt
}
```

**Batching Operations:**
```c
// Bad: Frequent wake-ups
void periodic_sensor_read(void) {
    wake_sensor();
    value = read_sensor();
    sleep_sensor();
    // Process immediately
}

// Good: Batch processing
void batched_sensor_read(void) {
    wake_sensor();
    for (int i = 0; i < BATCH_SIZE; i++) {
        buffer[i] = read_sensor();
        delay_ms(10);
    }
    sleep_sensor();
    // Process batch
}
```

### Peripheral Power Management Strategy

**Systematic Shutdown:**
1. Identify all peripherals and their power modes
2. Create power state matrix
3. Implement coordinated shutdown sequence

```c
typedef struct {
    bool uart_needed;
    bool spi_needed;
    bool i2c_needed;
    bool adc_needed;
} system_requirements_t;

void configure_power_state(system_requirements_t *req) {
    if (!req->uart_needed) {
        uart_clock_disable();
    }
    if (!req->spi_needed) {
        spi_clock_disable();
    }
    // etc...
}
```

### DMA for Power Efficiency

DMA transfers allow CPU to sleep during data movement:

```c
// Without DMA: CPU stays active
void copy_data_polling(uint8_t *src, uint8_t *dst, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dst[i] = src[i];  // CPU active entire time
    }
}

// With DMA: CPU can sleep
void copy_data_dma(uint8_t *src, uint8_t *dst, size_t len) {
    dma_configure(src, dst, len);
    dma_start();
    enter_sleep_mode();  // CPU sleeps, DMA works
    // Wake on DMA completion interrupt
}
```

## Common Pitfalls and Debugging

### Race Conditions in Power State Transitions

**Problem**: Peripheral access during power-down
```c
// Dangerous: UART might be powered down
void send_data(char *data) {
    uart_transmit(data);  // What if UART clock is disabled?
}
```

**Solution**: Reference counting or state checks
```c
struct peripheral_context {
    atomic_t ref_count;
    bool powered;
};

int uart_acquire(struct peripheral_context *ctx) {
    if (atomic_inc_return(&ctx->ref_count) == 1) {
        uart_power_on();
        ctx->powered = true;
    }
    return 0;
}

void uart_release(struct peripheral_context *ctx) {
    if (atomic_dec_return(&ctx->ref_count) == 0) {
        uart_power_off();
        ctx->powered = false;
    }
}
```

### Wake-up Latency Violations

Real-time tasks must account for wake-up latency:
```c
void time_critical_task(void) {
    // Allow time for wake-up
    uint32_t wakeup_latency_us = get_current_sleep_latency();
    uint32_t deadline_us = get_task_deadline();

    if (wakeup_latency_us > deadline_us) {
        // Constrain sleep mode
        set_max_sleep_state(SHALLOW_SLEEP);
    }
}
```

### Debugging Power Issues

**Common Tools and Techniques:**

1. **Current measurement with timestamps:**
   - Correlate current spikes with system events
   - Use oscilloscope with current probe

2. **GPIO toggling for profiling:**
```c
void enter_sleep(void) {
    GPIO_SET_LOW(DEBUG_PIN);  // Visible on logic analyzer
    __WFI();
    GPIO_SET_HIGH(DEBUG_PIN);
}
```

3. **Power state logging:**
```c
void log_power_event(const char *event, uint32_t timestamp) {
    // Circular buffer for power state transitions
    power_log[log_idx].timestamp = timestamp;
    strncpy(power_log[log_idx].event, event, 32);
    log_idx = (log_idx + 1) % LOG_SIZE;
}
```

### Unexpected Power Consumption Sources

- **Pull-up/pull-down resistors on floating pins**: Can cause current drain
- **Analog peripherals**: ADC/DAC can consume significant static power
- **Oscillators**: External crystals may not stop in some sleep modes
- **LEDs**: Often forgotten, can consume 5-20mA each
- **Voltage regulators**: Quiescent current varies widely (1μA to 1mA)

## Performance vs. Power Trade-offs

### Quantitative Analysis

**Race-to-Sleep Strategy:**
Run at maximum frequency to complete task quickly, then enter deep sleep.

**Example Calculation:**
Task requires 1M CPU cycles

**Option 1**: 100 MHz, 100mW active, 1mW sleep
- Time: 10ms active, 90ms sleep
- Energy: (100mW × 10ms) + (1mW × 90ms) = 1.09 mJ

**Option 2**: 50 MHz, 60mW active, 1mW sleep
- Time: 20ms active, 80ms sleep
- Energy: (60mW × 20ms) + (1mW × 80ms) = 1.28 mJ

Race-to-sleep wins by 15% energy savings!

### Real-Time Constraints

**Latency-Sensitive Applications:**
- Audio processing: Cannot tolerate >10ms interruptions
- Motor control: Requires <1ms response times
- Communication protocols: Strict timing requirements

**Strategy**: Use QoS (Quality of Service) requirements
```c
struct pm_qos_request {
    int latency_us;
    int throughput_kbps;
};

void set_latency_requirement(int max_latency_us) {
    // Constrain sleep states based on wake latency
    if (max_latency_us < 100)
        disable_deep_sleep();
    else
        enable_deep_sleep();
}
```

### Energy Proportional Computing

Ideal system: Energy consumption proportional to utilization
- Real systems: Fixed overhead power
- Challenge: Minimize idle power while maintaining responsiveness

## Applications

### Mobile Devices

Modern smartphones employ aggressive power management:
- **Display**: Largest power consumer (200-800mW), adaptive brightness, OLED pixel-level control
- **Cellular modem**: Discontinuous reception (DRX), power class adaptation
- **Application processor**: big.LITTLE architecture (high-performance + efficient cores)
- **GPU**: Per-block power gating, DVFS
- **Sensors**: Always-on low-power sensor hub, batched sensor data

**Typical power profile:**
- Screen on, active use: 2-4W
- Screen off, background: 50-200mW
- Airplane mode, sleep: 5-20mW
- Powered off (RTC only): <1mW

### Data Centers

Power and cooling represent 30-50% of operational costs:
- **Server-level**: CPU power capping, DVFS, core parking
- **Rack-level**: Dynamic power allocation across servers
- **Facility-level**: Free cooling, hot/cold aisle containment, renewable energy integration

**Power Usage Effectiveness (PUE):**
PUE = Total Facility Power / IT Equipment Power
- Typical PUE: 1.5-2.0
- Efficient designs: 1.1-1.3
- Ideal (theoretical): 1.0

### Embedded Systems

Battery-powered embedded systems prioritize ultra-low power:

**Energy Harvesting Applications:**
- Solar: 10-100 μW/cm² indoors, 10-100 mW/cm² outdoors
- Thermoelectric: 10-100 μW/cm²
- Vibration: 1-100 μW/cm³
- RF harvesting: 0.1-10 μW/cm²

**Power budgets must match harvesting rates:**
```c
// Energy-neutral operation
float harvested_power_uw = 50.0;
float active_power_uw = 5000.0;
float sleep_power_uw = 5.0;
float duty_cycle = (harvested_power_uw - sleep_power_uw) /
                   (active_power_uw - sleep_power_uw);
// duty_cycle ≈ 0.9% (9ms active per second)
```

## References and Standards

### Industry Standards

**ACPI (Advanced Configuration and Power Interface):**
- Defines sleep states (S0-S5), performance states (P-states), processor states (C-states)
- Standard for x86 systems
- Specification: https://uefi.org/specifications

**ARM Power State Coordination Interface (PSCI):**
- Standard power management interface for ARM systems
- Defines CPU_SUSPEND, CPU_OFF, SYSTEM_OFF operations

**IEEE 1801 (UPF - Unified Power Format):**
- Standard for specifying power intent in chip design
- Defines power domains, isolation, retention strategies

**Energy Star:**
- EPA program for energy efficiency
- Defines maximum idle power consumption for computers and servers

### Key Specifications

- **USB Power Delivery (USB-PD)**: Up to 240W power delivery specification
- **Qi Wireless Charging**: Inductive power transfer standard
- **Bluetooth Low Energy (BLE)**: Low-power wireless protocol (~10-20mA active, <1μA sleep)
- **LoRaWAN**: Long-range, low-power wide-area network protocol

### Recommended Resources

**Books:**
- "Low-Power CMOS Design" by Anantha Chandrakasan
- "Power Management Integrated Circuits" by Maloberti and Davies

**Online Resources:**
- Linux kernel power management documentation: https://www.kernel.org/doc/html/latest/power/
- ARM power management documentation
- Intel Software Developer Manuals (Volume 3B: System Programming)

## Conclusion

Effective power management is a multi-faceted challenge requiring careful coordination between hardware design, firmware implementation, and software policies. Modern systems employ a hierarchy of power management techniques—from transistor-level clock gating to system-level thermal management—to achieve optimal energy efficiency.

Key takeaways:
- **No single solution**: Combine multiple techniques (DVFS, sleep modes, power gating, clock gating)
- **Measure, don't guess**: Use profiling tools to identify actual power consumers
- **Consider the whole system**: CPU power is often dominated by peripherals and I/O
- **Trade-offs are inevitable**: Balance power, performance, latency, and cost
- **Software matters**: Poor software design can negate hardware power savings

As devices become more complex and energy constraints tighten, power management will continue to be a critical differentiator in system design. The principles outlined in this document provide a foundation for developing energy-efficient embedded systems that meet the demanding requirements of modern applications while maximizing battery life and minimizing operational costs.
