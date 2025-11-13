# Device Tree

A comprehensive guide to Linux Device Tree, a data structure for describing hardware configuration that can be passed to the kernel at boot time.

## Table of Contents

- [Overview](#overview)
- [Why Device Tree?](#why-device-tree)
- [Device Tree Basics](#device-tree-basics)
- [Device Tree Syntax](#device-tree-syntax)
- [Device Tree Structure](#device-tree-structure)
- [Standard Properties](#standard-properties)
- [Writing Device Tree Files](#writing-device-tree-files)
- [Device Tree Compiler](#device-tree-compiler)
- [Parsing Device Tree in Drivers](#parsing-device-tree-in-drivers)
- [Common Bindings](#common-bindings)
- [Platform-Specific Details](#platform-specific-details)
- [Debugging Device Tree](#debugging-device-tree)
- [Best Practices](#best-practices)
- [Real-World Examples](#real-world-examples)

---

## Overview

**Device Tree** is a data structure and language for describing hardware that cannot be dynamically detected by the operating system. It's used extensively in embedded systems, especially ARM-based platforms.

### Key Concepts

- **Device Tree Source (.dts)**: Human-readable text file describing hardware
- **Device Tree Blob (.dtb)**: Compiled binary format loaded by bootloader
- **Device Tree Overlay (.dtbo)**: Runtime modifications to base device tree
- **Bindings**: Documentation defining properties for specific device types

### Purpose

```
+-----------------+
|   Bootloader    |
|   (U-Boot)      |
+-----------------+
        |
        | Passes DTB
        v
+-----------------+
|  Linux Kernel   |
+-----------------+
        |
        | Parses DT
        v
+-----------------+
| Device Drivers  |
+-----------------+
```

Device Tree allows:
- Hardware description separated from kernel code
- Single kernel binary supporting multiple boards
- Board-specific configuration without recompiling kernel
- Runtime hardware configuration via overlays

---

## Why Device Tree?

### Problems Device Tree Solves

**Before Device Tree:**
```c
/* ARM board file - arch/arm/mach-vendor/board-xyz.c */
static struct platform_device uart0 = {
	.name = "vendor-uart",
	.id = 0,
	.resource = {
		.start = 0x44e09000,
		.end = 0x44e09fff,
		.flags = IORESOURCE_MEM,
	},
	.dev = {
		.platform_data = &uart0_data,
	},
};

platform_device_register(&uart0);
```

**Problems:**
- Board-specific code in kernel
- One kernel per board variant
- Difficult to maintain
- No standardization

**With Device Tree:**
```dts
uart0: serial@44e09000 {
	compatible = "vendor,uart";
	reg = <0x44e09000 0x1000>;
	interrupts = <72>;
	clock-frequency = <48000000>;
};
```

**Benefits:**
- Hardware description in separate file
- Single kernel for multiple boards
- Standardized bindings
- Easier to maintain

---

## Device Tree Basics

### Device Tree Hierarchy

Device Tree represents hardware as a tree of nodes:

```dts
/ {
	model = "Vendor Board XYZ";
	compatible = "vendor,board-xyz";

	cpus {
		cpu@0 {
			compatible = "arm,cortex-a8";
			device_type = "cpu";
			reg = <0>;
		};
	};

	memory@80000000 {
		device_type = "memory";
		reg = <0x80000000 0x20000000>;  /* 512MB */
	};

	soc {
		compatible = "simple-bus";
		#address-cells = <1>;
		#size-cells = <1>;
		ranges;

		uart0: serial@44e09000 {
			compatible = "vendor,uart";
			reg = <0x44e09000 0x1000>;
		};
	};
};
```

### Key Terminology

- **Node**: Represents a device or bus (`uart0`, `cpus`)
- **Property**: Key-value pair in a node (`compatible = "vendor,uart"`)
- **Label**: Reference to a node (`uart0:`)
- **Phandle**: Reference to another node (pointer to node)
- **Unit Address**: Address part after `@` (`serial@44e09000`)

---

## Device Tree Syntax

### Basic Syntax

```dts
/* Comments use C-style syntax */

/ {
	/* Root node - always present */

	node-name {
		/* Properties */
		property-name = "string value";
		another-property = <0x12345678>;
		multi-value = <0x1 0x2 0x3>;
		boolean-property;  /* Presence indicates true */
	};

	node@unit-address {
		/* Node with unit address */
		reg = <0x12340000 0x1000>;
	};
};
```

### Property Value Types

```dts
/ {
	/* String */
	model = "Vendor Board XYZ";

	/* String list */
	compatible = "vendor,board-xyz", "vendor,board";

	/* 32-bit unsigned integers (cells) */
	reg = <0x44e09000 0x1000>;

	/* Multiple cells */
	interrupts = <0 72 4>;

	/* Boolean (empty property) */
	dma-coherent;

	/* Byte sequence */
	mac-address = [00 11 22 33 44 55];

	/* Mixed */
	property = "string", <0x1234>, [AB CD];

	/* Phandle reference */
	interrupt-parent = <&intc>;
	clocks = <&osc 0>;
};
```

### Cell Size Specifiers

```dts
/ {
	#address-cells = <1>;  /* Address takes 1 cell (32-bit) */
	#size-cells = <1>;     /* Size takes 1 cell */

	soc {
		#address-cells = <1>;
		#size-cells = <1>;

		/* reg = <address size> */
		uart0: serial@44e09000 {
			reg = <0x44e09000 0x1000>;
		};
	};
};

/ {
	#address-cells = <2>;  /* 64-bit addressing */
	#size-cells = <2>;

	memory@0 {
		/* reg = <address-high address-low size-high size-low> */
		reg = <0x00000000 0x80000000 0x00000000 0x40000000>;
	};
};
```

### Labels and References

```dts
/ {
	/* Define label */
	intc: interrupt-controller@48200000 {
		compatible = "arm,gic";
		reg = <0x48200000 0x1000>;
		interrupt-controller;
		#interrupt-cells = <3>;
	};

	uart0: serial@44e09000 {
		compatible = "vendor,uart";
		/* Reference using phandle */
		interrupt-parent = <&intc>;
		interrupts = <0 72 4>;
		clocks = <&sysclk>;
	};
};
```

### Includes

```dts
/* Include common definitions */
/include/ "vendor-common.dtsi"

/* Or using C preprocessor */
#include "vendor-common.dtsi"
#include <dt-bindings/gpio/gpio.h>

/ {
	compatible = "vendor,board";
};
```

---

## Device Tree Structure

### Complete Example

```dts
/dts-v1/;

/ {
	model = "Vendor Development Board";
	compatible = "vendor,dev-board", "vendor,soc";

	#address-cells = <1>;
	#size-cells = <1>;

	chosen {
		bootargs = "console=ttyS0,115200 root=/dev/mmcblk0p2";
		stdout-path = "/serial@44e09000:115200n8";
	};

	memory@80000000 {
		device_type = "memory";
		reg = <0x80000000 0x40000000>;  /* 1GB */
	};

	cpus {
		#address-cells = <1>;
		#size-cells = <0>;

		cpu0: cpu@0 {
			compatible = "arm,cortex-a8";
			device_type = "cpu";
			reg = <0>;
			operating-points = <
				/* kHz    uV */
				1000000 1350000
				800000  1300000
				600000  1200000
			>;
			clock-latency = <300000>; /* 300 us */
		};
	};

	clocks {
		osc: oscillator {
			compatible = "fixed-clock";
			#clock-cells = <0>;
			clock-frequency = <24000000>;
		};

		sysclk: system-clock {
			compatible = "fixed-clock";
			#clock-cells = <0>;
			clock-frequency = <48000000>;
		};
	};

	soc {
		compatible = "simple-bus";
		#address-cells = <1>;
		#size-cells = <1>;
		ranges;

		intc: interrupt-controller@48200000 {
			compatible = "arm,cortex-a8-gic";
			interrupt-controller;
			#interrupt-cells = <3>;
			reg = <0x48200000 0x1000>,
			      <0x48210000 0x2000>;
		};

		uart0: serial@44e09000 {
			compatible = "vendor,uart", "ns16550a";
			reg = <0x44e09000 0x1000>;
			interrupt-parent = <&intc>;
			interrupts = <0 72 4>;
			clocks = <&sysclk>;
			clock-names = "uart";
			status = "okay";
		};

		i2c0: i2c@44e0b000 {
			compatible = "vendor,i2c";
			reg = <0x44e0b000 0x1000>;
			interrupts = <0 70 4>;
			#address-cells = <1>;
			#size-cells = <0>;
			clocks = <&sysclk>;
			status = "okay";

			/* I2C device */
			eeprom@50 {
				compatible = "atmel,24c256";
				reg = <0x50>;
				pagesize = <64>;
			};
		};

		gpio0: gpio@44e07000 {
			compatible = "vendor,gpio";
			reg = <0x44e07000 0x1000>;
			interrupts = <0 96 4>;
			gpio-controller;
			#gpio-cells = <2>;
			interrupt-controller;
			#interrupt-cells = <2>;
		};

		mmc0: mmc@48060000 {
			compatible = "vendor,mmc";
			reg = <0x48060000 0x1000>;
			interrupts = <0 64 4>;
			bus-width = <4>;
			cd-gpios = <&gpio0 6 0>;
			status = "okay";
		};
	};

	leds {
		compatible = "gpio-leds";

		led0 {
			label = "board:green:user0";
			gpios = <&gpio0 21 0>;
			linux,default-trigger = "heartbeat";
		};

		led1 {
			label = "board:green:user1";
			gpios = <&gpio0 22 0>;
			default-state = "off";
		};
	};

	regulators {
		compatible = "simple-bus";

		vdd_3v3: regulator@0 {
			compatible = "regulator-fixed";
			regulator-name = "vdd_3v3";
			regulator-min-microvolt = <3300000>;
			regulator-max-microvolt = <3300000>;
			regulator-always-on;
		};
	};
};
```

---

## Standard Properties

### Compatible Property

The `compatible` property is the most important - it binds the node to a driver:

```dts
uart0: serial@44e09000 {
	/* Most specific first, generic last */
	compatible = "vendor,soc-uart", "vendor,uart", "ns16550a";
	...
};
```

**Driver matching:**
```c
static const struct of_device_id uart_of_match[] = {
	{ .compatible = "vendor,soc-uart", .data = &soc_uart_data },
	{ .compatible = "vendor,uart", .data = &generic_uart_data },
	{ .compatible = "ns16550a", .data = &ns16550_data },
	{ }
};
MODULE_DEVICE_TABLE(of, uart_of_match);
```

### Reg Property

Specifies address ranges (MMIO, I2C address, SPI chip select):

```dts
/* MMIO register range */
uart0: serial@44e09000 {
	reg = <0x44e09000 0x1000>;  /* Base address, size */
};

/* Multiple ranges */
intc: interrupt-controller@48200000 {
	reg = <0x48200000 0x1000>,   /* Distributor */
	      <0x48210000 0x2000>;   /* CPU interface */
};

/* I2C device */
eeprom@50 {
	reg = <0x50>;  /* I2C address */
};

/* SPI device */
flash@0 {
	reg = <0>;  /* Chip select 0 */
};
```

### Status Property

Enables or disables devices:

```dts
uart0: serial@44e09000 {
	status = "okay";     /* Enable */
};

uart1: serial@44e0a000 {
	status = "disabled"; /* Disable */
};

uart2: serial@44e0b000 {
	status = "fail";     /* Error detected */
};
```

### Interrupt Properties

```dts
uart0: serial@44e09000 {
	/* Parent interrupt controller */
	interrupt-parent = <&intc>;

	/* Interrupt specifier (format defined by parent) */
	/* For GIC: <type number flags> */
	interrupts = <0 72 4>;  /* SPI, IRQ 72, level-high */
};

/* Shared interrupt */
device@0 {
	interrupts = <0 50 4>;
	interrupt-names = "tx", "rx", "error";
};
```

### Clock Properties

```dts
uart0: serial@44e09000 {
	clocks = <&sysclk>, <&pclk>;
	clock-names = "uart", "apb_pclk";
};

/* Clock frequency for fixed clocks */
osc: oscillator {
	compatible = "fixed-clock";
	#clock-cells = <0>;
	clock-frequency = <24000000>;
};
```

### GPIO Properties

```dts
device {
	/* GPIO specifier: <&controller pin flags> */
	reset-gpios = <&gpio0 15 GPIO_ACTIVE_LOW>;
	enable-gpios = <&gpio0 16 GPIO_ACTIVE_HIGH>;
};

#include <dt-bindings/gpio/gpio.h>
/* GPIO_ACTIVE_LOW, GPIO_ACTIVE_HIGH */
```

### DMA Properties

```dts
uart0: serial@44e09000 {
	dmas = <&dma 25>, <&dma 26>;
	dma-names = "tx", "rx";
};
```

---

## Writing Device Tree Files

### Device Tree Source (.dts)

Board-specific file:

```dts
/dts-v1/;

#include "vendor-soc.dtsi"

/ {
	model = "Vendor Board XYZ";
	compatible = "vendor,board-xyz", "vendor,soc";

	memory@80000000 {
		device_type = "memory";
		reg = <0x80000000 0x40000000>;
	};
};

/* Enable and configure UART0 */
&uart0 {
	status = "okay";
	pinctrl-names = "default";
	pinctrl-0 = <&uart0_pins>;
};

/* Disable UART1 (not used on this board) */
&uart1 {
	status = "disabled";
};

/* Add I2C devices */
&i2c0 {
	status = "okay";
	clock-frequency = <400000>;

	/* Board-specific I2C device */
	rtc@68 {
		compatible = "dallas,ds1307";
		reg = <0x68>;
	};
};
```

### Device Tree Include (.dtsi)

SoC-level common definitions:

```dts
/* vendor-soc.dtsi */
/ {
	#address-cells = <1>;
	#size-cells = <1>;

	cpus {
		#address-cells = <1>;
		#size-cells = <0>;

		cpu@0 {
			compatible = "arm,cortex-a8";
			device_type = "cpu";
			reg = <0>;
		};
	};

	soc {
		compatible = "simple-bus";
		#address-cells = <1>;
		#size-cells = <1>;
		ranges;

		uart0: serial@44e09000 {
			compatible = "vendor,uart";
			reg = <0x44e09000 0x1000>;
			interrupts = <0 72 4>;
			clocks = <&sysclk>;
			status = "disabled";  /* Disabled by default */
		};

		uart1: serial@44e0a000 {
			compatible = "vendor,uart";
			reg = <0x44e0a000 0x1000>;
			interrupts = <0 73 4>;
			clocks = <&sysclk>;
			status = "disabled";
		};

		i2c0: i2c@44e0b000 {
			compatible = "vendor,i2c";
			reg = <0x44e0b000 0x1000>;
			interrupts = <0 70 4>;
			#address-cells = <1>;
			#size-cells = <0>;
			clocks = <&sysclk>;
			status = "disabled";
		};
	};
};
```

### Overriding and Extending Nodes

```dts
/* Base definition in .dtsi */
&uart0 {
	compatible = "vendor,uart";
	reg = <0x44e09000 0x1000>;
	status = "disabled";
};

/* Board-specific .dts */
&uart0 {
	status = "okay";
	pinctrl-names = "default";
	pinctrl-0 = <&uart0_pins>;
	/* Adds new properties while keeping existing ones */
};
```

### Deleting Nodes/Properties

```dts
/* Delete property */
&uart0 {
	/delete-property/ dmas;
	/delete-property/ dma-names;
};

/* Delete node */
&uart1 {
	/delete-node/ device@0;
};
```

---

## Device Tree Compiler

### Compiling Device Tree

```bash
# Compile .dts to .dtb
dtc -I dts -O dtb -o board.dtb board.dts

# With includes
dtc -I dts -O dtb -o board.dtb -i include_path board.dts

# Using C preprocessor
cpp -nostdinc -I include_path -undef -x assembler-with-cpp \
    board.dts board.preprocessed.dts
dtc -I dts -O dtb -o board.dtb board.preprocessed.dts
```

### Decompiling Device Tree

```bash
# Decompile .dtb to .dts
dtc -I dtb -O dts -o board.dts board.dtb

# With symbols for overlays
dtc -I dtb -O dts -o board.dts board.dtb -@
```

### Building with Kernel

```makefile
# In kernel Makefile
dtb-$(CONFIG_BOARD_XYZ) += board-xyz.dtb

# Build
make dtbs

# Output in: arch/arm/boot/dts/board-xyz.dtb
```

### Validation

```bash
# Check syntax
dtc -I dts -O dtb -o /dev/null board.dts

# Validate against schema (Linux 5.4+)
make dt_binding_check
make dtbs_check
```

---

## Parsing Device Tree in Drivers

### Getting Device Tree Node

```c
#include <linux/of.h>
#include <linux/of_device.h>

static int my_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct device_node *np = dev->of_node;

	if (!np) {
		dev_err(dev, "No device tree node\n");
		return -ENODEV;
	}

	/* Node is available */
	return 0;
}
```

### Reading Properties

```c
/* Read string */
const char *model;
if (of_property_read_string(np, "model", &model) == 0) {
	pr_info("Model: %s\n", model);
}

/* Read u32 */
u32 clock_freq;
if (of_property_read_u32(np, "clock-frequency", &clock_freq) == 0) {
	pr_info("Clock: %u Hz\n", clock_freq);
}

/* Read u32 array */
u32 values[3];
int count = of_property_read_u32_array(np, "interrupts", values, 3);

/* Read u64 */
u64 reg_base;
of_property_read_u64(np, "reg", &reg_base);

/* Check if property exists */
if (of_property_read_bool(np, "dma-coherent")) {
	pr_info("DMA coherent enabled\n");
}
```

### Getting Resources

```c
/* Get memory resource */
struct resource *res;
res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
if (!res)
	return -ENODEV;

void __iomem *base = devm_ioremap_resource(dev, res);
if (IS_ERR(base))
	return PTR_ERR(base);

/* Get IRQ */
int irq = platform_get_irq(pdev, 0);
if (irq < 0)
	return irq;

/* Get register address/size directly */
u64 addr, size;
of_property_read_u64_index(np, "reg", 0, &addr);
of_property_read_u64_index(np, "reg", 1, &size);
```

### Parsing Phandles

```c
/* Get referenced node */
struct device_node *clk_np;
clk_np = of_parse_phandle(np, "clocks", 0);
if (!clk_np) {
	dev_err(dev, "No clock specified\n");
	return -EINVAL;
}

/* Get clock */
struct clk *clk = of_clk_get(np, 0);
if (IS_ERR(clk))
	return PTR_ERR(clk);

/* Or by name */
clk = of_clk_get_by_name(np, "uart");
```

### GPIO Handling

```c
#include <linux/of_gpio.h>

/* Get GPIO */
int reset_gpio = of_get_named_gpio(np, "reset-gpios", 0);
if (!gpio_is_valid(reset_gpio))
	return -EINVAL;

/* Request and configure */
devm_gpio_request_one(dev, reset_gpio, GPIOF_OUT_INIT_LOW, "reset");

/* Using GPIO descriptor API (preferred) */
#include <linux/gpio/consumer.h>

struct gpio_desc *reset_gpiod;
reset_gpiod = devm_gpiod_get(dev, "reset", GPIOD_OUT_LOW);
if (IS_ERR(reset_gpiod))
	return PTR_ERR(reset_gpiod);

gpiod_set_value(reset_gpiod, 1);
```

### Iterating Child Nodes

```c
struct device_node *child;

for_each_child_of_node(np, child) {
	const char *name;
	u32 reg;

	of_property_read_string(child, "label", &name);
	of_property_read_u32(child, "reg", &reg);

	pr_info("Child: %s at 0x%x\n", name, reg);
}
```

### Complete Driver Example

```c
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/of_device.h>
#include <linux/clk.h>
#include <linux/gpio/consumer.h>

struct my_device {
	void __iomem *base;
	struct clk *clk;
	int irq;
	struct gpio_desc *reset_gpio;
	u32 clock_freq;
};

static int my_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct device_node *np = dev->of_node;
	struct my_device *priv;
	struct resource *res;
	int ret;

	priv = devm_kzalloc(dev, sizeof(*priv), GFP_KERNEL);
	if (!priv)
		return -ENOMEM;

	/* Get memory resource */
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	priv->base = devm_ioremap_resource(dev, res);
	if (IS_ERR(priv->base))
		return PTR_ERR(priv->base);

	/* Get IRQ */
	priv->irq = platform_get_irq(pdev, 0);
	if (priv->irq < 0)
		return priv->irq;

	/* Get clock */
	priv->clk = devm_clk_get(dev, "uart");
	if (IS_ERR(priv->clk)) {
		dev_err(dev, "Failed to get clock\n");
		return PTR_ERR(priv->clk);
	}

	/* Get GPIO */
	priv->reset_gpio = devm_gpiod_get_optional(dev, "reset", GPIOD_OUT_LOW);
	if (IS_ERR(priv->reset_gpio))
		return PTR_ERR(priv->reset_gpio);

	/* Read clock frequency */
	ret = of_property_read_u32(np, "clock-frequency", &priv->clock_freq);
	if (ret) {
		/* Use default if not specified */
		priv->clock_freq = 48000000;
	}

	/* Enable clock */
	ret = clk_prepare_enable(priv->clk);
	if (ret)
		return ret;

	/* Reset device */
	if (priv->reset_gpio) {
		gpiod_set_value(priv->reset_gpio, 1);
		msleep(10);
		gpiod_set_value(priv->reset_gpio, 0);
	}

	platform_set_drvdata(pdev, priv);

	dev_info(dev, "Device initialized (clock=%u Hz)\n", priv->clock_freq);
	return 0;
}

static int my_remove(struct platform_device *pdev)
{
	struct my_device *priv = platform_get_drvdata(pdev);

	clk_disable_unprepare(priv->clk);
	return 0;
}

static const struct of_device_id my_of_match[] = {
	{ .compatible = "vendor,my-device" },
	{ }
};
MODULE_DEVICE_TABLE(of, my_of_match);

static struct platform_driver my_driver = {
	.probe = my_probe,
	.remove = my_remove,
	.driver = {
		.name = "my-device",
		.of_match_table = my_of_match,
	},
};
module_platform_driver(my_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Device Tree Example Driver");
```

**Device Tree:**
```dts
my_device: device@44e09000 {
	compatible = "vendor,my-device";
	reg = <0x44e09000 0x1000>;
	interrupts = <0 72 4>;
	clocks = <&sysclk>;
	clock-names = "uart";
	reset-gpios = <&gpio0 15 GPIO_ACTIVE_LOW>;
	clock-frequency = <48000000>;
};
```

---

## Common Bindings

### I2C Devices

```dts
&i2c0 {
	#address-cells = <1>;
	#size-cells = <0>;

	eeprom@50 {
		compatible = "atmel,24c256";
		reg = <0x50>;
		pagesize = <64>;
	};

	rtc@68 {
		compatible = "dallas,ds1307";
		reg = <0x68>;
		interrupts = <0 75 IRQ_TYPE_EDGE_FALLING>;
	};
};
```

### SPI Devices

```dts
&spi0 {
	#address-cells = <1>;
	#size-cells = <0>;

	flash@0 {
		compatible = "jedec,spi-nor";
		reg = <0>;  /* Chip select 0 */
		spi-max-frequency = <20000000>;

		partitions {
			compatible = "fixed-partitions";
			#address-cells = <1>;
			#size-cells = <1>;

			partition@0 {
				label = "bootloader";
				reg = <0x000000 0x100000>;
				read-only;
			};

			partition@100000 {
				label = "kernel";
				reg = <0x100000 0x400000>;
			};

			partition@500000 {
				label = "rootfs";
				reg = <0x500000 0xb00000>;
			};
		};
	};
};
```

### Regulators

```dts
regulators {
	compatible = "simple-bus";
	#address-cells = <1>;
	#size-cells = <0>;

	vdd_core: regulator@0 {
		compatible = "regulator-fixed";
		reg = <0>;
		regulator-name = "vdd_core";
		regulator-min-microvolt = <1200000>;
		regulator-max-microvolt = <1200000>;
		regulator-always-on;
		regulator-boot-on;
	};

	vdd_3v3: regulator@1 {
		compatible = "regulator-gpio";
		reg = <1>;
		regulator-name = "vdd_3v3";
		regulator-min-microvolt = <3300000>;
		regulator-max-microvolt = <3300000>;
		enable-gpio = <&gpio0 20 GPIO_ACTIVE_HIGH>;
		enable-active-high;
	};
};

/* Usage */
&uart0 {
	vdd-supply = <&vdd_3v3>;
};
```

### Pinctrl (Pin Multiplexing)

```dts
pinctrl: pinctrl@44e10800 {
	compatible = "vendor,pinctrl";
	reg = <0x44e10800 0x1000>;

	uart0_pins: uart0_pins {
		pinctrl-single,pins = <
			0x170 (PIN_INPUT_PULLUP | MUX_MODE0)  /* uart0_rxd */
			0x174 (PIN_OUTPUT_PULLDOWN | MUX_MODE0) /* uart0_txd */
		>;
	};

	i2c0_pins: i2c0_pins {
		pinctrl-single,pins = <
			0x188 (PIN_INPUT_PULLUP | MUX_MODE0)   /* i2c0_sda */
			0x18c (PIN_INPUT_PULLUP | MUX_MODE0)   /* i2c0_scl */
		>;
	};
};

&uart0 {
	pinctrl-names = "default";
	pinctrl-0 = <&uart0_pins>;
};

&i2c0 {
	pinctrl-names = "default";
	pinctrl-0 = <&i2c0_pins>;
};
```

---

## Platform-Specific Details

### ARM Device Tree

```dts
/dts-v1/;

/ {
	model = "ARM Versatile Express";
	compatible = "arm,vexpress";

	#address-cells = <1>;
	#size-cells = <1>;

	cpus {
		#address-cells = <1>;
		#size-cells = <0>;

		cpu@0 {
			device_type = "cpu";
			compatible = "arm,cortex-a9";
			reg = <0>;
		};

		cpu@1 {
			device_type = "cpu";
			compatible = "arm,cortex-a9";
			reg = <1>;
		};
	};
};
```

### ARM64 Device Tree

```dts
/dts-v1/;

/ {
	#address-cells = <2>;  /* 64-bit addressing */
	#size-cells = <2>;

	cpus {
		#address-cells = <1>;
		#size-cells = <0>;

		cpu@0 {
			device_type = "cpu";
			compatible = "arm,cortex-a57";
			reg = <0x0>;
			enable-method = "psci";
		};
	};

	memory@80000000 {
		device_type = "memory";
		reg = <0x0 0x80000000 0x0 0x80000000>; /* 2GB */
	};
};
```

### Raspberry Pi Example

```dts
/dts-v1/;

#include "bcm2835.dtsi"

/ {
	compatible = "raspberrypi,model-b", "brcm,bcm2835";
	model = "Raspberry Pi Model B";

	memory@0 {
		device_type = "memory";
		reg = <0 0x20000000>; /* 512 MB */
	};
};

&uart0 {
	status = "okay";
};

&i2c1 {
	status = "okay";
	clock-frequency = <100000>;
};

&sdhci {
	status = "okay";
	bus-width = <4>;
};
```

---

## Debugging Device Tree

### Viewing Loaded Device Tree

```bash
# View device tree in /proc
cat /proc/device-tree/model

# Or using dtc
dtc -I fs -O dts /proc/device-tree

# Better formatting
dtc -I fs -O dts -o /tmp/current.dts /proc/device-tree
```

### Sysfs Device Tree

```bash
# Navigate device tree in sysfs
ls /sys/firmware/devicetree/base/

# View property
cat /sys/firmware/devicetree/base/model

# View all properties of a node
ls -la /sys/firmware/devicetree/base/soc/serial@44e09000/
```

### Kernel Boot Messages

```bash
# Check device tree loading
dmesg | grep -i "device tree"
dmesg | grep -i "dtb"

# Check OF (Open Firmware) messages
dmesg | grep -i "of:"
```

### Driver Matching Debug

```c
/* In driver code */
static int my_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct device_node *np = dev->of_node;

	dev_info(dev, "Device tree node: %pOF\n", np);
	dev_info(dev, "Compatible: %s\n",
		 of_get_property(np, "compatible", NULL));

	/* Print all properties */
	struct property *prop;
	for_each_property_of_node(np, prop) {
		dev_info(dev, "Property: %s\n", prop->name);
	}

	return 0;
}
```

### Common Issues

**Device not probing:**
```bash
# Check if device is in device tree
ls /sys/firmware/devicetree/base/soc/

# Check driver registration
ls /sys/bus/platform/drivers/

# Check devices without driver
cat /sys/kernel/debug/devices_deferred
```

**Compatible string mismatch:**
```c
/* Check driver's compatible strings */
static const struct of_device_id my_of_match[] = {
	{ .compatible = "vendor,device-v2" },  /* Try this first */
	{ .compatible = "vendor,device" },      /* Then this */
	{ }
};
```

---

## Best Practices

### DO's

1. **Use specific compatible strings first:**
```dts
compatible = "vendor,soc-uart-v2", "vendor,uart", "ns16550a";
```

2. **Disable devices by default in SoC .dtsi:**
```dts
/* In SoC .dtsi */
uart0: serial@44e09000 {
	status = "disabled";
};

/* In board .dts */
&uart0 {
	status = "okay";
};
```

3. **Use labels for references:**
```dts
uart0: serial@44e09000 { ... };

&uart0 {
	/* Override properties */
};
```

4. **Document bindings:**
```yaml
# Documentation/devicetree/bindings/serial/vendor-uart.yaml
title: Vendor UART Controller

properties:
  compatible:
    const: vendor,uart

  reg:
    maxItems: 1

  interrupts:
    maxItems: 1
```

5. **Use standard property names:**
- `clock-frequency` not `clock-freq`
- `reset-gpios` not `reset-gpio`
- Follow bindings in `Documentation/devicetree/bindings/`

### DON'Ts

1. **Don't duplicate information:**
```dts
/* Bad - IRQ already specified in interrupts */
uart0 {
	interrupts = <72>;
	irq-number = <72>;  /* Redundant */
};

/* Good */
uart0 {
	interrupts = <72>;
};
```

2. **Don't use Linux-specific information:**
```dts
/* Bad - driver name is Linux-specific */
uart0 {
	linux,driver-name = "vendor-uart";
};

/* Good - use compatible */
uart0 {
	compatible = "vendor,uart";
};
```

3. **Don't hardcode board-specific data in drivers:**
```c
/* Bad - hardcoded in driver */
#define UART_BASE 0x44e09000

/* Good - read from device tree */
res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
```

---

## Real-World Examples

### BeagleBone Black

```dts
/dts-v1/;

#include "am33xx.dtsi"

/ {
	model = "TI AM335x BeagleBone Black";
	compatible = "ti,am335x-bone-black", "ti,am335x-bone", "ti,am33xx";

	memory@80000000 {
		device_type = "memory";
		reg = <0x80000000 0x20000000>; /* 512 MB */
	};

	leds {
		compatible = "gpio-leds";
		pinctrl-names = "default";
		pinctrl-0 = <&user_leds_s0>;

		led0 {
			label = "beaglebone:green:usr0";
			gpios = <&gpio1 21 GPIO_ACTIVE_HIGH>;
			linux,default-trigger = "heartbeat";
			default-state = "off";
		};
	};
};

&uart0 {
	pinctrl-names = "default";
	pinctrl-0 = <&uart0_pins>;
	status = "okay";
};

&mmc1 {
	vmmc-supply = <&vmmcsd_fixed>;
	bus-width = <4>;
	status = "okay";
};
```

### Raspberry Pi 4

```dts
/dts-v1/;

#include "bcm2711.dtsi"

/ {
	compatible = "raspberrypi,4-model-b", "brcm,bcm2711";
	model = "Raspberry Pi 4 Model B";

	memory@0 {
		device_type = "memory";
		reg = <0x0 0x0 0x0 0x80000000>; /* 2GB */
	};

	aliases {
		serial0 = &uart0;
		serial1 = &uart1;
	};
};

&uart0 {
	pinctrl-names = "default";
	pinctrl-0 = <&uart0_gpio14>;
	status = "okay";
};

&i2c1 {
	pinctrl-names = "default";
	pinctrl-0 = <&i2c1_gpio2>;
	clock-frequency = <100000>;
	status = "okay";
};
```

---

## Summary

Device Tree provides:
- Hardware description separated from kernel code
- Single kernel for multiple boards
- Runtime configuration
- Standardized hardware description

**Key points:**
1. Use `.dts` for board-specific, `.dtsi` for SoC common definitions
2. `compatible` property binds nodes to drivers
3. Use standard properties and follow bindings documentation
4. Parse device tree in drivers using OF APIs
5. Debug using `/proc/device-tree` and `/sys/firmware/devicetree`

**Resources:**
- [Device Tree Specification](https://www.devicetree.org/)
- [Linux Device Tree Documentation](https://www.kernel.org/doc/Documentation/devicetree/)
- [Device Tree Bindings](https://www.kernel.org/doc/Documentation/devicetree/bindings/)
