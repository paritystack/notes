# Linux Driver Development

Comprehensive guide to developing device drivers for the Linux kernel, covering the driver model, device types, and best practices.

## Table of Contents
- [Introduction](#introduction)
- [Linux Driver Model](#linux-driver-model)
- [Device Types](#device-types)
- [Character Device Drivers](#character-device-drivers)
- [Platform Drivers](#platform-drivers)
- [Bus Drivers](#bus-drivers)
- [Block Device Drivers](#block-device-drivers)
- [Network Device Drivers](#network-device-drivers)
- [Device Tree](#device-tree)
- [Power Management](#power-management)
- [DMA](#dma)
- [Interrupts](#interrupts)
- [sysfs and Device Model](#sysfs-and-device-model)
- [Debugging](#debugging)
- [Best Practices](#best-practices)

---

## Introduction

Linux device drivers are kernel modules that provide an interface between hardware devices and the kernel. They abstract hardware complexity and provide a uniform API for user space.

### Driver Architecture

```
┌─────────────────────────────────────┐
│        User Space                    │
│    (Applications, Libraries)         │
└─────────────────────────────────────┘
              │ System Calls
┌─────────────────────────────────────┐
│        Kernel Space                  │
│  ┌───────────────────────────────┐  │
│  │  Virtual File System (VFS)    │  │
│  └───────────────────────────────┘  │
│              │                       │
│  ┌───────────────────────────────┐  │
│  │    Device Drivers              │  │
│  │  - Character Drivers           │  │
│  │  - Block Drivers               │  │
│  │  - Network Drivers             │  │
│  └───────────────────────────────┘  │
│              │                       │
│  ┌───────────────────────────────┐  │
│  │    Bus Subsystems              │  │
│  │  - PCI, USB, I2C, SPI, etc.   │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
              │
┌─────────────────────────────────────┐
│        Hardware                      │
└─────────────────────────────────────┘
```

### Driver Types

1. **Character Drivers**: Sequential access (serial ports, keyboards)
2. **Block Drivers**: Random access (hard drives, SSDs)
3. **Network Drivers**: Network interfaces (Ethernet, WiFi)

---

## Linux Driver Model

The Linux driver model provides a unified framework for device management.

### Core Components

```
Device ←→ Driver ←→ Bus
   ↓         ↓       ↓
 struct   struct   struct
 device   driver    bus
```

### Key Structures

```c
#include <linux/device.h>

/* Device structure */
struct device {
	struct device		*parent;
	struct device_private	*p;

	struct kobject kobj;
	const char		*init_name;
	const struct device_type *type;

	struct bus_type	*bus;
	struct device_driver *driver;

	void		*platform_data;
	void		*driver_data;

	struct dev_pm_info	power;
	struct dev_pm_domain	*pm_domain;

	int		numa_node;
	u64		*dma_mask;
	u64		coherent_dma_mask;

	struct device_dma_parameters *dma_parms;

	struct list_head	dma_pools;
	struct dma_coherent_mem	*dma_mem;

	struct dev_archdata	archdata;
	struct device_node	*of_node;
	struct fwnode_handle	*fwnode;

	dev_t			devt;
	u32			id;

	spinlock_t		devres_lock;
	struct list_head	devres_head;
};

/* Driver structure */
struct device_driver {
	const char		*name;
	struct bus_type		*bus;

	struct module		*owner;
	const char		*mod_name;

	bool suppress_bind_attrs;

	const struct of_device_id	*of_match_table;
	const struct acpi_device_id	*acpi_match_table;

	int (*probe) (struct device *dev);
	int (*remove) (struct device *dev);
	void (*shutdown) (struct device *dev);
	int (*suspend) (struct device *dev, pm_message_t state);
	int (*resume) (struct device *dev);

	const struct attribute_group **groups;

	const struct dev_pm_ops *pm;

	struct driver_private *p;
};

/* Bus type structure */
struct bus_type {
	const char		*name;
	const char		*dev_name;
	struct device		*dev_root;

	const struct attribute_group **bus_groups;
	const struct attribute_group **dev_groups;
	const struct attribute_group **drv_groups;

	int (*match)(struct device *dev, struct device_driver *drv);
	int (*uevent)(struct device *dev, struct kobj_uevent_env *env);
	int (*probe)(struct device *dev);
	int (*remove)(struct device *dev);
	void (*shutdown)(struct device *dev);

	int (*suspend)(struct device *dev, pm_message_t state);
	int (*resume)(struct device *dev);

	const struct dev_pm_ops *pm;

	struct subsys_private *p;
};
```

### Device Registration

```c
/* Register a device */
int device_register(struct device *dev)
{
	device_initialize(dev);
	return device_add(dev);
}

/* Example: Create and register a device */
static int create_my_device(struct device *parent)
{
	struct device *dev;
	int ret;

	dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	if (!dev)
		return -ENOMEM;

	dev->parent = parent;
	dev->bus = &my_bus_type;
	dev_set_name(dev, "mydevice%d", id);

	ret = device_register(dev);
	if (ret) {
		put_device(dev);
		return ret;
	}

	return 0;
}

/* Unregister device */
void device_unregister(struct device *dev)
{
	device_del(dev);
	put_device(dev);
}
```

### Driver Registration

```c
/* Register a driver */
int driver_register(struct device_driver *drv)
{
	int ret;

	ret = bus_add_driver(drv);
	if (ret)
		return ret;

	ret = driver_add_groups(drv, drv->groups);
	if (ret) {
		bus_remove_driver(drv);
		return ret;
	}

	return 0;
}

/* Example: Register a driver */
static struct device_driver my_driver = {
	.name = "my_driver",
	.bus = &my_bus_type,
	.probe = my_probe,
	.remove = my_remove,
	.pm = &my_pm_ops,
};

static int __init my_driver_init(void)
{
	return driver_register(&my_driver);
}

static void __exit my_driver_exit(void)
{
	driver_unregister(&my_driver);
}

module_init(my_driver_init);
module_exit(my_driver_exit);
```

### Matching Devices and Drivers

```c
/* Bus match function */
static int my_bus_match(struct device *dev, struct device_driver *drv)
{
	struct my_device *my_dev = to_my_device(dev);
	struct my_driver *my_drv = to_my_driver(drv);

	/* Match by name */
	if (strcmp(dev_name(dev), drv->name) == 0)
		return 1;

	/* Match by compatible string (device tree) */
	if (of_driver_match_device(dev, drv))
		return 1;

	return 0;
}
```

---

## Device Types

### Character Devices

Sequential access devices. Most common type.

```c
#include <linux/cdev.h>
#include <linux/fs.h>

struct my_char_dev {
	struct cdev cdev;
	dev_t devt;
	struct class *class;
	struct device *device;
	/* Device-specific data */
	void __iomem *base;
	struct mutex lock;
};

static int my_open(struct inode *inode, struct file *filp)
{
	struct my_char_dev *dev;

	dev = container_of(inode->i_cdev, struct my_char_dev, cdev);
	filp->private_data = dev;

	pr_info("Device opened\n");
	return 0;
}

static int my_release(struct inode *inode, struct file *filp)
{
	pr_info("Device closed\n");
	return 0;
}

static ssize_t my_read(struct file *filp, char __user *buf,
		       size_t count, loff_t *f_pos)
{
	struct my_char_dev *dev = filp->private_data;
	char kbuf[256];
	size_t len;

	/* Read from hardware */
	len = snprintf(kbuf, sizeof(kbuf), "Hello from device\n");

	if (count < len)
		len = count;

	if (copy_to_user(buf, kbuf, len))
		return -EFAULT;

	*f_pos += len;
	return len;
}

static ssize_t my_write(struct file *filp, const char __user *buf,
			size_t count, loff_t *f_pos)
{
	struct my_char_dev *dev = filp->private_data;
	char kbuf[256];

	if (count > sizeof(kbuf) - 1)
		count = sizeof(kbuf) - 1;

	if (copy_from_user(kbuf, buf, count))
		return -EFAULT;

	kbuf[count] = '\0';

	pr_info("Received: %s\n", kbuf);

	/* Write to hardware */

	return count;
}

static long my_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct my_char_dev *dev = filp->private_data;

	switch (cmd) {
	case MY_IOCTL_RESET:
		/* Reset device */
		pr_info("Reset device\n");
		break;

	case MY_IOCTL_GET_STATUS:
		/* Get device status */
		if (copy_to_user((void __user *)arg, &dev->status,
				 sizeof(dev->status)))
			return -EFAULT;
		break;

	default:
		return -ENOTTY;
	}

	return 0;
}

static const struct file_operations my_fops = {
	.owner = THIS_MODULE,
	.open = my_open,
	.release = my_release,
	.read = my_read,
	.write = my_write,
	.unlocked_ioctl = my_ioctl,
};
```

### Block Devices

Random access storage devices.

```c
#include <linux/blkdev.h>
#include <linux/genhd.h>

struct my_block_dev {
	spinlock_t lock;
	struct request_queue *queue;
	struct gendisk *gd;

	u8 *data;		/* Virtual disk storage */
	size_t size;		/* Size in bytes */
};

static void my_request(struct request_queue *q)
{
	struct request *req;
	struct my_block_dev *dev = q->queuedata;

	while ((req = blk_fetch_request(q)) != NULL) {
		sector_t sector = blk_rq_pos(req);
		unsigned long offset = sector * KERNEL_SECTOR_SIZE;
		size_t len = blk_rq_bytes(req);

		if (offset + len > dev->size) {
			pr_err("Beyond device size\n");
			__blk_end_request_all(req, -EIO);
			continue;
		}

		if (rq_data_dir(req) == WRITE) {
			/* Write to virtual disk */
			memcpy(dev->data + offset, bio_data(req->bio), len);
		} else {
			/* Read from virtual disk */
			memcpy(bio_data(req->bio), dev->data + offset, len);
		}

		__blk_end_request_all(req, 0);
	}
}

static int my_block_open(struct block_device *bdev, fmode_t mode)
{
	pr_info("Block device opened\n");
	return 0;
}

static void my_block_release(struct gendisk *gd, fmode_t mode)
{
	pr_info("Block device released\n");
}

static const struct block_device_operations my_bdev_ops = {
	.owner = THIS_MODULE,
	.open = my_block_open,
	.release = my_block_release,
};

static int create_block_device(struct my_block_dev *dev)
{
	int ret;

	/* Allocate request queue */
	spin_lock_init(&dev->lock);
	dev->queue = blk_init_queue(my_request, &dev->lock);
	if (!dev->queue)
		return -ENOMEM;

	dev->queue->queuedata = dev;

	/* Allocate gendisk */
	dev->gd = alloc_disk(1);
	if (!dev->gd) {
		blk_cleanup_queue(dev->queue);
		return -ENOMEM;
	}

	dev->gd->major = MY_MAJOR;
	dev->gd->first_minor = 0;
	dev->gd->fops = &my_bdev_ops;
	dev->gd->queue = dev->queue;
	dev->gd->private_data = dev;
	snprintf(dev->gd->disk_name, 32, "myblock");
	set_capacity(dev->gd, dev->size / KERNEL_SECTOR_SIZE);

	add_disk(dev->gd);

	return 0;
}
```

### Network Devices

```c
#include <linux/netdevice.h>
#include <linux/etherdevice.h>

struct my_net_priv {
	struct net_device *dev;
	struct napi_struct napi;

	void __iomem *base;
	spinlock_t lock;
};

static int my_net_open(struct net_device *dev)
{
	struct my_net_priv *priv = netdev_priv(dev);

	/* Enable hardware */
	/* Request IRQ */
	/* Enable NAPI */
	napi_enable(&priv->napi);

	netif_start_queue(dev);

	pr_info("Network device opened\n");
	return 0;
}

static int my_net_stop(struct net_device *dev)
{
	struct my_net_priv *priv = netdev_priv(dev);

	netif_stop_queue(dev);

	napi_disable(&priv->napi);
	/* Free IRQ */
	/* Disable hardware */

	pr_info("Network device closed\n");
	return 0;
}

static netdev_tx_t my_net_start_xmit(struct sk_buff *skb,
				     struct net_device *dev)
{
	struct my_net_priv *priv = netdev_priv(dev);

	/* Transmit packet */
	/* Write to hardware TX ring */

	dev->stats.tx_packets++;
	dev->stats.tx_bytes += skb->len;

	dev_kfree_skb(skb);

	return NETDEV_TX_OK;
}

static int my_net_poll(struct napi_struct *napi, int budget)
{
	struct my_net_priv *priv = container_of(napi, struct my_net_priv, napi);
	struct net_device *dev = priv->dev;
	int work_done = 0;
	struct sk_buff *skb;

	/* Process RX packets */
	while (work_done < budget) {
		/* Get packet from hardware */
		skb = my_get_rx_packet(priv);
		if (!skb)
			break;

		skb->dev = dev;
		skb->protocol = eth_type_trans(skb, dev);

		netif_receive_skb(skb);

		dev->stats.rx_packets++;
		dev->stats.rx_bytes += skb->len;

		work_done++;
	}

	if (work_done < budget) {
		napi_complete(napi);
		/* Re-enable interrupts */
	}

	return work_done;
}

static const struct net_device_ops my_netdev_ops = {
	.ndo_open = my_net_open,
	.ndo_stop = my_net_stop,
	.ndo_start_xmit = my_net_start_xmit,
};

static int create_net_device(struct device *parent)
{
	struct net_device *dev;
	struct my_net_priv *priv;
	int ret;

	dev = alloc_etherdev(sizeof(*priv));
	if (!dev)
		return -ENOMEM;

	priv = netdev_priv(dev);
	priv->dev = dev;

	dev->netdev_ops = &my_netdev_ops;
	dev->watchdog_timeo = 5 * HZ;

	/* Set MAC address */
	eth_hw_addr_random(dev);

	/* Setup NAPI */
	netif_napi_add(dev, &priv->napi, my_net_poll, 64);

	SET_NETDEV_DEV(dev, parent);

	ret = register_netdev(dev);
	if (ret) {
		free_netdev(dev);
		return ret;
	}

	return 0;
}
```

---

## Character Device Drivers

Complete example with multiple features.

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "mychardev"
#define CLASS_NAME "myclass"

static int major_number;
static struct class *my_class;
static struct device *my_device;
static struct cdev my_cdev;

static char message[256] = "Hello from driver";
static short message_len;
static int times_opened = 0;

static int dev_open(struct inode *inode, struct file *file)
{
	times_opened++;
	pr_info("Device opened %d times\n", times_opened);
	return 0;
}

static int dev_release(struct inode *inode, struct file *file)
{
	pr_info("Device closed\n");
	return 0;
}

static ssize_t dev_read(struct file *file, char __user *buffer,
			size_t len, loff_t *offset)
{
	int bytes_to_read;

	if (*offset >= message_len)
		return 0;

	bytes_to_read = min(len, (size_t)(message_len - *offset));

	if (copy_to_user(buffer, message + *offset, bytes_to_read))
		return -EFAULT;

	*offset += bytes_to_read;

	pr_info("Sent %d characters to user\n", bytes_to_read);

	return bytes_to_read;
}

static ssize_t dev_write(struct file *file, const char __user *buffer,
			 size_t len, loff_t *offset)
{
	size_t bytes_to_write = min(len, sizeof(message) - 1);

	if (copy_from_user(message, buffer, bytes_to_write))
		return -EFAULT;

	message[bytes_to_write] = '\0';
	message_len = bytes_to_write;

	pr_info("Received %zu characters from user\n", bytes_to_write);

	return bytes_to_write;
}

static struct file_operations fops = {
	.owner = THIS_MODULE,
	.open = dev_open,
	.release = dev_release,
	.read = dev_read,
	.write = dev_write,
};

static int __init chardev_init(void)
{
	int ret;
	dev_t dev;

	/* Allocate major number */
	ret = alloc_chrdev_region(&dev, 0, 1, DEVICE_NAME);
	if (ret < 0) {
		pr_err("Failed to allocate major number\n");
		return ret;
	}

	major_number = MAJOR(dev);
	pr_info("Registered with major number %d\n", major_number);

	/* Initialize cdev */
	cdev_init(&my_cdev, &fops);
	my_cdev.owner = THIS_MODULE;

	/* Add cdev */
	ret = cdev_add(&my_cdev, dev, 1);
	if (ret < 0) {
		unregister_chrdev_region(dev, 1);
		pr_err("Failed to add cdev\n");
		return ret;
	}

	/* Create class */
	my_class = class_create(THIS_MODULE, CLASS_NAME);
	if (IS_ERR(my_class)) {
		cdev_del(&my_cdev);
		unregister_chrdev_region(dev, 1);
		pr_err("Failed to create class\n");
		return PTR_ERR(my_class);
	}

	/* Create device */
	my_device = device_create(my_class, NULL, dev, NULL, DEVICE_NAME);
	if (IS_ERR(my_device)) {
		class_destroy(my_class);
		cdev_del(&my_cdev);
		unregister_chrdev_region(dev, 1);
		pr_err("Failed to create device\n");
		return PTR_ERR(my_device);
	}

	message_len = strlen(message);

	pr_info("Character device driver loaded\n");
	return 0;
}

static void __exit chardev_exit(void)
{
	dev_t dev = MKDEV(major_number, 0);

	device_destroy(my_class, dev);
	class_destroy(my_class);
	cdev_del(&my_cdev);
	unregister_chrdev_region(dev, 1);

	pr_info("Character device driver unloaded\n");
}

module_init(chardev_init);
module_exit(chardev_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Driver Developer");
MODULE_DESCRIPTION("Simple character device driver");
```

---

## Platform Drivers

Platform drivers are for devices that are not discoverable (embedded SoCs).

```c
#include <linux/platform_device.h>
#include <linux/mod_devicetable.h>
#include <linux/io.h>
#include <linux/of.h>

struct my_platform_dev {
	struct device *dev;
	void __iomem *base;
	struct resource *res;
	int irq;
};

static int my_platform_probe(struct platform_device *pdev)
{
	struct my_platform_dev *priv;
	struct resource *res;
	int ret;

	pr_info("Platform driver probe\n");

	priv = devm_kzalloc(&pdev->dev, sizeof(*priv), GFP_KERNEL);
	if (!priv)
		return -ENOMEM;

	priv->dev = &pdev->dev;

	/* Get memory resource */
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (!res) {
		dev_err(&pdev->dev, "No memory resource\n");
		return -ENODEV;
	}

	/* Map registers */
	priv->base = devm_ioremap_resource(&pdev->dev, res);
	if (IS_ERR(priv->base))
		return PTR_ERR(priv->base);

	/* Get IRQ */
	priv->irq = platform_get_irq(pdev, 0);
	if (priv->irq < 0) {
		dev_err(&pdev->dev, "No IRQ resource\n");
		return priv->irq;
	}

	/* Request IRQ */
	ret = devm_request_irq(&pdev->dev, priv->irq, my_irq_handler,
			       IRQF_SHARED, dev_name(&pdev->dev), priv);
	if (ret) {
		dev_err(&pdev->dev, "Failed to request IRQ\n");
		return ret;
	}

	/* Store private data */
	platform_set_drvdata(pdev, priv);

	/* Initialize hardware */
	writel(0x1, priv->base + CTRL_REG);

	dev_info(&pdev->dev, "Device initialized\n");

	return 0;
}

static int my_platform_remove(struct platform_device *pdev)
{
	struct my_platform_dev *priv = platform_get_drvdata(pdev);

	/* Shutdown hardware */
	writel(0x0, priv->base + CTRL_REG);

	dev_info(&pdev->dev, "Device removed\n");

	return 0;
}

/* Device tree match table */
static const struct of_device_id my_of_match[] = {
	{ .compatible = "vendor,my-device" },
	{ .compatible = "vendor,my-device-v2" },
	{ }
};
MODULE_DEVICE_TABLE(of, my_of_match);

/* Platform device ID table (for non-DT systems) */
static const struct platform_device_id my_platform_ids[] = {
	{ .name = "my-device" },
	{ }
};
MODULE_DEVICE_TABLE(platform, my_platform_ids);

static struct platform_driver my_platform_driver = {
	.probe = my_platform_probe,
	.remove = my_platform_remove,
	.driver = {
		.name = "my-device",
		.of_match_table = my_of_match,
	},
	.id_table = my_platform_ids,
};

module_platform_driver(my_platform_driver);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Platform device driver");
```

---

## Bus Drivers

### I2C Driver

```c
#include <linux/i2c.h>

struct my_i2c_dev {
	struct i2c_client *client;
	struct device *dev;
};

static int my_i2c_probe(struct i2c_client *client,
			const struct i2c_device_id *id)
{
	struct my_i2c_dev *priv;
	u8 buf[2];
	int ret;

	dev_info(&client->dev, "I2C device probed\n");

	priv = devm_kzalloc(&client->dev, sizeof(*priv), GFP_KERNEL);
	if (!priv)
		return -ENOMEM;

	priv->client = client;
	priv->dev = &client->dev;

	i2c_set_clientdata(client, priv);

	/* Read device ID */
	ret = i2c_smbus_read_byte_data(client, REG_ID);
	if (ret < 0) {
		dev_err(&client->dev, "Failed to read device ID\n");
		return ret;
	}

	dev_info(&client->dev, "Device ID: 0x%02x\n", ret);

	/* Write configuration */
	buf[0] = REG_CONFIG;
	buf[1] = 0x80;
	ret = i2c_master_send(client, buf, 2);
	if (ret < 0) {
		dev_err(&client->dev, "Failed to write config\n");
		return ret;
	}

	return 0;
}

static int my_i2c_remove(struct i2c_client *client)
{
	dev_info(&client->dev, "I2C device removed\n");
	return 0;
}

static const struct i2c_device_id my_i2c_ids[] = {
	{ "my-i2c-device", 0 },
	{ }
};
MODULE_DEVICE_TABLE(i2c, my_i2c_ids);

static const struct of_device_id my_i2c_of_match[] = {
	{ .compatible = "vendor,my-i2c-device" },
	{ }
};
MODULE_DEVICE_TABLE(of, my_i2c_of_match);

static struct i2c_driver my_i2c_driver = {
	.driver = {
		.name = "my-i2c-device",
		.of_match_table = my_i2c_of_match,
	},
	.probe = my_i2c_probe,
	.remove = my_i2c_remove,
	.id_table = my_i2c_ids,
};

module_i2c_driver(my_i2c_driver);
```

### SPI Driver

```c
#include <linux/spi/spi.h>

struct my_spi_dev {
	struct spi_device *spi;
	struct device *dev;
};

static int my_spi_probe(struct spi_device *spi)
{
	struct my_spi_dev *priv;
	u8 tx_buf[2], rx_buf[2];
	int ret;

	dev_info(&spi->dev, "SPI device probed\n");

	priv = devm_kzalloc(&spi->dev, sizeof(*priv), GFP_KERNEL);
	if (!priv)
		return -ENOMEM;

	priv->spi = spi;
	priv->dev = &spi->dev;

	spi_set_drvdata(spi, priv);

	/* Configure SPI mode and speed */
	spi->mode = SPI_MODE_0;
	spi->max_speed_hz = 1000000;
	spi->bits_per_word = 8;

	ret = spi_setup(spi);
	if (ret < 0) {
		dev_err(&spi->dev, "Failed to setup SPI\n");
		return ret;
	}

	/* Read register */
	tx_buf[0] = READ_CMD | REG_ID;
	tx_buf[1] = 0x00;

	ret = spi_write_then_read(spi, tx_buf, 1, rx_buf, 1);
	if (ret < 0) {
		dev_err(&spi->dev, "Failed to read register\n");
		return ret;
	}

	dev_info(&spi->dev, "Device ID: 0x%02x\n", rx_buf[0]);

	return 0;
}

static int my_spi_remove(struct spi_device *spi)
{
	dev_info(&spi->dev, "SPI device removed\n");
	return 0;
}

static const struct of_device_id my_spi_of_match[] = {
	{ .compatible = "vendor,my-spi-device" },
	{ }
};
MODULE_DEVICE_TABLE(of, my_spi_of_match);

static const struct spi_device_id my_spi_ids[] = {
	{ "my-spi-device", 0 },
	{ }
};
MODULE_DEVICE_TABLE(spi, my_spi_ids);

static struct spi_driver my_spi_driver = {
	.driver = {
		.name = "my-spi-device",
		.of_match_table = my_spi_of_match,
	},
	.probe = my_spi_probe,
	.remove = my_spi_remove,
	.id_table = my_spi_ids,
};

module_spi_driver(my_spi_driver);
```

### USB Driver

```c
#include <linux/usb.h>

struct my_usb_dev {
	struct usb_device *udev;
	struct usb_interface *interface;
	struct urb *int_in_urb;
	unsigned char *int_in_buffer;
};

static void my_int_callback(struct urb *urb)
{
	struct my_usb_dev *dev = urb->context;
	int status = urb->status;

	switch (status) {
	case 0:
		/* Success */
		dev_info(&dev->interface->dev, "Data: %*ph\n",
			 urb->actual_length, dev->int_in_buffer);
		break;
	case -ECONNRESET:
	case -ENOENT:
	case -ESHUTDOWN:
		/* URB killed */
		return;
	default:
		dev_err(&dev->interface->dev, "URB error: %d\n", status);
		break;
	}

	/* Resubmit URB */
	usb_submit_urb(urb, GFP_ATOMIC);
}

static int my_usb_probe(struct usb_interface *interface,
			const struct usb_device_id *id)
{
	struct my_usb_dev *dev;
	struct usb_host_interface *iface_desc;
	struct usb_endpoint_descriptor *endpoint;
	int ret;

	dev_info(&interface->dev, "USB device probed\n");

	dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	if (!dev)
		return -ENOMEM;

	dev->udev = usb_get_dev(interface_to_usbdev(interface));
	dev->interface = interface;

	/* Get endpoint descriptors */
	iface_desc = interface->cur_altsetting;

	for (int i = 0; i < iface_desc->desc.bNumEndpoints; i++) {
		endpoint = &iface_desc->endpoint[i].desc;

		if (usb_endpoint_is_int_in(endpoint)) {
			/* Found interrupt IN endpoint */
			dev->int_in_buffer = kmalloc(
				le16_to_cpu(endpoint->wMaxPacketSize),
				GFP_KERNEL);
			if (!dev->int_in_buffer) {
				ret = -ENOMEM;
				goto error;
			}

			dev->int_in_urb = usb_alloc_urb(0, GFP_KERNEL);
			if (!dev->int_in_urb) {
				ret = -ENOMEM;
				goto error;
			}

			usb_fill_int_urb(dev->int_in_urb, dev->udev,
				usb_rcvintpipe(dev->udev, endpoint->bEndpointAddress),
				dev->int_in_buffer,
				le16_to_cpu(endpoint->wMaxPacketSize),
				my_int_callback,
				dev,
				endpoint->bInterval);

			/* Submit URB */
			ret = usb_submit_urb(dev->int_in_urb, GFP_KERNEL);
			if (ret) {
				dev_err(&interface->dev, "Failed to submit URB\n");
				goto error;
			}
		}
	}

	usb_set_intfdata(interface, dev);

	return 0;

error:
	if (dev->int_in_urb)
		usb_free_urb(dev->int_in_urb);
	kfree(dev->int_in_buffer);
	usb_put_dev(dev->udev);
	kfree(dev);
	return ret;
}

static void my_usb_disconnect(struct usb_interface *interface)
{
	struct my_usb_dev *dev;

	dev = usb_get_intfdata(interface);

	usb_set_intfdata(interface, NULL);

	if (dev->int_in_urb) {
		usb_kill_urb(dev->int_in_urb);
		usb_free_urb(dev->int_in_urb);
	}

	kfree(dev->int_in_buffer);
	usb_put_dev(dev->udev);
	kfree(dev);

	dev_info(&interface->dev, "USB device disconnected\n");
}

static const struct usb_device_id my_usb_table[] = {
	{ USB_DEVICE(VENDOR_ID, PRODUCT_ID) },
	{ }
};
MODULE_DEVICE_TABLE(usb, my_usb_table);

static struct usb_driver my_usb_driver = {
	.name = "my-usb-device",
	.probe = my_usb_probe,
	.disconnect = my_usb_disconnect,
	.id_table = my_usb_table,
};

module_usb_driver(my_usb_driver);
```

---

## Block Device Drivers

(See earlier section for complete example)

### Modern Block Layer (blk-mq)

```c
#include <linux/blk-mq.h>

struct my_blk_dev {
	struct blk_mq_tag_set tag_set;
	struct request_queue *queue;
	struct gendisk *disk;

	void *data;
	size_t size;
};

static blk_status_t my_queue_rq(struct blk_mq_hw_ctx *hctx,
				const struct blk_mq_queue_data *bd)
{
	struct request *rq = bd->rq;
	struct my_blk_dev *dev = rq->q->queuedata;
	struct bio_vec bvec;
	struct req_iterator iter;
	sector_t pos = blk_rq_pos(rq);
	void *buffer;
	unsigned long offset = pos * SECTOR_SIZE;

	blk_mq_start_request(rq);

	rq_for_each_segment(bvec, rq, iter) {
		buffer = page_address(bvec.bv_page) + bvec.bv_offset;

		if (rq_data_dir(rq) == WRITE)
			memcpy(dev->data + offset, buffer, bvec.bv_len);
		else
			memcpy(buffer, dev->data + offset, bvec.bv_len);

		offset += bvec.bv_len;
	}

	blk_mq_end_request(rq, BLK_STS_OK);

	return BLK_STS_OK;
}

static const struct blk_mq_ops my_mq_ops = {
	.queue_rq = my_queue_rq,
};

static int create_blkmq_device(struct my_blk_dev *dev)
{
	int ret;

	/* Initialize tag set */
	memset(&dev->tag_set, 0, sizeof(dev->tag_set));
	dev->tag_set.ops = &my_mq_ops;
	dev->tag_set.nr_hw_queues = 1;
	dev->tag_set.queue_depth = 128;
	dev->tag_set.numa_node = NUMA_NO_NODE;
	dev->tag_set.cmd_size = 0;
	dev->tag_set.flags = BLK_MQ_F_SHOULD_MERGE;
	dev->tag_set.driver_data = dev;

	ret = blk_mq_alloc_tag_set(&dev->tag_set);
	if (ret)
		return ret;

	/* Allocate queue */
	dev->queue = blk_mq_init_queue(&dev->tag_set);
	if (IS_ERR(dev->queue)) {
		blk_mq_free_tag_set(&dev->tag_set);
		return PTR_ERR(dev->queue);
	}

	dev->queue->queuedata = dev;

	/* Allocate disk */
	dev->disk = alloc_disk(1);
	if (!dev->disk) {
		blk_cleanup_queue(dev->queue);
		blk_mq_free_tag_set(&dev->tag_set);
		return -ENOMEM;
	}

	dev->disk->major = MY_MAJOR;
	dev->disk->first_minor = 0;
	dev->disk->fops = &my_bdev_ops;
	dev->disk->queue = dev->queue;
	dev->disk->private_data = dev;
	snprintf(dev->disk->disk_name, 32, "myblkmq");
	set_capacity(dev->disk, dev->size / SECTOR_SIZE);

	add_disk(dev->disk);

	return 0;
}
```

---

## Network Device Drivers

(See earlier section for complete example)

---

## Device Tree

Device tree describes hardware topology for non-discoverable devices.

### Device Tree Syntax

```dts
/* my-device.dts */
/dts-v1/;

/ {
	compatible = "vendor,my-board";
	#address-cells = <1>;
	#size-cells = <1>;

	my_device: my-device@40000000 {
		compatible = "vendor,my-device";
		reg = <0x40000000 0x1000>;
		interrupts = <0 25 4>;
		clocks = <&clk_peripheral>;
		clock-names = "peripheral";
		status = "okay";

		/* Custom properties */
		vendor,feature-enable;
		vendor,threshold = <100>;
		vendor,string-prop = "value";
	};

	i2c@40005000 {
		compatible = "vendor,i2c";
		reg = <0x40005000 0x1000>;
		#address-cells = <1>;
		#size-cells = <0>;

		sensor@48 {
			compatible = "vendor,temperature-sensor";
			reg = <0x48>;
		};
	};
};
```

### Parsing Device Tree in Driver

```c
#include <linux/of.h>
#include <linux/of_device.h>
#include <linux/of_irq.h>

static int my_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct device_node *np = dev->of_node;
	u32 threshold;
	const char *string_prop;
	int ret;

	/* Check compatible string */
	if (!of_device_is_compatible(np, "vendor,my-device"))
		return -ENODEV;

	/* Read u32 property */
	ret = of_property_read_u32(np, "vendor,threshold", &threshold);
	if (ret) {
		dev_err(dev, "Failed to read threshold\n");
		return ret;
	}

	dev_info(dev, "Threshold: %u\n", threshold);

	/* Read string property */
	ret = of_property_read_string(np, "vendor,string-prop", &string_prop);
	if (ret == 0)
		dev_info(dev, "String property: %s\n", string_prop);

	/* Check boolean property */
	if (of_property_read_bool(np, "vendor,feature-enable"))
		dev_info(dev, "Feature enabled\n");

	/* Get resource from reg property */
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);

	/* Get IRQ */
	irq = irq_of_parse_and_map(np, 0);

	/* Get clock */
	clk = devm_clk_get(dev, "peripheral");
	if (IS_ERR(clk))
		return PTR_ERR(clk);

	/* Get regulator */
	regulator = devm_regulator_get(dev, "vdd");

	return 0;
}
```

---

## Power Management

```c
#include <linux/pm.h>
#include <linux/pm_runtime.h>

/* System suspend/resume */
static int my_suspend(struct device *dev)
{
	struct my_dev *priv = dev_get_drvdata(dev);

	dev_info(dev, "Suspending\n");

	/* Save state */
	priv->saved_state = readl(priv->base + STATE_REG);

	/* Disable device */
	writel(0, priv->base + CTRL_REG);

	/* Gate clock */
	clk_disable_unprepare(priv->clk);

	return 0;
}

static int my_resume(struct device *dev)
{
	struct my_dev *priv = dev_get_drvdata(dev);
	int ret;

	dev_info(dev, "Resuming\n");

	/* Ungate clock */
	ret = clk_prepare_enable(priv->clk);
	if (ret)
		return ret;

	/* Restore state */
	writel(priv->saved_state, priv->base + STATE_REG);

	/* Enable device */
	writel(1, priv->base + CTRL_REG);

	return 0;
}

/* Runtime PM */
static int my_runtime_suspend(struct device *dev)
{
	struct my_dev *priv = dev_get_drvdata(dev);

	dev_dbg(dev, "Runtime suspend\n");

	clk_disable_unprepare(priv->clk);

	return 0;
}

static int my_runtime_resume(struct device *dev)
{
	struct my_dev *priv = dev_get_drvdata(dev);
	int ret;

	dev_dbg(dev, "Runtime resume\n");

	ret = clk_prepare_enable(priv->clk);
	if (ret)
		return ret;

	return 0;
}

static const struct dev_pm_ops my_pm_ops = {
	SET_SYSTEM_SLEEP_PM_OPS(my_suspend, my_resume)
	SET_RUNTIME_PM_OPS(my_runtime_suspend, my_runtime_resume, NULL)
};

/* Using runtime PM */
static int my_do_something(struct my_dev *priv)
{
	int ret;

	/* Get PM reference (resume device if suspended) */
	ret = pm_runtime_get_sync(priv->dev);
	if (ret < 0) {
		pm_runtime_put_noidle(priv->dev);
		return ret;
	}

	/* Do work */
	writel(0x1, priv->base + CMD_REG);

	/* Release PM reference */
	pm_runtime_mark_last_busy(priv->dev);
	pm_runtime_put_autosuspend(priv->dev);

	return 0;
}
```

---

## DMA

```c
#include <linux/dma-mapping.h>

struct my_dma_dev {
	struct device *dev;
	dma_addr_t dma_handle;
	void *cpu_addr;
	size_t size;
};

/* Coherent (consistent) DMA mapping */
static int setup_coherent_dma(struct my_dma_dev *priv)
{
	priv->size = 4096;

	priv->cpu_addr = dma_alloc_coherent(priv->dev, priv->size,
					    &priv->dma_handle, GFP_KERNEL);
	if (!priv->cpu_addr)
		return -ENOMEM;

	pr_info("DMA buffer: cpu=%p dma=%pad\n",
		priv->cpu_addr, &priv->dma_handle);

	/* Write data to DMA buffer */
	memset(priv->cpu_addr, 0xAA, priv->size);

	/* Program hardware with DMA address */
	writel(priv->dma_handle, priv->base + DMA_ADDR_REG);
	writel(priv->size, priv->base + DMA_SIZE_REG);
	writel(DMA_START, priv->base + DMA_CTRL_REG);

	return 0;
}

static void cleanup_coherent_dma(struct my_dma_dev *priv)
{
	if (priv->cpu_addr) {
		dma_free_coherent(priv->dev, priv->size,
				  priv->cpu_addr, priv->dma_handle);
		priv->cpu_addr = NULL;
	}
}

/* Streaming DMA mapping */
static int do_streaming_dma_tx(struct my_dma_dev *priv, void *buffer,
			       size_t len)
{
	dma_addr_t dma_addr;

	/* Map buffer for DMA */
	dma_addr = dma_map_single(priv->dev, buffer, len, DMA_TO_DEVICE);
	if (dma_mapping_error(priv->dev, dma_addr))
		return -ENOMEM;

	/* Program hardware */
	writel(dma_addr, priv->base + DMA_ADDR_REG);
	writel(len, priv->base + DMA_SIZE_REG);
	writel(DMA_START, priv->base + DMA_CTRL_REG);

	/* Wait for DMA completion (in real driver, use interrupt) */

	/* Unmap buffer */
	dma_unmap_single(priv->dev, dma_addr, len, DMA_TO_DEVICE);

	return 0;
}

/* Scatter-gather DMA */
static int do_sg_dma(struct my_dma_dev *priv, struct scatterlist *sgl,
		     int nents)
{
	int mapped_nents;
	struct scatterlist *sg;
	int i;

	/* Map scatter-gather list */
	mapped_nents = dma_map_sg(priv->dev, sgl, nents, DMA_TO_DEVICE);
	if (!mapped_nents)
		return -ENOMEM;

	/* Program hardware with each SG entry */
	for_each_sg(sgl, sg, mapped_nents, i) {
		writel(sg_dma_address(sg),
		       priv->base + DMA_SG_ADDR_REG(i));
		writel(sg_dma_len(sg),
		       priv->base + DMA_SG_LEN_REG(i));
	}

	writel(mapped_nents, priv->base + DMA_SG_COUNT_REG);
	writel(DMA_SG_START, priv->base + DMA_CTRL_REG);

	/* Wait for completion */

	/* Unmap */
	dma_unmap_sg(priv->dev, sgl, nents, DMA_TO_DEVICE);

	return 0;
}

/* Set DMA mask */
static int setup_dma(struct device *dev)
{
	int ret;

	/* Try 64-bit DMA */
	ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(64));
	if (ret) {
		/* Fall back to 32-bit */
		ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(32));
		if (ret) {
			dev_err(dev, "No suitable DMA available\n");
			return ret;
		}
	}

	return 0;
}
```

---

## Interrupts

```c
#include <linux/interrupt.h>

/* Interrupt handler (top half) */
static irqreturn_t my_irq_handler(int irq, void *dev_id)
{
	struct my_dev *priv = dev_id;
	u32 status;

	/* Read interrupt status */
	status = readl(priv->base + INT_STATUS_REG);

	if (!(status & MY_INT_MASK))
		return IRQ_NONE;  /* Not our interrupt */

	/* Clear interrupt */
	writel(status, priv->base + INT_STATUS_REG);

	/* Minimal processing */
	if (status & INT_ERROR)
		priv->errors++;

	/* Schedule bottom half */
	schedule_work(&priv->work);
	/* Or */
	tasklet_schedule(&priv->tasklet);

	return IRQ_HANDLED;
}

/* Bottom half (workqueue) */
static void my_work_func(struct work_struct *work)
{
	struct my_dev *priv = container_of(work, struct my_dev, work);

	/* Heavy processing that can sleep */
	mutex_lock(&priv->lock);
	/* Process data */
	mutex_unlock(&priv->lock);
}

/* Bottom half (tasklet) */
static void my_tasklet_func(unsigned long data)
{
	struct my_dev *priv = (struct my_dev *)data;

	/* Processing that cannot sleep */
	spin_lock(&priv->lock);
	/* Process data */
	spin_unlock(&priv->lock);
}

/* Threaded IRQ handler */
static irqreturn_t my_threaded_irq(int irq, void *dev_id)
{
	struct my_dev *priv = dev_id;

	/* This runs in a kernel thread, can sleep */
	mutex_lock(&priv->lock);

	/* Heavy processing */

	mutex_unlock(&priv->lock);

	return IRQ_HANDLED;
}

/* Setup interrupts */
static int setup_interrupts(struct my_dev *priv)
{
	int ret;

	/* Regular IRQ */
	ret = devm_request_irq(priv->dev, priv->irq, my_irq_handler,
			       IRQF_SHARED, "my-device", priv);
	if (ret) {
		dev_err(priv->dev, "Failed to request IRQ\n");
		return ret;
	}

	/* Threaded IRQ */
	ret = devm_request_threaded_irq(priv->dev, priv->irq,
					NULL, my_threaded_irq,
					IRQF_ONESHOT, "my-device", priv);
	if (ret) {
		dev_err(priv->dev, "Failed to request threaded IRQ\n");
		return ret;
	}

	/* Initialize work */
	INIT_WORK(&priv->work, my_work_func);

	/* Initialize tasklet */
	tasklet_init(&priv->tasklet, my_tasklet_func, (unsigned long)priv);

	return 0;
}
```

---

## sysfs and Device Model

```c
#include <linux/sysfs.h>

/* sysfs attribute */
static ssize_t threshold_show(struct device *dev,
			      struct device_attribute *attr,
			      char *buf)
{
	struct my_dev *priv = dev_get_drvdata(dev);

	return sprintf(buf, "%u\n", priv->threshold);
}

static ssize_t threshold_store(struct device *dev,
			       struct device_attribute *attr,
			       const char *buf, size_t count)
{
	struct my_dev *priv = dev_get_drvdata(dev);
	unsigned int val;
	int ret;

	ret = kstrtouint(buf, 0, &val);
	if (ret)
		return ret;

	if (val > MAX_THRESHOLD)
		return -EINVAL;

	priv->threshold = val;

	/* Update hardware */
	writel(val, priv->base + THRESHOLD_REG);

	return count;
}

static DEVICE_ATTR_RW(threshold);

/* Binary attribute (for large data) */
static ssize_t firmware_read(struct file *filp, struct kobject *kobj,
			     struct bin_attribute *attr,
			     char *buf, loff_t pos, size_t count)
{
	struct device *dev = kobj_to_dev(kobj);
	struct my_dev *priv = dev_get_drvdata(dev);

	if (pos >= priv->firmware_size)
		return 0;

	if (pos + count > priv->firmware_size)
		count = priv->firmware_size - pos;

	memcpy(buf, priv->firmware + pos, count);

	return count;
}

static BIN_ATTR_RO(firmware, 0);

/* Attribute group */
static struct attribute *my_attrs[] = {
	&dev_attr_threshold.attr,
	NULL,
};

static struct bin_attribute *my_bin_attrs[] = {
	&bin_attr_firmware,
	NULL,
};

static const struct attribute_group my_attr_group = {
	.attrs = my_attrs,
	.bin_attrs = my_bin_attrs,
};

/* Register attributes */
static int register_sysfs(struct my_dev *priv)
{
	return sysfs_create_group(&priv->dev->kobj, &my_attr_group);
}

static void unregister_sysfs(struct my_dev *priv)
{
	sysfs_remove_group(&priv->dev->kobj, &my_attr_group);
}

/* Alternative: Use device attribute groups directly */
static const struct attribute_group *my_attr_groups[] = {
	&my_attr_group,
	NULL,
};

/* Set in driver structure */
static struct device_driver my_driver = {
	.groups = my_attr_groups,
};
```

---

## Debugging

### printk and dev_* Functions

```c
/* Use appropriate log level */
pr_emerg("System is unusable\n");
pr_alert("Action must be taken immediately\n");
pr_crit("Critical conditions\n");
pr_err("Error conditions\n");
pr_warn("Warning conditions\n");
pr_notice("Normal but significant\n");
pr_info("Informational\n");
pr_debug("Debug-level messages\n");

/* Device-specific logging (preferred) */
dev_err(dev, "Device error: %d\n", err);
dev_warn(dev, "Device warning\n");
dev_info(dev, "Device information\n");
dev_dbg(dev, "Device debug\n");

/* Rate limited logging */
dev_err_ratelimited(dev, "This might happen often\n");
dev_warn_once(dev, "Only print once\n");
```

### debugfs

```c
#include <linux/debugfs.h>

struct my_dev {
	struct dentry *debugfs_dir;
	u32 debug_value;
};

static int register_debugfs(struct my_dev *priv)
{
	priv->debugfs_dir = debugfs_create_dir("my-device", NULL);
	if (!priv->debugfs_dir)
		return -ENOMEM;

	/* Create files */
	debugfs_create_u32("debug_value", 0644, priv->debugfs_dir,
			   &priv->debug_value);

	debugfs_create_file("registers", 0444, priv->debugfs_dir,
			    priv, &registers_fops);

	return 0;
}

static void unregister_debugfs(struct my_dev *priv)
{
	debugfs_remove_recursive(priv->debugfs_dir);
}

/* Custom debugfs file operations */
static int registers_show(struct seq_file *s, void *unused)
{
	struct my_dev *priv = s->private;

	seq_printf(s, "CTRL:   0x%08x\n", readl(priv->base + CTRL_REG));
	seq_printf(s, "STATUS: 0x%08x\n", readl(priv->base + STATUS_REG));
	seq_printf(s, "DATA:   0x%08x\n", readl(priv->base + DATA_REG));

	return 0;
}

static int registers_open(struct inode *inode, struct file *file)
{
	return single_open(file, registers_show, inode->i_private);
}

static const struct file_operations registers_fops = {
	.open = registers_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};
```

### Tracing

```c
/* Use trace_printk for fast debugging */
trace_printk("Fast trace: value=%d\n", value);

/* Define tracepoints */
#include <trace/events/my_driver.h>

TRACE_EVENT(my_event,
	TP_PROTO(int value, const char *msg),
	TP_ARGS(value, msg),
	TP_STRUCT__entry(
		__field(int, value)
		__string(msg, msg)
	),
	TP_fast_assign(
		__entry->value = value;
		__assign_str(msg, msg);
	),
	TP_printk("value=%d msg=%s", __entry->value, __get_str(msg))
);

/* Use tracepoint */
trace_my_event(42, "test message");
```

---

## Best Practices

### Error Handling

```c
/* Always check return values */
ret = device_register(&my_device);
if (ret) {
	pr_err("Failed to register device: %d\n", ret);
	goto err_register;
}

/* Use goto for cleanup */
err_register:
	kfree(buffer);
err_alloc:
	return ret;

/* Use devm_* functions for automatic cleanup */
priv = devm_kzalloc(dev, sizeof(*priv), GFP_KERNEL);
priv->base = devm_ioremap_resource(dev, res);
devm_request_irq(dev, irq, handler, flags, name, dev_id);
```

### Memory Management

```c
/* Use appropriate allocation flags */
/* GFP_KERNEL: Can sleep (process context) */
ptr = kmalloc(size, GFP_KERNEL);

/* GFP_ATOMIC: Cannot sleep (interrupt context) */
ptr = kmalloc(size, GFP_ATOMIC);

/* Always check for NULL */
if (!ptr)
	return -ENOMEM;

/* Free memory */
kfree(ptr);

/* Use devm_* for automatic cleanup */
ptr = devm_kmalloc(dev, size, GFP_KERNEL);
/* No need to explicitly free */
```

### Locking

```c
/* Choose appropriate lock type */

/* Mutex: Can sleep, process context only */
mutex_lock(&priv->lock);
/* ... */
mutex_unlock(&priv->lock);

/* Spinlock: Cannot sleep, short critical sections */
spin_lock(&priv->lock);
/* ... */
spin_unlock(&priv->lock);

/* Spinlock with IRQ disable (accessed from IRQ) */
unsigned long flags;
spin_lock_irqsave(&priv->lock, flags);
/* ... */
spin_unlock_irqrestore(&priv->lock, flags);
```

### Module Parameters

```c
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug output");

static char *mode = "auto";
module_param(mode, charp, 0444);
MODULE_PARM_DESC(mode, "Operating mode");

/* Use in code */
if (debug)
	pr_info("Debug mode enabled\n");
```

### Module Metadata

```c
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name <your.email@example.com>");
MODULE_DESCRIPTION("Device driver for XYZ hardware");
MODULE_VERSION("1.0");
MODULE_ALIAS("platform:my-device");
```

---

## Resources

- **Linux Device Drivers** (LDD3): [https://lwn.net/Kernel/LDD3/](https://lwn.net/Kernel/LDD3/)
- **Kernel Documentation**: `Documentation/driver-api/` in kernel source
- **Device Tree**: `Documentation/devicetree/` in kernel source
- **Example Drivers**: `drivers/` in kernel source tree
- **Linux Driver Development for Embedded Processors** (Alberto Liberal)
- **Essential Linux Device Drivers** (Sreekrishnan Venkateswaran)

---

Linux driver development requires understanding of kernel internals, hardware interfaces, and proper resource management. Following best practices and using the kernel's device model framework ensures drivers are maintainable, efficient, and safe.
