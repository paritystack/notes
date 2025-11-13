# Linux Kernel Development Patterns

Common patterns, idioms, and best practices used throughout the Linux kernel codebase.

## Table of Contents
- [Coding Style](#coding-style)
- [Design Patterns](#design-patterns)
- [Memory Management Patterns](#memory-management-patterns)
- [Locking and Synchronization](#locking-and-synchronization)
- [Error Handling](#error-handling)
- [Device Driver Patterns](#device-driver-patterns)
- [Data Structures](#data-structures)
- [Kernel APIs](#kernel-apis)
- [Debugging Patterns](#debugging-patterns)
- [Best Practices](#best-practices)

---

## Coding Style

### Basic Rules

The Linux kernel has strict coding style guidelines documented in `Documentation/process/coding-style.rst`.

**Indentation and Formatting:**
```c
// Use tabs (8 characters) for indentation, not spaces
int function_name(int arg1, int arg2)
{
	int local_var;

	if (condition) {
		do_something();
	} else {
		do_something_else();
	}

	return 0;
}
```

**Line Length:**
```c
// Prefer 80 columns, maximum 100 columns
// Break long lines sensibly
static const struct file_operations my_fops = {
	.owner = THIS_MODULE,
	.open = my_open,
	.read = my_read,
	.write = my_write,
	.release = my_release,
};
```

**Naming Conventions:**
```c
// Use descriptive, lowercase names with underscores
int count_active_users(struct user_struct *user);

// Global functions should be prefixed with subsystem name
int netdev_register_device(struct net_device *dev);

// Static functions can be shorter
static int validate(void);

// Avoid Hungarian notation
int nr_pages;           // Good
int iPageCount;         // Bad
```

**Braces:**
```c
// Opening brace on same line for functions, structs, etc.
struct my_struct {
	int member;
};

// But on next line for functions
int my_function(void)
{
	// function body
}

// Single statement doesn't need braces (but be careful)
if (condition)
	return -EINVAL;

// Multiple statements always need braces
if (condition) {
	do_something();
	return 0;
}
```

### Comments

```c
/*
 * Multi-line comments use this format.
 * Each line starts with a star.
 * The closing star-slash is on its own line.
 */

// Single-line comments can use C++ style, but prefer /* */ style

/**
 * function_name - Short description
 * @param1: Description of param1
 * @param2: Description of param2
 *
 * Longer description of what the function does.
 * This can span multiple lines.
 *
 * Return: Description of return value
 */
int function_name(int param1, char *param2)
{
	/* Implementation */
}
```

---

## Design Patterns

### Registration Pattern

The kernel uses registration callbacks extensively for hooking into subsystems.

```c
/* Define operations structure */
struct my_operations {
	int (*init)(void);
	void (*cleanup)(void);
	int (*process)(void *data);
};

/* Define registration structure */
struct my_driver {
	const char *name;
	struct my_operations *ops;
	struct list_head list;
};

/* Registration function */
int register_my_driver(struct my_driver *driver)
{
	if (!driver || !driver->ops)
		return -EINVAL;

	/* Add to global list with locking */
	mutex_lock(&drivers_mutex);
	list_add_tail(&driver->list, &drivers_list);
	mutex_unlock(&drivers_mutex);

	/* Initialize if needed */
	if (driver->ops->init)
		return driver->ops->init();

	return 0;
}

/* Unregistration */
void unregister_my_driver(struct my_driver *driver)
{
	mutex_lock(&drivers_mutex);
	list_del(&driver->list);
	mutex_unlock(&drivers_mutex);

	if (driver->ops->cleanup)
		driver->ops->cleanup();
}
```

### Object-Oriented Patterns in C

The kernel implements inheritance-like patterns using structure embedding.

```c
/* Base "class" */
struct device {
	const char *name;
	struct device *parent;
	void (*release)(struct device *dev);
};

/* Derived "class" */
struct pci_device {
	struct device dev;      /* Embedded base */
	unsigned int vendor;
	unsigned int device_id;
};

/* Upcast: derived to base */
struct pci_device *pci_dev;
struct device *dev = &pci_dev->dev;

/* Downcast: base to derived using container_of */
struct device *dev;
struct pci_device *pci_dev = container_of(dev, struct pci_device, dev);
```

### Reference Counting Pattern

```c
struct my_object {
	atomic_t refcount;
	/* other fields */
};

/* Initialize reference count */
static void my_object_init(struct my_object *obj)
{
	atomic_set(&obj->refcount, 1);
}

/* Get reference (increment) */
static inline struct my_object *my_object_get(struct my_object *obj)
{
	if (obj)
		atomic_inc(&obj->refcount);
	return obj;
}

/* Put reference (decrement and free if zero) */
static inline void my_object_put(struct my_object *obj)
{
	if (obj && atomic_dec_and_test(&obj->refcount))
		my_object_destroy(obj);
}

/* Usage */
struct my_object *obj = my_object_alloc();  /* refcount = 1 */
struct my_object *obj2 = my_object_get(obj); /* refcount = 2 */
my_object_put(obj);   /* refcount = 1 */
my_object_put(obj2);  /* refcount = 0, object destroyed */
```

### Kernel Object (kobject) Pattern

```c
#include <linux/kobject.h>

struct my_object {
	struct kobject kobj;
	int value;
};

static struct kobj_type my_ktype = {
	.release = my_release,
	.sysfs_ops = &my_sysfs_ops,
	.default_attrs = my_attrs,
};

/* Create object */
struct my_object *obj = kzalloc(sizeof(*obj), GFP_KERNEL);
kobject_init(&obj->kobj, &my_ktype);
kobject_add(&obj->kobj, parent, "my_object");

/* Get reference */
kobject_get(&obj->kobj);

/* Release reference */
kobject_put(&obj->kobj);
```

---

## Memory Management Patterns

### Allocation Patterns

```c
/* Kernel memory allocation */
void *ptr = kmalloc(size, GFP_KERNEL);  /* Can sleep */
void *ptr = kmalloc(size, GFP_ATOMIC);  /* Cannot sleep, use in interrupt */
void *ptr = kzalloc(size, GFP_KERNEL);  /* Zeroed memory */

/* Large allocations */
void *ptr = vmalloc(size);  /* Virtually contiguous, physically may not be */

/* Page allocation */
struct page *page = alloc_page(GFP_KERNEL);
struct page *pages = alloc_pages(GFP_KERNEL, order);  /* 2^order pages */

/* Per-CPU variables */
DEFINE_PER_CPU(int, my_var);
int val = get_cpu_var(my_var);
put_cpu_var(my_var);

/* Slab/KMEM cache for frequent allocations */
struct kmem_cache *my_cache;

my_cache = kmem_cache_create("my_cache",
                              sizeof(struct my_struct),
                              0, SLAB_HWCACHE_ALIGN, NULL);

struct my_struct *obj = kmem_cache_alloc(my_cache, GFP_KERNEL);
kmem_cache_free(my_cache, obj);
```

### Memory Barriers

```c
/* Compiler barrier - prevent compiler reordering */
barrier();

/* Memory barriers - prevent CPU reordering */
mb();    /* Full memory barrier */
rmb();   /* Read memory barrier */
wmb();   /* Write memory barrier */
smp_mb(); /* SMP memory barrier */

/* Example: Producer-consumer */
/* Producer */
data->value = 42;
smp_wmb();  /* Ensure value is written before flag */
data->ready = 1;

/* Consumer */
while (!data->ready)
	cpu_relax();
smp_rmb();  /* Ensure flag is read before value */
value = data->value;
```

### Page Flags and Reference Counting

```c
/* Get a page reference */
get_page(page);

/* Release a page reference */
put_page(page);

/* Check if page is locked */
if (PageLocked(page))
	/* ... */

/* Lock a page */
lock_page(page);
unlock_page(page);

/* Page flags */
SetPageDirty(page);
ClearPageDirty(page);
TestSetPageLocked(page);
```

---

## Locking and Synchronization

### Spinlock Pattern

```c
/* Define spinlock */
spinlock_t my_lock;

/* Initialize */
spin_lock_init(&my_lock);

/* Use in process context */
spin_lock(&my_lock);
/* Critical section */
spin_unlock(&my_lock);

/* Use with IRQ disabling (if accessed from interrupt) */
unsigned long flags;
spin_lock_irqsave(&my_lock, flags);
/* Critical section */
spin_unlock_irqrestore(&my_lock, flags);

/* Bottom-half (softirq) protection */
spin_lock_bh(&my_lock);
/* Critical section */
spin_unlock_bh(&my_lock);
```

### Mutex Pattern

```c
/* Define mutex */
struct mutex my_mutex;

/* Initialize */
mutex_init(&my_mutex);

/* Use (can sleep, so only in process context) */
mutex_lock(&my_mutex);
/* Critical section */
mutex_unlock(&my_mutex);

/* Trylock */
if (mutex_trylock(&my_mutex)) {
	/* Got the lock */
	mutex_unlock(&my_mutex);
}

/* Interruptible lock */
if (mutex_lock_interruptible(&my_mutex))
	return -EINTR;
/* Critical section */
mutex_unlock(&my_mutex);
```

### Read-Write Locks

```c
/* Spinlock version */
rwlock_t my_rwlock;
rwlock_init(&my_rwlock);

/* Readers */
read_lock(&my_rwlock);
/* Read data */
read_unlock(&my_rwlock);

/* Writer */
write_lock(&my_rwlock);
/* Modify data */
write_unlock(&my_rwlock);

/* Semaphore version (can sleep) */
struct rw_semaphore my_rwsem;
init_rwsem(&my_rwsem);

down_read(&my_rwsem);
/* Read data */
up_read(&my_rwsem);

down_write(&my_rwsem);
/* Modify data */
up_write(&my_rwsem);
```

### RCU (Read-Copy-Update) Pattern

```c
/* RCU list */
struct my_data {
	int value;
	struct list_head list;
	struct rcu_head rcu;
};

static LIST_HEAD(my_list);
static DEFINE_SPINLOCK(list_lock);

/* Read (no lock needed!) */
rcu_read_lock();
list_for_each_entry_rcu(entry, &my_list, list) {
	/* Read entry->value */
}
rcu_read_unlock();

/* Update (needs lock) */
spin_lock(&list_lock);
new = kmalloc(sizeof(*new), GFP_KERNEL);
new->value = 42;
list_add_rcu(&new->list, &my_list);
spin_unlock(&list_lock);

/* Delete */
static void my_data_free(struct rcu_head *head)
{
	struct my_data *entry = container_of(head, struct my_data, rcu);
	kfree(entry);
}

spin_lock(&list_lock);
list_del_rcu(&entry->list);
spin_unlock(&list_lock);
call_rcu(&entry->rcu, my_data_free);  /* Deferred free */
```

### Completion Pattern

```c
/* Declare completion */
struct completion my_completion;

/* Initialize */
init_completion(&my_completion);

/* Wait for completion */
wait_for_completion(&my_completion);

/* Timeout version */
if (!wait_for_completion_timeout(&my_completion, msecs_to_jiffies(5000)))
	printk(KERN_ERR "Timeout waiting for completion\n");

/* Signal completion */
complete(&my_completion);

/* Signal all waiters */
complete_all(&my_completion);
```

### Atomic Operations

```c
/* Atomic integer */
atomic_t counter = ATOMIC_INIT(0);

atomic_inc(&counter);
atomic_dec(&counter);
atomic_add(5, &counter);
atomic_sub(3, &counter);

/* Read */
int val = atomic_read(&counter);

/* Set */
atomic_set(&counter, 10);

/* Conditional operations */
if (atomic_dec_and_test(&counter))
	/* Counter reached zero */

if (atomic_inc_and_test(&counter))
	/* Counter is zero after increment */

/* Compare and swap */
int old = 5;
int new = 10;
atomic_cmpxchg(&counter, old, new);

/* Bitops */
unsigned long flags = 0;
set_bit(0, &flags);
clear_bit(0, &flags);
if (test_bit(0, &flags))
	/* Bit is set */

/* Atomic bitops */
test_and_set_bit(0, &flags);
test_and_clear_bit(0, &flags);
```

---

## Error Handling

### Error Code Pattern

```c
/* Return negative error codes, 0 for success */
int my_function(void)
{
	if (error_condition)
		return -EINVAL;  /* Invalid argument */

	if (no_memory)
		return -ENOMEM;  /* Out of memory */

	if (timeout)
		return -ETIMEDOUT;

	return 0;  /* Success */
}

/* Caller checks return value */
int ret = my_function();
if (ret) {
	printk(KERN_ERR "Function failed: %d\n", ret);
	return ret;  /* Propagate error */
}
```

### Common Error Codes

```c
-EINVAL    /* Invalid argument */
-ENOMEM    /* Out of memory */
-EFAULT    /* Bad address (copy_from/to_user failed) */
-EBUSY     /* Device or resource busy */
-EAGAIN    /* Try again (non-blocking operation) */
-EINTR     /* Interrupted system call */
-EIO       /* I/O error */
-ENODEV    /* No such device */
-ENOTTY    /* Inappropriate ioctl for device */
-EPERM     /* Operation not permitted */
-EACCES    /* Permission denied */
-EEXIST    /* File exists */
-ENOENT    /* No such file or directory */
-ETIMEDOUT /* Connection timed out */
```

### Cleanup with goto Pattern

```c
int complex_function(void)
{
	struct resource1 *res1 = NULL;
	struct resource2 *res2 = NULL;
	struct resource3 *res3 = NULL;
	int ret;

	res1 = allocate_resource1();
	if (!res1) {
		ret = -ENOMEM;
		goto out;
	}

	res2 = allocate_resource2();
	if (!res2) {
		ret = -ENOMEM;
		goto free_res1;
	}

	res3 = allocate_resource3();
	if (!res3) {
		ret = -ENOMEM;
		goto free_res2;
	}

	/* Do work */
	ret = do_work(res1, res2, res3);
	if (ret)
		goto free_res3;

	/* Success path */
	return 0;

free_res3:
	free_resource3(res3);
free_res2:
	free_resource2(res2);
free_res1:
	free_resource1(res1);
out:
	return ret;
}
```

### ERR_PTR Pattern

```c
/* Return pointer or error */
struct my_struct *my_function(void)
{
	struct my_struct *ptr;

	ptr = kmalloc(sizeof(*ptr), GFP_KERNEL);
	if (!ptr)
		return ERR_PTR(-ENOMEM);

	if (some_error) {
		kfree(ptr);
		return ERR_PTR(-EINVAL);
	}

	return ptr;
}

/* Caller checks for error */
struct my_struct *ptr = my_function();
if (IS_ERR(ptr)) {
	int err = PTR_ERR(ptr);
	printk(KERN_ERR "Function failed: %d\n", err);
	return err;
}

/* Use ptr */
kfree(ptr);
```

---

## Device Driver Patterns

### Character Device Pattern

```c
#include <linux/fs.h>
#include <linux/cdev.h>

static dev_t dev_num;
static struct cdev my_cdev;
static struct class *my_class;

static int my_open(struct inode *inode, struct file *filp)
{
	/* Initialize private data */
	return 0;
}

static int my_release(struct inode *inode, struct file *filp)
{
	/* Cleanup */
	return 0;
}

static ssize_t my_read(struct file *filp, char __user *buf,
                       size_t count, loff_t *pos)
{
	/* Read data and copy to user space */
	if (copy_to_user(buf, kernel_buf, count))
		return -EFAULT;

	return count;
}

static ssize_t my_write(struct file *filp, const char __user *buf,
                        size_t count, loff_t *pos)
{
	/* Copy from user space and write */
	if (copy_from_user(kernel_buf, buf, count))
		return -EFAULT;

	return count;
}

static long my_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	switch (cmd) {
	case MY_IOCTL_CMD:
		/* Handle command */
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

static int __init my_init(void)
{
	int ret;

	/* Allocate device number */
	ret = alloc_chrdev_region(&dev_num, 0, 1, "mydev");
	if (ret < 0)
		return ret;

	/* Initialize cdev */
	cdev_init(&my_cdev, &my_fops);
	my_cdev.owner = THIS_MODULE;

	/* Add cdev */
	ret = cdev_add(&my_cdev, dev_num, 1);
	if (ret < 0)
		goto unregister_chrdev;

	/* Create device class */
	my_class = class_create(THIS_MODULE, "myclass");
	if (IS_ERR(my_class)) {
		ret = PTR_ERR(my_class);
		goto del_cdev;
	}

	/* Create device */
	device_create(my_class, NULL, dev_num, NULL, "mydev");

	return 0;

del_cdev:
	cdev_del(&my_cdev);
unregister_chrdev:
	unregister_chrdev_region(dev_num, 1);
	return ret;
}

static void __exit my_exit(void)
{
	device_destroy(my_class, dev_num);
	class_destroy(my_class);
	cdev_del(&my_cdev);
	unregister_chrdev_region(dev_num, 1);
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```

### Platform Device Pattern

```c
#include <linux/platform_device.h>

static int my_probe(struct platform_device *pdev)
{
	struct resource *res;
	void __iomem *base;

	/* Get resources */
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (!res)
		return -ENODEV;

	/* Map registers */
	base = devm_ioremap_resource(&pdev->dev, res);
	if (IS_ERR(base))
		return PTR_ERR(base);

	/* Store in device private data */
	platform_set_drvdata(pdev, base);

	return 0;
}

static int my_remove(struct platform_device *pdev)
{
	/* Cleanup */
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
		.name = "my-driver",
		.of_match_table = my_of_match,
	},
};

module_platform_driver(my_driver);
```

### Interrupt Handler Pattern

```c
#include <linux/interrupt.h>

static irqreturn_t my_interrupt(int irq, void *dev_id)
{
	struct my_device *dev = dev_id;
	u32 status;

	/* Read interrupt status */
	status = readl(dev->base + STATUS_REG);

	if (!(status & MY_IRQ_FLAG))
		return IRQ_NONE;  /* Not our interrupt */

	/* Clear interrupt */
	writel(status, dev->base + STATUS_REG);

	/* Handle interrupt - do minimal work */
	/* Schedule bottom half if needed */
	tasklet_schedule(&dev->tasklet);

	return IRQ_HANDLED;
}

/* Bottom half (tasklet) */
static void my_tasklet_func(unsigned long data)
{
	struct my_device *dev = (struct my_device *)data;

	/* Do heavy work here */
}

/* Request IRQ */
ret = request_irq(irq, my_interrupt, IRQF_SHARED, "mydev", dev);

/* Free IRQ */
free_irq(irq, dev);

/* Threaded IRQ (for handlers that can sleep) */
ret = request_threaded_irq(irq, NULL, my_threaded_handler,
                            IRQF_ONESHOT, "mydev", dev);
```

---

## Data Structures

### Linked Lists

```c
#include <linux/list.h>

struct my_node {
	int data;
	struct list_head list;
};

/* Define and initialize list head */
static LIST_HEAD(my_list);

/* Add entry */
struct my_node *node = kmalloc(sizeof(*node), GFP_KERNEL);
node->data = 42;
list_add(&node->list, &my_list);       /* Add to head */
list_add_tail(&node->list, &my_list);  /* Add to tail */

/* Iterate */
struct my_node *entry;
list_for_each_entry(entry, &my_list, list) {
	printk(KERN_INFO "data: %d\n", entry->data);
}

/* Safe iteration (allows deletion) */
struct my_node *tmp;
list_for_each_entry_safe(entry, tmp, &my_list, list) {
	if (entry->data == 42) {
		list_del(&entry->list);
		kfree(entry);
	}
}

/* Check if empty */
if (list_empty(&my_list))
	printk(KERN_INFO "List is empty\n");
```

### Hash Tables

```c
#include <linux/hashtable.h>

#define HASH_BITS 8

struct my_entry {
	int key;
	int value;
	struct hlist_node hash;
};

/* Declare hash table */
static DEFINE_HASHTABLE(my_hash, HASH_BITS);

/* Initialize */
hash_init(my_hash);

/* Add entry */
struct my_entry *entry = kmalloc(sizeof(*entry), GFP_KERNEL);
entry->key = 123;
entry->value = 456;
hash_add(my_hash, &entry->hash, entry->key);

/* Find entry */
struct my_entry *found = NULL;
hash_for_each_possible(my_hash, entry, hash, key) {
	if (entry->key == key) {
		found = entry;
		break;
	}
}

/* Delete entry */
hash_del(&entry->hash);

/* Iterate all entries */
int bkt;
hash_for_each(my_hash, bkt, entry, hash) {
	printk(KERN_INFO "key=%d value=%d\n", entry->key, entry->value);
}
```

### Radix Tree

```c
#include <linux/radix-tree.h>

static RADIX_TREE(my_tree, GFP_KERNEL);

/* Insert */
void *item = kmalloc(sizeof(struct my_data), GFP_KERNEL);
radix_tree_insert(&my_tree, index, item);

/* Lookup */
void *found = radix_tree_lookup(&my_tree, index);

/* Delete */
void *deleted = radix_tree_delete(&my_tree, index);
kfree(deleted);

/* Iterate */
struct radix_tree_iter iter;
void **slot;
radix_tree_for_each_slot(slot, &my_tree, &iter, start) {
	void *item = radix_tree_deref_slot(slot);
	/* Process item */
}
```

### Red-Black Tree

```c
#include <linux/rbtree.h>

struct my_node {
	int key;
	struct rb_node node;
};

static struct rb_root my_tree = RB_ROOT;

/* Insert */
int my_insert(struct rb_root *root, struct my_node *data)
{
	struct rb_node **new = &(root->rb_node), *parent = NULL;

	while (*new) {
		struct my_node *this = container_of(*new, struct my_node, node);

		parent = *new;
		if (data->key < this->key)
			new = &((*new)->rb_left);
		else if (data->key > this->key)
			new = &((*new)->rb_right);
		else
			return -EEXIST;
	}

	rb_link_node(&data->node, parent, new);
	rb_insert_color(&data->node, root);

	return 0;
}

/* Search */
struct my_node *my_search(struct rb_root *root, int key)
{
	struct rb_node *node = root->rb_node;

	while (node) {
		struct my_node *data = container_of(node, struct my_node, node);

		if (key < data->key)
			node = node->rb_left;
		else if (key > data->key)
			node = node->rb_right;
		else
			return data;
	}

	return NULL;
}

/* Erase */
rb_erase(&node->node, &my_tree);
```

---

## Kernel APIs

### Workqueues

```c
#include <linux/workqueue.h>

struct work_struct my_work;

/* Work function */
static void my_work_func(struct work_struct *work)
{
	/* Do work in process context */
}

/* Initialize */
INIT_WORK(&my_work, my_work_func);

/* Schedule work */
schedule_work(&my_work);

/* Delayed work */
struct delayed_work my_delayed_work;
INIT_DELAYED_WORK(&my_delayed_work, my_work_func);
schedule_delayed_work(&my_delayed_work, msecs_to_jiffies(1000));

/* Cancel work */
cancel_work_sync(&my_work);
cancel_delayed_work_sync(&my_delayed_work);
```

### Timers

```c
#include <linux/timer.h>

struct timer_list my_timer;

/* Timer callback */
static void my_timer_callback(struct timer_list *t)
{
	/* Timer expired */
	printk(KERN_INFO "Timer expired\n");

	/* Reschedule if needed */
	mod_timer(&my_timer, jiffies + msecs_to_jiffies(1000));
}

/* Initialize and start timer */
timer_setup(&my_timer, my_timer_callback, 0);
mod_timer(&my_timer, jiffies + msecs_to_jiffies(1000));

/* Stop timer */
del_timer_sync(&my_timer);

/* High-resolution timers */
#include <linux/hrtimer.h>

struct hrtimer my_hrtimer;

static enum hrtimer_restart my_hrtimer_callback(struct hrtimer *timer)
{
	/* Timer expired */
	return HRTIMER_NORESTART;  /* Or HRTIMER_RESTART */
}

hrtimer_init(&my_hrtimer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
my_hrtimer.function = my_hrtimer_callback;
hrtimer_start(&my_hrtimer, ms_to_ktime(1000), HRTIMER_MODE_REL);
```

### Wait Queues

```c
#include <linux/wait.h>

static DECLARE_WAIT_QUEUE_HEAD(my_wait_queue);
static int condition = 0;

/* Wait for condition */
wait_event(my_wait_queue, condition != 0);

/* Wait with timeout */
int ret = wait_event_timeout(my_wait_queue, condition != 0,
                              msecs_to_jiffies(5000));

/* Interruptible wait */
if (wait_event_interruptible(my_wait_queue, condition != 0))
	return -ERESTARTSYS;

/* Wake up waiters */
condition = 1;
wake_up(&my_wait_queue);        /* Wake one */
wake_up_all(&my_wait_queue);    /* Wake all */
wake_up_interruptible(&my_wait_queue);
```

### Kernel Threads

```c
#include <linux/kthread.h>

static struct task_struct *my_thread;

static int my_thread_func(void *data)
{
	while (!kthread_should_stop()) {
		/* Do work */

		/* Sleep */
		msleep(1000);

		/* Or wait for condition */
		wait_event_interruptible(queue, condition || kthread_should_stop());
	}

	return 0;
}

/* Create and start thread */
my_thread = kthread_run(my_thread_func, NULL, "my_thread");
if (IS_ERR(my_thread))
	return PTR_ERR(my_thread);

/* Stop thread */
kthread_stop(my_thread);
```

---

## Debugging Patterns

### Print Debugging

```c
/* Use appropriate log level */
printk(KERN_EMERG   "Emergency\n");    /* System unusable */
printk(KERN_ALERT   "Alert\n");        /* Action must be taken */
printk(KERN_CRIT    "Critical\n");     /* Critical conditions */
printk(KERN_ERR     "Error\n");        /* Error conditions */
printk(KERN_WARNING "Warning\n");      /* Warning conditions */
printk(KERN_NOTICE  "Notice\n");       /* Normal but significant */
printk(KERN_INFO    "Info\n");         /* Informational */
printk(KERN_DEBUG   "Debug\n");        /* Debug messages */

/* Modern API */
pr_emerg("Emergency\n");
pr_err("Error\n");
pr_info("Info\n");
pr_debug("Debug\n");  /* Only if DEBUG is defined */

/* Device-specific logging */
dev_err(&pdev->dev, "Device error\n");
dev_info(&pdev->dev, "Device info\n");
```

### Dynamic Debug

```c
/* Compile with CONFIG_DYNAMIC_DEBUG */

/* Use pr_debug or dev_dbg */
pr_debug("Debug message: value=%d\n", value);
dev_dbg(&dev->dev, "Device debug: %s\n", msg);

/* Enable at runtime */
/* echo 'file mydriver.c +p' > /sys/kernel/debug/dynamic_debug/control */
```

### Assertions

```c
/* BUG and WARN macros */
BUG_ON(bad_condition);        /* Panic if true */
WARN_ON(warning_condition);   /* Warning if true */

if (WARN_ON_ONCE(ptr == NULL))
	return -EINVAL;

/* Better: return error instead of crashing */
if (WARN(bad_condition, "Something went wrong: %d\n", value))
	return -EINVAL;
```

### Tracing

```c
#include <linux/trace_events.h>

/* Use ftrace */
trace_printk("Fast trace message: %d\n", value);

/* Define tracepoints */
#include <trace/events/mydriver.h>

TRACE_EVENT(my_event,
	TP_PROTO(int value),
	TP_ARGS(value),
	TP_STRUCT__entry(
		__field(int, value)
	),
	TP_fast_assign(
		__entry->value = value;
	),
	TP_printk("value=%d", __entry->value)
);

/* Use tracepoint */
trace_my_event(42);
```

---

## Best Practices

### Resource Management

```c
/* Use devm_* functions for automatic cleanup on error/remove */
void __iomem *base = devm_ioremap_resource(&pdev->dev, res);
int *ptr = devm_kmalloc(&pdev->dev, size, GFP_KERNEL);
int irq = devm_request_irq(&pdev->dev, irq_num, handler, flags, name, dev);

/* These are automatically freed when device is removed */
```

### Copy to/from User Space

```c
/* Always use copy_to_user/copy_from_user */
if (copy_to_user(user_buf, kernel_buf, count))
	return -EFAULT;

if (copy_from_user(kernel_buf, user_buf, count))
	return -EFAULT;

/* For single values */
int value;
if (get_user(value, (int __user *)arg))
	return -EFAULT;

if (put_user(value, (int __user *)arg))
	return -EFAULT;

/* Check access */
if (!access_ok(user_buf, count))
	return -EFAULT;
```

### Module Parameters

```c
/* Define module parameters */
static int debug = 0;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Enable debug mode");

static char *name = "default";
module_param(name, charp, 0644);
MODULE_PARM_DESC(name, "Device name");

/* Load module with parameters */
/* insmod mymodule.ko debug=1 name="custom" */
```

### SMP Safety

```c
/* Always consider SMP (multiprocessor) safety */

/* Use per-CPU variables for lock-free data */
DEFINE_PER_CPU(int, my_counter);

int val = get_cpu_var(my_counter);
val++;
put_cpu_var(my_counter);

/* Use proper locking */
/* Identify data that needs protection */
/* Choose appropriate lock type (spinlock vs mutex) */
/* Keep critical sections short */
/* Avoid nested locks (lock ordering) */
```

### Power Management

```c
/* Implement PM operations */
static int my_suspend(struct device *dev)
{
	/* Save state, disable device */
	return 0;
}

static int my_resume(struct device *dev)
{
	/* Restore state, enable device */
	return 0;
}

static const struct dev_pm_ops my_pm_ops = {
	.suspend = my_suspend,
	.resume = my_resume,
};

static struct platform_driver my_driver = {
	.driver = {
		.name = "my-driver",
		.pm = &my_pm_ops,
	},
};
```

---

## Common Pitfalls

### Don't Do This

```c
/* DON'T use floating point in kernel */
// float x = 3.14;  /* Wrong! */

/* DON'T use large stack allocations */
// char buffer[8192];  /* Too big for stack */
/* Use kmalloc instead */

/* DON'T sleep in atomic context */
spin_lock(&lock);
// msleep(100);  /* Wrong! */
spin_unlock(&lock);

/* DON'T access user space directly */
// int *user_ptr;
// *user_ptr = 5;  /* Wrong! Use copy_to_user */

/* DON'T ignore return values */
// kmalloc(size, GFP_KERNEL);  /* Check for NULL! */

/* DON'T use unbounded loops */
// while (1) { }  /* Use kthread_should_stop() */
```

---

## Resources

- **Kernel Documentation**: `Documentation/` in kernel source
- **Coding Style**: `Documentation/process/coding-style.rst`
- **API Documentation**: `Documentation/core-api/`
- **Linux Kernel Development** by Robert Love
- **Linux Device Drivers** by Corbet, Rubini, and Kroah-Hartman
- **Understanding the Linux Kernel** by Bovet and Cesati

---

Linux kernel development follows well-established patterns that promote consistency, safety, and performance. Understanding these patterns is essential for writing quality kernel code that integrates well with the rest of the kernel.
