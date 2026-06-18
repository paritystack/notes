# Android Binder

## Overview

Binder is Android's primary inter-process communication (IPC) mechanism: an
**object-oriented, kernel-mediated RPC system** that lets a process invoke methods on an
object living in another process as if it were local. Almost every framework call an app
makes — `startActivity`, `getSystemService`, querying packages, talking to a HAL — is a
Binder transaction under the hood. It is the connective tissue of the
[Android Internals](internals.md) stack: apps talk to
[SystemServer & Core Services](system_server.md) over Binder, processes forked by
[Zygote & App Startup](zygote_startup.md) join a Binder thread pool at startup, and
[Project Treble & HALs](treble_hal.md) expose vendor hardware through stable AIDL/HIDL
Binder interfaces. Native code reaches the same driver through `libbinder` (see
[NDK & JNI](ndk_jni.md)).

Why a custom mechanism instead of pipes, sockets, or System V IPC?

- **One copy, not two.** A normal socket copies data user → kernel → user. Binder copies
  the payload once, from the sender's buffer directly into a region the receiver has
  `mmap`ed, then hands the receiver a pointer.
- **Kernel-guaranteed identity.** The driver stamps each transaction with the caller's
  real UID/PID. The receiver can trust it — a malicious app cannot forge who it claims to
  be. This is the foundation of Android's permission model.
- **Object semantics.** Binder passes *object references* across process boundaries with
  automatic reference counting and **death notifications**, not just bytes.

## Architecture

A client holds a **proxy**; the real object (a **stub** / `Bn...`) lives in the server. The
kernel `/dev/binder` driver routes transactions between them and translates object handles
across the boundary.

```text
   Client process                 kernel                 Server process
 ┌────────────────┐                                    ┌────────────────┐
 │ IFoo proxy     │                                    │ IFoo.Stub      │
 │  (BpBinder)    │                                    │  (BBinder)     │
 │    .doThing()  │                                    │  onTransact()  │
 └──────┬─────────┘                                    └──────▲─────────┘
        │ Parcel + code                                       │
        ▼                                                     │
   ioctl(BINDER_WRITE_READ)  ──►  ┌──────────────┐  ──►  binder thread
                                  │ binder driver │       wakes, reads
   reply ◄───────────────────────│  /dev/binder  │◄──── writes reply
                                  └──────────────┘
                            single copy into receiver's mmap'd buffer
```

Marshalling: the proxy packs arguments into a **Parcel**, issues a `BINDER_WRITE_READ`
ioctl with a transaction *code* (which method) and the target handle. The driver copies the
Parcel once into the server's mapped transaction buffer and wakes a server binder thread,
which unpacks the Parcel, dispatches on the code in `onTransact()`, and writes a reply
Parcel back the same way.

## The Binder driver

`/dev/binder` is a character device backed by a kernel module
(`CONFIG_ANDROID_BINDER_IPC=y`; modern devices also expose `/dev/hwbinder` for HALs and
`/dev/vndbinder` for vendor-to-vendor traffic). Each process `mmap`s a region the driver
uses as its receive buffer — that mapping is what makes the single-copy transfer possible.

All traffic flows through one multiplexing ioctl:

```c
BINDER_WRITE_READ       // write outgoing + read incoming transactions (the hot path)
BINDER_SET_MAX_THREADS  // cap the binder thread pool
BINDER_SET_CONTEXT_MGR  // claim handle 0 — only servicemanager does this
BINDER_VERSION          // driver/protocol version check
```

The transaction payload the driver moves looks roughly like:

```c
struct binder_transaction_data {
    union { __u32 handle; binder_uintptr_t ptr; } target; // who to call
    binder_uintptr_t cookie;
    __u32 code;            // which method (the transaction code)
    __u32 flags;           // e.g. TF_ONE_WAY
    pid_t sender_pid;      // \  filled in by the kernel —
    uid_t sender_euid;     // /  the caller cannot fake these
    binder_size_t data_size;
    binder_size_t offsets_size;   // where embedded binder/FD objects live
    union { struct { binder_uintptr_t buffer, offsets; } ptr; __u8 buf[8]; } data;
};
```

The `offsets` array tells the driver which positions in the buffer hold *special* objects
(binder references or file descriptors) so it can translate handles and duplicate FDs into
the receiving process. Each process's transaction buffer is **~1 MB and shared across all
in-flight transactions** — exceed it and the call fails (see Pitfalls).

```bash
adb shell cat /sys/kernel/debug/binder/stats         # transaction counts, errors
adb shell cat /sys/kernel/debug/binder/transactions  # in-flight transactions
```

## ServiceManager — the context manager

Binder transactions target a numeric *handle*, but clients want services by *name*.
**`servicemanager`** is the registry that bridges the two. It is the first process to open
`/dev/binder` and claims the well-known **handle 0** via `BINDER_SET_CONTEXT_MGR`; everyone
can reach handle 0 without a prior introduction.

```text
system_server:  ServiceManager.addService("activity", amsBinder)
app:            IBinder b = ServiceManager.getService("activity")
                IActivityManager am = IActivityManager.Stub.asInterface(b)
                am.startActivity(...)        // Binder txn → system_server
```

App developers rarely touch `ServiceManager` directly — `Context.getSystemService()` returns
a manager object that already wraps the right proxy. The full service catalog and the
services hosted in `system_server` are covered in
[SystemServer & Core Services](system_server.md).

```bash
adb shell service list                 # every Binder service registered with servicemanager
adb shell dumpsys activity processes   # callers + state for a given service
```

## AIDL: interface to Stub and Proxy

You almost never write Binder marshalling by hand. You declare an interface in **AIDL**
(Android Interface Definition Language) and the build generates the proxy and stub.

```java
// ICalculator.aidl
package com.example;

interface ICalculator {
    int add(int a, int b);
    int subtract(int a, int b);
}
```

Server side — implement the generated `Stub` and return it from `onBind`:

```java
public class CalculatorService extends Service {
    private final ICalculator.Stub binder = new ICalculator.Stub() {
        @Override public int add(int a, int b)      { return a + b; }
        @Override public int subtract(int a, int b) { return a - b; }
    };

    @Override public IBinder onBind(Intent intent) { return binder; }
}
```

Client side — bind, then turn the raw `IBinder` into a typed proxy with `asInterface`:

```java
ServiceConnection connection = new ServiceConnection() {
    public void onServiceConnected(ComponentName name, IBinder service) {
        ICalculator calc = ICalculator.Stub.asInterface(service);
        int result = calc.add(5, 3);   // 8 — a Binder transaction across processes
    }
    public void onServiceDisconnected(ComponentName name) { /* clean up */ }
};
bindService(intent, connection, Context.BIND_AUTO_CREATE);
```

What AIDL generates is the boilerplate the architecture diagram described — a `Stub` that
dispatches on the transaction code, and a `Proxy` that marshals into a Parcel:

```java
// Server: Stub.onTransact() runs on a binder thread
public boolean onTransact(int code, Parcel data, Parcel reply, int flags) {
    switch (code) {
        case TRANSACTION_add:
            int a = data.readInt(), b = data.readInt();
            reply.writeNoException();
            reply.writeInt(this.add(a, b));
            return true;
    }
    return super.onTransact(code, data, reply, flags);
}

// Client: Proxy marshals args, blocks on transact(), unmarshals the reply
public int add(int a, int b) throws RemoteException {
    Parcel data = Parcel.obtain(), reply = Parcel.obtain();
    try {
        data.writeInt(a); data.writeInt(b);
        mRemote.transact(TRANSACTION_add, data, reply, 0);
        reply.readException();
        return reply.readInt();
    } finally { reply.recycle(); data.recycle(); }
}
```

If the remote process is gone the `transact` throws `RemoteException` (specifically
`DeadObjectException`) — every cross-process call can fail and callers must handle it.

## Parcels and marshalling

A **Parcel** is the serialization container Binder moves. It is *not* general-purpose
storage — its byte layout is tied to the live transaction (it can carry live handles), so
never persist a Parcel to disk or across versions. A Parcel can hold:

- primitives, `String`/`CharSequence`, arrays and lists;
- `Parcelable` objects (you implement `writeToParcel`/`CREATOR`);
- **Binder references** via `writeStrongBinder()` / `readStrongBinder()` — this is how you
  pass a callback object back to a server;
- **file descriptors** via `ParcelFileDescriptor` — the driver `dup()`s the FD into the
  receiving process. This is the escape hatch for large data: instead of stuffing megabytes
  into a Parcel, pass an FD to ashmem/a pipe/a file.

`Parcelable` (Binder-oriented, fast) is preferred over Java `Serializable` (reflection-based,
slow) for anything that crosses a Binder boundary.

## Threads and the binder thread pool

Each process serving Binder calls runs a **thread pool**; the kernel hands an incoming
transaction to an idle binder thread and spawns more on demand up to the cap.

```cpp
ProcessState::self()->setThreadPoolMaxThreads(15); // default cap is 15 (+ the main thread)
ProcessState::self()->startThreadPool();
IPCThreadState::self()->joinThreadPool();          // donate the calling thread too
```

Consequences worth internalizing:

- **A synchronous Binder call blocks the calling thread** until the reply arrives. Never make
  one from the UI/main thread to a service that might be slow.
- **Calls are re-entrant.** If A calls B and B calls back into A on the same logical chain,
  the driver preserves the thread of control, so nested transactions don't deadlock against
  themselves.
- **The pool is finite.** If every binder thread in a service is blocked (e.g. on a lock held
  by a stuck thread), new callers queue and eventually time out. In `system_server` this is
  exactly what the Watchdog watches for — see [SystemServer & Core Services](system_server.md).

## Synchronous calls vs `oneway`

By default a transaction is synchronous: the caller blocks for a reply. Mark an AIDL method
(or whole interface) `oneway` to make it **fire-and-forget**:

```java
interface IEvents {
    oneway void onChanged(int what);   // returns immediately, no reply, cannot return a value
}
```

`oneway` calls return as soon as the kernel queues them. They are useful for callbacks and
notifications, but note the trade-offs: no return value or exception propagation, and
**ordering is only guaranteed per-destination-object**, not globally. `oneway` traffic also
draws from a smaller per-process async buffer (roughly half the transaction buffer), so a
flood of one-way calls can still hit `TransactionTooLargeException`.

## Security model

Because the kernel stamps every transaction with the caller's identity, the *server* enforces
access at the boundary — this is where Android's permission checks actually live.

```java
public void doPrivileged() {
    int uid = Binder.getCallingUid();   // kernel-guaranteed; cannot be spoofed
    int pid = Binder.getCallingPid();

    // throws SecurityException if the caller lacks the permission
    mContext.enforceCallingPermission(
        android.Manifest.permission.WRITE_SECURE_SETTINGS, "need WRITE_SECURE_SETTINGS");

    // When the service must do work as ITSELF (not the caller), drop the caller identity:
    long token = Binder.clearCallingIdentity();
    try {
        accessSystemOnlyResource();     // now runs under the service's UID
    } finally {
        Binder.restoreCallingIdentity(token);  // ALWAYS restore in finally
    }
}
```

On top of UID checks, **SELinux** governs which domains may even talk to which over Binder —
a transaction needs the `binder { call transfer }` permission between the two SELinux types,
independent of any framework permission. See [SELinux on Android](selinux_android.md) and the
app-facing view in [App Security](app_security.md).

## Reference counting and death recipients

Binder maintains **strong and weak reference counts** on objects across processes, so a server
object stays alive as long as some other process holds a proxy to it, and is reclaimed when
the last reference drops. To learn when the *remote* end disappears, register a
**death recipient**:

```java
IBinder remote = service.asBinder();
IBinder.DeathRecipient recipient = () -> {
    // remote process died — drop caches, unbind, attempt reconnect
    remote.unlinkToDeath(this, 0);
    reconnect();
};
remote.linkToDeath(recipient, 0);
```

This is the standard way a server cleans up per-client state when a client crashes (and vice
versa) without polling. Forgetting to `linkToDeath` (or to `unlinkToDeath`) is a common source
of leaked callbacks and stale state.

## AIDL vs HIDL vs stable AIDL

The app-facing AIDL above is **unstable** — both sides are built together from the same
`.aidl`. Framework-to-vendor boundaries need a *stable*, versioned ABI:

- **HIDL** (HAL Interface Definition Language) was introduced with
  [Project Treble & HALs](treble_hal.md) to decouple the vendor image from the framework. It
  runs over `/dev/hwbinder`. It is now **deprecated**.
- **Stable AIDL** is the modern replacement: the same AIDL language with versioning,
  `@VintfStability`, and frozen interface snapshots, used for new HALs and for
  [Project Mainline & APEX](mainline_apex.md) module boundaries.

The transport is still Binder in every case — only the stability/versioning contract differs.

## Debugging Binder

```bash
adb shell service list                       # registered services + their interfaces
adb shell dumpsys <service>                  # service-specific state (first diagnostic step)
adb shell cat /sys/kernel/debug/binder/stats # per-process transaction & error counters
adb shell cat /sys/kernel/debug/binder/transactions
adb logcat | grep -i TransactionTooLarge     # oversized Parcels
adb logcat | grep -i DeadObjectException     # calling into a dead process
```

`dumpsys` itself is a Binder call — each service implements `dump()` and writes its state over
a passed-in FD, which is why it works uniformly across the whole framework.

## Where this connects

- [Android Internals](internals.md) — Binder is the central IPC mechanism of the whole stack.
- [SystemServer & Core Services](system_server.md) — every system service is reached over Binder
  via `servicemanager`; the Watchdog guards against binder-thread starvation.
- [Zygote & App Startup](zygote_startup.md) — forked app processes join a binder thread pool
  during startup.
- [Project Treble & HALs](treble_hal.md) / [Project Mainline & APEX](mainline_apex.md) —
  HIDL and stable AIDL define versioned Binder interfaces across the framework/vendor boundary.
- [NDK & JNI](ndk_jni.md) — native code uses the same driver through `libbinder`.
- [SELinux on Android](selinux_android.md) / [App Security](app_security.md) — the
  `binder` SELinux class and UID-based permission checks enforced at transactions.
- [Performance & Profiling](performance_profiling.md) — synchronous Binder calls are a common
  source of main-thread jank.

## Pitfalls

- **`TransactionTooLargeException`.** The ~1 MB transaction buffer is **shared across all
  in-flight calls in the process**, not per call — a payload far smaller than 1 MB can still
  fail under concurrency. Pass large data via `ParcelFileDescriptor`/ashmem, not in the Parcel.
- **Blocking the main thread.** A synchronous Binder call waits for the remote process. From
  the UI thread this causes jank or ANRs; do IPC off the main thread.
- **Exhausting the thread pool.** If service binder threads block on a contended lock, callers
  queue and time out. Keep transaction handlers fast and non-blocking.
- **Forgetting `clearCallingIdentity`.** Accessing a system-only resource while still carrying
  the caller's identity throws `SecurityException`; always pair clear/restore in a `finally`.
- **Leaking remote references / skipping `linkToDeath`.** Without death notifications a server
  accumulates dead clients' callbacks and state; without `unlinkToDeath` you leak recipients.
- **Assuming `oneway` ordering.** One-way calls preserve order only per target object, never
  return errors, and draw from a smaller async buffer.
- **Persisting or reusing a Parcel.** Its layout is transaction-specific and may hold live
  handles — marshal fresh each time; never write one to disk.
- **Ignoring `RemoteException`/`DeadObjectException`.** Any cross-process call can fail because
  the peer died; every proxy call needs a failure path.
