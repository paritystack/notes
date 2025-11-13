# Android Binder

Binder is Android's inter-process communication (IPC) mechanism. It's a custom implementation allowing processes to communicate efficiently and securely.

## Overview

Binder enables:
- Cross-process method invocation
- Object reference passing
- Security via UID/PID checking
- Death notification

## Architecture

```
Client Process          Binder Driver         Server Process
    │                        │                       │
    │──Service Request──────>│                       │
    │                        │──Forward Request────>│
    │                        │<──Response───────────│
    │<──Return Result────────│                       │
```

## AIDL (Android Interface Definition Language)

```java
// ICalculator.aidl
package com.example;

interface ICalculator {
    int add(int a, int b);
    int subtract(int a, int b);
}
```

## Service Implementation

```java
// CalculatorService.java
public class CalculatorService extends Service {
    private final ICalculator.Stub binder = new ICalculator.Stub() {
        @Override
        public int add(int a, int b) {
            return a + b;
        }

        @Override
        public int subtract(int a, int b) {
            return a - b;
        }
    };

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }
}
```

## Client Usage

```java
// Client code
ServiceConnection connection = new ServiceConnection() {
    public void onServiceConnected(ComponentName name, IBinder service) {
        ICalculator calculator = ICalculator.Stub.asInterface(service);
        int result = calculator.add(5, 3);  // Result: 8
    }

    public void onServiceDisconnected(ComponentName name) {
        // Handle disconnection
    }
};

bindService(intent, connection, Context.BIND_AUTO_CREATE);
```

## Key Features

- **Security**: Permission checking at IPC boundaries
- **Reference Counting**: Automatic resource management  
- **Death Recipients**: Notification when remote process dies
- **Asynchronous**: Non-blocking calls with oneway keyword

Binder is fundamental to Android's architecture, enabling system services and app communication.
