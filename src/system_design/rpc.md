# RPC (Remote Procedure Call)

RPC is a protocol that allows a program to execute a procedure on another computer as if it were a local procedure call.

## Overview

RPC abstracts network communication, making distributed computing appear like local function calls.

**Key Concepts:**
- Client-Server model
- Stub generation
- Marshalling/Unmarshalling
- Synchronous or asynchronous calls

## Common RPC Frameworks

| Framework | Protocol | Language |
|-----------|----------|----------|
| gRPC | HTTP/2, Protobuf | Multi-language |
| JSON-RPC | HTTP, JSON | Multi-language |
| XML-RPC | HTTP, XML | Multi-language |
| Apache Thrift | Binary | Multi-language |

## gRPC Example

```protobuf
// service.proto
service Calculator {
  rpc Add(Numbers) returns (Result);
}

message Numbers {
  int32 a = 1;
  int32 b = 2;
}

message Result {
  int32 value = 1;
}
```

## JSON-RPC Example

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "add",
  "params": {"a": 5, "b": 3},
  "id": 1
}

// Response
{
  "jsonrpc": "2.0",
  "result": 8,
  "id": 1
}
```

## Advantages

- Simple interface (like local calls)
- Language-agnostic
- Abstraction of network details
- Type safety (with IDL)

## Challenges

- Network failures
- Latency
- Versioning
- Error handling complexity

RPC simplifies distributed system development by providing procedure call semantics over network communication.
