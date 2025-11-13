# gRPC

gRPC is a high-performance, open-source universal RPC framework. It uses HTTP/2 for transport, Protocol Buffers as the interface description language, and provides features like authentication, load balancing, and more.

## Overview

**Key Features:**
- HTTP/2 based transport
- Protocol Buffers for serialization
- Bidirectional streaming
- Pluggable auth, tracing, load balancing
- Language-agnostic

## Protocol Buffers

```protobuf
// user.proto
syntax = "proto3";

package user;

service UserService {
  rpc GetUser(UserRequest) returns (UserResponse);
  rpc ListUsers(ListUsersRequest) returns (stream UserResponse);
}

message UserRequest {
  int32 id = 1;
}

message UserResponse {
  int32 id = 1;
  string name = 2;
  string email = 3;
}

message ListUsersRequest {
  int32 page = 1;
  int32 page_size = 2;
}
```

## Server Implementation (Python)

```python
import grpc
from concurrent import futures
import user_pb2
import user_pb2_grpc

class UserServiceServicer(user_pb2_grpc.UserServiceServicer):
    def GetUser(self, request, context):
        # Fetch user from database
        return user_pb2.UserResponse(
            id=request.id,
            name="John Doe",
            email="john@example.com"
        )

    def ListUsers(self, request, context):
        # Stream users
        for user in get_users():
            yield user_pb2.UserResponse(
                id=user.id,
                name=user.name,
                email=user.email
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    user_pb2_grpc.add_UserServiceServicer_to_server(
        UserServiceServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

## Client Implementation

```python
import grpc
import user_pb2
import user_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = user_pb2_grpc.UserServiceStub(channel)

        # Unary call
        response = stub.GetUser(user_pb2.UserRequest(id=1))
        print(f"User: {response.name}")

        # Server streaming
        for user in stub.ListUsers(user_pb2.ListUsersRequest(page=1)):
            print(f"User: {user.name}")
```

## Stream Types

| Type | Description |
|------|-------------|
| Unary | Single request/response |
| Server streaming | Client sends one request, server streams responses |
| Client streaming | Client streams requests, server sends one response |
| Bidirectional | Both stream |

gRPC provides efficient, type-safe communication between services, ideal for microservices architectures.
