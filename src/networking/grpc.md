# gRPC

## Overview

gRPC is a high-performance RPC framework originally developed at Google, open-sourced in 2015. It uses **[HTTP/2](http2.md)** for transport, **Protocol Buffers** (protobuf) for serialization, and supports four call patterns including bidirectional streaming. It's the dominant RPC system for inter-service communication in microservice architectures. [TLS](tls_ssl.md) secures gRPC in production; [WebSocket](websocket.md) is a simpler alternative when browser compatibility matters.

The "g" stands for nothing — Google likes recursive backronyms (officially "gRPC Remote Procedure Calls").

## Why gRPC?

```
Versus REST/JSON:
  ✓ Strongly-typed contracts (.proto schemas)
  ✓ Smaller payloads (binary protobuf)
  ✓ Faster (no JSON parsing, HTTP/2 multiplexing)
  ✓ Streaming built-in
  ✓ Multi-language code generation
  ✗ Not browser-native (need gRPC-Web proxy)
  ✗ Harder to debug (binary, not curl-friendly)

Versus raw protobuf over TCP:
  ✓ Standard wire format (interop)
  ✓ Built-in deadlines, cancellation, metadata
  ✓ Ecosystem (load balancing, auth, observability)

Versus GraphQL:
  ✓ Lower latency, simpler model
  ✗ Less flexible (no on-the-fly field selection)
```

## Architecture Layers

```
┌──────────────────────────────────────────┐
│  Application: stub methods, Server impl  │
├──────────────────────────────────────────┤
│  Generated code (from .proto)            │
├──────────────────────────────────────────┤
│  gRPC core: deadlines, metadata, retries │
├──────────────────────────────────────────┤
│  Protocol Buffers (serialization)        │
├──────────────────────────────────────────┤
│  HTTP/2 (transport)                      │
├──────────────────────────────────────────┤
│  TLS                                     │
├──────────────────────────────────────────┤
│  TCP                                     │
└──────────────────────────────────────────┘
```

## Protocol Buffers (.proto)

Define services and messages in a `.proto` file:

```protobuf
syntax = "proto3";

package user.v1;

option go_package = "example.com/api/user/v1;userv1";

service UserService {
  // Unary
  rpc GetUser(GetUserRequest) returns (User);

  // Server streaming
  rpc ListUsers(ListUsersRequest) returns (stream User);

  // Client streaming
  rpc BatchCreate(stream CreateUserRequest) returns (BatchCreateResponse);

  // Bidirectional streaming
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

message GetUserRequest {
  string user_id = 1;
}

message User {
  string id = 1;
  string email = 2;
  string display_name = 3;
  int64 created_at_unix = 4;
  repeated string roles = 5;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message CreateUserRequest {
  string email = 1;
  string display_name = 2;
}

message BatchCreateResponse {
  int32 created_count = 1;
}

message ChatMessage {
  string from = 1;
  string text = 2;
  int64 ts_unix = 3;
}
```

### Code Generation

```bash
# Install plugins (Go example)
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       user/v1/user.proto

# Output:
#   user/v1/user.pb.go        (messages)
#   user/v1/user_grpc.pb.go   (service stubs)
```

Modern build tooling: **buf** is the go-to (`buf generate`, lint, breaking-change detection).

## The Four Call Types

### 1. Unary (request/response, like REST)

```
Client → 1 message  → Server
Client ← 1 message  ← Server
```

```go
// Go server
func (s *Server) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    user, err := s.db.FindUser(ctx, req.UserId)
    if err != nil {
        return nil, status.Errorf(codes.NotFound, "user not found: %v", err)
    }
    return &pb.User{
        Id:          user.ID,
        Email:       user.Email,
        DisplayName: user.Name,
    }, nil
}

// Go client
resp, err := client.GetUser(ctx, &pb.GetUserRequest{UserId: "u_123"})
if err != nil {
    log.Fatal(err)
}
fmt.Println(resp.Email)
```

### 2. Server Streaming

```
Client → 1 message      → Server
Client ← N messages     ← Server
```

```go
// Server
func (s *Server) ListUsers(req *pb.ListUsersRequest, stream pb.UserService_ListUsersServer) error {
    users, err := s.db.AllUsers(stream.Context())
    if err != nil {
        return err
    }
    for _, u := range users {
        if err := stream.Send(&pb.User{Id: u.ID, Email: u.Email}); err != nil {
            return err
        }
    }
    return nil
}

// Client
stream, err := client.ListUsers(ctx, &pb.ListUsersRequest{PageSize: 100})
for {
    user, err := stream.Recv()
    if err == io.EOF { break }
    if err != nil { log.Fatal(err) }
    fmt.Println(user.Email)
}
```

### 3. Client Streaming

```
Client → N messages → Server
Client ← 1 message  ← Server (after client EOF)
```

```go
// Client
stream, _ := client.BatchCreate(ctx)
for _, req := range requests {
    stream.Send(req)
}
resp, _ := stream.CloseAndRecv()
fmt.Println("created", resp.CreatedCount)
```

### 4. Bidirectional Streaming

```
Client ⇄ N messages ⇄ Server (independent in both directions)
```

```go
stream, _ := client.Chat(ctx)

go func() {
    for { // recv
        msg, err := stream.Recv()
        if err != nil { return }
        fmt.Printf("[%s] %s\n", msg.From, msg.Text)
    }
}()

for { // send
    stream.Send(&pb.ChatMessage{From: "alice", Text: "hello"})
}
```

## How gRPC Maps to HTTP/2

Each gRPC call is one HTTP/2 stream.

### Request

```
:method     POST
:scheme     https
:path       /user.v1.UserService/GetUser
:authority  api.example.com:443
content-type             application/grpc+proto
grpc-encoding            identity        (or gzip, etc.)
grpc-accept-encoding     identity,gzip
te                       trailers
grpc-timeout             5S              (5 seconds)
authorization            Bearer eyJ...
user-agent               grpc-go/1.60.0
```

Body: framed protobuf messages.

### Message Framing on the Wire

```
+--------+----------------+----------------+
| 0x00   | uint32 BE len  | protobuf bytes |
| flag   | (length)       | (payload)      |
+--------+----------------+----------------+

flag bit 0: 1 = compressed, 0 = uncompressed
```

Multiple length-prefixed messages = streaming.

### Response Status

gRPC status comes in **trailers** (HTTP/2 trailing headers), so a streaming response can fail mid-stream:

```
HTTP/2 200                            ← always 200 if call started!
content-type: application/grpc+proto

[message frames ...]

[TRAILERS]
grpc-status: 0                        ← real status here
grpc-message:
```

If the server crashes before sending data:
```
HTTP/2 200
grpc-status: 13                        ← INTERNAL
grpc-message: server failed
```

## Status Codes

gRPC defines its own status codes (NOT HTTP status codes):

| Code | Name | Meaning |
|------|------|---------|
| 0 | OK | Success |
| 1 | CANCELLED | Client cancelled |
| 2 | UNKNOWN | Unspecified error |
| 3 | INVALID_ARGUMENT | Bad input |
| 4 | DEADLINE_EXCEEDED | Timeout |
| 5 | NOT_FOUND | Resource not found |
| 6 | ALREADY_EXISTS | Conflict |
| 7 | PERMISSION_DENIED | Auth'd but not authorized |
| 8 | RESOURCE_EXHAUSTED | Quota / rate limit |
| 9 | FAILED_PRECONDITION | State check failed |
| 10 | ABORTED | Concurrency conflict, retry may help |
| 11 | OUT_OF_RANGE | Iterator past end |
| 12 | UNIMPLEMENTED | Method not implemented |
| 13 | INTERNAL | Server bug |
| 14 | UNAVAILABLE | Service down, retry with backoff |
| 15 | DATA_LOSS | Unrecoverable data corruption |
| 16 | UNAUTHENTICATED | No / bad credentials |

### Returning Status with Details

```go
import (
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

st := status.New(codes.InvalidArgument, "email already taken")
detail := &errdetails.BadRequest{
    FieldViolations: []*errdetails.BadRequest_FieldViolation{
        {Field: "email", Description: "already in use"},
    },
}
stWithDetail, _ := st.WithDetails(detail)
return nil, stWithDetail.Err()
```

## Deadlines & Cancellation

Every call carries a **deadline**, propagated via `grpc-timeout` header. Servers should respect `ctx.Done()`.

```go
// Client side
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

resp, err := client.GetUser(ctx, req)
if status.Code(err) == codes.DeadlineExceeded {
    // handle timeout
}
```

```go
// Server side — always honor ctx
func (s *Server) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    user, err := s.db.QueryContext(ctx, req.UserId)  // db driver should accept ctx
    if err != nil {
        return nil, status.FromContextError(ctx.Err()).Err()
    }
    return user, nil
}
```

### Deadline Propagation

If service A calls service B (gRPC → gRPC), the remaining deadline should propagate. Most clients do this automatically via the incoming `ctx`:

```go
// Service A handler — pass the incoming ctx onward
func (a *A) Handler(ctx context.Context, req *Req) (*Resp, error) {
    return b.Call(ctx, req)   // ctx carries deadline from upstream
}
```

## Metadata (gRPC headers)

Per-call key-value pairs, sent as HTTP/2 headers and trailers.

```go
// Client: outgoing metadata
ctx := metadata.AppendToOutgoingContext(ctx,
    "authorization", "Bearer "+token,
    "x-request-id", reqID,
)
resp, _ := client.GetUser(ctx, req)

// Server: read incoming metadata
md, ok := metadata.FromIncomingContext(ctx)
if ok {
    auths := md.Get("authorization")
}

// Server: send headers/trailers
header := metadata.Pairs("x-server-id", "node-7")
grpc.SendHeader(ctx, header)
```

Binary metadata keys end in `-bin`; values are base64-encoded on the wire.

## Interceptors (middleware)

Cross-cutting concerns (auth, logging, retries, tracing) live in interceptors.

```go
// Unary server interceptor
func loggingUnary(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
    start := time.Now()
    resp, err := handler(ctx, req)
    log.Printf("method=%s dur=%s err=%v", info.FullMethod, time.Since(start), err)
    return resp, err
}

server := grpc.NewServer(grpc.UnaryInterceptor(loggingUnary))
```

```go
// Stream interceptors are similar (wrap the stream)
grpc.StreamInterceptor(streamingMiddleware)
```

Use `grpc-middleware` package for chains, validators, auth, recovery, OpenTelemetry, etc.

## Authentication

### TLS (transport security)

```go
creds, _ := credentials.NewClientTLSFromFile("ca.pem", "")
conn, _ := grpc.NewClient("api.example.com:443", grpc.WithTransportCredentials(creds))
```

### mTLS

```go
// Server validates client certificate
tlsCfg := &tls.Config{
    ClientAuth: tls.RequireAndVerifyClientCert,
    ClientCAs:  certPool,
    Certificates: []tls.Certificate{serverCert},
}
server := grpc.NewServer(grpc.Creds(credentials.NewTLS(tlsCfg)))
```

### Per-Call Token (bearer / JWT)

```go
type tokenAuth struct{ token string }
func (t tokenAuth) GetRequestMetadata(_ context.Context, _ ...string) (map[string]string, error) {
    return map[string]string{"authorization": "Bearer " + t.token}, nil
}
func (t tokenAuth) RequireTransportSecurity() bool { return true }

conn, _ := grpc.NewClient(addr,
    grpc.WithTransportCredentials(creds),
    grpc.WithPerRPCCredentials(tokenAuth{token: jwt}),
)
```

## Load Balancing

```
1. Proxy-based (envoy, Linkerd, Nginx)
   Client → LB → many backends
   Pros: dumb client
   Cons: extra hop, single bottleneck

2. Client-side (gRPC's preferred model)
   Client resolves multiple backend addresses
   Picks one per RPC (round-robin, weighted, etc.)
   Uses Name Resolver + Balancer plugins
```

```go
// Client-side round-robin via DNS
conn, _ := grpc.NewClient(
    "dns:///api.example.com:50051",
    grpc.WithDefaultServiceConfig(`{"loadBalancingConfig":[{"round_robin":{}}]}`),
    grpc.WithTransportCredentials(creds),
)
```

xDS (Envoy's API) is the modern dynamic LB protocol — gRPC clients can directly consume it without a sidecar.

## Reflection & Tooling

### Server Reflection

Server advertises its service definitions at runtime:

```go
import "google.golang.org/grpc/reflection"
reflection.Register(server)
```

Then clients can discover without `.proto` files:

```bash
# grpcurl (the gRPC equivalent of curl)
grpcurl -plaintext localhost:50051 list
grpcurl -plaintext localhost:50051 list user.v1.UserService
grpcurl -plaintext localhost:50051 describe user.v1.User

# Invoke
grpcurl -plaintext -d '{"user_id":"u_123"}' \
  localhost:50051 user.v1.UserService/GetUser

# TLS
grpcurl -d '{"user_id":"u_123"}' api.example.com:443 user.v1.UserService/GetUser
```

### Health Checking

Standard health-check protocol (`grpc.health.v1.Health`):

```go
import healthpb "google.golang.org/grpc/health/grpc_health_v1"
import "google.golang.org/grpc/health"

healthServer := health.NewServer()
healthpb.RegisterHealthServer(server, healthServer)
healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
```

Used by Kubernetes via `grpc_health_probe` or native gRPC probes.

## Compression

Configure per call or per server. `gzip` is standard.

```go
// Client side
resp, _ := client.GetUser(ctx, req, grpc.UseCompressor(gzip.Name))

// Server side: compression applied if accept-encoding allows
```

Default = no compression. Worth enabling for large payloads, not for small ones (CPU > bandwidth savings).

## gRPC-Web

Browsers can't speak HTTP/2 trailers cleanly, so **gRPC-Web** is a slightly different framing that translates via a proxy:

```
Browser ──HTTP/1.1 or HTTP/2─→ Envoy ──gRPC─→ Backend
        (gRPC-Web)                   (regular gRPC)
```

Limitations: no client-streaming, no bidi-streaming (only unary + server-streaming).

Alternative: **Connect** (Buf's protocol) — HTTP/1.1-compatible, JSON or protobuf, works directly from browsers without a proxy.

## Common Pitfalls

### Resending Idempotent Requests

gRPC has built-in retry policy via service config:
```json
{
  "methodConfig": [{
    "name": [{"service": "user.v1.UserService"}],
    "retryPolicy": {
      "maxAttempts": 4,
      "initialBackoff": "0.1s",
      "maxBackoff": "1s",
      "backoffMultiplier": 2,
      "retryableStatusCodes": ["UNAVAILABLE"]
    }
  }]
}
```

Only retry idempotent methods. POST-like creates need idempotency keys.

### Long-lived Streams Pin Connections

Streams keep an HTTP/2 stream open. If you have thousands of long-lived subscriptions, watch SETTINGS_MAX_CONCURRENT_STREAMS and increase as needed.

### Proto Breaking Changes

```
Safe:
  - Adding new optional fields with new numbers
  - Adding new RPCs
  - Adding new enum values

Breaking:
  - Reusing or removing field numbers
  - Changing field types
  - Renaming services/packages
  - Changing message field cardinality

Tool: `buf breaking` catches violations in CI.
```

### Wire Format vs JSON

Two serializations exist: binary protobuf (`application/grpc+proto`) and protobuf-JSON (`application/grpc+json`). Most clients/servers default to binary; debugging often easier with JSON.

### Status Codes ≠ HTTP Status

A gRPC call that returns HTTP 200 may still be an error. Always check `grpc-status` trailer or library error.

## gRPC vs Alternatives

| Aspect | REST/JSON | gRPC | GraphQL | Connect |
|--------|-----------|------|---------|---------|
| Transport | HTTP/1.1+ | HTTP/2 | HTTP/1.1+ | HTTP/1.1+ |
| Encoding | JSON | Protobuf | JSON | Protobuf/JSON |
| Schema | OpenAPI (optional) | Required (.proto) | Required (SDL) | Required (.proto) |
| Streaming | SSE / WebSocket | Built-in (all 4 kinds) | Subscriptions | Built-in |
| Browser | Native | Needs proxy | Native | Native |
| Tooling | Mature | Mature | Mature | Newer |
| Best for | Public APIs | Internal microservices | Client-driven querying | Both browser + backend |

## Performance Tips

```
✓ Reuse client connections (gRPC handles multiplexing)
✓ Use streaming for high-throughput data (no per-call HTTP/2 overhead)
✓ Enable compression for >1KB payloads
✓ Set sensible deadlines (always; never infinite for user-facing)
✓ Use client-side load balancing for low-latency
✓ Tune flow control windows for high-bandwidth-delay paths
✓ Profile protobuf encoding (sometimes the bottleneck)
✓ Use grpc.SharedWriteBuffer for high-throughput servers
```

## Debugging

```bash
# Verbose logs
export GRPC_GO_LOG_VERBOSITY_LEVEL=99
export GRPC_GO_LOG_SEVERITY_LEVEL=info

# Trace
import _ "net/http/pprof"
http://localhost:6060/debug/grpc

# Wireshark / tcpdump (only useful without TLS or with SSLKEYLOGFILE)
ssldump -A -nq -k key.pem -i any port 50051
```

## ELI10

gRPC is like sending a robot to your friend's house with a typed-up shopping list:

- The **shopping list** is a `.proto` file. Both you and your friend agreed on the format ahead of time, so there's no confusion.
- The **robot** is the gRPC stub — generated automatically. You don't write the delivery code, you just hand it a list.
- **Streaming** is like sending a robot that can both bring items and take requests back, all in one trip.
- **Deadlines** are "be back by dinner, or come back empty-handed" so robots don't get lost forever.
- **Metadata** is sticky notes on the shopping bag (auth tokens, request IDs) that don't change the items but tell the other side useful context.
- **Interceptors** are the supervisor robot who logs every trip and checks IDs before sending the real robot.

The reason gRPC is faster than JSON-over-HTTP is that the shopping list is in compact code numbers instead of long English sentences, and the trucks (HTTP/2) can carry many robots at once.

## Further Resources

- [grpc.io official docs](https://grpc.io/docs/)
- [protobuf Language Guide](https://protobuf.dev/programming-guides/proto3/)
- [Google API Design Guide](https://cloud.google.com/apis/design)
- [buf — modern protobuf tooling](https://buf.build/)
## Where this connects

- [HTTP/2](http2.md) — gRPC is built on HTTP/2 streams for multiplexed RPC calls
- [TLS/SSL](tls_ssl.md) — TLS is required for gRPC in production (mTLS for service-to-service)
- [TCP](tcp.md) — the underlying transport HTTP/2 rides
- [WebSocket](websocket.md) — simpler alternative when bidirectional streaming over HTTP/1.1 suffices

- [Connect — gRPC-compatible alternative](https://connectrpc.com/)
- [gRPC + Go Tutorial](https://grpc.io/docs/languages/go/quickstart/)
