# Serialization

## Overview

Serialization turns in-memory data structures into a byte stream that can be stored or sent,
and deserialization reconstructs them on the other side. It is the lingua franca between
programs, services, and storage — every API call, message queue, cache entry, and save file
relies on it. It cuts across [type systems](type_systems.md) (mapping types to wire formats)
and [metaprogramming](metaprogramming.md) (reflection/codegen generate the (de)serializers),
and it underpins much of the [networking](../networking/grpc.md) and
[databases](../databases/index.html) sections.

## Text vs binary formats

```
TEXT (JSON, XML, YAML, TOML)            BINARY (Protobuf, MessagePack, Avro, CBOR)
----------------------------            ------------------------------------------
human-readable, debuggable              compact, fast to parse
self-describing                         needs a schema (Protobuf/Avro) or is typed (MsgPack)
larger, slower                          smaller, faster
ubiquitous (web APIs, config)           RPC, high-throughput pipelines, storage
```

JSON dominates web APIs for its readability and universal support; binary formats win when
size and speed matter (internal RPC via [gRPC](../web_development/grpc.md), event streams via
[Kafka](../databases/kafka.md)).

## Schema vs schemaless

- **Schemaless** (JSON, MessagePack) — structure is implicit in the data; flexible, but both
  sides must agree by convention and validate at the boundary.
- **Schema-driven** (Protobuf, Avro, Thrift) — a declared schema generates typed code and
  enforces structure. Self-documenting, smaller (field names aren't repeated), and enables
  safe evolution.

```protobuf
message User {
  int64  id    = 1;   // field numbers are the contract, not the names
  string name  = 2;
  string email = 3;
}
```

## Schema evolution

The hard part of serialization isn't a single encode/decode — it's letting producers and
consumers change *independently over time*. Good formats support **backward** (new code reads
old data) and **forward** (old code reads new data) compatibility.

```
Safe:    add an optional field, add an enum value (with a default/unknown case)
Unsafe:  remove/reuse a field number, change a type, rename a required field
Rule:    never reuse a tag/field number; deprecate, don't delete.
```

Avro pairs each payload with (or references) the writer's schema; Protobuf relies on stable
field numbers and optional fields. This is what lets you deploy services on rolling schedules
without coordinated big-bang upgrades.

## Performance and safety notes

Binary formats can be **zero-copy** (FlatBuffers, Cap'n Proto) — read fields directly out of
the buffer without parsing into objects, valuable in latency-critical paths. On the safety
side, deserializing untrusted input is a classic vulnerability class (see
[web security](../web_development/web_security.md)): language-native serializers (Java
`Serializable`, Python `pickle`) can execute code or instantiate arbitrary types — never
deserialize untrusted data with them.

## Where this connects

- [Metaprogramming](metaprogramming.md) — reflection/codegen generate (de)serializers from types.
- [Type systems](type_systems.md) — mapping language types to wire types, handling nullability.
- [gRPC](../web_development/grpc.md) / [Kafka](../databases/kafka.md) — Protobuf/Avro in practice.
- [API design](../web_development/api_design.md) — JSON contracts and versioning.
- [Web security](../web_development/web_security.md) — insecure deserialization.

## Pitfalls

- **Insecure deserialization.** `pickle`/Java `Serializable`/`yaml.load` on untrusted input can
  run arbitrary code. Use data-only formats and explicit schemas.
- **Reusing field numbers/tags.** Breaks every old client silently; always deprecate instead.
- **Floating-point and integer precision.** JSON numbers are doubles — large int64 IDs lose
  precision; send them as strings.
- **Timezones, encodings, and `null` vs absent.** Ambiguous date formats, non-UTF-8 bytes, and
  conflating "missing" with "null" cause cross-language mismatches.
- **Unbounded input.** No size/depth limits invites memory-exhaustion DoS; cap during parse.
- **Assuming round-trip identity.** Map ordering, number formatting, and unknown fields may not
  survive a decode/encode cycle.
