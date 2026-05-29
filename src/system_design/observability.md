# Observability

You can't operate what you can't see. Observability is the discipline of inferring internal system state from external signals — the **three pillars** are logs, metrics, and traces.

## The Three Pillars

| Pillar | Question it answers | Cost per event | Cardinality |
|---|---|---|---|
| **Logs** | What happened? (precise event) | High (KB) | Unlimited |
| **Metrics** | How much / how often? | Cheap (bytes) | Bounded |
| **Traces** | How did a request flow? | Medium | Sampled |

```
   Request enters ──► Service A ──► Service B ──► DB
                         │              │           │
                Logs:    │              │           │
                Metrics: │ rps/lat/err  │ rps/...   │ qps/...
                Traces:  └──── span ───►└── span ──►└── span ──►
```

## Logs

Discrete event records, timestamped, contextualized.

### Structured > unstructured
```json
{
  "ts": "2026-05-26T10:00:00Z",
  "level": "ERROR",
  "service": "checkout",
  "trace_id": "abc123",
  "user_id": "u_456",
  "order_id": "o_789",
  "msg": "payment declined",
  "code": "CARD_DECLINED",
  "elapsed_ms": 245
}
```

Structured = machine-queryable. `printf("user %s failed payment", u)` is dead on arrival in a real ops scenario.

### Log levels
| Level | Use |
|---|---|
| `TRACE` | Per-line debugging. Off in prod. |
| `DEBUG` | Detailed flow. Off in prod default. |
| `INFO` | Significant events. Always on. |
| `WARN` | Recovered/unexpected condition. |
| `ERROR` | Unrecovered, user-impacting. |
| `FATAL` | Process is dying. |

### Sampling logs
At scale, INFO logs explode. Mitigations:
- **Sample by trace_id**: keep 1% of traces' INFO; keep 100% of errors.
- **Dynamic sampling**: high-rate boring events sampled; rare events kept.
- **Drop at agent**: filter before shipping (Fluent Bit, Vector).

### Stack
```
App ──► Stdout ──► Agent (Fluent Bit/Vector) ──► Buffer (Kafka)
                                                   │
                                                   ▼
                                          Storage (Loki, Elastic, S3+Athena)
                                                   │
                                                   ▼
                                                 Search UI (Grafana/Kibana)
```

## Metrics

Aggregate numerical measurements, low cardinality, time-series.

### Types (Prometheus / OpenMetrics)
| Type | Description | Example |
|---|---|---|
| **Counter** | Monotonic, ever-increasing | `http_requests_total` |
| **Gauge** | Snapshot, can go up/down | `mem_used_bytes` |
| **Histogram** | Buckets for distributions | `http_latency_seconds_bucket{le=0.1}` |
| **Summary** | Pre-computed quantiles | `http_latency_seconds{quantile=0.99}` |

### Histogram vs Summary
- **Histogram**: bucket counts shipped, percentiles computed in query. ✅ Can aggregate across instances. Choose this.
- **Summary**: percentiles computed per-instance, shipped pre-baked. ❌ Cannot meaningfully aggregate (you can't average p99s).

### Cardinality is the killer
```
http_requests_total{method, status, path}
   methods: 5
   statuses: 6
   paths: 100   → 3000 time-series. Fine.

http_requests_total{method, status, path, user_id}
   + 10M user_ids → 30B time-series. System dies.
```

**Rule:** never use unbounded identifiers (user_id, request_id, IP) as metric labels. Those belong in logs/traces.

### Methods for choosing metrics

**RED** (request services):
- **R**ate — requests per second
- **E**rrors — errors per second (or %)
- **D**uration — latency distribution (p50, p95, p99)

**USE** (resources):
- **U**tilization — % busy
- **S**aturation — queue length / how overloaded
- **E**rrors — error count

**Golden signals** (Google SRE): latency, traffic, errors, saturation.

### Stack
```
App ──exposes──► /metrics HTTP endpoint
                       │ scrape every 15s
                       ▼
                 Prometheus (TSDB)
                       │
                       ▼
                  Grafana (viz)
                       │
                       ▼
                Alertmanager (alerts)
```

For long-term: Mimir / Cortex / Thanos / VictoriaMetrics / Datadog / Honeycomb.

## Traces

Records of how a single request flows through a distributed system.

### Anatomy
```
Trace (one request, has trace_id)
├── Root span: API gateway (200ms)
│   ├── Span: auth service (5ms)
│   ├── Span: order service (150ms)
│   │   ├── Span: inventory check (30ms)
│   │   └── Span: pricing (110ms)
│   │       └── Span: cache lookup (2ms) [miss]
│   │       └── Span: pricing DB (105ms)  ← culprit
│   └── Span: notification (40ms)
```

Each span has: `trace_id`, `span_id`, `parent_span_id`, name, start, duration, attributes, status.

### Context propagation
The crucial bit: trace context (W3C `traceparent` header) travels with every request hop. Without propagation, you get disconnected fragments.

```
traceparent: 00-{trace_id}-{span_id}-{flags}
```

Library auto-instrumentation injects/extracts these on HTTP, gRPC, message queues.

### Sampling
Tracing 100% of requests is expensive. Strategies:
| Strategy | Description |
|---|---|
| **Head sampling** | Decide at request start (e.g., 1%). Cheap, biased. |
| **Tail sampling** | Buffer, then decide based on outcome (always keep errors, slow). Best signal. |
| **Adaptive** | Sample more from rare endpoints. |
| **Per-user** | Sample by user_id hash for consistency. |

### Stack: OpenTelemetry (OTel)

```
App + OTel SDK ──OTLP──► OTel Collector ──► Backend
                            │
                            ├──► Tempo / Jaeger (traces)
                            ├──► Prometheus (metrics)
                            └──► Loki (logs)
```

**OpenTelemetry** is the vendor-neutral standard. Instrument once, export to any backend. Replaces OpenTracing + OpenCensus.

## Correlation Across Pillars

The trace_id is the unifier. A modern observability stack lets you:
- Click an error log → jump to the trace.
- Click a slow span → see logs from that span.
- Click a spiking metric → drill to exemplar traces.

Bake `trace_id` into every log line. It's the single most valuable field.

## Alerting

### Symptom vs cause
- **Symptom alerts** (good): "checkout error rate > 1%". User-impact-tied. Page on these.
- **Cause alerts** (sometimes good): "DB CPU > 90%". Often noisy if no symptom follows.

### SLO-based alerting
Define an SLO (e.g., "99.9% of checkout requests succeed in <300ms"). Track **error budget burn rate**. Page on:
- **Fast burn:** 2% budget burned in 1 hour → page immediately.
- **Slow burn:** 10% burned in 3 days → ticket.

### Alert hygiene
- Every page must be actionable. If you can't act, it's a ticket, not a page.
- Runbook link in every alert.
- Suppress during deploys.
- Multi-window, multi-burn-rate alerts (Google SRE workbook).

## Profiles (Sometimes a 4th Pillar)

Continuous profiling: capture CPU / heap / allocations from running services. Pyroscope, Parca, Datadog Profiler.

Useful when:
- Metrics show high CPU, you need to know **which function**.
- Diagnosing memory leaks without lab repro.
- Optimizing perf-critical paths.

## eBPF: Observability Without Instrumentation

Kernel-level probes that observe syscalls, network, CPU, no app changes. Cilium/Hubble, Pixie, Inspektor.

Use when you want:
- Network flow visibility without sidecars.
- Coverage of services you can't instrument (legacy, closed-source).
- Kernel-level perf data.

## Logs vs Metrics vs Traces: Decision Matrix

| Question | Use |
|---|---|
| What's the error rate right now? | Metric |
| What's the p99 latency trend? | Metric |
| What did this specific failed request do? | Trace |
| What was the exact error message? | Log |
| Did a specific user hit this bug? | Log (with user_id) |
| Which downstream service is slowest? | Trace |
| Is the box swapping? | Metric (USE) |

## Cost Realities at Scale

Logs are expensive — easily $5–50/GB ingested at vendor pricing. Mitigate:
- **Sample aggressively**. 1% INFO is plenty.
- **Tier storage**: hot 7d, warm 30d, cold S3+Athena beyond.
- **Pre-aggregate**: convert frequent log events to metrics.
- **Cardinality discipline** on metrics (no user_id labels).
- **Self-host** if scale > vendor pricing inflection (~$50K/mo).

## Interview Cheat: Observability in a Design

If the interview asks "how would you monitor/operate this?", hit:
1. Golden signals per service (RED).
2. Trace_id propagated through every hop.
3. Structured logs with trace_id.
4. SLOs + error-budget burn alerts.
5. Cardinality discipline.

## Common Pitfalls

- **Logging everything**: storage explodes, signal drowns in noise.
- **High-cardinality metrics**: kills Prometheus, breaks dashboards.
- **No trace context propagation**: traces are fragmented & useless.
- **Symptom blindness**: alerts on CPU but not user-visible failures.
- **Vendor lock-in**: don't ship in proprietary formats; use OTel.

## Related

- `design_patterns.md#observability` — observability patterns at architecture level
- `microservices.md` — why observability matters more in microservices
- `distributed_systems.md` — debugging distributed bugs starts with traces
