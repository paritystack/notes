# Logging & Log Aggregation

## Overview

Logs are the timestamped, event-by-event record of what a system did — the first thing you
reach for when debugging an incident. This note focuses on the **operational logging
pipeline**: how to emit good logs, ship them off the machine, aggregate them centrally, and
search them at scale. It's the "logs" pillar of [observability](observability.md) (which
covers metrics and traces too, and the conceptual three-pillars model) made concrete, and
it leans on [Elasticsearch](../databases/elasticsearch.md) as the dominant search backend.

```
Why centralize? In a distributed system a single request touches many services on many
hosts (that scale up/down). Logs on individual machines are useless when:
  - the container is already gone   - you must correlate across 10 services
  - you need to search millions of lines   - you want alerts on patterns
⇒ ship every log to ONE searchable place.
```

## Structured logging

The single highest-leverage practice: emit logs as **structured key-value data (JSON)**, not
free-text prose, so machines can parse, filter, and aggregate them.

```
✗ Unstructured:  "User 123 failed login from 1.2.3.4 at 10:05"
✓ Structured:    {"ts":"...","level":"warn","event":"login_failed",
                  "user_id":123,"ip":"1.2.3.4","trace_id":"abc"}

Now you can: filter level=warn, group by event, count by ip, JOIN to a trace — none of
which is possible by grepping prose.
```

```
Always include: timestamp, level, service name, and a CORRELATION ID (trace/request id)
so a single request can be followed across every service. See observability.md for trace
propagation. NEVER log secrets, passwords, tokens, or PII — see ../security/secrets_management.md.
```

## Log levels

```
ERROR   something failed and needs attention (exceptions, failed requests)
WARN    unexpected but handled (retries, fallbacks, deprecations)
INFO    normal significant events (startup, request completed, state change)
DEBUG   detailed diagnostic info (off in prod, or sampled)
TRACE   very fine-grained (rarely on in prod)

Set the level per environment; too verbose = cost + noise, too quiet = blind in an incident.
```

## The aggregation pipeline

Logs flow through four stages from app to query:

```
  [app emits] → [collect/ship] → [parse/enrich] → [store/index] → [visualize/alert]
       JSON       agent on host    add fields,      searchable      dashboards,
       to stdout  or sidecar       drop noise       store           alerts
```

```
1. Emit        log JSON to stdout/stderr (12-factor) — let the platform handle the rest.
2. Collect     a node agent tails container logs:
                 Fluentd / Fluent Bit (CNCF), Vector, Logstash, Filebeat, Promtail.
3. Parse/enrich  add k8s metadata (pod, namespace), parse fields, drop/sample noisy logs,
                 redact sensitive data.
4. Store/index   Elasticsearch/OpenSearch (full-text), Loki (label-indexed, cheaper),
                 or a managed service (Datadog, CloudWatch, Splunk).
5. Query/visualize  Kibana (ES), Grafana (Loki), or the vendor UI.
```

## The two stack archetypes

```
ELK / Elastic Stack   Elasticsearch + Logstash/Beats + Kibana.
                      Full-text indexing of log CONTENT → powerful search, higher
                      storage/compute cost. See ../databases/elasticsearch.md.
Grafana Loki          indexes only LABELS (service, level), stores log bodies compressed.
                      Much cheaper at scale; "grep-like" queries (LogQL) rather than
                      full inverted-index search. Pairs with Grafana + Prometheus.
Choose: rich search & analytics → ELK;  cost-efficient, label-scoped → Loki.
```

## Cost control

Logging volume (and bill) explodes silently — manage it deliberately:

```
Sampling       keep all ERRORs, sample high-volume INFO/DEBUG.
Filtering      drop health-check/noise logs at the collector, before storage.
Retention/tiering  hot (searchable, days) → warm → cold/archive (S3, cheap) → delete.
                   Index lifecycle management (ILM) automates this. See elasticsearch.md.
Cardinality    avoid high-cardinality labels in Loki (they explode the index) — keep
               unique IDs in the log BODY, not in labels.
```

## Logs vs metrics vs traces

Use the right pillar — logs are not the answer to everything (see [observability](observability.md)):

```
Metrics  aggregated numbers over time ("error rate is 5%")        → alerting, trends, cheap
Logs     discrete events with context ("WHY did request X fail")   → debugging specifics
Traces   one request's path across services ("WHERE is the latency")→ distributed debugging
Correlate them with a shared trace_id to jump metric → trace → log.
```

## Where this connects

- **[Observability](observability.md)** — the three-pillars model; logs are one pillar.
- **[Monitoring](monitoring.md)** — metrics & Prometheus; alerting on log-derived signals.
- **[Elasticsearch](../databases/elasticsearch.md)** — the search backend behind ELK.
- **[SRE](sre.md)** — logs as evidence during incident response.
- **[Secrets management](../security/secrets_management.md)** — never log secrets/PII.

## Pitfalls

- **Unstructured logs** — prose you can't filter or aggregate; emit JSON.
- **Logging secrets/PII** — redact at the source/collector; a top compliance & leak risk.
- **No correlation ID** — impossible to follow one request across services.
- **Unbounded volume/retention** — the logging bill quietly dwarfs compute; sample, filter, tier.
- **High-cardinality Loki labels** — explodes the index; keep IDs in the body.
- **Using logs for what metrics do** — counting events by scanning logs is slow and costly;
  emit a metric.
