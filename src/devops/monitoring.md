# Monitoring & Observability

Comprehensive monitoring, logging, and observability practices for maintaining reliable, performant systems.

## Core Concepts

### The Three Pillars of Observability

#### 1. Metrics
Numerical data points measured over time:
- **System Metrics**: CPU, memory, disk, network
- **Application Metrics**: Request rate, error rate, latency
- **Business Metrics**: User signups, transactions, revenue

#### 2. Logs
Discrete event records with timestamps:
- **Application Logs**: Debug, info, warning, error messages
- **Access Logs**: HTTP requests, API calls
- **Audit Logs**: Security events, user actions

#### 3. Traces
Request flow through distributed systems:
- **Distributed Tracing**: Track requests across services
- **Span**: Single operation in trace
- **Context Propagation**: Pass trace context between services

### Monitoring vs Observability

**Monitoring**: Known unknowns - track predefined metrics
- "Is the system up?"
- "What's the error rate?"

**Observability**: Unknown unknowns - understand system behavior
- "Why is this request slow?"
- "What caused this error?"

## Key Metrics

### Golden Signals (SRE)

#### Latency
Time to serve requests:
```promql
# P95 latency
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m])
)

# Average latency
rate(http_request_duration_seconds_sum[5m]) /
rate(http_request_duration_seconds_count[5m])
```

#### Traffic
Demand on system:
```promql
# Requests per second
rate(http_requests_total[5m])

# Request volume by endpoint
sum by (endpoint) (rate(http_requests_total[5m]))
```

#### Errors
Failed requests:
```promql
# Error rate
rate(http_requests_total{status=~"5.."}[5m]) /
rate(http_requests_total[5m])

# Error count
sum(rate(http_requests_total{status=~"5.."}[5m]))
```

#### Saturation
System utilization:
```promql
# CPU usage
100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) /
node_memory_MemTotal_bytes * 100

# Disk usage
(node_filesystem_size_bytes - node_filesystem_avail_bytes) /
node_filesystem_size_bytes * 100
```

### RED Method (Services)

**Rate**: Requests per second
**Errors**: Failed requests per second
**Duration**: Request latency

### USE Method (Resources)

**Utilization**: % time resource is busy
**Saturation**: Amount of queued work
**Errors**: Error count

## Prometheus

### Architecture
```
Applications → Prometheus → Grafana
     ↓
  Exporters
```

### Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    region: 'us-east-1'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node1:9100', 'node2:9100']

  - job_name: 'application'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - https://example.com
          - https://api.example.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
```

### Metrics Instrumentation

#### Node.js (prom-client)
```javascript
const client = require('prom-client');

// Default metrics (CPU, memory, etc.)
client.collectDefaultMetrics();

// Counter
const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['method', 'route', 'status']
});

// Histogram
const httpRequestDuration = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request latency',
  labelNames: ['method', 'route', 'status'],
  buckets: [0.1, 0.5, 1, 2, 5]
});

// Gauge
const activeConnections = new client.Gauge({
  name: 'active_connections',
  help: 'Active connections'
});

// Middleware
app.use((req, res, next) => {
  const start = Date.now();

  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestsTotal.inc({
      method: req.method,
      route: req.route?.path || req.path,
      status: res.statusCode
    });
    httpRequestDuration.observe({
      method: req.method,
      route: req.route?.path || req.path,
      status: res.statusCode
    }, duration);
  });

  next();
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', client.register.contentType);
  res.end(await client.register.metrics());
});
```

#### Python (prometheus_client)
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

active_users = Gauge(
    'active_users',
    'Currently active users'
)

# Decorator
def track_metrics(func):
    def wrapper(*args, **kwargs):
        method = request.method
        endpoint = request.endpoint

        with request_duration.labels(method, endpoint).time():
            response = func(*args, **kwargs)

        requests_total.labels(
            method,
            endpoint,
            response.status_code
        ).inc()

        return response
    return wrapper

# Start metrics server
start_http_server(8000)
```

### PromQL Queries

#### Rate and Increase
```promql
# Requests per second
rate(http_requests_total[5m])

# Total requests in 5 minutes
increase(http_requests_total[5m])

# Delta for gauges
delta(cpu_temperature_celsius[1h])
```

#### Aggregation
```promql
# Sum across all instances
sum(rate(http_requests_total[5m]))

# Average by instance
avg by(instance) (rate(http_requests_total[5m]))

# Top 5 endpoints
topk(5, sum by(endpoint) (rate(http_requests_total[5m])))

# Count of targets
count(up == 1)
```

#### Functions
```promql
# Percentiles
histogram_quantile(0.99,
  rate(http_request_duration_seconds_bucket[5m])
)

# Prediction (linear regression)
predict_linear(node_filesystem_free_bytes[1h], 4 * 3600)

# Absolute value
abs(delta(cpu_temp[5m]))

# Rounding
round(node_memory_MemAvailable_bytes / 1024 / 1024)
```

## Grafana

### Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Application Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m]))",
            "legendFormat": "Total RPS"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100",
            "legendFormat": "Error %"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P95"
          }
        ]
      }
    ]
  }
}
```

### Variables
```json
{
  "templating": {
    "list": [
      {
        "name": "environment",
        "type": "query",
        "query": "label_values(http_requests_total, environment)",
        "current": {
          "text": "production",
          "value": "production"
        }
      },
      {
        "name": "instance",
        "type": "query",
        "query": "label_values(http_requests_total{environment=\"$environment\"}, instance)"
      }
    ]
  }
}
```

## ELK Stack (Elasticsearch, Logstash, Kibana)

### Logstash Configuration
```ruby
input {
  beats {
    port => 5044
  }

  tcp {
    port => 5000
    codec => json
  }
}

filter {
  if [type] == "nginx" {
    grok {
      match => {
        "message" => "%{COMBINEDAPACHELOG}"
      }
    }

    date {
      match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
    }

    geoip {
      source => "clientip"
    }
  }

  if [type] == "application" {
    json {
      source => "message"
    }

    mutate {
      add_field => {
        "[@metadata][index]" => "app-logs-%{+YYYY.MM.dd}"
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[@metadata][index]}"
  }

  if [level] == "ERROR" {
    slack {
      url => "${SLACK_WEBHOOK}"
      format => "Error: %{message}"
    }
  }
}
```

### Filebeat Configuration
```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/app/*.log
    fields:
      type: application
      environment: production
    multiline.pattern: '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
    multiline.negate: true
    multiline.match: after

  - type: docker
    containers.ids: '*'
    processors:
      - add_docker_metadata: ~

output.logstash:
  hosts: ["logstash:5044"]

processors:
  - add_host_metadata: ~
  - add_cloud_metadata: ~
```

### Kibana Search Queries
```
# Search by field
level: ERROR

# Time range
@timestamp: [now-1h TO now]

# Boolean
level: ERROR AND service: api

# Wildcard
message: *timeout*

# Regex
message: /error \d+/

# Range
response_time: [500 TO *]

# Exists
_exists_: user_id

# Aggregation
service: api | stats count() by status_code
```

## Distributed Tracing

### OpenTelemetry

#### Node.js Setup
```javascript
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');

const sdk = new NodeSDK({
  traceExporter: new JaegerExporter({
    endpoint: 'http://jaeger:14268/api/traces',
  }),
  instrumentations: [getNodeAutoInstrumentations()],
  serviceName: 'my-service',
});

sdk.start();

// Custom spans
const { trace } = require('@opentelemetry/api');

async function processOrder(orderId) {
  const tracer = trace.getTracer('order-service');

  return tracer.startActiveSpan('processOrder', async (span) => {
    span.setAttribute('order.id', orderId);

    try {
      await validateOrder(orderId);
      await chargePayment(orderId);
      await fulfillOrder(orderId);

      span.setStatus({ code: SpanStatusCode.OK });
      return { success: true };
    } catch (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error.message
      });
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  });
}
```

#### Context Propagation
```javascript
const { propagation, context } = require('@opentelemetry/api');

// Extract context from headers
const extractedContext = propagation.extract(
  context.active(),
  req.headers
);

// Inject context into headers
const carrier = {};
propagation.inject(context.active(), carrier);
axios.get('http://downstream-service', {
  headers: carrier
});
```

## Alerting

### Prometheus Alerting Rules
```yaml
# alerts.yml
groups:
  - name: application
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) /
            sum(rate(http_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "High error rate on {{ $labels.instance }}"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"

      - alert: ServiceDown
        expr: up{job="application"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"

      - alert: DiskSpaceLow
        expr: |
          (
            node_filesystem_avail_bytes{mountpoint="/"} /
            node_filesystem_size_bytes{mountpoint="/"}
          ) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space on {{ $labels.instance }}"
          description: "Only {{ $value | humanizePercentage }} remaining"
```

### Alertmanager Configuration
```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/XXX'

route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'team-notifications'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      continue: true

    - match:
        team: backend
      receiver: 'backend-team'

receivers:
  - name: 'team-notifications'
    slack_configs:
      - channel: '#alerts'
        title: 'Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'critical-alerts'
    slack_configs:
      - channel: '#critical'
    pagerduty_configs:
      - service_key: 'xxx'

  - name: 'backend-team'
    email_configs:
      - to: 'backend@example.com'
        from: 'alerts@example.com'
        smarthost: 'smtp.gmail.com:587'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

## SLIs, SLOs, SLAs

### Service Level Indicators (SLIs)
Metrics that measure service performance:
```promql
# Availability SLI
sum(rate(http_requests_total{status!~"5.."}[30d])) /
sum(rate(http_requests_total[30d]))

# Latency SLI
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[30d])) by (le)
)

# Throughput SLI
sum(rate(http_requests_total[30d]))
```

### Service Level Objectives (SLOs)
Target values for SLIs:
```yaml
slos:
  - name: availability
    target: 99.9%  # 3 nines
    sli: |
      sum(rate(http_requests_total{status!~"5.."}[30d])) /
      sum(rate(http_requests_total[30d]))

  - name: latency
    target: 95%    # 95% of requests < 200ms
    sli: |
      sum(rate(http_request_duration_seconds_bucket{le="0.2"}[30d])) /
      sum(rate(http_request_duration_seconds_count[30d]))
```

### Error Budget
```promql
# Error budget remaining
1 - (
  (1 - (sum(rate(http_requests_total{status!~"5.."}[30d])) /
        sum(rate(http_requests_total[30d]))))
  /
  (1 - 0.999)  # SLO target
)

# Error budget burn rate
(1 - availability_sli) / (1 - availability_slo)
```

## Best Practices

### Logging

#### Structured Logging
```javascript
const logger = require('pino')();

logger.info({
  req_id: req.id,
  user_id: req.user.id,
  method: req.method,
  path: req.path,
  duration_ms: duration,
  status: res.statusCode
}, 'Request completed');
```

#### Log Levels
```
DEBUG: Detailed diagnostic information
INFO: General informational messages
WARN: Warning messages for degraded state
ERROR: Error events that still allow app to continue
FATAL: Critical errors that cause shutdown
```

#### What to Log
- Request/response details
- Errors with stack traces
- Authentication events
- Data changes (audit trail)
- Performance metrics
- External service calls

#### What NOT to Log
- Passwords or secrets
- Credit card numbers
- Personal identifiable information (PII)
- Session tokens

### Metrics

#### Naming Conventions
```
# Format: <namespace>_<name>_<unit>
http_requests_total
http_request_duration_seconds
database_connections_active
```

#### Cardinality
```
# Low cardinality - Good
http_requests_total{method="GET", status="200"}

# High cardinality - Bad (avoid)
http_requests_total{user_id="12345"}  # Too many unique values
```

### Alerting

#### Alert Design
- **Actionable**: Can be resolved by on-call engineer
- **Specific**: Clear what's wrong and where
- **Severe**: Requires immediate attention
- **Sustainable**: Won't cause alert fatigue

#### Alert Thresholds
```
Critical: User-facing impact, immediate action
Warning: Potential future impact, review during business hours
Info: FYI, no action required
```

## Dashboard Design

### Key Principles
1. **Above the fold**: Most important metrics visible without scrolling
2. **Consistent layout**: Similar dashboards use same structure
3. **Clear labels**: Descriptive titles and legends
4. **Appropriate time ranges**: Match use case (1h for ops, 30d for trends)
5. **Color coding**: Red for errors, yellow for warnings, green for OK

### Dashboard Types

#### Overview Dashboard
- Service health at a glance
- Golden signals
- Active alerts
- Key business metrics

#### Detail Dashboard
- Deep dive into specific service
- All relevant metrics
- Logs integration
- Trace links

#### SLO Dashboard
- SLI current values
- SLO targets
- Error budget remaining
- Historical trends

## Tools Comparison

| Tool | Best For | Strengths | Limitations |
|------|----------|-----------|-------------|
| Prometheus | Metrics | Time-series, powerful queries | Limited long-term storage |
| Grafana | Visualization | Beautiful dashboards | Not a data source |
| ELK | Logs | Full-text search, scalable | Resource intensive |
| Jaeger | Tracing | Distributed tracing | Sampling overhead |
| Datadog | All-in-one | Integrated platform | Expensive |
| New Relic | APM | Easy setup, great UI | Cost scales with data |

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [ELK Stack Guide](https://www.elastic.co/guide/)
- [OpenTelemetry](https://opentelemetry.io/)
- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
