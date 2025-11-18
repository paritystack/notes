# Observability

Comprehensive guide to building observable systems through strategic instrumentation, analysis, and continuous improvement.

## What is Observability?

Observability is the ability to understand internal system state from external outputs. Unlike monitoring (tracking known issues), observability helps debug unknown unknowns.

### Monitoring vs Observability

**Monitoring** (Known unknowns):
- "Is the service up?"
- "Is CPU above 80%?"
- Predefined dashboards and alerts

**Observability** (Unknown unknowns):
- "Why is this specific request slow?"
- "What changed between deployments?"
- Ad-hoc queries and exploration

```
Monitoring: Health checks, uptime
Observability: Understanding system behavior
```

## The Three Pillars

### 1. Metrics
Aggregated numerical measurements over time:

```javascript
// Counter: Monotonically increasing
requests_total.inc({ path: '/api', status: 200 });

// Gauge: Point-in-time value
memory_usage.set(process.memoryUsage().heapUsed);

// Histogram: Distribution of values
request_duration.observe(duration);

// Summary: Similar to histogram, quantiles calculated client-side
latency_summary.observe(duration);
```

**When to use**:
- System health (CPU, memory, disk)
- Business metrics (signups, revenue)
- Aggregated patterns (request rate, error rate)

### 2. Logs
Discrete event records with context. Logs are what happened and why.

#### Structured Logging
Use JSON format for machine-readable logs:

```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "level": "error",
  "message": "Database connection failed",
  "service": "api-gateway",
  "trace_id": "abc123",
  "span_id": "def456",
  "error": {
    "type": "ConnectionTimeout",
    "stack": "..."
  },
  "context": {
    "user_id": "user_789",
    "endpoint": "/checkout",
    "retry_count": 3
  }
}
```

**Log Levels:**
- **DEBUG**: Detailed diagnostic info (development only)
- **INFO**: General informational messages
- **WARN**: Warning for potentially harmful situations
- **ERROR**: Error events allowing app to continue
- **FATAL**: Severe errors causing shutdown

**Structured Logging Example:**
```javascript
// ❌ Bad: String interpolation
logger.info('User ' + userId + ' purchased item ' + itemId + ' for $' + price);

// ✅ Good: Structured fields
logger.info('Purchase completed', {
  user_id: userId,
  item_id: itemId,
  price: price,
  currency: 'USD',
  payment_method: 'credit_card'
});
```

**When to use**:
- Debugging specific issues
- Audit trails
- Unstructured investigation
- Compliance and security events

### 3. Traces
Request journey through distributed system:

```
[User Request] → API Gateway (50ms)
                 ├─ Auth Service (10ms)
                 ├─ Product Service (20ms)
                 │  └─ Database (15ms)
                 └─ Payment Service (200ms) ← SLOW!
                    └─ External API (190ms)
```

**When to use**:
- Debugging latency
- Understanding dependencies
- Visualizing request flow

## Observability Strategy

### Maturity Model

#### Level 1: Reactive
- Basic logging
- Simple uptime monitoring
- Manual investigation
- **Goal**: Know when things break

#### Level 2: Proactive
- Structured logs
- Metrics dashboards
- Basic alerting
- **Goal**: Detect issues before users

#### Level 3: Strategic
- Distributed tracing
- SLOs and error budgets
- Advanced correlation
- **Goal**: Understand system behavior

#### Level 4: Predictive
- Anomaly detection
- Predictive analytics
- Auto-remediation
- **Goal**: Prevent issues before they occur

### Observability-Driven Development

Build observability into development process:

```javascript
// 1. Add instrumentation during development
async function processOrder(orderId) {
  const span = trace.startSpan('processOrder');
  span.setAttribute('order.id', orderId);

  try {
    logger.info({ orderId }, 'Processing order');

    const order = await fetchOrder(orderId);
    metrics.orderValue.observe(order.total);

    await validateInventory(order);
    await chargePayment(order);

    logger.info({ orderId, total: order.total }, 'Order processed');
    return order;
  } catch (error) {
    logger.error({ orderId, error }, 'Order processing failed');
    metrics.orderErrors.inc({ reason: error.code });
    span.recordException(error);
    throw error;
  } finally {
    span.end();
  }
}
```

**Best practices**:
- Instrument code as you write it
- Include observability in code reviews
- Test instrumentation in development
- Document expected metrics and logs

## Architecture Patterns

### Centralized Observability Platform

```
┌─────────────┐
│ Application │──┐
└─────────────┘  │
                 ├──→ ┌──────────────┐
┌─────────────┐  │    │  Collector   │
│   Service   │──┤    │  (OpenTelem) │
└─────────────┘  │    └──────────────┘
                 │           │
┌─────────────┐  │           ├──→ Metrics (Prometheus)
│  Database   │──┘           ├──→ Logs (Loki/ES)
└─────────────┘              └──→ Traces (Jaeger/Tempo)
```

**Benefits**:
- Unified data collection
- Single configuration point
- Vendor neutrality
- Cost optimization

### Sampling Strategy

Not all data needs to be collected:

```javascript
// Head-based sampling (decision at start)
const sampler = new TraceIdRatioBasedSampler(0.1); // 10%

// Tail-based sampling (decision at end)
if (span.duration > 1000 || span.hasError) {
  span.setSampled(true); // Always keep slow/error traces
} else {
  span.setSampled(Math.random() < 0.01); // 1% of normal traces
}

// Adaptive sampling
const rate = errorRate > 0.01 ? 1.0 : 0.1; // 100% when errors high
```

**Strategies**:
- **Always sample**: Errors, slow requests, critical paths
- **Never sample**: Health checks, static assets
- **Adaptive**: Increase during incidents

### Context Propagation

Link observability data across services:

```javascript
// Service A: Create context
const trace = tracer.startSpan('handleRequest');
const context = {
  'trace-id': trace.spanContext().traceId,
  'span-id': trace.spanContext().spanId,
  'request-id': generateRequestId(),
  'user-id': req.user.id
};

// Pass to Service B
await axios.post('http://service-b/api', data, {
  headers: context
});

// Service B: Extract context
const traceId = req.headers['trace-id'];
const parentSpan = req.headers['span-id'];
const childSpan = tracer.startSpan('processData', {
  parent: parentSpan
});

// Both services now linked in distributed trace
```

## Implementation Patterns

### Instrumentation Layers

#### 1. Infrastructure Layer
```yaml
# Kubernetes metrics via prometheus
apiVersion: v1
kind: Service
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
    prometheus.io/path: "/metrics"
```

#### 2. Application Layer
```javascript
// Auto-instrumentation
const { registerInstrumentations } = require('@opentelemetry/instrumentation');
registerInstrumentations({
  instrumentations: [
    new HttpInstrumentation(),
    new ExpressInstrumentation(),
    new PgInstrumentation(),
    new RedisInstrumentation()
  ]
});

// Custom business metrics
const checkoutMetrics = {
  started: new Counter({ name: 'checkout_started_total' }),
  completed: new Counter({ name: 'checkout_completed_total' }),
  abandoned: new Counter({ name: 'checkout_abandoned_total' }),
  value: new Histogram({
    name: 'checkout_value_dollars',
    buckets: [10, 50, 100, 500, 1000]
  })
};
```

#### 3. Business Layer
```javascript
// Business events
async function completeCheckout(cart) {
  const startTime = Date.now();

  try {
    const order = await createOrder(cart);

    // Business observability
    events.emit('checkout.completed', {
      order_id: order.id,
      user_id: cart.userId,
      total: order.total,
      items: order.items.length,
      duration_ms: Date.now() - startTime,
      payment_method: order.paymentMethod,
      promocode: order.promoCode || null
    });

    return order;
  } catch (error) {
    events.emit('checkout.failed', {
      user_id: cart.userId,
      error_type: error.name,
      step: error.step
    });
    throw error;
  }
}
```

### Correlation Patterns

#### Correlating Metrics and Logs
```javascript
// Add trace context to logs
logger.info({
  trace_id: span.spanContext().traceId,
  span_id: span.spanContext().spanId,
  message: 'Order processed'
});

// Query logs by trace ID
// logs: trace_id:"abc123"
// See all logs for this request
```

#### Correlating Logs and Traces
```javascript
// Add log events to traces
span.addEvent('Payment authorized', {
  'payment.id': paymentId,
  'payment.method': 'credit_card'
});

// Link from trace span to logs
// Grafana: Click span → "View logs"
```

#### Correlating Metrics and Traces
```javascript
// Exemplars link metrics to traces
histogram.observe(
  { endpoint: '/checkout' },
  duration,
  { trace_id: traceId } // Exemplar
);

// In Grafana: Click metric spike → See example traces
```

## Advanced Patterns

### High Cardinality Data

**Problem**: Too many unique label values

```javascript
// ❌ Bad: User ID as label (millions of users)
requests.inc({ user_id: req.user.id });

// ✅ Good: User ID in logs only
logger.info({ user_id: req.user.id }, 'Request');
requests.inc({ endpoint: req.path });
```

**Solutions**:
- Use aggregated labels in metrics
- Store high-cardinality data in logs/traces
- Use cardinality limits and alerts

### Dynamic Sampling

Adjust sampling based on conditions:

```javascript
class AdaptiveSampler {
  constructor() {
    this.errorRate = 0;
    this.baseRate = 0.01; // 1%
  }

  shouldSample(span) {
    // Always sample errors
    if (span.hasError) return true;

    // Sample more during high error rates
    const rate = this.errorRate > 0.05
      ? 0.5  // 50% when errors high
      : this.baseRate;

    // Sample all slow requests
    if (span.duration > 1000) return true;

    return Math.random() < rate;
  }

  updateErrorRate(rate) {
    this.errorRate = rate;
  }
}
```

### Real User Monitoring (RUM)

Frontend observability:

```javascript
// Browser instrumentation
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import { DocumentLoadInstrumentation } from '@opentelemetry/instrumentation-document-load';

const provider = new WebTracerProvider();
provider.register();

// Measure web vitals
import { onCLS, onFID, onLCP } from 'web-vitals';

onCLS(metric => {
  sendMetric('web_vitals_cls', metric.value, {
    page: window.location.pathname
  });
});

onFID(metric => {
  sendMetric('web_vitals_fid', metric.value);
});

onLCP(metric => {
  sendMetric('web_vitals_lcp', metric.value);
});

// User journey tracking
class UserJourney {
  constructor() {
    this.sessionId = generateSessionId();
    this.events = [];
  }

  track(event) {
    this.events.push({
      timestamp: Date.now(),
      type: event.type,
      data: event.data,
      page: window.location.pathname
    });

    // Send to analytics
    if (this.events.length >= 10) {
      this.flush();
    }
  }

  flush() {
    sendEvents(this.sessionId, this.events);
    this.events = [];
  }
}
```

### Synthetic Monitoring

Proactive monitoring with simulated traffic:

```javascript
// Synthetic health checks
const syntheticChecks = [
  {
    name: 'api_health',
    interval: '1m',
    endpoint: 'https://api.example.com/health',
    assertions: [
      { type: 'status', value: 200 },
      { type: 'latency', max: 500 },
      { type: 'body', contains: '"status":"ok"' }
    ]
  },
  {
    name: 'user_flow',
    interval: '5m',
    steps: [
      { action: 'visit', url: '/login' },
      { action: 'fill', field: 'email', value: 'test@example.com' },
      { action: 'fill', field: 'password', value: 'test123' },
      { action: 'click', selector: '#submit' },
      { action: 'assert', selector: '.dashboard', exists: true }
    ]
  }
];
```

## Cost Optimization

### Data Retention Strategy

```yaml
# Tiered retention
retention:
  metrics:
    raw: 15d        # Full resolution
    5m: 90d         # 5min aggregation
    1h: 1y          # 1hour aggregation

  logs:
    hot: 7d         # Fast search (ES)
    warm: 30d       # Slower search (S3)
    cold: 90d       # Archive (Glacier)

  traces:
    full: 7d        # All traces
    sampled: 30d    # 10% sample
    errors: 90d     # Error traces only
```

### Reducing Volume

```javascript
// 1. Smart log levels
if (process.env.NODE_ENV === 'production') {
  logger.level = 'info'; // Skip debug logs
}

// 2. Sampling
const shouldLog = req.path.startsWith('/api')
  || Math.random() < 0.01; // 1% of other requests

// 3. Deduplication
const errorCache = new Map();
function logError(error) {
  const key = `${error.code}:${error.message}`;
  const lastSeen = errorCache.get(key);

  if (!lastSeen || Date.now() - lastSeen > 60000) {
    logger.error(error);
    errorCache.set(key, Date.now());
  }
}

// 4. Aggregation
// Instead of individual request logs
metrics.httpRequests.inc(); // Much cheaper

// 5. Filtering
// Don't log health checks
if (req.path === '/health') return next();
```

## Observability for Microservices

### Service Mesh Integration

```yaml
# Istio automatic observability
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
    - reviews
  http:
    - match:
        - headers:
            end-user:
              exact: jason
      route:
        - destination:
            host: reviews
            subset: v2
      # Automatic metrics for:
      # - Request rate
      # - Error rate
      # - Latency distribution
      # - Distributed traces
```

### Distributed Tracing Best Practices

```javascript
// 1. Consistent naming
span.name = `${method} ${resource}`; // GET /users
span.name = `${service}.${operation}`; // userService.createUser

// 2. Rich attributes
span.setAttributes({
  // HTTP semantic conventions
  'http.method': 'POST',
  'http.url': req.url,
  'http.status_code': res.statusCode,

  // Database semantic conventions
  'db.system': 'postgresql',
  'db.statement': query,
  'db.name': 'users',

  // Custom business context
  'user.id': userId,
  'order.total': orderTotal
});

// 3. Span hierarchy
async function processOrder(orderId) {
  return tracer.startActiveSpan('processOrder', async (orderSpan) => {
    await tracer.startActiveSpan('validateOrder', async (validateSpan) => {
      // validation logic
      validateSpan.end();
    });

    await tracer.startActiveSpan('chargePayment', async (paymentSpan) => {
      // payment logic
      paymentSpan.end();
    });

    orderSpan.end();
  });
}
```

### Service Dependencies

Track and visualize service relationships:

```javascript
// Dependency graph
const dependencies = {
  'api-gateway': ['auth-service', 'user-service', 'order-service'],
  'order-service': ['inventory-service', 'payment-service', 'notification-service'],
  'payment-service': ['stripe-api', 'fraud-detection']
};

// Detect circular dependencies
// Alert on new dependencies
// Visualize in Grafana/Jaeger
```

## Alerting Strategy

### Alert Fatigue Prevention

```yaml
# Good alert characteristics
alert: HighErrorRate
expr: error_rate > 0.05  # 5% errors
for: 5m                   # Sustained for 5 minutes
annotations:
  summary: "High error rate detected"
  runbook: "https://wiki.company.com/runbooks/high-error-rate"
  dashboard: "https://grafana.company.com/d/errors"
labels:
  severity: critical
  team: backend
  oncall: true

# Avoid alert fatigue
# ❌ Don't alert on:
#   - Symptoms without impact
#   - Transient spikes
#   - Non-actionable metrics
#   - Too many conditions

# ✅ Do alert on:
#   - User-facing issues
#   - SLO violations
#   - Security events
#   - Clear action needed
```

### Progressive Rollout Observability

Monitor during deployments:

```javascript
// Canary analysis
const canaryMetrics = {
  baseline: await queryMetrics('version=v1', timeRange),
  canary: await queryMetrics('version=v2', timeRange)
};

const analysis = {
  errorRate: canary.errors / canary.requests,
  errorIncrease: (canary.errors / canary.requests) -
                 (baseline.errors / baseline.requests),
  p95Latency: canary.p95,
  latencyIncrease: canary.p95 - baseline.p95
};

if (analysis.errorIncrease > 0.01 || analysis.latencyIncrease > 100) {
  rollback();
} else {
  promote();
}
```

## Logging Platforms

### ELK Stack (Elasticsearch, Logstash, Kibana)

Complete log aggregation and analysis platform.

#### Architecture
```
Applications → Filebeat/Fluentd → Logstash → Elasticsearch → Kibana
                                      ↓
                                  Filtering
                                  Enrichment
```

#### Logstash Pipeline
```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  # Parse JSON logs
  json {
    source => "message"
  }

  # Extract fields from message
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }

  # Add geolocation
  geoip {
    source => "client_ip"
  }

  # Parse timestamps
  date {
    match => [ "timestamp", "ISO8601" ]
    target => "@timestamp"
  }

  # Add custom fields
  mutate {
    add_field => {
      "environment" => "production"
      "indexed_at" => "%{@timestamp}"
    }
    remove_field => ["temp_field"]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-%{+YYYY.MM.dd}"
  }
}
```

#### Kibana Query Language (KQL)
```
# Simple field match
status: 500

# Boolean operators
status: 500 AND service: api

# Wildcards
message: *timeout*

# Range queries
response_time >= 1000

# Exists query
_exists_: error.stack

# Time range
@timestamp >= "2025-01-15T00:00:00"

# Aggregations
service: api | stats count() by status_code
```

### Grafana Loki

Lightweight, cost-effective log aggregation system.

#### Why Loki?
- **Cost-effective**: Only indexes labels, not full text
- **Simple**: Easy to operate, horizontally scalable
- **Integrated**: Works seamlessly with Grafana
- **Prometheus-like**: Uses familiar label model

#### Architecture
```
Applications → Promtail → Loki → Grafana
                ↓
          Log files
```

#### Promtail Configuration
```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          __path__: /var/log/*.log

  - job_name: containers
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'
```

#### LogQL Queries
```logql
# Stream selector
{service="api", environment="production"}

# Filter by content
{service="api"} |= "error"
{service="api"} != "health"

# JSON parsing
{service="api"} | json | status_code >= 500

# Pattern matching
{service="api"} |~ "timeout|deadline"

# Rate queries
rate({service="api"}[5m])

# Count over time
count_over_time({service="api", level="error"}[1h])

# Aggregation
sum(rate({service="api"}[5m])) by (status_code)
```

#### Loki vs ELK Comparison

| Feature | Loki | ELK |
|---------|------|-----|
| **Index Strategy** | Labels only | Full-text |
| **Cost** | Lower | Higher |
| **Query Speed** | Fast for label queries | Fast for full-text |
| **Storage** | More efficient | More expensive |
| **Complexity** | Simple | Complex |
| **Best For** | Metrics-style logs | Full-text search |

### Log Sampling

Reduce volume while maintaining visibility.

#### Head Sampling
Decide at creation time:

```javascript
const shouldLog = (level, req) => {
  // Always log errors
  if (level === 'error' || level === 'fatal') {
    return true;
  }

  // Always log important endpoints
  if (req.path.startsWith('/api/payment')) {
    return true;
  }

  // Sample 10% of info logs
  if (level === 'info') {
    return Math.random() < 0.1;
  }

  // Sample 1% of debug logs
  return Math.random() < 0.01;
};

if (shouldLog('info', req)) {
  logger.info({ req }, 'Request processed');
}
```

#### Tail Sampling
Decide after processing:

```javascript
class LogBuffer {
  constructor() {
    this.buffer = [];
    this.maxSize = 1000;
  }

  add(logEntry) {
    this.buffer.push(logEntry);

    if (this.buffer.length > this.maxSize) {
      this.flush();
    }
  }

  flush() {
    const hasErrors = this.buffer.some(log => log.level === 'error');
    const isSlow = this.buffer.some(log => log.duration > 1000);

    if (hasErrors || isSlow) {
      // Send all logs for this request
      this.sendLogs(this.buffer);
    } else {
      // Sample 1% of normal requests
      if (Math.random() < 0.01) {
        this.sendLogs(this.buffer);
      }
    }

    this.buffer = [];
  }
}
```

#### Dynamic Sampling
Adjust based on conditions:

```javascript
class AdaptiveLogSampler {
  constructor() {
    this.errorRate = 0;
    this.baseRate = 0.01;
  }

  getSampleRate(level) {
    if (level === 'error' || level === 'fatal') {
      return 1.0; // 100%
    }

    // Increase sampling during high error rates
    if (this.errorRate > 0.05) {
      return 0.5; // 50%
    }

    if (this.errorRate > 0.01) {
      return 0.1; // 10%
    }

    return this.baseRate; // 1%
  }

  updateErrorRate(rate) {
    this.errorRate = rate;
  }
}
```

### Log Aggregation Patterns

#### Centralized Logging
```
Service A ──┐
Service B ──┼──→ Log Aggregator → Storage → Analysis
Service C ──┘
```

#### Multi-Tier Logging
```
Edge Logs → Regional Aggregator → Central Storage
                                      ↓
                                  Archive (S3)
```

#### Hybrid Approach
```
High-value logs → Real-time (Loki/ES)
All logs → Cold storage (S3)
```

## Distributed Tracing Platforms

### OpenTelemetry
Industry-standard observability framework.

```javascript
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const { ZipkinExporter } = require('@opentelemetry/exporter-zipkin');

const sdk = new NodeSDK({
  serviceName: 'my-service',
  traceExporter: new JaegerExporter({
    endpoint: 'http://jaeger:14268/api/traces',
  }),
  instrumentations: [getNodeAutoInstrumentations()],
});

sdk.start();
```

### Jaeger
Distributed tracing platform inspired by Dapper and OpenZipkin.

**Features:**
- Distributed context propagation
- Distributed transaction monitoring
- Root cause analysis
- Service dependency analysis
- Performance optimization

**Architecture:**
```
Client → Agent → Collector → Storage (Cassandra/ES) → UI
```

### Zipkin
Distributed tracing system.

**Features:**
- Simpler than Jaeger
- Good for smaller deployments
- Native support in Spring Boot
- Compatible with OpenTelemetry

**Jaeger vs Zipkin:**

| Feature | Jaeger | Zipkin |
|---------|--------|--------|
| **Origin** | Uber | Twitter |
| **Storage** | Cassandra, ES, Badger | ES, MySQL, Cassandra |
| **Sampling** | Adaptive | Fixed |
| **Architecture** | More components | Simpler |
| **Best For** | Large scale | Simple setups |

## Incident Response

### Incident Lifecycle

```
Detection → Response → Resolution → Postmortem
```

#### 1. Detection
```javascript
// Automated detection
if (errorRate > SLO_THRESHOLD) {
  incident.create({
    severity: 'critical',
    title: 'High error rate detected',
    affected_service: 'api-gateway',
    metrics: {
      current_error_rate: errorRate,
      threshold: SLO_THRESHOLD
    }
  });
}
```

#### 2. Response
```yaml
# Incident response runbook
steps:
  1. Acknowledge alert
  2. Check dashboard: https://grafana.company.com/d/incident
  3. Review recent deployments
  4. Check dependencies status
  5. Enable debug logging if needed
  6. Communicate in #incidents channel
```

#### 3. Resolution
- Rollback bad deployment
- Scale up resources
- Fix configuration
- Deploy hotfix

#### 4. Postmortem

**Blameless Postmortem Template:**

```markdown
# Incident Postmortem: [Title]

## Summary
Brief description of what happened

## Impact
- Duration: 2 hours 15 minutes
- Affected users: ~15% of traffic
- Revenue impact: $XX,XXX
- Service: api-gateway

## Timeline (all times UTC)
- 14:00 - Deploy v2.3.4 to production
- 14:05 - Error rate increases to 5%
- 14:08 - PagerDuty alert triggers
- 14:10 - On-call engineer starts investigation
- 14:20 - Root cause identified: DB connection pool exhausted
- 14:25 - Decision to rollback
- 14:30 - Rollback initiated
- 14:35 - Service recovered
- 15:00 - Confirmed stable

## Root Cause
Database connection pool size was too small for new traffic pattern.
New feature made 3x more DB calls per request than expected.

## Resolution
1. Rolled back to v2.3.3
2. Increased connection pool size
3. Re-deployed with fix

## What Went Well
- Alert triggered within 3 minutes
- Clear runbooks enabled fast response
- Communication was effective
- Rollback process worked smoothly

## What Went Wrong
- Connection pool not load tested
- No gradual rollout (canary)
- Missing query count metrics
- Load tests didn't simulate production pattern

## Action Items
- [ ] Add connection pool metrics (@alice, 2025-01-20)
- [ ] Implement canary deployments (@bob, 2025-01-25)
- [ ] Add query count per request metric (@charlie, 2025-01-22)
- [ ] Update load test scenarios (@dave, 2025-01-30)
- [ ] Document DB connection tuning (@eve, 2025-01-23)

## Lessons Learned
- Always canary deploy
- Monitor connection pools
- Load test with production-like data
```

### On-Call Best Practices

#### On-Call Rotation
```yaml
rotation:
  primary: 7 days
  secondary: 7 days
  handoff: Monday 10:00 AM

responsibilities:
  - Respond to pages within 15 minutes
  - Investigate and mitigate incidents
  - Write postmortems
  - Update runbooks

compensation:
  - Shift differential
  - Time off in lieu
  - Rotation credits
```

#### Runbook Template
```markdown
# Runbook: High API Error Rate

## Symptoms
- Alert: "High error rate on api-gateway"
- Dashboard: Error rate > 5%
- User impact: API requests failing

## Severity
Critical (user-facing)

## Diagnosis
1. Check Grafana dashboard:
   https://grafana.company.com/d/api-errors

2. Query recent errors:
   ```
   {service="api"} | json | status_code >= 500
   ```

3. Check recent deployments:
   ```bash
   kubectl rollout history deployment/api-gateway
   ```

4. Check dependencies:
   - Database: https://status.db.company.com
   - Cache: https://status.redis.company.com
   - External APIs: Check status pages

## Mitigation
### If recent deployment:
```bash
kubectl rollout undo deployment/api-gateway
```

### If database issue:
```bash
# Check connection pool
kubectl exec -it api-gateway-xxx -- curl localhost:9090/metrics | grep db_connections

# Scale up if needed
kubectl scale deployment/api-gateway --replicas=10
```

### If external API down:
```bash
# Enable circuit breaker
kubectl set env deployment/api-gateway CIRCUIT_BREAKER_ENABLED=true
```

## Escalation
- Primary: @team-backend
- Secondary: @team-platform
- Manager: @engineering-manager

## Postmortem
Required for all critical incidents
```

### Alerting Integration

#### PagerDuty Integration
```javascript
const { PagerDutyClient } = require('pagerduty-client');

const pd = new PagerDutyClient({
  integrationKey: process.env.PD_INTEGRATION_KEY
});

async function triggerIncident(alert) {
  await pd.sendEvent({
    event_action: 'trigger',
    payload: {
      summary: alert.title,
      severity: alert.severity,
      source: alert.source,
      custom_details: {
        error_rate: alert.metrics.error_rate,
        threshold: alert.threshold,
        dashboard: alert.dashboard_url
      }
    },
    links: [
      {
        href: alert.dashboard_url,
        text: 'View Dashboard'
      },
      {
        href: alert.runbook_url,
        text: 'View Runbook'
      }
    ]
  });
}
```

#### Opsgenie Integration
```javascript
const opsgenie = require('opsgenie-sdk');

const client = new opsgenie.AlertApi({
  apiKey: process.env.OPSGENIE_API_KEY
});

async function createAlert(alert) {
  await client.createAlert({
    message: alert.title,
    description: alert.description,
    priority: alertSeverityToPriority(alert.severity),
    tags: [alert.service, alert.environment],
    details: {
      error_rate: alert.metrics.error_rate,
      affected_users: alert.affected_users
    },
    responders: [
      { type: 'team', name: 'Backend Team' }
    ],
    actions: ['View Dashboard', 'View Logs'],
    entity: alert.service,
    source: 'Prometheus'
  });
}

function alertSeverityToPriority(severity) {
  const map = {
    critical: 'P1',
    high: 'P2',
    medium: 'P3',
    low: 'P4',
    info: 'P5'
  };
  return map[severity] || 'P3';
}
```

## Debugging Production Issues

### Systematic Debugging Approach

#### 1. Gather Context
```bash
# What changed recently?
git log --since="2 hours ago" --oneline

# When did it start?
# Check metrics dashboard for inflection point

# What's the scope?
# All users? Specific region? Specific feature?
```

#### 2. Form Hypothesis
```
Theory: Database connection pool exhausted
Evidence needed:
  - Connection pool metrics
  - Database query latency
  - Error messages mentioning connections
```

#### 3. Test Hypothesis
```bash
# Check connection pool
curl http://api:9090/metrics | grep db_pool

# Check database
kubectl logs -l app=api --tail=100 | grep -i connection

# Check traces for slow DB queries
# Jaeger UI: Filter by service=api, minDuration=1000ms
```

#### 4. Mitigate
```bash
# Quick fix: Scale up
kubectl scale deployment/api --replicas=10

# Better fix: Increase pool size
kubectl set env deployment/api DB_POOL_SIZE=50
```

#### 5. Verify
```bash
# Check error rate returned to normal
curl -s http://prometheus:9090/api/v1/query?query='error_rate' | jq .

# Check latency
curl -s http://prometheus:9090/api/v1/query?query='p95_latency' | jq .
```

### Production Debugging Tools

#### Live Debugging
```bash
# Attach debugger to running container (Node.js)
kubectl exec -it api-gateway-xxx -- kill -USR1 1
kubectl port-forward api-gateway-xxx 9229:9229
# Chrome DevTools: chrome://inspect

# Python
kubectl exec -it api-gateway-xxx -- python -m pdb app.py

# Go (requires delve)
kubectl exec -it api-gateway-xxx -- dlv attach $(pidof app)
```

#### Dynamic Logging
```javascript
// Enable debug logs for specific user
app.use((req, res, next) => {
  if (req.headers['x-debug-user'] === 'user_123') {
    req.log = logger.child({ level: 'debug' });
  }
  next();
});

// Enable via feature flag
if (featureFlags.isEnabled('debug-logging', userId)) {
  logger.level = 'debug';
}
```

#### Traffic Replay
```bash
# Capture traffic with tcpdump
tcpdump -i eth0 -w capture.pcap port 8080

# Replay with tcpreplay
tcpreplay --topspeed -i eth0 capture.pcap

# Or use gor for more control
gor --input-raw :8080 --output-http="http://staging:8080"
```

#### Query Analysis
```javascript
// Add query explanation
const explain = await db.query('EXPLAIN ANALYZE ' + sqlQuery);
logger.info({ explain }, 'Query plan');

// Log slow queries
const start = Date.now();
const result = await db.query(sqlQuery);
const duration = Date.now() - start;

if (duration > 1000) {
  logger.warn({
    query: sqlQuery,
    duration,
    rows: result.rowCount
  }, 'Slow query detected');
}
```

### Common Production Issues

#### Memory Leaks
```javascript
// Detect memory leaks
const heapdump = require('heapdump');

setInterval(() => {
  const usage = process.memoryUsage();
  logger.info({ memory: usage }, 'Memory usage');

  if (usage.heapUsed > THRESHOLD) {
    heapdump.writeSnapshot(`/tmp/heap-${Date.now()}.heapsnapshot`);
  }
}, 60000);

// Analyze with Chrome DevTools
```

#### Connection Leaks
```javascript
// Track connection lifecycle
class ConnectionPool {
  constructor() {
    this.active = new Set();
  }

  async acquire() {
    const conn = await this.pool.acquire();
    this.active.add(conn);
    conn._acquiredAt = Date.now();
    return conn;
  }

  release(conn) {
    this.active.delete(conn);
    this.pool.release(conn);
  }

  checkLeaks() {
    const now = Date.now();
    for (const conn of this.active) {
      if (now - conn._acquiredAt > 30000) {
        logger.warn({
          age: now - conn._acquiredAt,
          stack: conn._stack
        }, 'Potential connection leak');
      }
    }
  }
}
```

#### Race Conditions
```javascript
// Add request tracing
const traceRequest = (req, res, next) => {
  req.id = generateId();
  req.startTime = Date.now();

  logger.info({
    req_id: req.id,
    method: req.method,
    path: req.path
  }, 'Request start');

  res.on('finish', () => {
    logger.info({
      req_id: req.id,
      duration: Date.now() - req.startTime,
      status: res.statusCode
    }, 'Request end');
  });

  next();
};
```

## Tools and Platforms

### Open Source Stack

```yaml
# Metrics
prometheus:
  image: prom/prometheus
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - 9090:9090

# Visualization
grafana:
  image: grafana/grafana
  ports:
    - 3000:3000
  environment:
    - GF_AUTH_ANONYMOUS_ENABLED=true

# Logs
loki:
  image: grafana/loki
  ports:
    - 3100:3100

# Traces
jaeger:
  image: jaegertracing/all-in-one
  ports:
    - 16686:16686  # UI
    - 14268:14268  # Collector

# Collector
otel-collector:
  image: otel/opentelemetry-collector
  volumes:
    - ./otel-config.yaml:/etc/otel-collector-config.yaml
```

### Commercial Platforms

| Platform | Strengths | Best For |
|----------|-----------|----------|
| **Datadog** | All-in-one, great UX | Teams wanting simplicity |
| **New Relic** | APM, easy setup | Application monitoring |
| **Splunk** | Log analysis, enterprise | Large organizations |
| **Honeycomb** | High-cardinality, exploration | Complex debugging |
| **Lightstep** | Distributed tracing | Microservices |
| **Grafana Cloud** | Managed open source | OSS stack without ops |

### Evaluation Criteria

```
✓ Data retention policies
✓ Query performance
✓ Cost at scale
✓ Integration ecosystem
✓ Team expertise
✓ Vendor lock-in
✓ SLA guarantees
✓ Support quality
```

## Observability Culture

### Building Observability Practice

**Phase 1: Foundation** (Months 1-3)
- Standardize logging format
- Deploy metrics collection
- Create first dashboards
- Document on-call process

**Phase 2: Expansion** (Months 4-6)
- Add distributed tracing
- Define SLOs
- Build runbooks
- Train team

**Phase 3: Maturity** (Months 7-12)
- Observability in code reviews
- Automated analysis
- Predictive alerting
- Continuous improvement

### Team Practices

```
Daily:
  - Check dashboards
  - Review overnight alerts
  - Triage new issues

Weekly:
  - Alert review (remove noise)
  - Incident retrospectives
  - Dashboard improvements

Monthly:
  - SLO review
  - Cost optimization
  - Tool evaluation
  - Training sessions

Quarterly:
  - Observability roadmap
  - Platform upgrades
  - Process improvements
```

## Common Pitfalls

### 1. Too Much Data
**Problem**: Collecting everything, analyzing nothing
**Solution**: Start with golden signals, expand based on needs

### 2. Vanity Metrics
**Problem**: Tracking metrics that don't drive decisions
**Solution**: Ask "what action would we take?" for each metric

### 3. Alert Fatigue
**Problem**: Too many alerts, all ignored
**Solution**: Ruthlessly prune non-actionable alerts

### 4. Tool Sprawl
**Problem**: Different tool for each team
**Solution**: Standardize on platform, federate access

### 5. Missing Context
**Problem**: Metrics without business meaning
**Solution**: Link technical metrics to business outcomes

### 6. Inconsistent Instrumentation
**Problem**: Each service does it differently
**Solution**: Shared libraries, code generation, conventions

## Measuring Success

### Observability KPIs

```javascript
const observabilityKPIs = {
  // Detection
  meanTimeToDetect: 'MTTD',      // How fast we notice issues

  // Investigation
  meanTimeToUnderstand: 'MTTU',  // How fast we understand root cause

  // Resolution
  meanTimeToResolve: 'MTTR',     // How fast we fix issues

  // Prevention
  changeFailureRate: 'CFR',      // % of changes causing issues
  deploymentFrequency: 'DF'      // How often we can deploy
};

// Track improvement over time
// Before observability: MTTR = 4 hours
// After observability: MTTR = 20 minutes
```

### ROI Calculation

```
Downtime cost reduction:
  Before: 10 hours/month × $10k/hour = $100k/month
  After:  2 hours/month × $10k/hour = $20k/month
  Savings: $80k/month

Development efficiency:
  Faster debugging: 5 hours/week × 10 engineers = 50 hours
  Value: $10k/month

Total value: $90k/month
Tool cost: $5k/month
ROI: 18x
```

## Resources

### Books
- Site Reliability Engineering (Google)
- Observability Engineering (Honeycomb)
- Distributed Systems Observability (Cindy Sridharan)

### Tools
- [OpenTelemetry](https://opentelemetry.io/) - Vendor-neutral observability
- [Prometheus](https://prometheus.io/) - Metrics collection
- [Grafana](https://grafana.com/) - Visualization
- [Jaeger](https://www.jaegertracing.io/) - Distributed tracing

### Learning
- [Grafana Labs Tutorials](https://grafana.com/tutorials/)
- [OpenTelemetry Demo](https://github.com/open-telemetry/opentelemetry-demo)
- [Charity Majors Blog](https://charity.wtf/)
