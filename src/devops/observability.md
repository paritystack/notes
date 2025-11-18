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
Discrete event records with context:

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

**When to use**:
- Debugging specific issues
- Audit trails
- Unstructured investigation

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
