# Performance Testing

Performance testing evaluates how a system performs under various workloads, measuring speed, scalability, and stability.

## Types of Performance Testing

### Load Testing
Tests system behavior under expected load.

**Goal**: Verify system handles normal and peak load
**Metrics**: Response time, throughput, error rate

```python
# Example: Test 100 concurrent users
Users: 100
Duration: 30 minutes
Expected: < 2s response time
```

### Stress Testing
Tests system limits by increasing load beyond normal capacity.

**Goal**: Find breaking point
**Metrics**: Maximum capacity, failure mode

```python
# Example: Gradually increase load until failure
Start: 100 users
Increment: +50 users every 5 minutes
Until: System fails or reaches 1000 users
```

### Spike Testing
Tests system response to sudden load increases.

**Goal**: Verify system handles sudden traffic spikes
**Metrics**: Recovery time, error rate during spike

```python
# Example: Sudden traffic spike
Normal: 100 users
Spike: 1000 users for 2 minutes
Normal: 100 users
```

### Endurance Testing (Soak Testing)
Tests system over extended period.

**Goal**: Find memory leaks, resource exhaustion
**Metrics**: Memory usage, CPU usage over time

```python
# Example: Run for 24 hours
Users: 200
Duration: 24 hours
Monitor: Memory, CPU, disk I/O
```

### Scalability Testing
Tests how system scales with increased resources.

**Goal**: Verify horizontal/vertical scaling
**Metrics**: Performance vs resources

### Volume Testing
Tests system with large data volumes.

**Goal**: Verify database performance
**Metrics**: Query time with millions of records

## Load Testing Tools

### Locust (Python)

**Installation:**
```bash
pip install locust
```

**Basic test (locustfile.py):**
```python
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3s between requests

    @task
    def view_homepage(self):
        self.client.get("/")

    @task(3)  # 3x more likely than other tasks
    def view_product(self):
        self.client.get("/product/1")

    @task(2)
    def add_to_cart(self):
        self.client.post("/cart", json={
            "product_id": 1,
            "quantity": 2
        })

    def on_start(self):
        # Login once per user
        self.client.post("/login", json={
            "username": "test",
            "password": "test"
        })
```

**Advanced features:**
```python
from locust import HttpUser, task, between, events
import logging

class AdvancedUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def load_dashboard(self):
        # Measure specific operation
        with self.client.get("/dashboard", catch_response=True) as response:
            if response.elapsed.total_seconds() > 2:
                response.failure(f"Too slow: {response.elapsed.total_seconds()}s")
            elif response.status_code != 200:
                response.failure(f"Failed: {response.status_code}")
            else:
                response.success()

    @task
    def search(self):
        # Test search with various queries
        queries = ["laptop", "phone", "tablet"]
        for query in queries:
            self.client.get(f"/search?q={query}")

    @task
    def checkout_flow(self):
        # Sequential steps
        self.client.get("/cart")
        self.client.post("/checkout/address", json={"address": "123 Main St"})
        self.client.post("/checkout/payment", json={"card": "4111111111111111"})
        self.client.post("/checkout/confirm")

# Custom event handlers
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logging.info("Load test starting")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logging.info("Load test stopping")
```

**Run tests:**
```bash
# Web UI mode
locust -f locustfile.py --host=http://localhost:8000

# Headless mode
locust -f locustfile.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 30m --headless

# Distributed mode (master)
locust -f locustfile.py --master

# Distributed mode (workers)
locust -f locustfile.py --worker --master-host=localhost
```

### k6 (JavaScript)

**Installation:**
```bash
# macOS
brew install k6

# Linux
wget https://github.com/grafana/k6/releases/download/v0.46.0/k6-v0.46.0-linux-amd64.tar.gz
```

**Basic test (script.js):**
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests < 500ms
    http_req_failed: ['rate<0.01'],     // Error rate < 1%
    errors: ['rate<0.1'],               // Custom error rate < 10%
  },
};

export default function () {
  // Homepage
  let response = http.get('http://localhost:8000/');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'page loaded': (r) => r.body.includes('<title>'),
  });

  sleep(1);

  // API request
  response = http.post('http://localhost:8000/api/users', JSON.stringify({
    name: 'Test User',
    email: 'test@example.com',
  }), {
    headers: { 'Content-Type': 'application/json' },
  });

  const success = check(response, {
    'user created': (r) => r.status === 201,
  });

  errorRate.add(!success);

  sleep(2);
}
```

**Advanced test:**
```javascript
import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Counter, Trend } from 'k6/metrics';

// Custom metrics
const checkoutCounter = new Counter('checkouts');
const checkoutDuration = new Trend('checkout_duration');

export const options = {
  scenarios: {
    // Scenario 1: Constant load
    constant_load: {
      executor: 'constant-vus',
      vus: 50,
      duration: '5m',
    },
    // Scenario 2: Ramping load
    ramping_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 0 },
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    'http_req_duration': ['p(99)<1500'],
    'checkout_duration': ['p(95)<3000'],
  },
};

export default function () {
  group('Browse products', function () {
    http.get('http://localhost:8000/products');
    sleep(1);
    http.get('http://localhost:8000/products/1');
    sleep(2);
  });

  group('Checkout', function () {
    const start = Date.now();

    http.post('http://localhost:8000/cart/add', { product_id: 1 });
    http.post('http://localhost:8000/checkout', {
      payment: { card: '4111111111111111' },
    });

    const duration = Date.now() - start;
    checkoutDuration.add(duration);
    checkoutCounter.add(1);

    sleep(1);
  });
}

export function handleSummary(data) {
  return {
    'summary.json': JSON.stringify(data),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}
```

**Run tests:**
```bash
# Basic run
k6 run script.js

# Run with custom settings
k6 run --vus 10 --duration 30s script.js

# Run with environment variables
k6 run -e BASE_URL=http://example.com script.js

# Output to file
k6 run --out json=results.json script.js
```

### Apache JMeter

**GUI mode:**
1. Add Thread Group
2. Add HTTP Request
3. Add Listeners (graphs, tables)
4. Configure and run

**Command line:**
```bash
# Run test
jmeter -n -t test-plan.jmx -l results.jtl

# Generate report
jmeter -g results.jtl -o report/
```

**JMeter test plan (XML):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2">
  <hashTree>
    <TestPlan>
      <ThreadGroup>
        <stringProp name="ThreadGroup.num_threads">100</stringProp>
        <stringProp name="ThreadGroup.ramp_time">60</stringProp>
        <stringProp name="ThreadGroup.duration">300</stringProp>
      </ThreadGroup>
      <HTTPSamplerProxy>
        <stringProp name="HTTPSampler.domain">localhost</stringProp>
        <stringProp name="HTTPSampler.port">8000</stringProp>
        <stringProp name="HTTPSampler.path">/api/users</stringProp>
        <stringProp name="HTTPSampler.method">GET</stringProp>
      </HTTPSamplerProxy>
    </TestPlan>
  </hashTree>
</jmeterTestPlan>
```

### Artillery

**Installation:**
```bash
npm install -g artillery
```

**Test configuration (load-test.yml):**
```yaml
config:
  target: 'http://localhost:8000'
  phases:
    - duration: 60
      arrivalRate: 10
      name: Warm up
    - duration: 300
      arrivalRate: 50
      name: Sustained load
    - duration: 120
      arrivalRate: 100
      name: Spike

scenarios:
  - name: Browse and purchase
    flow:
      - get:
          url: "/"
      - think: 2
      - get:
          url: "/products"
      - think: 3
      - post:
          url: "/cart/add"
          json:
            product_id: 1
            quantity: 2
      - think: 1
      - post:
          url: "/checkout"
          json:
            payment:
              card: "4111111111111111"
```

**Run test:**
```bash
# Run test
artillery run load-test.yml

# Quick test
artillery quick --count 10 --num 20 http://localhost:8000

# Generate report
artillery run --output report.json load-test.yml
artillery report report.json
```

## Benchmarking

### Python: pytest-benchmark

**Installation:**
```bash
pip install pytest-benchmark
```

**Usage:**
```python
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def test_fibonacci_performance(benchmark):
    result = benchmark(fibonacci, 20)
    assert result == 6765

def test_with_setup(benchmark):
    def setup():
        return [1, 2, 3, 4, 5], {}

    def process(items):
        return sum(items)

    result = benchmark.pedantic(process, setup=setup, rounds=100)
    assert result == 15

# Compare multiple implementations
def test_compare_sorting(benchmark):
    data = list(range(1000, 0, -1))

    @benchmark
    def sort_builtin():
        return sorted(data)

def test_compare_custom(benchmark):
    data = list(range(1000, 0, -1))

    @benchmark
    def sort_custom():
        # Custom sort implementation
        return bubble_sort(data)
```

**Run benchmarks:**
```bash
# Run all benchmarks
pytest --benchmark-only

# Compare with baseline
pytest --benchmark-compare

# Save baseline
pytest --benchmark-save=baseline

# Compare against baseline
pytest --benchmark-compare=baseline
```

### JavaScript: Benchmark.js

**Installation:**
```bash
npm install --save-dev benchmark
```

**Usage:**
```javascript
const Benchmark = require('benchmark');
const suite = new Benchmark.Suite();

// Add tests
suite
  .add('RegExp#test', function() {
    /o/.test('Hello World!');
  })
  .add('String#indexOf', function() {
    'Hello World!'.indexOf('o') > -1;
  })
  .add('String#match', function() {
    !!'Hello World!'.match(/o/);
  })
  .on('cycle', function(event) {
    console.log(String(event.target));
  })
  .on('complete', function() {
    console.log('Fastest is ' + this.filter('fastest').map('name'));
  })
  .run({ async: true });
```

## Profiling

### Python: cProfile

```python
import cProfile
import pstats

def main():
    # Your code here
    process_data()

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Top 10 functions
```

**Command line:**
```bash
python -m cProfile -o profile.stats script.py

# Analyze results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(10)"
```

### Python: line_profiler

```bash
pip install line_profiler
```

```python
@profile
def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total
```

```bash
kernprof -l -v script.py
```

### JavaScript: Chrome DevTools

1. Open DevTools (F12)
2. Go to Performance tab
3. Click Record
4. Perform actions
5. Stop recording
6. Analyze flame graph

### Node.js: clinic

```bash
npm install -g clinic

# CPU profiling
clinic doctor -- node app.js

# Memory profiling
clinic heapprofiler -- node app.js

# Event loop monitoring
clinic bubbleprof -- node app.js
```

## Performance Metrics

### Response Time Percentiles

```
p50 (median): 100ms  - Half of requests faster than this
p90: 200ms           - 90% of requests faster than this
p95: 300ms           - 95% of requests faster than this
p99: 500ms           - 99% of requests faster than this
```

**Why percentiles matter:**
- Average can be misleading
- p95/p99 show worst-case user experience
- Service Level Objectives (SLOs) often use percentiles

### Throughput

```
Requests per second (RPS)
Transactions per second (TPS)
```

**Example:**
```
1000 requests / 60 seconds = 16.67 RPS
```

### Error Rate

```
Error rate = (Failed requests / Total requests) × 100%
```

**Acceptable rates:**
- < 0.1% for critical services
- < 1% for non-critical services

### Resource Utilization

```
CPU usage: < 80% under load
Memory usage: < 80% of available
Network bandwidth: < 70% of capacity
Disk I/O: < 70% of capacity
```

## Best Practices

### 1. Define Performance Goals

```yaml
Requirements:
  - Response time: p95 < 500ms
  - Throughput: > 1000 RPS
  - Error rate: < 0.1%
  - Concurrent users: 10,000
  - CPU usage: < 70%
```

### 2. Test Realistic Scenarios

```python
# Bad: Unrealistic test
def test_single_user():
    response = requests.get('/api/users')

# Good: Realistic load
def test_concurrent_users():
    # Simulate 100 concurrent users
    # with realistic wait times and navigation
    pass
```

### 3. Use Production-Like Environment

```
Test environment should match:
- Similar hardware specs
- Same network latency
- Comparable data volume
- Identical software versions
```

### 4. Monitor During Tests

```python
# Monitor:
- Response times
- Error rates
- CPU/Memory usage
- Database connections
- Queue depths
- Cache hit rates
```

### 5. Analyze Bottlenecks

```
Common bottlenecks:
- Database queries (N+1 problem)
- External API calls
- Inefficient algorithms
- Memory leaks
- Blocking I/O
```

### 6. Test Incrementally

```bash
# Start small, increase gradually
Test 1: 10 users
Test 2: 50 users
Test 3: 100 users
Test 4: 500 users
Test 5: 1000 users
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Performance Tests

on:
  push:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install k6
        run: |
          wget https://github.com/grafana/k6/releases/download/v0.46.0/k6-v0.46.0-linux-amd64.tar.gz
          tar -xzf k6-v0.46.0-linux-amd64.tar.gz
          sudo mv k6 /usr/local/bin/

      - name: Run load test
        run: k6 run tests/load-test.js

      - name: Check thresholds
        run: |
          if grep -q "✗" k6-results.txt; then
            echo "Performance thresholds failed"
            exit 1
          fi
```

## Quick Reference

### Load Testing Tools Comparison

| Tool | Language | Best For | Learning Curve |
|------|----------|----------|----------------|
| Locust | Python | Python developers, flexible scenarios | Easy |
| k6 | JavaScript | Modern APIs, CI/CD | Medium |
| JMeter | GUI/XML | Enterprise, complex scenarios | Hard |
| Artillery | YAML | Quick setup, simple tests | Easy |
| Gatling | Scala | High performance, detailed reports | Medium |

### Common Commands

```bash
# Locust
locust -f locustfile.py --host=http://localhost --users 100 --spawn-rate 10

# k6
k6 run --vus 100 --duration 30s script.js

# Artillery
artillery quick --count 10 --num 100 http://localhost

# JMeter
jmeter -n -t test.jmx -l results.jtl

# Benchmark
pytest --benchmark-only
```

## Further Resources

- [Locust Documentation](https://docs.locust.io/)
- [k6 Documentation](https://k6.io/docs/)
- [JMeter Documentation](https://jmeter.apache.org/usermanual/index.html)
- [Artillery Documentation](https://artillery.io/docs/)
- [Performance Testing Guidance](https://martinfowler.com/articles/practical-test-pyramid.html)
