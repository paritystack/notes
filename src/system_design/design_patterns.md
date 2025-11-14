# Design Patterns

## Overview

Common architectural patterns for building scalable, reliable, and maintainable distributed systems.

## Communication Patterns

### API Gateway

Single entry point for clients:

```
Mobile App  ─┐
Web App     ─┼─→ API Gateway ─┬─→ User Service
Desktop App ─┘                ├─→ Order Service
                              ├─→ Payment Service
                              └─→ Notification Service
```

**Responsibilities**:
- Routing requests
- Authentication/authorization
- Rate limiting
- Request/response transformation
- Logging and monitoring

```python
# API Gateway example
class APIGateway:
    def __init__(self):
        self.services = {
            '/users': UserService(),
            '/orders': OrderService(),
            '/payments': PaymentService()
        }

    def handle_request(self, request):
        # Authentication
        if not self.authenticate(request):
            return 401, "Unauthorized"

        # Rate limiting
        if not self.check_rate_limit(request.user_id):
            return 429, "Too Many Requests"

        # Route to service
        service = self.find_service(request.path)
        response = service.handle(request)

        # Transform response
        return self.transform_response(response)
```

**Pros**: Single entry point, centralized logic
**Cons**: Single point of failure, can become bottleneck

### Backend for Frontend (BFF)

Separate API gateway per client type:

```
Mobile App → Mobile BFF ─┐
                         ├─→ Microservices
Web App → Web BFF ───────┘
```

```python
# Mobile BFF - lightweight responses
class MobileBFF:
    def get_user_dashboard(self, user_id):
        user = user_service.get(user_id)
        orders = order_service.get_recent(user_id, limit=5)

        return {
            'name': user.name,
            'recent_orders': [
                {'id': o.id, 'total': o.total}
                for o in orders
            ]
        }

# Web BFF - detailed responses
class WebBFF:
    def get_user_dashboard(self, user_id):
        user = user_service.get(user_id)
        orders = order_service.get_all(user_id)
        analytics = analytics_service.get(user_id)

        return {
            'user': user.to_dict(),
            'orders': [o.to_dict() for o in orders],
            'analytics': analytics.to_dict()
        }
```

**Pros**: Optimized per client, team ownership
**Cons**: Code duplication, more services to maintain

### Service Mesh

Infrastructure layer for service-to-service communication:

```
Service A ←→ Sidecar Proxy ←→ Service Mesh ←→ Sidecar Proxy ←→ Service B
```

**Features**:
- Load balancing
- Service discovery
- Encryption (mTLS)
- Observability
- Circuit breaking

**Popular**: Istio, Linkerd, Consul

## Resilience Patterns

### Circuit Breaker

Prevent cascading failures:

```
Closed (Normal) → Failures → Open (Reject immediately) → Timer → Half-Open (Try again)
                                                                         ↓
                                                                   Success → Closed
                                                                   Failure → Open
```

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = None

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)

            # Success
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = "open"

            raise e

# Usage
breaker = CircuitBreaker(failure_threshold=5, timeout=60)

try:
    result = breaker.call(external_service.fetch_data, user_id)
except CircuitBreakerOpenError:
    # Fallback to cache or default value
    result = get_cached_data(user_id)
```

### Retry with Exponential Backoff

Retry failed requests with increasing delays:

```python
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e

            # Exponential backoff: 1s, 2s, 4s, 8s...
            delay = base_delay * (2 ** attempt)

            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, delay * 0.1)

            time.sleep(delay + jitter)

# Usage
result = retry_with_backoff(
    lambda: api.call_external_service(),
    max_retries=3,
    base_delay=1
)
```

### Bulkhead

Isolate resources to prevent total failure:

```
Thread Pool A (50 threads) → Service A
Thread Pool B (50 threads) → Service B
Thread Pool C (50 threads) → Service C

If Service A fails, pools B and C unaffected
```

```python
from concurrent.futures import ThreadPoolExecutor

class Bulkhead:
    def __init__(self):
        self.pools = {
            'user_service': ThreadPoolExecutor(max_workers=20),
            'order_service': ThreadPoolExecutor(max_workers=30),
            'payment_service': ThreadPoolExecutor(max_workers=10)
        }

    def execute(self, service_name, func, *args):
        pool = self.pools[service_name]
        future = pool.submit(func, *args)
        return future.result(timeout=5)

# Usage
bulkhead = Bulkhead()

try:
    user = bulkhead.execute('user_service', get_user, user_id)
    orders = bulkhead.execute('order_service', get_orders, user_id)
except TimeoutError:
    # Handle timeout
    pass
```

### Timeout

Set maximum wait time:

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(func, timeout_seconds):
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        result = func()
        signal.alarm(0)  # Cancel timeout
        return result
    except TimeoutError:
        # Handle timeout
        return None

# Or with threading
import threading

def with_timeout_thread(func, args=(), timeout=5):
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Timeout occurred
        return None
    elif exception[0]:
        raise exception[0]
    else:
        return result[0]
```

### Fallback

Provide alternative when primary fails:

```python
def get_user_profile(user_id):
    try:
        # Try primary source
        return primary_db.get_user(user_id)
    except DatabaseError:
        try:
            # Fallback to cache
            return cache.get(f"user:{user_id}")
        except CacheError:
            # Fallback to default
            return {
                'id': user_id,
                'name': 'Unknown User',
                'status': 'unavailable'
            }
```

## Data Patterns

### Database per Service

Each microservice owns its database:

```
User Service → User DB
Order Service → Order DB
Payment Service → Payment DB
```

**Pros**: Service independence, technology choice
**Cons**: Distributed transactions, data duplication

```python
# User Service
class UserService:
    def __init__(self):
        self.db = UserDatabase()

    def create_user(self, user_data):
        return self.db.insert(user_data)

# Order Service
class OrderService:
    def __init__(self):
        self.db = OrderDatabase()

    def create_order(self, order_data):
        # Need user data? Call User Service API
        user = user_service_client.get_user(order_data['user_id'])
        return self.db.insert(order_data)
```

### Shared Database

Multiple services share one database:

```
User Service ─┐
              ├─→ Shared DB
Order Service ─┘
```

**Pros**: Simple, easy transactions
**Cons**: Tight coupling, schema conflicts, scaling issues

**Avoid**: Only for very small applications

### CQRS (Command Query Responsibility Segregation)

Separate read and write models:

```
Commands (Write) → Write Model → Event Store
                                      ↓
                                  Events
                                      ↓
                                 Read Model ← Queries (Read)
```

```python
# Write Model - optimized for writes
class WriteModel:
    def create_order(self, order_data):
        order = Order(**order_data)
        self.validate(order)

        # Store as event
        event = {
            'type': 'OrderCreated',
            'data': order_data,
            'timestamp': now()
        }
        event_store.append(event)

        # Publish event
        event_bus.publish(event)

# Read Model - optimized for reads
class ReadModel:
    def __init__(self):
        self.db = ReadDatabase()  # Denormalized, fast queries

    def get_order_summary(self, user_id):
        # Pre-computed summary
        return self.db.query(
            "SELECT * FROM order_summary WHERE user_id = ?",
            user_id
        )

# Event handler updates read model
@event_handler('OrderCreated')
def update_read_model(event):
    order = event['data']

    # Update denormalized views
    read_db.execute("""
        INSERT INTO order_summary (user_id, order_count, total_spent)
        VALUES (?, 1, ?)
        ON CONFLICT (user_id) DO UPDATE
            SET order_count = order_count + 1,
                total_spent = total_spent + ?
    """, order['user_id'], order['total'], order['total'])
```

### Event Sourcing

Store all changes as immutable events:

```
Event 1: OrderCreated(id=123, items=[...], total=99.99)
Event 2: OrderPaid(id=123, payment_id=456)
Event 3: OrderShipped(id=123, tracking=789)

Current State = Replay events
```

```python
class OrderEventStore:
    def __init__(self):
        self.events = []

    def append(self, event):
        event['version'] = len(self.events)
        event['timestamp'] = now()
        self.events.append(event)

    def get_state(self, order_id):
        # Rebuild state from events
        order = None

        for event in self.events:
            if event.get('order_id') != order_id:
                continue

            if event['type'] == 'OrderCreated':
                order = {
                    'id': event['order_id'],
                    'items': event['items'],
                    'total': event['total'],
                    'status': 'pending'
                }
            elif event['type'] == 'OrderPaid':
                order['status'] = 'paid'
                order['payment_id'] = event['payment_id']
            elif event['type'] == 'OrderShipped':
                order['status'] = 'shipped'
                order['tracking'] = event['tracking']

        return order
```

**Pros**: Full audit trail, time travel, debugging
**Cons**: Complexity, storage costs, eventual consistency

### Saga Pattern

Manage distributed transactions:

```
Step 1: Reserve Inventory → Success
Step 2: Charge Payment → Success
Step 3: Create Shipment → Failure
  ↓
Compensate: Refund Payment
Compensate: Release Inventory
```

**Orchestration** (coordinator):
```python
class OrderSaga:
    def __init__(self):
        self.completed_steps = []

    def execute(self, order):
        try:
            # Step 1
            inventory = inventory_service.reserve(order.items)
            self.completed_steps.append(('inventory', inventory))

            # Step 2
            payment = payment_service.charge(order.total)
            self.completed_steps.append(('payment', payment))

            # Step 3
            shipment = shipping_service.create(order)
            self.completed_steps.append(('shipment', shipment))

            return success

        except Exception as e:
            # Compensate in reverse order
            self.compensate()
            raise e

    def compensate(self):
        for step_type, step_data in reversed(self.completed_steps):
            if step_type == 'shipment':
                shipping_service.cancel(step_data)
            elif step_type == 'payment':
                payment_service.refund(step_data)
            elif step_type == 'inventory':
                inventory_service.release(step_data)
```

**Choreography** (event-driven):
```python
# Each service listens to events and reacts

@event_handler('OrderCreated')
def reserve_inventory(event):
    try:
        inventory_service.reserve(event['items'])
        publish('InventoryReserved', event['order_id'])
    except Exception:
        publish('InventoryReservationFailed', event['order_id'])

@event_handler('InventoryReserved')
def charge_payment(event):
    try:
        payment_service.charge(event['total'])
        publish('PaymentCharged', event['order_id'])
    except Exception:
        publish('PaymentFailed', event['order_id'])
        # Trigger compensation
        publish('ReleaseInventory', event['order_id'])

@event_handler('PaymentCharged')
def create_shipment(event):
    try:
        shipping_service.create(event['order_id'])
        publish('OrderCompleted', event['order_id'])
    except Exception:
        publish('ShipmentFailed', event['order_id'])
        # Trigger compensation
        publish('RefundPayment', event['order_id'])
        publish('ReleaseInventory', event['order_id'])
```

## Caching Patterns

### Cache-Aside (Lazy Loading)

Application manages cache:

```python
def get_user(user_id):
    # Check cache
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached

    # Load from DB
    user = db.get_user(user_id)

    # Store in cache
    cache.set(f"user:{user_id}", user, ttl=3600)

    return user
```

### Read-Through

Cache manages database loading:

```python
class ReadThroughCache:
    def get(self, key, loader_func):
        cached = self.cache.get(key)
        if cached:
            return cached

        # Cache loads from DB automatically
        value = loader_func()
        self.cache.set(key, value)
        return value

# Usage
user = cache.get(
    f"user:{user_id}",
    lambda: db.get_user(user_id)
)
```

### Write-Through

Write to cache and DB simultaneously:

```python
def update_user(user_id, data):
    # Update DB
    db.update_user(user_id, data)

    # Update cache
    cache.set(f"user:{user_id}", data)
```

### Write-Behind (Write-Back)

Write to cache, async to DB:

```python
def update_user(user_id, data):
    # Update cache immediately
    cache.set(f"user:{user_id}", data)

    # Queue DB update
    queue.send({
        'action': 'update_user',
        'user_id': user_id,
        'data': data
    })

    return success  # Fast response
```

## Scalability Patterns

### Load Balancer

Distribute requests across servers:

```
       Load Balancer
      /      |      \
  Server1 Server2 Server3
```

**Algorithms**:
```python
# Round Robin
class RoundRobinLB:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0

    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# Least Connections
class LeastConnectionsLB:
    def __init__(self, servers):
        self.servers = servers
        self.connections = {s: 0 for s in servers}

    def get_server(self):
        return min(self.connections, key=self.connections.get)

    def on_request_complete(self, server):
        self.connections[server] -= 1

# Weighted
class WeightedLB:
    def __init__(self, servers_with_weights):
        self.servers = []
        for server, weight in servers_with_weights:
            self.servers.extend([server] * weight)

    def get_server(self):
        return random.choice(self.servers)
```

### Horizontal Scaling (Scale Out)

Add more servers:

```python
# Stateless service - easy to scale
class StatelessAPI:
    def handle_request(self, request):
        # No local state, can run on any server
        data = db.query(request.query)
        return data

# Deploy multiple instances
instances = [
    StatelessAPI(),
    StatelessAPI(),
    StatelessAPI()
]

# Load balancer distributes requests
lb = LoadBalancer(instances)
```

### Auto-Scaling

Automatically adjust capacity:

```python
class AutoScaler:
    def __init__(self, min_instances=2, max_instances=10):
        self.min = min_instances
        self.max = max_instances
        self.instances = []

    def check_metrics(self):
        cpu_usage = get_average_cpu()
        request_rate = get_request_rate()

        if cpu_usage > 80 and len(self.instances) < self.max:
            self.scale_up()
        elif cpu_usage < 20 and len(self.instances) > self.min:
            self.scale_down()

    def scale_up(self):
        new_instance = create_instance()
        self.instances.append(new_instance)
        lb.add_server(new_instance)

    def scale_down(self):
        instance = self.instances.pop()
        lb.remove_server(instance)
        instance.graceful_shutdown()
```

## Observability Patterns

### Distributed Tracing

Track requests across services:

```python
import uuid

class Tracer:
    def start_trace(self, operation_name):
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        return {
            'trace_id': trace_id,
            'span_id': span_id,
            'operation': operation_name,
            'start_time': time.time()
        }

    def inject(self, span, request):
        # Add trace context to outgoing request
        request.headers['X-Trace-ID'] = span['trace_id']
        request.headers['X-Span-ID'] = span['span_id']

    def extract(self, request):
        # Extract trace context from incoming request
        return {
            'trace_id': request.headers.get('X-Trace-ID'),
            'parent_span_id': request.headers.get('X-Span-ID')
        }

# Service A
def handle_request(request):
    span = tracer.start_trace('handle_request')

    # Call Service B
    b_request = prepare_request()
    tracer.inject(span, b_request)
    response = service_b.call(b_request)

    tracer.finish(span)
```

### Health Check

Expose service health:

```python
@app.get('/health')
def health_check():
    checks = {
        'database': check_database(),
        'cache': check_cache(),
        'external_api': check_external_api()
    }

    all_healthy = all(checks.values())

    return {
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks
    }, 200 if all_healthy else 503

def check_database():
    try:
        db.execute("SELECT 1")
        return True
    except Exception:
        return False
```

### Metrics

Collect system metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters - always increasing
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Histograms - measure distributions
request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Gauges - current value
active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

# Usage
@app.route('/api/users')
def get_users():
    with request_duration.labels('GET', '/api/users').time():
        users = db.query("SELECT * FROM users")

    request_count.labels('GET', '/api/users', 200).inc()

    return users
```

## ELI10

Design patterns are like recipes for building systems:

- **Circuit Breaker**: Like a fuse - stops trying if too many failures
- **Retry**: Try again if it doesn't work (like knocking on a door)
- **API Gateway**: Front door where everyone enters
- **CQRS**: Separate reading (looking at menu) from writing (placing order)
- **Event Sourcing**: Keep diary of everything that happened
- **Saga**: Multi-step process with undo if something fails
- **Load Balancer**: Traffic cop directing cars to different lanes

Use the right pattern for the right problem!

## Further Resources

- [Microservices Patterns by Chris Richardson](https://microservices.io/patterns/)
- [Cloud Design Patterns (Microsoft)](https://learn.microsoft.com/en-us/azure/architecture/patterns/)
- [Designing Data-Intensive Applications](https://dataintensive.app/)
- [Pattern: Saga](https://microservices.io/patterns/data/saga.html)
- [Martin Fowler's Blog](https://martinfowler.com/)
