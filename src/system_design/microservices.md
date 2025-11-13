# Microservices Architecture

Microservices is an architectural style that structures an application as a collection of loosely coupled, independently deployable services. Each service is self-contained, implements a specific business capability, and communicates with other services through well-defined APIs.

## Table of Contents
- [Introduction](#introduction)
- [Core Principles](#core-principles)
- [Service Design](#service-design)
- [Communication Patterns](#communication-patterns)
- [Service Discovery](#service-discovery)
- [API Gateway](#api-gateway)
- [Data Management](#data-management)
- [Deployment and DevOps](#deployment-and-devops)
- [Best Practices](#best-practices)
- [Challenges and Solutions](#challenges-and-solutions)

---

## Introduction

**What are Microservices?**
Microservices break down a large application into smaller, independent services that:
- Run in their own processes
- Communicate via lightweight protocols (HTTP, message queues)
- Can be deployed independently
- Can use different technologies
- Are organized around business capabilities

**Benefits:**
- Independent deployment and scaling
- Technology diversity
- Fault isolation
- Team autonomy
- Faster development cycles
- Easier to understand and maintain small services

**Challenges:**
- Distributed system complexity
- Network latency and failures
- Data consistency
- Testing complexity
- Operational overhead
- Service coordination

---

## Core Principles

### 1. Single Responsibility
Each service handles one business capability.

```
❌ Monolith: One service handles users, orders, payments, inventory
✅ Microservices:
   - User Service: Authentication, profiles
   - Order Service: Order management
   - Payment Service: Payment processing
   - Inventory Service: Stock management
```

### 2. Decentralized Data Management
Each service owns its data store.

```javascript
// Each service has its own database
User Service → Users DB (PostgreSQL)
Order Service → Orders DB (MongoDB)
Inventory Service → Inventory DB (MySQL)
```

### 3. Smart Endpoints, Dumb Pipes
Services are intelligent; communication is simple.

```javascript
// Services handle business logic
// Communication uses simple protocols (HTTP, AMQP)
```

### 4. Design for Failure
Expect services to fail; build resilience.

```javascript
// Circuit breakers
// Retries
// Fallbacks
// Timeouts
```

---

## Service Design

### Domain-Driven Design

```javascript
// Bounded Contexts
Order Context {
  - Order
  - OrderItem
  - OrderStatus
}

User Context {
  - User
  - Profile
  - Authentication
}

Payment Context {
  - Payment
  - Transaction
  - PaymentMethod
}
```

### Service Size

```javascript
// Small enough to:
// - Be maintained by a small team (2-pizza team)
// - Be rewritten in 2-4 weeks
// - Have a clear purpose

// Large enough to:
// - Provide business value
// - Minimize inter-service communication
// - Have a clear domain boundary
```

### Example Service Structure

```
order-service/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   └── controllers/
│   ├── domain/
│   │   ├── models/
│   │   └── services/
│   ├── infrastructure/
│   │   ├── database/
│   │   └── messaging/
│   ├── config/
│   └── main.ts
├── tests/
├── Dockerfile
├── package.json
└── README.md
```

---

## Communication Patterns

### Synchronous Communication (REST/HTTP)

**Example: Order Service calling User Service**

```javascript
// order-service/userClient.js
const axios = require('axios');

class UserServiceClient {
  constructor(baseURL) {
    this.client = axios.create({
      baseURL: baseURL || process.env.USER_SERVICE_URL,
      timeout: 5000
    });
  }

  async getUser(userId) {
    try {
      const response = await this.client.get(`/users/${userId}`);
      return response.data;
    } catch (error) {
      if (error.code === 'ECONNABORTED') {
        throw new Error('User service timeout');
      }
      throw error;
    }
  }
}

// Usage in order service
async function createOrder(orderData) {
  const userClient = new UserServiceClient();
  const user = await userClient.getUser(orderData.userId);

  if (!user) {
    throw new Error('User not found');
  }

  // Create order logic...
}
```

### Asynchronous Communication (Message Queues)

**Example: Event-Driven Communication**

```javascript
// order-service/publisher.js
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
  clientId: 'order-service',
  brokers: ['kafka:9092']
});

const producer = kafka.producer();

async function publishOrderCreated(order) {
  await producer.send({
    topic: 'order.created',
    messages: [{
      key: `order:${order.id}`,
      value: JSON.stringify({
        orderId: order.id,
        userId: order.userId,
        items: order.items,
        total: order.total,
        timestamp: Date.now()
      })
    }]
  });
}

// inventory-service/consumer.js
const consumer = kafka.consumer({
  groupId: 'inventory-service'
});

async function start() {
  await consumer.subscribe({ topic: 'order.created' });

  await consumer.run({
    eachMessage: async ({ message }) => {
      const order = JSON.parse(message.value.toString());

      console.log('Reserving inventory for order:', order.orderId);

      await reserveInventory(order.items);

      // Publish inventory.reserved event
      await publishInventoryReserved(order.orderId);
    }
  });
}
```

### API Composition Pattern

```javascript
// api-gateway/orderComposer.js
class OrderComposer {
  constructor(userService, orderService, inventoryService) {
    this.userService = userService;
    this.orderService = orderService;
    this.inventoryService = inventoryService;
  }

  async getOrderDetails(orderId) {
    // Parallel requests
    const [order, user, inventory] = await Promise.all([
      this.orderService.getOrder(orderId),
      this.userService.getUser(order.userId),
      this.inventoryService.checkAvailability(order.items)
    ]);

    return {
      order,
      user: {
        id: user.id,
        name: user.name,
        email: user.email
      },
      inventory
    };
  }
}
```

---

## Service Discovery

### Client-Side Discovery

```javascript
// service-registry.js
class ServiceRegistry {
  constructor() {
    this.services = new Map();
  }

  register(serviceName, instance) {
    if (!this.services.has(serviceName)) {
      this.services.set(serviceName, []);
    }
    this.services.get(serviceName).push(instance);
  }

  discover(serviceName) {
    const instances = this.services.get(serviceName) || [];
    if (instances.length === 0) {
      throw new Error(`No instances available for ${serviceName}`);
    }
    // Round-robin load balancing
    return instances[Math.floor(Math.random() * instances.length)];
  }
}

// Usage
const registry = new ServiceRegistry();
registry.register('user-service', { host: 'localhost', port: 3001 });
registry.register('user-service', { host: 'localhost', port: 3002 });

const instance = registry.discover('user-service');
```

### Consul Integration

```javascript
const Consul = require('consul');

const consul = new Consul({
  host: 'consul-server',
  port: 8500
});

// Register service
async function registerService() {
  await consul.agent.service.register({
    name: 'order-service',
    id: `order-service-${process.env.INSTANCE_ID}`,
    address: process.env.SERVICE_HOST,
    port: parseInt(process.env.SERVICE_PORT),
    check: {
      http: `http://${process.env.SERVICE_HOST}:${process.env.SERVICE_PORT}/health`,
      interval: '10s'
    }
  });
}

// Discover service
async function discoverService(serviceName) {
  const result = await consul.health.service({
    service: serviceName,
    passing: true
  });

  const instances = result.map(item => ({
    address: item.Service.Address,
    port: item.Service.Port
  }));

  return instances;
}
```

---

## API Gateway

### Basic API Gateway

```javascript
const express = require('express');
const proxy = require('express-http-proxy');

const app = express();

// Service URLs
const USER_SERVICE = process.env.USER_SERVICE_URL;
const ORDER_SERVICE = process.env.ORDER_SERVICE_URL;
const PRODUCT_SERVICE = process.env.PRODUCT_SERVICE_URL;

// Authentication middleware
app.use(async (req, res, next) => {
  const token = req.headers.authorization;

  if (!token) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  try {
    const user = await verifyToken(token);
    req.user = user;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
});

// Rate limiting
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
});

app.use(limiter);

// Route to services
app.use('/api/users', proxy(USER_SERVICE));
app.use('/api/orders', proxy(ORDER_SERVICE));
app.use('/api/products', proxy(PRODUCT_SERVICE));

// Aggregation endpoint
app.get('/api/dashboard', async (req, res) => {
  try {
    const [user, orders, recommendations] = await Promise.all([
      axios.get(`${USER_SERVICE}/users/${req.user.id}`),
      axios.get(`${ORDER_SERVICE}/users/${req.user.id}/orders`),
      axios.get(`${PRODUCT_SERVICE}/recommendations/${req.user.id}`)
    ]);

    res.json({
      user: user.data,
      recentOrders: orders.data,
      recommendations: recommendations.data
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to load dashboard' });
  }
});

app.listen(3000);
```

---

## Data Management

### Database Per Service

```javascript
// Each service has its own database
services/
├── user-service/
│   └── database: PostgreSQL
├── order-service/
│   └── database: MongoDB
└── inventory-service/
    └── database: MySQL
```

### Saga Pattern (Distributed Transactions)

**Choreography-Based Saga:**

```javascript
// Order Service
async function createOrder(orderData) {
  const order = await Order.create({
    ...orderData,
    status: 'PENDING'
  });

  // Publish event
  await publishEvent('order.created', order);

  return order;
}

// Inventory Service
consumer.on('order.created', async (order) => {
  try {
    await reserveInventory(order.items);
    await publishEvent('inventory.reserved', { orderId: order.id });
  } catch (error) {
    await publishEvent('inventory.failed', {
      orderId: order.id,
      error: error.message
    });
  }
});

// Payment Service
consumer.on('inventory.reserved', async ({ orderId }) => {
  try {
    await processPayment(orderId);
    await publishEvent('payment.completed', { orderId });
  } catch (error) {
    await publishEvent('payment.failed', { orderId, error: error.message });
  }
});

// Order Service - Handle success/failure
consumer.on('payment.completed', async ({ orderId }) => {
  await Order.update({ id: orderId }, { status: 'CONFIRMED' });
});

consumer.on('payment.failed', async ({ orderId }) => {
  await Order.update({ id: orderId }, { status: 'CANCELLED' });
  await publishEvent('order.cancelled', { orderId });
});

// Inventory Service - Compensating transaction
consumer.on('order.cancelled', async ({ orderId }) => {
  await releaseInventory(orderId);
});
```

### CQRS (Command Query Responsibility Segregation)

```javascript
// Write Model (Commands)
class OrderWriteService {
  async createOrder(command) {
    const order = await Order.create(command);

    // Publish event
    await eventBus.publish('OrderCreated', {
      orderId: order.id,
      userId: order.userId,
      items: order.items
    });

    return order.id;
  }
}

// Read Model (Queries)
class OrderReadService {
  constructor(readDatabase) {
    this.db = readDatabase;
  }

  async getOrderById(orderId) {
    return await this.db.orders.findOne({ id: orderId });
  }

  async getOrdersByUser(userId) {
    return await this.db.orders.find({ userId });
  }
}

// Event Handler (updates read model)
eventBus.on('OrderCreated', async (event) => {
  await readDatabase.orders.insert({
    id: event.orderId,
    userId: event.userId,
    items: event.items,
    createdAt: new Date()
  });
});
```

---

## Deployment and DevOps

### Docker Compose

```yaml
version: '3.8'

services:
  api-gateway:
    build: ./api-gateway
    ports:
      - "3000:3000"
    environment:
      USER_SERVICE_URL: http://user-service:3001
      ORDER_SERVICE_URL: http://order-service:3002
    depends_on:
      - user-service
      - order-service

  user-service:
    build: ./user-service
    environment:
      DATABASE_URL: postgresql://postgres:password@user-db:5432/users
    depends_on:
      - user-db

  order-service:
    build: ./order-service
    environment:
      DATABASE_URL: mongodb://order-db:27017/orders
      KAFKA_BROKERS: kafka:9092
    depends_on:
      - order-db
      - kafka

  user-db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password

  order-db:
    image: mongo:6

  kafka:
    image: confluentinc/cp-kafka:latest
```

### Kubernetes

```yaml
# order-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
    spec:
      containers:
      - name: order-service
        image: myregistry/order-service:1.0.0
        ports:
        - containerPort: 3000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: order-service-secrets
              key: database-url
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
  ports:
  - port: 80
    targetPort: 3000
  type: ClusterIP
```

---

## Best Practices

### 1. Circuit Breaker Pattern

```javascript
const CircuitBreaker = require('opossum');

const options = {
  timeout: 3000,
  errorThresholdPercentage: 50,
  resetTimeout: 30000
};

const breaker = new CircuitBreaker(callExternalService, options);

breaker.fallback(() => ({ fallback: 'value' }));

breaker.on('open', () => console.log('Circuit opened'));
breaker.on('halfOpen', () => console.log('Circuit half-open'));
breaker.on('close', () => console.log('Circuit closed'));

async function callExternalService() {
  const response = await axios.get('http://external-service/api');
  return response.data;
}

// Usage
try {
  const result = await breaker.fire();
  console.log(result);
} catch (error) {
  console.error('Service call failed');
}
```

### 2. Health Checks

```javascript
const express = require('express');
const app = express();

app.get('/health', async (req, res) => {
  const health = {
    uptime: process.uptime(),
    message: 'OK',
    timestamp: Date.now()
  };

  try {
    // Check database connection
    await database.ping();
    health.database = 'connected';
  } catch (error) {
    health.database = 'disconnected';
    health.message = 'Degraded';
    return res.status(503).json(health);
  }

  res.json(health);
});

app.get('/ready', async (req, res) => {
  try {
    // Check if service is ready to accept traffic
    await database.ping();
    await cache.ping();

    res.json({ status: 'ready' });
  } catch (error) {
    res.status(503).json({ status: 'not ready' });
  }
});
```

### 3. Distributed Tracing

```javascript
const { trace } = require('@opentelemetry/api');
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');

const provider = new NodeTracerProvider();
provider.register();

const tracer = trace.getTracer('order-service');

async function createOrder(orderData) {
  const span = tracer.startSpan('createOrder');

  try {
    // Add attributes
    span.setAttribute('user.id', orderData.userId);
    span.setAttribute('order.total', orderData.total);

    // Business logic
    const order = await Order.create(orderData);

    span.setStatus({ code: 0 }); // OK
    return order;
  } catch (error) {
    span.setStatus({
      code: 2, // ERROR
      message: error.message
    });
    throw error;
  } finally {
    span.end();
  }
}
```

### 4. Logging

```javascript
const winston = require('winston');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  defaultMeta: {
    service: 'order-service',
    version: '1.0.0'
  },
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Structured logging
logger.info('Order created', {
  orderId: order.id,
  userId: order.userId,
  total: order.total,
  timestamp: Date.now()
});
```

---

## Challenges and Solutions

### Challenge 1: Data Consistency

**Solution:** Use eventual consistency with event-driven architecture

```javascript
// Use events to propagate data changes
await publishEvent('user.updated', { userId, email: newEmail });

// Other services listen and update their local views
consumer.on('user.updated', async (event) => {
  await updateLocalUserCache(event.userId, event.email);
});
```

### Challenge 2: Service Communication Failures

**Solution:** Implement retry logic with exponential backoff

```javascript
async function callServiceWithRetry(fn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await sleep(Math.pow(2, i) * 1000);
    }
  }
}
```

### Challenge 3: Testing

**Solution:** Use contract testing and integration tests

```javascript
// Contract test (using Pact)
const { Pact } = require('@pact-foundation/pact');

const provider = new Pact({
  consumer: 'order-service',
  provider: 'user-service'
});

describe('User Service Contract', () => {
  it('should get user by ID', async () => {
    await provider.addInteraction({
      state: 'user 123 exists',
      uponReceiving: 'a request for user 123',
      withRequest: {
        method: 'GET',
        path: '/users/123'
      },
      willRespondWith: {
        status: 200,
        body: { id: 123, name: 'John' }
      }
    });

    // Test your client code
    const user = await userClient.getUser(123);
    expect(user.id).toBe(123);
  });
});
```

---

## Resources

**Books:**
- Building Microservices by Sam Newman
- Microservices Patterns by Chris Richardson
- Release It! by Michael Nygard

**Frameworks:**
- [Express.js](https://expressjs.com/)
- [NestJS](https://nestjs.com/)
- [Spring Boot](https://spring.io/projects/spring-boot)
- [Go Kit](https://gokit.io/)

**Tools:**
- [Kubernetes](https://kubernetes.io/)
- [Istio](https://istio.io/) - Service Mesh
- [Consul](https://www.consul.io/) - Service Discovery
- [Jaeger](https://www.jaegertracing.io/) - Distributed Tracing

**Learning:**
- [Microservices.io](https://microservices.io/)
- [Martin Fowler's Blog](https://martinfowler.com/articles/microservices.html)
