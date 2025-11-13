# Apache Kafka

Apache Kafka is a distributed event streaming platform capable of handling trillions of events a day. It's used for building real-time data pipelines and streaming applications, providing high-throughput, fault-tolerant, and scalable messaging.

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
- [Installation and Setup](#installation-and-setup)
- [Producers](#producers)
- [Consumers](#consumers)
- [Topics and Partitions](#topics-and-partitions)
- [Kafka with Node.js](#kafka-with-nodejs)
- [Best Practices](#best-practices)
- [Production Considerations](#production-considerations)

---

## Introduction

**Key Features:**
- High-throughput message streaming
- Fault-tolerant and durable
- Horizontal scalability
- Low latency (sub-millisecond)
- Replay capability
- Stream processing with Kafka Streams
- Connect framework for integrations

**Use Cases:**
- Event-driven architectures
- Log aggregation
- Real-time analytics
- Change Data Capture (CDC)
- Microservices communication
- Stream processing
- Message queuing
- Activity tracking

---

## Core Concepts

### Topics
Logical channels for messages, similar to database tables.

### Partitions
Topics are split into partitions for parallel processing.

### Producers
Applications that publish messages to topics.

### Consumers
Applications that subscribe to topics and process messages.

### Consumer Groups
Multiple consumers working together to process messages from a topic.

### Brokers
Kafka servers that store and serve data.

### Zookeeper/KRaft
Coordination service for managing Kafka cluster (KRaft is the newer alternative).

---

## Installation and Setup

### Docker Compose Setup

**docker-compose.yml:**
```yaml
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

```bash
docker-compose up -d
```

### CLI Commands

```bash
# Create topic
kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --partitions 3 \
  --replication-factor 1

# List topics
kafka-topics --list --bootstrap-server localhost:9092

# Describe topic
kafka-topics --describe \
  --bootstrap-server localhost:9092 \
  --topic my-topic

# Delete topic
kafka-topics --delete \
  --bootstrap-server localhost:9092 \
  --topic my-topic

# Produce messages
kafka-console-producer \
  --bootstrap-server localhost:9092 \
  --topic my-topic

# Consume messages
kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --from-beginning
```

---

## Producers

### Basic Producer Concept

```javascript
// Producer sends messages to topics
Message → Producer → Kafka Broker → Topic Partition
```

### Producer Configuration

```javascript
{
  'bootstrap.servers': 'localhost:9092',
  'client.id': 'my-producer',
  'acks': 'all',                    // Wait for all replicas
  'compression.type': 'gzip',       // Compress messages
  'max.in.flight.requests.per.connection': 5,
  'retries': 3,                     // Retry failed sends
  'batch.size': 16384,              // Batch size in bytes
  'linger.ms': 10                   // Wait time before sending batch
}
```

---

## Consumers

### Basic Consumer Concept

```javascript
// Consumers read messages from topics
Kafka Broker → Topic Partition → Consumer Group → Consumer
```

### Consumer Groups
- Multiple consumers in a group share the workload
- Each partition is consumed by only one consumer in a group
- Enables parallel processing and fault tolerance

### Consumer Configuration

```javascript
{
  'bootstrap.servers': 'localhost:9092',
  'group.id': 'my-consumer-group',
  'auto.offset.reset': 'earliest',  // Start from beginning if no offset
  'enable.auto.commit': false,      // Manual commit for reliability
  'max.poll.records': 500           // Max records per poll
}
```

---

## Topics and Partitions

### Topic Design

```javascript
// Good topic naming
user.events
order.created
payment.processed
notification.email.sent

// Partition strategy
// - More partitions = more parallelism
// - But more partitions = more overhead
// Start with: partitions = throughput (MB/s) / partition throughput (MB/s)
```

### Message Keys

```javascript
// Messages with same key go to same partition
// Ensures ordering for related events
{
  key: 'user:123',        // All events for user 123 in same partition
  value: { ... }
}
```

---

## Kafka with Node.js

### Installation

```bash
npm install kafkajs
```

### Producer Example

```javascript
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
  clientId: 'my-app',
  brokers: ['localhost:9092']
});

const producer = kafka.producer();

async function sendMessage() {
  await producer.connect();

  // Send single message
  await producer.send({
    topic: 'user-events',
    messages: [
      {
        key: 'user:123',
        value: JSON.stringify({
          userId: 123,
          action: 'login',
          timestamp: Date.now()
        })
      }
    ]
  });

  // Send multiple messages
  await producer.sendBatch({
    topicMessages: [
      {
        topic: 'user-events',
        messages: [
          { key: 'user:123', value: JSON.stringify({ action: 'login' }) },
          { key: 'user:124', value: JSON.stringify({ action: 'logout' }) }
        ]
      }
    ]
  });

  await producer.disconnect();
}

sendMessage().catch(console.error);
```

### Consumer Example

```javascript
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
  clientId: 'my-app',
  brokers: ['localhost:9092']
});

const consumer = kafka.consumer({
  groupId: 'my-consumer-group'
});

async function consume() {
  await consumer.connect();

  await consumer.subscribe({
    topic: 'user-events',
    fromBeginning: true
  });

  await consumer.run({
    eachMessage: async ({ topic, partition, message }) => {
      console.log({
        topic,
        partition,
        offset: message.offset,
        key: message.key?.toString(),
        value: message.value.toString()
      });

      // Process message
      const event = JSON.parse(message.value.toString());
      await processEvent(event);
    }
  });
}

async function processEvent(event) {
  console.log('Processing:', event);
  // Your business logic here
}

consume().catch(console.error);
```

### Batch Processing

```javascript
await consumer.run({
  eachBatch: async ({
    batch,
    resolveOffset,
    heartbeat,
    isRunning,
    isStale
  }) => {
    const messages = batch.messages;

    for (let message of messages) {
      if (!isRunning() || isStale()) break;

      await processMessage(message);

      // Commit offset for this message
      resolveOffset(message.offset);

      // Send heartbeat to keep consumer alive
      await heartbeat();
    }
  }
});
```

### Error Handling

```javascript
const consumer = kafka.consumer({
  groupId: 'my-group',
  retry: {
    retries: 8,
    initialRetryTime: 100,
    multiplier: 2
  }
});

consumer.on('consumer.crash', async (event) => {
  console.error('Consumer crashed:', event);
  // Implement restart logic
});

await consumer.run({
  eachMessage: async ({ topic, partition, message }) => {
    try {
      await processMessage(message);
    } catch (error) {
      console.error('Processing error:', error);

      // Dead letter queue
      await producer.send({
        topic: 'dead-letter-queue',
        messages: [{
          key: message.key,
          value: message.value,
          headers: {
            originalTopic: topic,
            error: error.message
          }
        }]
      });
    }
  }
});
```

### Express Integration

```javascript
const express = require('express');
const { Kafka } = require('kafkajs');

const app = express();
app.use(express.json());

const kafka = new Kafka({
  clientId: 'api-server',
  brokers: ['localhost:9092']
});

const producer = kafka.producer();

// Connect producer on startup
producer.connect();

// API endpoint to publish events
app.post('/api/events', async (req, res) => {
  try {
    const { userId, action, data } = req.body;

    await producer.send({
      topic: 'user-events',
      messages: [{
        key: `user:${userId}`,
        value: JSON.stringify({
          userId,
          action,
          data,
          timestamp: Date.now()
        })
      }]
    });

    res.json({ success: true, message: 'Event published' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  await producer.disconnect();
  process.exit(0);
});

app.listen(3000);
```

### Microservices Communication

**Order Service (Producer):**
```javascript
// order-service/producer.js
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
  clientId: 'order-service',
  brokers: ['localhost:9092']
});

const producer = kafka.producer();

async function createOrder(orderData) {
  await producer.connect();

  // Publish order created event
  await producer.send({
    topic: 'order.created',
    messages: [{
      key: `order:${orderData.id}`,
      value: JSON.stringify(orderData)
    }]
  });

  console.log('Order created event published');
}
```

**Inventory Service (Consumer):**
```javascript
// inventory-service/consumer.js
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
  clientId: 'inventory-service',
  brokers: ['localhost:9092']
});

const consumer = kafka.consumer({
  groupId: 'inventory-service-group'
});

async function start() {
  await consumer.connect();
  await consumer.subscribe({ topic: 'order.created' });

  await consumer.run({
    eachMessage: async ({ message }) => {
      const order = JSON.parse(message.value.toString());

      console.log('Processing order:', order.id);

      // Update inventory
      await updateInventory(order.items);

      // Publish inventory updated event
      await producer.send({
        topic: 'inventory.updated',
        messages: [{
          key: `order:${order.id}`,
          value: JSON.stringify({
            orderId: order.id,
            status: 'inventory_reserved'
          })
        }]
      });
    }
  });
}

start().catch(console.error);
```

---

## Best Practices

### 1. Message Design

```javascript
// Include metadata
{
  id: 'uuid',
  type: 'order.created',
  timestamp: 1234567890,
  version: '1.0',
  data: {
    orderId: 123,
    userId: 456,
    items: [...]
  }
}

// Use schema registry for validation
// Use Avro or Protobuf for efficient serialization
```

### 2. Error Handling

```javascript
// Implement retry logic
async function processWithRetry(message, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      await processMessage(message);
      return;
    } catch (error) {
      if (attempt === maxRetries) {
        // Send to dead letter queue
        await sendToDeadLetterQueue(message, error);
      } else {
        await sleep(Math.pow(2, attempt) * 1000); // Exponential backoff
      }
    }
  }
}
```

### 3. Consumer Groups

```javascript
// Use consumer groups for scalability
// Same group = load balancing
// Different groups = broadcast
const consumer = kafka.consumer({
  groupId: 'order-processing-group'
});
```

### 4. Idempotency

```javascript
// Ensure idempotent message processing
async function processMessage(message) {
  const messageId = message.headers.messageId;

  // Check if already processed
  const processed = await redis.get(`processed:${messageId}`);

  if (processed) {
    console.log('Message already processed');
    return;
  }

  // Process message
  await doWork(message);

  // Mark as processed
  await redis.set(`processed:${messageId}`, '1', 'EX', 86400);
}
```

### 5. Monitoring

```javascript
const producer = kafka.producer({
  // Enable metrics
  metricReporters: [
    {
      name: 'my-metrics',
      interval: 5000,
      async report(event) {
        console.log('Metrics:', event);
      }
    }
  ]
});

// Monitor lag
await admin.fetchOffsets({
  groupId: 'my-group',
  topics: ['my-topic']
});
```

---

## Production Considerations

### High Availability

```javascript
// Multiple brokers for redundancy
const kafka = new Kafka({
  clientId: 'my-app',
  brokers: [
    'kafka1:9092',
    'kafka2:9092',
    'kafka3:9092'
  ],
  retry: {
    retries: 10,
    initialRetryTime: 300,
    multiplier: 2
  }
});

// Replication factor for topics
await admin.createTopics({
  topics: [{
    topic: 'critical-events',
    numPartitions: 6,
    replicationFactor: 3  // Data replicated on 3 brokers
  }]
});
```

### Performance Tuning

```javascript
// Producer optimization
const producer = kafka.producer({
  idempotent: true,          // Exactly-once semantics
  maxInFlightRequests: 5,
  compression: CompressionTypes.GZIP,
  batch: {
    size: 16384,
    lingerMs: 10
  }
});

// Consumer optimization
const consumer = kafka.consumer({
  groupId: 'my-group',
  sessionTimeout: 30000,
  heartbeatInterval: 3000,
  maxBytesPerPartition: 1048576,
  maxWaitTimeInMs: 5000
});
```

### Security

```javascript
const kafka = new Kafka({
  clientId: 'secure-app',
  brokers: ['kafka:9093'],
  ssl: true,
  sasl: {
    mechanism: 'plain',
    username: 'my-username',
    password: 'my-password'
  }
});
```

---

## Resources

**Official Documentation:**
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [KafkaJS Documentation](https://kafka.js.org/)
- [Confluent Documentation](https://docs.confluent.io/)

**Tools:**
- [Kafka UI](https://github.com/provectus/kafka-ui) - Web UI for Kafka
- [Kafdrop](https://github.com/obsidiandynamics/kafdrop) - Kafka Web UI
- [Kafka Tool](https://www.kafkatool.com/) - GUI

**Learning:**
- [Kafka: The Definitive Guide](https://www.confluent.io/resources/kafka-the-definitive-guide/)
- [Confluent Platform](https://www.confluent.io/)
- [Apache Kafka Quickstart](https://kafka.apache.org/quickstart)
