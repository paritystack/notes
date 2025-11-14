# Message Queues

## Overview

Message queues enable asynchronous communication between services, decoupling producers from consumers and providing reliability and scalability.

## Why Message Queues?

### Without Queue (Synchronous)

```
Client → API → Process → Database → Response
         (waits for everything to complete)
```

**Problems**:
- Slow response times
- Lost requests if service down
- Tight coupling
- No retry mechanism

### With Queue (Asynchronous)

```
Client → API → Queue → Response (fast)
                 ↓
              Worker → Process → Database
              (background)
```

**Benefits**:
- Fast responses
- Decoupled services
- Automatic retries
- Load smoothing
- Guaranteed delivery

## Message Queue Patterns

### Point-to-Point (Queue)

One message, one consumer:

```
Producer → [Queue] → Consumer
               ↓
           Message deleted after consumed
```

**Example**: Order processing
```python
# Producer
queue.send({
    "order_id": "123",
    "user_id": "456",
    "total": 99.99
})

# Consumer
message = queue.receive()
process_order(message)
queue.delete(message)  # Acknowledge
```

### Publish-Subscribe (Topic)

One message, many consumers:

```
               ┌→ Consumer A
Producer → [Topic] → Consumer B
               └→ Consumer C
```

**Example**: User signup event
```python
# Publisher
topic.publish({
    "event": "user.signup",
    "user_id": "123",
    "email": "alice@example.com"
})

# Subscriber 1: Send welcome email
# Subscriber 2: Create user profile
# Subscriber 3: Track analytics
```

## Popular Message Queue Systems

### RabbitMQ (AMQP)

```python
import pika

# Connect
connection = pika.BlockingConnection(
    pika.ConnectionParameters('localhost')
)
channel = connection.channel()

# Declare queue
channel.queue_declare(queue='tasks', durable=True)

# Publish
channel.basic_publish(
    exchange='',
    routing_key='tasks',
    body='Process this task',
    properties=pika.BasicProperties(
        delivery_mode=2,  # Persistent
    )
)

# Consume
def callback(ch, method, properties, body):
    print(f"Received {body}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(
    queue='tasks',
    on_message_callback=callback
)

channel.start_consuming()
```

### Apache Kafka (Event Streaming)

```python
from kafka import KafkaProducer, KafkaConsumer

# Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092']
)

producer.send('orders', b'{"order_id": 123}')
producer.flush()

# Consumer
consumer = KafkaConsumer(
    'orders',
    bootstrap_servers=['localhost:9092'],
    group_id='order-processors',
    auto_offset_reset='earliest'
)

for message in consumer:
    print(f"Processing: {message.value}")
```

### AWS SQS (Simple Queue Service)

```python
import boto3

sqs = boto3.client('sqs')

# Send message
sqs.send_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/123/myqueue',
    MessageBody='Process this order',
    MessageAttributes={
        'Priority': {'StringValue': 'high', 'DataType': 'String'}
    }
)

# Receive message
messages = sqs.receive_message(
    QueueUrl='...',
    MaxNumberOfMessages=10,
    WaitTimeSeconds=20  # Long polling
)

for message in messages.get('Messages', []):
    # Process
    process(message['Body'])

    # Delete
    sqs.delete_message(
        QueueUrl='...',
        ReceiptHandle=message['ReceiptHandle']
    )
```

### Redis (Lightweight)

```python
import redis

r = redis.Redis()

# Queue with lists
r.lpush('tasks', '{"task": "send_email"}')

# Worker
while True:
    task = r.brpop('tasks', timeout=5)  # Blocking pop
    if task:
        process(task[1])
```

## Message Queue Concepts

### Acknowledgment (ACK)

Confirm message processed:

```
1. Receive message from queue
2. Process message
3. Send ACK
4. Queue deletes message

If no ACK: Message returns to queue (retry)
```

```python
# Auto ACK (dangerous - lose messages)
channel.basic_consume(queue='tasks', auto_ack=True)

# Manual ACK (safe)
def callback(ch, method, properties, body):
    try:
        process(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
```

### Dead Letter Queue (DLQ)

Failed messages go to DLQ:

```
Queue → Process → Success ✓
  ↓
Retry 3x → Still fails → DLQ
```

```python
# RabbitMQ DLQ
channel.queue_declare(
    queue='tasks',
    arguments={
        'x-dead-letter-exchange': 'dlx',
        'x-max-retries': 3
    }
)

# SQS DLQ
sqs.create_queue(
    QueueName='tasks',
    Attributes={
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': dlq_arn,
            'maxReceiveCount': '3'
        })
    }
)
```

### Message Ordering

**FIFO Queues**: Guaranteed order
```
Send: [A, B, C] → Receive: [A, B, C]
```

**Standard Queues**: Best-effort order
```
Send: [A, B, C] → Receive: [B, A, C] (possible)
```

```python
# SQS FIFO
sqs.send_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/123/myqueue.fifo',
    MessageBody='Task',
    MessageGroupId='user-123',  # Same group = ordered
    MessageDeduplicationId='unique-id'
)
```

### Message Persistence

**In-Memory**: Fast but lost on crash
```
RAM Queue → Crash → Messages lost
```

**Persistent**: Slower but durable
```
Disk Queue → Crash → Messages recovered
```

```python
# RabbitMQ persistent
channel.queue_declare(queue='tasks', durable=True)
channel.basic_publish(
    exchange='',
    routing_key='tasks',
    body='data',
    properties=pika.BasicProperties(delivery_mode=2)
)
```

## Queue Patterns

### Task Queue

Background job processing:

```
Web Request → Queue Task → Return immediately
                  ↓
               Worker picks up task
                  ↓
              Process (slow operation)
```

**Example**: Image processing
```python
# API endpoint
@app.post('/upload')
def upload_image(image):
    # Save to storage
    storage.save(image)

    # Queue processing
    queue.send({
        'task': 'process_image',
        'image_id': image.id,
        'operations': ['resize', 'thumbnail', 'watermark']
    })

    return {'status': 'processing'}

# Worker
def worker():
    while True:
        task = queue.receive()
        if task['task'] == 'process_image':
            process_image(task['image_id'], task['operations'])
```

### Priority Queue

High-priority messages first:

```
High Priority:   [A, B] → Process first
Medium Priority: [C, D] → Process next
Low Priority:    [E, F] → Process last
```

```python
# With priority
queue.send(message, priority=10)  # High
queue.send(message, priority=5)   # Medium
queue.send(message, priority=1)   # Low

# Consumer gets highest priority first
```

### Delay Queue

Delayed message delivery:

```
Send message with delay=300s
  ↓
Wait 5 minutes
  ↓
Message becomes visible
  ↓
Consumer receives
```

```python
# SQS delay
sqs.send_message(
    QueueUrl='...',
    MessageBody='Reminder email',
    DelaySeconds=300  # 5 minutes
)

# RabbitMQ delay
channel.basic_publish(
    exchange='delayed',
    routing_key='tasks',
    body='data',
    properties=pika.BasicProperties(
        headers={'x-delay': 5000}  # 5 seconds
    )
)
```

### Fan-Out Pattern

One message to multiple queues:

```
         Exchange
         /  |  \
        ↓   ↓   ↓
      Q1  Q2  Q3
```

```python
# RabbitMQ fan-out
channel.exchange_declare(exchange='logs', exchange_type='fanout')

# Bind multiple queues
channel.queue_bind(exchange='logs', queue='email-service')
channel.queue_bind(exchange='logs', queue='sms-service')
channel.queue_bind(exchange='logs', queue='analytics')

# Publish once
channel.basic_publish(exchange='logs', routing_key='', body='event')
# All 3 queues receive the message
```

## Event-Driven Architecture

### Event Sourcing

Store all changes as events:

```
Event 1: UserCreated(id=123, name="Alice")
Event 2: EmailUpdated(id=123, email="new@example.com")
Event 3: UserDeleted(id=123)

State = Replay all events
```

```python
# Store events
events = [
    {"type": "UserCreated", "data": {"id": 123, "name": "Alice"}},
    {"type": "EmailUpdated", "data": {"id": 123, "email": "new@..."}},
]

# Rebuild state
def get_user_state(user_id):
    user = None
    for event in events:
        if event['type'] == 'UserCreated':
            user = event['data']
        elif event['type'] == 'EmailUpdated':
            user['email'] = event['data']['email']
    return user
```

### CQRS (Command Query Responsibility Segregation)

Separate read and write models:

```
Commands (Write) → Event Store → Events
                                    ↓
                              Read Model DB
                                    ↑
Queries (Read) ← ← ← ← ← ← ← ← ← ←
```

```python
# Command (write)
def create_order(order_data):
    event = {
        "type": "OrderCreated",
        "data": order_data,
        "timestamp": now()
    }
    event_store.append(event)
    event_bus.publish(event)

# Query (read)
def get_order_summary():
    return read_db.query("SELECT * FROM order_summary")

# Event handler updates read model
@event_handler('OrderCreated')
def update_read_model(event):
    read_db.execute(
        "INSERT INTO order_summary VALUES (...)",
        event['data']
    )
```

### Saga Pattern

Distributed transactions:

```
Service A → Success → Service B → Success → Service C
                ↓
              Failure → Compensate A
```

```python
# Order saga
def process_order(order):
    # Step 1: Reserve inventory
    inventory_result = inventory_service.reserve(order.items)
    if not inventory_result.success:
        return fail("Inventory unavailable")

    # Step 2: Charge payment
    payment_result = payment_service.charge(order.total)
    if not payment_result.success:
        # Compensate: Release inventory
        inventory_service.release(order.items)
        return fail("Payment failed")

    # Step 3: Create shipment
    shipment_result = shipping_service.create(order)
    if not shipment_result.success:
        # Compensate: Refund payment
        payment_service.refund(payment_result.id)
        # Compensate: Release inventory
        inventory_service.release(order.items)
        return fail("Shipping failed")

    return success("Order completed")
```

## Message Queue Best Practices

### Idempotency

Handle duplicate messages safely:

```python
# Non-idempotent (bad)
def process(message):
    balance = get_balance(user_id)
    set_balance(user_id, balance + 100)  # Duplicate = double credit!

# Idempotent (good)
def process(message):
    if processed_messages.exists(message.id):
        return  # Already processed

    balance = get_balance(user_id)
    set_balance(user_id, balance + 100)
    processed_messages.add(message.id)
```

### Poison Messages

Handle messages that always fail:

```python
def process_message(message):
    try:
        # Process
        result = process(message.body)
    except Exception as e:
        message.retry_count += 1

        if message.retry_count > 3:
            # Move to DLQ
            dlq.send(message)
            logger.error(f"Poison message: {message.id}")
        else:
            # Retry with backoff
            queue.send(message, delay=2 ** message.retry_count)
```

### Batching

Process multiple messages together:

```python
# Instead of 1 at a time
messages = queue.receive_batch(max_messages=10)

# Batch insert to DB
db.bulk_insert([process(m) for m in messages])

# Batch ACK
queue.delete_batch([m.receipt_handle for m in messages])
```

### Monitoring

Track queue health:

```python
# Queue metrics
metrics = {
    'messages_in_queue': queue.size(),
    'messages_in_flight': queue.in_flight(),
    'oldest_message_age': queue.oldest_age(),
    'consumer_count': queue.consumers()
}

# Alerts
if metrics['messages_in_queue'] > 10000:
    alert("Queue backing up!")

if metrics['oldest_message_age'] > 3600:
    alert("Messages not being processed!")
```

## Comparison

| System | Type | Ordering | Persistence | Use Case |
|--------|------|----------|-------------|----------|
| **RabbitMQ** | Broker | Yes | Yes | Task queues, RPC |
| **Kafka** | Log | Partition | Yes | Event streaming, logs |
| **SQS** | Broker | FIFO optional | Yes | Cloud-native, AWS |
| **Redis** | In-memory | Lists | Optional | Simple queues, caching |
| **ActiveMQ** | Broker | Yes | Yes | Enterprise, JMS |

## ELI10

Message queues are like a postal service:

- **Queue**: Mailbox where messages wait
- **Producer**: Person sending letters
- **Consumer**: Person receiving letters
- **ACK**: Confirmation letter was read
- **DLQ**: Return to sender for undeliverable mail

**Why useful?**
- Send letters even if recipient not home (asynchronous)
- Letters don't get lost (reliability)
- Can handle many letters (scalability)
- Different mailboxes for different types (routing)

Don't wait for responses - queue it up!

## Further Resources

- [RabbitMQ Tutorials](https://www.rabbitmq.com/getstarted.html)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [AWS SQS Best Practices](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-best-practices.html)
- [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/)
