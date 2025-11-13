# WebSocket

## Overview

WebSocket is a communication protocol that provides full-duplex communication channels over a single TCP connection. It enables real-time, bidirectional communication between a client and server with low overhead, making it ideal for interactive web applications.

## Key Characteristics

```
Protocol: ws:// (unencrypted) or wss:// (encrypted)
Port: 80 (ws) or 443 (wss)
Transport: TCP
Connection: Long-lived, persistent
Communication: Full-duplex (bidirectional)
Latency: Low (no HTTP overhead after handshake)
Overhead: 2-14 bytes per frame
Status: RFC 6455 (2011)

Benefits:
✓ Real-time bidirectional communication
✓ Low latency (no polling overhead)
✓ Efficient (minimal frame overhead)
✓ Server can push data to client
✓ Single TCP connection
✓ Works through proxies and firewalls
✓ Subprotocol support
```

## WebSocket vs Alternatives

### HTTP Polling

```
Traditional HTTP Request/Response:

Client                           Server
  |                                |
  |──── HTTP GET (new data?) ─────>|
  |                                |
  |<─── HTTP Response (no) ────────|
  |                                |
  [wait 1 second]
  |                                |
  |──── HTTP GET (new data?) ─────>|
  |                                |
  |<─── HTTP Response (yes!) ──────|
  |                                |

Problems:
- High latency (constant polling)
- Wasted requests (most return nothing)
- Server load (many unnecessary requests)
- HTTP overhead on every request
```

### Long Polling

```
HTTP Long Polling:

Client                           Server
  |                                |
  |──── HTTP GET (wait) ──────────>|
  |                                | [server holds request]
  |                                | [data arrives]
  |<─── HTTP Response (data!) ─────|
  |                                |
  |──── HTTP GET (wait) ──────────>|
  |                                |

Better, but:
- Still HTTP overhead
- Reconnect after each message
- Server must handle many pending connections
- Not truly bidirectional
```

### Server-Sent Events (SSE)

```
Server-Sent Events:

Client                           Server
  |                                |
  |──── HTTP GET (subscribe) ─────>|
  |                                |
  |<═══ Event stream ══════════════| (one-way)
  |<═══ data: message 1 ═══════════|
  |<═══ data: message 2 ═══════════|
  |<═══ data: message 3 ═══════════|
  |                                |

Good for:
✓ Server → Client only
✓ Text-based data
✓ Auto-reconnect
✓ Simpler than WebSocket

Limited:
✗ One-way only (server to client)
✗ HTTP/1.1 connection limit (6 per domain)
✗ Text only (no binary)
```

### WebSocket

```
WebSocket:

Client                           Server
  |                                |
  |──── HTTP Upgrade ─────────────>|
  |<─── 101 Switching Protocols ───|
  |                                |
  |<══════ WebSocket Open ═════════>|
  |                                |
  |──── Message 1 ────────────────>|
  |<─── Message 2 ─────────────────|
  |──── Message 3 ────────────────>|
  |──── Message 4 ────────────────>|
  |<─── Message 5 ─────────────────|
  |                                |

Best for:
✓ Bidirectional communication
✓ Real-time updates
✓ Low latency required
✓ High message frequency
✓ Binary data support
```

## WebSocket Protocol

### Connection Handshake

WebSocket starts with an HTTP upgrade request:

```http
Client Request:
GET /chat HTTP/1.1
Host: example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
Origin: https://example.com

Server Response:
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

Key Fields:

Upgrade: websocket
- Request protocol upgrade from HTTP to WebSocket

Connection: Upgrade
- Indicates connection upgrade needed

Sec-WebSocket-Key: <base64-encoded-random>
- 16-byte random value, base64 encoded
- Prevents caching proxies from confusing requests

Sec-WebSocket-Version: 13
- WebSocket protocol version (13 is current)

Sec-WebSocket-Accept: <computed-hash>
- Server proves it understands WebSocket
- Computed as: base64(SHA-1(Key + magic-string))
- Magic string: 258EAFA5-E914-47DA-95CA-C5AB0DC85B11

Origin: https://example.com
- Browser sends origin for CORS check
- Server can validate allowed origins

After handshake:
- HTTP connection becomes WebSocket connection
- Both sides can send messages anytime
- Connection stays open until explicitly closed
```

### Handshake Validation

```javascript
// Server-side validation (conceptual)
const crypto = require('crypto');

function computeAcceptKey(clientKey) {
  const MAGIC = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
  const hash = crypto
    .createHash('sha1')
    .update(clientKey + MAGIC)
    .digest('base64');
  return hash;
}

// Example:
const clientKey = 'dGhlIHNhbXBsZSBub25jZQ==';
const acceptKey = computeAcceptKey(clientKey);
console.log(acceptKey);
// Output: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

### Frame Format

After handshake, data is sent in frames:

```
WebSocket Frame Structure:

 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-------+-+-------------+-------------------------------+
|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
|I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
|N|V|V|V|       |S|             |   (if payload len==126/127)   |
| |1|2|3|       |K|             |                               |
+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
|     Extended payload length continued, if payload len == 127  |
+ - - - - - - - - - - - - - - - +-------------------------------+
|                               | Masking-key, if MASK set to 1 |
+-------------------------------+-------------------------------+
| Masking-key (continued)       |          Payload Data         |
+-------------------------------- - - - - - - - - - - - - - - - +
:                     Payload Data continued ...                :
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
|                     Payload Data continued ...                |
+---------------------------------------------------------------+

Fields:

FIN (1 bit):
- 1 = final fragment
- 0 = more fragments coming

RSV1, RSV2, RSV3 (3 bits):
- Reserved for extensions
- Must be 0 unless extension negotiated

Opcode (4 bits):
- 0x0 = Continuation frame
- 0x1 = Text frame (UTF-8)
- 0x2 = Binary frame
- 0x8 = Connection close
- 0x9 = Ping
- 0xA = Pong

MASK (1 bit):
- 1 = payload is masked (required for client → server)
- 0 = payload not masked (server → client)

Payload Length (7 bits, or 7+16, or 7+64):
- 0-125: actual length
- 126: next 16 bits contain length
- 127: next 64 bits contain length

Masking Key (32 bits):
- Present if MASK = 1
- Random 4-byte key
- Client must mask all frames to server

Payload Data:
- Actual message data
- If masked, XOR with masking key

Minimum Frame Size:
- 2 bytes (no masking, payload ≤ 125 bytes)
- 6 bytes (with masking, payload ≤ 125 bytes)
```

### Message Types

```
Text Frame (Opcode 0x1):
- UTF-8 encoded text
- Most common for JSON, strings

Binary Frame (Opcode 0x2):
- Raw binary data
- Images, files, protocol buffers

Ping Frame (Opcode 0x9):
- Sent by either side
- Keep connection alive
- Check if peer responsive

Pong Frame (Opcode 0xA):
- Response to ping
- Sent automatically
- Contains same data as ping

Close Frame (Opcode 0x8):
- Initiates connection close
- Contains optional close code and reason
- Peer responds with close frame
```

## Client-Side Implementation

### JavaScript (Browser)

```javascript
// Create WebSocket connection
const socket = new WebSocket('ws://localhost:8080');

// Alternative: secure WebSocket
// const socket = new WebSocket('wss://example.com/socket');

// Connection opened
socket.addEventListener('open', (event) => {
  console.log('Connected to server');

  // Send message
  socket.send('Hello Server!');

  // Send JSON
  socket.send(JSON.stringify({
    type: 'chat',
    message: 'Hello!',
    timestamp: Date.now()
  }));

  // Send binary data
  const buffer = new Uint8Array([1, 2, 3, 4]);
  socket.send(buffer);
});

// Receive message
socket.addEventListener('message', (event) => {
  console.log('Message from server:', event.data);

  // Handle text data
  if (typeof event.data === 'string') {
    try {
      const data = JSON.parse(event.data);
      handleMessage(data);
    } catch (e) {
      console.log('Text:', event.data);
    }
  }

  // Handle binary data
  if (event.data instanceof Blob) {
    event.data.arrayBuffer().then(buffer => {
      const view = new Uint8Array(buffer);
      console.log('Binary data:', view);
    });
  }

  // Or receive as ArrayBuffer
  // socket.binaryType = 'arraybuffer';
});

// Connection closed
socket.addEventListener('close', (event) => {
  console.log('Disconnected from server');
  console.log('Code:', event.code);
  console.log('Reason:', event.reason);
  console.log('Clean:', event.wasClean);
});

// Connection error
socket.addEventListener('error', (error) => {
  console.error('WebSocket error:', error);
});

// Send messages
function sendMessage(text) {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(text);
  } else {
    console.error('WebSocket not connected');
  }
}

// Close connection
function closeConnection() {
  socket.close(1000, 'User closed connection');
}

// WebSocket states
console.log('CONNECTING:', WebSocket.CONNECTING); // 0
console.log('OPEN:', WebSocket.OPEN);             // 1
console.log('CLOSING:', WebSocket.CLOSING);       // 2
console.log('CLOSED:', WebSocket.CLOSED);         // 3

// Check current state
console.log('Current state:', socket.readyState);
```

### Advanced Client Features

```javascript
class WebSocketClient {
  constructor(url, options = {}) {
    this.url = url;
    this.options = {
      reconnect: true,
      reconnectInterval: 1000,
      reconnectDecay: 1.5,
      maxReconnectInterval: 30000,
      maxReconnectAttempts: 10,
      ...options
    };

    this.ws = null;
    this.reconnectAttempts = 0;
    this.messageQueue = [];
    this.handlers = new Map();

    this.connect();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('Connected');
      this.reconnectAttempts = 0;

      // Send queued messages
      while (this.messageQueue.length > 0) {
        this.send(this.messageQueue.shift());
      }

      this.emit('connect');
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.emit(data.type || 'message', data);
      } catch (e) {
        this.emit('message', event.data);
      }
    };

    this.ws.onclose = (event) => {
      console.log('Disconnected:', event.code, event.reason);
      this.emit('disconnect', event);

      if (this.options.reconnect) {
        this.reconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    };
  }

  reconnect() {
    if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      this.emit('reconnect_failed');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.options.reconnectInterval *
        Math.pow(this.options.reconnectDecay, this.reconnectAttempts - 1),
      this.options.maxReconnectInterval
    );

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.emit('reconnecting', this.reconnectAttempts);
      this.connect();
    }, delay);
  }

  send(data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      const message = typeof data === 'string'
        ? data
        : JSON.stringify(data);
      this.ws.send(message);
    } else {
      console.log('Queueing message (not connected)');
      this.messageQueue.push(data);
    }
  }

  on(event, handler) {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, []);
    }
    this.handlers.get(event).push(handler);
  }

  emit(event, data) {
    if (this.handlers.has(event)) {
      this.handlers.get(event).forEach(handler => handler(data));
    }
  }

  close() {
    this.options.reconnect = false;
    if (this.ws) {
      this.ws.close(1000, 'Client closed');
    }
  }
}

// Usage
const client = new WebSocketClient('ws://localhost:8080', {
  reconnect: true,
  maxReconnectAttempts: 5
});

client.on('connect', () => {
  console.log('Connected!');
  client.send({ type: 'auth', token: 'abc123' });
});

client.on('message', (data) => {
  console.log('Received:', data);
});

client.on('disconnect', () => {
  console.log('Connection lost');
});

client.send({ type: 'chat', message: 'Hello' });
```

## Server-Side Implementation

### Node.js with 'ws' Library

```javascript
const WebSocket = require('ws');
const http = require('http');

// Create HTTP server
const server = http.createServer((req, res) => {
  res.writeHead(200);
  res.end('WebSocket server running');
});

// Create WebSocket server
const wss = new WebSocket.Server({ server });

// Track connected clients
const clients = new Set();

// Connection handler
wss.on('connection', (ws, req) => {
  console.log('Client connected from', req.socket.remoteAddress);

  // Add to client set
  clients.add(ws);

  // Send welcome message
  ws.send(JSON.stringify({
    type: 'welcome',
    message: 'Connected to server',
    clients: clients.size
  }));

  // Broadcast new connection to all clients
  broadcast({
    type: 'user-joined',
    clients: clients.size
  }, ws);

  // Message handler
  ws.on('message', (data) => {
    console.log('Received:', data.toString());

    try {
      const message = JSON.parse(data);

      // Handle different message types
      switch (message.type) {
        case 'chat':
          // Broadcast chat message
          broadcast({
            type: 'chat',
            message: message.message,
            timestamp: Date.now()
          });
          break;

        case 'ping':
          // Respond to ping
          ws.send(JSON.stringify({
            type: 'pong',
            timestamp: Date.now()
          }));
          break;

        default:
          console.log('Unknown message type:', message.type);
      }
    } catch (e) {
      console.error('Invalid JSON:', e);
    }
  });

  // Pong handler (heartbeat)
  ws.on('pong', () => {
    ws.isAlive = true;
  });

  // Close handler
  ws.on('close', (code, reason) => {
    console.log('Client disconnected:', code, reason.toString());
    clients.delete(ws);

    // Notify others
    broadcast({
      type: 'user-left',
      clients: clients.size
    });
  });

  // Error handler
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });

  // Mark as alive for heartbeat
  ws.isAlive = true;
});

// Broadcast to all clients
function broadcast(data, exclude = null) {
  const message = JSON.stringify(data);

  clients.forEach(client => {
    if (client !== exclude && client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

// Heartbeat (detect dead connections)
const heartbeatInterval = setInterval(() => {
  clients.forEach(ws => {
    if (!ws.isAlive) {
      console.log('Terminating dead connection');
      ws.terminate();
      clients.delete(ws);
      return;
    }

    ws.isAlive = false;
    ws.ping();
  });
}, 30000); // Every 30 seconds

// Cleanup on server close
wss.on('close', () => {
  clearInterval(heartbeatInterval);
});

// Start server
const PORT = 8080;
server.listen(PORT, () => {
  console.log(`WebSocket server listening on port ${PORT}`);
});
```

### Advanced Server Features

```javascript
const WebSocket = require('ws');
const http = require('http');
const url = require('url');

class WebSocketServer {
  constructor(options = {}) {
    this.options = {
      port: 8080,
      pingInterval: 30000,
      maxClients: 1000,
      ...options
    };

    this.server = http.createServer();
    this.wss = new WebSocket.Server({ server: this.server });

    this.rooms = new Map(); // roomId -> Set of clients
    this.clients = new Map(); // ws -> client info

    this.setupHandlers();
    this.startHeartbeat();
  }

  setupHandlers() {
    this.wss.on('connection', (ws, req) => {
      // Check max clients
      if (this.clients.size >= this.options.maxClients) {
        ws.close(1008, 'Server full');
        return;
      }

      // Parse URL parameters
      const params = url.parse(req.url, true).query;

      // Create client info
      const clientInfo = {
        id: this.generateId(),
        ip: req.socket.remoteAddress,
        rooms: new Set(),
        authenticated: false,
        metadata: {}
      };

      this.clients.set(ws, clientInfo);
      ws.isAlive = true;

      console.log(`Client ${clientInfo.id} connected`);

      // Send client ID
      this.send(ws, {
        type: 'connected',
        clientId: clientInfo.id
      });

      // Message handler
      ws.on('message', (data) => {
        this.handleMessage(ws, data);
      });

      // Pong handler
      ws.on('pong', () => {
        ws.isAlive = true;
      });

      // Close handler
      ws.on('close', () => {
        this.handleDisconnect(ws);
      });

      // Error handler
      ws.on('error', (error) => {
        console.error('Error:', error);
      });
    });
  }

  handleMessage(ws, data) {
    const client = this.clients.get(ws);
    if (!client) return;

    try {
      const message = JSON.parse(data);

      switch (message.type) {
        case 'auth':
          this.handleAuth(ws, message);
          break;

        case 'join-room':
          this.joinRoom(ws, message.room);
          break;

        case 'leave-room':
          this.leaveRoom(ws, message.room);
          break;

        case 'message':
          this.handleRoomMessage(ws, message);
          break;

        default:
          console.log('Unknown message type:', message.type);
      }
    } catch (e) {
      console.error('Invalid message:', e);
      this.send(ws, {
        type: 'error',
        message: 'Invalid message format'
      });
    }
  }

  handleAuth(ws, message) {
    const client = this.clients.get(ws);

    // Validate token (simplified)
    if (message.token === 'valid-token') {
      client.authenticated = true;
      client.metadata.username = message.username;

      this.send(ws, {
        type: 'auth-success',
        username: message.username
      });
    } else {
      this.send(ws, {
        type: 'auth-failed',
        message: 'Invalid token'
      });

      ws.close(1008, 'Authentication failed');
    }
  }

  joinRoom(ws, roomId) {
    const client = this.clients.get(ws);
    if (!client?.authenticated) return;

    // Create room if doesn't exist
    if (!this.rooms.has(roomId)) {
      this.rooms.set(roomId, new Set());
    }

    // Add client to room
    this.rooms.get(roomId).add(ws);
    client.rooms.add(roomId);

    console.log(`Client ${client.id} joined room ${roomId}`);

    // Notify client
    this.send(ws, {
      type: 'joined-room',
      room: roomId,
      members: this.rooms.get(roomId).size
    });

    // Notify room members
    this.broadcastToRoom(roomId, {
      type: 'user-joined',
      userId: client.id,
      username: client.metadata.username,
      members: this.rooms.get(roomId).size
    }, ws);
  }

  leaveRoom(ws, roomId) {
    const client = this.clients.get(ws);
    if (!client) return;

    if (this.rooms.has(roomId)) {
      this.rooms.get(roomId).delete(ws);
      client.rooms.delete(roomId);

      // Notify others
      this.broadcastToRoom(roomId, {
        type: 'user-left',
        userId: client.id,
        members: this.rooms.get(roomId).size
      });

      // Clean up empty rooms
      if (this.rooms.get(roomId).size === 0) {
        this.rooms.delete(roomId);
      }
    }
  }

  handleRoomMessage(ws, message) {
    const client = this.clients.get(ws);
    if (!client?.authenticated) return;

    if (message.room && this.rooms.has(message.room)) {
      this.broadcastToRoom(message.room, {
        type: 'message',
        userId: client.id,
        username: client.metadata.username,
        message: message.content,
        timestamp: Date.now()
      });
    }
  }

  handleDisconnect(ws) {
    const client = this.clients.get(ws);
    if (!client) return;

    console.log(`Client ${client.id} disconnected`);

    // Remove from all rooms
    client.rooms.forEach(roomId => {
      this.leaveRoom(ws, roomId);
    });

    // Remove from clients
    this.clients.delete(ws);
  }

  send(ws, data) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  }

  broadcastToRoom(roomId, data, exclude = null) {
    if (!this.rooms.has(roomId)) return;

    const message = JSON.stringify(data);
    this.rooms.get(roomId).forEach(client => {
      if (client !== exclude && client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  }

  broadcastToAll(data, exclude = null) {
    const message = JSON.stringify(data);
    this.clients.forEach((clientInfo, ws) => {
      if (ws !== exclude && ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    });
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      this.clients.forEach((clientInfo, ws) => {
        if (!ws.isAlive) {
          console.log(`Terminating dead connection: ${clientInfo.id}`);
          ws.terminate();
          return;
        }

        ws.isAlive = false;
        ws.ping();
      });
    }, this.options.pingInterval);
  }

  generateId() {
    return Math.random().toString(36).substring(2, 15);
  }

  start() {
    this.server.listen(this.options.port, () => {
      console.log(`WebSocket server listening on port ${this.options.port}`);
    });
  }

  stop() {
    clearInterval(this.heartbeatInterval);
    this.wss.close();
    this.server.close();
  }
}

// Usage
const server = new WebSocketServer({
  port: 8080,
  pingInterval: 30000,
  maxClients: 1000
});

server.start();
```

## Use Cases

### 1. Chat Application

```javascript
// Client
class ChatClient {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.setupHandlers();
  }

  setupHandlers() {
    this.socket.onopen = () => {
      console.log('Connected to chat');
      this.authenticate();
    };

    this.socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'message':
          this.displayMessage(data);
          break;
        case 'user-joined':
          this.showNotification(`${data.username} joined`);
          break;
        case 'user-left':
          this.showNotification(`${data.username} left`);
          break;
      }
    };
  }

  authenticate() {
    this.socket.send(JSON.stringify({
      type: 'auth',
      token: localStorage.getItem('token'),
      username: localStorage.getItem('username')
    }));
  }

  joinRoom(roomId) {
    this.socket.send(JSON.stringify({
      type: 'join-room',
      room: roomId
    }));
  }

  sendMessage(roomId, message) {
    this.socket.send(JSON.stringify({
      type: 'message',
      room: roomId,
      content: message
    }));
  }

  displayMessage(data) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message';
    messageElement.innerHTML = `
      <span class="username">${data.username}:</span>
      <span class="content">${data.message}</span>
      <span class="timestamp">${new Date(data.timestamp).toLocaleTimeString()}</span>
    `;
    document.getElementById('messages').appendChild(messageElement);
  }

  showNotification(text) {
    console.log(text);
  }
}

const chat = new ChatClient('ws://localhost:8080');
chat.joinRoom('general');
```

### 2. Real-Time Dashboard

```javascript
// Server: Push updates to dashboard
function broadcastMetrics() {
  const metrics = {
    type: 'metrics',
    cpu: getCpuUsage(),
    memory: getMemoryUsage(),
    activeUsers: clients.size,
    requestsPerSecond: getRequestRate(),
    timestamp: Date.now()
  };

  broadcastToAll(metrics);
}

setInterval(broadcastMetrics, 1000);

// Client: Display real-time metrics
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'metrics') {
    updateChart('cpu', data.cpu);
    updateChart('memory', data.memory);
    updateCounter('users', data.activeUsers);
    updateCounter('rps', data.requestsPerSecond);
  }
};
```

### 3. Live Notifications

```javascript
// Server: Send notifications
function notifyUser(userId, notification) {
  const client = getUserWebSocket(userId);

  if (client && client.readyState === WebSocket.OPEN) {
    client.send(JSON.stringify({
      type: 'notification',
      title: notification.title,
      message: notification.message,
      priority: notification.priority,
      timestamp: Date.now()
    }));
  }
}

// Client: Display notifications
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'notification') {
    showNotification(data.title, data.message);

    // Play sound for high priority
    if (data.priority === 'high') {
      playNotificationSound();
    }

    // Desktop notification
    if (Notification.permission === 'granted') {
      new Notification(data.title, {
        body: data.message,
        icon: '/icon.png'
      });
    }
  }
};
```

### 4. Collaborative Editing

```javascript
// Server: Broadcast document changes
wss.on('connection', (ws) => {
  ws.on('message', (data) => {
    const change = JSON.parse(data);

    if (change.type === 'edit') {
      // Apply change to document
      applyChange(change.documentId, change.operation);

      // Broadcast to others in same document
      broadcastToDocument(change.documentId, {
        type: 'edit',
        operation: change.operation,
        userId: ws.userId
      }, ws);
    }
  });
});

// Client: Send and receive edits
let editor = document.getElementById('editor');

editor.addEventListener('input', debounce((e) => {
  socket.send(JSON.stringify({
    type: 'edit',
    documentId: currentDocId,
    operation: {
      type: 'insert',
      position: e.target.selectionStart,
      text: e.data
    }
  }));
}, 100));

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'edit' && data.userId !== myUserId) {
    applyRemoteEdit(data.operation);
  }
};
```

### 5. Gaming/Multiplayer

```javascript
// Server: Game state synchronization
const gameState = {
  players: new Map(),
  entities: []
};

function updateGameState() {
  broadcastToAll({
    type: 'state',
    players: Array.from(gameState.players.values()),
    entities: gameState.entities,
    timestamp: Date.now()
  });
}

// 60 updates per second
setInterval(updateGameState, 1000 / 60);

// Client: Send player input
const input = {
  keys: {},
  mouse: { x: 0, y: 0 }
};

document.addEventListener('keydown', (e) => {
  input.keys[e.key] = true;

  socket.send(JSON.stringify({
    type: 'input',
    keys: input.keys,
    timestamp: Date.now()
  }));
});

// Receive game state
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'state') {
    renderGameState(data.players, data.entities);
  }
};
```

## Security

### Authentication

```javascript
// Server: Verify token on connection
wss.on('connection', (ws, req) => {
  // Extract token from query string
  const params = new URLSearchParams(req.url.split('?')[1]);
  const token = params.get('token');

  // Verify token
  if (!verifyToken(token)) {
    ws.close(1008, 'Invalid authentication');
    return;
  }

  ws.userId = decodeToken(token).userId;
});

// Or: Authenticate after connection
ws.on('message', (data) => {
  const message = JSON.parse(data);

  if (message.type === 'auth') {
    if (verifyToken(message.token)) {
      ws.authenticated = true;
      ws.userId = decodeToken(message.token).userId;
      ws.send(JSON.stringify({ type: 'auth-success' }));
    } else {
      ws.close(1008, 'Authentication failed');
    }
  } else if (!ws.authenticated) {
    ws.send(JSON.stringify({
      type: 'error',
      message: 'Not authenticated'
    }));
  }
});
```

### Origin Validation

```javascript
// Server: Validate origin
wss.on('connection', (ws, req) => {
  const origin = req.headers.origin;
  const allowedOrigins = [
    'https://example.com',
    'https://app.example.com'
  ];

  if (!allowedOrigins.includes(origin)) {
    console.log('Rejected connection from:', origin);
    ws.close(1008, 'Origin not allowed');
    return;
  }

  // Accept connection
});
```

### Rate Limiting

```javascript
// Server: Rate limit messages
const rateLimits = new Map(); // clientId -> message count

ws.on('message', (data) => {
  const clientId = ws.userId || ws.ip;

  if (!rateLimits.has(clientId)) {
    rateLimits.set(clientId, { count: 0, resetAt: Date.now() + 60000 });
  }

  const limit = rateLimits.get(clientId);

  // Reset if window expired
  if (Date.now() > limit.resetAt) {
    limit.count = 0;
    limit.resetAt = Date.now() + 60000;
  }

  // Check limit (100 messages per minute)
  if (limit.count >= 100) {
    ws.send(JSON.stringify({
      type: 'error',
      message: 'Rate limit exceeded'
    }));
    return;
  }

  limit.count++;

  // Process message
  handleMessage(ws, data);
});
```

### Input Validation

```javascript
// Server: Validate and sanitize input
function handleMessage(ws, data) {
  let message;

  try {
    message = JSON.parse(data);
  } catch (e) {
    ws.send(JSON.stringify({
      type: 'error',
      message: 'Invalid JSON'
    }));
    return;
  }

  // Validate message structure
  if (!message.type || typeof message.type !== 'string') {
    ws.send(JSON.stringify({
      type: 'error',
      message: 'Invalid message format'
    }));
    return;
  }

  // Validate message size
  if (data.length > 10000) {
    ws.send(JSON.stringify({
      type: 'error',
      message: 'Message too large'
    }));
    return;
  }

  // Sanitize text content
  if (message.content) {
    message.content = sanitizeHtml(message.content);
  }

  // Process validated message
  processMessage(ws, message);
}
```

### Secure WebSocket (wss://)

```javascript
// Server: Use HTTPS/WSS
const https = require('https');
const fs = require('fs');

const server = https.createServer({
  cert: fs.readFileSync('cert.pem'),
  key: fs.readFileSync('key.pem')
});

const wss = new WebSocket.Server({ server });

server.listen(443);

// Client: Connect with wss://
const socket = new WebSocket('wss://example.com/socket');
```

## Best Practices

### 1. Heartbeat/Ping-Pong

```javascript
// Server: Detect dead connections
const heartbeatInterval = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) {
      return ws.terminate();
    }

    ws.isAlive = false;
    ws.ping();
  });
}, 30000);

wss.on('connection', (ws) => {
  ws.isAlive = true;

  ws.on('pong', () => {
    ws.isAlive = true;
  });
});

// Client: Respond to pings (automatic in browsers)
// Or implement custom heartbeat:
setInterval(() => {
  socket.send(JSON.stringify({ type: 'ping' }));
}, 30000);
```

### 2. Reconnection Strategy

```javascript
// Client: Exponential backoff
class ReconnectingWebSocket {
  constructor(url) {
    this.url = url;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.reconnectAttempts = 0;
    this.connect();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('Connected');
      this.reconnectDelay = 1000;
      this.reconnectAttempts = 0;
    };

    this.ws.onclose = () => {
      console.log('Disconnected');
      this.scheduleReconnect();
    };
  }

  scheduleReconnect() {
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay
    );

    console.log(`Reconnecting in ${delay}ms`);

    setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }
}
```

### 3. Message Queuing

```javascript
// Client: Queue messages when disconnected
class QueuedWebSocket {
  constructor(url) {
    this.url = url;
    this.queue = [];
    this.connect();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      // Send queued messages
      while (this.queue.length > 0) {
        this.ws.send(this.queue.shift());
      }
    };
  }

  send(data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      this.queue.push(data);
    }
  }
}
```

### 4. Binary Data

```javascript
// Send binary efficiently
const buffer = new ArrayBuffer(8);
const view = new DataView(buffer);

view.setUint32(0, 12345);
view.setFloat32(4, 3.14);

socket.send(buffer);

// Receive binary
socket.binaryType = 'arraybuffer';

socket.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    const view = new DataView(event.data);
    const num = view.getUint32(0);
    const float = view.getFloat32(4);
  }
};
```

### 5. Compression

```javascript
// Server: Enable per-message deflate
const wss = new WebSocket.Server({
  server,
  perMessageDeflate: {
    zlibDeflateOptions: {
      chunkSize: 1024,
      memLevel: 7,
      level: 3
    },
    zlibInflateOptions: {
      chunkSize: 10 * 1024
    },
    clientNoContextTakeover: true,
    serverNoContextTakeover: true,
    serverMaxWindowBits: 10,
    concurrencyLimit: 10,
    threshold: 1024 // Compress only messages > 1KB
  }
});
```

## Debugging

### Browser DevTools

```javascript
// Chrome/Firefox DevTools
// Network tab → WS/Messages

// View frames
// - Sent (green arrow)
// - Received (red arrow)
// - Click to view content

// Console logging
const socket = new WebSocket('ws://localhost:8080');

socket.addEventListener('message', (event) => {
  console.log('%c⬇ Received', 'color: blue', event.data);
});

socket.send = new Proxy(socket.send, {
  apply(target, thisArg, args) {
    console.log('%c⬆ Sent', 'color: green', args[0]);
    return target.apply(thisArg, args);
  }
});
```

### Command-Line Tools

```bash
# wscat - WebSocket client
npm install -g wscat

# Connect to server
wscat -c ws://localhost:8080

# Send message
> {"type": "chat", "message": "Hello"}

# Listen for messages
< {"type": "message", "content": "Hi there"}

# WebSocket with headers
wscat -c ws://localhost:8080 -H "Authorization: Bearer token"

# Test wss:// with self-signed cert
wscat -c wss://localhost:443 -n

# websocat - More features
cargo install websocat

# Connect
websocat ws://localhost:8080

# Binary mode
websocat --binary ws://localhost:8080

# tcpdump - Capture WebSocket traffic
sudo tcpdump -i any -A 'tcp port 8080'

# Wireshark
# Filter: websocket
# Analyze → Decode As → WebSocket
```

## Common Issues

```
Issue: Connection fails immediately
Causes:
- Wrong URL (ws:// vs wss://)
- Server not running
- Firewall blocking port
- CORS/Origin mismatch

Solution:
- Check server logs
- Verify URL and port
- Check browser console for errors
- Validate origin on server

Issue: Connection drops frequently
Causes:
- No heartbeat/ping
- Idle timeout
- Network issues
- Proxy timeout

Solution:
- Implement ping/pong
- Send periodic messages
- Reduce ping interval
- Use wss:// for better stability

Issue: Messages not received
Causes:
- Wrong readyState
- Connection closed
- Message too large
- Server not broadcasting

Solution:
- Check socket.readyState === OPEN
- Add message queuing
- Split large messages
- Verify server broadcast logic

Issue: High memory usage
Causes:
- Not closing connections
- Large message buffers
- Too many connections
- Memory leaks

Solution:
- Close unused connections
- Limit message size
- Set max connections
- Use heartbeat to detect dead connections
```

## ELI10: WebSocket Explained Simply

WebSocket is like having a phone call instead of sending letters:

### Traditional HTTP (Letters)
```
You: "Any new messages?" [wait for response]
Server: "No"

[1 second later]
You: "Any new messages?" [wait for response]
Server: "No"

[1 second later]
You: "Any new messages?" [wait for response]
Server: "Yes! Here's one"

Problem: Lots of wasted "letters" (requests)
```

### WebSocket (Phone Call)
```
You: "Hello!" [open connection]
Server: "Hi!" [connection open]

[Connection stays open]

Server: "New message for you!"
You: "Thanks! Here's my reply"
Server: "Got it!"
You: "Question?"
Server: "Answer!"

Connection stays open until you hang up
```

### Key Differences
```
HTTP:
- Ask → Wait → Answer → Close
- Repeat every time
- Like knocking on door for each question

WebSocket:
- Open door once
- Walk in and stay
- Talk back and forth
- Like having a conversation
```

### Real Examples
```
HTTP: Checking email every minute
WebSocket: Email app shows new mail instantly

HTTP: Refreshing page to see chat messages
WebSocket: Messages appear as sent

HTTP: Reloading dashboard for new data
WebSocket: Dashboard updates in real-time
```

## Further Resources

### Specifications
- [RFC 6455 - WebSocket Protocol](https://tools.ietf.org/html/rfc6455)
- [WebSocket API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [WebSocket Extensions](https://www.iana.org/assignments/websocket/websocket.xml)

### Libraries

**JavaScript (Client)**
- Native WebSocket API (built-in)
- [Socket.IO](https://socket.io/) - High-level library with fallbacks
- [SockJS](https://github.com/sockjs/sockjs-client) - WebSocket emulation

**Node.js (Server)**
- [ws](https://github.com/websockets/ws) - Fast, standards-compliant
- [Socket.IO](https://socket.io/) - Client + server library
- [uWebSockets.js](https://github.com/uNetworking/uWebSockets.js) - Ultra fast

**Python**
- [websockets](https://websockets.readthedocs.io/) - asyncio library
- [aiohttp](https://docs.aiohttp.org/) - WebSocket support
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/) - Flask integration

**Go**
- [gorilla/websocket](https://github.com/gorilla/websocket)
- [nhooyr/websocket](https://github.com/nhooyr/websocket)

**Rust**
- [tokio-tungstenite](https://github.com/snapview/tokio-tungstenite)
- [actix-web](https://actix.rs/) - WebSocket support

### Tools
- [wscat](https://github.com/websockets/wscat) - WebSocket CLI
- [websocat](https://github.com/vi/websocat) - netcat for WebSockets
- [Postman](https://www.postman.com/) - WebSocket testing

### Testing
- [WebSocket King](https://websocketking.com/) - Online tester
- [PieSocket](https://www.piesocket.com/websocket-tester) - Testing tool

### Books & Tutorials
- [The Definitive Guide to HTML5 WebSocket](https://www.apress.com/gp/book/9781430247401)
- [Real-Time Web Application Development](https://www.oreilly.com/library/view/real-time-web-application/9781484232705/)
