# WebRTC (Web Real-Time Communication)

## Overview

WebRTC (Web Real-Time Communication) is an open-source framework that enables real-time peer-to-peer communication directly between web browsers and mobile applications. It supports video, audio, and arbitrary data transfer without requiring plugins or third-party software.

## Key Features

```
1. Peer-to-Peer Communication
   - Direct browser-to-browser connections
   - Low latency (no server relay required*)
   - Reduced bandwidth costs

2. Media Support
   - Audio streaming
   - Video streaming
   - Screen sharing
   - Data channels for arbitrary data

3. Built-in Security
   - Mandatory encryption (DTLS, SRTP)
   - No unencrypted media transmission
   - Secure signaling required

4. NAT/Firewall Traversal
   - ICE protocol for connectivity
   - STUN for public address discovery
   - TURN as relay fallback

5. Adaptive Quality
   - Bandwidth estimation
   - Codec negotiation
   - Quality adjusts to network conditions

* Direct P2P when possible; TURN relay as fallback
```

## WebRTC Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WebRTC Application                        │
│  (JavaScript API in browser or native mobile app)           │
└────────────────────┬────────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     ▼               ▼               ▼
┌─────────┐   ┌──────────┐   ┌──────────┐
│  Media  │   │   Data   │   │ Signaling│
│ Streams │   │ Channels │   │ (Custom) │
└─────────┘   └──────────┘   └──────────┘
     │               │               │
     ▼               ▼               │
┌─────────────────────────┐         │
│   WebRTC Core APIs      │         │
│                         │         │
│ - getUserMedia()        │         │
│ - RTCPeerConnection     │         │
│ - RTCDataChannel        │         │
└─────────────────────────┘         │
     │                               │
     ▼                               │
┌─────────────────────────┐         │
│   ICE/STUN/TURN         │         │
│ (NAT Traversal)         │         │
└─────────────────────────┘         │
     │                               │
     └───────────────┬───────────────┘
                     │
                     ▼
           ┌──────────────────┐
           │   Network Layer   │
           │  (UDP/TCP/TLS)    │
           └──────────────────┘
```

## Core Components

### 1. getUserMedia API

Access local camera and microphone:

```javascript
// Basic usage
async function getLocalMedia() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true
    });

    // Display local video
    document.getElementById('localVideo').srcObject = stream;
    return stream;
  } catch (error) {
    console.error('Error accessing media devices:', error);
  }
}

// Advanced constraints
const constraints = {
  video: {
    width: { min: 640, ideal: 1280, max: 1920 },
    height: { min: 480, ideal: 720, max: 1080 },
    frameRate: { ideal: 30, max: 60 },
    facingMode: 'user' // or 'environment' for rear camera
  },
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true
  }
};

const stream = await navigator.mediaDevices.getUserMedia(constraints);

// List available devices
const devices = await navigator.mediaDevices.enumerateDevices();
devices.forEach(device => {
  console.log(`${device.kind}: ${device.label} (${device.deviceId})`);
});

// Screen sharing
const screenStream = await navigator.mediaDevices.getDisplayMedia({
  video: {
    cursor: 'always',
    displaySurface: 'monitor' // 'window', 'application', 'browser'
  },
  audio: false
});
```

### 2. RTCPeerConnection

Core API for peer-to-peer connection:

```javascript
// Create peer connection
const configuration = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    {
      urls: 'turn:turn.example.com:3478',
      username: 'user',
      credential: 'pass'
    }
  ],
  iceCandidatePoolSize: 10
};

const peerConnection = new RTCPeerConnection(configuration);

// Add local stream to connection
localStream.getTracks().forEach(track => {
  peerConnection.addTrack(track, localStream);
});

// Listen for remote stream
peerConnection.ontrack = (event) => {
  const remoteVideo = document.getElementById('remoteVideo');
  if (remoteVideo.srcObject !== event.streams[0]) {
    remoteVideo.srcObject = event.streams[0];
    console.log('Received remote stream');
  }
};

// Handle ICE candidates
peerConnection.onicecandidate = (event) => {
  if (event.candidate) {
    // Send candidate to remote peer via signaling
    sendToSignalingServer({
      type: 'ice-candidate',
      candidate: event.candidate
    });
  }
};

// Monitor connection state
peerConnection.onconnectionstatechange = () => {
  console.log('Connection state:', peerConnection.connectionState);
  // States: new, connecting, connected, disconnected, failed, closed
};

peerConnection.oniceconnectionstatechange = () => {
  console.log('ICE state:', peerConnection.iceConnectionState);
  // States: new, checking, connected, completed, failed, disconnected, closed
};
```

### 3. RTCDataChannel

Bi-directional data transfer:

```javascript
// Sender creates data channel
const dataChannel = peerConnection.createDataChannel('chat', {
  ordered: true,        // Guarantee order
  maxRetransmits: 3     // Retry failed messages 3 times
  // OR: maxPacketLifeTime: 3000  // Drop after 3 seconds
});

dataChannel.onopen = () => {
  console.log('Data channel opened');
  dataChannel.send('Hello!');
};

dataChannel.onmessage = (event) => {
  console.log('Received:', event.data);
};

dataChannel.onerror = (error) => {
  console.error('Data channel error:', error);
};

dataChannel.onclose = () => {
  console.log('Data channel closed');
};

// Receiver listens for data channel
peerConnection.ondatachannel = (event) => {
  const receiveChannel = event.channel;

  receiveChannel.onmessage = (event) => {
    console.log('Received:', event.data);
  };

  receiveChannel.onopen = () => {
    console.log('Receive channel opened');
  };
};

// Send different data types
dataChannel.send('Text message');
dataChannel.send(JSON.stringify({ type: 'chat', message: 'Hi' }));
dataChannel.send(new Uint8Array([1, 2, 3, 4])); // Binary
dataChannel.send(new Blob(['file content'])); // Blob

// Check buffered amount before sending large data
if (dataChannel.bufferedAmount === 0) {
  dataChannel.send(largeData);
}
```

## Connection Establishment (Signaling)

WebRTC doesn't define signaling - you implement it yourself:

### Offer/Answer Exchange (SDP)

```javascript
// ============================================
// Caller (Initiator)
// ============================================

// 1. Create offer
const offer = await peerConnection.createOffer({
  offerToReceiveAudio: true,
  offerToReceiveVideo: true
});

// 2. Set local description
await peerConnection.setLocalDescription(offer);

// 3. Send offer to remote peer via signaling
sendToSignalingServer({
  type: 'offer',
  sdp: peerConnection.localDescription
});

// 4. Receive answer from signaling server
signalingSocket.on('answer', async (answer) => {
  await peerConnection.setRemoteDescription(
    new RTCSessionDescription(answer)
  );
});

// ============================================
// Callee (Responder)
// ============================================

// 1. Receive offer from signaling server
signalingSocket.on('offer', async (offer) => {
  // 2. Set remote description
  await peerConnection.setRemoteDescription(
    new RTCSessionDescription(offer)
  );

  // 3. Create answer
  const answer = await peerConnection.createAnswer();

  // 4. Set local description
  await peerConnection.setLocalDescription(answer);

  // 5. Send answer back via signaling
  sendToSignalingServer({
    type: 'answer',
    sdp: peerConnection.localDescription
  });
});

// ============================================
// Both Peers
// ============================================

// Handle ICE candidates
peerConnection.onicecandidate = (event) => {
  if (event.candidate) {
    sendToSignalingServer({
      type: 'ice-candidate',
      candidate: event.candidate
    });
  }
};

// Receive ICE candidates from signaling
signalingSocket.on('ice-candidate', async (candidate) => {
  try {
    await peerConnection.addIceCandidate(
      new RTCIceCandidate(candidate)
    );
  } catch (error) {
    console.error('Error adding ICE candidate:', error);
  }
});
```

### SDP (Session Description Protocol)

SDP describes the media session:

```
Example SDP Offer:

v=0
o=- 123456789 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
a=msid-semantic: WMS stream1

m=audio 9 UDP/TLS/RTP/SAVPF 111 103 104
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:F7gI
a=ice-pwd:x9cml6RvRClHPcAy
a=ice-options:trickle
a=fingerprint:sha-256 8B:87:09:8A:5D:C2:...
a=setup:actpass
a=mid:0
a=sendrecv
a=rtcp-mux
a=rtpmap:111 opus/48000/2
a=rtpmap:103 ISAC/16000
a=rtpmap:104 ISAC/32000

m=video 9 UDP/TLS/RTP/SAVPF 96 97 98
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:F7gI
a=ice-pwd:x9cml6RvRClHPcAy
a=ice-options:trickle
a=fingerprint:sha-256 8B:87:09:8A:5D:C2:...
a=setup:actpass
a=mid:1
a=sendrecv
a=rtcp-mux
a=rtpmap:96 VP8/90000
a=rtpmap:97 VP9/90000
a=rtpmap:98 H264/90000

Key Fields:
- v=0: SDP version
- m=: Media description (audio/video)
- c=: Connection information
- a=: Attributes (ICE, codecs, etc.)
- rtpmap: RTP payload mapping
- ice-ufrag/ice-pwd: ICE credentials
- fingerprint: DTLS certificate fingerprint
```

## Signaling Implementation Examples

### WebSocket Signaling Server (Node.js)

```javascript
// Server
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

const rooms = new Map(); // roomId -> Set of clients

wss.on('connection', (ws) => {
  console.log('Client connected');

  ws.on('message', (data) => {
    const message = JSON.parse(data);

    switch (message.type) {
      case 'join':
        // Join room
        if (!rooms.has(message.room)) {
          rooms.set(message.room, new Set());
        }
        rooms.get(message.room).add(ws);
        ws.room = message.room;

        // Notify others in room
        broadcast(message.room, ws, {
          type: 'user-joined',
          userId: message.userId
        });
        break;

      case 'offer':
      case 'answer':
      case 'ice-candidate':
        // Forward to specific peer or broadcast
        if (message.target) {
          sendToUser(message.target, message);
        } else {
          broadcast(ws.room, ws, message);
        }
        break;

      case 'leave':
        leaveRoom(ws);
        break;
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
    leaveRoom(ws);
  });
});

function broadcast(room, sender, message) {
  if (!rooms.has(room)) return;

  rooms.get(room).forEach(client => {
    if (client !== sender && client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(message));
    }
  });
}

function leaveRoom(ws) {
  if (ws.room && rooms.has(ws.room)) {
    rooms.get(ws.room).delete(ws);
    broadcast(ws.room, ws, {
      type: 'user-left',
      userId: ws.userId
    });
  }
}

console.log('Signaling server running on ws://localhost:8080');
```

### Client-Side Signaling

```javascript
// Client
class SignalingClient {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.handlers = new Map();

    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const handler = this.handlers.get(message.type);
      if (handler) {
        handler(message);
      }
    };

    this.socket.onopen = () => {
      console.log('Signaling connected');
    };

    this.socket.onerror = (error) => {
      console.error('Signaling error:', error);
    };

    this.socket.onclose = () => {
      console.log('Signaling disconnected');
    };
  }

  on(type, handler) {
    this.handlers.set(type, handler);
  }

  send(message) {
    this.socket.send(JSON.stringify(message));
  }

  join(room, userId) {
    this.send({ type: 'join', room, userId });
  }

  sendOffer(offer, target) {
    this.send({ type: 'offer', sdp: offer, target });
  }

  sendAnswer(answer, target) {
    this.send({ type: 'answer', sdp: answer, target });
  }

  sendIceCandidate(candidate, target) {
    this.send({ type: 'ice-candidate', candidate, target });
  }
}

// Usage
const signaling = new SignalingClient('ws://localhost:8080');

signaling.on('offer', handleOffer);
signaling.on('answer', handleAnswer);
signaling.on('ice-candidate', handleIceCandidate);

signaling.join('room123', 'user1');
```

## Complete WebRTC Example

### Simple Video Chat Application

```javascript
class WebRTCVideoChat {
  constructor(signalingUrl) {
    this.signaling = new SignalingClient(signalingUrl);
    this.peerConnection = null;
    this.localStream = null;

    this.setupSignaling();
  }

  setupSignaling() {
    this.signaling.on('offer', async (message) => {
      await this.handleOffer(message.sdp, message.sender);
    });

    this.signaling.on('answer', async (message) => {
      await this.handleAnswer(message.sdp);
    });

    this.signaling.on('ice-candidate', async (message) => {
      await this.handleIceCandidate(message.candidate);
    });

    this.signaling.on('user-joined', (message) => {
      console.log('User joined:', message.userId);
      // Initiate call if you're the caller
    });
  }

  async start(localVideoElement, remoteVideoElement) {
    // Get local media
    this.localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: true
    });

    localVideoElement.srcObject = this.localStream;

    // Create peer connection
    this.peerConnection = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' }
      ]
    });

    // Add local stream
    this.localStream.getTracks().forEach(track => {
      this.peerConnection.addTrack(track, this.localStream);
    });

    // Handle remote stream
    this.peerConnection.ontrack = (event) => {
      remoteVideoElement.srcObject = event.streams[0];
    };

    // Handle ICE candidates
    this.peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        this.signaling.sendIceCandidate(event.candidate);
      }
    };

    // Monitor connection
    this.peerConnection.onconnectionstatechange = () => {
      console.log('Connection state:',
        this.peerConnection.connectionState);
    };
  }

  async call() {
    // Create and send offer
    const offer = await this.peerConnection.createOffer();
    await this.peerConnection.setLocalDescription(offer);
    this.signaling.sendOffer(offer);
  }

  async handleOffer(offer, sender) {
    await this.peerConnection.setRemoteDescription(
      new RTCSessionDescription(offer)
    );

    const answer = await this.peerConnection.createAnswer();
    await this.peerConnection.setLocalDescription(answer);

    this.signaling.sendAnswer(answer, sender);
  }

  async handleAnswer(answer) {
    await this.peerConnection.setRemoteDescription(
      new RTCSessionDescription(answer)
    );
  }

  async handleIceCandidate(candidate) {
    await this.peerConnection.addIceCandidate(
      new RTCIceCandidate(candidate)
    );
  }

  hangup() {
    if (this.peerConnection) {
      this.peerConnection.close();
      this.peerConnection = null;
    }

    if (this.localStream) {
      this.localStream.getTracks().forEach(track => track.stop());
      this.localStream = null;
    }
  }

  toggleAudio() {
    const audioTrack = this.localStream.getAudioTracks()[0];
    audioTrack.enabled = !audioTrack.enabled;
    return audioTrack.enabled;
  }

  toggleVideo() {
    const videoTrack = this.localStream.getVideoTracks()[0];
    videoTrack.enabled = !videoTrack.enabled;
    return videoTrack.enabled;
  }
}

// Usage
const chat = new WebRTCVideoChat('ws://localhost:8080');

const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');

await chat.start(localVideo, remoteVideo);
chat.signaling.join('room123', 'user1');

// When ready to call
document.getElementById('callButton').onclick = () => chat.call();
document.getElementById('hangupButton').onclick = () => chat.hangup();
document.getElementById('muteButton').onclick = () => chat.toggleAudio();
document.getElementById('videoButton').onclick = () => chat.toggleVideo();
```

## Media Codecs

### Audio Codecs

```
Opus (Preferred)
- Bitrate: 6-510 kbps
- Latency: 5-66.5 ms
- Best quality and efficiency
- Supports stereo and mono
- Adaptive bitrate

G.711 (PCMU/PCMA)
- Bitrate: 64 kbps
- Latency: Low
- Widely supported
- Lower quality than Opus

iSAC
- Bitrate: 10-32 kbps
- Adaptive bitrate
- Good for low bandwidth

iLBC
- Bitrate: 13.33 or 15.2 kbps
- Packet loss resilience
- Voice only
```

### Video Codecs

```
VP8 (Mandatory in WebRTC)
- Open source
- Good quality
- Hardware acceleration common
- Bitrate: 100-2000 kbps typically

VP9 (Better than VP8)
- 50% better compression than VP8
- Supports 4K
- Lower bandwidth usage
- Newer, less hardware support

H.264 (Most compatible)
- Patent-encumbered
- Excellent hardware support
- Multiple profiles (Baseline, Main, High)
- Most widely supported

AV1 (Future)
- Best compression
- Open source
- Still emerging
- Limited hardware support
```

### Codec Selection

```javascript
// Prefer specific codec
function preferCodec(sdp, codecName) {
  const lines = sdp.split('\n');
  const mLineIndex = lines.findIndex(line => line.startsWith('m=video'));
  if (mLineIndex === -1) return sdp;

  const codecRegex = new RegExp(`rtpmap:(\\d+) ${codecName}`, 'i');
  const codecPayload = lines
    .find(line => codecRegex.test(line))
    ?.match(codecRegex)?.[1];

  if (!codecPayload) return sdp;

  const mLine = lines[mLineIndex].split(' ');
  const codecs = mLine.slice(3);

  // Move preferred codec to front
  const newCodecs = [
    codecPayload,
    ...codecs.filter(c => c !== codecPayload)
  ];

  mLine.splice(3, codecs.length, ...newCodecs);
  lines[mLineIndex] = mLine.join(' ');

  return lines.join('\n');
}

// Usage
const offer = await peerConnection.createOffer();
offer.sdp = preferCodec(offer.sdp, 'VP9');
await peerConnection.setLocalDescription(offer);
```

## Quality Adaptation

### Bandwidth Estimation

```javascript
// Monitor bandwidth
peerConnection.getStats().then(stats => {
  stats.forEach(report => {
    if (report.type === 'candidate-pair' && report.state === 'succeeded') {
      console.log('Available bandwidth:',
        report.availableOutgoingBitrate);
      console.log('Current bandwidth:',
        report.currentRoundTripTime);
    }

    if (report.type === 'inbound-rtp' && report.mediaType === 'video') {
      console.log('Bytes received:', report.bytesReceived);
      console.log('Packets lost:', report.packetsLost);
      console.log('Jitter:', report.jitter);
    }
  });
});

// Periodic monitoring
setInterval(async () => {
  const stats = await peerConnection.getStats();
  analyzeStats(stats);
}, 1000);
```

### Simulcast (Multiple Qualities)

```javascript
// Sender: Send multiple resolutions
const sender = peerConnection
  .getSenders()
  .find(s => s.track.kind === 'video');

const parameters = sender.getParameters();
if (!parameters.encodings) {
  parameters.encodings = [
    { rid: 'h', maxBitrate: 1500000 },  // High quality
    { rid: 'm', maxBitrate: 600000, scaleResolutionDownBy: 2 },  // Medium
    { rid: 'l', maxBitrate: 200000, scaleResolutionDownBy: 4 }   // Low
  ];
}

await sender.setParameters(parameters);

// Receiver: Select layer
const receiver = peerConnection
  .getReceivers()
  .find(r => r.track.kind === 'video');

// Request specific layer
receiver.getParameters().encodings = [
  { active: true, rid: 'm' }  // Request medium quality
];
```

### Manual Bitrate Control

```javascript
async function setMaxBitrate(peerConnection, maxBitrate) {
  const sender = peerConnection
    .getSenders()
    .find(s => s.track.kind === 'video');

  const parameters = sender.getParameters();

  if (!parameters.encodings) {
    parameters.encodings = [{}];
  }

  parameters.encodings[0].maxBitrate = maxBitrate;

  await sender.setParameters(parameters);
  console.log(`Set max bitrate to ${maxBitrate} bps`);
}

// Usage
setMaxBitrate(peerConnection, 500000); // 500 kbps
```

## Data Channels Use Cases

### File Transfer

```javascript
class FileTransfer {
  constructor(dataChannel) {
    this.channel = dataChannel;
    this.chunkSize = 16384; // 16 KB chunks
  }

  async sendFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const totalChunks = Math.ceil(arrayBuffer.byteLength / this.chunkSize);

    // Send metadata
    this.channel.send(JSON.stringify({
      type: 'file-start',
      name: file.name,
      size: file.size,
      totalChunks: totalChunks
    }));

    // Send chunks
    for (let i = 0; i < totalChunks; i++) {
      const start = i * this.chunkSize;
      const end = Math.min(start + this.chunkSize, arrayBuffer.byteLength);
      const chunk = arrayBuffer.slice(start, end);

      // Wait if buffer is filling up
      while (this.channel.bufferedAmount > this.chunkSize * 10) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      this.channel.send(chunk);

      // Progress update
      const progress = ((i + 1) / totalChunks * 100).toFixed(1);
      console.log(`Sending: ${progress}%`);
    }

    // Send completion
    this.channel.send(JSON.stringify({ type: 'file-end' }));
  }

  receiveFile(onProgress, onComplete) {
    const chunks = [];
    let metadata = null;

    this.channel.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const message = JSON.parse(event.data);

        if (message.type === 'file-start') {
          metadata = message;
          chunks.length = 0;
        } else if (message.type === 'file-end') {
          const blob = new Blob(chunks);
          onComplete(blob, metadata);
        }
      } else {
        // Binary chunk
        chunks.push(event.data);

        if (metadata) {
          const progress = (chunks.length / metadata.totalChunks * 100)
            .toFixed(1);
          onProgress(progress);
        }
      }
    };
  }
}

// Usage
const fileTransfer = new FileTransfer(dataChannel);

// Sender
document.getElementById('fileInput').onchange = async (e) => {
  const file = e.target.files[0];
  await fileTransfer.sendFile(file);
};

// Receiver
fileTransfer.receiveFile(
  (progress) => console.log(`Receiving: ${progress}%`),
  (blob, metadata) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = metadata.name;
    a.click();
  }
);
```

### Gaming/Real-time Data

```javascript
class GameDataChannel {
  constructor(dataChannel) {
    this.channel = dataChannel;
    this.channel.binaryType = 'arraybuffer';

    // Unreliable, unordered for low latency
    this.channel = peerConnection.createDataChannel('game', {
      ordered: false,
      maxRetransmits: 0
    });
  }

  sendPlayerPosition(x, y, angle) {
    const buffer = new ArrayBuffer(12);
    const view = new DataView(buffer);

    view.setFloat32(0, x, true);
    view.setFloat32(4, y, true);
    view.setFloat32(8, angle, true);

    this.channel.send(buffer);
  }

  onPlayerPosition(callback) {
    this.channel.onmessage = (event) => {
      const view = new DataView(event.data);

      const x = view.getFloat32(0, true);
      const y = view.getFloat32(4, true);
      const angle = view.getFloat32(8, true);

      callback(x, y, angle);
    };
  }
}

// Usage
const gameChannel = new GameDataChannel(dataChannel);

// Send position 60 times per second
setInterval(() => {
  gameChannel.sendPlayerPosition(
    player.x,
    player.y,
    player.angle
  );
}, 1000 / 60);

gameChannel.onPlayerPosition((x, y, angle) => {
  updateRemotePlayer(x, y, angle);
});
```

## Security Considerations

### Encryption

```
WebRTC Security Stack:

Application Data
       ↓
SRTP (Secure RTP)
  - Encrypts media (audio/video)
  - AES encryption
  - HMAC authentication
       ↓
DTLS (Datagram TLS)
  - Encrypts data channels
  - Key exchange for SRTP
  - Certificate verification
       ↓
UDP/TCP Transport

All WebRTC traffic is encrypted!
No option for unencrypted communication.
```

### Certificate Verification

```javascript
// Verify peer certificate fingerprint
peerConnection.onicecandidate = (event) => {
  if (event.candidate === null) {
    // Get local certificate
    peerConnection.getConfiguration().certificates.forEach(cert => {
      cert.getFingerprints().forEach(fingerprint => {
        console.log('Local fingerprint:', fingerprint);
        // Send to peer via secure signaling
        // Peer should verify this matches SDP
      });
    });
  }
};

// Check SDP fingerprint matches expected
function verifySdpFingerprint(sdp, expectedFingerprint) {
  const fingerprintMatch = sdp.match(/a=fingerprint:(\S+) (\S+)/);
  if (!fingerprintMatch) {
    throw new Error('No fingerprint in SDP');
  }

  const [, algorithm, fingerprint] = fingerprintMatch;

  if (fingerprint !== expectedFingerprint) {
    throw new Error('Fingerprint mismatch! Possible MITM attack.');
  }

  return true;
}
```

### Best Practices

```
1. Secure Signaling
   - Use TLS/WSS for signaling
   - Authenticate users
   - Verify peer identity

2. Certificate Pinning
   - Verify SDP fingerprints
   - Out-of-band verification if possible

3. Access Control
   - Verify room/session authorization
   - Implement user authentication
   - Rate limiting

4. Media Permissions
   - Request minimal permissions
   - Explain why access is needed
   - Allow users to deny

5. Privacy
   - Minimize data collection
   - No recording without consent
   - Clear privacy policy

6. Network Security
   - Use TURN with authentication
   - Restrict TURN access
   - Monitor for abuse
```

## Debugging and Troubleshooting

### Enable Debug Logs

```javascript
// Chrome: Enable WebRTC internals
// Navigate to: chrome://webrtc-internals

// Firefox: Enable logging
// Navigate to: about:webrtc

// Console logging
peerConnection.addEventListener('track', e => {
  console.log('Track event:', e);
});

peerConnection.addEventListener('icecandidate', e => {
  console.log('ICE candidate:', e.candidate);
});

peerConnection.addEventListener('icecandidateerror', e => {
  console.error('ICE candidate error:', e);
});

peerConnection.addEventListener('connectionstatechange', e => {
  console.log('Connection state:', peerConnection.connectionState);
});

peerConnection.addEventListener('iceconnectionstatechange', e => {
  console.log('ICE connection state:',
    peerConnection.iceConnectionState);
});
```

### Get Detailed Statistics

```javascript
async function getDetailedStats(peerConnection) {
  const stats = await peerConnection.getStats();
  const report = {};

  stats.forEach(stat => {
    if (stat.type === 'inbound-rtp' && stat.kind === 'video') {
      report.video = {
        bytesReceived: stat.bytesReceived,
        packetsReceived: stat.packetsReceived,
        packetsLost: stat.packetsLost,
        jitter: stat.jitter,
        frameWidth: stat.frameWidth,
        frameHeight: stat.frameHeight,
        framesPerSecond: stat.framesPerSecond,
        framesDecoded: stat.framesDecoded,
        framesDropped: stat.framesDropped
      };
    }

    if (stat.type === 'inbound-rtp' && stat.kind === 'audio') {
      report.audio = {
        bytesReceived: stat.bytesReceived,
        packetsReceived: stat.packetsReceived,
        packetsLost: stat.packetsLost,
        jitter: stat.jitter,
        audioLevel: stat.audioLevel
      };
    }

    if (stat.type === 'candidate-pair' && stat.state === 'succeeded') {
      report.connection = {
        localCandidateType: stat.localCandidateType,
        remoteCandidateType: stat.remoteCandidateType,
        currentRoundTripTime: stat.currentRoundTripTime,
        availableOutgoingBitrate: stat.availableOutgoingBitrate,
        bytesReceived: stat.bytesReceived,
        bytesSent: stat.bytesSent
      };
    }
  });

  return report;
}

// Monitor every second
setInterval(async () => {
  const stats = await getDetailedStats(peerConnection);
  console.table(stats);
}, 1000);
```

### Common Issues and Solutions

```
Issue: ICE connection fails
Solutions:
- Check STUN/TURN server configuration
- Verify firewall allows UDP traffic
- Add TURN server as fallback
- Check ICE candidate gathering

Issue: No video/audio
Solutions:
- Verify getUserMedia constraints
- Check browser permissions
- Verify tracks added to peer connection
- Check ontrack event handler

Issue: One-way audio/video
Solutions:
- Verify both peers add tracks
- Check SDP offer/answer exchange
- Verify both peers handle ontrack
- Check NAT/firewall rules

Issue: Poor quality
Solutions:
- Reduce resolution/bitrate
- Enable simulcast
- Check network bandwidth
- Monitor packet loss
- Verify codec support

Issue: High latency
Solutions:
- Use TURN server closer to users
- Enable unreliable data channels for gaming
- Reduce buffering
- Optimize codec settings
```

## Browser Support

```
Desktop Browsers:
✓ Chrome 23+
✓ Firefox 22+
✓ Safari 11+
✓ Edge 79+ (Chromium-based)
✓ Opera 18+

Mobile Browsers:
✓ Chrome Android 28+
✓ Firefox Android 24+
✓ Safari iOS 11+
✓ Samsung Internet 4+

Feature Support:
- getUserMedia: All modern browsers
- RTCPeerConnection: All modern browsers
- RTCDataChannel: All modern browsers
- Screen sharing: Desktop only (most browsers)
- VP9 codec: Chrome, Firefox, Edge
- H.264 codec: All browsers (licensing)

Check: https://caniuse.com/rtcpeerconnection
```

## Performance Optimization

### Tips for Better Performance

```javascript
// 1. Reuse peer connections
const peerConnections = new Map();

function getOrCreatePeerConnection(peerId) {
  if (!peerConnections.has(peerId)) {
    peerConnections.set(peerId, createPeerConnection());
  }
  return peerConnections.get(peerId);
}

// 2. Batch ICE candidates (trickle ICE)
const pendingCandidates = [];

peerConnection.onicecandidate = (event) => {
  if (event.candidate) {
    pendingCandidates.push(event.candidate);

    // Send in batches
    if (pendingCandidates.length >= 5) {
      signaling.send({
        type: 'ice-candidates',
        candidates: pendingCandidates.splice(0)
      });
    }
  }
};

// 3. Use efficient codecs
// VP9 or H.264 for video, Opus for audio

// 4. Enable hardware acceleration
// Automatic in most browsers

// 5. Limit resolution based on network
async function adaptToNetwork(peerConnection) {
  const stats = await peerConnection.getStats();
  // Analyze and adjust bitrate/resolution
}

// 6. Use object fit for video elements
<video style="object-fit: cover;" />

// 7. Clean up resources
function cleanup() {
  localStream?.getTracks().forEach(track => track.stop());
  peerConnection?.close();
  dataChannel?.close();
}
```

## ELI10: WebRTC Explained Simply

WebRTC lets browsers talk directly to each other without a server in the middle:

### Traditional Communication
```
Your Browser → Server → Friend's Browser
- Everything goes through server
- Server sees all your data
- Costs more (server bandwidth)
- Higher latency
```

### WebRTC Communication
```
Your Browser ←→ Friend's Browser
- Direct connection (peer-to-peer)
- Server only introduces you
- Private (server can't see)
- Faster (no middleman)
```

### The Process
```
1. Get Permission
   "Can I use your camera and microphone?"

2. Signaling (Meeting)
   Server: "Hey Browser A, meet Browser B"
   Exchange: "Here's how to reach me"

3. ICE/STUN (Finding the Path)
   "What's my public address?"
   "Can we connect directly?"

4. Connection!
   Direct video/audio/data
   Encrypted automatically

5. If Direct Fails
   TURN server relays traffic
   Still encrypted
```

### Real-World Analogy
```
Traditional: Passing notes through teacher
WebRTC: Sitting next to friend and talking
Signaling: Teacher introduces you
STUN: Finding where each person sits
TURN: Using walkie-talkies if too far
```

## Further Resources

### Documentation
- [MDN WebRTC API](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
- [WebRTC Specification](https://www.w3.org/TR/webrtc/)
- [WebRTC Samples](https://webrtc.github.io/samples/)

### Tools
- [chrome://webrtc-internals](chrome://webrtc-internals) - Chrome debugging
- [about:webrtc](about:webrtc) - Firefox debugging
- [WebRTC Troubleshooter](https://test.webrtc.org/)

### Libraries
- [SimpleWebRTC](https://simplewebrtc.com/) - Simplified WebRTC
- [PeerJS](https://peerjs.com/) - Easy peer-to-peer
- [Janus Gateway](https://janus.conf.meetecho.com/) - WebRTC server
- [Kurento](https://www.kurento.org/) - Media server

### Testing
- [WebRTC Network Tester](https://networktest.twilio.com/)
- [STUN/TURN Server Test](https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/)

### Books
- *Real-Time Communication with WebRTC* by Salvatore Loreto
- *WebRTC Cookbook* by Andrii Sergiienko
- *High Performance Browser Networking* by Ilya Grigorik
