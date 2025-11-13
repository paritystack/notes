# ICE (Interactive Connectivity Establishment)

## Overview

ICE (Interactive Connectivity Establishment) is a framework used to establish peer-to-peer connections through NATs and firewalls. It's primarily used by WebRTC, VoIP applications, and other real-time communication systems to find the best path for connecting two endpoints on the internet.

## The NAT Problem

### Why ICE is Needed

```
Traditional Scenario:
┌──────────────┐                    ┌──────────────┐
│  Client A    │                    │  Client B    │
│ 10.0.0.5     │                    │ 192.168.1.10 │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │ NAT                          NAT  │
       │                                   │
┌──────▼───────┐                    ┌──────▼───────┐
│    Router    │                    │    Router    │
│ 203.0.113.5  │                    │ 198.51.100.3 │
└──────────────┘                    └──────────────┘
       │                                   │
       └───────────── Internet ────────────┘

Problems:
1. Client A only knows its private IP (10.0.0.5)
2. Client B can't reach 10.0.0.5 (not routable)
3. Client A doesn't know Client B's public IP
4. Routers block unsolicited incoming connections

ICE Solution:
1. Discover public IPs (STUN)
2. Try multiple connection paths
3. Use relay as fallback (TURN)
4. Select best working path
```

## How ICE Works

### The ICE Process

```
1. Gather Candidates
   Collect all possible ways to reach this peer:
   - Host candidate (local IP)
   - Server reflexive (public IP from STUN)
   - Relayed candidate (TURN relay)

2. Exchange Candidates
   Send candidates to remote peer via signaling

3. Pair Candidates
   Create pairs: local candidate + remote candidate

4. Check Connectivity
   Test all pairs in priority order

5. Select Best Pair
   Use the pair with highest priority that works

6. Keep Alive
   Maintain selected connection
```

### Detailed Flow Diagram

```
Peer A                    Signaling Server                    Peer B
  |                              |                               |
  |──① Gather Candidates         |                               |
  |    - host                    |                               |
  |    - srflx (STUN)            |                               |
  |    - relay (TURN)            |                               |
  |                              |                               |
  |──② Send Candidates──────────►|                               |
  |    via SDP offer             |                               |
  |                              |──③ Forward Candidates────────►|
  |                              |                               |
  |                              |                               |──④ Gather Candidates
  |                              |                               |    - host
  |                              |                               |    - srflx (STUN)
  |                              |                               |    - relay (TURN)
  |                              |                               |
  |                              |◄──⑤ Send Candidates───────────|
  |                              |    via SDP answer             |
  |◄─⑥ Forward Candidates────────|                               |
  |                              |                               |
  |──⑦ Connectivity Checks───────────────────────────────────────►|
  |   (test all candidate pairs)                                 |
  |                              |                               |
  |◄─⑧ Connectivity Checks───────────────────────────────────────|
  |   (test all candidate pairs)                                 |
  |                              |                               |
  |──⑨ Nomination (best pair)────────────────────────────────────►|
  |                              |                               |
  |◄─⑩ Confirmation──────────────────────────────────────────────|
  |                              |                               |
  |══⑪ Media/Data Flow ═══════════════════════════════════════════|
  |   (using selected pair)                                      |
```

## ICE Candidate Types

### 1. Host Candidate

```
Local network interface address:

Type: host
Address: 10.0.0.5:54321
Foundation: 1

Characteristics:
- Actual IP address of the interface
- No NAT traversal
- Works only on same local network
- Lowest latency
- Priority: High for local connections

Example:
candidate:1 1 UDP 2130706431 10.0.0.5 54321 typ host

Use case:
- Devices on same LAN
- No NAT between peers
```

### 2. Server Reflexive Candidate (srflx)

```
Public IP address as seen by STUN server:

Type: srflx
Address: 203.0.113.5:61234
Related: 10.0.0.5:54321
Foundation: 2

Characteristics:
- Discovered via STUN server
- Public IP:port after NAT
- Most common for internet connections
- Priority: Medium-High

Example:
candidate:2 1 UDP 1694498815 203.0.113.5 61234 typ srflx
    raddr 10.0.0.5 rport 54321

Discovery:
1. Client sends STUN request from 10.0.0.5:54321
2. STUN server sees request from 203.0.113.5:61234
3. STUN responds with "Your IP:port is 203.0.113.5:61234"
4. Client creates srflx candidate

Use case:
- Typical internet connections
- NAT traversal
- Peer-to-peer over internet
```

### 3. Peer Reflexive Candidate (prflx)

```
Public IP discovered during connectivity checks:

Type: prflx
Address: 203.0.113.5:61235
Foundation: 3

Characteristics:
- Discovered during checks (not via STUN)
- Learned from peer's connectivity checks
- Alternative to srflx
- Priority: Medium

Example:
candidate:3 1 UDP 1862270975 203.0.113.5 61235 typ prflx
    raddr 10.0.0.5 rport 54321

Discovery:
1. Peer B sends connectivity check
2. Peer A receives from unexpected address
3. Peer A learns new reflexive address
4. Creates prflx candidate

Use case:
- Discovered during connection attempts
- Additional connectivity options
```

### 4. Relayed Candidate (relay)

```
Address on TURN relay server:

Type: relay
Address: 198.51.100.10:55555
Related: 203.0.113.5:61234
Foundation: 4

Characteristics:
- Allocated on TURN server
- Relay forwards all traffic
- Works through any NAT/firewall
- Highest latency and bandwidth cost
- Priority: Low (fallback)

Example:
candidate:4 1 UDP 16777215 198.51.100.10 55555 typ relay
    raddr 203.0.113.5 rport 61234

Discovery:
1. Client requests allocation from TURN server
2. TURN allocates 198.51.100.10:55555
3. Client creates relay candidate
4. All traffic flows through TURN

Use case:
- Symmetric NATs
- Restrictive firewalls
- When direct connection fails
- Corporate networks
```

## Candidate Priority

### Priority Calculation

```
Priority Formula:
priority = (2^24 × type preference) +
           (2^8 × local preference) +
           (256 - component ID)

Type Preference (higher = better):
- host: 126
- prflx: 110
- srflx: 100
- relay: 0

Local Preference:
- Higher for interfaces you prefer
- Typically: 65535 for best interface

Component ID:
- 1 for RTP (main media)
- 2 for RTCP (control)

Example Calculations:

Host candidate:
(2^24 × 126) + (2^8 × 65535) + (256 - 1)
= 2113667071

Srflx candidate:
(2^24 × 100) + (2^8 × 65535) + (256 - 1)
= 1694498815

Relay candidate:
(2^24 × 0) + (2^8 × 65535) + (256 - 1)
= 16777215
```

### Priority in Practice

```
Sorted by priority (high to low):

1. host (LAN)           Priority: 2113667071
   - Try first
   - Works if same network
   - Lowest latency

2. srflx (NAT)          Priority: 1694498815
   - Try second
   - Works through NAT
   - Good latency

3. prflx (Discovered)   Priority: 1862270975
   - Try if discovered
   - Alternative path

4. relay (TURN)         Priority: 16777215
   - Try last
   - Always works
   - Higher latency/cost

Best path:
host → host (LAN)
host → srflx (NAT traversal)
srflx → srflx (Both behind NAT)
relay → relay (Fallback)
```

## Candidate Gathering

### ICE Gathering States

```javascript
// ICE gathering state machine

peerConnection.onicegatheringstatechange = () => {
  console.log('ICE gathering state:',
    peerConnection.iceGatheringState);
};

/*
States:

1. new
   - Initial state
   - No gathering started

2. gathering
   - Actively gathering candidates
   - STUN/TURN requests in progress

3. complete
   - All candidates gathered
   - Ready to connect
*/

// Monitor gathering
peerConnection.addEventListener('icecandidate', (event) => {
  if (event.candidate) {
    console.log('New candidate:', event.candidate);
    // Send to remote peer
  } else {
    console.log('Gathering complete');
    // All candidates collected
  }
});
```

### Gathering Configuration

```javascript
// Configure ICE servers
const configuration = {
  iceServers: [
    // Public STUN servers (Google)
    {
      urls: 'stun:stun.l.google.com:19302'
    },
    {
      urls: 'stun:stun1.l.google.com:19302'
    },

    // STUN server (custom)
    {
      urls: 'stun:stun.example.com:3478'
    },

    // TURN server (UDP and TCP)
    {
      urls: [
        'turn:turn.example.com:3478',
        'turn:turn.example.com:3478?transport=tcp'
      ],
      username: 'user',
      credential: 'password',
      credentialType: 'password'
    },

    // TURN server (TLS)
    {
      urls: 'turns:turn.example.com:5349',
      username: 'user',
      credential: 'password'
    }
  ],

  // ICE transport policy
  iceTransportPolicy: 'all', // 'all' or 'relay'
  // 'all': Try all candidates
  // 'relay': Only use TURN (force relay)

  // Candidate pool size
  iceCandidatePoolSize: 10
  // Pre-allocate TURN allocations
  // Higher = faster but more resources
};

const peerConnection = new RTCPeerConnection(configuration);
```

### Trickle ICE

Instead of waiting for all candidates, send them as discovered:

```javascript
// Sender: Send candidates as discovered
peerConnection.onicecandidate = (event) => {
  if (event.candidate) {
    // Send immediately (trickle)
    signaling.send({
      type: 'ice-candidate',
      candidate: event.candidate
    });
  } else {
    // Signal end of candidates
    signaling.send({
      type: 'ice-candidate',
      candidate: null
    });
  }
};

// Receiver: Add candidates as received
signaling.on('ice-candidate', async (message) => {
  if (message.candidate) {
    try {
      await peerConnection.addIceCandidate(
        new RTCIceCandidate(message.candidate)
      );
    } catch (error) {
      console.error('Error adding candidate:', error);
    }
  } else {
    // End of candidates
    console.log('Remote candidate gathering complete');
  }
});

Benefits:
- Faster connection establishment
- Start checks before all candidates gathered
- Improved user experience
```

## Connectivity Checks

### STUN Binding Requests

ICE uses STUN messages to test connectivity:

```
Connectivity Check Process:

1. Create Candidate Pairs
   Local Candidate    Remote Candidate         Pair
   10.0.0.5:54321  +  192.168.1.10:44444  =  Pair 1
   10.0.0.5:54321  +  198.51.100.3:55555  =  Pair 2
   203.0.113.5:61234 + 192.168.1.10:44444 =  Pair 3
   203.0.113.5:61234 + 198.51.100.3:55555 =  Pair 4
   198.51.100.10:55555 + 192.168.1.10:44444 = Pair 5
   198.51.100.10:55555 + 198.51.100.3:55555 = Pair 6

2. Sort Pairs by Priority
   Priority = min(local priority, remote priority)

3. Send STUN Binding Request
   From: Local candidate
   To: Remote candidate
   Message: STUN Binding Request
   Attributes:
   - USERNAME: ice-ufrag
   - PRIORITY: candidate priority
   - ICE-CONTROLLING or ICE-CONTROLLED
   - MESSAGE-INTEGRITY: HMAC

4. Receive STUN Binding Response
   From: Remote candidate
   To: Local candidate
   Message: STUN Binding Response (Success)
   Attributes:
   - XOR-MAPPED-ADDRESS
   - MESSAGE-INTEGRITY

5. Mark Pair as Valid
   If response received, pair works!

6. Nominate Best Pair
   Controlling agent nominates highest priority valid pair
```

### Controlling vs Controlled

```
ICE Roles:

Controlling Agent (Caller):
- Makes final decision on selected pair
- Sends nomination
- Usually the offerer

Controlled Agent (Callee):
- Responds to checks
- Accepts nomination
- Usually the answerer

Role Conflict Resolution:
If both think they're controlling:
- Compare ICE tie-breaker values
- Higher value becomes controlling
- Lower value becomes controlled

Attribute:
ICE-CONTROLLING: <tie-breaker>
  or
ICE-CONTROLLED: <tie-breaker>
```

### Connectivity Check States

```
Candidate Pair States:

1. Frozen
   - Initial state
   - Waiting to be checked
   - Not yet sent binding request

2. Waiting
   - Ready to check
   - Will check soon
   - Waiting for resources

3. In Progress
   - Binding request sent
   - Waiting for response
   - Timeout if no response

4. Succeeded
   - Binding response received
   - Pair is valid
   - Can be used for media

5. Failed
   - No response (timeout)
   - Or error response
   - Cannot use this pair

State Machine:
Frozen → Waiting → In Progress → Succeeded ✓
                               → Failed ✗
```

## Connection States

### ICE Connection States

```javascript
peerConnection.oniceconnectionstatechange = () => {
  console.log('ICE connection state:',
    peerConnection.iceConnectionState);

  switch (peerConnection.iceConnectionState) {
    case 'new':
      // Initial state, gathering not started
      console.log('ICE gathering starting...');
      break;

    case 'checking':
      // Checking candidate pairs
      console.log('Testing connectivity...');
      break;

    case 'connected':
      // At least one working pair found
      console.log('Connection established!');
      break;

    case 'completed':
      // All checks done, best pair selected
      console.log('ICE completed');
      break;

    case 'failed':
      // All pairs failed
      console.error('Connection failed');
      // Fallback: restart ICE or use TURN
      handleConnectionFailure();
      break;

    case 'disconnected':
      // Lost connectivity (temporary?)
      console.warn('Connection lost, attempting to reconnect...');
      break;

    case 'closed':
      // Connection closed
      console.log('Connection closed');
      break;
  }
};

// Overall connection state (combines ICE + DTLS)
peerConnection.onconnectionstatechange = () => {
  console.log('Connection state:',
    peerConnection.connectionState);
  // States: new, connecting, connected, disconnected, failed, closed
};
```

## ICE Restart

When connection fails or degrades:

```javascript
// Restart ICE
async function restartIce(peerConnection) {
  console.log('Restarting ICE...');

  // Create new offer with iceRestart option
  const offer = await peerConnection.createOffer({
    iceRestart: true
  });

  await peerConnection.setLocalDescription(offer);

  // Send new offer to peer
  signaling.send({
    type: 'offer',
    sdp: offer
  });

  // New candidates will be gathered
  // New connectivity checks will be performed
}

// Trigger restart on failure
peerConnection.oniceconnectionstatechange = () => {
  if (peerConnection.iceConnectionState === 'failed') {
    console.error('ICE failed, restarting...');
    restartIce(peerConnection);
  }
};

// Or restart on disconnection timeout
let disconnectTimeout;

peerConnection.oniceconnectionstatechange = () => {
  if (peerConnection.iceConnectionState === 'disconnected') {
    // Wait 5 seconds before restart
    disconnectTimeout = setTimeout(() => {
      if (peerConnection.iceConnectionState !== 'connected') {
        restartIce(peerConnection);
      }
    }, 5000);
  } else if (peerConnection.iceConnectionState === 'connected') {
    clearTimeout(disconnectTimeout);
  }
};
```

## Advanced ICE Features

### ICE Lite

Simplified ICE for servers:

```
ICE Lite:
- Only responds to checks (doesn't initiate)
- Only gathers host candidates
- Simpler implementation
- Used by servers (not browsers)

Standard ICE vs ICE Lite:

Standard ICE (Full Agent):
- Gathers all candidate types
- Sends connectivity checks
- Can be controlling or controlled
- Used by clients

ICE Lite:
- Only host candidates
- Only responds to checks
- Always controlled role
- Used by servers

Example: Media server
- Server uses ICE Lite
- Client uses full ICE
- Client initiates all checks
- Server just responds
```

### Consent Freshness

Keep-alive mechanism:

```
Purpose:
- Verify peer still wants to receive
- Detect path changes
- Prevent unwanted traffic

Process:
1. Every 5 seconds, send STUN Binding Request
2. Peer responds with Binding Response
3. If no response for 30 seconds → disconnected

STUN Binding Request:
- From selected local candidate
- To selected remote candidate
- Authenticated (MESSAGE-INTEGRITY)

Failure:
- 30 seconds without response
- ICE state → disconnected
- May trigger ICE restart

Automatic in WebRTC:
- Browser handles automatically
- No manual intervention needed
```

### Aggressive Nomination

Faster connection establishment:

```
Regular Nomination:
1. Check all pairs
2. Wait for all checks to complete
3. Nominate best pair
   Time: Slow but optimal

Aggressive Nomination:
1. Check pairs in priority order
2. Nominate first working pair immediately
3. Continue checking in background
   Time: Fast but may not be optimal

Trade-off:
- Aggressive: Faster connection, may not be best path
- Regular: Slower connection, guaranteed best path

Most WebRTC implementations use regular nomination
for better quality.
```

## Debugging ICE

### Analyzing ICE Candidates

```javascript
// Log all candidates
peerConnection.onicecandidate = (event) => {
  if (event.candidate) {
    const candidate = event.candidate.candidate;
    console.log('Candidate:', candidate);

    // Parse candidate
    const parts = candidate.split(' ');
    const parsed = {
      foundation: parts[0].split(':')[1],
      component: parts[1],
      protocol: parts[2],
      priority: parts[3],
      ip: parts[4],
      port: parts[5],
      type: parts[7],
      relAddr: parts[9],
      relPort: parts[11]
    };

    console.log('Parsed:', parsed);

    // Identify issues
    if (parsed.type === 'relay') {
      console.warn('Using TURN relay (may indicate NAT/firewall issues)');
    }
    if (parsed.protocol === 'tcp') {
      console.warn('Using TCP (UDP may be blocked)');
    }
  }
};

// Monitor selected pair
async function getSelectedPair(peerConnection) {
  const stats = await peerConnection.getStats();

  stats.forEach(report => {
    if (report.type === 'candidate-pair' && report.state === 'succeeded') {
      console.log('Selected pair:');
      console.log('  Local:', report.localCandidateId);
      console.log('  Remote:', report.remoteCandidateId);
      console.log('  State:', report.state);
      console.log('  Priority:', report.priority);
      console.log('  RTT:', report.currentRoundTripTime);
      console.log('  Bytes sent:', report.bytesSent);
      console.log('  Bytes received:', report.bytesReceived);
    }
  });
}

// Check every second
setInterval(() => getSelectedPair(peerConnection), 1000);
```

### Common ICE Issues

```
Issue: No candidates gathered
Cause: Missing or incorrect STUN/TURN config
Solution: Verify iceServers configuration

Issue: Only relay candidates
Cause: Restrictive firewall blocks UDP
Solution:
- Enable UDP ports
- Use TURN with TCP
- Check firewall rules

Issue: Connectivity checks fail
Cause: Firewall blocks STUN packets
Solution:
- Allow UDP 3478 (STUN)
- Allow UDP 49152-65535 (RTP)
- Use TURN as fallback

Issue: Connection works then fails
Cause: NAT binding timeout
Solution:
- Shorter keep-alive interval
- Use consent freshness
- ICE restart on failure

Issue: High latency
Cause: Using TURN relay when direct possible
Solution:
- Verify STUN server reachable
- Check NAT type (symmetric NAT requires TURN)
- Verify candidate priorities

Issue: One-way media
Cause: Asymmetric connectivity
Solution:
- Check firewall rules both directions
- Verify both peers send candidates
- Use TURN if necessary
```

### ICE Testing Tools

```bash
# Test STUN server
stunclient stun.l.google.com 19302

# Output shows:
# - Your public IP
# - NAT type
# - Whether server is reachable

# Test TURN server
turnutils_uclient -v -u username -w password \
  turn.example.com

# Test with ICE
# Browser: chrome://webrtc-internals
# - View all ICE candidates
# - See connectivity checks
# - Monitor selected pair

# Command-line ICE test
npm install -g wrtc-ice-tester
wrtc-ice-tester --stun stun:stun.l.google.com:19302

# Network debugging
tcpdump -i any -n udp port 3478 or portrange 49152-65535

# WebRTC test page
https://test.webrtc.org/
```

## ICE Configuration Examples

### Basic Configuration

```javascript
// Minimal config (STUN only)
const config = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' }
  ]
};

// With TURN fallback
const config = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    {
      urls: 'turn:turn.example.com:3478',
      username: 'user',
      credential: 'pass'
    }
  ]
};

// Production config (redundancy)
const config = {
  iceServers: [
    // Multiple STUN servers
    { urls: 'stun:stun1.example.com:3478' },
    { urls: 'stun:stun2.example.com:3478' },

    // TURN with TCP fallback
    {
      urls: [
        'turn:turn.example.com:3478',           // UDP
        'turn:turn.example.com:3478?transport=tcp', // TCP
        'turns:turn.example.com:5349'           // TLS
      ],
      username: 'user',
      credential: 'pass'
    }
  ],
  iceCandidatePoolSize: 10,
  iceTransportPolicy: 'all' // Try everything
};
```

### Dynamic TURN Credentials

```javascript
// Get temporary TURN credentials from your server
async function getTurnCredentials() {
  const response = await fetch('/api/turn-credentials', {
    headers: { 'Authorization': 'Bearer ' + token }
  });

  return await response.json();
  /*
  Returns:
  {
    urls: ['turn:turn.example.com:3478'],
    username: 'temporary-user-12345',
    credential: 'temporary-password',
    ttl: 86400  // 24 hours
  }
  */
}

// Use dynamic credentials
const turnCreds = await getTurnCredentials();

const config = {
  iceServers: [
    { urls: 'stun:stun.example.com:3478' },
    {
      urls: turnCreds.urls,
      username: turnCreds.username,
      credential: turnCreds.credential,
      credentialType: 'password'
    }
  ]
};

const pc = new RTCPeerConnection(config);
```

### Server-Side TURN Credential Generation

```javascript
// Node.js server
const crypto = require('crypto');

function generateTurnCredentials(username, secret, ttl = 86400) {
  const timestamp = Math.floor(Date.now() / 1000) + ttl;
  const turnUsername = `${timestamp}:${username}`;

  const hmac = crypto.createHmac('sha1', secret);
  hmac.update(turnUsername);
  const turnPassword = hmac.digest('base64');

  return {
    urls: [
      'turn:turn.example.com:3478',
      'turn:turn.example.com:3478?transport=tcp',
      'turns:turn.example.com:5349'
    ],
    username: turnUsername,
    credential: turnPassword,
    ttl: ttl
  };
}

// API endpoint
app.get('/api/turn-credentials', authenticate, (req, res) => {
  const credentials = generateTurnCredentials(
    req.user.id,
    process.env.TURN_SECRET,
    86400  // 24 hours
  );

  res.json(credentials);
});
```

## Performance Considerations

### Minimizing Connection Time

```javascript
// 1. Pre-gather candidates
const pc = new RTCPeerConnection({
  iceServers: [...],
  iceCandidatePoolSize: 10  // Pre-allocate TURN
});

// 2. Use trickle ICE (send candidates immediately)
pc.onicecandidate = (event) => {
  if (event.candidate) {
    signaling.send({ type: 'candidate', candidate: event.candidate });
  }
};

// 3. Start gathering early
await pc.setLocalDescription(await pc.createOffer());

// 4. Use multiple STUN servers (parallel queries)
const config = {
  iceServers: [
    { urls: 'stun:stun1.example.com:3478' },
    { urls: 'stun:stun2.example.com:3478' },
    { urls: 'stun:stun3.example.com:3478' }
  ]
};

// 5. Close old connection before creating new one
if (oldPeerConnection) {
  oldPeerConnection.close();
}

// Typical connection time:
// - LAN: 100-500ms
// - Internet (NAT): 1-3 seconds
// - TURN relay: 2-5 seconds
```

### Bandwidth Considerations

```
TURN Relay Bandwidth:

Scenario: 10 users in video call, all using TURN

Without TURN (P2P mesh):
Each user sends to 9 others directly
Total: 10 × 9 = 90 connections
User bandwidth: 9 video streams (upload + download)

With TURN relay:
Each user → TURN server → other users
Total: 10 × 9 through TURN
TURN bandwidth: 90 video streams
User bandwidth: Same (9 streams)

TURN costs:
- P2P: No relay bandwidth
- TURN: All traffic through server
- Solution: Use TURN only when necessary

Check if using TURN:
const stats = await pc.getStats();
stats.forEach(report => {
  if (report.type === 'local-candidate' &&
      report.candidateType === 'relay') {
    console.warn('Using TURN relay!');
  }
});
```

## NAT Types and ICE Success

### NAT Type Matrix

```
NAT Types (restrictiveness):

1. No NAT
   ✓ Direct connection
   Success rate: 100%

2. Full Cone NAT
   ✓ Any external host can connect
   Success rate: 100%

3. Restricted Cone NAT
   ✓ Can connect after outbound packet
   Success rate: 95%

4. Port Restricted Cone NAT
   ✓ Can connect after outbound to specific port
   Success rate: 90%

5. Symmetric NAT
   ✗ Different port for each destination
   Needs TURN relay
   Success rate: 100% (with TURN)

Connection Matrix:

                 Peer B
              Full  Restricted  Symmetric
Peer A
Full          ✓     ✓           ✓*
Restricted    ✓     ✓           ✓*
Symmetric     ✓*    ✓*          ✗ (need TURN)

✓ = Direct connection (STUN sufficient)
✓* = May need TURN
✗ = Requires TURN relay
```

## ELI10: ICE Explained Simply

ICE is like finding the best way to connect two phones:

### The Problem
```
You: Inside your house (private network)
Friend: Inside their house (private network)

Can't call directly:
- You don't know their full address
- Their house blocks unknown callers
- Your house blocks incoming calls
```

### ICE Solution
```
1. Find All Your Phone Numbers
   - Room extension (host): 101
   - House number (srflx): (555) 123-4567
   - Call-forwarding service (relay): (555) 999-0000

2. Share Numbers
   - You send your 3 numbers to friend
   - Friend sends their 3 numbers to you

3. Try All Combinations (9 attempts)
   Your 101 → Their 101 (works if same house)
   Your 101 → Their (555) 234-5678 (fails)
   Your (555) 123-4567 → Their (555) 234-5678 (works!)
   ... etc

4. Use Best Connection
   - Direct if possible (faster, cheaper)
   - Through forwarding if necessary (works always)

5. Keep Checking
   - "Are you still there?"
   - If no answer, try again
```

### Real Terms
```
House = Private network
House number = Public IP (STUN)
Call forwarding = Relay (TURN)
Trying combinations = Connectivity checks
Best connection = Selected candidate pair
```

## Further Resources

### Specifications
- [RFC 8445 - ICE](https://tools.ietf.org/html/rfc8445)
- [RFC 5389 - STUN](https://tools.ietf.org/html/rfc5389)
- [RFC 5766 - TURN](https://tools.ietf.org/html/rfc5766)
- [RFC 8656 - TURN Extensions](https://tools.ietf.org/html/rfc8656)

### Tools
- [Trickle ICE](https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/) - Test ICE candidates
- [WebRTC Troubleshooter](https://test.webrtc.org/) - Connection testing
- [NAT Type Test](https://www.nattest.net/) - Identify NAT type

### Debugging
- [chrome://webrtc-internals](chrome://webrtc-internals) - Chrome ICE debug
- [about:webrtc](about:webrtc) - Firefox ICE debug

### STUN/TURN Servers
- [Coturn](https://github.com/coturn/coturn) - Open source TURN server
- [Xirsys](https://xirsys.com/) - TURN server hosting
- [Twilio STUN/TURN](https://www.twilio.com/stun-turn) - Managed service

### Articles
- [WebRTC ICE](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Connectivity)
- [Understanding ICE](https://bloggeek.me/webrtc-ice/)
