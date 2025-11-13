# Web APIs

## Overview

Web APIs are interfaces provided by browsers that allow JavaScript to interact with browser features, device hardware, and web platform capabilities. These APIs enable rich, interactive web applications without requiring plugins or native code.

## Storage APIs

### Web Storage (localStorage & sessionStorage)

Simple key-value storage for strings:

```javascript
// LocalStorage (persists across sessions)
// Store data
localStorage.setItem('username', 'john_doe');
localStorage.setItem('theme', 'dark');

// Retrieve data
const username = localStorage.getItem('username');
console.log(username); // 'john_doe'

// Store objects (must serialize)
const user = { name: 'John', age: 30 };
localStorage.setItem('user', JSON.stringify(user));

// Retrieve objects (must parse)
const storedUser = JSON.parse(localStorage.getItem('user'));

// Remove item
localStorage.removeItem('username');

// Clear all
localStorage.clear();

// Get all keys
for (let i = 0; i < localStorage.length; i++) {
  const key = localStorage.key(i);
  console.log(key, localStorage.getItem(key));
}

// SessionStorage (cleared when tab closes)
sessionStorage.setItem('sessionId', '12345');
sessionStorage.getItem('sessionId');

// Storage event (listen for changes in other tabs)
window.addEventListener('storage', (e) => {
  console.log('Storage changed:');
  console.log('Key:', e.key);
  console.log('Old value:', e.oldValue);
  console.log('New value:', e.newValue);
  console.log('URL:', e.url);
});

// Limitations:
// - 5-10 MB limit (varies by browser)
// - Strings only (must serialize objects)
// - Synchronous (blocks main thread)
// - No expiration mechanism
```

### IndexedDB

Powerful client-side database for structured data:

```javascript
// Open database
const request = indexedDB.open('MyDatabase', 1);

// Create object stores (like tables)
request.onupgradeneeded = (event) => {
  const db = event.target.result;

  // Create object store
  const objectStore = db.createObjectStore('users', {
    keyPath: 'id',
    autoIncrement: true
  });

  // Create indexes
  objectStore.createIndex('email', 'email', { unique: true });
  objectStore.createIndex('name', 'name', { unique: false });

  console.log('Database upgraded');
};

request.onsuccess = (event) => {
  const db = event.target.result;
  console.log('Database opened successfully');

  // Add data
  const transaction = db.transaction(['users'], 'readwrite');
  const objectStore = transaction.objectStore('users');

  const user = {
    name: 'John Doe',
    email: 'john@example.com',
    age: 30
  };

  const addRequest = objectStore.add(user);

  addRequest.onsuccess = () => {
    console.log('User added with ID:', addRequest.result);
  };

  // Get data by key
  const getRequest = objectStore.get(1);
  getRequest.onsuccess = () => {
    console.log('User:', getRequest.result);
  };

  // Get by index
  const index = objectStore.index('email');
  const emailRequest = index.get('john@example.com');
  emailRequest.onsuccess = () => {
    console.log('User by email:', emailRequest.result);
  };

  // Update data
  const updateRequest = objectStore.put({
    id: 1,
    name: 'John Smith',
    email: 'john@example.com',
    age: 31
  });

  // Delete data
  const deleteRequest = objectStore.delete(1);

  // Get all data
  const getAllRequest = objectStore.getAll();
  getAllRequest.onsuccess = () => {
    console.log('All users:', getAllRequest.result);
  };

  // Cursor (iterate over records)
  const cursorRequest = objectStore.openCursor();
  cursorRequest.onsuccess = (event) => {
    const cursor = event.target.result;
    if (cursor) {
      console.log('Record:', cursor.value);
      cursor.continue(); // Move to next record
    }
  };
};

request.onerror = (event) => {
  console.error('Database error:', event.target.error);
};

// Promised-based wrapper (easier to use)
class IndexedDBHelper {
  constructor(dbName, version) {
    this.dbName = dbName;
    this.version = version;
    this.db = null;
  }

  async open(upgrade) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onupgradeneeded = (event) => {
        if (upgrade) {
          upgrade(event.target.result);
        }
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        resolve(this.db);
      };

      request.onerror = () => reject(request.error);
    });
  }

  async add(storeName, data) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.add(data);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async get(storeName, key) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(key);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getAll(storeName) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async update(storeName, data) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.put(data);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async delete(storeName, key) {
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
}

// Usage
const db = new IndexedDBHelper('MyApp', 1);

await db.open((database) => {
  const store = database.createObjectStore('users', { keyPath: 'id', autoIncrement: true });
  store.createIndex('email', 'email', { unique: true });
});

await db.add('users', { name: 'John', email: 'john@example.com' });
const users = await db.getAll('users');
console.log(users);
```

### Cache API

Store network requests and responses:

```javascript
// Open cache
const cache = await caches.open('my-cache-v1');

// Add to cache
await cache.add('/api/data');
await cache.addAll([
  '/styles.css',
  '/script.js',
  '/image.png'
]);

// Put custom response in cache
const response = new Response(JSON.stringify({ data: 'cached' }), {
  headers: { 'Content-Type': 'application/json' }
});
await cache.put('/api/custom', response);

// Get from cache
const cachedResponse = await cache.match('/api/data');
if (cachedResponse) {
  const data = await cachedResponse.json();
  console.log('Cached data:', data);
}

// Delete from cache
await cache.delete('/api/data');

// Get all keys
const keys = await cache.keys();
console.log('Cached URLs:', keys.map(req => req.url));

// Delete old caches
const cacheWhitelist = ['my-cache-v2'];
const cacheNames = await caches.keys();
await Promise.all(
  cacheNames.map(cacheName => {
    if (!cacheWhitelist.includes(cacheName)) {
      return caches.delete(cacheName);
    }
  })
);

// Cache-first strategy
async function fetchWithCache(url) {
  const cachedResponse = await caches.match(url);
  if (cachedResponse) {
    return cachedResponse;
  }

  const response = await fetch(url);
  const cache = await caches.open('my-cache-v1');
  cache.put(url, response.clone());
  return response;
}

// Network-first strategy
async function fetchNetworkFirst(url) {
  try {
    const response = await fetch(url);
    const cache = await caches.open('my-cache-v1');
    cache.put(url, response.clone());
    return response;
  } catch (error) {
    const cachedResponse = await caches.match(url);
    if (cachedResponse) {
      return cachedResponse;
    }
    throw error;
  }
}
```

## Web Workers

### Worker (Background Threads)

Run JavaScript in background threads:

```javascript
// main.js - Main thread
const worker = new Worker('worker.js');

// Send message to worker
worker.postMessage({ type: 'calculate', data: [1, 2, 3, 4, 5] });

// Receive message from worker
worker.onmessage = (event) => {
  console.log('Result from worker:', event.data);
};

worker.onerror = (error) => {
  console.error('Worker error:', error.message);
};

// Terminate worker
worker.terminate();

// ============================================
// worker.js - Worker thread
self.onmessage = (event) => {
  const { type, data } = event.data;

  if (type === 'calculate') {
    // Perform heavy computation
    const result = data.reduce((sum, num) => sum + num, 0);

    // Send result back to main thread
    self.postMessage(result);
  }
};

// Worker can't access:
// - DOM
// - window object
// - document object
// - parent object

// Worker can access:
// - navigator
// - location (read-only)
// - XMLHttpRequest / fetch
// - setTimeout / setInterval
// - IndexedDB
// - Cache API

// ============================================
// Advanced: Transferable objects (zero-copy)
const buffer = new ArrayBuffer(1024 * 1024); // 1 MB
worker.postMessage({ buffer }, [buffer]); // Transfer ownership
// buffer is now unusable in main thread

// ============================================
// Inline worker (no separate file)
const code = `
  self.onmessage = (e) => {
    self.postMessage(e.data * 2);
  };
`;

const blob = new Blob([code], { type: 'application/javascript' });
const workerUrl = URL.createObjectURL(blob);
const inlineWorker = new Worker(workerUrl);

inlineWorker.postMessage(5);
inlineWorker.onmessage = (e) => {
  console.log('Result:', e.data); // 10
};

// ============================================
// Shared Worker (shared across tabs)
const sharedWorker = new SharedWorker('shared-worker.js');

sharedWorker.port.postMessage('hello');
sharedWorker.port.onmessage = (event) => {
  console.log('From shared worker:', event.data);
};

// shared-worker.js
const connections = [];

self.onconnect = (event) => {
  const port = event.ports[0];
  connections.push(port);

  port.onmessage = (e) => {
    // Broadcast to all connections
    connections.forEach(conn => {
      conn.postMessage(`Broadcast: ${e.data}`);
    });
  };
};
```

### Service Worker

Powerful worker for offline capabilities and caching:

```javascript
// Register service worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/service-worker.js')
    .then(registration => {
      console.log('Service Worker registered:', registration.scope);

      // Check for updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        console.log('New service worker installing');

        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed') {
            if (navigator.serviceWorker.controller) {
              console.log('New version available, please refresh');
            } else {
              console.log('Content cached for offline use');
            }
          }
        });
      });
    })
    .catch(error => {
      console.error('Service Worker registration failed:', error);
    });

  // Listen for messages from service worker
  navigator.serviceWorker.addEventListener('message', (event) => {
    console.log('Message from SW:', event.data);
  });
}

// ============================================
// service-worker.js
const CACHE_NAME = 'my-app-v1';
const urlsToCache = [
  '/',
  '/styles.css',
  '/script.js',
  '/offline.html'
];

// Install event - cache resources
self.addEventListener('install', (event) => {
  console.log('Service Worker installing');

  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Caching resources');
        return cache.addAll(urlsToCache);
      })
      .then(() => self.skipWaiting()) // Activate immediately
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker activating');

  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim()) // Take control immediately
  );
});

// Fetch event - serve from cache
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Return cached version or fetch from network
        return response || fetch(event.request)
          .then(fetchResponse => {
            // Cache new resources
            return caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(event.request, fetchResponse.clone());
                return fetchResponse;
              });
          })
          .catch(() => {
            // Return offline page if fetch fails
            return caches.match('/offline.html');
          });
      })
  );
});

// Push notification event
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};

  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: '/icon.png',
      badge: '/badge.png',
      data: data.url
    })
  );
});

// Notification click event
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  event.waitUntil(
    clients.openWindow(event.notification.data)
  );
});

// Sync event (background sync)
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-messages') {
    event.waitUntil(syncMessages());
  }
});

async function syncMessages() {
  // Sync pending messages
  const messages = await getUnsyncedMessages();
  await Promise.all(
    messages.map(msg => fetch('/api/messages', {
      method: 'POST',
      body: JSON.stringify(msg)
    }))
  );
}

// Message from client
self.addEventListener('message', (event) => {
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
```

## Notification API

Display system notifications:

```javascript
// Request permission
async function requestNotificationPermission() {
  const permission = await Notification.requestPermission();

  if (permission === 'granted') {
    console.log('Notification permission granted');
  } else if (permission === 'denied') {
    console.log('Notification permission denied');
  }
}

// Check current permission
console.log('Permission:', Notification.permission);
// 'default', 'granted', or 'denied'

// Show notification (simple)
if (Notification.permission === 'granted') {
  new Notification('Hello!', {
    body: 'This is a notification',
    icon: '/icon.png',
    badge: '/badge.png'
  });
}

// Show notification (advanced)
const notification = new Notification('New Message', {
  body: 'You have 3 new messages',
  icon: '/icon.png',
  badge: '/badge.png',
  image: '/banner.png',
  tag: 'message-notification', // Replaces notifications with same tag
  renotify: true, // Notify even if same tag exists
  requireInteraction: false, // Auto-dismiss
  silent: false,
  vibrate: [200, 100, 200], // Vibration pattern
  timestamp: Date.now(),
  actions: [
    { action: 'view', title: 'View', icon: '/view.png' },
    { action: 'dismiss', title: 'Dismiss', icon: '/dismiss.png' }
  ],
  data: { url: '/messages' } // Custom data
});

// Event handlers
notification.onclick = (event) => {
  console.log('Notification clicked');
  window.focus();
  notification.close();
};

notification.onclose = () => {
  console.log('Notification closed');
};

notification.onerror = (error) => {
  console.error('Notification error:', error);
};

notification.onshow = () => {
  console.log('Notification shown');
};

// Close notification
setTimeout(() => {
  notification.close();
}, 5000);

// Service Worker notifications (recommended)
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.ready.then(registration => {
    registration.showNotification('Title', {
      body: 'Body text',
      icon: '/icon.png',
      actions: [
        { action: 'yes', title: 'Yes' },
        { action: 'no', title: 'No' }
      ]
    });
  });
}
```

## Geolocation API

Access device location:

```javascript
// Check if available
if ('geolocation' in navigator) {
  console.log('Geolocation is available');
}

// Get current position (one-time)
navigator.geolocation.getCurrentPosition(
  // Success callback
  (position) => {
    console.log('Latitude:', position.coords.latitude);
    console.log('Longitude:', position.coords.longitude);
    console.log('Accuracy:', position.coords.accuracy, 'meters');
    console.log('Altitude:', position.coords.altitude);
    console.log('Altitude accuracy:', position.coords.altitudeAccuracy);
    console.log('Heading:', position.coords.heading); // Direction of travel
    console.log('Speed:', position.coords.speed); // meters/second
    console.log('Timestamp:', position.timestamp);
  },
  // Error callback
  (error) => {
    switch (error.code) {
      case error.PERMISSION_DENIED:
        console.error('User denied geolocation');
        break;
      case error.POSITION_UNAVAILABLE:
        console.error('Position unavailable');
        break;
      case error.TIMEOUT:
        console.error('Request timeout');
        break;
    }
  },
  // Options
  {
    enableHighAccuracy: true, // Use GPS (more battery)
    timeout: 5000, // 5 seconds
    maximumAge: 0 // Don't use cached position
  }
);

// Watch position (continuous updates)
const watchId = navigator.geolocation.watchPosition(
  (position) => {
    console.log('Position updated:', position.coords);
    updateMapMarker(position.coords.latitude, position.coords.longitude);
  },
  (error) => {
    console.error('Watch error:', error);
  },
  {
    enableHighAccuracy: true,
    timeout: 10000,
    maximumAge: 5000 // Use cached position if < 5 seconds old
  }
);

// Stop watching
navigator.geolocation.clearWatch(watchId);

// Promised-based wrapper
function getPosition(options = {}) {
  return new Promise((resolve, reject) => {
    navigator.geolocation.getCurrentPosition(resolve, reject, options);
  });
}

// Usage
try {
  const position = await getPosition({ enableHighAccuracy: true });
  console.log('Position:', position.coords);
} catch (error) {
  console.error('Error getting position:', error);
}
```

## File API

Read and manipulate files:

```javascript
// File input
const input = document.getElementById('fileInput');

input.addEventListener('change', async (event) => {
  const files = event.target.files;

  for (const file of files) {
    console.log('Name:', file.name);
    console.log('Size:', file.size, 'bytes');
    console.log('Type:', file.type);
    console.log('Last modified:', new Date(file.lastModified));

    // Read as text
    const text = await file.text();
    console.log('Content:', text);

    // Read as ArrayBuffer
    const buffer = await file.arrayBuffer();
    console.log('Buffer:', new Uint8Array(buffer));

    // Read as Data URL (base64)
    const reader = new FileReader();
    reader.onload = (e) => {
      console.log('Data URL:', e.target.result);
      // Can use as img src
      document.getElementById('preview').src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Read as text with FileReader
    reader.onload = (e) => {
      console.log('Text:', e.target.result);
    };
    reader.readAsText(file);

    // Read as ArrayBuffer with FileReader
    reader.onload = (e) => {
      const buffer = e.target.result;
      console.log('Buffer:', new Uint8Array(buffer));
    };
    reader.readAsArrayBuffer(file);

    // Progress event
    reader.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = (e.loaded / e.total) * 100;
        console.log('Progress:', percent.toFixed(2) + '%');
      }
    };
  }
});

// Drag and drop
const dropZone = document.getElementById('dropZone');

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');

  const files = e.dataTransfer.files;
  for (const file of files) {
    console.log('Dropped file:', file.name);
  }
});

// Create Blob
const blob = new Blob(['Hello, World!'], { type: 'text/plain' });
console.log('Blob size:', blob.size);
console.log('Blob type:', blob.type);

// Read Blob
const text = await blob.text();
console.log('Blob text:', text);

// Blob to URL
const url = URL.createObjectURL(blob);
console.log('Blob URL:', url);
// Don't forget to revoke
URL.revokeObjectURL(url);

// Download file
function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url);
}

downloadFile('Hello, World!', 'hello.txt', 'text/plain');

// File from Blob
const file = new File([blob], 'example.txt', {
  type: 'text/plain',
  lastModified: Date.now()
});
```

## Clipboard API

Read and write clipboard:

```javascript
// Write text to clipboard
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    console.log('Text copied to clipboard');
  } catch (error) {
    console.error('Failed to copy:', error);
  }
}

copyText('Hello, clipboard!');

// Read text from clipboard
async function pasteText() {
  try {
    const text = await navigator.clipboard.readText();
    console.log('Pasted text:', text);
    return text;
  } catch (error) {
    console.error('Failed to read clipboard:', error);
  }
}

// Write images/rich content
async function copyImage(blob) {
  try {
    const item = new ClipboardItem({ 'image/png': blob });
    await navigator.clipboard.write([item]);
    console.log('Image copied');
  } catch (error) {
    console.error('Failed to copy image:', error);
  }
}

// Read images/rich content
async function pasteImage() {
  try {
    const items = await navigator.clipboard.read();

    for (const item of items) {
      for (const type of item.types) {
        const blob = await item.getType(type);

        if (type.startsWith('image/')) {
          const url = URL.createObjectURL(blob);
          const img = document.createElement('img');
          img.src = url;
          document.body.appendChild(img);
        }
      }
    }
  } catch (error) {
    console.error('Failed to paste:', error);
  }
}

// Copy button example
document.getElementById('copyBtn').addEventListener('click', async () => {
  const text = document.getElementById('text').textContent;
  await copyText(text);
  alert('Copied!');
});

// Legacy approach (fallback)
function copyTextLegacy(text) {
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand('copy');
  document.body.removeChild(textarea);
}
```

## Intersection Observer API

Detect element visibility:

```javascript
// Create observer
const observer = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        console.log('Element is visible:', entry.target);

        // Lazy load image
        if (entry.target.tagName === 'IMG') {
          entry.target.src = entry.target.dataset.src;
          observer.unobserve(entry.target); // Stop observing
        }

        // Animation on scroll
        entry.target.classList.add('animate-in');
      } else {
        console.log('Element is hidden:', entry.target);
      }
    });
  },
  {
    root: null, // viewport
    rootMargin: '0px', // margin around root
    threshold: 0.5 // 50% visible
  }
);

// Observe elements
const images = document.querySelectorAll('img[data-src]');
images.forEach(img => observer.observe(img));

// Multiple thresholds
const detailedObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      console.log('Visibility:', entry.intersectionRatio);
      // 0 = not visible, 1 = fully visible
    });
  },
  {
    threshold: [0, 0.25, 0.5, 0.75, 1.0]
  }
);

// Infinite scroll example
const loadMore = document.getElementById('loadMore');

const infiniteObserver = new IntersectionObserver(
  (entries) => {
    if (entries[0].isIntersecting) {
      console.log('Load more items');
      loadMoreItems().then(items => {
        appendItems(items);
      });
    }
  },
  { threshold: 1.0 }
);

infiniteObserver.observe(loadMore);

// Unobserve element
observer.unobserve(element);

// Disconnect observer
observer.disconnect();
```

## Mutation Observer API

Watch for DOM changes:

```javascript
// Create observer
const mutationObserver = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    console.log('Type:', mutation.type);

    if (mutation.type === 'childList') {
      console.log('Children changed');
      console.log('Added:', mutation.addedNodes);
      console.log('Removed:', mutation.removedNodes);
    }

    if (mutation.type === 'attributes') {
      console.log('Attribute changed:', mutation.attributeName);
      console.log('Old value:', mutation.oldValue);
    }

    if (mutation.type === 'characterData') {
      console.log('Text content changed');
      console.log('Old value:', mutation.oldValue);
    }
  });
});

// Observe element
const targetNode = document.getElementById('target');

mutationObserver.observe(targetNode, {
  childList: true, // Watch for child additions/removals
  attributes: true, // Watch for attribute changes
  characterData: true, // Watch for text content changes
  subtree: true, // Watch descendants too
  attributeOldValue: true, // Record old attribute value
  characterDataOldValue: true, // Record old text value
  attributeFilter: ['class', 'style'] // Only watch specific attributes
});

// Disconnect observer
mutationObserver.disconnect();

// Example: Watch for dynamically added elements
const bodyObserver = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    mutation.addedNodes.forEach(node => {
      if (node.classList && node.classList.contains('dynamic-content')) {
        console.log('Dynamic content added:', node);
        initializeDynamicContent(node);
      }
    });
  });
});

bodyObserver.observe(document.body, {
  childList: true,
  subtree: true
});
```

## Resize Observer API

Detect element size changes:

```javascript
// Create observer
const resizeObserver = new ResizeObserver((entries) => {
  entries.forEach(entry => {
    console.log('Element:', entry.target);
    console.log('Content box:', entry.contentBoxSize);
    console.log('Border box:', entry.borderBoxSize);
    console.log('Device pixel box:', entry.devicePixelContentBoxSize);

    const width = entry.contentRect.width;
    const height = entry.contentRect.height;
    console.log('Size:', width, 'x', height);

    // Responsive behavior
    if (width < 600) {
      entry.target.classList.add('mobile');
    } else {
      entry.target.classList.remove('mobile');
    }
  });
});

// Observe element
const element = document.getElementById('resizable');
resizeObserver.observe(element);

// Observe multiple elements
const elements = document.querySelectorAll('.resizable');
elements.forEach(el => resizeObserver.observe(el));

// Unobserve
resizeObserver.unobserve(element);

// Disconnect
resizeObserver.disconnect();

// Example: Canvas responsive rendering
const canvas = document.getElementById('canvas');

const canvasObserver = new ResizeObserver((entries) => {
  const entry = entries[0];
  const width = entry.contentRect.width;
  const height = entry.contentRect.height;

  // Update canvas size
  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;

  // Re-render
  renderCanvas();
});

canvasObserver.observe(canvas);
```

## Page Visibility API

Detect when page is visible:

```javascript
// Check current visibility
console.log('Hidden:', document.hidden);
console.log('Visibility state:', document.visibilityState);
// 'visible', 'hidden', 'prerender'

// Listen for visibility changes
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    console.log('Page is hidden');

    // Pause video
    video.pause();

    // Stop animations
    stopAnimations();

    // Reduce network activity
    clearInterval(pollingInterval);
  } else {
    console.log('Page is visible');

    // Resume video
    video.play();

    // Resume animations
    startAnimations();

    // Resume polling
    startPolling();
  }
});

// Example: Pause game when tab is hidden
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    game.pause();
  } else {
    game.resume();
  }
});

// Example: Analytics
let startTime = Date.now();

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    const visibleTime = Date.now() - startTime;
    analytics.track('time-visible', visibleTime);
  } else {
    startTime = Date.now();
  }
});
```

## Broadcast Channel API

Communicate between tabs/windows:

```javascript
// Create channel
const channel = new BroadcastChannel('my-channel');

// Send message
channel.postMessage('Hello from tab 1');
channel.postMessage({ type: 'update', data: { count: 5 } });

// Receive messages
channel.onmessage = (event) => {
  console.log('Received message:', event.data);

  if (event.data.type === 'update') {
    updateUI(event.data.data);
  }
};

channel.onerror = (error) => {
  console.error('Channel error:', error);
};

// Close channel
channel.close();

// Example: Sync state across tabs
const stateChannel = new BroadcastChannel('app-state');

// Tab 1: Update state
function updateState(newState) {
  state = newState;
  localStorage.setItem('state', JSON.stringify(state));
  stateChannel.postMessage({ type: 'state-update', state });
}

// All tabs: Listen for updates
stateChannel.onmessage = (event) => {
  if (event.data.type === 'state-update') {
    state = event.data.state;
    renderUI();
  }
};

// Example: Logout all tabs
const authChannel = new BroadcastChannel('auth');

// Tab with logout button
function logout() {
  clearAuthToken();
  authChannel.postMessage({ type: 'logout' });
  redirectToLogin();
}

// All tabs
authChannel.onmessage = (event) => {
  if (event.data.type === 'logout') {
    clearAuthToken();
    redirectToLogin();
  }
};
```

## History API

Manipulate browser history:

```javascript
// Push new state
history.pushState(
  { page: 1 }, // State object
  'Title',     // Title (ignored by most browsers)
  '/page/1'    // URL
);

// Replace current state
history.replaceState({ page: 2 }, 'Title', '/page/2');

// Go back
history.back();

// Go forward
history.forward();

// Go to specific point
history.go(-2); // Go back 2 pages
history.go(1);  // Go forward 1 page

// Listen for state changes
window.addEventListener('popstate', (event) => {
  console.log('State:', event.state);
  console.log('URL:', location.pathname);

  // Restore page state
  if (event.state && event.state.page) {
    loadPage(event.state.page);
  }
});

// Get current state
console.log('Current state:', history.state);

// Length of history
console.log('History length:', history.length);

// Example: Single Page App navigation
function navigateTo(url, state = {}) {
  history.pushState(state, '', url);
  loadContent(url);
}

document.querySelectorAll('a[data-link]').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    navigateTo(link.href);
  });
});

window.addEventListener('popstate', () => {
  loadContent(location.pathname);
});
```

## Performance API

Measure performance:

```javascript
// Mark time points
performance.mark('start-task');

// Do some work
await doSomethingExpensive();

performance.mark('end-task');

// Measure duration
performance.measure('task-duration', 'start-task', 'end-task');

// Get measurements
const measures = performance.getEntriesByName('task-duration');
console.log('Duration:', measures[0].duration, 'ms');

// Navigation timing
const navTiming = performance.getEntriesByType('navigation')[0];
console.log('DNS lookup:', navTiming.domainLookupEnd - navTiming.domainLookupStart);
console.log('TCP connect:', navTiming.connectEnd - navTiming.connectStart);
console.log('Request time:', navTiming.responseEnd - navTiming.requestStart);
console.log('DOM load:', navTiming.domContentLoadedEventEnd - navTiming.domContentLoadedEventStart);
console.log('Page load:', navTiming.loadEventEnd - navTiming.loadEventStart);

// Resource timing
const resources = performance.getEntriesByType('resource');
resources.forEach(resource => {
  console.log('Resource:', resource.name);
  console.log('Duration:', resource.duration);
  console.log('Size:', resource.transferSize);
});

// Paint timing
const paintTiming = performance.getEntriesByType('paint');
paintTiming.forEach(entry => {
  console.log(`${entry.name}:`, entry.startTime);
});
// first-paint, first-contentful-paint

// Clear marks and measures
performance.clearMarks();
performance.clearMeasures();

// Observer for performance entries
const perfObserver = new PerformanceObserver((list) => {
  list.getEntries().forEach(entry => {
    console.log('Entry:', entry.name, entry.duration);
  });
});

perfObserver.observe({ entryTypes: ['measure', 'navigation', 'resource'] });

// Memory usage (Chrome only)
if (performance.memory) {
  console.log('Used heap:', performance.memory.usedJSHeapSize);
  console.log('Total heap:', performance.memory.totalJSHeapSize);
  console.log('Heap limit:', performance.memory.jsHeapSizeLimit);
}

// Current time (high-resolution)
const start = performance.now();
// Do work
const end = performance.now();
console.log('Elapsed:', end - start, 'ms');
```

## Battery Status API

Get battery information:

```javascript
if ('getBattery' in navigator) {
  const battery = await navigator.getBattery();

  console.log('Charging:', battery.charging);
  console.log('Level:', battery.level * 100 + '%');
  console.log('Charging time:', battery.chargingTime, 'seconds');
  console.log('Discharging time:', battery.dischargingTime, 'seconds');

  // Listen for changes
  battery.addEventListener('chargingchange', () => {
    console.log('Charging:', battery.charging);
  });

  battery.addEventListener('levelchange', () => {
    console.log('Battery level:', battery.level * 100 + '%');

    if (battery.level < 0.2 && !battery.charging) {
      alert('Low battery! Please charge your device.');
    }
  });

  battery.addEventListener('chargingtimechange', () => {
    console.log('Charging time:', battery.chargingTime);
  });

  battery.addEventListener('dischargingtimechange', () => {
    console.log('Discharging time:', battery.dischargingTime);
  });

  // Adaptive features based on battery
  if (battery.level < 0.2 && !battery.charging) {
    // Reduce animations, polling, etc.
    enablePowerSavingMode();
  }
}
```

## Web Share API

Share content from web app:

```javascript
// Check if supported
if (navigator.share) {
  console.log('Web Share API supported');
}

// Share text
async function shareText() {
  try {
    await navigator.share({
      title: 'Check this out!',
      text: 'This is amazing content',
      url: 'https://example.com'
    });
    console.log('Shared successfully');
  } catch (error) {
    console.error('Error sharing:', error);
  }
}

// Share files
async function shareFiles(files) {
  if (navigator.canShare && navigator.canShare({ files })) {
    try {
      await navigator.share({
        files: files,
        title: 'Shared files',
        text: 'Check out these files'
      });
      console.log('Files shared successfully');
    } catch (error) {
      console.error('Error sharing files:', error);
    }
  } else {
    console.log('File sharing not supported');
  }
}

// Example: Share button
document.getElementById('shareBtn').addEventListener('click', async () => {
  if (navigator.share) {
    await shareText();
  } else {
    // Fallback: Copy link
    await navigator.clipboard.writeText(window.location.href);
    alert('Link copied to clipboard');
  }
});

// Example: Share image
const canvas = document.getElementById('canvas');
canvas.toBlob(async (blob) => {
  const file = new File([blob], 'image.png', { type: 'image/png' });
  await shareFiles([file]);
});
```

## Browser Support and Feature Detection

Always check for API availability:

```javascript
// Feature detection
const features = {
  serviceWorker: 'serviceWorker' in navigator,
  pushNotifications: 'PushManager' in window,
  notifications: 'Notification' in window,
  geolocation: 'geolocation' in navigator,
  webWorker: typeof Worker !== 'undefined',
  indexedDB: 'indexedDB' in window,
  webRTC: 'RTCPeerConnection' in window,
  webGL: (() => {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
  })(),
  mediaDevices: 'mediaDevices' in navigator,
  clipboard: 'clipboard' in navigator,
  share: 'share' in navigator,
  battery: 'getBattery' in navigator
};

console.table(features);

// Polyfill loading
if (!window.IntersectionObserver) {
  await import('intersection-observer');
}

// Progressive enhancement
if ('serviceWorker' in navigator) {
  // Enable offline support
  registerServiceWorker();
} else {
  // Gracefully degrade
  console.log('Service Worker not supported');
}
```

## Best Practices

```javascript
// 1. Always check feature support
if ('geolocation' in navigator) {
  // Use geolocation
}

// 2. Handle errors gracefully
try {
  await navigator.clipboard.writeText('text');
} catch (error) {
  // Fallback
  fallbackCopyMethod('text');
}

// 3. Request permissions appropriately
// Don't request permission immediately on page load
document.getElementById('enableNotifications').addEventListener('click', async () => {
  await Notification.requestPermission();
});

// 4. Clean up resources
const observer = new IntersectionObserver(callback);
// When done:
observer.disconnect();

const worker = new Worker('worker.js');
// When done:
worker.terminate();

// 5. Use Promises/async-await for better readability
// Instead of callbacks
async function loadData() {
  const data = await fetch('/api/data').then(r => r.json());
  return data;
}

// 6. Respect user privacy
// Check permission status before requesting
const status = await navigator.permissions.query({ name: 'geolocation' });
if (status.state === 'granted') {
  // Already have permission
}

// 7. Optimize performance
// Debounce expensive operations
function debounce(func, wait) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

window.addEventListener('resize', debounce(() => {
  console.log('Resized');
}, 250));
```

## Further Resources

### Documentation
- [MDN Web APIs](https://developer.mozilla.org/en-US/docs/Web/API)
- [Can I Use](https://caniuse.com/) - Browser support tables
- [Web.dev](https://web.dev/) - Modern web development guides

### Specifications
- [W3C Web APIs](https://www.w3.org/TR/)
- [WHATWG Standards](https://spec.whatwg.org/)

### Tools
- [Lighthouse](https://developers.google.com/web/tools/lighthouse) - Performance auditing
- [Workbox](https://developers.google.com/web/tools/workbox) - Service Worker library

### Libraries
- [Dexie.js](https://dexie.org/) - IndexedDB wrapper
- [localForage](https://localforage.github.io/localForage/) - Unified storage API
- [Comlink](https://github.com/GoogleChromeLabs/comlink) - Web Worker RPC
