# Frontend Performance

## Overview

Frontend performance is critical for user experience, SEO rankings, and conversion rates. Studies show that a 1-second delay in page load time can result in a 7% reduction in conversions. This guide covers essential strategies for optimizing web application performance.

## Core Web Vitals

Google's Core Web Vitals are key metrics for measuring user experience:

### Largest Contentful Paint (LCP)

Measures loading performance - when the largest content element becomes visible.

**Target**: < 2.5 seconds

```javascript
// Monitor LCP
new PerformanceObserver((entryList) => {
  for (const entry of entryList.getEntries()) {
    console.log('LCP:', entry.renderTime || entry.loadTime);
  }
}).observe({ entryTypes: ['largest-contentful-paint'] });
```

**Optimizations**:
- Optimize server response time
- Use CDN for static assets
- Preload critical resources
- Lazy load non-critical content

### First Input Delay (FID)

Measures interactivity - time from user interaction to browser response.

**Target**: < 100 milliseconds

**Optimizations**:
- Minimize JavaScript execution
- Break up long tasks
- Use web workers for heavy computation
- Defer non-critical JavaScript

### Cumulative Layout Shift (CLS)

Measures visual stability - unexpected layout shifts.

**Target**: < 0.1

```css
/* Reserve space for images */
img {
  aspect-ratio: 16 / 9;
  width: 100%;
  height: auto;
}

/* Prevent flash of unstyled content */
.skeleton {
  min-height: 200px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
}
```

**Optimizations**:
- Set dimensions for images and videos
- Reserve space for dynamic content
- Avoid inserting content above existing content
- Use `transform` instead of layout properties

## Loading Strategies

### Critical Rendering Path

```
HTML → DOM Tree
CSS  → CSSOM Tree
     ↓
Render Tree → Layout → Paint → Composite
     ↑
JavaScript
```

### Resource Loading

```html
<!-- Preload critical resources -->
<link rel="preload" href="critical.css" as="style">
<link rel="preload" href="hero.jpg" as="image">
<link rel="preload" href="app.js" as="script">

<!-- Prefetch for next page -->
<link rel="prefetch" href="next-page.js">

<!-- DNS prefetch for external domains -->
<link rel="dns-prefetch" href="https://api.example.com">

<!-- Preconnect for critical third-party origins -->
<link rel="preconnect" href="https://fonts.googleapis.com">

<!-- Async scripts (don't block parsing) -->
<script src="analytics.js" async></script>

<!-- Defer scripts (execute after parsing) -->
<script src="app.js" defer></script>
```

### Code Splitting

Split bundles to load only what's needed:

```javascript
// Dynamic imports
const handleClick = async () => {
  const module = await import('./heavyFeature.js');
  module.default();
};

// React lazy loading
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./Dashboard'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Dashboard />
    </Suspense>
  );
}

// Webpack code splitting
import(/* webpackChunkName: "lodash" */ 'lodash')
  .then(({ default: _ }) => {
    console.log(_.join(['Hello', 'webpack'], ' '));
  });
```

### Lazy Loading Images

```javascript
// Intersection Observer API
const imageObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src;
      img.classList.remove('lazy');
      observer.unobserve(img);
    }
  });
});

document.querySelectorAll('img.lazy').forEach(img => {
  imageObserver.observe(img);
});

// Native lazy loading
<img src="image.jpg" loading="lazy" alt="Description">
```

## Image Optimization

### Modern Formats

```html
<!-- Serve WebP/AVIF with fallbacks -->
<picture>
  <source srcset="image.avif" type="image/avif">
  <source srcset="image.webp" type="image/webp">
  <img src="image.jpg" alt="Description">
</picture>
```

### Responsive Images

```html
<!-- Different sizes for different viewports -->
<img
  srcset="
    small.jpg 400w,
    medium.jpg 800w,
    large.jpg 1200w
  "
  sizes="
    (max-width: 400px) 400px,
    (max-width: 800px) 800px,
    1200px
  "
  src="medium.jpg"
  alt="Description"
>
```

### Image Compression

| Format | Use Case | Quality |
|--------|----------|---------|
| **AVIF** | Modern browsers, best compression | Excellent |
| **WebP** | Wide support, good compression | Very good |
| **JPEG** | Photos, gradients | Good |
| **PNG** | Graphics with transparency | Lossless |
| **SVG** | Icons, logos, illustrations | Vector |

**Best Practices**:
- Compress images (TinyPNG, ImageOptim)
- Use appropriate dimensions
- Implement responsive images
- Serve via CDN
- Use `srcset` for retina displays

## JavaScript Optimization

### Bundle Size Reduction

```javascript
// Tree shaking - remove unused code
import { debounce } from 'lodash-es'; // Instead of entire lodash

// Analyze bundle
// npm install --save-dev webpack-bundle-analyzer

// Dynamic imports for routes
const routes = [
  {
    path: '/dashboard',
    component: () => import('./Dashboard.vue')
  }
];
```

### Performance Patterns

```javascript
// Debounce expensive operations
function debounce(fn, delay) {
  let timeoutId;
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

const handleSearch = debounce((query) => {
  fetch(`/api/search?q=${query}`);
}, 300);

// Throttle scroll/resize handlers
function throttle(fn, limit) {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

const handleScroll = throttle(() => {
  console.log('Scroll position:', window.scrollY);
}, 100);

// Memoization for expensive calculations
const memoize = (fn) => {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
};

const fibonacci = memoize((n) => {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
});
```

### Web Workers

Offload heavy computation to prevent UI blocking:

```javascript
// main.js
const worker = new Worker('worker.js');

worker.postMessage({ data: largeDataset });

worker.onmessage = (e) => {
  console.log('Result:', e.data);
};

// worker.js
self.onmessage = (e) => {
  const result = processData(e.data);
  self.postMessage(result);
};

function processData(data) {
  // Heavy computation here
  return data.map(item => complexCalculation(item));
}
```

### RequestIdleCallback

```javascript
// Run non-critical tasks when browser is idle
requestIdleCallback((deadline) => {
  while (deadline.timeRemaining() > 0 && tasks.length > 0) {
    const task = tasks.shift();
    task();
  }
}, { timeout: 2000 });
```

## CSS Optimization

### Critical CSS

Inline above-the-fold styles:

```html
<head>
  <!-- Inline critical CSS -->
  <style>
    /* Above-fold styles */
    .header { background: #fff; height: 60px; }
    .hero { min-height: 400px; }
  </style>

  <!-- Load full stylesheet async -->
  <link rel="preload" href="styles.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="styles.css"></noscript>
</head>
```

### CSS Performance

```css
/* Avoid expensive properties */
/* Bad - triggers layout */
.box {
  width: 100px;
  height: 100px;
}

/* Good - uses transform (composite only) */
.box {
  transform: scale(1);
  will-change: transform;
}

/* Optimize selectors - avoid deep nesting */
/* Bad */
.container .sidebar .menu ul li a { }

/* Good */
.menu-link { }

/* Use CSS containment */
.article {
  contain: layout style paint;
}

/* GPU acceleration */
.animated {
  transform: translateZ(0);
  will-change: transform;
}
```

## Caching Strategies

### HTTP Caching

```
Cache-Control Headers:
┌─────────────────────────────────┐
│ no-cache: Validate before use   │
│ no-store: Never cache           │
│ public: Cache in shared caches  │
│ private: Browser cache only     │
│ max-age: Cache duration (sec)   │
│ immutable: Never revalidate     │
└─────────────────────────────────┘
```

```javascript
// Service worker caching
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('v1').then((cache) => {
      return cache.addAll([
        '/',
        '/styles.css',
        '/app.js',
        '/logo.png'
      ]);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
```

### Browser Storage

```javascript
// LocalStorage (5-10MB, synchronous)
localStorage.setItem('theme', 'dark');
const theme = localStorage.getItem('theme');

// SessionStorage (per-tab, cleared on close)
sessionStorage.setItem('tempData', JSON.stringify(data));

// IndexedDB (large datasets, asynchronous)
const request = indexedDB.open('myDB', 1);

request.onsuccess = (event) => {
  const db = event.target.result;
  const transaction = db.transaction(['store'], 'readwrite');
  const store = transaction.objectStore('store');
  store.add({ id: 1, data: 'value' });
};

// Cache API (for offline-first)
caches.open('api-cache').then((cache) => {
  cache.put('/api/data', new Response(JSON.stringify(data)));
});
```

## Network Optimization

### HTTP/2 & HTTP/3

```
HTTP/1.1:
Request 1 → Response 1
Request 2 → Response 2
Request 3 → Response 3

HTTP/2 (Multiplexing):
Request 1 ↘
Request 2 → Single Connection → Response 1, 2, 3
Request 3 ↗

HTTP/3 (QUIC):
- Faster connection setup
- Better packet loss recovery
- Built-in encryption
```

### Resource Hints

```html
<!-- DNS Prefetch -->
<link rel="dns-prefetch" href="//cdn.example.com">

<!-- Preconnect (DNS + TCP + TLS) -->
<link rel="preconnect" href="https://fonts.googleapis.com">

<!-- Prefetch (low priority) -->
<link rel="prefetch" href="/next-page.js">

<!-- Preload (high priority) -->
<link rel="preload" href="/critical.css" as="style">

<!-- Prerender (full page) -->
<link rel="prerender" href="/next-page.html">
```

### API Optimization

```javascript
// Batch requests
// Bad
await fetch('/api/user/1');
await fetch('/api/user/2');
await fetch('/api/user/3');

// Good
await fetch('/api/users?ids=1,2,3');

// GraphQL - request only needed fields
const query = `
  query {
    user(id: 1) {
      name
      email
      # Only request what you need
    }
  }
`;

// Compression
fetch('/api/data', {
  headers: {
    'Accept-Encoding': 'gzip, deflate, br'
  }
});

// Pagination/Infinite scroll
const fetchPage = async (page, limit = 20) => {
  const response = await fetch(`/api/posts?page=${page}&limit=${limit}`);
  return response.json();
};
```

## Rendering Strategies

### Server-Side Rendering (SSR)

```
Request → Server renders HTML → Send to browser → Hydrate
```

**Pros**: Better SEO, faster FCP
**Cons**: Slower TTFB, server load

### Static Site Generation (SSG)

```
Build time → Generate HTML → Deploy static files
```

**Pros**: Fastest delivery, CDN cache
**Cons**: Rebuild for updates

### Client-Side Rendering (CSR)

```
Load JS → Render in browser → Fetch data → Update UI
```

**Pros**: Rich interactions, no server rendering
**Cons**: Slower FCP, poor SEO

### Incremental Static Regeneration (ISR)

```
Static pages + Background regeneration at intervals
```

```javascript
// Next.js example
export async function getStaticProps() {
  const data = await fetchData();

  return {
    props: { data },
    revalidate: 60 // Regenerate every 60 seconds
  };
}
```

## Performance Monitoring

### Performance API

```javascript
// Navigation timing
const perfData = performance.getEntriesByType('navigation')[0];
console.log('DNS:', perfData.domainLookupEnd - perfData.domainLookupStart);
console.log('TCP:', perfData.connectEnd - perfData.connectStart);
console.log('TTFB:', perfData.responseStart - perfData.requestStart);
console.log('Load:', perfData.loadEventEnd - perfData.loadEventStart);

// Resource timing
performance.getEntriesByType('resource').forEach(resource => {
  console.log(resource.name, resource.duration);
});

// Custom marks and measures
performance.mark('start-render');
// ... rendering logic
performance.mark('end-render');
performance.measure('render-time', 'start-render', 'end-render');

const measure = performance.getEntriesByName('render-time')[0];
console.log('Render time:', measure.duration);
```

### Real User Monitoring (RUM)

```javascript
// Send metrics to analytics
const sendMetrics = () => {
  const perfData = performance.getEntriesByType('navigation')[0];

  fetch('/analytics', {
    method: 'POST',
    body: JSON.stringify({
      ttfb: perfData.responseStart - perfData.requestStart,
      domLoad: perfData.domContentLoadedEventEnd - perfData.fetchStart,
      windowLoad: perfData.loadEventEnd - perfData.fetchStart,
      url: window.location.href
    }),
    keepalive: true // Ensures request completes even if page unloads
  });
};

window.addEventListener('load', sendMetrics);
```

## Framework-Specific Optimizations

### React

```javascript
// Memoization
import { memo, useMemo, useCallback } from 'react';

// Prevent re-renders
const ExpensiveComponent = memo(({ data }) => {
  return <div>{data}</div>;
});

// Memoize calculated values
const sortedData = useMemo(() => {
  return data.sort((a, b) => a.value - b.value);
}, [data]);

// Memoize callbacks
const handleClick = useCallback(() => {
  console.log('Clicked');
}, []);

// Virtualization for long lists
import { FixedSizeList } from 'react-window';

<FixedSizeList
  height={400}
  itemCount={1000}
  itemSize={35}
  width="100%"
>
  {({ index, style }) => (
    <div style={style}>Item {index}</div>
  )}
</FixedSizeList>
```

### Vue

```javascript
// Keep-alive for component caching
<keep-alive>
  <component :is="currentView"></component>
</keep-alive>

// Lazy load components
const Dashboard = () => import('./Dashboard.vue');

// Computed properties (cached)
computed: {
  filteredList() {
    return this.items.filter(item => item.active);
  }
}

// v-once for static content
<div v-once>{{ staticContent }}</div>
```

## Performance Budget

Set limits to maintain performance:

```javascript
// webpack.config.js
module.exports = {
  performance: {
    maxAssetSize: 244000, // 244 KB
    maxEntrypointSize: 244000,
    hints: 'error'
  }
};
```

| Metric | Budget |
|--------|--------|
| **Total page size** | < 1.5 MB |
| **JavaScript** | < 300 KB |
| **CSS** | < 100 KB |
| **Images** | < 500 KB |
| **Time to Interactive** | < 3.8s |
| **First Contentful Paint** | < 1.8s |

## Tools & Testing

### Performance Testing Tools

- **Lighthouse**: Automated auditing (Chrome DevTools)
- **WebPageTest**: Real device testing
- **PageSpeed Insights**: Google's performance analysis
- **Chrome DevTools**: Performance profiling
- **Bundle Analyzer**: Visualize bundle composition

```bash
# Lighthouse CI
npm install -g @lhci/cli
lhci autorun --collect.url=https://example.com

# Webpack Bundle Analyzer
npm install --save-dev webpack-bundle-analyzer
```

### Lighthouse Score Factors

```
Performance Score (0-100)
├─ FCP (10%)
├─ SI (10%)
├─ LCP (25%)
├─ TTI (10%)
├─ TBT (30%)
└─ CLS (15%)
```

## Best Practices Checklist

**Loading**:
- [ ] Minify and compress assets (gzip/brotli)
- [ ] Enable HTTP/2 or HTTP/3
- [ ] Use CDN for static assets
- [ ] Implement resource hints (preload, prefetch)
- [ ] Lazy load images and non-critical resources

**JavaScript**:
- [ ] Code splitting and tree shaking
- [ ] Remove unused dependencies
- [ ] Defer non-critical JavaScript
- [ ] Use web workers for heavy tasks
- [ ] Implement service workers for offline support

**CSS**:
- [ ] Extract and inline critical CSS
- [ ] Remove unused CSS
- [ ] Use CSS containment
- [ ] Optimize animations (transform/opacity)

**Images**:
- [ ] Use modern formats (WebP/AVIF)
- [ ] Implement responsive images
- [ ] Compress images appropriately
- [ ] Use lazy loading
- [ ] Set explicit dimensions

**Monitoring**:
- [ ] Set performance budgets
- [ ] Monitor Core Web Vitals
- [ ] Implement RUM
- [ ] Regular Lighthouse audits

## ELI10

Think of your website like a pizza delivery:

**Fast Pizza = Happy Customer**
- **LCP** (Loading): How fast the pizza arrives
- **FID** (Interactivity): How quickly you can take a bite
- **CLS** (Stability): Pizza doesn't slide around in the box

**Optimization = Faster Delivery**
- **Code splitting**: Don't send the whole menu, just what's ordered
- **Lazy loading**: Deliver toppings as needed, not all at once
- **Caching**: Keep popular items ready (no wait time!)
- **CDN**: Multiple pizza shops closer to customers
- **Compression**: Pack the box efficiently

**Result**: Faster website = More users = Better business!

## Further Resources

- [Web.dev Performance](https://web.dev/performance/)
- [Chrome DevTools Performance](https://developer.chrome.com/docs/devtools/performance/)
- [Lighthouse Documentation](https://developer.chrome.com/docs/lighthouse/)
- [WebPageTest](https://www.webpagetest.org/)
- [Performance Budget Calculator](https://www.performancebudget.io/)
- [Can I Use](https://caniuse.com/) - Browser feature support
- [HTTP Archive](https://httparchive.org/) - Web performance trends
