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

## Additional Performance Metrics

### First Contentful Paint (FCP)

Time when the first text or image is painted.

**Target**: < 1.8 seconds

```javascript
// Monitor FCP
new PerformanceObserver((entryList) => {
  for (const entry of entryList.getEntries()) {
    if (entry.name === 'first-contentful-paint') {
      console.log('FCP:', entry.startTime);
    }
  }
}).observe({ entryTypes: ['paint'] });
```

**Optimizations**:
- Eliminate render-blocking resources
- Minify CSS
- Remove unused CSS
- Preconnect to required origins
- Reduce server response times

### Time to Interactive (TTI)

Time until page is fully interactive (can respond to user input).

**Target**: < 3.8 seconds

```javascript
// Approximate TTI detection
let ttiTime;
const observer = new PerformanceObserver((list) => {
  const entries = list.getEntries();
  // Look for a 5-second window with no long tasks
  entries.forEach(entry => {
    if (entry.duration < 50) {
      ttiTime = entry.startTime + entry.duration;
    }
  });
});
observer.observe({ entryTypes: ['longtask'] });
```

**Optimizations**:
- Minimize main thread work
- Reduce JavaScript execution time
- Break up long tasks (> 50ms)
- Defer non-critical third-party scripts
- Use code splitting and lazy loading

### Time to First Byte (TTFB)

Time from request to first byte of response.

**Target**: < 600 milliseconds

```javascript
// Measure TTFB
const perfData = performance.getEntriesByType('navigation')[0];
const ttfb = perfData.responseStart - perfData.requestStart;
console.log('TTFB:', ttfb);
```

**Optimizations**:
- Use a CDN
- Optimize server processing time
- Enable database query caching
- Implement server-side caching (Redis, Memcached)
- Use HTTP/2 or HTTP/3
- Reduce redirects

### Total Blocking Time (TBT)

Sum of all time periods between FCP and TTI where task length exceeded 50ms.

**Target**: < 200 milliseconds

```javascript
// Monitor long tasks
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    // Tasks longer than 50ms block the main thread
    console.log('Long task:', entry.duration);
  }
});
observer.observe({ entryTypes: ['longtask'] });
```

**Optimizations**:
- Break up long tasks
- Optimize third-party scripts
- Use web workers
- Implement code splitting
- Defer non-critical JavaScript

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

### Image CDNs

Automatically optimize and deliver images:

```html
<!-- Cloudinary -->
<img src="https://res.cloudinary.com/demo/image/upload/w_400,f_auto,q_auto/sample.jpg">

<!-- Parameters:
     w_400: width 400px
     f_auto: automatic format (WebP/AVIF)
     q_auto: automatic quality optimization
-->

<!-- imgix -->
<img src="https://demo.imgix.net/sample.jpg?w=400&auto=format,compress">
```

**Features**:
- Automatic format selection (WebP/AVIF)
- On-the-fly resizing
- Smart compression
- Global CDN delivery
- Lazy loading support

```javascript
// Responsive images with Cloudinary
const cloudinaryUrl = (publicId, width) => {
  return `https://res.cloudinary.com/demo/image/upload/w_${width},f_auto,q_auto,dpr_auto/${publicId}`;
};

// Usage
<img
  srcset="
    ${cloudinaryUrl('sample', 400)} 400w,
    ${cloudinaryUrl('sample', 800)} 800w,
    ${cloudinaryUrl('sample', 1200)} 1200w
  "
  sizes="(max-width: 400px) 400px, (max-width: 800px) 800px, 1200px"
  src="${cloudinaryUrl('sample', 800)}"
  alt="Sample"
>
```

**Popular Image CDNs**:
- Cloudinary
- imgix
- Cloudflare Images
- ImageKit
- AWS CloudFront with Lambda@Edge

## Bundle Optimization

### Minification

Remove unnecessary characters without changing functionality:

```javascript
// webpack.config.js
const TerserPlugin = require('terser-webpack-plugin');

module.exports = {
  optimization: {
    minimize: true,
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: true, // Remove console.logs
            dead_code: true,    // Remove unreachable code
          },
          mangle: true,         // Shorten variable names
        },
      }),
    ],
  },
};
```

### Compression (Gzip & Brotli)

Compress assets before sending to browser:

```
Uncompressed: 1000 KB
Gzip:         300 KB (70% reduction)
Brotli:       250 KB (75% reduction)
```

```nginx
# Nginx configuration
http {
  # Gzip compression
  gzip on;
  gzip_vary on;
  gzip_min_length 1024;
  gzip_types text/plain text/css text/xml text/javascript
             application/javascript application/json application/xml+rss;
  gzip_comp_level 6;

  # Brotli compression (better than gzip)
  brotli on;
  brotli_comp_level 6;
  brotli_types text/plain text/css text/xml text/javascript
               application/javascript application/json application/xml+rss;
}
```

```javascript
// Express.js
const compression = require('compression');
const express = require('express');
const app = express();

// Enable gzip compression
app.use(compression({
  level: 6,
  threshold: 1024, // Only compress responses > 1KB
  filter: (req, res) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return compression.filter(req, res);
  }
}));
```

### Tree Shaking

Remove unused code during bundling:

```javascript
// Bad - imports entire library
import _ from 'lodash';
const result = _.debounce(fn, 300);

// Good - only imports what's needed
import debounce from 'lodash-es/debounce';
const result = debounce(fn, 300);

// package.json - mark as side-effect free
{
  "name": "my-app",
  "sideEffects": false, // Enable tree shaking
  // or specify files with side effects
  "sideEffects": ["*.css", "*.scss"]
}
```

### Analyzing Bundle Size

```bash
# Webpack Bundle Analyzer
npm install --save-dev webpack-bundle-analyzer

# Add to webpack.config.js
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;

module.exports = {
  plugins: [
    new BundleAnalyzerPlugin({
      analyzerMode: 'static',
      openAnalyzer: true,
      reportFilename: 'bundle-report.html'
    })
  ]
};

# Run analysis
npm run build
# Opens interactive treemap visualization
```

```bash
# Source Map Explorer
npm install -g source-map-explorer

# Analyze bundle
source-map-explorer bundle.min.js bundle.min.js.map

# Vite
npm run build -- --report

# Next.js
npm install @next/bundle-analyzer
```

```javascript
// Monitor bundle size in CI/CD
// package.json
{
  "scripts": {
    "analyze": "webpack-bundle-analyzer dist/stats.json",
    "size": "size-limit"
  },
  "size-limit": [
    {
      "path": "dist/bundle.js",
      "limit": "300 KB"
    }
  ]
}
```

## JavaScript Optimization

### Bundle Size Reduction

```javascript
// Tree shaking - remove unused code
import { debounce } from 'lodash-es'; // Instead of entire lodash

// Dynamic imports for routes
const routes = [
  {
    path: '/dashboard',
    component: () => import('./Dashboard.vue')
  }
];
```

### Reducing JavaScript Execution Time

Break up long-running tasks to keep UI responsive:

```javascript
// Bad - blocks main thread
function processLargeArray(items) {
  items.forEach(item => {
    heavyProcessing(item); // Takes 200ms total
  });
}

// Good - break into chunks
async function processLargeArray(items) {
  const chunkSize = 50;

  for (let i = 0; i < items.length; i += chunkSize) {
    const chunk = items.slice(i, i + chunkSize);

    // Process chunk
    chunk.forEach(item => heavyProcessing(item));

    // Yield to browser for UI updates
    await new Promise(resolve => setTimeout(resolve, 0));
  }
}

// Using requestIdleCallback
function processWhenIdle(items) {
  function processChunk(deadline) {
    while (deadline.timeRemaining() > 0 && items.length > 0) {
      const item = items.shift();
      heavyProcessing(item);
    }

    if (items.length > 0) {
      requestIdleCallback(processChunk);
    }
  }

  requestIdleCallback(processChunk);
}
```

### Long Tasks Detection & Prevention

Tasks > 50ms block user input:

```javascript
// Detect long tasks
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.warn('Long task detected:', {
      duration: entry.duration,
      startTime: entry.startTime,
      name: entry.name
    });

    // Send to analytics
    analytics.track('long-task', {
      duration: entry.duration,
      url: window.location.href
    });
  }
});

observer.observe({ entryTypes: ['longtask'] });

// Break up long tasks with Task Scheduler API
if ('scheduler' in window) {
  await scheduler.yield(); // Give browser chance to render
}

// Polyfill for older browsers
const yieldToMain = () => {
  return new Promise(resolve => {
    setTimeout(resolve, 0);
  });
};

async function processData(data) {
  for (const item of data) {
    processItem(item);

    // Yield every 50ms
    if (performance.now() % 50 < 1) {
      await yieldToMain();
    }
  }
}
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

### Unused CSS Removal

Remove CSS that isn't used on your pages:

```bash
# PurgeCSS
npm install --save-dev @fullhuman/postcss-purgecss

# Configure in postcss.config.js
module.exports = {
  plugins: [
    require('@fullhuman/postcss-purgecss')({
      content: [
        './src/**/*.html',
        './src/**/*.js',
        './src/**/*.jsx',
        './src/**/*.vue',
      ],
      safelist: ['active', 'disabled'], // Don't remove these
      defaultExtractor: content => content.match(/[\w-/:]+(?<!:)/g) || []
    })
  ]
};
```

```bash
# UnCSS
npm install -g uncss

uncss https://example.com > cleaned.css

# Reduces CSS by 90%+
Before: 150 KB
After:   15 KB
```

```javascript
// Tailwind CSS built-in purging
// tailwind.config.js
module.exports = {
  content: [
    './src/**/*.{html,js,jsx,ts,tsx}',
  ],
  // Automatically removes unused utility classes
};
```

**Chrome DevTools Coverage**:
1. Open DevTools (F12)
2. Cmd+Shift+P → "Show Coverage"
3. Reload page
4. See unused CSS/JS percentages

### CSS-in-JS Performance

Runtime vs compile-time CSS-in-JS:

```javascript
// ❌ Runtime CSS-in-JS (slower)
// styled-components, Emotion (without compilation)
import styled from 'styled-components';

const Button = styled.button`
  background: ${props => props.primary ? 'blue' : 'gray'};
  padding: 10px 20px;
`;
// Generates styles at runtime (impacts performance)

// ✅ Zero-runtime CSS-in-JS (faster)
// Linaria, vanilla-extract, Compiled
import { styled } from '@linaria/react';

const Button = styled.button`
  background: blue;
  padding: 10px 20px;
`;
// Styles extracted at build time (no runtime cost)
```

**Performance comparison**:

| Library | Runtime | Initial Paint | Re-render |
|---------|---------|---------------|-----------|
| **Plain CSS** | None | Fast | Fast |
| **CSS Modules** | None | Fast | Fast |
| **Styled-components** | Yes | Slower | Slower |
| **Emotion** | Yes | Slower | Slower |
| **Linaria** | No | Fast | Fast |
| **vanilla-extract** | No | Fast | Fast |

**Best practices**:
```javascript
// ✅ Use static styles when possible
const Button = styled.button`
  padding: 10px 20px; /* Static */
`;

// ✅ Memoize dynamic styles
const DynamicButton = memo(styled.button`
  background: ${({ color }) => color};
`);

// ❌ Avoid creating styled components in render
function Component() {
  // Bad - new component every render
  const Button = styled.button`...`;
  return <Button />;
}

// ✅ Define outside component
const Button = styled.button`...`;
function Component() {
  return <Button />;
}
```

## Font Optimization

Web fonts can significantly impact performance if not optimized:

### Font Loading Strategies

```css
/* font-display property */
@font-face {
  font-family: 'MyFont';
  src: url('/fonts/myfont.woff2') format('woff2');
  font-display: swap; /* Show fallback immediately */
}

/* Options:
   auto: Browser default
   block: Hide text up to 3s (FOIT - Flash of Invisible Text)
   swap: Show fallback immediately (FOUT - Flash of Unstyled Text)
   fallback: 100ms block, then swap
   optional: 100ms block, may not download font
*/
```

### Preload Critical Fonts

```html
<!-- Preload fonts used above-fold -->
<link
  rel="preload"
  href="/fonts/myfont.woff2"
  as="font"
  type="font/woff2"
  crossorigin="anonymous"
>
```

### Self-Host Fonts

```html
<!-- ❌ External font (extra DNS lookup, connection) -->
<link href="https://fonts.googleapis.com/css2?family=Roboto" rel="stylesheet">

<!-- ✅ Self-hosted fonts (faster) -->
<link rel="stylesheet" href="/fonts/fonts.css">
```

```css
/* fonts.css */
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url('/fonts/roboto-v30-latin-regular.woff2') format('woff2');
  /* Only load Latin subset */
  unicode-range: U+0000-00FF, U+0131, U+0152-0153;
}
```

### Font Subsetting

Reduce font file size by including only needed characters:

```bash
# pyftsubset (fonttools)
pip install fonttools brotli

# Create subset with only needed characters
pyftsubset font.ttf \
  --output-file=font-subset.woff2 \
  --flavor=woff2 \
  --layout-features=* \
  --unicodes=U+0020-007F

# Result:
Original: 150 KB
Subset:    30 KB (80% reduction)
```

### Variable Fonts

Use variable fonts for multiple weights/styles in one file:

```css
/* Traditional: 3 separate files */
@font-face {
  font-family: 'Roboto';
  font-weight: 400;
  src: url('roboto-regular.woff2'); /* 50 KB */
}
@font-face {
  font-family: 'Roboto';
  font-weight: 700;
  src: url('roboto-bold.woff2'); /* 50 KB */
}
@font-face {
  font-family: 'Roboto';
  font-weight: 900;
  src: url('roboto-black.woff2'); /* 50 KB */
}
/* Total: 150 KB */

/* Variable font: 1 file with all weights */
@font-face {
  font-family: 'Roboto';
  font-weight: 100 900; /* Full weight range */
  src: url('roboto-variable.woff2'); /* 80 KB */
}
/* Total: 80 KB (47% reduction) */
```

### System Font Stack

Fastest option - use system fonts (no download):

```css
body {
  font-family:
    -apple-system,        /* macOS, iOS */
    BlinkMacSystemFont,   /* macOS Chrome */
    "Segoe UI",           /* Windows */
    Roboto,               /* Android */
    "Helvetica Neue",     /* macOS legacy */
    Arial,                /* Fallback */
    sans-serif;           /* Generic */
}
```

### Google Fonts Optimization

```html
<!-- ❌ Bad - blocks rendering -->
<link href="https://fonts.googleapis.com/css2?family=Roboto" rel="stylesheet">

<!-- ✅ Better - preconnect -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">

<!-- ✅ Best - async load -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link
  rel="stylesheet"
  href="https://fonts.googleapis.com/css2?family=Roboto&display=swap"
  media="print"
  onload="this.media='all'"
>
```

**Font loading checklist**:
- [ ] Use WOFF2 format (best compression)
- [ ] Subset fonts (remove unused characters)
- [ ] Use `font-display: swap`
- [ ] Preload critical fonts
- [ ] Self-host when possible
- [ ] Consider variable fonts
- [ ] Limit font families and weights

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

```nginx
# Nginx cache configuration
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
  expires 1y;
  add_header Cache-Control "public, immutable";
}

location ~* \.(html)$ {
  expires 0;
  add_header Cache-Control "no-cache, must-revalidate";
}
```

```javascript
// Express.js caching
app.use('/static', express.static('public', {
  maxAge: '1y',
  immutable: true
}));

// Set cache headers manually
app.get('/api/data', (req, res) => {
  res.set('Cache-Control', 'public, max-age=300'); // 5 minutes
  res.json(data);
});
```

### CDN Caching

Distribute static assets globally for faster delivery:

```
User Request → CDN Edge Server (nearest location)
               ↓
         Cache Hit? → Return cached content
               ↓ No
         Origin Server → Cache & return content
```

**CDN Benefits**:
- Lower latency (geographic proximity)
- Reduced origin server load
- DDoS protection
- Automatic compression
- SSL/TLS termination

```javascript
// Cloudflare cache configuration
// cloudflare-worker.js
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  const cache = caches.default;
  let response = await cache.match(request);

  if (!response) {
    response = await fetch(request);
    // Cache for 1 hour
    const headers = new Headers(response.headers);
    headers.set('Cache-Control', 's-maxage=3600');

    response = new Response(response.body, {
      status: response.status,
      headers: headers
    });

    event.waitUntil(cache.put(request, response.clone()));
  }

  return response;
}
```

**Popular CDNs**:
- Cloudflare
- AWS CloudFront
- Fastly
- Akamai
- Vercel Edge Network
- Netlify Edge

**Cache invalidation**:
```bash
# Versioned URLs (best practice)
<link rel="stylesheet" href="/styles.abc123.css">

# Query string versioning
<link rel="stylesheet" href="/styles.css?v=1.2.3">

# Cloudflare purge
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache" \
  -H "Authorization: Bearer {api_token}" \
  -d '{"files":["https://example.com/styles.css"]}'
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
