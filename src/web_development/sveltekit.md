# SvelteKit

SvelteKit is the official application framework for Svelte that provides server-side rendering, routing, code splitting, and more. It's a full-stack framework that enables you to build web applications of any size with excellent performance and developer experience.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Project Structure](#project-structure)
- [Routing](#routing)
- [Loading Data](#loading-data)
- [Form Actions](#form-actions)
- [Hooks](#hooks)
- [Page Options](#page-options)
- [API Routes](#api-routes)
- [State Management](#state-management)
- [Navigation](#navigation)
- [Error Handling](#error-handling)
- [Advanced Patterns](#advanced-patterns)
- [Authentication](#authentication)
- [Database Integration](#database-integration)
- [Building and Deployment](#building-and-deployment)
- [Performance Optimization](#performance-optimization)
- [Testing](#testing)
- [Best Practices](#best-practices)

---

## Introduction

**Key Features:**
- Server-side rendering (SSR) by default
- Static site generation (SSG)
- API routes
- File-based routing
- Code splitting and lazy loading
- Hot module replacement (HMR)
- TypeScript support
- Adaptable to any platform
- Progressive enhancement
- Zero-config deployment

**Comparison with Other Frameworks:**

| Feature | SvelteKit | Next.js | Nuxt.js |
|---------|-----------|---------|---------|
| Base Framework | Svelte | React | Vue |
| Build Tool | Vite | Webpack/Turbopack | Vite/Webpack |
| Rendering | SSR/SSG/CSR | SSR/SSG/ISR | SSR/SSG/CSR |
| Bundle Size | Smallest | Medium | Medium |
| Learning Curve | Gentle | Moderate | Moderate |
| Performance | Excellent | Very Good | Very Good |

**Use Cases:**
- Full-stack web applications
- E-commerce platforms
- Content management systems
- Dashboards and admin panels
- Documentation sites
- Progressive web apps (PWAs)
- API-driven applications

---

## Installation and Setup

### Create New Project

```bash
# Create new SvelteKit project
npm create svelte@latest my-app
cd my-app

# Install dependencies
npm install

# Start development server
npm run dev

# Or use other package managers
pnpm create svelte@latest my-app
yarn create svelte my-app
```

### Project Setup Options

During `npm create svelte`, you'll be asked:

1. **Which template?**
   - Skeleton project (minimal)
   - SvelteKit demo app (examples)
   - Library project

2. **Type checking?**
   - TypeScript
   - JavaScript with JSDoc
   - None

3. **Additional options:**
   - ESLint
   - Prettier
   - Playwright (E2E testing)
   - Vitest (unit testing)

### Basic Configuration

**svelte.config.js:**
```javascript
import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/kit/vite';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Preprocessor for Svelte files
  preprocess: vitePreprocess(),

  kit: {
    // Adapter for deployment
    adapter: adapter(),

    // Alias configuration
    alias: {
      $components: 'src/lib/components',
      $utils: 'src/lib/utils',
      $stores: 'src/lib/stores'
    },

    // CSP headers
    csp: {
      directives: {
        'script-src': ['self']
      }
    },

    // Environment variables prefix
    env: {
      publicPrefix: 'PUBLIC_'
    }
  }
};

export default config;
```

**vite.config.js:**
```javascript
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],

  server: {
    port: 3000,
    strictPort: false
  },

  preview: {
    port: 4173
  },

  optimizeDeps: {
    include: ['lodash']
  }
});
```

---

## Project Structure

```
my-sveltekit-app/
├── src/
│   ├── lib/
│   │   ├── components/        # Reusable components
│   │   ├── server/           # Server-only code
│   │   ├── stores/           # Svelte stores
│   │   └── utils/            # Utility functions
│   ├── routes/               # File-based routing
│   │   ├── +layout.svelte    # Root layout
│   │   ├── +layout.js        # Root layout data
│   │   ├── +page.svelte      # Home page
│   │   ├── +page.js          # Home page data
│   │   ├── about/
│   │   │   └── +page.svelte
│   │   ├── blog/
│   │   │   ├── +page.svelte
│   │   │   ├── +page.server.js
│   │   │   └── [slug]/
│   │   │       └── +page.svelte
│   │   └── api/
│   │       └── posts/
│   │           └── +server.js
│   ├── app.html              # HTML template
│   ├── app.css              # Global styles
│   ├── hooks.client.js      # Client hooks
│   └── hooks.server.js      # Server hooks
├── static/                   # Static assets
│   ├── favicon.png
│   └── robots.txt
├── tests/                    # Test files
├── .env                      # Environment variables
├── svelte.config.js         # SvelteKit config
├── vite.config.js           # Vite config
├── package.json
└── tsconfig.json            # TypeScript config
```

### Important Files

**src/app.html:**
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%sveltekit.assets%/favicon.png" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    %sveltekit.head%
  </head>
  <body data-sveltekit-preload-data="hover">
    <div style="display: contents">%sveltekit.body%</div>
  </body>
</html>
```

---

## Routing

### File-Based Routing

SvelteKit uses filesystem-based routing where the structure of your `src/routes` directory defines your app's routes.

```
routes/
├── +page.svelte              # / (home)
├── about/
│   └── +page.svelte          # /about
├── blog/
│   ├── +page.svelte          # /blog
│   ├── +layout.svelte        # Blog layout
│   └── [slug]/
│       └── +page.svelte      # /blog/my-post
├── products/
│   ├── +page.svelte          # /products
│   ├── [id]/
│   │   └── +page.svelte      # /products/123
│   └── [...path]/
│       └── +page.svelte      # /products/a/b/c
└── admin/
    └── (dashboard)/          # Route group (no URL segment)
        ├── users/
        │   └── +page.svelte  # /admin/users
        └── settings/
            └── +page.svelte  # /admin/settings
```

### Route Files

| File | Purpose |
|------|---------|
| `+page.svelte` | Page component |
| `+page.js` | Universal load function |
| `+page.server.js` | Server-only load function |
| `+layout.svelte` | Layout component |
| `+layout.js` | Layout universal load |
| `+layout.server.js` | Layout server load |
| `+server.js` | API endpoint |
| `+error.svelte` | Error page |

### Basic Page

**src/routes/+page.svelte:**
```svelte
<script>
  export let data;
</script>

<h1>Welcome to SvelteKit</h1>
<p>Data from server: {data.message}</p>

<style>
  h1 {
    color: #ff3e00;
  }
</style>
```

**src/routes/+page.js:**
```javascript
export async function load({ fetch }) {
  return {
    message: 'Hello from load function'
  };
}
```

### Layouts

Layouts wrap pages and can be nested.

**src/routes/+layout.svelte:**
```svelte
<script>
  import '../app.css';
  import Header from '$lib/components/Header.svelte';
  import Footer from '$lib/components/Footer.svelte';

  export let data;
</script>

<div class="app">
  <Header user={data.user} />

  <main>
    <slot />
  </main>

  <Footer />
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }

  main {
    flex: 1;
  }
</style>
```

**src/routes/+layout.server.js:**
```javascript
export async function load({ locals }) {
  return {
    user: locals.user || null
  };
}
```

### Nested Layouts

**src/routes/blog/+layout.svelte:**
```svelte
<script>
  export let data;
</script>

<div class="blog-layout">
  <aside>
    <h2>Categories</h2>
    <ul>
      {#each data.categories as category}
        <li><a href="/blog/category/{category.slug}">{category.name}</a></li>
      {/each}
    </ul>
  </aside>

  <div class="content">
    <slot />
  </div>
</div>

<style>
  .blog-layout {
    display: grid;
    grid-template-columns: 250px 1fr;
    gap: 2rem;
  }
</style>
```

### Dynamic Routes

**src/routes/blog/[slug]/+page.svelte:**
```svelte
<script>
  export let data;
</script>

<article>
  <h1>{data.post.title}</h1>
  <div class="meta">
    <time>{data.post.date}</time>
    <span>By {data.post.author}</span>
  </div>
  <div class="content">
    {@html data.post.content}
  </div>
</article>
```

**src/routes/blog/[slug]/+page.server.js:**
```javascript
import { error } from '@sveltejs/kit';
import { getPostBySlug } from '$lib/server/database';

export async function load({ params }) {
  const post = await getPostBySlug(params.slug);

  if (!post) {
    throw error(404, {
      message: 'Post not found'
    });
  }

  return {
    post
  };
}
```

### Optional Parameters

**src/routes/archive/[[page]]/+page.svelte:**
```svelte
<!-- Matches /archive and /archive/2 -->
<script>
  export let data;
</script>

<h1>Archive - Page {data.page}</h1>

{#each data.posts as post}
  <article>
    <h2>{post.title}</h2>
  </article>
{/each}
```

**src/routes/archive/[[page]]/+page.js:**
```javascript
export async function load({ params }) {
  const page = params.page ? parseInt(params.page) : 1;
  const posts = await fetchPosts(page);

  return {
    page,
    posts
  };
}
```

### Rest Parameters

**src/routes/docs/[...path]/+page.svelte:**
```svelte
<!-- Matches /docs/getting-started, /docs/api/reference, etc. -->
<script>
  export let data;
</script>

<nav>
  {#each data.breadcrumbs as crumb, i}
    {#if i > 0}<span>/</span>{/if}
    <a href={crumb.href}>{crumb.label}</a>
  {/each}
</nav>

<div>
  {@html data.content}
</div>
```

**src/routes/docs/[...path]/+page.js:**
```javascript
export async function load({ params }) {
  const path = params.path || '';
  const segments = path.split('/').filter(Boolean);

  const breadcrumbs = segments.map((segment, i) => ({
    label: segment,
    href: '/docs/' + segments.slice(0, i + 1).join('/')
  }));

  const content = await loadDocumentation(path);

  return {
    breadcrumbs,
    content
  };
}
```

### Route Groups

Route groups allow you to organize routes without affecting the URL structure.

```
routes/
└── (marketing)/            # Group name in parentheses
    ├── +layout.svelte     # Shared layout
    ├── about/
    │   └── +page.svelte   # /about (not /(marketing)/about)
    └── contact/
        └── +page.svelte   # /contact
```

### Route Matching

**src/routes/products/[id=integer]/+page.svelte:**

Uses a matcher to validate route parameters.

**src/params/integer.js:**
```javascript
export function match(param) {
  return /^\d+$/.test(param);
}
```

This ensures `/products/123` matches but `/products/abc` does not.

---

## Loading Data

### Universal Load Functions

Run on both server and client.

**src/routes/blog/+page.js:**
```javascript
export async function load({ fetch, params, url }) {
  // Use SvelteKit's fetch for credentials and relative URLs
  const response = await fetch('/api/posts');
  const posts = await response.json();

  return {
    posts,
    currentPath: url.pathname
  };
}
```

### Server Load Functions

Run only on the server. Can access databases, environment variables, etc.

**src/routes/dashboard/+page.server.js:**
```javascript
import { db } from '$lib/server/database';

export async function load({ locals, cookies }) {
  // Access server-only resources
  const userId = locals.user?.id;

  if (!userId) {
    redirect(303, '/login');
  }

  const stats = await db.query('SELECT * FROM stats WHERE user_id = ?', [userId]);
  const preferences = cookies.get('preferences');

  return {
    stats,
    preferences: preferences ? JSON.parse(preferences) : {}
  };
}
```

### Load Function Parameters

```javascript
export async function load({
  params,       // Route parameters
  url,          // URL object
  route,        // Route information
  fetch,        // Enhanced fetch
  setHeaders,   // Set response headers
  depends,      // Track dependencies
  parent,       // Parent load data
  locals,       // Server-only locals
  cookies       // Server-only cookies
}) {
  // Load logic
}
```

### Streaming Data with Promises

**src/routes/dashboard/+page.server.js:**
```javascript
export async function load() {
  // Fast data loads immediately
  const user = await getUser();

  // Slow data streams in later
  return {
    user,
    // Return promise directly - SvelteKit will await it
    stats: getStats(),
    notifications: getNotifications()
  };
}
```

**src/routes/dashboard/+page.svelte:**
```svelte
<script>
  export let data;
</script>

<h1>Welcome, {data.user.name}</h1>

{#await data.stats}
  <p>Loading stats...</p>
{:then stats}
  <div class="stats">
    <div>Posts: {stats.posts}</div>
    <div>Views: {stats.views}</div>
  </div>
{/await}

{#await data.notifications}
  <p>Loading notifications...</p>
{:then notifications}
  <ul>
    {#each notifications as notif}
      <li>{notif.message}</li>
    {/each}
  </ul>
{/await}
```

### Parent Data

Access data from parent layouts.

**src/routes/blog/[slug]/+page.js:**
```javascript
export async function load({ params, parent }) {
  // Get data from parent layout
  const { categories } = await parent();

  const post = await getPost(params.slug);

  return {
    post,
    relatedPosts: getRelatedPosts(post, categories)
  };
}
```

### Invalidation

**Manual invalidation:**
```javascript
import { invalidate, invalidateAll } from '$app/navigation';

// Invalidate specific URL
await invalidate('/api/posts');

// Invalidate by dependency
await invalidate('posts:list');

// Invalidate all
await invalidateAll();
```

**Dependency tracking:**
```javascript
// In load function
export async function load({ fetch, depends }) {
  depends('posts:list');

  const posts = await fetch('/api/posts').then(r => r.json());
  return { posts };
}

// Later, in component
import { invalidate } from '$app/navigation';

async function refreshPosts() {
  await invalidate('posts:list');
}
```

---

## Form Actions

Form actions enable progressive enhancement for form submissions.

### Basic Form Action

**src/routes/login/+page.server.js:**
```javascript
import { fail, redirect } from '@sveltejs/kit';
import { db } from '$lib/server/database';

export const actions = {
  default: async ({ request, cookies }) => {
    const data = await request.formData();
    const email = data.get('email');
    const password = data.get('password');

    // Validation
    if (!email || !password) {
      return fail(400, {
        email,
        missing: true
      });
    }

    // Authenticate
    const user = await db.authenticate(email, password);

    if (!user) {
      return fail(400, {
        email,
        credentials: true
      });
    }

    // Set session cookie
    cookies.set('session', user.sessionId, {
      path: '/',
      httpOnly: true,
      sameSite: 'strict',
      secure: process.env.NODE_ENV === 'production',
      maxAge: 60 * 60 * 24 * 7 // 1 week
    });

    throw redirect(303, '/dashboard');
  }
};
```

**src/routes/login/+page.svelte:**
```svelte
<script>
  import { enhance } from '$app/forms';
  export let form;
</script>

<form method="POST" use:enhance>
  <label>
    Email
    <input
      name="email"
      type="email"
      value={form?.email ?? ''}
      required
    />
  </label>

  <label>
    Password
    <input name="password" type="password" required />
  </label>

  {#if form?.missing}
    <p class="error">Please fill in all fields</p>
  {/if}

  {#if form?.credentials}
    <p class="error">Invalid credentials</p>
  {/if}

  <button type="submit">Log in</button>
</form>
```

### Named Actions

**src/routes/todos/+page.server.js:**
```javascript
import { fail } from '@sveltejs/kit';
import { db } from '$lib/server/database';

export async function load() {
  const todos = await db.getTodos();
  return { todos };
}

export const actions = {
  create: async ({ request, locals }) => {
    const data = await request.formData();
    const text = data.get('text');

    if (!text) {
      return fail(400, { text, missing: true });
    }

    await db.createTodo({
      userId: locals.user.id,
      text
    });

    return { success: true };
  },

  update: async ({ request }) => {
    const data = await request.formData();
    const id = data.get('id');
    const completed = data.get('completed') === 'true';

    await db.updateTodo(id, { completed });

    return { success: true };
  },

  delete: async ({ request }) => {
    const data = await request.formData();
    const id = data.get('id');

    await db.deleteTodo(id);

    return { success: true };
  }
};
```

**src/routes/todos/+page.svelte:**
```svelte
<script>
  import { enhance } from '$app/forms';
  export let data;
  export let form;
</script>

<!-- Create form -->
<form method="POST" action="?/create" use:enhance>
  <input name="text" placeholder="New todo..." />
  <button type="submit">Add</button>
  {#if form?.missing}
    <span class="error">Required</span>
  {/if}
</form>

<!-- Todo list -->
{#each data.todos as todo}
  <div class="todo">
    <!-- Update form -->
    <form method="POST" action="?/update" use:enhance>
      <input type="hidden" name="id" value={todo.id} />
      <input type="hidden" name="completed" value={!todo.completed} />
      <button type="submit">
        {todo.completed ? '☑' : '☐'}
      </button>
    </form>

    <span class:completed={todo.completed}>
      {todo.text}
    </span>

    <!-- Delete form -->
    <form method="POST" action="?/delete" use:enhance>
      <input type="hidden" name="id" value={todo.id} />
      <button type="submit">Delete</button>
    </form>
  </div>
{/each}
```

### Custom use:enhance

**src/routes/newsletter/+page.svelte:**
```svelte
<script>
  import { enhance } from '$app/forms';

  let loading = false;
  let message = '';

  function handleSubmit() {
    loading = true;
    message = '';

    return async ({ result, update }) => {
      loading = false;

      if (result.type === 'success') {
        message = 'Subscribed successfully!';
        // Optionally don't update form
        // await update();
      } else if (result.type === 'failure') {
        message = result.data?.message || 'Subscription failed';
        await update();
      }
    };
  }
</script>

<form
  method="POST"
  use:enhance={handleSubmit}
>
  <input
    name="email"
    type="email"
    placeholder="your@email.com"
    disabled={loading}
  />

  <button type="submit" disabled={loading}>
    {loading ? 'Subscribing...' : 'Subscribe'}
  </button>
</form>

{#if message}
  <p class="message">{message}</p>
{/if}
```

### File Upload Action

**src/routes/upload/+page.server.js:**
```javascript
import { fail } from '@sveltejs/kit';
import { writeFile } from 'fs/promises';
import path from 'path';

export const actions = {
  upload: async ({ request }) => {
    const data = await request.formData();
    const file = data.get('file');

    if (!file || !file.size) {
      return fail(400, { missing: true });
    }

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      return fail(400, { invalidType: true });
    }

    // Validate file size (5MB max)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
      return fail(400, { tooLarge: true });
    }

    // Save file
    const filename = `${Date.now()}-${file.name}`;
    const filepath = path.join('static', 'uploads', filename);

    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(filepath, buffer);

    return {
      success: true,
      url: `/uploads/${filename}`
    };
  }
};
```

**src/routes/upload/+page.svelte:**
```svelte
<script>
  import { enhance } from '$app/forms';
  export let form;

  let files;
  let preview = '';

  $: if (files && files[0]) {
    const reader = new FileReader();
    reader.onload = (e) => preview = e.target.result;
    reader.readAsDataURL(files[0]);
  }
</script>

<form method="POST" action="?/upload" enctype="multipart/form-data" use:enhance>
  <input
    type="file"
    name="file"
    accept="image/*"
    bind:files
    required
  />

  {#if preview}
    <img src={preview} alt="Preview" />
  {/if}

  {#if form?.missing}
    <p class="error">Please select a file</p>
  {/if}

  {#if form?.invalidType}
    <p class="error">Only images are allowed</p>
  {/if}

  {#if form?.tooLarge}
    <p class="error">File too large (max 5MB)</p>
  {/if}

  {#if form?.success}
    <p class="success">Upload successful!</p>
    <img src={form.url} alt="Uploaded" />
  {/if}

  <button type="submit">Upload</button>
</form>
```

---

## Hooks

### Server Hooks

**src/hooks.server.js:**
```javascript
import { sequence } from '@sveltejs/kit/hooks';
import { db } from '$lib/server/database';

// Authentication hook
async function handleAuth({ event, resolve }) {
  const sessionId = event.cookies.get('session');

  if (sessionId) {
    const user = await db.getUserBySessionId(sessionId);
    if (user) {
      event.locals.user = user;
    }
  }

  return resolve(event);
}

// Logging hook
async function handleLog({ event, resolve }) {
  const start = Date.now();

  const response = await resolve(event);

  const duration = Date.now() - start;
  console.log(`${event.request.method} ${event.url.pathname} ${response.status} ${duration}ms`);

  return response;
}

// Protected routes hook
async function handleProtected({ event, resolve }) {
  if (event.url.pathname.startsWith('/admin')) {
    if (!event.locals.user?.isAdmin) {
      return new Response('Forbidden', { status: 403 });
    }
  }

  return resolve(event);
}

// Custom response headers
async function handleHeaders({ event, resolve }) {
  const response = await resolve(event);

  response.headers.set('X-Custom-Header', 'SvelteKit');

  return response;
}

// Combine multiple hooks with sequence
export const handle = sequence(
  handleAuth,
  handleProtected,
  handleLog,
  handleHeaders
);

// Handle fetch requests
export async function handleFetch({ request, fetch }) {
  // Modify fetch requests made during SSR
  if (request.url.startsWith('https://api.example.com/')) {
    request = new Request(
      request.url,
      {
        ...request,
        headers: {
          ...request.headers,
          'Authorization': `Bearer ${process.env.API_TOKEN}`
        }
      }
    );
  }

  return fetch(request);
}

// Handle errors
export function handleError({ error, event }) {
  // Log error to monitoring service
  console.error('Error:', error, 'Event:', event);

  return {
    message: 'An error occurred',
    code: error?.code ?? 'UNKNOWN'
  };
}
```

### Client Hooks

**src/hooks.client.js:**
```javascript
import { dev } from '$app/environment';

// Handle errors on client
export function handleError({ error, event }) {
  if (dev) {
    console.error('Client error:', error, event);
  }

  // Send to error tracking service
  if (!dev && window.Sentry) {
    Sentry.captureException(error);
  }

  return {
    message: 'Something went wrong'
  };
}
```

---

## Page Options

Configure page behavior with export statements.

**src/routes/blog/+page.js:**
```javascript
// Prerender this page at build time
export const prerender = true;

// Disable server-side rendering
export const ssr = false;

// Disable client-side rendering
export const csr = false;

// Trailing slash behavior: 'always' | 'never' | 'ignore'
export const trailingSlash = 'never';

export async function load({ fetch }) {
  const posts = await fetch('/api/posts').then(r => r.json());
  return { posts };
}
```

### Prerendering

**Static site generation:**
```javascript
// src/routes/+layout.js
export const prerender = true;
```

**Per-route control:**
```javascript
// src/routes/blog/+page.js
export const prerender = true;

// src/routes/dashboard/+page.js
export const prerender = false;
```

**Dynamic prerendering:**
```javascript
// src/routes/blog/[slug]/+page.server.js
export const prerender = true;

export async function entries() {
  // Return all possible parameter values
  const posts = await getAllPosts();
  return posts.map(post => ({
    slug: post.slug
  }));
}

export async function load({ params }) {
  const post = await getPost(params.slug);
  return { post };
}
```

### SSR and CSR Control

```javascript
// Disable SSR for a specific page (SPA mode)
export const ssr = false;
export const csr = true;

// Server-only rendering (no hydration)
export const ssr = true;
export const csr = false;

// Full stack mode (default)
export const ssr = true;
export const csr = true;
```

---

## API Routes

Create API endpoints with `+server.js` files.

### Basic API Routes

**src/routes/api/hello/+server.js:**
```javascript
import { json } from '@sveltejs/kit';

export async function GET() {
  return json({
    message: 'Hello from SvelteKit API'
  });
}

export async function POST({ request }) {
  const data = await request.json();

  return json({
    received: data
  }, {
    status: 201
  });
}
```

### CRUD API

**src/routes/api/posts/+server.js:**
```javascript
import { json, error } from '@sveltejs/kit';
import { db } from '$lib/server/database';

// GET /api/posts
export async function GET({ url }) {
  const limit = parseInt(url.searchParams.get('limit') || '10');
  const offset = parseInt(url.searchParams.get('offset') || '0');

  const posts = await db.query(
    'SELECT * FROM posts ORDER BY created_at DESC LIMIT ? OFFSET ?',
    [limit, offset]
  );

  return json(posts);
}

// POST /api/posts
export async function POST({ request, locals }) {
  if (!locals.user) {
    throw error(401, 'Unauthorized');
  }

  const { title, content } = await request.json();

  if (!title || !content) {
    throw error(400, 'Title and content are required');
  }

  const post = await db.createPost({
    title,
    content,
    authorId: locals.user.id
  });

  return json(post, { status: 201 });
}
```

**src/routes/api/posts/[id]/+server.js:**
```javascript
import { json, error } from '@sveltejs/kit';
import { db } from '$lib/server/database';

// GET /api/posts/:id
export async function GET({ params }) {
  const post = await db.getPost(params.id);

  if (!post) {
    throw error(404, 'Post not found');
  }

  return json(post);
}

// PUT /api/posts/:id
export async function PUT({ params, request, locals }) {
  const post = await db.getPost(params.id);

  if (!post) {
    throw error(404, 'Post not found');
  }

  if (post.authorId !== locals.user?.id) {
    throw error(403, 'Forbidden');
  }

  const { title, content } = await request.json();

  const updated = await db.updatePost(params.id, {
    title,
    content
  });

  return json(updated);
}

// DELETE /api/posts/:id
export async function DELETE({ params, locals }) {
  const post = await db.getPost(params.id);

  if (!post) {
    throw error(404, 'Post not found');
  }

  if (post.authorId !== locals.user?.id && !locals.user?.isAdmin) {
    throw error(403, 'Forbidden');
  }

  await db.deletePost(params.id);

  return new Response(null, { status: 204 });
}
```

### Cookies and Headers

**src/routes/api/preferences/+server.js:**
```javascript
import { json } from '@sveltejs/kit';

export async function GET({ cookies }) {
  const preferences = cookies.get('preferences');

  return json(
    preferences ? JSON.parse(preferences) : {}
  );
}

export async function POST({ request, cookies }) {
  const preferences = await request.json();

  cookies.set('preferences', JSON.stringify(preferences), {
    path: '/',
    maxAge: 60 * 60 * 24 * 365, // 1 year
    httpOnly: false,
    sameSite: 'lax'
  });

  return json({ success: true });
}
```

### Streaming Responses

**src/routes/api/stream/+server.js:**
```javascript
export async function GET() {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      for (let i = 0; i < 10; i++) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        controller.enqueue(encoder.encode(`data: ${i}\n\n`));
      }
      controller.close();
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache'
    }
  });
}
```

### File Downloads

**src/routes/api/download/[filename]/+server.js:**
```javascript
import { error } from '@sveltejs/kit';
import { readFile } from 'fs/promises';
import path from 'path';

export async function GET({ params }) {
  const filename = params.filename;
  const filepath = path.join('static', 'downloads', filename);

  try {
    const file = await readFile(filepath);

    return new Response(file, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': `attachment; filename="${filename}"`
      }
    });
  } catch (err) {
    throw error(404, 'File not found');
  }
}
```

---

## State Management

### Built-in Stores

**$app/stores:**
```svelte
<script>
  import { page, navigating, updated } from '$app/stores';

  // page - contains url, params, route, status, error, data, form
  $: console.log('Current URL:', $page.url.pathname);
  $: console.log('Route params:', $page.params);
  $: console.log('Page data:', $page.data);

  // navigating - contains from, to, type during navigation
  $: if ($navigating) {
    console.log('Navigating from', $navigating.from, 'to', $navigating.to);
  }

  // updated - becomes true when new version deployed
  $: if ($updated) {
    // Reload to get new version
    location.reload();
  }
</script>

<nav>
  <a href="/" class:active={$page.url.pathname === '/'}>
    Home
  </a>
  <a href="/about" class:active={$page.url.pathname === '/about'}>
    About
  </a>
</nav>

{#if $navigating}
  <div class="loading-bar" />
{/if}
```

### Custom Stores

**src/lib/stores/cart.js:**
```javascript
import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

function createCart() {
  // Load from localStorage if in browser
  const stored = browser && localStorage.getItem('cart');
  const initial = stored ? JSON.parse(stored) : [];

  const { subscribe, set, update } = writable(initial);

  // Sync to localStorage
  if (browser) {
    subscribe(value => {
      localStorage.setItem('cart', JSON.stringify(value));
    });
  }

  return {
    subscribe,

    addItem: (item) => update(items => {
      const existing = items.find(i => i.id === item.id);
      if (existing) {
        return items.map(i =>
          i.id === item.id
            ? { ...i, quantity: i.quantity + 1 }
            : i
        );
      }
      return [...items, { ...item, quantity: 1 }];
    }),

    removeItem: (id) => update(items =>
      items.filter(i => i.id !== id)
    ),

    updateQuantity: (id, quantity) => update(items =>
      items.map(i =>
        i.id === id ? { ...i, quantity } : i
      )
    ),

    clear: () => set([])
  };
}

export const cart = createCart();

export const cartTotal = derived(
  cart,
  $cart => $cart.reduce((sum, item) =>
    sum + item.price * item.quantity, 0
  )
);

export const cartCount = derived(
  cart,
  $cart => $cart.reduce((sum, item) =>
    sum + item.quantity, 0
  )
);
```

**Using the cart store:**
```svelte
<script>
  import { cart, cartTotal, cartCount } from '$lib/stores/cart';

  export let data;
</script>

<header>
  <a href="/cart">
    Cart ({$cartCount}) - ${$cartTotal.toFixed(2)}
  </a>
</header>

{#each data.products as product}
  <div class="product">
    <h3>{product.name}</h3>
    <p>${product.price}</p>
    <button on:click={() => cart.addItem(product)}>
      Add to Cart
    </button>
  </div>
{/each}
```

### Context-based State

**src/routes/+layout.svelte:**
```svelte
<script>
  import { setContext } from 'svelte';
  import { writable } from 'svelte/store';

  export let data;

  // Create app-wide state
  const theme = writable(data.userPreferences?.theme || 'light');
  const user = writable(data.user);

  setContext('app', {
    theme,
    user,
    toggleTheme: () => theme.update(t => t === 'light' ? 'dark' : 'light')
  });
</script>

<div class="app" data-theme={$theme}>
  <slot />
</div>
```

**Using context in components:**
```svelte
<script>
  import { getContext } from 'svelte';

  const { theme, user, toggleTheme } = getContext('app');
</script>

<header>
  <span>Welcome, {$user?.name || 'Guest'}</span>
  <button on:click={toggleTheme}>
    {$theme === 'light' ? '🌙' : '☀️'}
  </button>
</header>
```

---

## Navigation

### Programmatic Navigation

```svelte
<script>
  import { goto, invalidate, invalidateAll } from '$app/navigation';
  import { beforeNavigate, afterNavigate } from '$app/navigation';

  async function navigate() {
    // Navigate to a route
    await goto('/dashboard');

    // Navigate with options
    await goto('/dashboard', {
      replaceState: true,  // Replace history instead of push
      noScroll: true,      // Don't scroll to top
      keepFocus: true,     // Keep focus on current element
      invalidateAll: true  // Re-run all load functions
    });
  }

  // Intercept navigation
  beforeNavigate(({ from, to, cancel }) => {
    console.log('Navigating from', from, 'to', to);

    // Cancel navigation based on condition
    if (unsavedChanges) {
      if (!confirm('You have unsaved changes. Leave anyway?')) {
        cancel();
      }
    }
  });

  // Handle after navigation
  afterNavigate(({ from, to, type }) => {
    console.log('Navigation complete:', type);
    // type: 'link' | 'goto' | 'popstate'
  });
</script>
```

### Prefetching

```svelte
<script>
  import { preloadData, preloadCode } from '$app/navigation';

  async function prefetch() {
    // Preload data and code
    await preloadData('/blog/post-1');

    // Just preload code
    await preloadCode('/dashboard');
  }
</script>

<!-- Automatic prefetch on hover/tap -->
<a href="/blog" data-sveltekit-preload-data="hover">
  Blog
</a>

<!-- Prefetch on viewport -->
<a href="/about" data-sveltekit-preload-data="viewport">
  About
</a>

<!-- Prefetch on tap (mobile) -->
<a href="/contact" data-sveltekit-preload-data="tap">
  Contact
</a>

<!-- Disable prefetch -->
<a href="/external" data-sveltekit-reload>
  External Site
</a>
```

### Link Options

```svelte
<!-- Standard navigation -->
<a href="/about">About</a>

<!-- External link (skip SvelteKit routing) -->
<a href="https://example.com" data-sveltekit-reload>
  External
</a>

<!-- Disable prefetch -->
<a href="/slow-page" data-sveltekit-preload-data="off">
  Slow Page
</a>

<!-- Programmatic prefetch -->
<a
  href="/dashboard"
  on:mouseenter={() => preloadData('/dashboard')}
>
  Dashboard
</a>
```

---

## Error Handling

### Custom Error Pages

**src/routes/+error.svelte:**
```svelte
<script>
  import { page } from '$app/stores';
</script>

<div class="error">
  <h1>{$page.status}</h1>
  <p>{$page.error?.message || 'An error occurred'}</p>

  {#if $page.status === 404}
    <p>The page you're looking for doesn't exist.</p>
  {:else if $page.status === 500}
    <p>Internal server error. Please try again later.</p>
  {/if}

  <a href="/">Go home</a>
</div>

<style>
  .error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    text-align: center;
  }

  h1 {
    font-size: 4rem;
    color: #ff3e00;
  }
</style>
```

### Throwing Errors

```javascript
import { error, redirect } from '@sveltejs/kit';

export async function load({ params, locals }) {
  // 404 error
  const post = await getPost(params.id);
  if (!post) {
    throw error(404, {
      message: 'Post not found'
    });
  }

  // 403 error
  if (post.authorId !== locals.user?.id) {
    throw error(403, 'You do not have permission to view this post');
  }

  // 500 error
  try {
    const data = await fetchData();
    return { data };
  } catch (err) {
    throw error(500, 'Failed to load data');
  }

  // Redirect
  if (!locals.user) {
    throw redirect(303, '/login');
  }

  return { post };
}
```

### Expected vs Unexpected Errors

```javascript
// src/routes/api/posts/+server.js
import { error } from '@sveltejs/kit';

export async function GET({ params }) {
  try {
    const post = await db.getPost(params.id);

    if (!post) {
      // Expected error - shown to user
      throw error(404, 'Post not found');
    }

    return json(post);
  } catch (err) {
    // Unexpected error - logged, generic message shown
    if (err.status) {
      throw err;
    }

    console.error('Database error:', err);
    throw error(500, 'Internal server error');
  }
}
```

---

## Advanced Patterns

### Authentication Pattern

**src/hooks.server.js:**
```javascript
import { redirect } from '@sveltejs/kit';
import { db } from '$lib/server/database';

export async function handle({ event, resolve }) {
  // Get session from cookie
  const sessionId = event.cookies.get('session');

  if (sessionId) {
    const user = await db.getUserBySession(sessionId);
    if (user) {
      event.locals.user = user;
    } else {
      // Invalid session
      event.cookies.delete('session', { path: '/' });
    }
  }

  // Protected routes
  if (event.url.pathname.startsWith('/dashboard')) {
    if (!event.locals.user) {
      throw redirect(303, '/login');
    }
  }

  // Admin routes
  if (event.url.pathname.startsWith('/admin')) {
    if (!event.locals.user?.isAdmin) {
      throw redirect(303, '/');
    }
  }

  return resolve(event);
}
```

**src/routes/login/+page.server.js:**
```javascript
import { fail, redirect } from '@sveltejs/kit';
import { db } from '$lib/server/database';
import bcrypt from 'bcrypt';

export const actions = {
  login: async ({ request, cookies }) => {
    const data = await request.formData();
    const email = data.get('email');
    const password = data.get('password');

    // Validate
    if (!email || !password) {
      return fail(400, {
        email,
        missing: true
      });
    }

    // Get user
    const user = await db.getUserByEmail(email);
    if (!user) {
      return fail(400, {
        email,
        invalid: true
      });
    }

    // Verify password
    const valid = await bcrypt.compare(password, user.passwordHash);
    if (!valid) {
      return fail(400, {
        email,
        invalid: true
      });
    }

    // Create session
    const sessionId = crypto.randomUUID();
    await db.createSession({
      id: sessionId,
      userId: user.id,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000) // 7 days
    });

    // Set cookie
    cookies.set('session', sessionId, {
      path: '/',
      httpOnly: true,
      sameSite: 'strict',
      secure: process.env.NODE_ENV === 'production',
      maxAge: 60 * 60 * 24 * 7 // 7 days
    });

    throw redirect(303, '/dashboard');
  },

  logout: async ({ cookies, locals }) => {
    const sessionId = cookies.get('session');
    if (sessionId) {
      await db.deleteSession(sessionId);
    }

    cookies.delete('session', { path: '/' });
    throw redirect(303, '/');
  }
};
```

### Pagination Pattern

**src/routes/blog/+page.server.js:**
```javascript
import { error } from '@sveltejs/kit';
import { db } from '$lib/server/database';

const POSTS_PER_PAGE = 10;

export async function load({ url }) {
  const page = parseInt(url.searchParams.get('page') || '1');

  if (page < 1) {
    throw error(400, 'Invalid page number');
  }

  const offset = (page - 1) * POSTS_PER_PAGE;

  const [posts, totalCount] = await Promise.all([
    db.getPosts({ limit: POSTS_PER_PAGE, offset }),
    db.getPostCount()
  ]);

  const totalPages = Math.ceil(totalCount / POSTS_PER_PAGE);

  if (page > totalPages && totalPages > 0) {
    throw error(404, 'Page not found');
  }

  return {
    posts,
    pagination: {
      page,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1
    }
  };
}
```

**src/routes/blog/+page.svelte:**
```svelte
<script>
  export let data;
</script>

<div class="posts">
  {#each data.posts as post}
    <article>
      <h2><a href="/blog/{post.slug}">{post.title}</a></h2>
      <p>{post.excerpt}</p>
    </article>
  {/each}
</div>

<nav class="pagination">
  {#if data.pagination.hasPrev}
    <a href="?page={data.pagination.page - 1}">
      ← Previous
    </a>
  {/if}

  <span>
    Page {data.pagination.page} of {data.pagination.totalPages}
  </span>

  {#if data.pagination.hasNext}
    <a href="?page={data.pagination.page + 1}">
      Next →
    </a>
  {/if}
</nav>
```

### Search Pattern

**src/routes/search/+page.svelte:**
```svelte
<script>
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';

  export let data;

  let query = $page.url.searchParams.get('q') || '';
  let timeout;

  function handleInput() {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      if (query) {
        goto(`?q=${encodeURIComponent(query)}`, {
          keepFocus: true,
          noScroll: true
        });
      }
    }, 300);
  }
</script>

<form on:submit|preventDefault>
  <input
    type="search"
    bind:value={query}
    on:input={handleInput}
    placeholder="Search..."
  />
</form>

{#if data.results}
  <div class="results">
    <p>{data.results.length} results for "{data.query}"</p>

    {#each data.results as result}
      <article>
        <h3><a href={result.url}>{result.title}</a></h3>
        <p>{result.excerpt}</p>
      </article>
    {/each}
  </div>
{/if}
```

**src/routes/search/+page.server.js:**
```javascript
export async function load({ url }) {
  const query = url.searchParams.get('q');

  if (!query) {
    return { results: null, query: '' };
  }

  const results = await searchDatabase(query);

  return {
    results,
    query
  };
}
```

---

## Authentication

### JWT-based Authentication

**src/lib/server/auth.js:**
```javascript
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { db } from './database';

const JWT_SECRET = process.env.JWT_SECRET;
const JWT_EXPIRES_IN = '7d';

export async function hashPassword(password) {
  return bcrypt.hash(password, 10);
}

export async function verifyPassword(password, hash) {
  return bcrypt.compare(password, hash);
}

export function generateToken(user) {
  return jwt.sign(
    { userId: user.id, email: user.email },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRES_IN }
  );
}

export function verifyToken(token) {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch {
    return null;
  }
}

export async function getUserFromToken(token) {
  const payload = verifyToken(token);
  if (!payload) return null;

  return db.getUserById(payload.userId);
}
```

**src/hooks.server.js:**
```javascript
import { getUserFromToken } from '$lib/server/auth';

export async function handle({ event, resolve }) {
  const token = event.cookies.get('auth_token');

  if (token) {
    const user = await getUserFromToken(token);
    if (user) {
      event.locals.user = user;
    }
  }

  return resolve(event);
}
```

### OAuth Integration

**src/routes/auth/github/+server.js:**
```javascript
import { redirect } from '@sveltejs/kit';

const GITHUB_CLIENT_ID = process.env.GITHUB_CLIENT_ID;
const GITHUB_CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET;
const CALLBACK_URL = process.env.GITHUB_CALLBACK_URL;

export async function GET() {
  const state = crypto.randomUUID();

  const url = new URL('https://github.com/login/oauth/authorize');
  url.searchParams.set('client_id', GITHUB_CLIENT_ID);
  url.searchParams.set('redirect_uri', CALLBACK_URL);
  url.searchParams.set('state', state);
  url.searchParams.set('scope', 'user:email');

  throw redirect(302, url.toString());
}
```

**src/routes/auth/github/callback/+server.js:**
```javascript
import { error, redirect } from '@sveltejs/kit';
import { db } from '$lib/server/database';
import { generateToken } from '$lib/server/auth';

export async function GET({ url, cookies }) {
  const code = url.searchParams.get('code');
  const state = url.searchParams.get('state');

  if (!code) {
    throw error(400, 'Missing code');
  }

  // Exchange code for access token
  const tokenResponse = await fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    body: JSON.stringify({
      client_id: process.env.GITHUB_CLIENT_ID,
      client_secret: process.env.GITHUB_CLIENT_SECRET,
      code
    })
  });

  const { access_token } = await tokenResponse.json();

  // Get user info
  const userResponse = await fetch('https://api.github.com/user', {
    headers: {
      'Authorization': `Bearer ${access_token}`
    }
  });

  const githubUser = await userResponse.json();

  // Create or update user
  let user = await db.getUserByGithubId(githubUser.id);

  if (!user) {
    user = await db.createUser({
      githubId: githubUser.id,
      email: githubUser.email,
      name: githubUser.name,
      avatar: githubUser.avatar_url
    });
  }

  // Generate JWT
  const token = generateToken(user);

  cookies.set('auth_token', token, {
    path: '/',
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 60 * 60 * 24 * 7 // 7 days
  });

  throw redirect(303, '/dashboard');
}
```

---

## Database Integration

### Prisma Integration

**Install Prisma:**
```bash
npm install -D prisma
npm install @prisma/client
npx prisma init
```

**prisma/schema.prisma:**
```prisma
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

generator client {
  provider = "prisma-client-js"
}

model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String?
  posts     Post[]
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}

model Post {
  id        String   @id @default(cuid())
  title     String
  content   String
  published Boolean  @default(false)
  author    User     @relation(fields: [authorId], references: [id])
  authorId  String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}
```

**src/lib/server/database.js:**
```javascript
import { PrismaClient } from '@prisma/client';

const prisma = global.prisma || new PrismaClient();

if (process.env.NODE_ENV === 'development') {
  global.prisma = prisma;
}

export { prisma };
```

**src/routes/blog/+page.server.js:**
```javascript
import { prisma } from '$lib/server/database';

export async function load() {
  const posts = await prisma.post.findMany({
    where: { published: true },
    include: { author: true },
    orderBy: { createdAt: 'desc' }
  });

  return { posts };
}

export const actions = {
  create: async ({ request, locals }) => {
    const data = await request.formData();
    const title = data.get('title');
    const content = data.get('content');

    const post = await prisma.post.create({
      data: {
        title,
        content,
        authorId: locals.user.id
      }
    });

    return { success: true, post };
  }
};
```

### Drizzle ORM Integration

**src/lib/server/db/schema.ts:**
```typescript
import { pgTable, serial, text, timestamp, boolean } from 'drizzle-orm/pg-core';

export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  email: text('email').notNull().unique(),
  name: text('name'),
  createdAt: timestamp('created_at').defaultNow()
});

export const posts = pgTable('posts', {
  id: serial('id').primaryKey(),
  title: text('title').notNull(),
  content: text('content').notNull(),
  published: boolean('published').default(false),
  authorId: serial('author_id').references(() => users.id),
  createdAt: timestamp('created_at').defaultNow()
});
```

**src/lib/server/db/index.ts:**
```typescript
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema';

const client = postgres(process.env.DATABASE_URL!);
export const db = drizzle(client, { schema });
```

---

## Building and Deployment

### Adapters

SvelteKit uses adapters to deploy to different platforms.

**Install adapter:**
```bash
# Automatic adapter selection
npm install -D @sveltejs/adapter-auto

# Node.js
npm install -D @sveltejs/adapter-node

# Vercel
npm install -D @sveltejs/adapter-vercel

# Netlify
npm install -D @sveltejs/adapter-netlify

# Cloudflare Pages
npm install -D @sveltejs/adapter-cloudflare

# Static site (SPA/SSG)
npm install -D @sveltejs/adapter-static
```

**svelte.config.js:**
```javascript
import adapter from '@sveltejs/adapter-node';

export default {
  kit: {
    adapter: adapter({
      out: 'build',
      precompress: true,
      envPrefix: 'MY_'
    })
  }
};
```

### Building for Production

```bash
# Build the application
npm run build

# Preview production build
npm run preview

# Run production build (with adapter-node)
node build
```

### Environment Variables

**.env:**
```bash
# Private (server-only)
DATABASE_URL="postgresql://..."
JWT_SECRET="secret"
API_KEY="key"

# Public (exposed to client)
PUBLIC_API_URL="https://api.example.com"
PUBLIC_SITE_NAME="My SvelteKit App"
```

**Using environment variables:**
```javascript
// src/routes/+page.server.js
import { env } from '$env/dynamic/private';
// or
import { DATABASE_URL } from '$env/static/private';

export async function load() {
  const apiKey = env.API_KEY; // or API_KEY from static import
  // ...
}
```

```svelte
<!-- src/routes/+page.svelte -->
<script>
  import { env } from '$env/dynamic/public';
  // or
  import { PUBLIC_API_URL } from '$env/static/public';

  const apiUrl = env.PUBLIC_API_URL; // or PUBLIC_API_URL from static import
</script>
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build
RUN npm prune --production

FROM node:18-alpine

WORKDIR /app

COPY --from=builder /app/build build/
COPY --from=builder /app/node_modules node_modules/
COPY package.json .

EXPOSE 3000
ENV NODE_ENV=production

CMD ["node", "build"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Vercel Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Production deployment
vercel --prod
```

**vercel.json (optional):**
```json
{
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "framework": "sveltekit",
  "installCommand": "npm install"
}
```

---

## Performance Optimization

### Code Splitting

```svelte
<script>
  // Dynamic imports for code splitting
  let HeavyComponent;

  async function loadComponent() {
    const module = await import('$lib/components/HeavyComponent.svelte');
    HeavyComponent = module.default;
  }
</script>

<button on:click={loadComponent}>
  Load Heavy Component
</button>

{#if HeavyComponent}
  <svelte:component this={HeavyComponent} />
{/if}
```

### Preloading

```svelte
<script>
  import { preloadData } from '$app/navigation';

  function handleMouseEnter() {
    preloadData('/dashboard');
  }
</script>

<a
  href="/dashboard"
  on:mouseenter={handleMouseEnter}
>
  Dashboard
</a>

<!-- Or use data attributes -->
<a href="/blog" data-sveltekit-preload-data="hover">
  Blog
</a>
```

### Image Optimization

**Using modern formats:**
```svelte
<picture>
  <source
    srcset="/images/hero.webp"
    type="image/webp"
  />
  <source
    srcset="/images/hero.jpg"
    type="image/jpeg"
  />
  <img
    src="/images/hero.jpg"
    alt="Hero image"
    loading="lazy"
    width="1200"
    height="600"
  />
</picture>
```

**Responsive images:**
```svelte
<img
  srcset="
    /images/small.jpg 400w,
    /images/medium.jpg 800w,
    /images/large.jpg 1200w
  "
  sizes="(max-width: 600px) 400px, (max-width: 1200px) 800px, 1200px"
  src="/images/medium.jpg"
  alt="Responsive image"
  loading="lazy"
/>
```

### Caching Strategies

**src/hooks.server.js:**
```javascript
export async function handle({ event, resolve }) {
  const response = await resolve(event);

  // Cache static assets
  if (event.url.pathname.startsWith('/images/')) {
    response.headers.set('Cache-Control', 'public, max-age=31536000, immutable');
  }

  // Cache API responses
  if (event.url.pathname.startsWith('/api/')) {
    response.headers.set('Cache-Control', 'public, max-age=60');
  }

  return response;
}
```

---

## Testing

### Unit Testing with Vitest

**vitest.config.js:**
```javascript
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [sveltekit()],
  test: {
    include: ['src/**/*.{test,spec}.{js,ts}'],
    environment: 'jsdom'
  }
});
```

**src/lib/utils/format.test.js:**
```javascript
import { describe, it, expect } from 'vitest';
import { formatCurrency, formatDate } from './format';

describe('formatCurrency', () => {
  it('formats USD correctly', () => {
    expect(formatCurrency(1234.56, 'USD')).toBe('$1,234.56');
  });

  it('handles zero', () => {
    expect(formatCurrency(0, 'USD')).toBe('$0.00');
  });
});

describe('formatDate', () => {
  it('formats date correctly', () => {
    const date = new Date('2024-01-15');
    expect(formatDate(date)).toBe('Jan 15, 2024');
  });
});
```

**Testing Svelte components:**
```javascript
import { render, screen, fireEvent } from '@testing-library/svelte';
import { describe, it, expect } from 'vitest';
import Button from './Button.svelte';

describe('Button', () => {
  it('renders with text', () => {
    render(Button, { props: { text: 'Click me' } });
    expect(screen.getByText('Click me')).toBeTruthy();
  });

  it('calls onClick when clicked', async () => {
    let clicked = false;
    render(Button, {
      props: {
        text: 'Click',
        onClick: () => { clicked = true; }
      }
    });

    const button = screen.getByText('Click');
    await fireEvent.click(button);

    expect(clicked).toBe(true);
  });
});
```

### E2E Testing with Playwright

**tests/home.spec.js:**
```javascript
import { expect, test } from '@playwright/test';

test('home page loads', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('h1')).toContainText('Welcome');
});

test('navigation works', async ({ page }) => {
  await page.goto('/');
  await page.click('a[href="/about"]');
  await expect(page).toHaveURL('/about');
  await expect(page.locator('h1')).toContainText('About');
});

test('form submission', async ({ page }) => {
  await page.goto('/contact');

  await page.fill('input[name="name"]', 'John Doe');
  await page.fill('input[name="email"]', 'john@example.com');
  await page.fill('textarea[name="message"]', 'Hello!');

  await page.click('button[type="submit"]');

  await expect(page.locator('.success')).toContainText('Message sent');
});
```

---

## Best Practices

### 1. Use Server Load Functions for Sensitive Data

```javascript
// ✅ Good - server-only
// src/routes/dashboard/+page.server.js
export async function load({ locals }) {
  const user = await db.getUser(locals.userId);
  return { user };
}

// ❌ Bad - exposes API keys
// src/routes/dashboard/+page.js
export async function load() {
  const data = await fetch('https://api.example.com', {
    headers: { 'API-Key': process.env.API_KEY } // Exposed to client!
  });
  return { data };
}
```

### 2. Leverage Progressive Enhancement

```svelte
<!-- Form works without JavaScript -->
<form method="POST" action="?/create" use:enhance>
  <input name="title" required />
  <button type="submit">Create</button>
</form>
```

### 3. Optimize Load Functions

```javascript
// ✅ Good - parallel loading
export async function load({ fetch }) {
  const [posts, categories, tags] = await Promise.all([
    fetch('/api/posts').then(r => r.json()),
    fetch('/api/categories').then(r => r.json()),
    fetch('/api/tags').then(r => r.json())
  ]);

  return { posts, categories, tags };
}

// ❌ Bad - sequential loading
export async function load({ fetch }) {
  const posts = await fetch('/api/posts').then(r => r.json());
  const categories = await fetch('/api/categories').then(r => r.json());
  const tags = await fetch('/api/tags').then(r => r.json());

  return { posts, categories, tags };
}
```

### 4. Use Layouts Effectively

```
routes/
├── +layout.svelte              # Root layout (navbar, footer)
├── (app)/
│   ├── +layout.svelte         # App layout (sidebar)
│   ├── dashboard/
│   │   └── +page.svelte
│   └── settings/
│       └── +page.svelte
└── (marketing)/
    ├── +layout.svelte         # Marketing layout (different header)
    ├── about/
    │   └── +page.svelte
    └── pricing/
        └── +page.svelte
```

### 5. Handle Errors Gracefully

```javascript
import { error } from '@sveltejs/kit';

export async function load({ params }) {
  try {
    const data = await fetchData(params.id);

    if (!data) {
      throw error(404, {
        message: 'Not found',
        hint: 'Check the URL and try again'
      });
    }

    return { data };
  } catch (err) {
    if (err.status) throw err;

    console.error('Unexpected error:', err);
    throw error(500, 'Something went wrong');
  }
}
```

### 6. Validate User Input

```javascript
import { fail } from '@sveltejs/kit';
import { z } from 'zod';

const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  age: z.number().min(18)
});

export const actions = {
  default: async ({ request }) => {
    const data = await request.formData();
    const values = {
      email: data.get('email'),
      password: data.get('password'),
      age: parseInt(data.get('age'))
    };

    const result = schema.safeParse(values);

    if (!result.success) {
      return fail(400, {
        errors: result.error.flatten().fieldErrors,
        values
      });
    }

    // Process valid data
    await createUser(result.data);

    return { success: true };
  }
};
```

### 7. Secure Cookies

```javascript
cookies.set('session', sessionId, {
  path: '/',
  httpOnly: true,        // Prevent JS access
  sameSite: 'strict',    // CSRF protection
  secure: true,          // HTTPS only
  maxAge: 60 * 60 * 24 * 7  // 7 days
});
```

### 8. Use Type Safety

```typescript
// src/routes/blog/[slug]/+page.ts
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
  const post = await fetch(`/api/posts/${params.slug}`).then(r => r.json());

  return {
    post
  };
};
```

---

## Resources

**Official Documentation:**
- [SvelteKit Documentation](https://kit.svelte.dev/docs)
- [Svelte Documentation](https://svelte.dev/docs)
- [SvelteKit Tutorial](https://learn.svelte.dev/)

**Deployment Platforms:**
- [Vercel](https://vercel.com/)
- [Netlify](https://www.netlify.com/)
- [Cloudflare Pages](https://pages.cloudflare.com/)

**Useful Libraries:**
- [Prisma ORM](https://www.prisma.io/)
- [Drizzle ORM](https://orm.drizzle.team/)
- [Lucia Auth](https://lucia-auth.com/)
- [Zod Validation](https://zod.dev/)
- [Paraglide i18n](https://inlang.com/m/gerre34r/library-inlang-paraglideJs)

**Community:**
- [SvelteKit Discord](https://svelte.dev/chat)
- [Svelte Society](https://sveltesociety.dev/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/sveltekit)
- [GitHub Discussions](https://github.com/sveltejs/kit/discussions)

**Learning Resources:**
- [Joy of Code](https://joyofcode.xyz/)
- [Svelte Mastery](https://sveltemastery.com/)
- [SvelteKit Examples](https://github.com/sveltejs/kit/tree/master/examples)
