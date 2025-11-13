# Next.js

Next.js is a production-ready React framework that provides server-side rendering, static site generation, API routes, and many other features out of the box. Built by Vercel, it's designed to give you the best developer experience with all the features needed for production.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [File-Based Routing](#file-based-routing)
- [Pages and Layouts](#pages-and-layouts)
- [Data Fetching](#data-fetching)
- [API Routes](#api-routes)
- [Dynamic Routes](#dynamic-routes)
- [Image Optimization](#image-optimization)
- [CSS and Styling](#css-and-styling)
- [Authentication](#authentication)
- [Deployment](#deployment)
- [Best Practices](#best-practices)

---

## Introduction

**Key Features:**
- Server-Side Rendering (SSR)
- Static Site Generation (SSG)
- Incremental Static Regeneration (ISR)
- API Routes
- File-based routing
- Automatic code splitting
- Built-in image optimization
- TypeScript support
- Fast Refresh
- Zero configuration

**Use Cases:**
- E-commerce websites
- Marketing websites
- Blogs and content sites
- Dashboards
- SaaS applications
- Mobile applications (with React Native)

---

## Installation and Setup

### Create New Project

```bash
# Create Next.js app
npx create-next-app@latest my-next-app
cd my-next-app

# Or with TypeScript
npx create-next-app@latest my-next-app --typescript

# Start development server
npm run dev
```

### Project Structure

```
my-next-app/
├── app/                    # App directory (Next.js 13+)
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   ├── api/               # API routes
│   └── [folder]/          # Routes
├── public/                # Static files
├── components/            # React components
├── lib/                   # Utility functions
├── styles/               # CSS files
├── next.config.js        # Next.js configuration
├── package.json
└── tsconfig.json         # TypeScript configuration
```

### Configuration

**next.config.js:**
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['example.com', 'cdn.example.com'],
  },
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://api.example.com/:path*',
      },
    ]
  },
}

module.exports = nextConfig
```

---

## File-Based Routing

### App Router (Next.js 13+)

```
app/
├── page.tsx              # / route
├── about/
│   └── page.tsx         # /about route
├── blog/
│   ├── page.tsx         # /blog route
│   └── [slug]/
│       └── page.tsx     # /blog/[slug] route
└── dashboard/
    ├── layout.tsx       # Dashboard layout
    ├── page.tsx         # /dashboard route
    └── settings/
        └── page.tsx     # /dashboard/settings route
```

**app/page.tsx:**
```typescript
import Link from 'next/link'

export default function Home() {
  return (
    <main>
      <h1>Welcome to Next.js</h1>
      <Link href="/about">About</Link>
      <Link href="/blog">Blog</Link>
    </main>
  )
}
```

**app/about/page.tsx:**
```typescript
export default function About() {
  return (
    <div>
      <h1>About Us</h1>
      <p>This is the about page</p>
    </div>
  )
}
```

---

## Pages and Layouts

### Root Layout

**app/layout.tsx:**
```typescript
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'My Next.js App',
  description: 'Built with Next.js',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav>
          <a href="/">Home</a>
          <a href="/about">About</a>
          <a href="/blog">Blog</a>
        </nav>
        {children}
        <footer>© 2024 My App</footer>
      </body>
    </html>
  )
}
```

### Nested Layouts

**app/dashboard/layout.tsx:**
```typescript
export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="dashboard">
      <aside>
        <nav>
          <a href="/dashboard">Overview</a>
          <a href="/dashboard/settings">Settings</a>
          <a href="/dashboard/profile">Profile</a>
        </nav>
      </aside>
      <main>{children}</main>
    </div>
  )
}
```

### Loading and Error States

**app/loading.tsx:**
```typescript
export default function Loading() {
  return <div>Loading...</div>
}
```

**app/error.tsx:**
```typescript
'use client'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div>
      <h2>Something went wrong!</h2>
      <p>{error.message}</p>
      <button onClick={reset}>Try again</button>
    </div>
  )
}
```

---

## Data Fetching

### Server Components (Default)

```typescript
async function getData() {
  const res = await fetch('https://api.example.com/data', {
    cache: 'no-store', // or 'force-cache'
  })

  if (!res.ok) {
    throw new Error('Failed to fetch data')
  }

  return res.json()
}

export default async function Page() {
  const data = await getData()

  return (
    <div>
      <h1>Data from API</h1>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  )
}
```

### Static Generation

```typescript
async function getStaticData() {
  const res = await fetch('https://api.example.com/posts')
  return res.json()
}

export default async function BlogPage() {
  const posts = await getStaticData()

  return (
    <div>
      {posts.map((post: any) => (
        <article key={post.id}>
          <h2>{post.title}</h2>
          <p>{post.excerpt}</p>
        </article>
      ))}
    </div>
  )
}

// Revalidate every hour
export const revalidate = 3600
```

### Dynamic Data with Params

```typescript
async function getPost(slug: string) {
  const res = await fetch(`https://api.example.com/posts/${slug}`)
  return res.json()
}

export default async function Post({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug)

  return (
    <article>
      <h1>{post.title}</h1>
      <div dangerouslySetInnerHTML={{ __html: post.content }} />
    </article>
  )
}

// Generate static params for dynamic routes
export async function generateStaticParams() {
  const posts = await fetch('https://api.example.com/posts').then((res) =>
    res.json()
  )

  return posts.map((post: any) => ({
    slug: post.slug,
  }))
}
```

### Client Components

```typescript
'use client'

import { useState, useEffect } from 'react'

export default function ClientComponent() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/data')
      .then((res) => res.json())
      .then((data) => {
        setData(data)
        setLoading(false)
      })
  }, [])

  if (loading) return <div>Loading...</div>

  return <div>{JSON.stringify(data)}</div>
}
```

---

## API Routes

### Basic API Route

**app/api/hello/route.ts:**
```typescript
import { NextResponse } from 'next/server'

export async function GET() {
  return NextResponse.json({ message: 'Hello from Next.js!' })
}

export async function POST(request: Request) {
  const body = await request.json()
  return NextResponse.json({ received: body })
}
```

### Dynamic API Routes

**app/api/users/[id]/route.ts:**
```typescript
import { NextResponse } from 'next/server'

export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  const id = params.id

  // Fetch user from database
  const user = await fetchUser(id)

  if (!user) {
    return NextResponse.json({ error: 'User not found' }, { status: 404 })
  }

  return NextResponse.json(user)
}

export async function PUT(
  request: Request,
  { params }: { params: { id: string } }
) {
  const id = params.id
  const body = await request.json()

  // Update user in database
  const updatedUser = await updateUser(id, body)

  return NextResponse.json(updatedUser)
}

export async function DELETE(
  request: Request,
  { params }: { params: { id: string } }
) {
  const id = params.id

  await deleteUser(id)

  return NextResponse.json({ message: 'User deleted' })
}
```

### API with Database

```typescript
import { NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'

export async function GET() {
  try {
    const users = await prisma.user.findMany()
    return NextResponse.json(users)
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch users' },
      { status: 500 }
    )
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const user = await prisma.user.create({
      data: body,
    })
    return NextResponse.json(user, { status: 201 })
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to create user' },
      { status: 500 }
    )
  }
}
```

---

## Dynamic Routes

### Catch-All Routes

**app/shop/[...slug]/page.tsx:**
```typescript
export default function ShopPage({ params }: { params: { slug: string[] } }) {
  return (
    <div>
      <h1>Shop</h1>
      <p>Category: {params.slug.join('/')}</p>
    </div>
  )
}

// Matches:
// /shop/electronics
// /shop/electronics/laptops
// /shop/electronics/laptops/gaming
```

### Optional Catch-All Routes

**app/docs/[[...slug]]/page.tsx:**
```typescript
export default function DocsPage({
  params,
}: {
  params: { slug?: string[] }
}) {
  if (!params.slug) {
    return <div>Documentation Home</div>
  }

  return <div>Path: {params.slug.join('/')}</div>
}

// Matches:
// /docs
// /docs/getting-started
// /docs/api/reference
```

---

## Image Optimization

```typescript
import Image from 'next/image'

export default function ImageExample() {
  return (
    <div>
      {/* Static Image */}
      <Image
        src="/hero.jpg"
        alt="Hero"
        width={1200}
        height={600}
        priority
      />

      {/* External Image */}
      <Image
        src="https://example.com/image.jpg"
        alt="External"
        width={800}
        height={600}
        quality={85}
      />

      {/* Responsive Image */}
      <Image
        src="/profile.jpg"
        alt="Profile"
        fill
        sizes="(max-width: 768px) 100vw, 50vw"
        style={{ objectFit: 'cover' }}
      />

      {/* With Placeholder */}
      <Image
        src="/photo.jpg"
        alt="Photo"
        width={600}
        height={400}
        placeholder="blur"
        blurDataURL="data:image/jpeg;base64,..."
      />
    </div>
  )
}
```

---

## CSS and Styling

### CSS Modules

**components/Button.module.css:**
```css
.button {
  padding: 12px 24px;
  background: blue;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.button:hover {
  background: darkblue;
}
```

**components/Button.tsx:**
```typescript
import styles from './Button.module.css'

export default function Button({ children }: { children: React.ReactNode }) {
  return <button className={styles.button}>{children}</button>
}
```

### Tailwind CSS

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**tailwind.config.js:**
```javascript
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**app/globals.css:**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Usage:**
```typescript
export default function Home() {
  return (
    <div className="min-h-screen bg-gray-100">
      <h1 className="text-4xl font-bold text-blue-600">
        Hello Tailwind!
      </h1>
      <button className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
        Click Me
      </button>
    </div>
  )
}
```

---

## Authentication

### NextAuth.js

```bash
npm install next-auth
```

**app/api/auth/[...nextauth]/route.ts:**
```typescript
import NextAuth from 'next-auth'
import GoogleProvider from 'next-auth/providers/google'
import CredentialsProvider from 'next-auth/providers/credentials'

const handler = NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        // Verify credentials
        const user = await verifyUser(credentials)

        if (user) {
          return user
        }
        return null
      }
    })
  ],
  pages: {
    signIn: '/auth/signin',
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id
      }
      return token
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string
      }
      return session
    },
  },
})

export { handler as GET, handler as POST }
```

**app/providers.tsx:**
```typescript
'use client'

import { SessionProvider } from 'next-auth/react'

export function Providers({ children }: { children: React.ReactNode }) {
  return <SessionProvider>{children}</SessionProvider>
}
```

**Protected Route:**
```typescript
import { getServerSession } from 'next-auth'
import { redirect } from 'next/navigation'

export default async function DashboardPage() {
  const session = await getServerSession()

  if (!session) {
    redirect('/auth/signin')
  }

  return (
    <div>
      <h1>Dashboard</h1>
      <p>Welcome, {session.user?.name}</p>
    </div>
  )
}
```

**Client-Side Auth:**
```typescript
'use client'

import { useSession, signIn, signOut } from 'next-auth/react'

export default function LoginButton() {
  const { data: session, status } = useSession()

  if (status === 'loading') {
    return <div>Loading...</div>
  }

  if (session) {
    return (
      <>
        <p>Signed in as {session.user?.email}</p>
        <button onClick={() => signOut()}>Sign out</button>
      </>
    )
  }

  return <button onClick={() => signIn()}>Sign in</button>
}
```

---

## Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Production deployment
vercel --prod
```

### Docker

**Dockerfile:**
```dockerfile
FROM node:18-alpine AS base

FROM base AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM base AS runner
WORKDIR /app
ENV NODE_ENV production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
```

**next.config.js:**
```javascript
module.exports = {
  output: 'standalone',
}
```

### Environment Variables

**.env.local:**
```bash
DATABASE_URL="postgresql://..."
NEXTAUTH_SECRET="your-secret"
NEXTAUTH_URL="http://localhost:3000"
GOOGLE_CLIENT_ID="..."
GOOGLE_CLIENT_SECRET="..."
```

---

## Best Practices

### 1. Server vs Client Components

```typescript
// Server Component (default) - Use for:
// - Data fetching
// - Direct database access
// - API calls
export default async function ServerComponent() {
  const data = await fetchData()
  return <div>{data}</div>
}

// Client Component - Use for:
// - Interactivity (onClick, onChange, etc.)
// - State management
// - Browser APIs
'use client'
export default function ClientComponent() {
  const [count, setCount] = useState(0)
  return <button onClick={() => setCount(count + 1)}>{count}</button>
}
```

### 2. Data Fetching Strategies

```typescript
// Static - Fetch at build time
export const revalidate = false

// ISR - Revalidate every 60 seconds
export const revalidate = 60

// Dynamic - Fetch on every request
export const dynamic = 'force-dynamic'

// Cache specific requests
fetch('https://api.example.com/data', {
  next: { revalidate: 3600 } // Revalidate every hour
})
```

### 3. Metadata

```typescript
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'My Page',
  description: 'Page description',
  openGraph: {
    title: 'My Page',
    description: 'Page description',
    images: ['/og-image.jpg'],
  },
  twitter: {
    card: 'summary_large_image',
  },
}
```

### 4. Error Boundaries

```typescript
// app/error.tsx
'use client'

export default function Error({
  error,
  reset,
}: {
  error: Error
  reset: () => void
}) {
  useEffect(() => {
    console.error(error)
  }, [error])

  return (
    <div>
      <h2>Something went wrong!</h2>
      <button onClick={reset}>Try again</button>
    </div>
  )
}
```

### 5. Performance Optimization

```typescript
// Dynamic imports
import dynamic from 'next/dynamic'

const DynamicComponent = dynamic(() => import('@/components/Heavy'), {
  loading: () => <p>Loading...</p>,
  ssr: false, // Disable SSR for this component
})

// Lazy load images
<Image
  src="/photo.jpg"
  alt="Photo"
  loading="lazy"
  width={600}
  height={400}
/>
```

---

## Resources

**Official Documentation:**
- [Next.js Documentation](https://nextjs.org/docs)
- [Next.js Learn Course](https://nextjs.org/learn)
- [Next.js Examples](https://github.com/vercel/next.js/tree/canary/examples)

**Tools and Ecosystem:**
- [Next Auth](https://next-auth.js.org/)
- [Prisma](https://www.prisma.io/)
- [TailwindCSS](https://tailwindcss.com/)
- [Vercel](https://vercel.com/)

**Community:**
- [Next.js GitHub](https://github.com/vercel/next.js)
- [Next.js Discord](https://nextjs.org/discord)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/next.js)

**Learning Resources:**
- [Next.js Conf](https://nextjs.org/conf)
- [Vercel Guide](https://vercel.com/guides)
- [Awesome Next.js](https://github.com/unicodeveloper/awesome-nextjs)
