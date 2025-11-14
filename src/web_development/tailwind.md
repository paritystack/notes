# Tailwind CSS

**Tailwind CSS** is a utility-first CSS framework for rapidly building custom user interfaces. Unlike traditional CSS frameworks that provide pre-designed components (like Bootstrap), Tailwind provides low-level utility classes that let you build completely custom designs without ever leaving your HTML.

**Key Philosophy**: Instead of fighting framework conventions, Tailwind gives you the building blocks to create your own design system with utility classes that can be composed to build any design directly in your markup.

## Table of Contents

- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Core Concepts](#core-concepts)
- [Utility Classes](#utility-classes)
  - [Layout](#layout)
  - [Spacing](#spacing)
  - [Sizing](#sizing)
  - [Typography](#typography)
  - [Colors](#colors)
  - [Borders](#borders)
  - [Effects and Filters](#effects-and-filters)
  - [Transitions and Animations](#transitions-and-animations)
  - [Transforms](#transforms)
- [Responsive Design](#responsive-design)
- [State Variants](#state-variants)
- [Dark Mode](#dark-mode)
- [Component Patterns](#component-patterns)
- [Layout Patterns](#layout-patterns)
- [Customization](#customization)
- [Plugin System](#plugin-system)
- [Framework Integration](#framework-integration)
- [Advanced Topics](#advanced-topics)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [Accessibility](#accessibility)
- [Migration and Comparison](#migration-and-comparison)
- [Tooling and Ecosystem](#tooling-and-ecosystem)
- [Resources](#resources)

---

## Introduction

### What is Tailwind CSS?

Tailwind CSS is a **utility-first CSS framework** that provides single-purpose utility classes for building user interfaces. Instead of writing custom CSS, you compose these utilities directly in your HTML.

**Traditional CSS approach:**
```html
<div class="chat-notification">
  <div class="chat-notification-logo-wrapper">
    <img class="chat-notification-logo" src="logo.svg" alt="Logo">
  </div>
  <div class="chat-notification-content">
    <h4 class="chat-notification-title">New message</h4>
    <p class="chat-notification-message">You have a new message!</p>
  </div>
</div>
```

**Tailwind approach:**
```html
<div class="flex items-center p-6 max-w-sm mx-auto bg-white rounded-xl shadow-lg">
  <div class="shrink-0">
    <img class="h-12 w-12" src="logo.svg" alt="Logo">
  </div>
  <div class="ml-4">
    <h4 class="text-xl font-medium text-black">New message</h4>
    <p class="text-gray-500">You have a new message!</p>
  </div>
</div>
```

### Key Features

1. **Utility-First**: Compose designs from utility classes instead of writing custom CSS
2. **Responsive**: Mobile-first breakpoints built into every utility
3. **Component-Friendly**: Easy to extract components when needed
4. **Customizable**: Extensive theming and configuration options
5. **Modern**: Supports CSS Grid, Flexbox, transforms, transitions, and more
6. **Dark Mode**: First-class dark mode support
7. **JIT Mode**: Generate styles on-demand for faster builds
8. **Production-Optimized**: Automatically removes unused CSS

### Use Cases

**Perfect for:**
- Web applications and dashboards
- Marketing websites and landing pages
- Rapid prototyping
- Design systems and component libraries
- Projects requiring custom designs

**Maybe not ideal for:**
- Simple static sites (might be overkill)
- Teams resistant to utility-first approach
- Projects with very limited HTML access

### Tailwind vs Traditional CSS

| Aspect | Tailwind | Traditional CSS |
|--------|----------|----------------|
| **Approach** | Utility-first | Semantic class names |
| **Workflow** | Compose in HTML | Write CSS separately |
| **File Switching** | Minimal | Constant (HTML ↔ CSS) |
| **Naming** | No naming needed | Need to invent class names |
| **Bundle Size** | Small (purged) | Grows over time |
| **Customization** | Config-based | Manual CSS |
| **Learning Curve** | Learn utilities | Learn CSS deeply |

### Tailwind vs Bootstrap

| Feature | Tailwind | Bootstrap |
|---------|----------|-----------|
| **Philosophy** | Utility-first | Component-first |
| **Customization** | Highly flexible | Limited to overrides |
| **Design** | Build your own | Pre-designed look |
| **File Size** | Smaller (purged) | Larger base |
| **Components** | Build from utilities | Ready-made |
| **Learning** | Utility classes | Component classes |

---

## Installation and Setup

### NPM/Yarn Installation

```bash
# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer

# Initialize configuration
npx tailwindcss init
```

### Complete Setup

**1. Create config files:**
```bash
# Create both tailwind.config.js and postcss.config.js
npx tailwindcss init -p
```

**2. Configure template paths** (tailwind.config.js):
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**3. Add Tailwind directives to CSS** (src/index.css):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**4. Import CSS in your app:**
```javascript
// main.js or App.jsx
import './index.css'
```

### Framework-Specific Setup

#### React / Next.js

```bash
# Next.js (automatic with create-next-app)
npx create-next-app@latest my-project --tailwind

# Manual setup for existing React project
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Next.js config:**
```javascript
// tailwind.config.js
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

#### Vue / Nuxt

```bash
# Nuxt 3
npm install -D @nuxtjs/tailwindcss
```

**nuxt.config.ts:**
```typescript
export default defineNuxtConfig({
  modules: ['@nuxtjs/tailwindcss']
})
```

#### Svelte / SvelteKit

```bash
npx svelte-add@latest tailwindcss
npm install
```

#### Vite

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**vite.config.js:**
```javascript
import { defineConfig } from 'vite'
export default defineConfig({
  css: {
    postcss: './postcss.config.js',
  },
})
```

### CDN (Development Only)

```html
<!DOCTYPE html>
<html>
<head>
  <!-- Include via CDN (no build step, but no customization) -->
  <script src="https://cdn.tailwindcss.com"></script>

  <!-- Optional: Configure via script tag -->
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            brand: '#3b82f6',
          }
        }
      }
    }
  </script>
</head>
<body>
  <h1 class="text-3xl font-bold text-brand">
    Hello Tailwind!
  </h1>
</body>
</html>
```

**⚠️ CDN Warning**: Don't use in production. No purging, no optimization, large file size.

### Tailwind CLI

For projects without a build tool:

```bash
# Install
npm install -D tailwindcss

# Initialize
npx tailwindcss init

# Build CSS
npx tailwindcss -i ./src/input.css -o ./dist/output.css --watch

# Production build
npx tailwindcss -i ./src/input.css -o ./dist/output.css --minify
```

---

## Configuration

### Basic tailwind.config.js

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  // Files to scan for class names
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx,vue,svelte}",
  ],

  // Dark mode configuration
  darkMode: 'class', // or 'media'

  // Theme customization
  theme: {
    // Replace default theme
    screens: {
      sm: '640px',
      md: '768px',
      lg: '1024px',
      xl: '1280px',
      '2xl': '1536px',
    },

    // Extend default theme (recommended)
    extend: {
      colors: {
        brand: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
      },
      spacing: {
        '128': '32rem',
        '144': '36rem',
      },
      borderRadius: {
        '4xl': '2rem',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Lexend', 'sans-serif'],
      },
    },
  },

  // Plugins
  plugins: [],
}
```

### Content Configuration

**Tell Tailwind where to look for classes:**

```javascript
module.exports = {
  content: [
    // HTML files
    './public/**/*.html',

    // JavaScript/TypeScript
    './src/**/*.{js,jsx,ts,tsx}',

    // Vue components
    './src/**/*.vue',

    // Svelte components
    './src/**/*.svelte',

    // PHP files (for WordPress, Laravel, etc.)
    './templates/**/*.php',

    // Use safelist for dynamic classes
  ],

  // Safelist classes that might be generated dynamically
  safelist: [
    'bg-red-500',
    'bg-green-500',
    'bg-blue-500',
    // Or use patterns
    {
      pattern: /bg-(red|green|blue)-(100|500|900)/,
    },
  ],
}
```

### Theme Extension

```javascript
module.exports = {
  theme: {
    // Extend default theme (adds to existing)
    extend: {
      // Custom colors
      colors: {
        primary: '#3b82f6',
        secondary: '#8b5cf6',
        danger: '#ef4444',
      },

      // Custom spacing values
      spacing: {
        '128': '32rem',
        '144': '36rem',
      },

      // Custom font sizes
      fontSize: {
        'xxs': '0.625rem',
      },

      // Custom breakpoints
      screens: {
        '3xl': '1920px',
      },

      // Custom z-index values
      zIndex: {
        '100': '100',
      },

      // Custom animations
      animation: {
        'spin-slow': 'spin 3s linear infinite',
      },

      // Custom keyframes
      keyframes: {
        wiggle: {
          '0%, 100%': { transform: 'rotate(-3deg)' },
          '50%': { transform: 'rotate(3deg)' },
        }
      }
    },

    // Replace default theme (use sparingly)
    // screens: { ... } // This replaces all default breakpoints
  },
}
```

### Using CSS Variables

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: 'var(--color-primary)',
        secondary: 'var(--color-secondary)',
      },
    },
  },
}
```

```css
/* In your CSS */
:root {
  --color-primary: 59 130 246; /* RGB values */
  --color-secondary: 139 92 246;
}

.dark {
  --color-primary: 96 165 250;
  --color-secondary: 167 139 250;
}
```

```html
<!-- Use with opacity modifiers -->
<div class="bg-primary/50">Semi-transparent background</div>
```

---

## Core Concepts

### Utility-First Fundamentals

**Instead of semantic class names, use utilities:**

```html
<!-- ❌ Traditional approach -->
<div class="card">
  <h2 class="card-title">Title</h2>
  <p class="card-body">Content</p>
</div>

<!-- ✅ Tailwind approach -->
<div class="bg-white rounded-lg shadow-md p-6">
  <h2 class="text-xl font-bold mb-2">Title</h2>
  <p class="text-gray-700">Content</p>
</div>
```

**Benefits:**
- No need to invent class names
- Changes are local (no cascade issues)
- CSS bundle size stays small
- Faster development

### Responsive Design (Mobile-First)

All utilities can be prefixed with breakpoint names:

```html
<!-- Mobile: full width, Desktop: half width -->
<div class="w-full md:w-1/2">
  Responsive element
</div>

<!-- Mobile: column, Tablet+: row -->
<div class="flex flex-col md:flex-row">
  <div>Item 1</div>
  <div>Item 2</div>
</div>
```

**Breakpoints:**
- `sm`: 640px
- `md`: 768px
- `lg`: 1024px
- `xl`: 1280px
- `2xl`: 1536px

### Hover, Focus, and Other States

```html
<!-- Hover state -->
<button class="bg-blue-500 hover:bg-blue-700">
  Hover me
</button>

<!-- Focus state -->
<input class="border focus:border-blue-500 focus:ring-2 focus:ring-blue-200">

<!-- Multiple states -->
<button class="bg-blue-500 hover:bg-blue-600 active:bg-blue-700 disabled:bg-gray-300">
  Button
</button>
```

### Design Tokens and Constraints

Tailwind provides a constrained set of values (design tokens) for consistency:

```html
<!-- Spacing scale: 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64... -->
<div class="p-4">  <!-- padding: 1rem -->
<div class="p-8">  <!-- padding: 2rem -->
<div class="p-16"> <!-- padding: 4rem -->

<!-- Color scale: 50, 100, 200, 300, 400, 500, 600, 700, 800, 900 -->
<div class="bg-blue-100">  <!-- Light blue -->
<div class="bg-blue-500">  <!-- Medium blue -->
<div class="bg-blue-900">  <!-- Dark blue -->
```

**Use arbitrary values when needed:**
```html
<!-- Arbitrary values with [value] syntax -->
<div class="w-[347px]">Exact width</div>
<div class="bg-[#1da1f2]">Twitter blue</div>
<div class="text-[2.35rem]">Custom font size</div>
```

---

## Utility Classes

### Layout

#### Container

```html
<!-- Centered container with max-width -->
<div class="container mx-auto px-4">
  Content
</div>

<!-- Responsive max-widths by default:
     sm: 640px
     md: 768px
     lg: 1024px
     xl: 1280px
     2xl: 1536px
-->
```

#### Display

```html
<!-- Block, inline, inline-block -->
<div class="block">Block</div>
<div class="inline">Inline</div>
<div class="inline-block">Inline-block</div>

<!-- Flex and Grid -->
<div class="flex">Flexbox container</div>
<div class="inline-flex">Inline flex container</div>
<div class="grid">Grid container</div>
<div class="inline-grid">Inline grid container</div>

<!-- Hidden -->
<div class="hidden">Not displayed</div>
<div class="md:block">Hidden on mobile, shown on tablet+</div>
```

#### Flexbox

```html
<!-- Flex direction -->
<div class="flex flex-row">Horizontal (default)</div>
<div class="flex flex-col">Vertical</div>
<div class="flex flex-row-reverse">Reversed horizontal</div>

<!-- Justify content (main axis) -->
<div class="flex justify-start">Start</div>
<div class="flex justify-center">Center</div>
<div class="flex justify-between">Space between</div>
<div class="flex justify-around">Space around</div>
<div class="flex justify-evenly">Space evenly</div>

<!-- Align items (cross axis) -->
<div class="flex items-start">Start</div>
<div class="flex items-center">Center</div>
<div class="flex items-end">End</div>
<div class="flex items-stretch">Stretch (default)</div>

<!-- Flex wrap -->
<div class="flex flex-wrap">Wrap</div>
<div class="flex flex-nowrap">No wrap (default)</div>

<!-- Flex grow/shrink -->
<div class="flex-1">Grow and shrink</div>
<div class="flex-auto">Auto sizing</div>
<div class="flex-none">Don't grow or shrink</div>
<div class="grow">Only grow</div>
<div class="shrink-0">Don't shrink</div>

<!-- Gap -->
<div class="flex gap-4">Gap between items</div>
<div class="flex gap-x-4 gap-y-2">Different x and y gaps</div>
```

#### Grid

```html
<!-- Grid columns -->
<div class="grid grid-cols-3 gap-4">
  <div>1</div>
  <div>2</div>
  <div>3</div>
</div>

<!-- Grid cols with different sizes -->
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  Responsive grid
</div>

<!-- Column span -->
<div class="grid grid-cols-3">
  <div class="col-span-2">Spans 2 columns</div>
  <div>1 column</div>
</div>

<!-- Auto-fit columns -->
<div class="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-4">
  Auto-sizing grid
</div>

<!-- Grid rows -->
<div class="grid grid-rows-3 gap-4 h-64">
  <div>Row 1</div>
  <div>Row 2</div>
  <div>Row 3</div>
</div>

<!-- Grid template areas (arbitrary value) -->
<div class="grid grid-rows-[auto_1fr_auto]">
  <header>Header</header>
  <main>Content</main>
  <footer>Footer</footer>
</div>
```

#### Position

```html
<!-- Position types -->
<div class="static">Default</div>
<div class="relative">Relative</div>
<div class="absolute">Absolute</div>
<div class="fixed">Fixed</div>
<div class="sticky">Sticky</div>

<!-- Positioning with inset -->
<div class="absolute top-0 left-0">Top-left</div>
<div class="absolute top-0 right-0">Top-right</div>
<div class="absolute bottom-0 left-0">Bottom-left</div>
<div class="absolute inset-0">All sides 0</div>
<div class="absolute inset-x-0">Left and right 0</div>
<div class="absolute inset-y-0">Top and bottom 0</div>

<!-- Sticky header -->
<header class="sticky top-0 bg-white z-10">
  Sticky navigation
</header>
```

#### Float and Clear

```html
<div class="float-left">Float left</div>
<div class="float-right">Float right</div>
<div class="clear-both">Clear floats</div>
```

### Spacing

#### Padding

```html
<!-- All sides -->
<div class="p-4">Padding 1rem (16px)</div>
<div class="p-0">No padding</div>
<div class="p-px">1px padding</div>

<!-- Horizontal/Vertical -->
<div class="px-4">Horizontal padding</div>
<div class="py-2">Vertical padding</div>

<!-- Individual sides -->
<div class="pt-4">Padding top</div>
<div class="pr-4">Padding right</div>
<div class="pb-4">Padding bottom</div>
<div class="pl-4">Padding left</div>

<!-- Spacing scale: 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96 */
```

#### Margin

```html
<!-- All sides -->
<div class="m-4">Margin 1rem</div>
<div class="m-auto">Auto margin (for centering)</div>
<div class="-m-4">Negative margin</div>

<!-- Horizontal/Vertical -->
<div class="mx-auto">Center horizontally</div>
<div class="my-4">Vertical margin</div>

<!-- Individual sides -->
<div class="mt-4">Margin top</div>
<div class="mr-4">Margin right</div>
<div class="mb-4">Margin bottom</div>
<div class="ml-4">Margin left</div>
```

#### Space Between

```html
<!-- Space between children (flex/grid) -->
<div class="flex space-x-4">
  <div>Item 1</div>
  <div>Item 2</div>
  <div>Item 3</div>
</div>

<div class="flex flex-col space-y-4">
  <div>Item 1</div>
  <div>Item 2</div>
</div>
```

### Sizing

#### Width

```html
<!-- Fixed widths -->
<div class="w-32">Width 8rem (128px)</div>
<div class="w-64">Width 16rem (256px)</div>

<!-- Fractional widths -->
<div class="w-1/2">50% width</div>
<div class="w-1/3">33.333% width</div>
<div class="w-2/3">66.666% width</div>
<div class="w-1/4">25% width</div>
<div class="w-3/4">75% width</div>

<!-- Full widths -->
<div class="w-full">100% width</div>
<div class="w-screen">100vw width</div>

<!-- Min/Max width -->
<div class="min-w-0">Min-width 0</div>
<div class="min-w-full">Min-width 100%</div>
<div class="max-w-sm">Max-width 24rem</div>
<div class="max-w-md">Max-width 28rem</div>
<div class="max-w-lg">Max-width 32rem</div>
<div class="max-w-xl">Max-width 36rem</div>
<div class="max-w-2xl">Max-width 42rem</div>
<div class="max-w-full">Max-width 100%</div>
<div class="max-w-prose">Max-width 65ch (for reading)</div>

<!-- Arbitrary values -->
<div class="w-[420px]">Exact 420px</div>
```

#### Height

```html
<!-- Fixed heights -->
<div class="h-32">Height 8rem</div>
<div class="h-64">Height 16rem</div>

<!-- Full heights -->
<div class="h-full">100% height</div>
<div class="h-screen">100vh height</div>

<!-- Min/Max height -->
<div class="min-h-screen">Min-height 100vh</div>
<div class="max-h-96">Max-height 24rem</div>
```

### Typography

#### Font Family

```html
<!-- Default font stacks -->
<p class="font-sans">Sans-serif font</p>
<p class="font-serif">Serif font</p>
<p class="font-mono">Monospace font</p>

<!-- Custom fonts (defined in config) -->
<p class="font-display">Display font</p>
```

#### Font Size

```html
<p class="text-xs">Extra small (0.75rem)</p>
<p class="text-sm">Small (0.875rem)</p>
<p class="text-base">Base (1rem)</p>
<p class="text-lg">Large (1.125rem)</p>
<p class="text-xl">Extra large (1.25rem)</p>
<p class="text-2xl">2x large (1.5rem)</p>
<p class="text-3xl">3x large (1.875rem)</p>
<p class="text-4xl">4x large (2.25rem)</p>
<p class="text-5xl">5x large (3rem)</p>
<p class="text-6xl">6x large (3.75rem)</p>
<p class="text-7xl">7x large (4.5rem)</p>
<p class="text-8xl">8x large (6rem)</p>
<p class="text-9xl">9x large (8rem)</p>
```

#### Font Weight

```html
<p class="font-thin">Thin (100)</p>
<p class="font-extralight">Extra light (200)</p>
<p class="font-light">Light (300)</p>
<p class="font-normal">Normal (400)</p>
<p class="font-medium">Medium (500)</p>
<p class="font-semibold">Semibold (600)</p>
<p class="font-bold">Bold (700)</p>
<p class="font-extrabold">Extra bold (800)</p>
<p class="font-black">Black (900)</p>
```

#### Text Alignment and Styling

```html
<!-- Alignment -->
<p class="text-left">Left aligned</p>
<p class="text-center">Center aligned</p>
<p class="text-right">Right aligned</p>
<p class="text-justify">Justified</p>

<!-- Decoration -->
<p class="underline">Underlined</p>
<p class="line-through">Strikethrough</p>
<p class="no-underline">No underline</p>

<!-- Transform -->
<p class="uppercase">UPPERCASE</p>
<p class="lowercase">lowercase</p>
<p class="capitalize">Capitalize Each Word</p>
<p class="normal-case">Normal case</p>

<!-- Style -->
<p class="italic">Italic</p>
<p class="not-italic">Not italic</p>
```

#### Line Height and Letter Spacing

```html
<!-- Line height -->
<p class="leading-none">Line height 1</p>
<p class="leading-tight">Line height 1.25</p>
<p class="leading-normal">Line height 1.5</p>
<p class="leading-loose">Line height 2</p>

<!-- Letter spacing -->
<p class="tracking-tighter">Very tight</p>
<p class="tracking-tight">Tight</p>
<p class="tracking-normal">Normal</p>
<p class="tracking-wide">Wide</p>
<p class="tracking-wider">Wider</p>
<p class="tracking-widest">Widest</p>
```

#### Text Overflow

```html
<!-- Truncate with ellipsis -->
<p class="truncate">
  This text will be truncated with ellipsis if it's too long
</p>

<!-- Overflow behavior -->
<p class="overflow-ellipsis">Ellipsis</p>
<p class="overflow-clip">Clip</p>

<!-- Whitespace -->
<p class="whitespace-normal">Normal</p>
<p class="whitespace-nowrap">No wrap</p>
<p class="whitespace-pre">Preserve whitespace</p>
<p class="whitespace-pre-wrap">Preserve and wrap</p>
```

### Colors

#### Background Colors

```html
<!-- Gray scale -->
<div class="bg-white">White</div>
<div class="bg-gray-50">Gray 50</div>
<div class="bg-gray-100">Gray 100</div>
<div class="bg-gray-500">Gray 500</div>
<div class="bg-gray-900">Gray 900</div>
<div class="bg-black">Black</div>

<!-- Color palette (50-950 for each color) -->
<div class="bg-red-500">Red</div>
<div class="bg-orange-500">Orange</div>
<div class="bg-amber-500">Amber</div>
<div class="bg-yellow-500">Yellow</div>
<div class="bg-lime-500">Lime</div>
<div class="bg-green-500">Green</div>
<div class="bg-emerald-500">Emerald</div>
<div class="bg-teal-500">Teal</div>
<div class="bg-cyan-500">Cyan</div>
<div class="bg-sky-500">Sky</div>
<div class="bg-blue-500">Blue</div>
<div class="bg-indigo-500">Indigo</div>
<div class="bg-violet-500">Violet</div>
<div class="bg-purple-500">Purple</div>
<div class="bg-fuchsia-500">Fuchsia</div>
<div class="bg-pink-500">Pink</div>
<div class="bg-rose-500">Rose</div>

<!-- With opacity -->
<div class="bg-blue-500/50">50% opacity</div>
<div class="bg-blue-500/75">75% opacity</div>
```

#### Text Colors

```html
<p class="text-gray-900">Dark gray text</p>
<p class="text-blue-600">Blue text</p>
<p class="text-red-500">Red text</p>

<!-- With opacity -->
<p class="text-gray-900/50">Semi-transparent text</p>
```

#### Border Colors

```html
<div class="border border-gray-300">Gray border</div>
<div class="border-2 border-blue-500">Blue border</div>
```

### Borders

```html
<!-- Border width -->
<div class="border">1px border</div>
<div class="border-0">No border</div>
<div class="border-2">2px border</div>
<div class="border-4">4px border</div>
<div class="border-8">8px border</div>

<!-- Individual sides -->
<div class="border-t">Top border</div>
<div class="border-r">Right border</div>
<div class="border-b">Bottom border</div>
<div class="border-l">Left border</div>

<!-- Border style -->
<div class="border border-solid">Solid</div>
<div class="border border-dashed">Dashed</div>
<div class="border border-dotted">Dotted</div>
<div class="border border-double">Double</div>

<!-- Border radius -->
<div class="rounded-none">No radius</div>
<div class="rounded-sm">Small radius</div>
<div class="rounded">Default radius (0.25rem)</div>
<div class="rounded-md">Medium radius</div>
<div class="rounded-lg">Large radius</div>
<div class="rounded-xl">Extra large radius</div>
<div class="rounded-2xl">2x large radius</div>
<div class="rounded-3xl">3x large radius</div>
<div class="rounded-full">Fully rounded (circle/pill)</div>

<!-- Individual corners -->
<div class="rounded-tl-lg">Top-left</div>
<div class="rounded-tr-lg">Top-right</div>
<div class="rounded-br-lg">Bottom-right</div>
<div class="rounded-bl-lg">Bottom-left</div>

<!-- Divide (borders between children) -->
<div class="divide-y divide-gray-200">
  <div class="py-2">Item 1</div>
  <div class="py-2">Item 2</div>
  <div class="py-2">Item 3</div>
</div>
```

### Effects and Filters

#### Box Shadow

```html
<div class="shadow-sm">Small shadow</div>
<div class="shadow">Default shadow</div>
<div class="shadow-md">Medium shadow</div>
<div class="shadow-lg">Large shadow</div>
<div class="shadow-xl">Extra large shadow</div>
<div class="shadow-2xl">2x large shadow</div>
<div class="shadow-inner">Inner shadow</div>
<div class="shadow-none">No shadow</div>

<!-- Colored shadows -->
<div class="shadow-lg shadow-blue-500/50">Blue shadow</div>
```

#### Opacity

```html
<div class="opacity-0">Invisible</div>
<div class="opacity-25">25% opacity</div>
<div class="opacity-50">50% opacity</div>
<div class="opacity-75">75% opacity</div>
<div class="opacity-100">Fully opaque</div>
```

#### Blur

```html
<div class="blur-none">No blur</div>
<div class="blur-sm">Small blur</div>
<div class="blur">Default blur</div>
<div class="blur-lg">Large blur</div>
<div class="blur-xl">Extra large blur</div>

<!-- Backdrop blur (for overlays) -->
<div class="backdrop-blur-sm">Backdrop blur</div>
```

#### Other Filters

```html
<!-- Brightness -->
<img class="brightness-50" src="image.jpg">
<img class="brightness-125" src="image.jpg">

<!-- Contrast -->
<img class="contrast-50" src="image.jpg">
<img class="contrast-150" src="image.jpg">

<!-- Grayscale -->
<img class="grayscale" src="image.jpg">

<!-- Sepia -->
<img class="sepia" src="image.jpg">
```

### Transitions and Animations

```html
<!-- Transition property -->
<button class="transition">All properties</button>
<button class="transition-colors">Colors only</button>
<button class="transition-opacity">Opacity only</button>
<button class="transition-transform">Transform only</button>

<!-- Duration -->
<button class="transition duration-150">150ms</button>
<button class="transition duration-300">300ms (default)</button>
<button class="transition duration-500">500ms</button>
<button class="transition duration-1000">1s</button>

<!-- Timing function -->
<button class="transition ease-linear">Linear</button>
<button class="transition ease-in">Ease in</button>
<button class="transition ease-out">Ease out</button>
<button class="transition ease-in-out">Ease in-out</button>

<!-- Complete transition example -->
<button class="bg-blue-500 hover:bg-blue-700 transition-colors duration-300">
  Smooth color transition
</button>

<!-- Animations -->
<div class="animate-spin">Spinning</div>
<div class="animate-ping">Pinging</div>
<div class="animate-pulse">Pulsing</div>
<div class="animate-bounce">Bouncing</div>
```

### Transforms

```html
<!-- Scale -->
<img class="scale-50 hover:scale-100">  <!-- 50% to 100% on hover -->
<img class="scale-100 hover:scale-110"> <!-- Zoom in on hover -->
<img class="scale-x-75">                <!-- Scale X only -->

<!-- Rotate -->
<img class="rotate-0 hover:rotate-45">   <!-- 0 to 45 degrees -->
<img class="rotate-90">                  <!-- 90 degrees -->
<img class="rotate-180">                 <!-- 180 degrees -->
<img class="-rotate-45">                 <!-- -45 degrees -->

<!-- Translate -->
<div class="translate-x-4">Move right 1rem</div>
<div class="translate-y-4">Move down 1rem</div>
<div class="-translate-x-1/2">Move left 50%</div>

<!-- Skew -->
<div class="skew-x-12">Skew X</div>
<div class="skew-y-6">Skew Y</div>

<!-- Transform origin -->
<div class="origin-center">Center origin (default)</div>
<div class="origin-top-left">Top-left origin</div>

<!-- Combined transforms with transition -->
<button class="transition-transform duration-300 hover:scale-110 hover:rotate-3">
  Hover for effect
</button>
```

---

## Responsive Design

Tailwind uses a **mobile-first** breakpoint system. Unprefixed utilities apply to all screen sizes, while prefixed utilities apply at the specified breakpoint and above.

### Breakpoint System

```javascript
// Default breakpoints
sm: '640px'   // Small devices (landscape phones)
md: '768px'   // Medium devices (tablets)
lg: '1024px'  // Large devices (desktops)
xl: '1280px'  // Extra large devices (large desktops)
2xl: '1536px' // 2x extra large devices
```

### Responsive Utilities

```html
<!-- Mobile: full width, Desktop: half width -->
<div class="w-full lg:w-1/2">
  Responsive width
</div>

<!-- Hide on mobile, show on desktop -->
<div class="hidden lg:block">
  Desktop only content
</div>

<!-- Responsive padding -->
<div class="p-4 md:p-6 lg:p-8">
  Increasing padding
</div>

<!-- Responsive grid -->
<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
  <div>Item 1</div>
  <div>Item 2</div>
  <div>Item 3</div>
  <div>Item 4</div>
</div>
```

### Responsive Layout Example

```html
<!-- Mobile: stacked, Desktop: side-by-side -->
<div class="flex flex-col lg:flex-row gap-4">
  <!-- Sidebar: full width mobile, 1/4 width desktop -->
  <aside class="w-full lg:w-1/4 bg-gray-100 p-4">
    Sidebar
  </aside>

  <!-- Main: full width mobile, 3/4 width desktop -->
  <main class="w-full lg:w-3/4 p-4">
    Main content
  </main>
</div>
```

### Responsive Typography

```html
<h1 class="text-2xl sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold">
  Responsive heading
</h1>

<p class="text-sm md:text-base lg:text-lg">
  Responsive paragraph
</p>
```

### Custom Breakpoints

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    screens: {
      'sm': '640px',
      'md': '768px',
      'lg': '1024px',
      'xl': '1280px',
      '2xl': '1536px',
      '3xl': '1920px', // Custom breakpoint
    },
  },
}
```

```html
<div class="hidden 3xl:block">
  Only on 1920px+ screens
</div>
```

### Container Queries (Plugin)

```bash
npm install @tailwindcss/container-queries
```

```javascript
// tailwind.config.js
module.exports = {
  plugins: [
    require('@tailwindcss/container-queries'),
  ],
}
```

```html
<div class="@container">
  <div class="@md:text-2xl @lg:text-4xl">
    Size based on container, not viewport
  </div>
</div>
```

---

## State Variants

Tailwind includes variants for styling elements based on their state.

### Hover, Focus, and Active

```html
<!-- Hover -->
<button class="bg-blue-500 hover:bg-blue-700">
  Hover me
</button>

<!-- Focus -->
<input class="border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200">

<!-- Active (being clicked) -->
<button class="bg-blue-500 active:bg-blue-800">
  Click me
</button>

<!-- Combined states -->
<button class="
  bg-blue-500
  hover:bg-blue-600
  focus:ring-2
  focus:ring-blue-300
  active:bg-blue-700
  transition-colors
">
  Full interaction states
</button>
```

### Focus Visible

```html
<!-- Only show focus ring for keyboard navigation -->
<button class="focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
  Keyboard accessible
</button>
```

### Form States

```html
<!-- Disabled -->
<button class="bg-blue-500 disabled:bg-gray-300 disabled:cursor-not-allowed" disabled>
  Disabled button
</button>

<!-- Required -->
<input class="border required:border-red-500" required>

<!-- Valid/Invalid -->
<input class="border invalid:border-red-500 valid:border-green-500" type="email">

<!-- Placeholder -->
<input class="placeholder:italic placeholder:text-gray-400" placeholder="Email address">
```

### Group Hover and Focus

Style child elements when hovering over parent:

```html
<div class="group hover:bg-blue-50 p-4 cursor-pointer">
  <h3 class="group-hover:text-blue-600">Heading</h3>
  <p class="group-hover:text-gray-700">
    Hover over the card to change colors
  </p>
  <button class="opacity-0 group-hover:opacity-100">
    Hidden button appears on card hover
  </button>
</div>
```

```html
<!-- Group with custom name -->
<div class="group/card hover:bg-blue-50">
  <div class="group/item">
    <p class="group-hover/card:text-blue-600">Card hover</p>
    <p class="group-hover/item:text-red-600">Item hover</p>
  </div>
</div>
```

### Peer Modifiers

Style an element based on sibling state:

```html
<label>
  <input type="checkbox" class="peer sr-only">
  <div class="
    w-11 h-6 bg-gray-200 rounded-full
    peer-checked:bg-blue-600
    peer-focus:ring-2 peer-focus:ring-blue-300
  ">
    <!-- Toggle switch styled by peer checkbox -->
  </div>
</label>
```

```html
<!-- Floating label -->
<div class="relative">
  <input
    id="email"
    class="peer w-full border-b-2 border-gray-300 focus:border-blue-500"
    placeholder=" "
  >
  <label
    for="email"
    class="
      absolute left-0 top-0
      text-gray-500
      peer-placeholder-shown:top-2
      peer-focus:top-0
      peer-focus:text-xs
      peer-focus:text-blue-500
      transition-all
    "
  >
    Email
  </label>
</div>
```

### Child Selectors

```html
<!-- First and last child -->
<ul>
  <li class="first:font-bold">First (bold)</li>
  <li>Middle</li>
  <li class="last:font-bold">Last (bold)</li>
</ul>

<!-- Odd and even -->
<table>
  <tr class="odd:bg-white even:bg-gray-50">
    <td>Row 1</td>
  </tr>
  <tr class="odd:bg-white even:bg-gray-50">
    <td>Row 2</td>
  </tr>
</table>
```

### Before and After Pseudo-elements

```html
<!-- Before -->
<div class="
  before:content-['→']
  before:mr-2
  before:text-blue-500
">
  Content with arrow before
</div>

<!-- After -->
<a class="
  after:content-['_↗']
  after:text-xs
  after:text-gray-400
">
  External link
</a>
```

---

## Dark Mode

Tailwind includes first-class dark mode support.

### Configuration

```javascript
// tailwind.config.js
module.exports = {
  // Choose strategy
  darkMode: 'class', // or 'media'
  // ...
}
```

**Two strategies:**
1. **'media'**: Uses `prefers-color-scheme` media query (system preference)
2. **'class'**: Requires `.dark` class on `<html>` or `<body>` (manual toggle)

### Using Dark Mode (Class Strategy)

```html
<!-- Light mode: white background, dark text -->
<!-- Dark mode: dark background, light text -->
<div class="bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
  Content adapts to dark mode
</div>
```

### Dark Mode Examples

```html
<!-- Card with dark mode -->
<div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
  <h2 class="text-gray-900 dark:text-white text-2xl font-bold">
    Heading
  </h2>
  <p class="text-gray-700 dark:text-gray-300">
    Description text
  </p>
  <button class="
    bg-blue-500 hover:bg-blue-600
    dark:bg-blue-600 dark:hover:bg-blue-700
    text-white
  ">
    Button
  </button>
</div>

<!-- Form input -->
<input class="
  bg-white dark:bg-gray-700
  border border-gray-300 dark:border-gray-600
  text-gray-900 dark:text-white
  focus:border-blue-500 dark:focus:border-blue-400
  focus:ring-2 focus:ring-blue-200 dark:focus:ring-blue-800
">

<!-- Image with different versions -->
<img
  class="block dark:hidden"
  src="logo-light.png"
  alt="Logo"
>
<img
  class="hidden dark:block"
  src="logo-dark.png"
  alt="Logo"
>
```

### Dark Mode Toggle Implementation

```html
<!-- HTML -->
<button id="theme-toggle" class="p-2 rounded-lg bg-gray-200 dark:bg-gray-700">
  <!-- Sun icon (show in dark mode) -->
  <svg class="hidden dark:block w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
    <path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"></path>
  </svg>
  <!-- Moon icon (show in light mode) -->
  <svg class="block dark:hidden w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
    <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"></path>
  </svg>
</button>

<script>
// JavaScript for toggle
const toggle = document.getElementById('theme-toggle');
const html = document.documentElement;

// Check localStorage or system preference
if (localStorage.theme === 'dark' ||
    (!('theme' in localStorage) &&
     window.matchMedia('(prefers-color-scheme: dark)').matches)) {
  html.classList.add('dark');
} else {
  html.classList.remove('dark');
}

toggle.addEventListener('click', () => {
  if (html.classList.contains('dark')) {
    html.classList.remove('dark');
    localStorage.theme = 'light';
  } else {
    html.classList.add('dark');
    localStorage.theme = 'dark';
  }
});
</script>
```

### React Dark Mode Toggle

```jsx
import { useState, useEffect } from 'react';

function DarkModeToggle() {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Check localStorage or system preference
    const isDark = localStorage.theme === 'dark' ||
      (!('theme' in localStorage) &&
       window.matchMedia('(prefers-color-scheme: dark)').matches);

    setDarkMode(isDark);
    if (isDark) {
      document.documentElement.classList.add('dark');
    }
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    if (!darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.theme = 'dark';
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.theme = 'light';
    }
  };

  return (
    <button
      onClick={toggleDarkMode}
      className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700"
    >
      {darkMode ? '☀️' : '🌙'}
    </button>
  );
}
```

---

## Component Patterns

Building real-world components with Tailwind utilities.

### Buttons

```html
<!-- Primary button -->
<button class="
  px-4 py-2
  bg-blue-600 hover:bg-blue-700
  active:bg-blue-800
  text-white font-medium
  rounded-lg
  transition-colors
  focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
">
  Primary Button
</button>

<!-- Secondary button -->
<button class="
  px-4 py-2
  bg-gray-200 hover:bg-gray-300
  text-gray-900 font-medium
  rounded-lg
  transition-colors
">
  Secondary Button
</button>

<!-- Outline button -->
<button class="
  px-4 py-2
  border-2 border-blue-600
  text-blue-600 hover:bg-blue-50
  font-medium rounded-lg
  transition-colors
">
  Outline Button
</button>

<!-- Ghost button -->
<button class="
  px-4 py-2
  text-blue-600 hover:bg-blue-50
  font-medium rounded-lg
  transition-colors
">
  Ghost Button
</button>

<!-- Danger button -->
<button class="
  px-4 py-2
  bg-red-600 hover:bg-red-700
  text-white font-medium
  rounded-lg
">
  Delete
</button>

<!-- Disabled button -->
<button
  class="
    px-4 py-2
    bg-blue-600
    text-white font-medium
    rounded-lg
    disabled:bg-gray-300 disabled:cursor-not-allowed
  "
  disabled
>
  Disabled
</button>

<!-- Button sizes -->
<button class="px-2 py-1 text-sm bg-blue-600 text-white rounded">Small</button>
<button class="px-4 py-2 text-base bg-blue-600 text-white rounded-lg">Medium</button>
<button class="px-6 py-3 text-lg bg-blue-600 text-white rounded-lg">Large</button>
<button class="px-8 py-4 text-xl bg-blue-600 text-white rounded-xl">XL</button>

<!-- Icon button -->
<button class="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
  </svg>
</button>

<!-- Button with icon -->
<button class="
  flex items-center gap-2
  px-4 py-2
  bg-blue-600 hover:bg-blue-700
  text-white rounded-lg
">
  <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
  </svg>
  Add Item
</button>

<!-- Loading button -->
<button class="
  flex items-center gap-2
  px-4 py-2
  bg-blue-600
  text-white rounded-lg
  cursor-wait
" disabled>
  <svg class="animate-spin h-5 w-5" viewBox="0 0 24 24">
    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
  Loading...
</button>

<!-- Button group -->
<div class="inline-flex rounded-lg shadow-sm">
  <button class="px-4 py-2 bg-white border border-gray-300 rounded-l-lg hover:bg-gray-50">
    Left
  </button>
  <button class="px-4 py-2 bg-white border-t border-b border-gray-300 hover:bg-gray-50">
    Middle
  </button>
  <button class="px-4 py-2 bg-white border border-gray-300 rounded-r-lg hover:bg-gray-50">
    Right
  </button>
</div>
```

### Cards

```html
<!-- Basic card -->
<div class="bg-white rounded-lg shadow-md p-6">
  <h3 class="text-xl font-bold mb-2">Card Title</h3>
  <p class="text-gray-700">
    This is a simple card component with rounded corners and shadow.
  </p>
</div>

<!-- Product card -->
<div class="group bg-white rounded-lg shadow-md overflow-hidden hover:shadow-xl transition-shadow">
  <!-- Image -->
  <div class="relative overflow-hidden">
    <img
      src="product.jpg"
      alt="Product"
      class="w-full h-48 object-cover group-hover:scale-110 transition-transform duration-300"
    >
    <!-- Badge -->
    <span class="absolute top-2 right-2 bg-red-500 text-white text-xs font-bold px-2 py-1 rounded">
      SALE
    </span>
  </div>

  <!-- Content -->
  <div class="p-4">
    <h3 class="text-lg font-semibold mb-2 group-hover:text-blue-600 transition-colors">
      Product Name
    </h3>
    <p class="text-gray-600 text-sm mb-4">
      Product description goes here
    </p>

    <!-- Price and button -->
    <div class="flex items-center justify-between">
      <div>
        <span class="text-gray-400 line-through text-sm">$99.00</span>
        <span class="text-2xl font-bold text-gray-900 ml-2">$79.00</span>
      </div>
      <button class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
        Add to Cart
      </button>
    </div>
  </div>
</div>

<!-- Profile card -->
<div class="bg-white rounded-xl shadow-lg p-6 max-w-sm">
  <!-- Avatar -->
  <div class="flex items-center gap-4 mb-4">
    <img
      src="avatar.jpg"
      alt="Profile"
      class="w-16 h-16 rounded-full object-cover"
    >
    <div>
      <h3 class="text-lg font-bold text-gray-900">John Doe</h3>
      <p class="text-gray-500 text-sm">Software Engineer</p>
    </div>
  </div>

  <!-- Bio -->
  <p class="text-gray-700 mb-4">
    Passionate about building great user experiences with modern web technologies.
  </p>

  <!-- Stats -->
  <div class="flex gap-4 mb-4">
    <div class="text-center">
      <div class="text-2xl font-bold text-gray-900">1.2K</div>
      <div class="text-gray-500 text-sm">Followers</div>
    </div>
    <div class="text-center">
      <div class="text-2xl font-bold text-gray-900">456</div>
      <div class="text-gray-500 text-sm">Following</div>
    </div>
    <div class="text-center">
      <div class="text-2xl font-bold text-gray-900">89</div>
      <div class="text-gray-500 text-sm">Posts</div>
    </div>
  </div>

  <!-- Actions -->
  <div class="flex gap-2">
    <button class="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
      Follow
    </button>
    <button class="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
      Message
    </button>
  </div>
</div>

<!-- Stats card with icon -->
<div class="bg-white rounded-lg shadow-md p-6">
  <div class="flex items-center justify-between mb-4">
    <div>
      <p class="text-gray-500 text-sm font-medium">Total Revenue</p>
      <p class="text-3xl font-bold text-gray-900">$45,231</p>
    </div>
    <div class="p-3 bg-green-100 rounded-full">
      <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
      </svg>
    </div>
  </div>
  <div class="flex items-center gap-1 text-sm">
    <span class="text-green-600 font-medium">↑ 12%</span>
    <span class="text-gray-500">from last month</span>
  </div>
</div>
```

### Forms

```html
<!-- Complete form -->
<form class="max-w-md mx-auto bg-white rounded-lg shadow-md p-6">
  <h2 class="text-2xl font-bold mb-6">Sign Up</h2>

  <!-- Text input -->
  <div class="mb-4">
    <label class="block text-gray-700 font-medium mb-2" for="name">
      Full Name
    </label>
    <input
      id="name"
      type="text"
      class="
        w-full px-4 py-2
        border border-gray-300 rounded-lg
        focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
        placeholder:text-gray-400
      "
      placeholder="John Doe"
    >
  </div>

  <!-- Email input with validation states -->
  <div class="mb-4">
    <label class="block text-gray-700 font-medium mb-2" for="email">
      Email
    </label>
    <input
      id="email"
      type="email"
      class="
        w-full px-4 py-2
        border rounded-lg
        focus:outline-none focus:ring-2 focus:ring-blue-500
        invalid:border-red-500 invalid:ring-red-500
        valid:border-green-500
      "
      placeholder="john@example.com"
      required
    >
    <p class="mt-1 text-sm text-red-600 hidden peer-invalid:block">
      Please enter a valid email
    </p>
  </div>

  <!-- Password input -->
  <div class="mb-4">
    <label class="block text-gray-700 font-medium mb-2" for="password">
      Password
    </label>
    <input
      id="password"
      type="password"
      class="
        w-full px-4 py-2
        border border-gray-300 rounded-lg
        focus:outline-none focus:ring-2 focus:ring-blue-500
      "
      required
    >
  </div>

  <!-- Select -->
  <div class="mb-4">
    <label class="block text-gray-700 font-medium mb-2" for="country">
      Country
    </label>
    <select
      id="country"
      class="
        w-full px-4 py-2
        border border-gray-300 rounded-lg
        focus:outline-none focus:ring-2 focus:ring-blue-500
        bg-white
      "
    >
      <option>United States</option>
      <option>Canada</option>
      <option>United Kingdom</option>
      <option>Australia</option>
    </select>
  </div>

  <!-- Textarea -->
  <div class="mb-4">
    <label class="block text-gray-700 font-medium mb-2" for="bio">
      Bio
    </label>
    <textarea
      id="bio"
      rows="4"
      class="
        w-full px-4 py-2
        border border-gray-300 rounded-lg
        focus:outline-none focus:ring-2 focus:ring-blue-500
        resize-none
      "
      placeholder="Tell us about yourself..."
    ></textarea>
  </div>

  <!-- Checkbox -->
  <div class="mb-4">
    <label class="flex items-center">
      <input
        type="checkbox"
        class="
          w-4 h-4
          text-blue-600
          border-gray-300 rounded
          focus:ring-2 focus:ring-blue-500
        "
      >
      <span class="ml-2 text-gray-700">I agree to the Terms and Conditions</span>
    </label>
  </div>

  <!-- Radio buttons -->
  <div class="mb-6">
    <p class="text-gray-700 font-medium mb-2">Newsletter</p>
    <label class="flex items-center mb-2">
      <input
        type="radio"
        name="newsletter"
        value="daily"
        class="w-4 h-4 text-blue-600 focus:ring-2 focus:ring-blue-500"
      >
      <span class="ml-2 text-gray-700">Daily</span>
    </label>
    <label class="flex items-center mb-2">
      <input
        type="radio"
        name="newsletter"
        value="weekly"
        class="w-4 h-4 text-blue-600 focus:ring-2 focus:ring-blue-500"
        checked
      >
      <span class="ml-2 text-gray-700">Weekly</span>
    </label>
    <label class="flex items-center">
      <input
        type="radio"
        name="newsletter"
        value="never"
        class="w-4 h-4 text-blue-600 focus:ring-2 focus:ring-blue-500"
      >
      <span class="ml-2 text-gray-700">Never</span>
    </label>
  </div>

  <!-- Submit button -->
  <button
    type="submit"
    class="
      w-full px-4 py-2
      bg-blue-600 hover:bg-blue-700
      text-white font-medium rounded-lg
      transition-colors
      focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
    "
  >
    Create Account
  </button>
</form>

<!-- File upload -->
<div class="max-w-md mx-auto">
  <label class="
    flex flex-col items-center justify-center
    w-full h-32
    border-2 border-gray-300 border-dashed rounded-lg
    cursor-pointer
    hover:bg-gray-50
    transition-colors
  ">
    <div class="flex flex-col items-center justify-center pt-5 pb-6">
      <svg class="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
      </svg>
      <p class="mb-2 text-sm text-gray-500">
        <span class="font-semibold">Click to upload</span> or drag and drop
      </p>
      <p class="text-xs text-gray-500">PNG, JPG or GIF (MAX. 800x400px)</p>
    </div>
    <input type="file" class="hidden">
  </label>
</div>
```

### Navigation

```html
<!-- Desktop navbar -->
<nav class="bg-white shadow-lg">
  <div class="container mx-auto px-4">
    <div class="flex items-center justify-between h-16">
      <!-- Logo -->
      <div class="flex items-center">
        <a href="/" class="text-xl font-bold text-gray-900">
          Logo
        </a>
      </div>

      <!-- Desktop menu -->
      <div class="hidden md:flex items-center space-x-4">
        <a href="#" class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md font-medium">
          Home
        </a>
        <a href="#" class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md font-medium">
          About
        </a>
        <a href="#" class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md font-medium">
          Services
        </a>
        <a href="#" class="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md font-medium">
          Contact
        </a>
        <button class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
          Sign In
        </button>
      </div>

      <!-- Mobile menu button -->
      <div class="md:hidden">
        <button class="p-2 rounded-md text-gray-700 hover:bg-gray-100">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
          </svg>
        </button>
      </div>
    </div>
  </div>

  <!-- Mobile menu (hidden by default) -->
  <div class="md:hidden hidden">
    <div class="px-2 pt-2 pb-3 space-y-1">
      <a href="#" class="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">Home</a>
      <a href="#" class="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">About</a>
      <a href="#" class="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">Services</a>
      <a href="#" class="block px-3 py-2 rounded-md text-gray-700 hover:bg-gray-100">Contact</a>
    </div>
  </div>
</nav>

<!-- Sidebar navigation -->
<aside class="w-64 bg-gray-900 min-h-screen">
  <div class="p-4">
    <h2 class="text-white text-xl font-bold mb-6">Dashboard</h2>
    <nav class="space-y-2">
      <a href="#" class="flex items-center gap-3 px-4 py-2 bg-blue-600 text-white rounded-lg">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path>
        </svg>
        Dashboard
      </a>
      <a href="#" class="flex items-center gap-3 px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
        </svg>
        Users
      </a>
      <a href="#" class="flex items-center gap-3 px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
        </svg>
        Settings
      </a>
    </nav>
  </div>
</aside>

<!-- Breadcrumbs -->
<nav class="flex" aria-label="Breadcrumb">
  <ol class="inline-flex items-center space-x-1 md:space-x-3">
    <li class="inline-flex items-center">
      <a href="#" class="text-gray-700 hover:text-blue-600">
        Home
      </a>
    </li>
    <li>
      <div class="flex items-center">
        <svg class="w-6 h-6 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"></path>
        </svg>
        <a href="#" class="ml-1 text-gray-700 hover:text-blue-600">
          Products
        </a>
      </div>
    </li>
    <li>
      <div class="flex items-center">
        <svg class="w-6 h-6 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"></path>
        </svg>
        <span class="ml-1 text-gray-500">Details</span>
      </div>
    </li>
  </ol>
</nav>

<!-- Tabs -->
<div class="border-b border-gray-200">
  <nav class="flex space-x-8">
    <a href="#" class="border-b-2 border-blue-500 text-blue-600 py-4 px-1 font-medium">
      Profile
    </a>
    <a href="#" class="border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 py-4 px-1 font-medium">
      Settings
    </a>
    <a href="#" class="border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 py-4 px-1 font-medium">
      Notifications
    </a>
  </nav>
</div>
```

### Modals and Overlays

```html
<!-- Modal -->
<div class="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center p-4 z-50">
  <!-- Modal content -->
  <div class="bg-white rounded-lg shadow-xl max-w-md w-full">
    <!-- Header -->
    <div class="flex items-center justify-between p-6 border-b">
      <h3 class="text-xl font-semibold text-gray-900">
        Modal Title
      </h3>
      <button class="text-gray-400 hover:text-gray-600">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
        </svg>
      </button>
    </div>

    <!-- Body -->
    <div class="p-6">
      <p class="text-gray-700">
        This is the modal content. You can add any content here.
      </p>
    </div>

    <!-- Footer -->
    <div class="flex justify-end gap-3 p-6 border-t">
      <button class="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50">
        Cancel
      </button>
      <button class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
        Confirm
      </button>
    </div>
  </div>
</div>

<!-- Dropdown menu -->
<div class="relative inline-block text-left">
  <button class="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50">
    Options
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
    </svg>
  </button>

  <!-- Dropdown panel -->
  <div class="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-lg ring-1 ring-black ring-opacity-5 z-10">
    <div class="py-1">
      <a href="#" class="block px-4 py-2 text-gray-700 hover:bg-gray-100">
        Edit
      </a>
      <a href="#" class="block px-4 py-2 text-gray-700 hover:bg-gray-100">
        Duplicate
      </a>
      <hr class="my-1">
      <a href="#" class="block px-4 py-2 text-red-600 hover:bg-gray-100">
        Delete
      </a>
    </div>
  </div>
</div>

<!-- Toast notification -->
<div class="fixed top-4 right-4 bg-white rounded-lg shadow-lg p-4 max-w-sm animate-slide-in">
  <div class="flex items-start gap-3">
    <!-- Success icon -->
    <div class="flex-shrink-0">
      <svg class="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
      </svg>
    </div>
    <!-- Content -->
    <div class="flex-1">
      <p class="font-medium text-gray-900">Success!</p>
      <p class="text-sm text-gray-500">Your changes have been saved.</p>
    </div>
    <!-- Close button -->
    <button class="flex-shrink-0 text-gray-400 hover:text-gray-600">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
      </svg>
    </button>
  </div>
</div>
```

### Alerts and Badges

```html
<!-- Alert variants -->
<div class="bg-blue-50 border border-blue-200 text-blue-800 px-4 py-3 rounded-lg" role="alert">
  <strong class="font-bold">Info!</strong>
  <span class="block sm:inline"> This is an informational message.</span>
</div>

<div class="bg-green-50 border border-green-200 text-green-800 px-4 py-3 rounded-lg" role="alert">
  <strong class="font-bold">Success!</strong>
  <span class="block sm:inline"> Operation completed successfully.</span>
</div>

<div class="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded-lg" role="alert">
  <strong class="font-bold">Warning!</strong>
  <span class="block sm:inline"> Please review before proceeding.</span>
</div>

<div class="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg" role="alert">
  <strong class="font-bold">Error!</strong>
  <span class="block sm:inline"> Something went wrong.</span>
</div>

<!-- Badges -->
<span class="px-2 py-1 text-xs font-semibold bg-gray-200 text-gray-800 rounded-full">
  Default
</span>

<span class="px-2 py-1 text-xs font-semibold bg-blue-100 text-blue-800 rounded-full">
  Primary
</span>

<span class="px-2 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded-full">
  Success
</span>

<span class="px-2 py-1 text-xs font-semibold bg-red-100 text-red-800 rounded-full">
  Danger
</span>

<!-- Badge with dot -->
<span class="inline-flex items-center gap-1 px-2 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded-full">
  <span class="w-2 h-2 bg-green-500 rounded-full"></span>
  Active
</span>

<!-- Loading skeleton -->
<div class="animate-pulse">
  <div class="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
  <div class="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
  <div class="h-4 bg-gray-200 rounded w-5/6"></div>
</div>

<!-- Progress bar -->
<div class="w-full bg-gray-200 rounded-full h-2.5">
  <div class="bg-blue-600 h-2.5 rounded-full" style="width: 45%"></div>
</div>
```

---

## Layout Patterns

### Dashboard Layout

```html
<div class="min-h-screen bg-gray-100">
  <!-- Sidebar -->
  <aside class="fixed inset-y-0 left-0 w-64 bg-gray-900">
    <!-- Sidebar content here -->
  </aside>

  <!-- Main content -->
  <div class="ml-64">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-10">
      <div class="px-6 py-4">
        <h1 class="text-2xl font-bold">Dashboard</h1>
      </div>
    </header>

    <!-- Content -->
    <main class="p-6">
      <!-- Grid of cards/widgets -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <!-- Cards here -->
      </div>
    </main>
  </div>
</div>
```

### Landing Page Hero

```html
<section class="relative bg-gradient-to-r from-blue-600 to-indigo-700 text-white">
  <div class="container mx-auto px-4 py-20 md:py-32">
    <div class="max-w-3xl mx-auto text-center">
      <h1 class="text-4xl md:text-5xl lg:text-6xl font-bold mb-6">
        Build Amazing Products
      </h1>
      <p class="text-xl md:text-2xl mb-8 text-blue-100">
        The fastest way to create beautiful, responsive websites
      </p>
      <div class="flex flex-col sm:flex-row gap-4 justify-center">
        <button class="px-8 py-3 bg-white text-blue-600 rounded-lg font-semibold hover:bg-gray-100">
          Get Started
        </button>
        <button class="px-8 py-3 border-2 border-white rounded-lg font-semibold hover:bg-white hover:text-blue-600 transition-colors">
          Learn More
        </button>
      </div>
    </div>
  </div>
</section>
```

### Centering Techniques

```html
<!-- Flexbox centering -->
<div class="flex items-center justify-center min-h-screen">
  <div class="text-center">
    Perfectly centered
  </div>
</div>

<!-- Grid centering -->
<div class="grid place-items-center min-h-screen">
  <div>
    Centered with grid
  </div>
</div>

<!-- Absolute centering -->
<div class="relative h-screen">
  <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
    Centered with transform
  </div>
</div>
```

### Holy Grail Layout

```html
<div class="min-h-screen flex flex-col">
  <!-- Header -->
  <header class="bg-gray-800 text-white p-4">
    Header
  </header>

  <!-- Main content area -->
  <div class="flex flex-1">
    <!-- Left sidebar -->
    <aside class="w-64 bg-gray-100 p-4">
      Left Sidebar
    </aside>

    <!-- Main content -->
    <main class="flex-1 p-4">
      Main Content
    </main>

    <!-- Right sidebar -->
    <aside class="w-64 bg-gray-100 p-4">
      Right Sidebar
    </aside>
  </div>

  <!-- Footer -->
  <footer class="bg-gray-800 text-white p-4">
    Footer
  </footer>
</div>
```

### Sticky Header/Footer

```html
<div class="min-h-screen flex flex-col">
  <!-- Sticky header -->
  <header class="sticky top-0 bg-white shadow-md p-4 z-10">
    Sticky Header
  </header>

  <!-- Main content (scrollable) -->
  <main class="flex-1 p-4">
    <!-- Long content here -->
  </main>

  <!-- Sticky footer -->
  <footer class="sticky bottom-0 bg-gray-800 text-white p-4">
    Sticky Footer
  </footer>
</div>
```

---

## Customization

### Extending Colors

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        // Brand colors
        brand: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          900: '#1e3a8a',
        },
        // Single color
        'accent': '#ff6b6b',
      },
    },
  },
}
```

```html
<!-- Use custom colors -->
<div class="bg-brand-500 text-white">Brand color</div>
<div class="bg-accent text-white">Accent color</div>
```

### Extending Spacing

```javascript
module.exports = {
  theme: {
    extend: {
      spacing: {
        '128': '32rem',
        '144': '36rem',
      },
    },
  },
}
```

```html
<div class="p-128">Extra large padding</div>
```

### Custom Fonts

```javascript
module.exports = {
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Lexend', 'sans-serif'],
        body: ['Open Sans', 'sans-serif'],
      },
    },
  },
}
```

```css
/* In your CSS */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
```

```html
<h1 class="font-display">Display font</h1>
<p class="font-body">Body font</p>
```

### Arbitrary Values

Use square brackets for one-off custom values:

```html
<!-- Custom width -->
<div class="w-[347px]">Exact width</div>

<!-- Custom color -->
<div class="bg-[#1da1f2]">Twitter blue</div>

<!-- Custom grid -->
<div class="grid-cols-[200px_1fr_200px]">Custom grid</div>

<!-- Custom shadow -->
<div class="shadow-[0_35px_60px_-15px_rgba(0,0,0,0.3)]">Custom shadow</div>
```

### Adding Custom Utilities

```javascript
// tailwind.config.js
const plugin = require('tailwindcss/plugin')

module.exports = {
  plugins: [
    plugin(function({ addUtilities }) {
      const newUtilities = {
        '.text-shadow': {
          textShadow: '2px 2px 4px rgba(0,0,0,0.1)',
        },
        '.text-shadow-lg': {
          textShadow: '4px 4px 8px rgba(0,0,0,0.2)',
        },
      }
      addUtilities(newUtilities)
    })
  ],
}
```

```html
<h1 class="text-shadow">Text with shadow</h1>
```

---

## Plugin System

### Official Plugins

#### @tailwindcss/forms

Provides better default styles for form elements.

```bash
npm install @tailwindcss/forms
```

```javascript
// tailwind.config.js
module.exports = {
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
```

```html
<!-- Forms are automatically styled nicely -->
<input type="text" class="mt-1 block w-full">
<select class="mt-1 block w-full">
  <option>Option 1</option>
</select>
```

#### @tailwindcss/typography

Adds prose class for styling user-generated content.

```bash
npm install @tailwindcss/typography
```

```javascript
module.exports = {
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
```

```html
<article class="prose lg:prose-xl">
  <!-- All HTML elements are beautifully styled -->
  <h1>Heading</h1>
  <p>Paragraph with nice defaults</p>
  <ul>
    <li>List item</li>
  </ul>
</article>

<!-- Dark mode -->
<article class="prose dark:prose-invert">
  Content
</article>
```

#### @tailwindcss/aspect-ratio

Maintains aspect ratios for elements.

```bash
npm install @tailwindcss/aspect-ratio
```

```html
<div class="aspect-w-16 aspect-h-9">
  <iframe src="video.mp4"></iframe>
</div>
```

#### @tailwindcss/container-queries

Enables container-based responsive design.

```bash
npm install @tailwindcss/container-queries
```

```html
<div class="@container">
  <div class="@lg:text-xl">
    Responds to container size, not viewport
  </div>
</div>
```

### Creating Custom Plugins

```javascript
// tailwind.config.js
const plugin = require('tailwindcss/plugin')

module.exports = {
  plugins: [
    // Simple utility plugin
    plugin(function({ addUtilities }) {
      addUtilities({
        '.rotate-y-180': {
          transform: 'rotateY(180deg)',
        },
      })
    }),

    // Plugin with options
    plugin(function({ addComponents, theme }) {
      addComponents({
        '.btn': {
          padding: theme('spacing.4'),
          borderRadius: theme('borderRadius.lg'),
          fontWeight: theme('fontWeight.semibold'),
          '&:hover': {
            opacity: 0.8,
          },
        },
        '.btn-primary': {
          backgroundColor: theme('colors.blue.500'),
          color: theme('colors.white'),
        },
      })
    }),
  ],
}
```

---

## Framework Integration

### React / Next.js

Next.js 13+ includes Tailwind by default with `create-next-app`:

```bash
npx create-next-app@latest my-app --tailwind
```

**Manual setup:**

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Example React component:**

```jsx
// components/Button.jsx
export default function Button({ children, variant = 'primary' }) {
  const baseClasses = "px-4 py-2 rounded-lg font-medium transition-colors";
  const variants = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white",
    secondary: "bg-gray-200 hover:bg-gray-300 text-gray-900",
    outline: "border-2 border-blue-600 text-blue-600 hover:bg-blue-50",
  };

  return (
    <button className={`${baseClasses} ${variants[variant]}`}>
      {children}
    </button>
  );
}
```

**Using clsx for conditional classes:**

```jsx
import clsx from 'clsx';

function Button({ variant, size, children }) {
  return (
    <button
      className={clsx(
        'font-semibold rounded-lg transition-colors',
        {
          'bg-blue-600 text-white hover:bg-blue-700': variant === 'primary',
          'bg-gray-200 text-gray-900 hover:bg-gray-300': variant === 'secondary',
          'px-3 py-1.5 text-sm': size === 'sm',
          'px-4 py-2 text-base': size === 'md',
          'px-6 py-3 text-lg': size === 'lg',
        }
      )}
    >
      {children}
    </button>
  );
}
```

### Vue / Nuxt

**Nuxt 3:**

```bash
npm install -D @nuxtjs/tailwindcss
```

```typescript
// nuxt.config.ts
export default defineNuxtConfig({
  modules: ['@nuxtjs/tailwindcss'],
})
```

**Vue 3 component:**

```vue
<template>
  <button
    :class="[
      'px-4 py-2 rounded-lg font-medium transition-colors',
      variantClasses
    ]"
  >
    <slot />
  </button>
</template>

<script setup>
const props = defineProps({
  variant: {
    type: String,
    default: 'primary'
  }
});

const variantClasses = computed(() => {
  const variants = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white',
    secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-900',
  };
  return variants[props.variant];
});
</script>
```

### Svelte / SvelteKit

```bash
npx svelte-add@latest tailwindcss
```

**Svelte component:**

```svelte
<script>
  export let variant = 'primary';

  $: variantClasses = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white',
    secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-900',
  }[variant];
</script>

<button class="px-4 py-2 rounded-lg font-medium transition-colors {variantClasses}">
  <slot />
</button>
```

---

## Advanced Topics

### @layer Directive

Organize custom styles into Tailwind's layers:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  h1 {
    @apply text-4xl font-bold;
  }

  a {
    @apply text-blue-600 hover:underline;
  }
}

@layer components {
  .btn {
    @apply px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700;
  }

  .card {
    @apply bg-white rounded-lg shadow-md p-6;
  }
}

@layer utilities {
  .text-shadow {
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
  }
}
```

### @apply Directive

Extract repeated utilities into custom classes:

```css
.btn-primary {
  @apply px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors;
}
```

**⚠️ Use sparingly**: Only extract when you have true component repetition across multiple files.

### Custom Variants

```javascript
// tailwind.config.js
const plugin = require('tailwindcss/plugin')

module.exports = {
  plugins: [
    plugin(function({ addVariant }) {
      // Custom variant for third child
      addVariant('third', '&:nth-child(3)');

      // Custom variant for optional elements
      addVariant('optional', '&:optional');

      // Custom variant for hocus (hover + focus)
      addVariant('hocus', ['&:hover', '&:focus']);
    })
  ],
}
```

```html
<div class="third:bg-blue-500">Third child is blue</div>
<input class="optional:border-gray-300" />
<button class="hocus:bg-blue-700">Hover or focus</button>
```

### Important Modifier

Force a utility to be `!important`:

```html
<!-- Without important -->
<p class="text-red-500">Red text</p>

<!-- With important (overrides everything) -->
<p class="!text-red-500">Always red text</p>
```

### Arbitrary Variants

Create one-off variants with square brackets:

```html
<!-- Target specific data attribute -->
<div class="[&[data-state='active']]:bg-blue-500" data-state="active">
  Blue when active
</div>

<!-- Target child elements -->
<ul class="[&>li]:mb-2">
  <li>Item 1</li>
  <li>Item 2</li>
</ul>

<!-- Complex selectors -->
<div class="[&:nth-child(3)]:text-red-500">
  Third child is red
</div>
```

---

## Performance Optimization

### Content Configuration

Tell Tailwind exactly where to look for class names:

```javascript
// tailwind.config.js
module.exports = {
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './public/index.html',
    // Don't include:
    // - node_modules (unless using Tailwind in a package)
    // - build/dist folders
  ],
}
```

### JIT (Just-In-Time) Mode

JIT is enabled by default in Tailwind 3+. It generates styles on-demand as you author your templates.

**Benefits:**
- Lightning fast build times
- All variants enabled by default
- Arbitrary values work everywhere
- Smaller CSS in development
- Better performance

### Production Build

Tailwind automatically purges unused styles in production:

```bash
NODE_ENV=production npx tailwindcss -o output.css --minify
```

**In build tools**, set `NODE_ENV=production`:

```json
// package.json
{
  "scripts": {
    "build": "NODE_ENV=production webpack build"
  }
}
```

### Bundle Size Tips

1. **Only import what you need** - The config already does this via purging
2. **Use PurgeCSS** - Automatically enabled in production
3. **Avoid safelist overuse** - Only safelist truly dynamic classes
4. **Enable minification** - Always in production builds

---

## Best Practices

1. **Use utility classes in HTML**
   - Keeps styles close to usage
   - Easier to understand and modify
   - No context switching

2. **Extract components when needed**
   - Repeated patterns across multiple files
   - True reusable components
   - Not just to reduce class count in one place

3. **Use consistent spacing scale**
   - Stick to Tailwind's spacing scale (4, 8, 16, 24, 32...)
   - Use arbitrary values sparingly
   - Creates visual rhythm

4. **Mobile-first responsive design**
   - Start with mobile layout
   - Add breakpoints for larger screens
   - `md:` for tablet, `lg:` for desktop

5. **Organize classes logically**
   - Layout → Spacing → Sizing → Typography → Colors → Effects
   - Example: `flex items-center px-4 py-2 text-lg font-bold bg-blue-500 rounded-lg shadow`

6. **Use editor extensions**
   - Tailwind CSS IntelliSense (VSCode)
   - Auto-complete and class sorting
   - Linting and validation

7. **Combine with component frameworks**
   - Headless UI for accessible components
   - Radix UI primitives
   - Build design system on top

8. **Don't fight the framework**
   - Use Tailwind's design tokens
   - Extend theme rather than arbitrary values
   - Embrace the constraints

9. **When NOT to use Tailwind**
   - Simple static sites
   - Teams that prefer CSS-in-JS
   - Projects with strict CSS architecture requirements
   - When you need maximum control over generated CSS

10. **Performance considerations**
    - Configure content paths correctly
    - Safelist only what's necessary
    - Use JIT mode (default in v3)
    - Minify in production

---

## Accessibility

### Focus States

```html
<!-- Always include focus styles -->
<button class="
  bg-blue-600
  focus:outline-none
  focus:ring-2
  focus:ring-blue-500
  focus:ring-offset-2
">
  Accessible button
</button>

<!-- Focus-visible (keyboard only) -->
<a href="#" class="
  focus:outline-none
  focus-visible:ring-2
  focus-visible:ring-blue-500
">
  Link
</a>
```

### Screen Reader Utilities

```html
<!-- Screen reader only text -->
<button class="p-2">
  <svg class="w-6 h-6" fill="currentColor">
    <!-- Icon -->
  </svg>
  <span class="sr-only">Close menu</span>
</button>

<!-- Hide from screen readers -->
<div aria-hidden="true" class="text-gray-400">
  Decorative element
</div>
```

### Color Contrast

```html
<!-- Good contrast -->
<div class="bg-gray-900 text-white">High contrast</div>

<!-- Ensure sufficient contrast -->
<p class="text-gray-600"><!-- Check contrast ratio --></p>

<!-- Use Tailwind's color scales appropriately -->
<!-- On white bg: gray-700, gray-800, gray-900 are safe -->
<!-- On dark bg: gray-100, gray-200, gray-300 are safe -->
```

### Keyboard Navigation

```html
<!-- Ensure tab order makes sense -->
<nav>
  <a href="#" class="focus:ring-2 tabindex="0">Link 1</a>
  <a href="#" class="focus:ring-2 tabindex="0">Link 2</a>
</nav>

<!-- Skip link for keyboard users -->
<a href="#main-content" class="sr-only focus:not-sr-only focus:absolute focus:top-0">
  Skip to main content
</a>
```

---

## Migration and Comparison

### Migrating from Bootstrap

**Bootstrap approach:**
```html
<div class="container">
  <div class="row">
    <div class="col-md-6">Column 1</div>
    <div class="col-md-6">Column 2</div>
  </div>
</div>
```

**Tailwind equivalent:**
```html
<div class="container mx-auto px-4">
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
    <div>Column 1</div>
    <div>Column 2</div>
  </div>
</div>
```

### Tailwind vs CSS-in-JS

| Aspect | Tailwind | CSS-in-JS (styled-components) |
|--------|----------|------------------------------|
| **Syntax** | HTML classes | JavaScript objects/strings |
| **Runtime** | No runtime | Runtime overhead |
| **File size** | Small (purged) | Depends on usage |
| **Theming** | Config file | Theme provider |
| **Learning curve** | Learn utilities | Learn library API |
| **Type safety** | Via LSP | Native TypeScript |

### Pros and Cons

**Pros:**
- ✅ Rapid development
- ✅ Consistent design system
- ✅ Small production bundle
- ✅ No naming fatigue
- ✅ Responsive by default
- ✅ Great developer experience
- ✅ Highly customizable

**Cons:**
- ❌ HTML can look cluttered
- ❌ Learning curve for utilities
- ❌ Team alignment needed
- ❌ Harder to enforce design patterns
- ❌ Some prefer separation of concerns

---

## Tooling and Ecosystem

### Editor Extensions

**VS Code:**
- **Tailwind CSS IntelliSense**: Auto-complete, syntax highlighting, linting
- **Tailwind Fold**: Fold long class strings
- **Headwind**: Auto-sort Tailwind classes

**Settings for VSCode:**
```json
{
  "tailwindCSS.experimental.classRegex": [
    ["class:\\s*?[\"'`]([^\"'`]*).*?[\"'`]", "[\"'`]([^\"'`]*).*?[\"'`]"],
  ],
  "editor.quickSuggestions": {
    "strings": true
  }
}
```

### Prettier Plugin

Auto-sort classes in consistent order:

```bash
npm install -D prettier prettier-plugin-tailwindcss
```

```json
// .prettierrc
{
  "plugins": ["prettier-plugin-tailwindcss"]
}
```

### Headless UI

Unstyled, accessible UI components:

```bash
npm install @headlessui/react
```

```jsx
import { Dialog } from '@headlessui/react'

function MyDialog({ isOpen, onClose }) {
  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="bg-white rounded-lg p-6 max-w-sm">
          <Dialog.Title className="text-lg font-medium">Title</Dialog.Title>
          <Dialog.Description>Description</Dialog.Description>

          <button onClick={onClose} className="mt-4 px-4 py-2 bg-blue-600 text-white rounded">
            Close
          </button>
        </Dialog.Panel>
      </div>
    </Dialog>
  )
}
```

### Component Libraries

**Free:**
- **daisyUI**: Component library built on Tailwind
- **Flowbite**: Open-source component library
- **Preline**: Free Tailwind components
- **Mamba UI**: Free Tailwind components

**Commercial:**
- **Tailwind UI**: Official component library (paid)
- **Meraki UI**: Premium components

---

## Resources

### Official Documentation

- **Tailwind CSS Docs**: https://tailwindcss.com/docs
- **Tailwind Play** (playground): https://play.tailwindcss.com/
- **GitHub**: https://github.com/tailwindlabs/tailwindcss

### Learning Resources

- **Tailwind CSS Tutorial** (official): https://tailwindcss.com/docs/installation
- **Scrimba Tailwind Course**: Interactive lessons
- **Tailwind from A to Z** (YouTube): Adam Wathan
- **Tailwind CSS From Scratch** (Traversy Media)

### Component Libraries

- **Headless UI**: https://headlessui.com/
- **daisyUI**: https://daisyui.com/
- **Flowbite**: https://flowbite.com/
- **Tailwind UI**: https://tailwindui.com/ (commercial)

### Tools

- **Tailwind CSS IntelliSense**: VS Code extension
- **Prettier Plugin**: Auto-sort classes
- **Tailwind Cheat Sheet**: https://nerdcave.com/tailwind-cheat-sheet
- **Tailwind Color Shades Generator**: Generate custom color palettes

### Icons

- **Heroicons**: https://heroicons.com/ (by Tailwind makers)
- **Tabler Icons**: https://tabler-icons.io/
- **Lucide Icons**: https://lucide.dev/

### Community

- **Discord**: Official Tailwind Discord server
- **Twitter**: @tailwindcss
- **GitHub Discussions**: Community Q&A
- **Reddit**: r/tailwindcss

---

**Last Updated**: January 2025
