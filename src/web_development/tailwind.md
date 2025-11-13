# Tailwind CSS

Tailwind CSS is a utility-first CSS framework for rapidly building custom user interfaces. Instead of opinionated predesigned components, Tailwind provides low-level utility classes.

## Installation

```bash
# Via npm
npm install -D tailwindcss
npx tailwindcss init

# Via CDN (development only)
<script src="https://cdn.tailwindcss.com"></script>
```

## Configuration

```javascript
// tailwind.config.js
module.exports = {
  content: ["./src/**/*.{html,js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: '#3490dc',
        secondary: '#ffed4e',
      },
    },
  },
  plugins: [],
}
```

## Basic Usage

```html
<!-- Flexbox layout -->
<div class="flex items-center justify-between p-4">
  <h1 class="text-2xl font-bold text-gray-900">Title</h1>
  <button class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
    Button
  </button>
</div>

<!-- Grid layout -->
<div class="grid grid-cols-3 gap-4">
  <div class="bg-gray-200 p-4">Item 1</div>
  <div class="bg-gray-200 p-4">Item 2</div>
  <div class="bg-gray-200 p-4">Item 3</div>
</div>

<!-- Responsive design -->
<div class="w-full md:w-1/2 lg:w-1/3 xl:w-1/4">
  Responsive width
</div>
```

## Common Classes

| Category | Classes |
|----------|---------|
| Layout | `container`, `mx-auto`, `flex`, `grid` |
| Spacing | `p-4`, `m-2`, `px-6`, `py-3` |
| Typography | `text-xl`, `font-bold`, `text-center` |
| Colors | `bg-blue-500`, `text-white` |
| Borders | `border`, `rounded`, `border-gray-300` |

Tailwind promotes rapid UI development with utility classes and responsive design built-in.
