# Vue.js

Vue.js is a progressive JavaScript framework for building user interfaces. It's designed to be incrementally adoptable and focuses on the view layer.

## Installation

```bash
# Create Vue 3 project
npm create vue@latest my-app
cd my-app
npm install
npm run dev

# Or via CDN
<script src="https://unpkg.com/vue@3"></script>
```

## Component Basics

```vue
<!-- HelloWorld.vue -->
<template>
  <div>
    <h1>{{ message }}</h1>
    <button @click="increment">Count: {{ count }}</button>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const message = ref('Hello Vue!')
const count = ref(0)

function increment() {
  count.value++
}
</script>

<style scoped>
h1 {
  color: #42b983;
}
</style>
```

## Reactivity

```vue
<script setup>
import { ref, reactive, computed, watch } from 'vue'

// Refs
const count = ref(0)

// Reactive objects
const state = reactive({
  name: 'John',
  age: 30
})

// Computed properties
const doubled = computed(() => count.value * 2)

// Watchers
watch(count, (newVal, oldVal) => {
  console.log(`Count changed from ${oldVal} to ${newVal}`)
})
</script>
```

## Props and Emits

```vue
<!-- Child.vue -->
<script setup>
const props = defineProps({
  title: String,
  count: {
    type: Number,
    default: 0
  }
})

const emit = defineEmits(['update', 'delete'])

function handleClick() {
  emit('update', { id: 1, value: 'new' })
}
</script>

<template>
  <h2>{{ title }}</h2>
  <button @click="handleClick">Update</button>
</template>

<!-- Parent.vue -->
<Child
  title="My Component"
  :count="10"
  @update="handleUpdate"
/>
```

## Directives

```vue
<template>
  <!-- Conditional rendering -->
  <div v-if="show">Visible</div>
  <div v-else>Hidden</div>

  <!-- List rendering -->
  <ul>
    <li v-for="item in items" :key="item.id">
      {{ item.name }}
    </li>
  </ul>

  <!-- Two-way binding -->
  <input v-model="text" />

  <!-- Event handling -->
  <button @click="handleClick">Click me</button>

  <!-- Dynamic attributes -->
  <img :src="imageUrl" :alt="description" />
</template>
```

## Lifecycle Hooks

```vue
<script setup>
import { onMounted, onUpdated, onUnmounted } from 'vue'

onMounted(() => {
  console.log('Component mounted')
})

onUpdated(() => {
  console.log('Component updated')
})

onUnmounted(() => {
  console.log('Component unmounted')
})
</script>
```

## Quick Reference

| Feature | Syntax |
|---------|--------|
| Data binding | `{{ variable }}` |
| Attribute binding | `:attribute="value"` |
| Event handling | `@event="handler"` |
| Two-way binding | `v-model="variable"` |
| Conditional | `v-if`, `v-else-if`, `v-else` |
| Loop | `v-for="item in items"` |

Vue.js provides an approachable, versatile, and performant framework for building modern web interfaces.
