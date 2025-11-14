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

## Composition API Advanced Patterns

### Composables (Reusable Logic)

```javascript
// composables/useCounter.js
import { ref, computed } from 'vue'

export function useCounter(initialValue = 0) {
  const count = ref(initialValue)
  const doubled = computed(() => count.value * 2)

  function increment() {
    count.value++
  }

  function decrement() {
    count.value--
  }

  function reset() {
    count.value = initialValue
  }

  return {
    count,
    doubled,
    increment,
    decrement,
    reset
  }
}

// Usage in component
<script setup>
import { useCounter } from './composables/useCounter'

const { count, doubled, increment, decrement, reset } = useCounter(10)
</script>
```

### Mouse Tracker Composable

```javascript
// composables/useMouse.js
import { ref, onMounted, onUnmounted } from 'vue'

export function useMouse() {
  const x = ref(0)
  const y = ref(0)

  function update(event) {
    x.value = event.pageX
    y.value = event.pageY
  }

  onMounted(() => window.addEventListener('mousemove', update))
  onUnmounted(() => window.removeEventListener('mousemove', update))

  return { x, y }
}
```

### Async Data Fetching Composable

```javascript
// composables/useFetch.js
import { ref, watchEffect, toValue } from 'vue'

export function useFetch(url) {
  const data = ref(null)
  const error = ref(null)
  const loading = ref(false)

  const fetchData = async () => {
    loading.value = true
    error.value = null
    data.value = null

    try {
      const response = await fetch(toValue(url))
      if (!response.ok) throw new Error('Network response was not ok')
      data.value = await response.json()
    } catch (e) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  watchEffect(() => {
    fetchData()
  })

  return { data, error, loading, refetch: fetchData }
}

// Usage
<script setup>
import { ref } from 'vue'
import { useFetch } from './composables/useFetch'

const userId = ref(1)
const url = computed(() => `https://api.example.com/users/${userId.value}`)
const { data, error, loading } = useFetch(url)
</script>
```

### toRef, toRefs, and unref

```vue
<script setup>
import { reactive, toRef, toRefs, unref } from 'vue'

const state = reactive({
  name: 'John',
  age: 30,
  email: 'john@example.com'
})

// Create a ref to a single property
const name = toRef(state, 'name')
name.value = 'Jane' // Updates state.name

// Convert all properties to refs (useful for destructuring)
const { age, email } = toRefs(state)
age.value = 31 // Updates state.age

// unref - get value from ref or non-ref
function logValue(maybeRef) {
  console.log(unref(maybeRef)) // Works with refs and plain values
}
</script>
```

### Readonly and Shallow Reactivity

```vue
<script setup>
import { reactive, readonly, shallowRef, shallowReactive } from 'vue'

// readonly - prevents mutations
const original = reactive({ count: 0 })
const copy = readonly(original)

original.count++ // Works
// copy.count++  // Warning: mutation on readonly proxy

// shallowRef - only .value is reactive
const shallowState = shallowRef({ nested: { count: 0 } })
shallowState.value = { nested: { count: 1 } } // Triggers update
shallowState.value.nested.count++ // Does NOT trigger update

// shallowReactive - only root level is reactive
const shallowObj = shallowReactive({
  count: 0,
  nested: { value: 1 }
})
shallowObj.count++ // Triggers update
shallowObj.nested.value++ // Does NOT trigger update
</script>
```

### Advanced Watchers

```vue
<script setup>
import { ref, watch, watchEffect, watchPostEffect } from 'vue'

const count = ref(0)
const name = ref('John')

// Watch specific sources
watch([count, name], ([newCount, newName], [oldCount, oldName]) => {
  console.log(`Count: ${oldCount} -> ${newCount}`)
  console.log(`Name: ${oldName} -> ${newName}`)
})

// Watch with options
watch(count, (newVal) => {
  console.log('Count changed:', newVal)
}, {
  immediate: true,  // Run immediately
  deep: true,       // Deep watch for objects
  flush: 'post'     // Run after component updates
})

// watchEffect - automatically tracks dependencies
watchEffect(() => {
  console.log(`Count is ${count.value}`)
})

// watchPostEffect - runs after component updates (access updated DOM)
watchPostEffect(() => {
  console.log('DOM has been updated')
})

// Stop a watcher
const stop = watchEffect(() => {
  console.log(count.value)
})
stop() // Stop watching
</script>
```

## Component Communication Patterns

### Provide / Inject

```vue
<!-- Parent.vue (Provider) -->
<script setup>
import { provide, ref } from 'vue'

const theme = ref('dark')
const updateTheme = (newTheme) => {
  theme.value = newTheme
}

// Provide values to descendants
provide('theme', theme)
provide('updateTheme', updateTheme)

// With injection key for type safety
import { InjectionKey, Ref } from 'vue'
export const themeKey = Symbol() as InjectionKey<Ref<string>>
provide(themeKey, theme)
</script>

<!-- Child.vue (Consumer) -->
<script setup>
import { inject } from 'vue'

// Inject provided values
const theme = inject('theme')
const updateTheme = inject('updateTheme')

// With default value
const config = inject('config', { defaultValue: true })

// With injection key
import { themeKey } from './Parent.vue'
const theme = inject(themeKey)
</script>

<template>
  <div :class="theme">
    <button @click="updateTheme('light')">Light Mode</button>
  </div>
</template>
```

### Template Refs

```vue
<script setup>
import { ref, onMounted } from 'vue'

// DOM element ref
const input = ref(null)
const list = ref([])

// Component ref
const childComponent = ref(null)

onMounted(() => {
  // Access DOM element
  input.value.focus()

  // Access component instance (only exposed properties)
  childComponent.value.someExposedMethod()
})

// Ref in v-for
function setItemRef(el) {
  if (el) {
    list.value.push(el)
  }
}
</script>

<template>
  <input ref="input" />
  <ChildComponent ref="childComponent" />

  <!-- Dynamic refs in v-for -->
  <div v-for="item in items" :key="item.id" :ref="setItemRef">
    {{ item.name }}
  </div>
</template>
```

### defineExpose

```vue
<!-- Child.vue -->
<script setup>
import { ref } from 'vue'

const count = ref(0)
const message = ref('Hello')

function increment() {
  count.value++
}

function reset() {
  count.value = 0
}

// Expose specific properties and methods to parent
defineExpose({
  count,
  increment,
  reset
  // message is NOT exposed
})
</script>

<!-- Parent.vue -->
<script setup>
import { ref } from 'vue'
import Child from './Child.vue'

const child = ref(null)

function callChildMethod() {
  child.value.increment()
  console.log(child.value.count) // Accessible
  // console.log(child.value.message) // undefined
}
</script>

<template>
  <Child ref="child" />
  <button @click="callChildMethod">Call Child Method</button>
</template>
```

### Custom v-model

```vue
<!-- CustomInput.vue -->
<script setup>
// Default v-model (modelValue prop, update:modelValue emit)
const props = defineProps(['modelValue'])
const emit = defineEmits(['update:modelValue'])

function updateValue(event) {
  emit('update:modelValue', event.target.value)
}
</script>

<template>
  <input
    :value="modelValue"
    @input="updateValue"
  />
</template>

<!-- Multiple v-models -->
<script setup>
defineProps(['firstName', 'lastName'])
defineEmits(['update:firstName', 'update:lastName'])
</script>

<template>
  <input
    :value="firstName"
    @input="$emit('update:firstName', $event.target.value)"
  />
  <input
    :value="lastName"
    @input="$emit('update:lastName', $event.target.value)"
  />
</template>

<!-- Usage -->
<CustomInput v-model="text" />
<CustomInput
  v-model:first-name="first"
  v-model:last-name="last"
/>

<!-- v-model with modifiers -->
<script setup>
const props = defineProps({
  modelValue: String,
  modelModifiers: { default: () => ({}) }
})

const emit = defineEmits(['update:modelValue'])

function emitValue(event) {
  let value = event.target.value
  if (props.modelModifiers.capitalize) {
    value = value.charAt(0).toUpperCase() + value.slice(1)
  }
  emit('update:modelValue', value)
}
</script>

<!-- Usage: <CustomInput v-model.capitalize="text" /> -->
```

### Slots and Scoped Slots

```vue
<!-- Card.vue -->
<template>
  <div class="card">
    <!-- Default slot -->
    <div class="card-header">
      <slot name="header">Default Header</slot>
    </div>

    <!-- Default slot -->
    <div class="card-body">
      <slot>Default Content</slot>
    </div>

    <!-- Scoped slot - passing data to parent -->
    <div class="card-footer">
      <slot name="footer" :date="new Date()" :version="1.0">
        Default Footer
      </slot>
    </div>
  </div>
</template>

<!-- Usage -->
<template>
  <Card>
    <template #header>
      <h1>Custom Header</h1>
    </template>

    <p>Custom content</p>

    <template #footer="{ date, version }">
      <p>Version {{ version }} - {{ date.toLocaleDateString() }}</p>
    </template>
  </Card>
</template>

<!-- List with scoped slots -->
<script setup>
defineProps(['items'])
</script>

<template>
  <ul>
    <li v-for="item in items" :key="item.id">
      <slot :item="item" :index="item.id">
        {{ item.name }}
      </slot>
    </li>
  </ul>
</template>

<!-- Usage -->
<List :items="users">
  <template #default="{ item, index }">
    <strong>{{ index }}:</strong> {{ item.name }} ({{ item.email }})
  </template>
</List>
```

## Advanced Component Patterns

### Dynamic Components

```vue
<script setup>
import { ref, shallowRef } from 'vue'
import ComponentA from './ComponentA.vue'
import ComponentB from './ComponentB.vue'
import ComponentC from './ComponentC.vue'

// Use shallowRef for component definitions (performance)
const currentComponent = shallowRef(ComponentA)

const tabs = {
  a: ComponentA,
  b: ComponentB,
  c: ComponentC
}

function switchComponent(key) {
  currentComponent.value = tabs[key]
}
</script>

<template>
  <div>
    <button @click="switchComponent('a')">Component A</button>
    <button @click="switchComponent('b')">Component B</button>
    <button @click="switchComponent('c')">Component C</button>

    <!-- Dynamic component -->
    <component :is="currentComponent" />

    <!-- With props and events -->
    <component
      :is="currentComponent"
      :some-prop="value"
      @some-event="handler"
    />
  </div>
</template>
```

### Async Components and Lazy Loading

```vue
<script setup>
import { defineAsyncComponent } from 'vue'

// Simple async component
const AsyncComponent = defineAsyncComponent(() =>
  import('./components/AsyncComponent.vue')
)

// With loading and error states
const AsyncComponentWithOptions = defineAsyncComponent({
  loader: () => import('./components/HeavyComponent.vue'),
  loadingComponent: LoadingSpinner,
  errorComponent: ErrorDisplay,
  delay: 200, // Delay before showing loading component
  timeout: 3000, // Timeout for loading
  suspensible: false,
  onError(error, retry, fail, attempts) {
    if (attempts <= 3) {
      retry()
    } else {
      fail()
    }
  }
})
</script>

<template>
  <AsyncComponent />
  <AsyncComponentWithOptions />
</template>
```

### Teleport (Portal)

```vue
<script setup>
import { ref } from 'vue'

const showModal = ref(false)
</script>

<template>
  <button @click="showModal = true">Open Modal</button>

  <!-- Teleport to body -->
  <Teleport to="body">
    <div v-if="showModal" class="modal">
      <div class="modal-content">
        <h2>Modal Title</h2>
        <p>Modal content</p>
        <button @click="showModal = false">Close</button>
      </div>
    </div>
  </Teleport>

  <!-- Teleport to specific element -->
  <Teleport to="#modals">
    <div>Teleported content</div>
  </Teleport>

  <!-- Conditional teleport -->
  <Teleport :disabled="isMobile" to="body">
    <div>Only teleported on desktop</div>
  </Teleport>
</template>
```

### KeepAlive (Component Caching)

```vue
<script setup>
import { ref } from 'vue'
import CompA from './CompA.vue'
import CompB from './CompB.vue'

const current = ref('CompA')
</script>

<template>
  <!-- Cache all components -->
  <KeepAlive>
    <component :is="current === 'CompA' ? CompA : CompB" />
  </KeepAlive>

  <!-- Cache specific components -->
  <KeepAlive include="CompA,CompB">
    <component :is="current" />
  </KeepAlive>

  <!-- Exclude specific components -->
  <KeepAlive exclude="CompC">
    <component :is="current" />
  </KeepAlive>

  <!-- With max cached instances -->
  <KeepAlive :max="10">
    <component :is="current" />
  </KeepAlive>
</template>

<!-- Component with KeepAlive lifecycle hooks -->
<script setup>
import { onActivated, onDeactivated } from 'vue'

onActivated(() => {
  console.log('Component activated (from cache)')
})

onDeactivated(() => {
  console.log('Component deactivated (cached)')
})
</script>
```

### Suspense (Async Boundaries)

```vue
<template>
  <Suspense>
    <!-- Component with async setup -->
    <template #default>
      <AsyncComponent />
    </template>

    <!-- Loading state -->
    <template #fallback>
      <div>Loading...</div>
    </template>
  </Suspense>
</template>

<!-- AsyncComponent.vue -->
<script setup>
// Top-level await in script setup
const data = await fetch('/api/data').then(r => r.json())
</script>

<template>
  <div>{{ data }}</div>
</template>

<!-- Multiple async components -->
<template>
  <Suspense>
    <div>
      <AsyncComponentA />
      <AsyncComponentB />
      <!-- Both must resolve before showing -->
    </div>

    <template #fallback>
      <LoadingSpinner />
    </template>
  </Suspense>
</template>

<!-- Error handling with Suspense -->
<script setup>
import { onErrorCaptured, ref } from 'vue'

const error = ref(null)

onErrorCaptured((err) => {
  error.value = err
  return false // Prevent error from propagating
})
</script>

<template>
  <div v-if="error">
    Error: {{ error.message }}
  </div>
  <Suspense v-else>
    <AsyncComponent />
    <template #fallback>Loading...</template>
  </Suspense>
</template>
```

## Form Handling & Validation

### Complex Form Patterns

```vue
<script setup>
import { reactive, computed } from 'vue'

const form = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
  acceptTerms: false
})

const errors = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: ''
})

// Validation rules
const rules = {
  username: (value) => {
    if (!value) return 'Username is required'
    if (value.length < 3) return 'Username must be at least 3 characters'
    return ''
  },
  email: (value) => {
    if (!value) return 'Email is required'
    if (!/^\S+@\S+\.\S+$/.test(value)) return 'Email is invalid'
    return ''
  },
  password: (value) => {
    if (!value) return 'Password is required'
    if (value.length < 8) return 'Password must be at least 8 characters'
    return ''
  },
  confirmPassword: (value) => {
    if (value !== form.password) return 'Passwords do not match'
    return ''
  }
}

function validateField(field) {
  errors[field] = rules[field](form[field])
}

function validateAll() {
  Object.keys(rules).forEach(validateField)
  return !Object.values(errors).some(error => error)
}

const isValid = computed(() => {
  return form.username &&
         form.email &&
         form.password &&
         form.password === form.confirmPassword &&
         form.acceptTerms
})

async function handleSubmit() {
  if (!validateAll()) {
    return
  }

  try {
    await submitForm(form)
  } catch (error) {
    console.error('Submit failed:', error)
  }
}
</script>

<template>
  <form @submit.prevent="handleSubmit">
    <div>
      <label>Username</label>
      <input
        v-model="form.username"
        @blur="validateField('username')"
        :class="{ error: errors.username }"
      />
      <span class="error-message">{{ errors.username }}</span>
    </div>

    <div>
      <label>Email</label>
      <input
        v-model="form.email"
        type="email"
        @blur="validateField('email')"
        :class="{ error: errors.email }"
      />
      <span class="error-message">{{ errors.email }}</span>
    </div>

    <div>
      <label>Password</label>
      <input
        v-model="form.password"
        type="password"
        @blur="validateField('password')"
        :class="{ error: errors.password }"
      />
      <span class="error-message">{{ errors.password }}</span>
    </div>

    <div>
      <label>Confirm Password</label>
      <input
        v-model="form.confirmPassword"
        type="password"
        @blur="validateField('confirmPassword')"
        :class="{ error: errors.confirmPassword }"
      />
      <span class="error-message">{{ errors.confirmPassword }}</span>
    </div>

    <div>
      <label>
        <input type="checkbox" v-model="form.acceptTerms" />
        Accept Terms and Conditions
      </label>
    </div>

    <button type="submit" :disabled="!isValid">
      Submit
    </button>
  </form>
</template>
```

### v-model Modifiers

```vue
<template>
  <!-- .lazy - update on change instead of input -->
  <input v-model.lazy="text" />

  <!-- .number - convert to number -->
  <input v-model.number="age" type="number" />

  <!-- .trim - trim whitespace -->
  <input v-model.trim="message" />

  <!-- Multiple modifiers -->
  <input v-model.lazy.trim="username" />
</template>
```

### Debounced Input

```vue
<script setup>
import { ref, watch } from 'vue'

const searchQuery = ref('')
const debouncedQuery = ref('')

// Debounce function
function debounce(fn, delay) {
  let timeoutId
  return (...args) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

const updateDebounced = debounce((value) => {
  debouncedQuery.value = value
}, 500)

watch(searchQuery, (newValue) => {
  updateDebounced(newValue)
})

// Or as a composable
import { customRef } from 'vue'

function useDebouncedRef(value, delay = 300) {
  return customRef((track, trigger) => {
    let timeout
    return {
      get() {
        track()
        return value
      },
      set(newValue) {
        clearTimeout(timeout)
        timeout = setTimeout(() => {
          value = newValue
          trigger()
        }, delay)
      }
    }
  })
}

const debouncedSearch = useDebouncedRef('', 500)
</script>

<template>
  <input v-model="searchQuery" placeholder="Search..." />
  <p>Debounced: {{ debouncedQuery }}</p>
</template>
```

## Vue Router

### Router Setup

```javascript
// router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: Home
  },
  {
    path: '/about',
    name: 'about',
    // Lazy-loaded route
    component: () => import('../views/About.vue')
  },
  {
    path: '/user/:id',
    name: 'user',
    component: () => import('../views/User.vue'),
    props: true // Pass route params as props
  },
  {
    path: '/posts/:id',
    component: () => import('../views/Post.vue'),
    // Route meta fields
    meta: { requiresAuth: true }
  },
  {
    // Nested routes
    path: '/dashboard',
    component: () => import('../views/Dashboard.vue'),
    children: [
      {
        path: '',
        component: () => import('../views/DashboardHome.vue')
      },
      {
        path: 'profile',
        component: () => import('../views/Profile.vue')
      },
      {
        path: 'settings',
        component: () => import('../views/Settings.vue')
      }
    ]
  },
  {
    // 404 catch all
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('../views/NotFound.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    } else {
      return { top: 0 }
    }
  }
})

export default router
```

### Navigation and Route Access

```vue
<script setup>
import { useRouter, useRoute } from 'vue-router'
import { computed } from 'vue'

const router = useRouter()
const route = useRoute()

// Access route params and query
const userId = computed(() => route.params.id)
const page = computed(() => route.query.page || 1)

// Programmatic navigation
function goToHome() {
  router.push('/')
}

function goToUser(id) {
  router.push({ name: 'user', params: { id } })
}

function goToUserWithQuery(id) {
  router.push({
    path: `/user/${id}`,
    query: { tab: 'posts', page: 1 }
  })
}

function goBack() {
  router.back()
}

function goForward() {
  router.forward()
}

// Replace (no history entry)
function replaceRoute() {
  router.replace('/new-location')
}
</script>

<template>
  <div>
    <!-- Declarative navigation -->
    <router-link to="/">Home</router-link>
    <router-link :to="{ name: 'user', params: { id: 123 } }">
      User 123
    </router-link>
    <router-link to="/about" active-class="active" exact>
      About
    </router-link>

    <!-- Current route info -->
    <p>Current path: {{ route.path }}</p>
    <p>User ID: {{ userId }}</p>
    <p>Page: {{ page }}</p>

    <!-- Programmatic navigation -->
    <button @click="goToHome">Go Home</button>
    <button @click="goBack">Go Back</button>

    <!-- Router view -->
    <router-view />

    <!-- Named views -->
    <router-view name="sidebar" />
    <router-view name="main" />
  </div>
</template>
```

### Navigation Guards

```javascript
// Global guards (in router/index.js)
router.beforeEach((to, from, next) => {
  // Check authentication
  if (to.meta.requiresAuth && !isAuthenticated()) {
    next({ name: 'login', query: { redirect: to.fullPath } })
  } else {
    next()
  }
})

router.afterEach((to, from) => {
  // Analytics, page title, etc.
  document.title = to.meta.title || 'Default Title'
})

// Per-route guards
const routes = [
  {
    path: '/admin',
    component: Admin,
    beforeEnter: (to, from, next) => {
      if (!isAdmin()) {
        next({ name: 'home' })
      } else {
        next()
      }
    }
  }
]

// Component guards
<script setup>
import { onBeforeRouteLeave, onBeforeRouteUpdate } from 'vue-router'

onBeforeRouteLeave((to, from) => {
  if (hasUnsavedChanges()) {
    const answer = window.confirm('You have unsaved changes. Leave anyway?')
    if (!answer) return false
  }
})

onBeforeRouteUpdate(async (to, from) => {
  // React to route changes on the same component
  if (to.params.id !== from.params.id) {
    await loadUser(to.params.id)
  }
})
</script>
```

## State Management

### Pinia Store

```javascript
// stores/counter.js
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Option syntax
export const useCounterStore = defineStore('counter', {
  state: () => ({
    count: 0,
    name: 'Counter'
  }),

  getters: {
    doubled: (state) => state.count * 2,
    doubledPlusOne() {
      return this.doubled + 1
    }
  },

  actions: {
    increment() {
      this.count++
    },
    async fetchData() {
      const data = await fetch('/api/data').then(r => r.json())
      this.count = data.count
    }
  }
})

// Composition API syntax (recommended)
export const useCounterStore = defineStore('counter', () => {
  const count = ref(0)
  const name = ref('Counter')

  const doubled = computed(() => count.value * 2)

  function increment() {
    count.value++
  }

  async function fetchData() {
    const data = await fetch('/api/data').then(r => r.json())
    count.value = data.count
  }

  return {
    count,
    name,
    doubled,
    increment,
    fetchData
  }
})
```

### Using Stores

```vue
<script setup>
import { useCounterStore } from '@/stores/counter'
import { storeToRefs } from 'pinia'

const counterStore = useCounterStore()

// Destructure actions (works directly)
const { increment, fetchData } = counterStore

// Destructure state (needs storeToRefs to maintain reactivity)
const { count, doubled } = storeToRefs(counterStore)

// Or use store directly
// counterStore.count
// counterStore.increment()
</script>

<template>
  <div>
    <p>Count: {{ count }}</p>
    <p>Doubled: {{ doubled }}</p>
    <button @click="increment">Increment</button>
    <button @click="fetchData">Fetch Data</button>
  </div>
</template>
```

### Shared State with Composables

```javascript
// composables/useSharedState.js
import { ref, readonly } from 'vue'

// Shared state (singleton)
const count = ref(0)
const isLoading = ref(false)

export function useSharedState() {
  function increment() {
    count.value++
  }

  function decrement() {
    count.value--
  }

  async function loadData() {
    isLoading.value = true
    try {
      // Fetch data
      await new Promise(resolve => setTimeout(resolve, 1000))
    } finally {
      isLoading.value = false
    }
  }

  return {
    count: readonly(count), // Expose as readonly
    isLoading: readonly(isLoading),
    increment,
    decrement,
    loadData
  }
}

// Usage in multiple components
<script setup>
import { useSharedState } from '@/composables/useSharedState'

const { count, increment } = useSharedState()
</script>
```

## Performance Optimization

### Component Lazy Loading

```javascript
// Lazy load in router
const routes = [
  {
    path: '/dashboard',
    component: () => import('./views/Dashboard.vue')
  }
]

// Lazy load component
<script setup>
import { defineAsyncComponent } from 'vue'

const HeavyComponent = defineAsyncComponent(() =>
  import('./components/HeavyComponent.vue')
)
</script>

// Webpack magic comments for chunk naming
const Dashboard = () => import(
  /* webpackChunkName: "dashboard" */
  './views/Dashboard.vue'
)
```

### Computed vs Methods vs Watchers

```vue
<script setup>
import { ref, computed, watch } from 'vue'

const count = ref(0)
const multiplier = ref(2)

// ✅ GOOD: Use computed for derived values (cached, reactive)
const doubled = computed(() => count.value * multiplier.value)

// ❌ BAD: Don't use methods for derived values (recalculated every render)
function getDoubled() {
  return count.value * multiplier.value
}

// ✅ GOOD: Use watchers for side effects
watch(count, (newValue, oldValue) => {
  console.log('Count changed:', newValue)
  // Side effects: API calls, DOM manipulation, etc.
})

// ❌ BAD: Don't use computed for side effects
const badComputed = computed(() => {
  console.log('This runs too often!')
  return count.value * 2
})
</script>

<template>
  <!-- ✅ Computed (cached) -->
  <p>{{ doubled }}</p>

  <!-- ❌ Method (recalculated every render) -->
  <p>{{ getDoubled() }}</p>
</template>
```

### v-memo and v-once

```vue
<template>
  <!-- v-once: render once, never update -->
  <div v-once>
    <h1>{{ title }}</h1>
    <p>This content never changes</p>
  </div>

  <!-- v-memo: conditional caching (Vue 3.2+) -->
  <div v-for="item in list" :key="item.id" v-memo="[item.id, item.selected]">
    <!-- Only re-render if item.id or item.selected changes -->
    <p>{{ item.name }}</p>
    <p>{{ item.description }}</p>
  </div>

  <!-- Without v-memo, entire item re-renders on any change -->
  <!-- With v-memo, only re-renders when dependencies change -->
</template>
```

### Virtual Scrolling Pattern

```vue
<script setup>
import { ref, computed } from 'vue'

const items = ref([/* thousands of items */])
const containerHeight = ref(600)
const itemHeight = 50
const scrollTop = ref(0)

const visibleStart = computed(() =>
  Math.floor(scrollTop.value / itemHeight)
)

const visibleEnd = computed(() =>
  Math.ceil((scrollTop.value + containerHeight.value) / itemHeight)
)

const visibleItems = computed(() =>
  items.value.slice(visibleStart.value, visibleEnd.value)
)

const totalHeight = computed(() =>
  items.value.length * itemHeight
)

const offsetY = computed(() =>
  visibleStart.value * itemHeight
)

function handleScroll(event) {
  scrollTop.value = event.target.scrollTop
}
</script>

<template>
  <div
    class="virtual-scroll-container"
    :style="{ height: containerHeight + 'px' }"
    @scroll="handleScroll"
  >
    <div :style="{ height: totalHeight + 'px', position: 'relative' }">
      <div
        :style="{ transform: `translateY(${offsetY}px)` }"
      >
        <div
          v-for="item in visibleItems"
          :key="item.id"
          :style="{ height: itemHeight + 'px' }"
        >
          {{ item.name }}
        </div>
      </div>
    </div>
  </div>
</template>
```

### Production Optimization

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    // Enable minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
      }
    },
    // Code splitting
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['vue', 'vue-router', 'pinia'],
          'ui': ['./src/components/ui']
        }
      }
    },
    // Chunk size warnings
    chunkSizeWarningLimit: 500
  }
})
```

## TypeScript Integration

### Typed Props and Emits

```vue
<script setup lang="ts">
import { ref } from 'vue'

// Define props with TypeScript
interface Props {
  title: string
  count?: number
  items: Array<{ id: number; name: string }>
  callback?: (value: number) => void
}

const props = withDefaults(defineProps<Props>(), {
  count: 0,
  callback: () => {}
})

// Define emits with TypeScript
interface Emits {
  (e: 'update', value: number): void
  (e: 'delete', id: number): void
  (e: 'submit', data: { name: string; email: string }): void
}

const emit = defineEmits<Emits>()

// Or inline
const emit = defineEmits<{
  update: [value: number]
  delete: [id: number]
}>()

function handleClick() {
  emit('update', props.count + 1)
}
</script>
```

### Typed Refs and Reactive

```vue
<script setup lang="ts">
import { ref, reactive, computed } from 'vue'

// Typed ref
const count = ref<number>(0)
const name = ref<string>('John')

// Typed ref with interface
interface User {
  id: number
  name: string
  email: string
}

const user = ref<User>({
  id: 1,
  name: 'John',
  email: 'john@example.com'
})

// Typed reactive
const state = reactive<{
  count: number
  name: string
}>({
  count: 0,
  name: 'John'
})

// Typed computed
const doubled = computed<number>(() => count.value * 2)

// Typed template ref
import { ComponentPublicInstance } from 'vue'
import ChildComponent from './ChildComponent.vue'

const child = ref<ComponentPublicInstance<typeof ChildComponent>>()
</script>
```

### Typed Composables

```typescript
// composables/useFetch.ts
import { ref, Ref } from 'vue'

interface UseFetchReturn<T> {
  data: Ref<T | null>
  error: Ref<Error | null>
  loading: Ref<boolean>
  refetch: () => Promise<void>
}

export function useFetch<T>(url: string): UseFetchReturn<T> {
  const data = ref<T | null>(null)
  const error = ref<Error | null>(null)
  const loading = ref<boolean>(false)

  async function fetchData() {
    loading.value = true
    error.value = null

    try {
      const response = await fetch(url)
      if (!response.ok) throw new Error('Network error')
      data.value = await response.json()
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  fetchData()

  return {
    data,
    error,
    loading,
    refetch: fetchData
  }
}

// Usage
interface User {
  id: number
  name: string
  email: string
}

const { data, error, loading } = useFetch<User[]>('/api/users')
```

### Generic Components

```vue
<script setup lang="ts" generic="T extends { id: number }">
import { computed } from 'vue'

interface Props {
  items: T[]
  selectedId?: number
}

const props = defineProps<Props>()

const emit = defineEmits<{
  select: [item: T]
}>()

const selectedItem = computed(() =>
  props.items.find(item => item.id === props.selectedId)
)
</script>

<template>
  <div>
    <div
      v-for="item in items"
      :key="item.id"
      @click="emit('select', item)"
    >
      <slot :item="item" />
    </div>
  </div>
</template>

<!-- Usage -->
<GenericList
  :items="users"
  @select="handleSelect"
>
  <template #default="{ item }">
    {{ item.name }}
  </template>
</GenericList>
```

## Common Utility Patterns

### Async Data Fetching with Loading States

```vue
<script setup>
import { ref, onMounted } from 'vue'

const data = ref(null)
const loading = ref(false)
const error = ref(null)

async function fetchData() {
  loading.value = true
  error.value = null

  try {
    const response = await fetch('/api/data')
    if (!response.ok) throw new Error('Failed to fetch')
    data.value = await response.json()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div>
    <div v-if="loading">Loading...</div>
    <div v-else-if="error">Error: {{ error }}</div>
    <div v-else-if="data">
      <!-- Display data -->
      <pre>{{ data }}</pre>
    </div>
    <button @click="fetchData">Retry</button>
  </div>
</template>
```

### Error Boundary Pattern

```vue
<script setup>
import { ref, onErrorCaptured } from 'vue'

const error = ref(null)

onErrorCaptured((err, instance, info) => {
  error.value = err
  console.error('Error captured:', err, info)
  // Return false to prevent propagation
  return false
})

function resetError() {
  error.value = null
}
</script>

<template>
  <div>
    <div v-if="error" class="error-boundary">
      <h2>Something went wrong</h2>
      <p>{{ error.message }}</p>
      <button @click="resetError">Try Again</button>
    </div>
    <slot v-else />
  </div>
</template>
```

### Conditional Classes and Styles

```vue
<script setup>
import { ref, computed } from 'vue'

const isActive = ref(true)
const hasError = ref(false)
const type = ref('primary')

const buttonClasses = computed(() => ({
  active: isActive.value,
  error: hasError.value,
  [`btn-${type.value}`]: true
}))

const dynamicStyles = computed(() => ({
  color: isActive.value ? 'blue' : 'gray',
  fontSize: '14px'
}))
</script>

<template>
  <!-- Class binding -->
  <div :class="{ active: isActive, error: hasError }">Basic</div>

  <!-- Array syntax -->
  <div :class="['btn', type, { active: isActive }]">Array</div>

  <!-- Computed classes -->
  <button :class="buttonClasses">Button</button>

  <!-- Style binding -->
  <div :style="{ color: 'red', fontSize: '14px' }">Inline</div>
  <div :style="dynamicStyles">Dynamic</div>

  <!-- Multiple style objects -->
  <div :style="[baseStyles, overrideStyles]">Multiple</div>
</template>
```

### Debounce and Throttle

```javascript
// utils/timing.js

// Debounce: wait for pause in calls
export function debounce(fn, delay) {
  let timeoutId
  return function (...args) {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn.apply(this, args), delay)
  }
}

// Throttle: limit call frequency
export function throttle(fn, limit) {
  let inThrottle
  return function (...args) {
    if (!inThrottle) {
      fn.apply(this, args)
      inThrottle = true
      setTimeout(() => inThrottle = false, limit)
    }
  }
}

// Usage
<script setup>
import { ref } from 'vue'
import { debounce, throttle } from '@/utils/timing'

const searchQuery = ref('')

const debouncedSearch = debounce((query) => {
  console.log('Searching for:', query)
  // API call here
}, 500)

const throttledScroll = throttle(() => {
  console.log('Scroll event')
}, 1000)

function handleInput(event) {
  searchQuery.value = event.target.value
  debouncedSearch(event.target.value)
}
</script>

<template>
  <input @input="handleInput" />
  <div @scroll="throttledScroll">Scrollable content</div>
</template>
```

### Intersection Observer (Lazy Loading)

```vue
<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const target = ref(null)
const isVisible = ref(false)

let observer

onMounted(() => {
  observer = new IntersectionObserver(
    ([entry]) => {
      isVisible.value = entry.isIntersecting

      // Load once and disconnect
      if (entry.isIntersecting) {
        loadContent()
        observer.disconnect()
      }
    },
    {
      threshold: 0.1,
      rootMargin: '50px'
    }
  )

  if (target.value) {
    observer.observe(target.value)
  }
})

onUnmounted(() => {
  if (observer) {
    observer.disconnect()
  }
})

function loadContent() {
  console.log('Loading content...')
}
</script>

<template>
  <div ref="target">
    <div v-if="isVisible">
      <!-- Lazy loaded content -->
      <img src="large-image.jpg" />
    </div>
    <div v-else>
      Loading...
    </div>
  </div>
</template>
```

## Testing Patterns

### Component Testing with Vitest

```javascript
// MyComponent.spec.js
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import MyComponent from './MyComponent.vue'

describe('MyComponent', () => {
  it('renders properly', () => {
    const wrapper = mount(MyComponent, {
      props: {
        title: 'Hello'
      }
    })

    expect(wrapper.text()).toContain('Hello')
  })

  it('emits update event when button clicked', async () => {
    const wrapper = mount(MyComponent)

    await wrapper.find('button').trigger('click')

    expect(wrapper.emitted()).toHaveProperty('update')
    expect(wrapper.emitted('update')[0]).toEqual([1])
  })

  it('updates count when increment is called', async () => {
    const wrapper = mount(MyComponent)

    expect(wrapper.vm.count).toBe(0)

    await wrapper.vm.increment()

    expect(wrapper.vm.count).toBe(1)
    expect(wrapper.html()).toContain('1')
  })

  it('handles async data fetching', async () => {
    // Mock fetch
    global.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ data: 'test' })
      })
    )

    const wrapper = mount(MyComponent)

    // Wait for async operations
    await wrapper.vm.$nextTick()
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(wrapper.vm.data).toEqual({ data: 'test' })
  })
})
```

### Testing Composables

```javascript
// useCounter.spec.js
import { describe, it, expect } from 'vitest'
import { useCounter } from './useCounter'

describe('useCounter', () => {
  it('initializes with default value', () => {
    const { count } = useCounter()
    expect(count.value).toBe(0)
  })

  it('initializes with custom value', () => {
    const { count } = useCounter(10)
    expect(count.value).toBe(10)
  })

  it('increments count', () => {
    const { count, increment } = useCounter()
    increment()
    expect(count.value).toBe(1)
  })

  it('computes doubled value', () => {
    const { count, doubled, increment } = useCounter()
    expect(doubled.value).toBe(0)
    increment()
    expect(doubled.value).toBe(2)
  })
})
```

### Mocking Composables and Stores

```javascript
// Component.spec.js
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import MyComponent from './MyComponent.vue'
import { useUserStore } from '@/stores/user'

// Mock composable
vi.mock('@/composables/useFetch', () => ({
  useFetch: vi.fn(() => ({
    data: { value: { name: 'Test' } },
    loading: { value: false },
    error: { value: null }
  }))
}))

describe('MyComponent with mocks', () => {
  it('uses mocked composable', () => {
    const wrapper = mount(MyComponent)
    expect(wrapper.text()).toContain('Test')
  })

  it('works with pinia store', () => {
    setActivePinia(createPinia())

    const store = useUserStore()
    store.name = 'John'

    const wrapper = mount(MyComponent, {
      global: {
        plugins: [createPinia()]
      }
    })

    expect(wrapper.text()).toContain('John')
  })
})
```

## Build & Tooling

### Vite Configuration

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],

  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@utils': resolve(__dirname, 'src/utils')
    }
  },

  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },

  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia']
        }
      }
    }
  },

  css: {
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/variables.scss";`
      }
    }
  }
})
```

### Environment Variables

```javascript
// .env
VITE_API_URL=https://api.example.com
VITE_APP_TITLE=My App

// .env.development
VITE_API_URL=http://localhost:3000

// .env.production
VITE_API_URL=https://api.production.com

// Usage in code
<script setup>
const apiUrl = import.meta.env.VITE_API_URL
const appTitle = import.meta.env.VITE_APP_TITLE
const isDev = import.meta.env.DEV
const isProd = import.meta.env.PROD

console.log('API URL:', apiUrl)
</script>

// Type definitions (env.d.ts)
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_APP_TITLE: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
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
| Ref | `ref(value)` |
| Reactive | `reactive({})` |
| Computed | `computed(() => value)` |
| Watch | `watch(source, callback)` |
| Lifecycle | `onMounted(() => {})` |
| Template Ref | `ref(null)` + `ref="name"` |
| Provide | `provide('key', value)` |
| Inject | `inject('key')` |
| Slot | `<slot name="header" />` |

Vue.js provides an approachable, versatile, and performant framework for building modern web interfaces with comprehensive tooling for state management, routing, testing, and production optimization.
