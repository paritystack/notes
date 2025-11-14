# Svelte

Svelte is a radical new approach to building user interfaces. Unlike frameworks that do the bulk of their work in the browser, Svelte shifts that work into a compile step.

## Installation

```bash
# Create new Svelte project
npm create vite@latest my-app -- --template svelte
cd my-app
npm install
npm run dev
```

## Component Basics

```svelte
<!-- App.svelte -->
<script>
  let count = 0;

  function increment() {
    count += 1;
  }
</script>

<button on:click={increment}>
  Clicked {count} {count === 1 ? 'time' : 'times'}
</button>

<style>
  button {
    background: #ff3e00;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
  }
</style>
```

## Reactivity

```svelte
<script>
  let count = 0;

  // Reactive declaration
  $: doubled = count * 2;

  // Reactive statement
  $: if (count >= 10) {
    alert('count is high!');
  }

  // Reactive block
  $: {
    console.log(`count is ${count}`);
  }
</script>
```

## Props

```svelte
<!-- Child.svelte -->
<script>
  export let name;
  export let age = 25; // default value
</script>

<p>{name} is {age} years old</p>

<!-- Parent.svelte -->
<script>
  import Child from './Child.svelte';
</script>

<Child name="John" age={30} />
```

## Events

```svelte
<script>
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  function handleClick() {
    dispatch('message', { text: 'Hello!' });
  }
</script>

<button on:click={handleClick}>
  Send message
</button>

<!-- Parent -->
<Child on:message={e => console.log(e.detail.text)} />
```

## Stores

```javascript
// store.js
import { writable } from 'svelte/store';

export const count = writable(0);
```

```svelte
<script>
  import { count } from './store.js';
</script>

<button on:click={() => $count += 1}>
  Count: {$count}
</button>
```

## Component Lifecycle

### Lifecycle Hooks

```svelte
<script>
  import { onMount, onDestroy, beforeUpdate, afterUpdate, tick } from 'svelte';

  // Runs after component is first rendered to DOM
  onMount(() => {
    console.log('Component mounted');

    // Return cleanup function
    return () => {
      console.log('onMount cleanup');
    };
  });

  // Runs before component is destroyed
  onDestroy(() => {
    console.log('Component destroyed');
  });

  // Runs before DOM is updated
  beforeUpdate(() => {
    console.log('Before update');
  });

  // Runs after DOM is updated
  afterUpdate(() => {
    console.log('After update');
  });

  // Example: scroll to bottom after update
  let messages = [];
  let container;

  async function addMessage(text) {
    messages = [...messages, text];
    await tick(); // Wait for DOM to update
    container.scrollTop = container.scrollHeight;
  }
</script>

<div bind:this={container}>
  {#each messages as message}
    <p>{message}</p>
  {/each}
</div>
```

### Async onMount Pattern

```svelte
<script>
  import { onMount } from 'svelte';

  let data = null;
  let loading = true;
  let error = null;

  onMount(async () => {
    try {
      const response = await fetch('/api/data');
      if (!response.ok) throw new Error('Failed to fetch');
      data = await response.json();
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  });
</script>

{#if loading}
  <p>Loading...</p>
{:else if error}
  <p>Error: {error}</p>
{:else}
  <pre>{JSON.stringify(data, null, 2)}</pre>
{/if}
```

## Advanced Reactivity

### Reactive Declarations

```svelte
<script>
  let firstName = 'John';
  let lastName = 'Doe';

  // Reactive declaration - automatically updates
  $: fullName = `${firstName} ${lastName}`;

  // Multiple dependencies
  let width = 100;
  let height = 100;
  $: area = width * height;
  $: perimeter = 2 * (width + height);

  // Reactive statements with side effects
  $: {
    console.log(`Area: ${area}`);
    console.log(`Perimeter: ${perimeter}`);
  }

  // Conditional reactive statements
  $: if (area > 10000) {
    console.warn('Area is very large!');
  }
</script>
```

### Reactive Arrays and Objects

```svelte
<script>
  let items = [1, 2, 3];

  // ❌ This won't trigger reactivity
  function addWrong() {
    items.push(4);
  }

  // ✅ Correct - create new array
  function addCorrect() {
    items = [...items, 4];
  }

  // ✅ Alternative - reassign
  function addAlternative() {
    items.push(4);
    items = items;
  }

  let user = { name: 'John', age: 30 };

  // ❌ Won't trigger reactivity
  function updateWrong() {
    user.age = 31;
  }

  // ✅ Create new object
  function updateCorrect() {
    user = { ...user, age: 31 };
  }
</script>
```

### Complex Reactive Chains

```svelte
<script>
  let numbers = [1, 2, 3, 4, 5];

  // Chain of reactive declarations
  $: doubled = numbers.map(n => n * 2);
  $: filtered = doubled.filter(n => n > 5);
  $: sum = filtered.reduce((a, b) => a + b, 0);
  $: average = filtered.length ? sum / filtered.length : 0;

  // Reactive with async
  let query = '';
  let results = [];

  $: if (query.length > 2) {
    searchAPI(query);
  }

  async function searchAPI(q) {
    const response = await fetch(`/api/search?q=${q}`);
    results = await response.json();
  }
</script>
```

## Bindings

### Form Input Bindings

```svelte
<script>
  let text = '';
  let number = 0;
  let checked = false;
  let selected = '';
  let group = [];
  let value = 50;
</script>

<!-- Text input -->
<input type="text" bind:value={text}>
<p>Text: {text}</p>

<!-- Number input -->
<input type="number" bind:value={number}>
<p>Number: {number}</p>

<!-- Checkbox -->
<input type="checkbox" bind:checked={checked}>
<p>Checked: {checked}</p>

<!-- Select -->
<select bind:value={selected}>
  <option value="a">Option A</option>
  <option value="b">Option B</option>
  <option value="c">Option C</option>
</select>

<!-- Radio group -->
<input type="radio" bind:group={group} value="one">
<input type="radio" bind:group={group} value="two">
<input type="radio" bind:group={group} value="three">

<!-- Range -->
<input type="range" bind:value={value} min="0" max="100">
<p>Value: {value}</p>

<!-- Textarea -->
<textarea bind:value={text}></textarea>
```

### Element Bindings

```svelte
<script>
  let div;
  let input;
  let canvas;

  $: if (div) {
    console.log('Div dimensions:', div.offsetWidth, div.offsetHeight);
  }

  function focusInput() {
    input.focus();
  }

  function getCanvasContext() {
    const ctx = canvas.getContext('2d');
    ctx.fillRect(0, 0, 100, 100);
  }

  // Bind to dimensions
  let w;
  let h;
</script>

<div bind:this={div} bind:clientWidth={w} bind:clientHeight={h}>
  Content here - {w}x{h}
</div>

<input bind:this={input} type="text">
<button on:click={focusInput}>Focus Input</button>

<canvas bind:this={canvas} width="200" height="200"></canvas>
<button on:click={getCanvasContext}>Draw</button>
```

### Component Bindings

```svelte
<!-- Child.svelte -->
<script>
  export let value = '';
</script>

<input type="text" bind:value={value}>

<!-- Parent.svelte -->
<script>
  import Child from './Child.svelte';
  let childValue = '';
</script>

<Child bind:value={childValue} />
<p>Parent sees: {childValue}</p>
```

### Contenteditable Bindings

```svelte
<script>
  let html = '<p>Edit me!</p>';
  let text = 'Plain text';
</script>

<div contenteditable="true" bind:innerHTML={html}></div>
<div contenteditable="true" bind:textContent={text}></div>
```

## Slots

### Default Slots

```svelte
<!-- Card.svelte -->
<div class="card">
  <slot>
    <!-- Fallback content if no slot provided -->
    <p>No content provided</p>
  </slot>
</div>

<!-- Usage -->
<Card>
  <h2>My Title</h2>
  <p>My content</p>
</Card>
```

### Named Slots

```svelte
<!-- Layout.svelte -->
<div class="layout">
  <header>
    <slot name="header">
      <h1>Default Header</h1>
    </slot>
  </header>

  <main>
    <slot></slot>
  </main>

  <footer>
    <slot name="footer">
      <p>Default Footer</p>
    </slot>
  </footer>
</div>

<!-- Usage -->
<Layout>
  <svelte:fragment slot="header">
    <h1>Custom Header</h1>
  </svelte:fragment>

  <p>Main content goes here</p>

  <svelte:fragment slot="footer">
    <p>Custom Footer</p>
  </svelte:fragment>
</Layout>
```

### Slot Props (Scoped Slots)

```svelte
<!-- List.svelte -->
<script>
  export let items = [];
</script>

<ul>
  {#each items as item, index}
    <li>
      <slot {item} {index}>
        <!-- Fallback -->
        {item}
      </slot>
    </li>
  {/each}
</ul>

<!-- Usage -->
<script>
  import List from './List.svelte';
  const items = ['Apple', 'Banana', 'Cherry'];
</script>

<List {items} let:item let:index>
  <strong>{index + 1}:</strong> {item}
</List>
```

### Advanced Slot Pattern

```svelte
<!-- DataTable.svelte -->
<script>
  export let data = [];
  export let columns = [];
</script>

<table>
  <thead>
    <tr>
      {#each columns as column}
        <th>
          <slot name="header" {column}>
            {column.label}
          </slot>
        </th>
      {/each}
    </tr>
  </thead>
  <tbody>
    {#each data as row, rowIndex}
      <tr>
        {#each columns as column}
          <td>
            <slot name="cell" {row} {column} {rowIndex}>
              {row[column.key]}
            </slot>
          </td>
        {/each}
      </tr>
    {/each}
  </tbody>
</table>

<!-- Usage -->
<DataTable {data} {columns}>
  <svelte:fragment slot="header" let:column>
    <strong>{column.label.toUpperCase()}</strong>
  </svelte:fragment>

  <svelte:fragment slot="cell" let:row let:column>
    {#if column.key === 'email'}
      <a href="mailto:{row.email}">{row.email}</a>
    {:else}
      {row[column.key]}
    {/if}
  </svelte:fragment>
</DataTable>
```

## Context API

### Basic Context Usage

```svelte
<!-- Parent.svelte -->
<script>
  import { setContext } from 'svelte';
  import Child from './Child.svelte';

  const theme = {
    primary: '#ff3e00',
    secondary: '#676778'
  };

  setContext('theme', theme);

  // Can also set functions
  setContext('api', {
    fetchUser: async (id) => {
      const res = await fetch(`/api/users/${id}`);
      return res.json();
    }
  });
</script>

<Child />

<!-- Child.svelte -->
<script>
  import { getContext } from 'svelte';

  const theme = getContext('theme');
  const api = getContext('api');

  let user;
  $: api.fetchUser(1).then(u => user = u);
</script>

<div style="color: {theme.primary}">
  {#if user}
    <p>{user.name}</p>
  {/if}
</div>
```

### Context with Stores

```svelte
<!-- App.svelte -->
<script>
  import { setContext } from 'svelte';
  import { writable } from 'svelte/store';

  const user = writable({ name: 'John', isAdmin: false });

  setContext('user', user);
</script>

<!-- AnyChildComponent.svelte -->
<script>
  import { getContext } from 'svelte';

  const user = getContext('user');
</script>

<p>Hello, {$user.name}!</p>
{#if $user.isAdmin}
  <button>Admin Panel</button>
{/if}

<button on:click={() => $user.name = 'Jane'}>
  Change Name
</button>
```

### Context Module Pattern

```javascript
// context.js
import { getContext, setContext } from 'svelte';
import { writable } from 'svelte/store';

const CONTEXT_KEY = 'myApp';

export function createAppContext() {
  const state = writable({
    user: null,
    theme: 'light'
  });

  const api = {
    setUser: (user) => state.update(s => ({ ...s, user })),
    toggleTheme: () => state.update(s => ({
      ...s,
      theme: s.theme === 'light' ? 'dark' : 'light'
    }))
  };

  setContext(CONTEXT_KEY, { state, ...api });
  return { state, ...api };
}

export function getAppContext() {
  return getContext(CONTEXT_KEY);
}
```

```svelte
<!-- Root.svelte -->
<script>
  import { createAppContext } from './context.js';
  createAppContext();
</script>

<!-- AnyDescendant.svelte -->
<script>
  import { getAppContext } from './context.js';
  const { state, setUser, toggleTheme } = getAppContext();
</script>

<p>Theme: {$state.theme}</p>
<button on:click={toggleTheme}>Toggle Theme</button>
```

## Actions (Custom Directives)

### Basic Action

```svelte
<script>
  function tooltip(node, text) {
    const tooltip = document.createElement('div');
    tooltip.textContent = text;
    tooltip.className = 'tooltip';

    function mouseOver() {
      document.body.appendChild(tooltip);
      const rect = node.getBoundingClientRect();
      tooltip.style.left = rect.left + 'px';
      tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
    }

    function mouseOut() {
      tooltip.remove();
    }

    node.addEventListener('mouseover', mouseOver);
    node.addEventListener('mouseout', mouseOut);

    return {
      destroy() {
        node.removeEventListener('mouseover', mouseOver);
        node.removeEventListener('mouseout', mouseOut);
      }
    };
  }
</script>

<button use:tooltip="This is a tooltip">
  Hover me
</button>

<style>
  :global(.tooltip) {
    position: absolute;
    background: black;
    color: white;
    padding: 5px 10px;
    border-radius: 3px;
    font-size: 12px;
  }
</style>
```

### Action with Parameters

```svelte
<script>
  function longpress(node, duration = 500) {
    let timer;

    const handleMousedown = () => {
      timer = setTimeout(() => {
        node.dispatchEvent(new CustomEvent('longpress'));
      }, duration);
    };

    const handleMouseup = () => {
      clearTimeout(timer);
    };

    node.addEventListener('mousedown', handleMousedown);
    node.addEventListener('mouseup', handleMouseup);

    return {
      update(newDuration) {
        duration = newDuration;
      },
      destroy() {
        node.removeEventListener('mousedown', handleMousedown);
        node.removeEventListener('mouseup', handleMouseup);
      }
    };
  }
</script>

<button
  use:longpress={2000}
  on:longpress={() => alert('Long pressed!')}
>
  Press and hold
</button>
```

### Practical Actions

```svelte
<script>
  // Click outside action
  function clickOutside(node) {
    const handleClick = (event) => {
      if (!node.contains(event.target)) {
        node.dispatchEvent(new CustomEvent('outclick'));
      }
    };

    document.addEventListener('click', handleClick, true);

    return {
      destroy() {
        document.removeEventListener('click', handleClick, true);
      }
    };
  }

  // Auto-resize textarea
  function autoresize(node) {
    function resize() {
      node.style.height = 'auto';
      node.style.height = node.scrollHeight + 'px';
    }

    node.addEventListener('input', resize);
    resize();

    return {
      destroy() {
        node.removeEventListener('input', resize);
      }
    };
  }

  let showDropdown = false;
</script>

<div
  class="dropdown"
  use:clickOutside
  on:outclick={() => showDropdown = false}
>
  <button on:click={() => showDropdown = !showDropdown}>
    Toggle
  </button>
  {#if showDropdown}
    <div class="menu">Dropdown content</div>
  {/if}
</div>

<textarea use:autoresize placeholder="Auto-resizing textarea"></textarea>
```

## Transitions & Animations

### Built-in Transitions

```svelte
<script>
  import { fade, fly, slide, scale, blur } from 'svelte/transition';
  import { quintOut } from 'svelte/easing';

  let visible = true;
</script>

<button on:click={() => visible = !visible}>Toggle</button>

{#if visible}
  <!-- Fade -->
  <div transition:fade={{ duration: 300 }}>Fades in and out</div>

  <!-- Fly -->
  <div transition:fly={{ y: 200, duration: 500, easing: quintOut }}>
    Flies in and out
  </div>

  <!-- Slide -->
  <div transition:slide={{ duration: 300 }}>Slides in and out</div>

  <!-- Scale -->
  <div transition:scale={{ start: 0.5, duration: 300 }}>
    Scales in and out
  </div>

  <!-- Blur -->
  <div transition:blur={{ duration: 300 }}>Blurs in and out</div>
{/if}
```

### Directional Transitions

```svelte
<script>
  import { fade, fly } from 'svelte/transition';

  let visible = true;
</script>

{#if visible}
  <!-- Different transitions for in and out -->
  <div
    in:fly={{ y: -100, duration: 300 }}
    out:fade={{ duration: 200 }}
  >
    Flies in, fades out
  </div>
{/if}
```

### Custom Transitions

```svelte
<script>
  import { cubicOut } from 'svelte/easing';

  function typewriter(node, { speed = 1 }) {
    const valid = node.childNodes.length === 1 &&
                  node.childNodes[0].nodeType === Node.TEXT_NODE;

    if (!valid) return {};

    const text = node.textContent;
    const duration = text.length / (speed * 0.01);

    return {
      duration,
      tick: t => {
        const i = Math.trunc(text.length * t);
        node.textContent = text.slice(0, i);
      }
    };
  }

  function spin(node, { duration = 1000 }) {
    return {
      duration,
      css: t => {
        const eased = cubicOut(t);
        return `
          transform: rotate(${eased * 360}deg);
          opacity: ${t};
        `;
      }
    };
  }

  let visible = true;
</script>

{#if visible}
  <p transition:typewriter={{ speed: 1 }}>
    This text will appear letter by letter
  </p>

  <div transition:spin={{ duration: 500 }}>
    Spinning element
  </div>
{/if}
```

### Deferred Transitions (Crossfade)

```svelte
<script>
  import { quintOut } from 'svelte/easing';
  import { crossfade } from 'svelte/transition';

  const [send, receive] = crossfade({
    duration: 300,
    easing: quintOut
  });

  let todos = [
    { id: 1, done: false, text: 'Learn Svelte' },
    { id: 2, done: false, text: 'Build an app' }
  ];

  function toggle(id) {
    todos = todos.map(todo =>
      todo.id === id ? { ...todo, done: !todo.done } : todo
    );
  }
</script>

<div class="columns">
  <div>
    <h2>Todo</h2>
    {#each todos.filter(t => !t.done) as todo (todo.id)}
      <div
        in:receive={{ key: todo.id }}
        out:send={{ key: todo.id }}
        on:click={() => toggle(todo.id)}
      >
        {todo.text}
      </div>
    {/each}
  </div>

  <div>
    <h2>Done</h2>
    {#each todos.filter(t => t.done) as todo (todo.id)}
      <div
        in:receive={{ key: todo.id }}
        out:send={{ key: todo.id }}
        on:click={() => toggle(todo.id)}
      >
        {todo.text}
      </div>
    {/each}
  </div>
</div>
```

### Animations (Motion)

```svelte
<script>
  import { flip } from 'svelte/animate';
  import { quintOut } from 'svelte/easing';

  let items = [1, 2, 3, 4, 5];

  function shuffle() {
    items = items.sort(() => Math.random() - 0.5);
  }
</script>

<button on:click={shuffle}>Shuffle</button>

{#each items as item (item)}
  <div animate:flip={{ duration: 300, easing: quintOut }}>
    {item}
  </div>
{/each}
```

## Advanced Store Patterns

### Derived Stores

```javascript
// stores.js
import { writable, derived } from 'svelte/store';

export const firstName = writable('John');
export const lastName = writable('Doe');

// Derived from single store
export const fullName = derived(
  [firstName, lastName],
  ([$firstName, $lastName]) => `${$firstName} ${$lastName}`
);

// Derived with custom logic
export const items = writable([
  { id: 1, price: 10, quantity: 2 },
  { id: 2, price: 20, quantity: 1 }
]);

export const total = derived(
  items,
  ($items) => $items.reduce((sum, item) => sum + item.price * item.quantity, 0)
);

// Async derived store
export const userId = writable(1);

export const user = derived(
  userId,
  ($userId, set) => {
    fetch(`/api/users/${$userId}`)
      .then(res => res.json())
      .then(data => set(data));

    return () => {
      // Cleanup function
    };
  },
  null // Initial value
);
```

### Readable Stores

```javascript
// stores.js
import { readable } from 'svelte/store';

// Time store
export const time = readable(new Date(), (set) => {
  const interval = setInterval(() => {
    set(new Date());
  }, 1000);

  return () => clearInterval(interval);
});

// Mouse position store
export const mousePosition = readable({ x: 0, y: 0 }, (set) => {
  const handleMouseMove = (event) => {
    set({ x: event.clientX, y: event.clientY });
  };

  document.addEventListener('mousemove', handleMouseMove);

  return () => {
    document.removeEventListener('mousemove', handleMouseMove);
  };
});

// WebSocket store
export const websocket = readable(null, (set) => {
  const ws = new WebSocket('wss://example.com/socket');

  ws.addEventListener('message', (event) => {
    set(JSON.parse(event.data));
  });

  return () => ws.close();
});
```

### Custom Stores

```javascript
// stores.js
import { writable } from 'svelte/store';

// Custom store with methods
function createCounter() {
  const { subscribe, set, update } = writable(0);

  return {
    subscribe,
    increment: () => update(n => n + 1),
    decrement: () => update(n => n - 1),
    reset: () => set(0)
  };
}

export const counter = createCounter();

// LocalStorage store
function createLocalStore(key, initial) {
  const stored = localStorage.getItem(key);
  const { subscribe, set, update } = writable(
    stored ? JSON.parse(stored) : initial
  );

  return {
    subscribe,
    set: (value) => {
      localStorage.setItem(key, JSON.stringify(value));
      set(value);
    },
    update: (fn) => {
      update(value => {
        const newValue = fn(value);
        localStorage.setItem(key, JSON.stringify(newValue));
        return newValue;
      });
    }
  };
}

export const preferences = createLocalStore('preferences', {
  theme: 'light',
  language: 'en'
});

// Async store with loading state
function createAsyncStore(url) {
  const { subscribe, set } = writable({
    loading: true,
    data: null,
    error: null
  });

  async function load() {
    try {
      set({ loading: true, data: null, error: null });
      const response = await fetch(url);
      if (!response.ok) throw new Error('Fetch failed');
      const data = await response.json();
      set({ loading: false, data, error: null });
    } catch (error) {
      set({ loading: false, data: null, error: error.message });
    }
  }

  load();

  return {
    subscribe,
    reload: load
  };
}

export const users = createAsyncStore('/api/users');
```

### Store Composition

```javascript
// stores.js
import { writable, derived, get } from 'svelte/store';

// Shopping cart example
function createCart() {
  const { subscribe, set, update } = writable([]);

  return {
    subscribe,
    addItem: (item) => update(items => {
      const existing = items.find(i => i.id === item.id);
      if (existing) {
        return items.map(i =>
          i.id === item.id ? { ...i, quantity: i.quantity + 1 } : i
        );
      }
      return [...items, { ...item, quantity: 1 }];
    }),
    removeItem: (id) => update(items => items.filter(i => i.id !== id)),
    updateQuantity: (id, quantity) => update(items =>
      items.map(i => i.id === id ? { ...i, quantity } : i)
    ),
    clear: () => set([])
  };
}

export const cart = createCart();

export const cartTotal = derived(
  cart,
  ($cart) => $cart.reduce((sum, item) => sum + item.price * item.quantity, 0)
);

export const cartCount = derived(
  cart,
  ($cart) => $cart.reduce((sum, item) => sum + item.quantity, 0)
);
```

## Component Communication Patterns

### Props Down, Events Up

```svelte
<!-- Parent.svelte -->
<script>
  import Child from './Child.svelte';

  let parentValue = 'Hello';

  function handleUpdate(event) {
    parentValue = event.detail.value;
  }
</script>

<Child value={parentValue} on:update={handleUpdate} />

<!-- Child.svelte -->
<script>
  import { createEventDispatcher } from 'svelte';

  export let value;
  const dispatch = createEventDispatcher();

  function updateValue() {
    dispatch('update', { value: 'Updated from child' });
  }
</script>

<button on:click={updateValue}>Update Parent</button>
```

### Event Forwarding

```svelte
<!-- Child.svelte -->
<button on:click>
  Click me
</button>

<!-- Parent.svelte -->
<script>
  import Child from './Child.svelte';
</script>

<Child on:click={() => console.log('Clicked!')} />
```

### Store-based Communication

```javascript
// shared.js
import { writable } from 'svelte/store';

export const sharedState = writable({ message: 'Hello' });
```

```svelte
<!-- ComponentA.svelte -->
<script>
  import { sharedState } from './shared.js';
</script>

<input bind:value={$sharedState.message}>

<!-- ComponentB.svelte -->
<script>
  import { sharedState } from './shared.js';
</script>

<p>{$sharedState.message}</p>
```

### Component Instance References

```svelte
<!-- Modal.svelte -->
<script>
  let visible = false;

  export function open() {
    visible = true;
  }

  export function close() {
    visible = false;
  }
</script>

{#if visible}
  <div class="modal">
    <slot {close} />
  </div>
{/if}

<!-- Parent.svelte -->
<script>
  import Modal from './Modal.svelte';

  let modal;
</script>

<button on:click={() => modal.open()}>Open Modal</button>

<Modal bind:this={modal}>
  <h2>Modal Title</h2>
  <button on:click={() => modal.close()}>Close</button>
</Modal>
```

## Conditional Rendering & Logic

### If/Else Blocks

```svelte
<script>
  let user = { loggedIn: false, isAdmin: false };
</script>

{#if user.loggedIn}
  <p>Welcome back!</p>
  {#if user.isAdmin}
    <button>Admin Panel</button>
  {:else}
    <p>Regular user</p>
  {/if}
{:else}
  <button>Log in</button>
{/if}

<!-- Else if -->
{#if x > 10}
  <p>x is greater than 10</p>
{:else if x < 5}
  <p>x is less than 5</p>
{:else}
  <p>x is between 5 and 10</p>
{/if}
```

### Each Blocks

```svelte
<script>
  let items = [
    { id: 1, name: 'Apple' },
    { id: 2, name: 'Banana' },
    { id: 3, name: 'Cherry' }
  ];
</script>

<!-- Basic each -->
{#each items as item}
  <p>{item.name}</p>
{/each}

<!-- With index -->
{#each items as item, index}
  <p>{index + 1}: {item.name}</p>
{/each}

<!-- With key (important for animations and performance) -->
{#each items as item (item.id)}
  <p>{item.name}</p>
{/each}

<!-- Destructuring -->
{#each items as { id, name }}
  <p>{id}: {name}</p>
{/each}

<!-- With else -->
{#each items as item}
  <p>{item.name}</p>
{:else}
  <p>No items</p>
{/each}
```

### Await Blocks

```svelte
<script>
  async function fetchData() {
    const response = await fetch('/api/data');
    if (!response.ok) throw new Error('Failed to fetch');
    return response.json();
  }

  let promise = fetchData();
</script>

<!-- Basic await -->
{#await promise}
  <p>Loading...</p>
{:then data}
  <p>Data: {JSON.stringify(data)}</p>
{:catch error}
  <p>Error: {error.message}</p>
{/await}

<!-- Only handle then -->
{#await promise then data}
  <p>{data.message}</p>
{/await}

<!-- Only handle catch -->
{#await promise catch error}
  <p>Error: {error.message}</p>
{/await}
```

### Key Blocks

```svelte
<script>
  let value = 0;

  // Component will be destroyed and recreated when value changes
</script>

{#key value}
  <Component {value} />
{/key}

<button on:click={() => value += 1}>Reset Component</button>
```

## Form Handling Patterns

### Basic Form

```svelte
<script>
  let formData = {
    name: '',
    email: '',
    age: '',
    terms: false
  };

  let errors = {};
  let submitted = false;

  function validate() {
    errors = {};

    if (!formData.name) {
      errors.name = 'Name is required';
    }

    if (!formData.email) {
      errors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errors.email = 'Invalid email format';
    }

    if (!formData.age) {
      errors.age = 'Age is required';
    } else if (formData.age < 18) {
      errors.age = 'Must be 18 or older';
    }

    if (!formData.terms) {
      errors.terms = 'You must accept the terms';
    }

    return Object.keys(errors).length === 0;
  }

  function handleSubmit(event) {
    event.preventDefault();

    if (validate()) {
      submitted = true;
      console.log('Form submitted:', formData);
    }
  }
</script>

<form on:submit={handleSubmit}>
  <div>
    <label for="name">Name:</label>
    <input
      id="name"
      type="text"
      bind:value={formData.name}
      class:error={errors.name}
    >
    {#if errors.name}
      <span class="error-message">{errors.name}</span>
    {/if}
  </div>

  <div>
    <label for="email">Email:</label>
    <input
      id="email"
      type="email"
      bind:value={formData.email}
      class:error={errors.email}
    >
    {#if errors.email}
      <span class="error-message">{errors.email}</span>
    {/if}
  </div>

  <div>
    <label for="age">Age:</label>
    <input
      id="age"
      type="number"
      bind:value={formData.age}
      class:error={errors.age}
    >
    {#if errors.age}
      <span class="error-message">{errors.age}</span>
    {/if}
  </div>

  <div>
    <label>
      <input type="checkbox" bind:checked={formData.terms}>
      I accept the terms
    </label>
    {#if errors.terms}
      <span class="error-message">{errors.terms}</span>
    {/if}
  </div>

  <button type="submit">Submit</button>
</form>

{#if submitted}
  <div class="success">Form submitted successfully!</div>
{/if}

<style>
  .error {
    border-color: red;
  }
  .error-message {
    color: red;
    font-size: 0.875rem;
  }
  .success {
    color: green;
    margin-top: 1rem;
  }
</style>
```

### Form with Real-time Validation

```svelte
<script>
  let email = '';
  let emailError = '';
  let emailTouched = false;

  $: if (emailTouched) {
    if (!email) {
      emailError = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      emailError = 'Invalid email format';
    } else {
      emailError = '';
    }
  }
</script>

<input
  type="email"
  bind:value={email}
  on:blur={() => emailTouched = true}
  class:error={emailTouched && emailError}
>
{#if emailTouched && emailError}
  <span class="error-message">{emailError}</span>
{/if}
```

### Dynamic Form Fields

```svelte
<script>
  let fields = [
    { id: 1, value: '' }
  ];

  let nextId = 2;

  function addField() {
    fields = [...fields, { id: nextId++, value: '' }];
  }

  function removeField(id) {
    fields = fields.filter(f => f.id !== id);
  }
</script>

<form>
  {#each fields as field, index (field.id)}
    <div>
      <input
        type="text"
        bind:value={field.value}
        placeholder="Field {index + 1}"
      >
      {#if fields.length > 1}
        <button type="button" on:click={() => removeField(field.id)}>
          Remove
        </button>
      {/if}
    </div>
  {/each}

  <button type="button" on:click={addField}>
    Add Field
  </button>
</form>
```

## Async Operations & API Integration

### Fetch with Loading States

```svelte
<script>
  let data = null;
  let loading = false;
  let error = null;

  async function fetchData() {
    loading = true;
    error = null;

    try {
      const response = await fetch('/api/data');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      data = await response.json();
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }
</script>

<button on:click={fetchData} disabled={loading}>
  Fetch Data
</button>

{#if loading}
  <div class="spinner">Loading...</div>
{:else if error}
  <div class="error">Error: {error}</div>
{:else if data}
  <div class="data">
    <pre>{JSON.stringify(data, null, 2)}</pre>
  </div>
{/if}
```

### API Hook Pattern

```javascript
// hooks.js
import { writable } from 'svelte/store';

export function useApi(url, options = {}) {
  const { subscribe, set, update } = writable({
    data: null,
    loading: false,
    error: null
  });

  async function execute(params = {}) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const queryString = new URLSearchParams(params).toString();
      const fullUrl = queryString ? `${url}?${queryString}` : url;

      const response = await fetch(fullUrl, options);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      set({ data, loading: false, error: null });
      return data;
    } catch (error) {
      set({ data: null, loading: false, error: error.message });
      throw error;
    }
  }

  return {
    subscribe,
    execute
  };
}
```

```svelte
<script>
  import { useApi } from './hooks.js';

  const users = useApi('/api/users');

  $: if ($users.data) {
    console.log('Users loaded:', $users.data);
  }
</script>

<button on:click={() => users.execute()}>
  Load Users
</button>

{#if $users.loading}
  <p>Loading...</p>
{:else if $users.error}
  <p>Error: {$users.error}</p>
{:else if $users.data}
  <ul>
    {#each $users.data as user}
      <li>{user.name}</li>
    {/each}
  </ul>
{/if}
```

### Debounced API Calls

```svelte
<script>
  let query = '';
  let results = [];
  let loading = false;
  let debounceTimer;

  async function search(q) {
    if (!q) {
      results = [];
      return;
    }

    loading = true;
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      results = await response.json();
    } catch (error) {
      console.error('Search failed:', error);
      results = [];
    } finally {
      loading = false;
    }
  }

  $: {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      search(query);
    }, 300);
  }
</script>

<input
  type="text"
  bind:value={query}
  placeholder="Search..."
>

{#if loading}
  <p>Searching...</p>
{:else if results.length}
  <ul>
    {#each results as result}
      <li>{result.title}</li>
    {/each}
  </ul>
{:else if query}
  <p>No results found</p>
{/if}
```

### Optimistic Updates

```svelte
<script>
  let todos = [];
  let optimisticTodos = [];

  $: optimisticTodos = todos;

  async function addTodo(text) {
    const tempId = Date.now();
    const optimisticTodo = { id: tempId, text, pending: true };

    // Add optimistically
    optimisticTodos = [...optimisticTodos, optimisticTodo];

    try {
      const response = await fetch('/api/todos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const newTodo = await response.json();

      // Replace optimistic with real
      todos = [...todos, newTodo];
      optimisticTodos = todos;
    } catch (error) {
      // Rollback on error
      optimisticTodos = todos;
      alert('Failed to add todo');
    }
  }

  async function deleteTodo(id) {
    const original = [...todos];
    todos = todos.filter(t => t.id !== id);

    try {
      await fetch(`/api/todos/${id}`, { method: 'DELETE' });
    } catch (error) {
      // Rollback
      todos = original;
      alert('Failed to delete todo');
    }
  }
</script>

{#each optimisticTodos as todo (todo.id)}
  <div class:pending={todo.pending}>
    {todo.text}
    <button on:click={() => deleteTodo(todo.id)}>Delete</button>
  </div>
{/each}
```

## Component Composition Patterns

### Higher-Order Component Pattern

```svelte
<!-- WithLoading.svelte -->
<script>
  export let loading = false;
</script>

{#if loading}
  <div class="loading">Loading...</div>
{:else}
  <slot />
{/if}

<!-- WithAuth.svelte -->
<script>
  export let user = null;
</script>

{#if user}
  <slot {user} />
{:else}
  <p>Please log in</p>
{/if}

<!-- Usage -->
<WithAuth {user} let:user>
  <WithLoading {loading}>
    <Dashboard {user} />
  </WithLoading>
</WithAuth>
```

### Render Props via Slots

```svelte
<!-- DataProvider.svelte -->
<script>
  import { onMount } from 'svelte';

  export let url;

  let data = null;
  let loading = true;
  let error = null;

  onMount(async () => {
    try {
      const response = await fetch(url);
      data = await response.json();
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  });
</script>

<slot {data} {loading} {error} />

<!-- Usage -->
<DataProvider url="/api/users" let:data let:loading let:error>
  {#if loading}
    <p>Loading...</p>
  {:else if error}
    <p>Error: {error}</p>
  {:else}
    <ul>
      {#each data as user}
        <li>{user.name}</li>
      {/each}
    </ul>
  {/if}
</DataProvider>
```

### Compound Components

```svelte
<!-- Tabs.svelte -->
<script>
  import { setContext } from 'svelte';
  import { writable } from 'svelte/store';

  export let active = 0;

  const activeTab = writable(active);
  setContext('tabs', { activeTab });

  $: activeTab.set(active);
</script>

<div class="tabs">
  <slot />
</div>

<!-- TabList.svelte -->
<div class="tab-list">
  <slot />
</div>

<!-- Tab.svelte -->
<script>
  import { getContext } from 'svelte';

  export let index;

  const { activeTab } = getContext('tabs');
</script>

<button
  class:active={$activeTab === index}
  on:click={() => activeTab.set(index)}
>
  <slot />
</button>

<!-- TabPanel.svelte -->
<script>
  import { getContext } from 'svelte';

  export let index;

  const { activeTab } = getContext('tabs');
</script>

{#if $activeTab === index}
  <div class="tab-panel">
    <slot />
  </div>
{/if}

<!-- Usage -->
<script>
  import Tabs from './Tabs.svelte';
  import TabList from './TabList.svelte';
  import Tab from './Tab.svelte';
  import TabPanel from './TabPanel.svelte';

  let active = 0;
</script>

<Tabs bind:active>
  <TabList>
    <Tab index={0}>Tab 1</Tab>
    <Tab index={1}>Tab 2</Tab>
    <Tab index={2}>Tab 3</Tab>
  </TabList>

  <TabPanel index={0}>Content 1</TabPanel>
  <TabPanel index={1}>Content 2</TabPanel>
  <TabPanel index={2}>Content 3</TabPanel>
</Tabs>
```

## Performance Optimization

### Keyed Each Blocks

```svelte
<script>
  let items = [
    { id: 1, name: 'Item 1' },
    { id: 2, name: 'Item 2' },
    { id: 3, name: 'Item 3' }
  ];

  function shuffle() {
    items = items.sort(() => Math.random() - 0.5);
  }
</script>

<!-- ❌ Without key - components recreated -->
{#each items as item}
  <ExpensiveComponent data={item} />
{/each}

<!-- ✅ With key - components reused -->
{#each items as item (item.id)}
  <ExpensiveComponent data={item} />
{/each}
```

### Immutable Data Patterns

```svelte
<script>
  // ❌ Mutating - won't trigger reactivity
  function badUpdate() {
    items[0].name = 'Updated';
  }

  // ✅ Immutable - triggers reactivity
  function goodUpdate() {
    items = items.map((item, i) =>
      i === 0 ? { ...item, name: 'Updated' } : item
    );
  }

  // ✅ Array operations
  function addItem(item) {
    items = [...items, item];
  }

  function removeItem(id) {
    items = items.filter(item => item.id !== id);
  }

  function updateItem(id, updates) {
    items = items.map(item =>
      item.id === id ? { ...item, ...updates } : item
    );
  }
</script>
```

### Lazy Loading Components

```svelte
<script>
  let HeavyComponent;
  let showHeavy = false;

  async function loadHeavy() {
    if (!HeavyComponent) {
      HeavyComponent = (await import('./HeavyComponent.svelte')).default;
    }
    showHeavy = true;
  }
</script>

<button on:click={loadHeavy}>Load Heavy Component</button>

{#if showHeavy && HeavyComponent}
  <svelte:component this={HeavyComponent} />
{/if}
```

### Memoization with Reactive Statements

```svelte
<script>
  let numbers = [1, 2, 3, 4, 5];
  let filter = 'all';

  // Memoized computation - only runs when dependencies change
  $: filtered = numbers.filter(n => {
    console.log('Filtering...');
    if (filter === 'even') return n % 2 === 0;
    if (filter === 'odd') return n % 2 === 1;
    return true;
  });

  $: sum = filtered.reduce((a, b) => a + b, 0);
</script>
```

### Virtual Lists for Long Lists

```svelte
<script>
  import { onMount, tick } from 'svelte';

  export let items = [];
  export let itemHeight = 50;

  let viewport;
  let contents;
  let viewportHeight = 0;
  let scrollTop = 0;

  $: visibleItems = Math.ceil(viewportHeight / itemHeight) + 1;
  $: start = Math.floor(scrollTop / itemHeight);
  $: end = start + visibleItems;
  $: visible = items.slice(start, end);
  $: paddingTop = start * itemHeight;
  $: paddingBottom = (items.length - end) * itemHeight;

  onMount(() => {
    viewportHeight = viewport.offsetHeight;
  });
</script>

<div
  class="viewport"
  bind:this={viewport}
  bind:offsetHeight={viewportHeight}
  on:scroll={() => scrollTop = viewport.scrollTop}
>
  <div
    class="contents"
    style="padding-top: {paddingTop}px; padding-bottom: {paddingBottom}px;"
  >
    {#each visible as item (item.id)}
      <div class="item" style="height: {itemHeight}px;">
        {item.name}
      </div>
    {/each}
  </div>
</div>

<style>
  .viewport {
    height: 400px;
    overflow-y: auto;
  }
</style>
```

## TypeScript Integration

### Typed Component Props

```svelte
<!-- Component.svelte -->
<script lang="ts">
  export let name: string;
  export let age: number = 0;
  export let optional?: string;
  export let callback: (value: string) => void;

  interface User {
    id: number;
    name: string;
    email: string;
  }

  export let user: User;

  let count: number = 0;

  function increment(): void {
    count += 1;
  }
</script>

<button on:click={increment}>
  {name} ({age}) - Count: {count}
</button>
```

### Typed Events

```svelte
<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  interface CustomEvents {
    submit: { value: string; timestamp: number };
    cancel: never;
  }

  const dispatch = createEventDispatcher<CustomEvents>();

  function handleSubmit() {
    dispatch('submit', {
      value: 'data',
      timestamp: Date.now()
    });
  }

  function handleCancel() {
    dispatch('cancel');
  }
</script>
```

### Typed Stores

```typescript
// stores.ts
import { writable, derived, type Writable, type Readable } from 'svelte/store';

interface User {
  id: number;
  name: string;
  email: string;
}

export const user: Writable<User | null> = writable(null);

export const userName: Readable<string> = derived(
  user,
  ($user) => $user?.name ?? 'Guest'
);

// Custom typed store
interface CounterStore extends Readable<number> {
  increment: () => void;
  decrement: () => void;
  reset: () => void;
}

function createCounter(): CounterStore {
  const { subscribe, set, update } = writable(0);

  return {
    subscribe,
    increment: () => update(n => n + 1),
    decrement: () => update(n => n - 1),
    reset: () => set(0)
  };
}

export const counter: CounterStore = createCounter();
```

### Generic Components

```svelte
<!-- List.svelte -->
<script lang="ts" generics="T">
  export let items: T[];
  export let getKey: (item: T) => string | number;
  export let renderItem: (item: T) => string;
</script>

<ul>
  {#each items as item (getKey(item))}
    <li>{renderItem(item)}</li>
  {/each}
</ul>

<!-- Usage -->
<script lang="ts">
  import List from './List.svelte';

  interface Product {
    id: number;
    name: string;
    price: number;
  }

  const products: Product[] = [
    { id: 1, name: 'Apple', price: 1.99 },
    { id: 2, name: 'Banana', price: 0.99 }
  ];
</script>

<List
  items={products}
  getKey={(p) => p.id}
  renderItem={(p) => `${p.name} - $${p.price}`}
/>
```

## Common Patterns & Best Practices

### Container/Presenter Pattern

```svelte
<!-- UserContainer.svelte (Smart Component) -->
<script>
  import { onMount } from 'svelte';
  import UserPresenter from './UserPresenter.svelte';

  let user = null;
  let loading = true;
  let error = null;

  onMount(async () => {
    try {
      const response = await fetch('/api/user');
      user = await response.json();
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  });

  function handleUpdate(event) {
    // Handle update logic
  }
</script>

<UserPresenter
  {user}
  {loading}
  {error}
  on:update={handleUpdate}
/>

<!-- UserPresenter.svelte (Dumb Component) -->
<script>
  import { createEventDispatcher } from 'svelte';

  export let user;
  export let loading;
  export let error;

  const dispatch = createEventDispatcher();
</script>

{#if loading}
  <p>Loading...</p>
{:else if error}
  <p>Error: {error}</p>
{:else if user}
  <div class="user">
    <h2>{user.name}</h2>
    <p>{user.email}</p>
    <button on:click={() => dispatch('update')}>
      Update
    </button>
  </div>
{/if}
```

### Singleton Store Pattern

```javascript
// auth.js
import { writable } from 'svelte/store';

function createAuth() {
  const { subscribe, set, update } = writable({
    user: null,
    token: null,
    loading: false
  });

  return {
    subscribe,
    login: async (credentials) => {
      update(state => ({ ...state, loading: true }));
      try {
        const response = await fetch('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(credentials)
        });
        const data = await response.json();
        set({ user: data.user, token: data.token, loading: false });
      } catch (error) {
        update(state => ({ ...state, loading: false }));
        throw error;
      }
    },
    logout: () => {
      set({ user: null, token: null, loading: false });
    }
  };
}

export const auth = createAuth();
```

### Feature Flags Pattern

```svelte
<script>
  import { getContext } from 'svelte';

  const features = getContext('features') || {};

  $: hasNewFeature = features.newFeature === true;
</script>

{#if hasNewFeature}
  <NewFeatureComponent />
{:else}
  <OldFeatureComponent />
{/if}
```

### Error Boundary Pattern

```svelte
<!-- ErrorBoundary.svelte -->
<script>
  import { onMount } from 'svelte';

  let error = null;

  onMount(() => {
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  });

  function handleError(event) {
    error = event.error;
  }

  export function reset() {
    error = null;
  }
</script>

{#if error}
  <div class="error-boundary">
    <h2>Something went wrong</h2>
    <p>{error.message}</p>
    <button on:click={reset}>Try again</button>
  </div>
{:else}
  <slot />
{/if}
```

## Common Gotchas & Troubleshooting

### Reactivity Gotchas

```svelte
<script>
  let obj = { count: 0 };
  let arr = [1, 2, 3];

  // ❌ These won't trigger updates
  obj.count += 1;
  arr.push(4);
  arr[0] = 10;

  // ✅ These will trigger updates
  obj = { ...obj, count: obj.count + 1 };
  arr = [...arr, 4];
  arr = arr.map((v, i) => i === 0 ? 10 : v);

  // ✅ Or reassign to trigger
  arr.push(4);
  arr = arr;
</script>
```

### Event Modifier Ordering

```svelte
<!-- Order matters! -->
<button on:click|preventDefault|stopPropagation={handler}>
  Click
</button>

<!-- Common modifiers -->
<div on:click|preventDefault>...</div>
<div on:click|stopPropagation>...</div>
<div on:click|capture>...</div>
<div on:click|once>...</div>
<div on:click|passive>...</div>
<div on:click|self>...</div>
<div on:click|trusted>...</div>
```

### Binding Lifecycle

```svelte
<script>
  let element;

  // ❌ element is undefined here
  console.log(element);

  // ✅ Use onMount
  import { onMount } from 'svelte';

  onMount(() => {
    console.log(element); // Now it's defined
  });

  // ✅ Or reactive statement
  $: if (element) {
    console.log(element);
  }
</script>

<div bind:this={element}>Content</div>
```

### Style Scoping

```svelte
<style>
  /* Scoped to this component */
  p {
    color: red;
  }

  /* Global styles */
  :global(body) {
    margin: 0;
  }

  /* Mixing scoped and global */
  div :global(.external-class) {
    color: blue;
  }

  /* Global modifier */
  :global(.global-class) p {
    color: green;
  }
</style>
```

### Await Block Pitfalls

```svelte
<script>
  // ❌ Promise doesn't update
  let promise = fetch('/api/data');

  function refresh() {
    fetch('/api/data'); // This doesn't update the promise
  }

  // ✅ Reassign the promise
  function refreshCorrect() {
    promise = fetch('/api/data');
  }
</script>

{#await promise}
  <p>Loading...</p>
{:then data}
  <p>Data loaded</p>
{/await}

<button on:click={refreshCorrect}>Refresh</button>
```

### Component Imports

```svelte
<script>
  // ❌ This won't work for conditional rendering
  import Component from './Component.svelte';
  let show = false;
</script>

{#if show}
  <Component /> <!-- Always imported even when hidden -->
{/if}

<!-- ✅ Use dynamic import for code splitting -->
<script>
  let Component;
  let show = false;

  async function loadComponent() {
    if (!Component) {
      Component = (await import('./Component.svelte')).default;
    }
    show = true;
  }
</script>

{#if show && Component}
  <svelte:component this={Component} />
{/if}
```

## Quick Reference

| Feature | Syntax |
|---------|--------|
| Reactive variable | `$: value = ...` |
| Event handler | `on:click={handler}` |
| Event modifiers | `on:click\|preventDefault\|stopPropagation` |
| Two-way binding | `bind:value={variable}` |
| Element reference | `bind:this={element}` |
| Conditional | `{#if condition}...{:else}...{/if}` |
| Loop | `{#each items as item}...{/each}` |
| Keyed loop | `{#each items as item (item.id)}...{/each}` |
| Await | `{#await promise}...{:then data}...{:catch error}...{/await}` |
| Slot | `<slot />` or `<slot name="header" />` |
| Slot props | `<slot {item} />` / `let:item` |
| Store subscription | `$storeName` |
| Component binding | `<Component bind:prop={value} />` |
| Dynamic component | `<svelte:component this={Component} />` |
| Self reference | `<svelte:self />` |
| Window events | `<svelte:window on:resize={handler} />` |
| Body events | `<svelte:body on:click={handler} />` |
| Head content | `<svelte:head><title>...</title></svelte:head>` |
| Transition | `transition:fade` or `in:fade out:fly` |
| Animation | `animate:flip` |
| Action | `use:action` or `use:action={params}` |
| Class directive | `class:active={isActive}` |
| Style directive | `style:color={color}` |

Svelte compiles components to highly efficient imperative code, resulting in small bundle sizes and excellent performance.
