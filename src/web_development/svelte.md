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

## Quick Reference

| Feature | Syntax |
|---------|--------|
| Reactive variable | `$: value = ...` |
| Event handler | `on:click={handler}` |
| Two-way binding | `bind:value={variable}` |
| Conditional | `{#if condition}...{/if}` |
| Loop | `{#each items as item}...{/each}` |
| Await | `{#await promise}...{/await}` |

Svelte compiles components to highly efficient imperative code, resulting in small bundle sizes and excellent performance.
