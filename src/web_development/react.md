# React

## Overview

React is a JavaScript library for building user interfaces with reusable components and efficient rendering.

## Components

### Functional Components (Modern)

```javascript
function Welcome({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// Arrow function
const Greeting = ({ message }) => <p>{message}</p>;
```

### Class Components (Legacy)

```javascript
class Welcome extends React.Component {
  render() {
    return <h1>Hello, {this.props.name}!</h1>;
  }
}
```

## Hooks

Modern way to manage state and effects:

```javascript
import { useState, useEffect } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  useEffect(() => {
    console.log('Count changed:', count);
    // Cleanup
    return () => console.log('Cleanup');
  }, [count]); // Dependencies

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

## Common Hooks

| Hook | Purpose |
|------|---------|
| **useState** | Manage state |
| **useEffect** | Side effects |
| **useContext** | Access context |
| **useReducer** | Complex state logic |
| **useCallback** | Memoize function |
| **useMemo** | Memoize value |

## Props

```javascript
// Parent
<Child name="John" age={30} onClick={handleClick} />

// Child
function Child({ name, age, onClick }) {
  return (
    <div onClick={onClick}>
      {name} is {age}
    </div>
  );
}
```

## Conditional Rendering

```javascript
{isLoggedIn && <Dashboard />}
{user ? <UserProfile /> : <LoginForm />}

{status === 'loading' && <Spinner />}
{status === 'error' && <Error />}
{status === 'success' && <Data />}
```

## Lists

```javascript
const users = [
  { id: 1, name: 'John' },
  { id: 2, name: 'Jane' }
];

<ul>
  {users.map(user => (
    <li key={user.id}>{user.name}</li>
  ))}
</ul>
```

## Event Handling

```javascript
function Button() {
  const handleClick = (e) => {
    console.log('Clicked');
  };

  const handleChange = (e) => {
    const value = e.target.value;
  };

  return (
    <>
      <button onClick={handleClick}>Click</button>
      <input onChange={handleChange} />
    </>
  );
}
```

## Forms

```javascript
function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(email, password);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <button type="submit">Login</button>
    </form>
  );
}
```

## State Management

### Local State (useState)
```javascript
const [state, setState] = useState(initialValue);
```

### Context API (Global)
```javascript
const UserContext = createContext();

function App() {
  return (
    <UserContext.Provider value={{ user: 'John' }}>
      <Child />
    </UserContext.Provider>
  );
}

function Child() {
  const { user } = useContext(UserContext);
}
```

### Redux (Complex)
- Centralized store
- Actions → Reducers → State

## Lifecycle (Class Components)

```javascript
componentDidMount() { } // After render
componentDidUpdate() { } // After update
componentWillUnmount() { } // Before remove
```

## Best Practices

1. **Functional components** (with hooks)
2. **Keep components small**
3. **Lift state up** when needed
4. **Use keys** in lists
5. **Memoize** expensive computations
6. **Lazy load** components

## ELI10

React is like LEGO blocks:
- Build reusable pieces (components)
- Combine to make complex UIs
- Reuse same piece many times
- Efficient updates when data changes!

## Further Resources

- [React Documentation](https://react.dev/)
- [React Hooks](https://react.dev/reference/react)
