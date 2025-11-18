# React

## Overview

React is a JavaScript library for building user interfaces with reusable components and efficient rendering. Developed by Meta (Facebook), React uses a virtual DOM for optimal performance and supports declarative programming, component-based architecture, and unidirectional data flow.

**Key Features:**
- Component-based architecture
- Virtual DOM for efficient updates
- JSX syntax (JavaScript XML)
- One-way data binding
- Rich ecosystem and community
- Server-side rendering (SSR) support
- Concurrent rendering (React 18+)

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

| Hook | Purpose | When to Use |
|------|---------|-------------|
| **useState** | Manage component state | Simple state values |
| **useEffect** | Side effects & lifecycle | API calls, subscriptions, DOM manipulation |
| **useContext** | Access context values | Avoid prop drilling |
| **useReducer** | Complex state logic | Multiple related state values |
| **useCallback** | Memoize functions | Prevent child re-renders |
| **useMemo** | Memoize computed values | Expensive calculations |
| **useRef** | Persist values/DOM refs | Access DOM, store mutable values |
| **useLayoutEffect** | Synchronous effects | Measure DOM, prevent flicker |
| **useImperativeHandle** | Customize ref exposure | Expose specific methods to parent |
| **useId** | Generate unique IDs | Accessibility IDs (React 18+) |
| **useTransition** | Mark updates as transitions | Non-urgent updates (React 18+) |
| **useDeferredValue** | Defer expensive updates | Debounce values (React 18+) |

### useRef Example

```javascript
function TextInput() {
  const inputRef = useRef(null);

  const focusInput = () => {
    inputRef.current.focus();
  };

  return (
    <>
      <input ref={inputRef} />
      <button onClick={focusInput}>Focus Input</button>
    </>
  );
}
```

### useReducer Example

```javascript
const initialState = { count: 0 };

function reducer(state, action) {
  switch (action.type) {
    case 'increment':
      return { count: state.count + 1 };
    case 'decrement':
      return { count: state.count - 1 };
    case 'reset':
      return initialState;
    default:
      throw new Error('Unknown action');
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <>
      <p>Count: {state.count}</p>
      <button onClick={() => dispatch({ type: 'increment' })}>+</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>-</button>
      <button onClick={() => dispatch({ type: 'reset' })}>Reset</button>
    </>
  );
}
```

### Custom Hooks

```javascript
// Custom hook for fetching data
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(url);
        const json = await response.json();
        setData(json);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [url]);

  return { data, loading, error };
}

// Usage
function UserProfile({ userId }) {
  const { data, loading, error } = useFetch(`/api/users/${userId}`);

  if (loading) return <Spinner />;
  if (error) return <Error message={error.message} />;
  return <div>{data.name}</div>;
}
```

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
2. **Keep components small** and focused
3. **Lift state up** when needed
4. **Use keys** in lists (stable, unique IDs)
5. **Memoize** expensive computations
6. **Lazy load** components
7. **Avoid inline functions** in JSX (use useCallback)
8. **Use fragments** to avoid extra DOM nodes
9. **Name components** for better debugging
10. **Follow hooks rules** (top level, React functions only)

## Performance Optimization

### React.memo

Prevents unnecessary re-renders when props haven't changed:

```javascript
const ExpensiveComponent = React.memo(({ data }) => {
  // Only re-renders if 'data' prop changes
  return <div>{/* expensive rendering */}</div>;
});
```

### useCallback & useMemo

```javascript
function Parent() {
  const [count, setCount] = useState(0);
  const [items, setItems] = useState([]);

  // Memoize callback to prevent child re-renders
  const handleClick = useCallback(() => {
    console.log('Clicked');
  }, []); // Dependencies

  // Memoize expensive computation
  const expensiveValue = useMemo(() => {
    return items.reduce((sum, item) => sum + item.value, 0);
  }, [items]);

  return <Child onClick={handleClick} total={expensiveValue} />;
}
```

### Code Splitting & Lazy Loading

```javascript
import { lazy, Suspense } from 'react';

// Lazy load component
const Dashboard = lazy(() => import('./Dashboard'));
const Profile = lazy(() => import('./Profile'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Dashboard />
    </Suspense>
  );
}
```

### Virtualization (Large Lists)

```javascript
// Using react-window or react-virtualized
import { FixedSizeList } from 'react-window';

function VirtualList({ items }) {
  const Row = ({ index, style }) => (
    <div style={style}>{items[index].name}</div>
  );

  return (
    <FixedSizeList
      height={400}
      itemCount={items.length}
      itemSize={35}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
}
```

## Error Boundaries

Catch JavaScript errors in component tree:

```javascript
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error:', error, errorInfo);
    // Log to error reporting service
  }

  render() {
    if (this.state.hasError) {
      return (
        <div>
          <h1>Something went wrong.</h1>
          <details>{this.state.error.toString()}</details>
        </div>
      );
    }

    return this.props.children;
  }
}

// Usage
<ErrorBoundary>
  <App />
</ErrorBoundary>
```

## TypeScript with React

### Component Types

```typescript
// Function component with props
interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({ label, onClick, disabled = false }) => {
  return <button onClick={onClick} disabled={disabled}>{label}</button>;
};

// Or without React.FC (preferred)
function Button({ label, onClick, disabled = false }: ButtonProps) {
  return <button onClick={onClick} disabled={disabled}>{label}</button>;
}
```

### Hooks with TypeScript

```typescript
// useState with type
const [count, setCount] = useState<number>(0);
const [user, setUser] = useState<User | null>(null);

// useRef with type
const inputRef = useRef<HTMLInputElement>(null);

// Custom hook with types
function useFetch<T>(url: string): {
  data: T | null;
  loading: boolean;
  error: Error | null;
} {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then((data: T) => setData(data))
      .catch(setError)
      .finally(() => setLoading(false));
  }, [url]);

  return { data, loading, error };
}
```

### Event Types

```typescript
function Input() {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log(e.target.value);
  };

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
  };

  return (
    <form onSubmit={handleSubmit}>
      <input onChange={handleChange} />
      <button onClick={handleClick}>Submit</button>
    </form>
  );
}
```

## React Router (v6)

```javascript
import { BrowserRouter, Routes, Route, Link, useParams, useNavigate } from 'react-router-dom';

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">Home</Link>
        <Link to="/about">About</Link>
        <Link to="/users/123">User 123</Link>
      </nav>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/users/:id" element={<User />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}

// Access URL parameters
function User() {
  const { id } = useParams();
  const navigate = useNavigate();

  return (
    <div>
      <h1>User {id}</h1>
      <button onClick={() => navigate('/about')}>Go to About</button>
    </div>
  );
}
```

## Concurrent Features (React 18+)

### Transitions

Mark non-urgent updates:

```javascript
import { useTransition } from 'react';

function SearchResults() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isPending, startTransition] = useTransition();

  const handleChange = (e) => {
    setQuery(e.target.value);

    // Mark this update as non-urgent
    startTransition(() => {
      setResults(filterResults(e.target.value));
    });
  };

  return (
    <>
      <input value={query} onChange={handleChange} />
      {isPending ? <Spinner /> : <ResultsList results={results} />}
    </>
  );
}
```

### Suspense for Data Fetching

```javascript
import { Suspense } from 'react';

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <UserProfile />
    </Suspense>
  );
}

// Component that suspends while loading
function UserProfile() {
  const user = use(fetchUser()); // Suspends until data loads
  return <div>{user.name}</div>;
}
```

## Testing

### React Testing Library

```javascript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

test('counter increments', () => {
  render(<Counter />);

  const button = screen.getByRole('button', { name: /increment/i });
  const count = screen.getByText(/count: 0/i);

  fireEvent.click(button);

  expect(screen.getByText(/count: 1/i)).toBeInTheDocument();
});

test('fetches and displays user', async () => {
  render(<UserProfile userId={1} />);

  expect(screen.getByText(/loading/i)).toBeInTheDocument();

  await waitFor(() => {
    expect(screen.getByText(/john doe/i)).toBeInTheDocument();
  });
});
```

### Component Testing

```javascript
import { render } from '@testing-library/react';

test('renders with props', () => {
  const { container } = render(
    <Button label="Click me" onClick={jest.fn()} />
  );

  expect(container.firstChild).toMatchSnapshot();
});
```

## Common Patterns

### Render Props

```javascript
function DataProvider({ render }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchData().then(setData);
  }, []);

  return render(data);
}

// Usage
<DataProvider render={(data) => <div>{data}</div>} />
```

### Higher-Order Components (HOC)

```javascript
function withAuth(Component) {
  return function AuthenticatedComponent(props) {
    const { user } = useAuth();

    if (!user) {
      return <Login />;
    }

    return <Component {...props} user={user} />;
  };
}

// Usage
const ProtectedPage = withAuth(Dashboard);
```

### Compound Components

```javascript
function Tabs({ children }) {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div>{children}</div>
    </TabsContext.Provider>
  );
}

Tabs.List = function TabsList({ children }) {
  return <div role="tablist">{children}</div>;
};

Tabs.Tab = function Tab({ index, children }) {
  const { activeTab, setActiveTab } = useContext(TabsContext);
  return (
    <button onClick={() => setActiveTab(index)}>
      {children}
    </button>
  );
};

Tabs.Panel = function TabPanel({ index, children }) {
  const { activeTab } = useContext(TabsContext);
  return activeTab === index ? <div>{children}</div> : null;
};

// Usage
<Tabs>
  <Tabs.List>
    <Tabs.Tab index={0}>Tab 1</Tabs.Tab>
    <Tabs.Tab index={1}>Tab 2</Tabs.Tab>
  </Tabs.List>
  <Tabs.Panel index={0}>Content 1</Tabs.Panel>
  <Tabs.Panel index={1}>Content 2</Tabs.Panel>
</Tabs>
```

## Portals

Render children outside parent DOM hierarchy:

```javascript
import { createPortal } from 'react-dom';

function Modal({ children, isOpen }) {
  if (!isOpen) return null;

  return createPortal(
    <div className="modal-overlay">
      <div className="modal-content">
        {children}
      </div>
    </div>,
    document.getElementById('modal-root')
  );
}
```

## Refs and Forward Refs

```javascript
import { forwardRef, useImperativeHandle, useRef } from 'react';

// Forward ref to child
const FancyInput = forwardRef((props, ref) => {
  return <input ref={ref} {...props} />;
});

// Usage
function Parent() {
  const inputRef = useRef();

  const focusInput = () => {
    inputRef.current.focus();
  };

  return <FancyInput ref={inputRef} />;
}

// Expose specific methods
const CustomInput = forwardRef((props, ref) => {
  const inputRef = useRef();

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current.focus(),
    clear: () => inputRef.current.value = ''
  }));

  return <input ref={inputRef} />;
});
```

## Common Anti-Patterns to Avoid

1. **Mutating state directly**: Use setState, never `state.value = x`
2. **Using index as key**: Causes re-render issues
3. **Forgetting useCallback dependencies**: Stale closures
4. **Too many useEffects**: Consider combining or custom hooks
5. **Props drilling**: Use Context or state management
6. **Large components**: Break into smaller, focused components
7. **Inline object/array creation in JSX**: Causes re-renders
8. **Not cleaning up effects**: Memory leaks in subscriptions

## ELI10

React is like LEGO blocks:
- Build reusable pieces (components)
- Combine to make complex UIs
- Reuse same piece many times
- Efficient updates when data changes!

## Further Resources

### Official Documentation
- [React Documentation](https://react.dev/) - Official React docs
- [React API Reference](https://react.dev/reference/react) - Complete API reference
- [React Hooks Reference](https://react.dev/reference/react/hooks) - All hooks documentation
- [React DevTools](https://react.dev/learn/react-developer-tools) - Browser extension

### Popular Libraries
- [React Router](https://reactrouter.com/) - Client-side routing
- [Redux Toolkit](https://redux-toolkit.js.org/) - State management
- [React Query](https://tanstack.com/query/latest) - Data fetching & caching
- [Zustand](https://zustand-demo.pmnd.rs/) - Lightweight state management
- [Jotai](https://jotai.org/) - Atomic state management
- [React Hook Form](https://react-hook-form.com/) - Form validation
- [React Testing Library](https://testing-library.com/react) - Testing utilities
- [Styled Components](https://styled-components.com/) - CSS-in-JS
- [Framer Motion](https://www.framer.com/motion/) - Animations

### Frameworks Built on React
- [Next.js](https://nextjs.org/) - Full-stack React framework with SSR/SSG
- [Remix](https://remix.run/) - Full-stack web framework
- [Gatsby](https://www.gatsbyjs.com/) - Static site generator
- [Expo](https://expo.dev/) - React Native for mobile apps

### Learning Resources
- [React Tutorial](https://react.dev/learn) - Official interactive tutorial
- [React Patterns](https://reactpatterns.com/) - Common design patterns
- [Awesome React](https://github.com/enaqx/awesome-react) - Curated list of resources
