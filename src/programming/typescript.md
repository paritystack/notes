# TypeScript

TypeScript is a strongly-typed superset of JavaScript developed by Microsoft that compiles to plain JavaScript. It adds optional static typing, classes, interfaces, and other features to JavaScript, making it easier to build and maintain large-scale applications.

## Table of Contents
- [Why TypeScript?](#why-typescript)
- [Basic Types](#basic-types)
- [Interfaces](#interfaces)
- [Type Aliases](#type-aliases)
- [Union and Intersection Types](#union-and-intersection-types)
- [Generics](#generics)
- [Classes](#classes)
- [Enums](#enums)
- [Type Assertions](#type-assertions)
- [Type Guards](#type-guards)
- [Utility Types](#utility-types)
- [TypeScript with React](#typescript-with-react)
- [TypeScript with Node.js](#typescript-with-nodejs)
- [Configuration (tsconfig.json)](#configuration-tsconfigjson)
- [Advanced Types](#advanced-types)
- [Best Practices](#best-practices)

---

## Why TypeScript?

**Benefits:**
- **Type Safety**: Catch errors at compile-time instead of runtime
- **Better IDE Support**: Enhanced autocomplete, navigation, and refactoring
- **Self-Documenting**: Types serve as inline documentation
- **Scalability**: Easier to maintain large codebases
- **Modern JavaScript**: Use latest JavaScript features with backward compatibility
- **Refactoring Confidence**: Safe refactoring with type checking

**When to Use:**
- Large-scale applications
- Team projects with multiple developers
- Projects requiring long-term maintenance
- When you need robust IDE support
- Enterprise applications

---

## Basic Types

### Primitive Types

```typescript
// Boolean
let isDone: boolean = false;

// Number
let decimal: number = 6;
let hex: number = 0xf00d;
let binary: number = 0b1010;
let octal: number = 0o744;

// String
let color: string = "blue";
let fullName: string = `Bob Bobbington`;
let sentence: string = `Hello, my name is ${fullName}.`;

// Array
let list: number[] = [1, 2, 3];
let list2: Array<number> = [1, 2, 3]; // Generic syntax

// Tuple - fixed-length array with known types
let x: [string, number];
x = ["hello", 10]; // OK
// x = [10, "hello"]; // Error

// Enum
enum Color {
  Red,
  Green,
  Blue,
}
let c: Color = Color.Green;

// Any - opt-out of type checking
let notSure: any = 4;
notSure = "maybe a string instead";
notSure = false; // OK

// Unknown - type-safe alternative to any
let userInput: unknown;
userInput = 5;
userInput = "hello";
// let str: string = userInput; // Error
if (typeof userInput === "string") {
  let str: string = userInput; // OK
}

// Void - absence of any type (typically for functions)
function warnUser(): void {
  console.log("This is a warning message");
}

// Null and Undefined
let u: undefined = undefined;
let n: null = null;

// Never - represents values that never occur
function error(message: string): never {
  throw new Error(message);
}

function infiniteLoop(): never {
  while (true) {}
}
```

---

## Interfaces

Interfaces define the structure of objects and enforce contracts in your code.

### Basic Interface

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  age?: number; // Optional property
  readonly createdAt: Date; // Read-only property
}

const user: User = {
  id: 1,
  name: "John Doe",
  email: "john@example.com",
  createdAt: new Date(),
};

// user.createdAt = new Date(); // Error: Cannot assign to 'createdAt'
```

### Function Types

```typescript
interface SearchFunc {
  (source: string, subString: string): boolean;
}

const mySearch: SearchFunc = (source, subString) => {
  return source.includes(subString);
};
```

### Indexable Types

```typescript
interface StringArray {
  [index: number]: string;
}

let myArray: StringArray = ["Bob", "Fred"];
let myStr: string = myArray[0];

interface NumberDictionary {
  [key: string]: number;
}

let dict: NumberDictionary = {
  age: 25,
  height: 180,
};
```

### Extending Interfaces

```typescript
interface Shape {
  color: string;
}

interface Square extends Shape {
  sideLength: number;
}

let square: Square = {
  color: "blue",
  sideLength: 10,
};

// Multiple inheritance
interface PenStroke {
  penWidth: number;
}

interface FilledSquare extends Square, PenStroke {
  filled: boolean;
}
```

### Implementing Interfaces

```typescript
interface ClockInterface {
  currentTime: Date;
  setTime(d: Date): void;
}

class Clock implements ClockInterface {
  currentTime: Date = new Date();

  setTime(d: Date): void {
    this.currentTime = d;
  }
}
```

---

## Type Aliases

Type aliases create a new name for a type. Similar to interfaces but more flexible.

```typescript
// Basic type alias
type ID = string | number;

type Point = {
  x: number;
  y: number;
};

// Union type
type Result = Success | Failure;

type Success = {
  status: "success";
  data: any;
};

type Failure = {
  status: "error";
  error: string;
};

// Function type
type GreetFunction = (name: string) => string;

const greet: GreetFunction = (name) => `Hello, ${name}!`;

// Intersection type
type Admin = {
  privileges: string[];
};

type Employee = {
  name: string;
  startDate: Date;
};

type AdminEmployee = Admin & Employee;

const ae: AdminEmployee = {
  privileges: ["create-server"],
  name: "Max",
  startDate: new Date(),
};
```

### Interface vs Type Alias

```typescript
// Interfaces can be merged (declaration merging)
interface Window {
  title: string;
}

interface Window {
  ts: number;
}

// Type aliases cannot be merged
// type Window = { title: string };
// type Window = { ts: number }; // Error: Duplicate identifier

// Type aliases can represent unions and tuples
type StringOrNumber = string | number;
type Tuple = [string, number];

// Both can be extended
interface Shape {
  color: string;
}

// Interface extending interface
interface Circle extends Shape {
  radius: number;
}

// Type extending type
type ColoredShape = Shape & { filled: boolean };

// Type extending interface
type ColoredCircle = Circle & { filled: boolean };

// Interface extending type
type Size = { width: number; height: number };
interface Rectangle extends Size {
  color: string;
}
```

---

## Union and Intersection Types

### Union Types

A union type can be one of several types.

```typescript
function printId(id: number | string) {
  console.log("Your ID is: " + id);
}

printId(101); // OK
printId("202"); // OK
// printId({ myID: 22342 }); // Error

// Discriminated Unions (Tagged Unions)
type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "square"; sideLength: number }
  | { kind: "rectangle"; width: number; height: number };

function getArea(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "square":
      return shape.sideLength ** 2;
    case "rectangle":
      return shape.width * shape.height;
  }
}
```

### Intersection Types

An intersection type combines multiple types into one.

```typescript
interface Colorful {
  color: string;
}

interface Circle {
  radius: number;
}

type ColorfulCircle = Colorful & Circle;

const cc: ColorfulCircle = {
  color: "red",
  radius: 42,
};
```

---

## Generics

Generics allow you to create reusable components that work with multiple types.

### Generic Functions

```typescript
function identity<T>(arg: T): T {
  return arg;
}

let output1 = identity<string>("myString");
let output2 = identity<number>(123);
let output3 = identity("myString"); // Type inference

// Generic with constraints
interface Lengthwise {
  length: number;
}

function loggingIdentity<T extends Lengthwise>(arg: T): T {
  console.log(arg.length);
  return arg;
}

loggingIdentity({ length: 10, value: 3 }); // OK
loggingIdentity([1, 2, 3]); // OK
// loggingIdentity(3); // Error: number doesn't have length
```

### Generic Interfaces

```typescript
interface GenericIdentityFn<T> {
  (arg: T): T;
}

let myIdentity: GenericIdentityFn<number> = identity;

// Generic container
interface Container<T> {
  value: T;
  getValue(): T;
  setValue(value: T): void;
}

class Box<T> implements Container<T> {
  constructor(public value: T) {}

  getValue(): T {
    return this.value;
  }

  setValue(value: T): void {
    this.value = value;
  }
}

const numberBox = new Box<number>(42);
const stringBox = new Box<string>("hello");
```

### Generic Classes

```typescript
class GenericNumber<T> {
  zeroValue: T;
  add: (x: T, y: T) => T;
}

let myGenericNumber = new GenericNumber<number>();
myGenericNumber.zeroValue = 0;
myGenericNumber.add = (x, y) => x + y;

let stringNumeric = new GenericNumber<string>();
stringNumeric.zeroValue = "";
stringNumeric.add = (x, y) => x + y;
```

### Generic Constraints

```typescript
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

let x = { a: 1, b: 2, c: 3, d: 4 };

getProperty(x, "a"); // OK
// getProperty(x, "m"); // Error: "m" is not in 'a' | 'b' | 'c' | 'd'
```

### Advanced Generic Patterns

```typescript
// Generic type with default
type APIResponse<T = any> = {
  data: T;
  status: number;
  message: string;
};

// Multiple type parameters
function merge<T, U>(obj1: T, obj2: U): T & U {
  return { ...obj1, ...obj2 };
}

const merged = merge({ name: "John" }, { age: 30 });
// merged: { name: string } & { age: number }

// Conditional types with generics
type NonNullable<T> = T extends null | undefined ? never : T;

type A = NonNullable<string | null>; // string
type B = NonNullable<number | undefined>; // number
```

---

## Classes

### Basic Class

```typescript
class Greeter {
  greeting: string;

  constructor(message: string) {
    this.greeting = message;
  }

  greet(): string {
    return `Hello, ${this.greeting}`;
  }
}

let greeter = new Greeter("world");
```

### Inheritance

```typescript
class Animal {
  name: string;

  constructor(name: string) {
    this.name = name;
  }

  move(distanceInMeters: number = 0): void {
    console.log(`${this.name} moved ${distanceInMeters}m.`);
  }
}

class Dog extends Animal {
  bark(): void {
    console.log("Woof! Woof!");
  }
}

const dog = new Dog("Buddy");
dog.bark();
dog.move(10);
```

### Access Modifiers

```typescript
class Person {
  public name: string; // Public by default
  private age: number; // Only accessible within the class
  protected email: string; // Accessible in class and subclasses
  readonly id: number; // Cannot be modified after initialization

  constructor(name: string, age: number, email: string, id: number) {
    this.name = name;
    this.age = age;
    this.email = email;
    this.id = id;
  }

  getAge(): number {
    return this.age;
  }
}

class Employee extends Person {
  constructor(name: string, age: number, email: string, id: number) {
    super(name, age, email, id);
  }

  getEmail(): string {
    return this.email; // OK: protected is accessible in subclass
  }
}

const person = new Person("John", 30, "john@example.com", 1);
console.log(person.name); // OK
// console.log(person.age); // Error: private
// console.log(person.email); // Error: protected
```

### Getters and Setters

```typescript
class Employee {
  private _fullName: string = "";

  get fullName(): string {
    return this._fullName;
  }

  set fullName(newName: string) {
    if (newName && newName.length > 0) {
      this._fullName = newName;
    } else {
      throw new Error("Invalid name");
    }
  }
}

let employee = new Employee();
employee.fullName = "Bob Smith";
console.log(employee.fullName);
```

### Abstract Classes

```typescript
abstract class Department {
  constructor(public name: string) {}

  printName(): void {
    console.log("Department name: " + this.name);
  }

  abstract printMeeting(): void; // Must be implemented in derived class
}

class AccountingDepartment extends Department {
  constructor() {
    super("Accounting and Auditing");
  }

  printMeeting(): void {
    console.log("The Accounting Department meets each Monday at 10am.");
  }

  generateReports(): void {
    console.log("Generating accounting reports...");
  }
}

let department: Department = new AccountingDepartment();
department.printName();
department.printMeeting();
// department.generateReports(); // Error: method doesn't exist on Department
```

### Static Members

```typescript
class Grid {
  static origin = { x: 0, y: 0 };

  calculateDistanceFromOrigin(point: { x: number; y: number }): number {
    let xDist = point.x - Grid.origin.x;
    let yDist = point.y - Grid.origin.y;
    return Math.sqrt(xDist * xDist + yDist * yDist);
  }
}

console.log(Grid.origin);
let grid = new Grid();
```

---

## Enums

Enums allow defining a set of named constants.

### Numeric Enums

```typescript
enum Direction {
  Up = 1,
  Down,
  Left,
  Right,
}

// Starts from 1 and auto-increments
console.log(Direction.Up); // 1
console.log(Direction.Down); // 2

enum Response {
  No = 0,
  Yes = 1,
}

function respond(recipient: string, message: Response): void {
  // ...
}

respond("Princess Caroline", Response.Yes);
```

### String Enums

```typescript
enum Direction {
  Up = "UP",
  Down = "DOWN",
  Left = "LEFT",
  Right = "RIGHT",
}

console.log(Direction.Up); // "UP"
```

### Const Enums

```typescript
const enum Enum {
  A = 1,
  B = A * 2,
}

// Compiled code is inlined (better performance)
let value = Enum.B; // Becomes: let value = 2;
```

### Enum as Type

```typescript
enum Status {
  Active,
  Inactive,
  Pending,
}

interface User {
  name: string;
  status: Status;
}

const user: User = {
  name: "John",
  status: Status.Active,
};
```

---

## Type Assertions

Type assertions tell the compiler to treat a value as a specific type.

```typescript
// Angle-bracket syntax
let someValue: any = "this is a string";
let strLength: number = (<string>someValue).length;

// As syntax (preferred in JSX/TSX)
let someValue2: any = "this is a string";
let strLength2: number = (someValue2 as string).length;

// Non-null assertion operator
function liveDangerously(x?: number | null) {
  // TypeScript will trust that x is not null/undefined
  console.log(x!.toFixed());
}

// Const assertions
let x = "hello" as const; // Type: "hello" (not string)

let y = [10, 20] as const; // Type: readonly [10, 20]

let z = {
  name: "John",
  age: 30,
} as const; // All properties are readonly
```

---

## Type Guards

Type guards allow you to narrow down the type of a variable within a conditional block.

### typeof Guards

```typescript
function padLeft(value: string, padding: string | number) {
  if (typeof padding === "number") {
    return Array(padding + 1).join(" ") + value;
  }
  if (typeof padding === "string") {
    return padding + value;
  }
  throw new Error(`Expected string or number, got '${typeof padding}'.`);
}
```

### instanceof Guards

```typescript
class Bird {
  fly() {
    console.log("Flying");
  }
}

class Fish {
  swim() {
    console.log("Swimming");
  }
}

function move(animal: Bird | Fish) {
  if (animal instanceof Bird) {
    animal.fly();
  } else {
    animal.swim();
  }
}
```

### in Operator

```typescript
type Fish = { swim: () => void };
type Bird = { fly: () => void };

function move(animal: Fish | Bird) {
  if ("swim" in animal) {
    animal.swim();
  } else {
    animal.fly();
  }
}
```

### Custom Type Guards

```typescript
interface Cat {
  meow(): void;
}

interface Dog {
  bark(): void;
}

function isCat(pet: Cat | Dog): pet is Cat {
  return (pet as Cat).meow !== undefined;
}

function makeSound(pet: Cat | Dog) {
  if (isCat(pet)) {
    pet.meow();
  } else {
    pet.bark();
  }
}
```

---

## Utility Types

TypeScript provides several utility types for common type transformations.

### Partial<T>

Makes all properties optional.

```typescript
interface User {
  id: number;
  name: string;
  email: string;
}

function updateUser(user: User, updates: Partial<User>): User {
  return { ...user, ...updates };
}

const user: User = { id: 1, name: "John", email: "john@example.com" };
const updated = updateUser(user, { name: "Jane" });
```

### Required<T>

Makes all properties required.

```typescript
interface Props {
  a?: number;
  b?: string;
}

const obj: Required<Props> = { a: 5, b: "text" };
```

### Readonly<T>

Makes all properties readonly.

```typescript
interface User {
  name: string;
  age: number;
}

const user: Readonly<User> = {
  name: "John",
  age: 30,
};

// user.name = "Jane"; // Error: Cannot assign to 'name'
```

### Pick<T, K>

Creates a type by picking specific properties from another type.

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  age: number;
}

type UserPreview = Pick<User, "id" | "name">;
// { id: number; name: string; }

const preview: UserPreview = { id: 1, name: "John" };
```

### Omit<T, K>

Creates a type by omitting specific properties.

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  password: string;
}

type UserPublic = Omit<User, "password">;
// { id: number; name: string; email: string; }
```

### Record<K, T>

Creates an object type with keys of type K and values of type T.

```typescript
type PageInfo = {
  title: string;
  url: string;
};

type Page = "home" | "about" | "contact";

const pages: Record<Page, PageInfo> = {
  home: { title: "Home", url: "/" },
  about: { title: "About", url: "/about" },
  contact: { title: "Contact", url: "/contact" },
};
```

### Exclude<T, U> and Extract<T, U>

```typescript
type T0 = Exclude<"a" | "b" | "c", "a">; // "b" | "c"
type T1 = Exclude<string | number | (() => void), Function>; // string | number

type T2 = Extract<"a" | "b" | "c", "a" | "f">; // "a"
type T3 = Extract<string | number | (() => void), Function>; // () => void
```

### ReturnType<T>

Extracts the return type of a function type.

```typescript
function getUser() {
  return { id: 1, name: "John", email: "john@example.com" };
}

type User = ReturnType<typeof getUser>;
// { id: number; name: string; email: string; }
```

### Parameters<T>

Extracts parameter types of a function type as a tuple.

```typescript
function createUser(name: string, age: number, email: string) {
  return { name, age, email };
}

type CreateUserParams = Parameters<typeof createUser>;
// [name: string, age: number, email: string]
```

---

## TypeScript with React

### Functional Components

```typescript
import React from "react";

// Props interface
interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: "primary" | "secondary";
}

// Functional component
const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  disabled = false,
  variant = "primary",
}) => {
  return (
    <button onClick={onClick} disabled={disabled} className={variant}>
      {label}
    </button>
  );
};

// Alternative (recommended in modern React)
function Button2(props: ButtonProps) {
  return <button {...props}>{props.label}</button>;
}

export default Button;
```

### Component with Children

```typescript
interface CardProps {
  title: string;
  children: React.ReactNode;
}

const Card: React.FC<CardProps> = ({ title, children }) => {
  return (
    <div className="card">
      <h2>{title}</h2>
      <div className="card-body">{children}</div>
    </div>
  );
};
```

### useState Hook

```typescript
import { useState } from "react";

interface User {
  id: number;
  name: string;
}

function UserComponent() {
  // Type inference
  const [count, setCount] = useState(0);

  // Explicit type
  const [user, setUser] = useState<User | null>(null);

  // With initial state
  const [users, setUsers] = useState<User[]>([]);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(count + 1)}>Increment</button>
    </div>
  );
}
```

### useEffect Hook

```typescript
import { useEffect, useState } from "react";

function DataFetcher() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    async function fetchData() {
      const response = await fetch("/api/data");
      const result = await response.json();
      setData(result);
    }

    fetchData();
  }, []); // Empty dependency array

  return <div>{data ? JSON.stringify(data) : "Loading..."}</div>;
}
```

### useRef Hook

```typescript
import { useRef, useEffect } from "react";

function TextInput() {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return <input ref={inputRef} type="text" />;
}
```

### useContext Hook

```typescript
import { createContext, useContext, useState } from "react";

interface AuthContextType {
  user: User | null;
  login: (user: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [user, setUser] = useState<User | null>(null);

  const login = (user: User) => setUser(user);
  const logout = () => setUser(null);

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
};
```

### Custom Hooks

```typescript
import { useState, useEffect } from "react";

function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch(url);
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err as Error);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [url]);

  return { data, loading, error };
}

// Usage
interface User {
  id: number;
  name: string;
}

function UserList() {
  const { data: users, loading, error } = useFetch<User[]>("/api/users");

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <ul>
      {users?.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

### Event Handlers

```typescript
import React from "react";

function Form() {
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    // Handle form submission
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log(e.target.value);
  };

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log("Button clicked");
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" onChange={handleChange} />
      <button onClick={handleClick}>Submit</button>
    </form>
  );
}
```

---

## TypeScript with Node.js

### Basic Express Server

```typescript
import express, { Request, Response, NextFunction } from "express";

const app = express();
const PORT = 3000;

app.use(express.json());

// Basic route
app.get("/", (req: Request, res: Response) => {
  res.json({ message: "Hello World" });
});

// Route with params
app.get("/users/:id", (req: Request, res: Response) => {
  const userId = req.params.id;
  res.json({ id: userId });
});

// POST route with body
interface CreateUserBody {
  name: string;
  email: string;
}

app.post("/users", (req: Request<{}, {}, CreateUserBody>, res: Response) => {
  const { name, email } = req.body;
  res.json({ id: 1, name, email });
});

// Middleware
const logger = (req: Request, res: Response, next: NextFunction) => {
  console.log(`${req.method} ${req.path}`);
  next();
};

app.use(logger);

// Error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error(err.stack);
  res.status(500).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### Custom Request Types

```typescript
import { Request } from "express";

interface UserRequest extends Request {
  user?: {
    id: number;
    email: string;
  };
}

app.get("/profile", (req: UserRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  res.json(req.user);
});
```

### Async/Await with Express

```typescript
import { Request, Response } from "express";

// Wrapper for async route handlers
const asyncHandler =
  (fn: Function) => (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };

app.get(
  "/users",
  asyncHandler(async (req: Request, res: Response) => {
    const users = await getUsersFromDB();
    res.json(users);
  })
);
```

### File System Operations

```typescript
import * as fs from "fs/promises";
import * as path from "path";

async function readConfig(): Promise<any> {
  try {
    const configPath = path.join(__dirname, "config.json");
    const data = await fs.readFile(configPath, "utf-8");
    return JSON.parse(data);
  } catch (error) {
    console.error("Error reading config:", error);
    throw error;
  }
}

async function writeLog(message: string): Promise<void> {
  const logPath = path.join(__dirname, "app.log");
  const timestamp = new Date().toISOString();
  const logEntry = `[${timestamp}] ${message}\n`;
  await fs.appendFile(logPath, logEntry);
}
```

---

## Configuration (tsconfig.json)

### Basic Configuration

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "moduleResolution": "node",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### Important Compiler Options

#### Strict Type Checking

```json
{
  "compilerOptions": {
    "strict": true,               // Enable all strict type checking
    "noImplicitAny": true,        // Error on expressions with implied 'any'
    "strictNullChecks": true,     // Enable strict null checks
    "strictFunctionTypes": true,  // Enable strict checking of function types
    "strictBindCallApply": true,  // Enable strict bind/call/apply methods
    "strictPropertyInitialization": true, // Ensure properties are initialized
    "noImplicitThis": true,       // Error on 'this' expressions with implied 'any'
    "alwaysStrict": true          // Parse in strict mode and emit "use strict"
  }
}
```

#### Module Resolution

```json
{
  "compilerOptions": {
    "module": "commonjs",         // Module code generation
    "moduleResolution": "node",   // Module resolution strategy
    "baseUrl": "./",              // Base directory for module resolution
    "paths": {                    // Path mappings
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"]
    },
    "esModuleInterop": true,      // Emit helpers for importing CommonJS modules
    "allowSyntheticDefaultImports": true  // Allow default imports from modules
  }
}
```

#### React Configuration

```json
{
  "compilerOptions": {
    "jsx": "react-jsx",           // JSX code generation (React 17+)
    // "jsx": "react",            // For React 16 and earlier
    "lib": ["DOM", "DOM.Iterable", "ES2020"]
  }
}
```

#### Node.js Configuration

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "types": ["node"],
    "esModuleInterop": true
  }
}
```

### Project References

For monorepos or multi-package projects:

```json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "declarationMap": true
  },
  "references": [
    { "path": "../common" },
    { "path": "../utils" }
  ]
}
```

---

## Advanced Types

### Conditional Types

```typescript
type IsString<T> = T extends string ? true : false;

type A = IsString<string>; // true
type B = IsString<number>; // false

// Distributive conditional types
type ToArray<T> = T extends any ? T[] : never;
type StrOrNumArray = ToArray<string | number>; // string[] | number[]

// Infer keyword
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type Func = () => number;
type Result = ReturnType<Func>; // number
```

### Mapped Types

```typescript
type Readonly<T> = {
  readonly [P in keyof T]: T[P];
};

type Optional<T> = {
  [P in keyof T]?: T[P];
};

type Nullable<T> = {
  [P in keyof T]: T[P] | null;
};

interface Person {
  name: string;
  age: number;
}

type ReadonlyPerson = Readonly<Person>;
// { readonly name: string; readonly age: number; }

type OptionalPerson = Optional<Person>;
// { name?: string; age?: number; }
```

### Template Literal Types

```typescript
type EventName = "click" | "scroll" | "mousemove";
type Handler = `on${Capitalize<EventName>}`;
// "onClick" | "onScroll" | "onMousemove"

type PropEventSource<Type> = {
  on<Key extends string & keyof Type>(
    eventName: `${Key}Changed`,
    callback: (newValue: Type[Key]) => void
  ): void;
};

declare function makeWatchedObject<Type>(
  obj: Type
): Type & PropEventSource<Type>;

const person = makeWatchedObject({
  firstName: "John",
  age: 26,
});

person.on("firstNameChanged", (newName) => {
  console.log(`New name: ${newName}`);
});
```

### Index Signatures

```typescript
interface StringArray {
  [index: number]: string;
}

interface StringByString {
  [key: string]: string | number;
  length: number; // OK: number is assignable to string | number
}

// Generic index signature
interface Dictionary<T> {
  [key: string]: T;
}

const userScores: Dictionary<number> = {
  john: 100,
  jane: 95,
};
```

### Discriminated Unions (Tagged Unions)

```typescript
interface Square {
  kind: "square";
  size: number;
}

interface Rectangle {
  kind: "rectangle";
  width: number;
  height: number;
}

interface Circle {
  kind: "circle";
  radius: number;
}

type Shape = Square | Rectangle | Circle;

function area(s: Shape): number {
  switch (s.kind) {
    case "square":
      return s.size * s.size;
    case "rectangle":
      return s.width * s.height;
    case "circle":
      return Math.PI * s.radius ** 2;
  }
}
```

---

## Best Practices

### 1. Use Strict Mode

Always enable `strict: true` in `tsconfig.json` for maximum type safety.

```json
{
  "compilerOptions": {
    "strict": true
  }
}
```

### 2. Avoid `any` Type

Use `unknown` instead of `any` when the type is truly unknown.

```typescript
// Bad
function process(data: any) {
  return data.value;
}

// Good
function process(data: unknown) {
  if (typeof data === "object" && data !== null && "value" in data) {
    return (data as { value: any }).value;
  }
  throw new Error("Invalid data");
}
```

### 3. Use Type Inference

Let TypeScript infer types when possible.

```typescript
// Bad
const numbers: number[] = [1, 2, 3];
const result: number = numbers.reduce((acc: number, n: number) => acc + n, 0);

// Good
const numbers = [1, 2, 3];
const result = numbers.reduce((acc, n) => acc + n, 0);
```

### 4. Use Readonly When Appropriate

```typescript
// Readonly arrays
const numbers: readonly number[] = [1, 2, 3];
// numbers.push(4); // Error

// Readonly objects
interface Config {
  readonly apiUrl: string;
  readonly timeout: number;
}

// Readonly function parameters
function printList(list: readonly string[]) {
  // list.push("new"); // Error
  console.log(list.join(", "));
}
```

### 5. Prefer Interfaces for Objects, Types for Unions/Intersections

```typescript
// Good: Use interface for object shapes
interface User {
  id: number;
  name: string;
}

// Good: Use type for unions
type Status = "pending" | "approved" | "rejected";

// Good: Use type for intersections
type AdminUser = User & { role: "admin" };
```

### 6. Use Discriminated Unions for Complex State

```typescript
type RequestState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: any }
  | { status: "error"; error: string };

function handleRequest(state: RequestState) {
  switch (state.status) {
    case "idle":
      return "Not started";
    case "loading":
      return "Loading...";
    case "success":
      return state.data;
    case "error":
      return state.error;
  }
}
```

### 7. Use Const Assertions

```typescript
// Without const assertion
const colors = ["red", "green", "blue"];
// Type: string[]

// With const assertion
const colors = ["red", "green", "blue"] as const;
// Type: readonly ["red", "green", "blue"]

const config = {
  apiUrl: "https://api.example.com",
  timeout: 5000,
} as const;
// All properties are readonly
```

### 8. Use Type Guards

```typescript
function isString(value: unknown): value is string {
  return typeof value === "string";
}

function processValue(value: string | number) {
  if (isString(value)) {
    console.log(value.toUpperCase());
  } else {
    console.log(value.toFixed(2));
  }
}
```

### 9. Use Generics for Reusable Code

```typescript
// Generic function
function firstOrNull<T>(arr: T[]): T | null {
  return arr.length > 0 ? arr[0] : null;
}

// Generic constraints
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}
```

### 10. Use Utility Types

```typescript
// Instead of manually creating partial types
interface User {
  id: number;
  name: string;
  email: string;
}

// Good: Use Partial utility type
function updateUser(id: number, updates: Partial<User>) {
  // Implementation
}

// Good: Use Pick for selecting specific properties
type UserPreview = Pick<User, "id" | "name">;

// Good: Use Omit to exclude properties
type UserWithoutId = Omit<User, "id">;
```

### 11. Avoid Type Assertions When Possible

```typescript
// Bad
const data = JSON.parse(jsonString) as User;

// Good: Validate at runtime
function isUser(data: any): data is User {
  return (
    typeof data === "object" &&
    typeof data.id === "number" &&
    typeof data.name === "string"
  );
}

const data = JSON.parse(jsonString);
if (isUser(data)) {
  // TypeScript knows data is User here
}
```

### 12. Use Enum Alternatives

```typescript
// Instead of enum
enum Status {
  Pending,
  Approved,
  Rejected,
}

// Consider union types
type Status = "pending" | "approved" | "rejected";

// Or const objects with 'as const'
const Status = {
  Pending: "pending",
  Approved: "approved",
  Rejected: "rejected",
} as const;

type StatusValue = (typeof Status)[keyof typeof Status];
```

### 13. Document Complex Types

```typescript
/**
 * Represents a user in the system
 * @property id - Unique identifier
 * @property name - Full name of the user
 * @property email - Contact email address
 */
interface User {
  id: number;
  name: string;
  email: string;
}

/**
 * Fetches user data from the API
 * @param userId - The ID of the user to fetch
 * @returns Promise resolving to user data
 * @throws {Error} When user is not found
 */
async function fetchUser(userId: number): Promise<User> {
  // Implementation
}
```

### 14. Use Namespace for Organization (Sparingly)

```typescript
namespace Validation {
  export interface StringValidator {
    isValid(s: string): boolean;
  }

  export class EmailValidator implements StringValidator {
    isValid(s: string): boolean {
      return /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/.test(s);
    }
  }
}

const validator = new Validation.EmailValidator();
```

### 15. Leverage TSConfig Paths

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@components/*": ["src/components/*"],
      "@utils/*": ["src/utils/*"],
      "@models/*": ["src/models/*"]
    }
  }
}
```

```typescript
// Instead of
import { Button } from "../../../components/Button";

// Use
import { Button } from "@components/Button";
```

---

## Common Patterns

### Factory Pattern

```typescript
interface Product {
  operation(): string;
}

class ConcreteProductA implements Product {
  operation(): string {
    return "Product A";
  }
}

class ConcreteProductB implements Product {
  operation(): string {
    return "Product B";
  }
}

class ProductFactory {
  createProduct(type: "A" | "B"): Product {
    switch (type) {
      case "A":
        return new ConcreteProductA();
      case "B":
        return new ConcreteProductB();
    }
  }
}
```

### Builder Pattern

```typescript
class QueryBuilder {
  private query: string = "";

  select(...fields: string[]): this {
    this.query += `SELECT ${fields.join(", ")} `;
    return this;
  }

  from(table: string): this {
    this.query += `FROM ${table} `;
    return this;
  }

  where(condition: string): this {
    this.query += `WHERE ${condition} `;
    return this;
  }

  build(): string {
    return this.query.trim();
  }
}

const query = new QueryBuilder()
  .select("id", "name")
  .from("users")
  .where("age > 18")
  .build();
```

### Singleton Pattern

```typescript
class Database {
  private static instance: Database;
  private connection: any;

  private constructor() {
    // Private constructor prevents instantiation
    this.connection = this.connect();
  }

  private connect() {
    // Connection logic
    return {};
  }

  static getInstance(): Database {
    if (!Database.instance) {
      Database.instance = new Database();
    }
    return Database.instance;
  }

  query(sql: string) {
    // Query logic
  }
}

const db1 = Database.getInstance();
const db2 = Database.getInstance();
console.log(db1 === db2); // true
```

---

## Resources

- **Official Documentation**: [https://www.typescriptlang.org/docs/](https://www.typescriptlang.org/docs/)
- **TypeScript Playground**: [https://www.typescriptlang.org/play](https://www.typescriptlang.org/play)
- **Definitely Typed**: [https://github.com/DefinitelyTyped/DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped)
- **TypeScript Deep Dive**: [https://basarat.gitbook.io/typescript/](https://basarat.gitbook.io/typescript/)
- **React TypeScript Cheatsheet**: [https://react-typescript-cheatsheet.netlify.app/](https://react-typescript-cheatsheet.netlify.app/)

---

TypeScript significantly improves the development experience by catching errors early, providing better tooling support, and making code more maintainable. The initial learning curve is worth the long-term benefits, especially for large-scale applications and team projects.
