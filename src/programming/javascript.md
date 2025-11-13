# JavaScript Programming

## Overview

JavaScript is a high-level, interpreted programming language primarily used for web development. It enables interactive web pages and is an essential part of web applications alongside HTML and CSS.

**Key Features:**
- Event-driven, functional, and imperative programming styles
- Dynamic typing
- Prototype-based object-orientation
- First-class functions
- Runs in browsers and on servers (Node.js)
- Asynchronous programming with Promises and async/await

---

## Basic Syntax

### Variables

```javascript
// var (function-scoped, avoid in modern code)
var x = 10;

// let (block-scoped, can be reassigned)
let y = 20;
y = 30;  // OK

// const (block-scoped, cannot be reassigned)
const z = 40;
// z = 50;  // ERROR!

// But const objects can be modified
const obj = { name: "Alice" };
obj.name = "Bob";  // OK
obj.age = 30;      // OK
```

### Data Types

```javascript
// Primitives
let num = 42;                    // Number
let str = "Hello";               // String
let bool = true;                 // Boolean
let undef = undefined;           // Undefined
let nul = null;                  // Null
let sym = Symbol("id");          // Symbol (ES6)
let bigInt = 123n;               // BigInt (ES2020)

// Objects
let obj = { name: "Alice" };
let arr = [1, 2, 3];
let func = function() {};

// Type checking
typeof num;      // "number"
typeof str;      // "string"
typeof obj;      // "object"
typeof arr;      // "object" (arrays are objects)
Array.isArray(arr);  // true

// Type conversion
String(42);      // "42"
Number("42");    // 42
parseInt("42");  // 42
parseFloat("3.14");  // 3.14
Boolean(0);      // false
Boolean(1);      // true
```

### Template Literals (ES6)

```javascript
const name = "Alice";
const age = 30;

// Template literals
const message = `Hello, ${name}! You are ${age} years old.`;

// Multi-line strings
const multiline = `
  This is a
  multi-line
  string
`;

// Tagged templates
function highlight(strings, ...values) {
  return strings.reduce((acc, str, i) => {
    return acc + str + (values[i] ? `<strong>${values[i]}</strong>` : '');
  }, '');
}

const result = highlight`Name: ${name}, Age: ${age}`;
```

---

## Arrays

```javascript
// Creating arrays
const arr = [1, 2, 3, 4, 5];
const mixed = [1, "hello", true, null, { name: "Alice" }];
const empty = [];

// Accessing elements
const first = arr[0];        // 1
const last = arr[arr.length - 1];  // 5

// Common methods
arr.push(6);                 // Add to end: [1, 2, 3, 4, 5, 6]
arr.pop();                   // Remove from end: 6
arr.unshift(0);              // Add to start: [0, 1, 2, 3, 4, 5]
arr.shift();                 // Remove from start: 0
arr.splice(2, 1);            // Remove 1 element at index 2
arr.slice(1, 3);             // Extract [2, 3]

// Iteration methods
arr.forEach((item, index) => {
  console.log(index, item);
});

// Map (transform array)
const squares = arr.map(x => x * x);

// Filter (select elements)
const evens = arr.filter(x => x % 2 === 0);

// Reduce (aggregate)
const sum = arr.reduce((acc, val) => acc + val, 0);

// Find
const found = arr.find(x => x > 3);      // First element > 3
const foundIndex = arr.findIndex(x => x > 3);

// Some and Every
const hasEven = arr.some(x => x % 2 === 0);     // true if any even
const allEven = arr.every(x => x % 2 === 0);    // true if all even

// Sorting
arr.sort((a, b) => a - b);   // Ascending
arr.sort((a, b) => b - a);   // Descending

// Spread operator
const arr1 = [1, 2, 3];
const arr2 = [4, 5, 6];
const combined = [...arr1, ...arr2];  // [1, 2, 3, 4, 5, 6]

// Destructuring
const [first, second, ...rest] = [1, 2, 3, 4, 5];
// first = 1, second = 2, rest = [3, 4, 5]
```

---

## Objects

```javascript
// Creating objects
const person = {
  name: "Alice",
  age: 30,
  greet() {
    return `Hello, I'm ${this.name}`;
  }
};

// Accessing properties
person.name;           // "Alice"
person["age"];         // 30

// Adding/modifying properties
person.email = "alice@example.com";
person.age = 31;

// Deleting properties
delete person.email;

// Object methods
Object.keys(person);       // ["name", "age", "greet"]
Object.values(person);     // ["Alice", 31, function]
Object.entries(person);    // [["name", "Alice"], ["age", 31], ...]

// Spread operator
const person2 = { ...person, city: "NYC" };

// Destructuring
const { name, age } = person;
const { name: personName, age: personAge } = person;  // Rename

// Computed property names
const key = "dynamicKey";
const obj = {
  [key]: "value"
};

// Object shorthand (ES6)
const name = "Bob";
const age = 25;
const user = { name, age };  // Same as { name: name, age: age }

// Object.assign (merge objects)
const merged = Object.assign({}, person, { city: "NYC" });

// Freeze object (immutable)
Object.freeze(person);
```

---

## Functions

### Function Declaration

```javascript
// Traditional function
function greet(name) {
  return `Hello, ${name}!`;
}

// Function with default parameters
function greet(name = "World") {
  return `Hello, ${name}!`;
}

// Rest parameters
function sum(...numbers) {
  return numbers.reduce((acc, val) => acc + val, 0);
}

sum(1, 2, 3, 4, 5);  // 15
```

### Function Expressions

```javascript
// Anonymous function
const greet = function(name) {
  return `Hello, ${name}!`;
};

// Named function expression
const factorial = function fact(n) {
  return n <= 1 ? 1 : n * fact(n - 1);
};
```

### Arrow Functions (ES6)

```javascript
// Basic arrow function
const greet = (name) => {
  return `Hello, ${name}!`;
};

// Implicit return (single expression)
const greet = name => `Hello, ${name}!`;

// No parameters
const sayHello = () => "Hello!";

// Multiple parameters
const add = (a, b) => a + b;

// Arrow functions and 'this'
const person = {
  name: "Alice",
  greet: function() {
    setTimeout(() => {
      console.log(`Hello, ${this.name}`);  // 'this' refers to person
    }, 1000);
  }
};
```

### Higher-Order Functions

```javascript
// Function that returns a function
function multiplier(factor) {
  return function(number) {
    return number * factor;
  };
}

const double = multiplier(2);
console.log(double(5));  // 10

// Function that takes a function as argument
function applyOperation(arr, operation) {
  return arr.map(operation);
}

const numbers = [1, 2, 3, 4, 5];
const squared = applyOperation(numbers, x => x * x);
```

### Closures

```javascript
function createCounter() {
  let count = 0;
  return {
    increment() {
      return ++count;
    },
    decrement() {
      return --count;
    },
    getCount() {
      return count;
    }
  };
}

const counter = createCounter();
counter.increment();  // 1
counter.increment();  // 2
counter.getCount();   // 2
```

---

## Asynchronous JavaScript

### Callbacks

```javascript
// Traditional callback pattern
function fetchData(callback) {
  setTimeout(() => {
    callback("Data loaded");
  }, 1000);
}

fetchData((data) => {
  console.log(data);
});

// Callback hell (pyramid of doom)
getData1((data1) => {
  getData2(data1, (data2) => {
    getData3(data2, (data3) => {
      console.log(data3);
    });
  });
});
```

### Promises

```javascript
// Creating a promise
const promise = new Promise((resolve, reject) => {
  setTimeout(() => {
    const success = true;
    if (success) {
      resolve("Data loaded");
    } else {
      reject("Error occurred");
    }
  }, 1000);
});

// Consuming a promise
promise
  .then(data => {
    console.log(data);
    return "Next data";
  })
  .then(nextData => {
    console.log(nextData);
  })
  .catch(error => {
    console.error(error);
  })
  .finally(() => {
    console.log("Cleanup");
  });

// Promise chaining
fetch('https://api.example.com/data')
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error(error));

// Promise.all (wait for all promises)
Promise.all([promise1, promise2, promise3])
  .then(([result1, result2, result3]) => {
    console.log(result1, result2, result3);
  });

// Promise.race (first to complete)
Promise.race([promise1, promise2])
  .then(result => console.log(result));

// Promise.allSettled (wait for all, regardless of result)
Promise.allSettled([promise1, promise2])
  .then(results => console.log(results));
```

### Async/Await (ES2017)

```javascript
// Async function
async function fetchData() {
  try {
    const response = await fetch('https://api.example.com/data');
    const data = await response.json();
    console.log(data);
    return data;
  } catch (error) {
    console.error('Error:', error);
  }
}

// Sequential execution
async function sequential() {
  const data1 = await fetchData1();
  const data2 = await fetchData2(data1);
  const data3 = await fetchData3(data2);
  return data3;
}

// Parallel execution
async function parallel() {
  const [data1, data2, data3] = await Promise.all([
    fetchData1(),
    fetchData2(),
    fetchData3()
  ]);
  return { data1, data2, data3 };
}

// Top-level await (ES2022)
const data = await fetchData();
```

---

## Classes and OOP

### ES6 Classes

```javascript
// Basic class
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  greet() {
    return `Hello, I'm ${this.name}`;
  }

  // Static method
  static species() {
    return "Homo sapiens";
  }

  // Getter
  get info() {
    return `${this.name}, ${this.age}`;
  }

  // Setter
  set info(value) {
    const [name, age] = value.split(', ');
    this.name = name;
    this.age = parseInt(age);
  }
}

const person = new Person("Alice", 30);
console.log(person.greet());
console.log(Person.species());
```

### Inheritance

```javascript
class Animal {
  constructor(name) {
    this.name = name;
  }

  speak() {
    return `${this.name} makes a sound`;
  }
}

class Dog extends Animal {
  constructor(name, breed) {
    super(name);  // Call parent constructor
    this.breed = breed;
  }

  speak() {
    return `${this.name} barks`;
  }

  fetch() {
    return `${this.name} is fetching`;
  }
}

const dog = new Dog("Buddy", "Golden Retriever");
console.log(dog.speak());   // "Buddy barks"
console.log(dog.fetch());   // "Buddy is fetching"
```

### Private Fields (ES2022)

```javascript
class BankAccount {
  #balance = 0;  // Private field

  deposit(amount) {
    this.#balance += amount;
  }

  withdraw(amount) {
    if (amount <= this.#balance) {
      this.#balance -= amount;
      return amount;
    }
    return 0;
  }

  getBalance() {
    return this.#balance;
  }
}

const account = new BankAccount();
account.deposit(100);
console.log(account.getBalance());  // 100
// console.log(account.#balance);   // SyntaxError
```

---

## Common Patterns

### Module Pattern

```javascript
const MyModule = (function() {
  // Private variables
  let privateVar = "I'm private";

  // Private function
  function privateFunction() {
    return "Private function called";
  }

  // Public API
  return {
    publicVar: "I'm public",
    publicFunction() {
      return privateFunction();
    },
    getPrivateVar() {
      return privateVar;
    }
  };
})();

console.log(MyModule.publicVar);
console.log(MyModule.publicFunction());
```

### Revealing Module Pattern

```javascript
const Calculator = (function() {
  let result = 0;

  function add(x) {
    result += x;
    return this;
  }

  function subtract(x) {
    result -= x;
    return this;
  }

  function getResult() {
    return result;
  }

  function reset() {
    result = 0;
    return this;
  }

  return {
    add,
    subtract,
    getResult,
    reset
  };
})();

Calculator.add(5).add(3).subtract(2);
console.log(Calculator.getResult());  // 6
```

### Singleton Pattern

```javascript
const Singleton = (function() {
  let instance;

  function createInstance() {
    return {
      name: "Singleton",
      getData() {
        return "Data from singleton";
      }
    };
  }

  return {
    getInstance() {
      if (!instance) {
        instance = createInstance();
      }
      return instance;
    }
  };
})();

const instance1 = Singleton.getInstance();
const instance2 = Singleton.getInstance();
console.log(instance1 === instance2);  // true
```

### Factory Pattern

```javascript
class Car {
  constructor(options) {
    this.doors = options.doors || 4;
    this.state = options.state || "brand new";
    this.color = options.color || "silver";
  }
}

class Truck {
  constructor(options) {
    this.wheels = options.wheels || 6;
    this.state = options.state || "used";
    this.color = options.color || "blue";
  }
}

class VehicleFactory {
  createVehicle(type, options) {
    switch(type) {
      case 'car':
        return new Car(options);
      case 'truck':
        return new Truck(options);
      default:
        throw new Error('Unknown vehicle type');
    }
  }
}

const factory = new VehicleFactory();
const car = factory.createVehicle('car', { color: 'red' });
const truck = factory.createVehicle('truck', { wheels: 8 });
```

### Observer Pattern

```javascript
class Subject {
  constructor() {
    this.observers = [];
  }

  subscribe(observer) {
    this.observers.push(observer);
  }

  unsubscribe(observer) {
    this.observers = this.observers.filter(obs => obs !== observer);
  }

  notify(data) {
    this.observers.forEach(observer => observer.update(data));
  }
}

class Observer {
  constructor(name) {
    this.name = name;
  }

  update(data) {
    console.log(`${this.name} received: ${data}`);
  }
}

const subject = new Subject();
const observer1 = new Observer('Observer 1');
const observer2 = new Observer('Observer 2');

subject.subscribe(observer1);
subject.subscribe(observer2);
subject.notify('Event occurred!');
```

---

## DOM Manipulation

```javascript
// Selecting elements
const element = document.getElementById('myId');
const elements = document.getElementsByClassName('myClass');
const element = document.querySelector('.myClass');
const elements = document.querySelectorAll('.myClass');

// Creating elements
const div = document.createElement('div');
div.textContent = 'Hello World';
div.className = 'my-class';
div.id = 'my-id';

// Appending elements
document.body.appendChild(div);
parentElement.insertBefore(newElement, referenceElement);

// Modifying content
element.textContent = 'New text';
element.innerHTML = '<strong>Bold text</strong>';

// Modifying attributes
element.setAttribute('data-id', '123');
element.getAttribute('data-id');
element.removeAttribute('data-id');

// Modifying styles
element.style.color = 'red';
element.style.fontSize = '20px';

// Adding/removing classes
element.classList.add('active');
element.classList.remove('inactive');
element.classList.toggle('visible');
element.classList.contains('active');

// Event listeners
element.addEventListener('click', (event) => {
  console.log('Element clicked!', event);
});

element.addEventListener('click', handleClick);
element.removeEventListener('click', handleClick);

// Event delegation
document.addEventListener('click', (event) => {
  if (event.target.matches('.my-button')) {
    console.log('Button clicked!');
  }
});

// Preventing default behavior
form.addEventListener('submit', (event) => {
  event.preventDefault();
  // Handle form submission
});
```

---

## ES6+ Features

### Destructuring

```javascript
// Array destructuring
const [a, b, c] = [1, 2, 3];
const [first, , third] = [1, 2, 3];
const [head, ...tail] = [1, 2, 3, 4, 5];

// Object destructuring
const { name, age } = { name: 'Alice', age: 30 };
const { name: personName } = { name: 'Alice' };  // Rename
const { name, age = 25 } = { name: 'Alice' };    // Default value

// Nested destructuring
const { address: { city, country } } = person;

// Function parameter destructuring
function greet({ name, age }) {
  return `Hello ${name}, you are ${age} years old`;
}
```

### Spread and Rest Operators

```javascript
// Spread in arrays
const arr1 = [1, 2, 3];
const arr2 = [...arr1, 4, 5, 6];

// Spread in objects
const obj1 = { a: 1, b: 2 };
const obj2 = { ...obj1, c: 3 };

// Rest in function parameters
function sum(...numbers) {
  return numbers.reduce((acc, val) => acc + val, 0);
}

// Rest in destructuring
const [first, ...rest] = [1, 2, 3, 4, 5];
```

### Optional Chaining (ES2020)

```javascript
const user = {
  name: 'Alice',
  address: {
    city: 'NYC'
  }
};

// Without optional chaining
const city = user && user.address && user.address.city;

// With optional chaining
const city = user?.address?.city;
const fn = obj?.method?.();  // Call method if exists
```

### Nullish Coalescing (ES2020)

```javascript
// Returns right operand when left is null or undefined
const value = null ?? 'default';        // 'default'
const value = undefined ?? 'default';   // 'default'
const value = 0 ?? 'default';           // 0
const value = '' ?? 'default';          // ''

// Compare with || operator
const value = 0 || 'default';           // 'default'
const value = '' || 'default';          // 'default'
```

---

## Error Handling

```javascript
// Try-catch
try {
  throw new Error('Something went wrong');
} catch (error) {
  console.error(error.message);
} finally {
  console.log('Cleanup');
}

// Custom errors
class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ValidationError';
  }
}

try {
  throw new ValidationError('Invalid input');
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Validation error:', error.message);
  } else {
    throw error;  // Re-throw unknown errors
  }
}

// Error handling with async/await
async function fetchData() {
  try {
    const response = await fetch('/api/data');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Fetch error:', error);
    throw error;
  }
}
```

---

## Common Array Methods

```javascript
const numbers = [1, 2, 3, 4, 5];

// map - transform each element
const doubled = numbers.map(n => n * 2);

// filter - select elements
const evens = numbers.filter(n => n % 2 === 0);

// reduce - aggregate
const sum = numbers.reduce((acc, n) => acc + n, 0);

// find - first matching element
const found = numbers.find(n => n > 3);

// findIndex - index of first match
const index = numbers.findIndex(n => n > 3);

// some - at least one matches
const hasEven = numbers.some(n => n % 2 === 0);

// every - all match
const allPositive = numbers.every(n => n > 0);

// flat - flatten nested arrays
const nested = [1, [2, 3], [4, [5, 6]]];
const flat = nested.flat(2);  // [1, 2, 3, 4, 5, 6]

// flatMap - map then flatten
const words = ['hello world', 'foo bar'];
const letters = words.flatMap(w => w.split(' '));
```

---

## Best Practices

1. **Use `const` by default**, `let` when reassignment is needed
2. **Avoid `var`** - it has function scope and hoisting issues
3. **Use arrow functions** for callbacks and short functions
4. **Use template literals** instead of string concatenation
5. **Use async/await** instead of promise chains when possible
6. **Use destructuring** for cleaner code
7. **Use spread operator** for copying arrays/objects
8. **Handle errors properly** with try-catch
9. **Use strict mode**: `'use strict';`
10. **Use meaningful variable names**

---

## Common Libraries/Frameworks

- **React**: UI library
- **Vue.js**: Progressive framework
- **Angular**: Full-featured framework
- **Express.js**: Web server framework (Node.js)
- **Lodash**: Utility library
- **Axios**: HTTP client
- **Moment.js/Day.js**: Date manipulation
- **D3.js**: Data visualization
