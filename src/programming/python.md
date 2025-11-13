# Python Programming

## Overview

Python is a high-level, interpreted, dynamically-typed programming language known for its simplicity and readability. It's widely used for web development, data science, machine learning, automation, and scripting.

**Key Features:**
- Clean, readable syntax emphasizing indentation
- Dynamic typing with strong type checking
- Extensive standard library ("batteries included")
- Large ecosystem of third-party packages (PyPI)
- Multi-paradigm: procedural, object-oriented, functional

---

## Basic Syntax

### Variables and Data Types

```python
# Variables (no declaration needed)
x = 10              # int
y = 3.14            # float
name = "Alice"      # str
is_valid = True     # bool

# Type checking and conversion
print(type(x))      # <class 'int'>
num_str = str(42)   # Convert to string
num_int = int("42") # Convert to int
```

### Print and String Formatting

```python
# Basic print
print("Hello, World!")

# f-strings (Python 3.6+)
name = "Bob"
age = 30
print(f"{name} is {age} years old")

# .format() method
print("{} is {} years old".format(name, age))

# %-formatting (older style)
print("%s is %d years old" % (name, age))
```

---

## Data Structures

### Lists

Lists are mutable, ordered sequences that can contain mixed types.

```python
# Creating lists
my_list = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# Common operations
my_list.append(6)           # Add to end: [1, 2, 3, 4, 5, 6]
my_list.insert(0, 0)        # Insert at index: [0, 1, 2, 3, 4, 5, 6]
my_list.pop()               # Remove and return last: 6
my_list.remove(3)           # Remove first occurrence of value
element = my_list[2]        # Access by index
my_list[1] = 10             # Modify by index

# Slicing
first_three = my_list[0:3]  # [0, 1, 2]
last_two = my_list[-2:]     # Last 2 elements
reversed_list = my_list[::-1]  # Reverse

# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# Common methods
len(my_list)                # Length
my_list.sort()              # Sort in-place
sorted(my_list)             # Return sorted copy
my_list.reverse()           # Reverse in-place
my_list.count(2)            # Count occurrences
my_list.index(2)            # Find first index
```

**List Characteristics:**
- Mutable (can change)
- Ordered (maintains insertion order)
- Allows duplicates
- Can be nested
- Dynamic sizing

### Tuples

Tuples are immutable, ordered sequences.

```python
# Creating tuples
my_tuple = (1, 2, 3, 4, 5)
single = (42,)              # Single element needs comma
empty = ()

# Accessing elements
first = my_tuple[0]
last = my_tuple[-1]
sub = my_tuple[1:3]

# Unpacking
x, y, z = (1, 2, 3)
a, *rest, b = (1, 2, 3, 4, 5)  # a=1, rest=[2,3,4], b=5

# Named tuples
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20
```

**Tuple Characteristics:**
- Immutable (cannot change)
- Ordered
- Faster than lists
- Can be used as dictionary keys
- Used for function return values

### Dictionaries

Dictionaries are mutable, unordered key-value collections (ordered in Python 3.7+).

```python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}
empty = {}
from_keys = dict.fromkeys(['a', 'b', 'c'], 0)  # {'a': 0, 'b': 0, 'c': 0}

# Accessing and modifying
name = person["name"]       # KeyError if not exists
age = person.get("age", 0)  # Returns default if not exists
person["email"] = "alice@example.com"  # Add/update
del person["city"]          # Delete key

# Methods
person.keys()               # dict_keys(['name', 'age', 'email'])
person.values()             # dict_values(['Alice', 30, 'alice@example.com'])
person.items()              # dict_items([('name', 'Alice'), ...])

# Iteration
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Merge dictionaries (Python 3.9+)
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged = dict1 | dict2
```

### Sets

Sets are mutable, unordered collections of unique elements.

```python
# Creating sets
my_set = {1, 2, 3, 4, 5}
empty = set()  # Note: {} creates empty dict
from_list = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}

# Operations
my_set.add(6)
my_set.remove(3)        # KeyError if not exists
my_set.discard(3)       # No error if not exists
my_set.pop()            # Remove and return arbitrary element

# Set operations
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
union = a | b           # {1, 2, 3, 4, 5, 6}
intersection = a & b    # {3, 4}
difference = a - b      # {1, 2}
symmetric_diff = a ^ b  # {1, 2, 5, 6}

# Set comprehension
evens = {x for x in range(10) if x % 2 == 0}
```

---

## Control Flow

### If-Elif-Else

```python
age = 18

if age < 13:
    print("Child")
elif age < 20:
    print("Teenager")
else:
    print("Adult")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"

# Check multiple conditions
if 10 < age < 20:
    print("Teenager")
```

### Loops

```python
# For loop
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for i in range(0, 10, 2):  # Start, stop, step
    print(i)  # 0, 2, 4, 6, 8

# Iterate over list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Enumerate (get index and value)
for idx, fruit in enumerate(fruits):
    print(f"{idx}: {fruit}")

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# Break and continue
for i in range(10):
    if i == 3:
        continue  # Skip 3
    if i == 7:
        break     # Stop at 7
    print(i)

# Else clause (runs if loop completes without break)
for i in range(5):
    print(i)
else:
    print("Loop completed")
```

---

## Functions

### Basic Functions

```python
# Simple function
def greet(name):
    return f"Hello, {name}!"

# Default arguments
def greet(name="World"):
    return f"Hello, {name}!"

# Multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

minimum, maximum, average = get_stats([1, 2, 3, 4, 5])

# *args (variable positional arguments)
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs (variable keyword arguments)
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="NYC")

# Lambda functions
square = lambda x: x**2
add = lambda x, y: x + y

# Map, Filter, Reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

from functools import reduce
product = reduce(lambda x, y: x * y, numbers)  # 120
```

### Decorators

```python
# Simple decorator
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Before function call
# Hello!
# After function call

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Prints 3 times

# Common built-in decorators
class MyClass:
    @staticmethod
    def static_method():
        print("Static method")

    @classmethod
    def class_method(cls):
        print(f"Class method of {cls}")

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
```

---

## Object-Oriented Programming

### Classes and Objects

```python
class Person:
    # Class variable (shared by all instances)
    species = "Homo sapiens"

    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age

    # Instance method
    def greet(self):
        return f"Hello, I'm {self.name} and I'm {self.age} years old"

    # Magic methods
    def __str__(self):
        return f"Person({self.name}, {self.age})"

    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age

# Creating objects
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

print(person1.greet())
print(str(person1))
```

### Inheritance

```python
# Single inheritance
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())  # Buddy says Woof!

# Multiple inheritance
class Flyer:
    def fly(self):
        return "Flying..."

class Swimmer:
    def swim(self):
        return "Swimming..."

class Duck(Animal, Flyer, Swimmer):
    def speak(self):
        return f"{self.name} says Quack!"

duck = Duck("Donald")
print(duck.speak())  # Donald says Quack!
print(duck.fly())    # Flying...
print(duck.swim())   # Swimming...

# super() for parent class
class Employee(Person):
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id

    def get_info(self):
        return f"{self.greet()}, ID: {self.employee_id}"
```

### Data Classes (Python 3.7+)

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Person:
    name: str
    age: int
    email: str = "unknown@example.com"  # Default value
    hobbies: List[str] = field(default_factory=list)

    def greet(self):
        return f"Hello, I'm {self.name}"

person = Person("Alice", 30)
print(person)  # Person(name='Alice', age=30, email='unknown@example.com', hobbies=[])

# Frozen (immutable) dataclass
@dataclass(frozen=True)
class Point:
    x: int
    y: int
```

---

## Common Patterns

### Singleton Pattern

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# All instances are the same
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True
```

### Factory Pattern

```python
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Usage
animal = AnimalFactory.create_animal("dog")
print(animal.speak())  # Woof!
```

### Context Manager Pattern

```python
# Custom context manager
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Usage
with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')

# Using contextlib
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()

with file_manager('test.txt', 'r') as f:
    content = f.read()
```

### Iterator and Generator Patterns

```python
# Iterator
class Counter:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        current = self.current
        self.current += 1
        return current

for num in Counter(1, 5):
    print(num)  # 1, 2, 3, 4, 5

# Generator
def counter(start, end):
    while start <= end:
        yield start
        start += 1

for num in counter(1, 5):
    print(num)

# Generator expressions
squares = (x**2 for x in range(10))
print(next(squares))  # 0
print(next(squares))  # 1
```

---

## File Handling

```python
# Reading files
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()          # Read entire file

with open('file.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()       # Read all lines as list

with open('file.txt', 'r', encoding='utf-8') as f:
    for line in f:              # Iterate line by line
        print(line.strip())

# Writing files
with open('file.txt', 'w', encoding='utf-8') as f:
    f.write('Hello, World!\n')

# Appending
with open('file.txt', 'a', encoding='utf-8') as f:
    f.write('New line\n')

# Binary files
with open('image.png', 'rb') as f:
    data = f.read()

# JSON files
import json

# Write JSON
data = {"name": "Alice", "age": 30}
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

# Read JSON
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# CSV files
import csv

# Write CSV
with open('data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerow(['Alice', 30, 'NYC'])

# Read CSV
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

---

## Error Handling

```python
# Try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int("abc")
except (ValueError, TypeError) as e:
    print(f"Error: {e}")

# Catch all exceptions
try:
    risky_operation()
except Exception as e:
    print(f"An error occurred: {e}")

# Finally block
try:
    f = open('file.txt', 'r')
    content = f.read()
except FileNotFoundError:
    print("File not found")
finally:
    f.close()  # Always executes

# Else block
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Error")
else:
    print("Success!")  # Runs if no exception

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

# Custom exceptions
class InvalidAgeError(Exception):
    pass

def check_age(age):
    if age < 0:
        raise InvalidAgeError("Age must be positive")
```

---

## Working with Excel and Pandas

```python
from dataclasses import dataclass
import pandas as pd
from typing import List

@dataclass
class Person:
    name: str
    age: int
    email: str

# Load data from Excel
def load_people_from_excel(file_path: str) -> List[Person]:
    df = pd.read_excel(file_path)
    return [
        Person(
            name=row['name'],
            age=row['age'],
            email=row['email']
        ) for _, row in df.iterrows()
    ]

# Usage
people = load_people_from_excel("data.xlsx")
for person in people:
    print(f"{person.name} is {person.age} years old")

# With column mapping
EXCEL_TO_CLASS_MAPPING = {
    'Full Name': 'name',
    'Person Age': 'age',
    'E-mail Address': 'email'
}

def load_with_mapping(file_path: str) -> List[Person]:
    df = pd.read_excel(file_path)
    df = df.rename(columns=EXCEL_TO_CLASS_MAPPING)
    return [Person(**row) for _, row in df.iterrows()]
```

---

## Common Python Idioms

```python
# List comprehension vs map/filter
numbers = range(10)
squares = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]

# Dictionary get with default
config = {"debug": True}
log_level = config.get("log_level", "INFO")

# String joining
words = ["Hello", "World"]
sentence = " ".join(words)

# Enumerate
for idx, value in enumerate(['a', 'b', 'c']):
    print(f"{idx}: {value}")

# Zip
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Unpacking
first, *middle, last = [1, 2, 3, 4, 5]

# Swapping variables
a, b = 10, 20
a, b = b, a

# Chaining comparisons
if 0 < x < 10:
    print("x is between 0 and 10")

# In-place operations
numbers = [1, 2, 3]
numbers += [4, 5]  # Extend list

# any() and all()
numbers = [2, 4, 6, 8]
all_even = all(x % 2 == 0 for x in numbers)
has_even = any(x % 2 == 0 for x in numbers)
```

---

## Virtual Environments

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install packages
pip install requests pandas numpy

# Save dependencies
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate
```

---

## Best Practices

1. **Follow PEP 8**: Python's style guide
   - Use 4 spaces for indentation
   - Max line length: 79 characters
   - Use snake_case for functions and variables
   - Use PascalCase for classes

2. **Use Type Hints** (Python 3.5+)
   ```python
   def greet(name: str) -> str:
       return f"Hello, {name}!"

   from typing import List, Dict, Optional

   def process_data(items: List[int]) -> Dict[str, int]:
       return {"sum": sum(items), "count": len(items)}
   ```

3. **Use List Comprehensions** for simple transformations
   ```python
   # Good
   squares = [x**2 for x in range(10)]

   # Avoid for complex logic
   # Use regular loops instead
   ```

4. **Use Context Managers** for resource management
   ```python
   with open('file.txt', 'r') as f:
       data = f.read()
   ```

5. **Use f-strings** for string formatting (Python 3.6+)
   ```python
   name = "Alice"
   age = 30
   print(f"{name} is {age} years old")
   ```

---

## Common Libraries

- **Requests**: HTTP requests
- **NumPy**: Numerical computing
- **Pandas**: Data analysis
- **Matplotlib/Seaborn**: Data visualization
- **Flask/Django**: Web frameworks
- **SQLAlchemy**: Database ORM
- **pytest**: Testing
- **Beautiful Soup**: Web scraping
- **Pillow**: Image processing
