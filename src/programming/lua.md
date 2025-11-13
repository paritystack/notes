# Lua Programming

## Overview

Lua is a lightweight, high-level, multi-paradigm programming language designed primarily for embedded use in applications. It's known for its simplicity, efficiency, and powerful data description constructs.

**Key Features:**
- Lightweight and embeddable
- Fast execution
- Simple and clean syntax
- Dynamic typing
- Automatic memory management (garbage collection)
- First-class functions
- Powerful table data structure
- Coroutines for concurrency

**Common Uses:**
- Game scripting (World of Warcraft, Roblox)
- Embedded systems
- Configuration files
- Application scripting
- Web development (OpenResty)

---

## Basic Syntax

### Variables and Data Types

```lua
-- Variables (global by default)
name = "Alice"
age = 30
pi = 3.14159

-- Local variables (recommended)
local x = 10
local y = 20

-- Multiple assignment
local a, b, c = 1, 2, 3

-- Swap variables
a, b = b, a

-- Nil (undefined/null)
local nothing = nil

-- Comments
-- Single line comment

--[[
    Multi-line
    comment
]]

-- Data types
local num = 42              -- number
local str = "Hello"         -- string
local bool = true           -- boolean
local tbl = {1, 2, 3}      -- table
local func = function() end -- function
local thread = coroutine.create(function() end)  -- thread
local nothing = nil         -- nil

-- Type checking
print(type(num))    -- number
print(type(str))    -- string
print(type(bool))   -- boolean
print(type(tbl))    -- table
print(type(func))   -- function
```

### Strings

```lua
-- String creation
local s1 = "Hello"
local s2 = 'World'
local s3 = [[Multi-line
string]]

-- String concatenation
local full = s1 .. " " .. s2

-- String length
local len = #s1
local len2 = string.len(s1)

-- String methods
local upper = string.upper(s1)          -- "HELLO"
local lower = string.lower(s1)          -- "hello"
local sub = string.sub(s1, 1, 3)        -- "Hel"
local find = string.find(s1, "ll")      -- 3, 4
local replace = string.gsub(s1, "l", "L")  -- "HeLLo"

-- String formatting
local formatted = string.format("Name: %s, Age: %d", "Alice", 30)

-- String to number
local num = tonumber("42")
local str = tostring(42)

-- String repetition
local repeated = string.rep("Ha", 3)  -- "HaHaHa"

-- Pattern matching (similar to regex)
local match = string.match("Hello123", "%d+")  -- "123"

-- Iterate characters
for i = 1, #s1 do
    local char = string.sub(s1, i, i)
    print(char)
end
```

---

## Tables

Tables are the only data structure in Lua - they can represent arrays, dictionaries, objects, and more.

### Arrays (1-indexed)

```lua
-- Array creation
local arr = {10, 20, 30, 40, 50}

-- Accessing elements (1-indexed!)
print(arr[1])  -- 10

-- Modifying elements
arr[1] = 15

-- Array length
local len = #arr

-- Append to array
table.insert(arr, 60)        -- Append to end
table.insert(arr, 2, 25)     -- Insert at position 2

-- Remove from array
local last = table.remove(arr)      -- Remove last
local second = table.remove(arr, 2) -- Remove at position 2

-- Iterate array
for i = 1, #arr do
    print(i, arr[i])
end

-- Iterate with ipairs
for i, v in ipairs(arr) do
    print(i, v)
end

-- Table functions
table.sort(arr)                      -- Sort ascending
table.sort(arr, function(a, b) return a > b end)  -- Sort descending
local str = table.concat(arr, ", ")  -- Join with separator
```

### Dictionaries/Maps

```lua
-- Dictionary creation
local person = {
    name = "Alice",
    age = 30,
    city = "NYC"
}

-- Alternative syntax
local person2 = {
    ["name"] = "Bob",
    ["age"] = 25
}

-- Accessing elements
print(person.name)       -- Dot notation
print(person["age"])     -- Bracket notation

-- Adding/modifying
person.email = "alice@example.com"
person["phone"] = "123-456-7890"

-- Removing
person.email = nil

-- Iterate dictionary
for key, value in pairs(person) do
    print(key, value)
end

-- Check if key exists
if person.name then
    print("Name exists")
end

-- Nested tables
local nested = {
    user = {
        name = "Alice",
        address = {
            city = "NYC",
            country = "USA"
        }
    }
}

print(nested.user.address.city)
```

### Mixed Tables

```lua
-- Table with both array and dictionary parts
local mixed = {
    "first",      -- [1] = "first"
    "second",     -- [2] = "second"
    name = "Alice",
    age = 30
}

print(mixed[1])      -- "first"
print(mixed.name)    -- "Alice"

-- Length only counts array part
print(#mixed)        -- 2
```

---

## Control Flow

### If-Else

```lua
local age = 18

if age < 13 then
    print("Child")
elseif age < 20 then
    print("Teenager")
else
    print("Adult")
end

-- Logical operators: and, or, not
if age >= 18 and age < 65 then
    print("Working age")
end

if age < 18 or age > 65 then
    print("Not working age")
end

if not (age < 18) then
    print("Adult")
end

-- Ternary-like operator
local status = age >= 18 and "Adult" or "Minor"
```

### Loops

```lua
-- While loop
local count = 0
while count < 5 do
    print(count)
    count = count + 1
end

-- Repeat-until loop (do-while)
local i = 0
repeat
    print(i)
    i = i + 1
until i >= 5

-- For loop (numeric)
for i = 1, 5 do
    print(i)  -- 1, 2, 3, 4, 5
end

-- For loop with step
for i = 0, 10, 2 do
    print(i)  -- 0, 2, 4, 6, 8, 10
end

-- For loop (reverse)
for i = 5, 1, -1 do
    print(i)  -- 5, 4, 3, 2, 1
end

-- Iterate array with ipairs
local arr = {10, 20, 30, 40, 50}
for i, v in ipairs(arr) do
    print(i, v)
end

-- Iterate table with pairs
local person = {name = "Alice", age = 30}
for key, value in pairs(person) do
    print(key, value)
end

-- Break
for i = 1, 10 do
    if i == 5 then
        break
    end
    print(i)
end

-- No continue in Lua (use goto in Lua 5.2+)
for i = 1, 10 do
    if i == 5 then
        goto continue
    end
    print(i)
    ::continue::
end
```

---

## Functions

```lua
-- Basic function
function greet(name)
    print("Hello, " .. name)
end

greet("Alice")

-- Function with return value
function add(a, b)
    return a + b
end

local result = add(5, 3)

-- Multiple return values
function swap(a, b)
    return b, a
end

local x, y = swap(10, 20)

-- Default parameters (manual)
function greet(name)
    name = name or "World"
    print("Hello, " .. name)
end

-- Variable arguments
function sum(...)
    local total = 0
    for _, v in ipairs({...}) do
        total = total + v
    end
    return total
end

print(sum(1, 2, 3, 4, 5))  -- 15

-- Anonymous functions
local add = function(a, b)
    return a + b
end

-- Function as argument
function applyOperation(a, b, operation)
    return operation(a, b)
end

local result = applyOperation(5, 3, function(x, y)
    return x * y
end)

-- Closures
function counter()
    local count = 0
    return function()
        count = count + 1
        return count
    end
end

local c = counter()
print(c())  -- 1
print(c())  -- 2
print(c())  -- 3

-- Local functions
local function helper()
    print("Helper function")
end

-- Recursive functions need forward declaration
local factorial
factorial = function(n)
    if n <= 1 then
        return 1
    else
        return n * factorial(n - 1)
    end
end
```

---

## Object-Oriented Programming

Lua doesn't have built-in classes, but tables and metatables provide OOP features.

### Tables as Objects

```lua
-- Simple object
local person = {
    name = "Alice",
    age = 30,

    greet = function(self)
        print("Hello, I'm " .. self.name)
    end
}

person:greet()  -- Colon syntax passes self automatically
-- Equivalent to: person.greet(person)
```

### Metatables and Classes

```lua
-- Define a class
local Person = {}
Person.__index = Person

-- Constructor
function Person:new(name, age)
    local instance = setmetatable({}, Person)
    instance.name = name
    instance.age = age
    return instance
end

-- Methods
function Person:greet()
    print("Hello, I'm " .. self.name)
end

function Person:getAge()
    return self.age
end

function Person:setAge(age)
    self.age = age
end

-- Usage
local alice = Person:new("Alice", 30)
alice:greet()
print(alice:getAge())

-- Inheritance
local Employee = setmetatable({}, {__index = Person})
Employee.__index = Employee

function Employee:new(name, age, salary)
    local instance = Person:new(name, age)
    setmetatable(instance, Employee)
    instance.salary = salary
    return instance
end

function Employee:getSalary()
    return self.salary
end

-- Usage
local emp = Employee:new("Bob", 25, 50000)
emp:greet()  -- Inherited from Person
print(emp:getSalary())

-- Operator overloading
local Vector = {}
Vector.__index = Vector

function Vector:new(x, y)
    return setmetatable({x = x, y = y}, Vector)
end

-- Overload addition
Vector.__add = function(a, b)
    return Vector:new(a.x + b.x, a.y + b.y)
end

-- Overload tostring
Vector.__tostring = function(v)
    return "(" .. v.x .. ", " .. v.y .. ")"
end

local v1 = Vector:new(1, 2)
local v2 = Vector:new(3, 4)
local v3 = v1 + v2
print(v3)  -- (4, 6)
```

---

## Modules

```lua
-- mymodule.lua
local M = {}

-- Private function
local function private()
    print("Private")
end

-- Public function
function M.public()
    print("Public")
end

function M.add(a, b)
    return a + b
end

return M

-- main.lua
local mymodule = require("mymodule")
mymodule.public()
local result = mymodule.add(5, 3)
```

---

## Error Handling

```lua
-- pcall (protected call)
local success, result = pcall(function()
    return 10 / 0
end)

if success then
    print("Result:", result)
else
    print("Error:", result)
end

-- Error with message
function divide(a, b)
    if b == 0 then
        error("Division by zero")
    end
    return a / b
end

local success, result = pcall(divide, 10, 0)
if not success then
    print("Error:", result)
end

-- Assert
local function checkPositive(n)
    assert(n > 0, "Number must be positive")
    return n
end

-- xpcall (with error handler)
local function errorHandler(err)
    print("Error occurred:", err)
    return err
end

local success, result = xpcall(function()
    error("Something went wrong")
end, errorHandler)
```

---

## File I/O

```lua
-- Read entire file
local file = io.open("input.txt", "r")
if file then
    local content = file:read("*all")
    print(content)
    file:close()
end

-- Read line by line
local file = io.open("input.txt", "r")
if file then
    for line in file:lines() do
        print(line)
    end
    file:close()
end

-- Write file
local file = io.open("output.txt", "w")
if file then
    file:write("Hello, World!\n")
    file:write("Second line\n")
    file:close()
end

-- Append to file
local file = io.open("output.txt", "a")
if file then
    file:write("Appended line\n")
    file:close()
end

-- Using io.input and io.output
io.input("input.txt")
local content = io.read("*all")
io.close()

io.output("output.txt")
io.write("Hello\n")
io.close()
```

---

## Coroutines

```lua
-- Create coroutine
local co = coroutine.create(function()
    for i = 1, 5 do
        print("Coroutine:", i)
        coroutine.yield()  -- Pause execution
    end
end)

-- Resume coroutine
coroutine.resume(co)  -- Prints 1
coroutine.resume(co)  -- Prints 2
coroutine.resume(co)  -- Prints 3

-- Check status
print(coroutine.status(co))  -- suspended or running or dead

-- Producer-consumer pattern
local function producer()
    return coroutine.create(function()
        for i = 1, 5 do
            coroutine.yield(i)
        end
    end)
end

local function consumer(prod)
    while true do
        local status, value = coroutine.resume(prod)
        if not status then break end
        print("Received:", value)
    end
end

consumer(producer())
```

---

## Common Patterns

### Singleton Pattern

```lua
local Singleton = {}
local instance

function Singleton:getInstance()
    if not instance then
        instance = {data = "singleton"}
    end
    return instance
end

local s1 = Singleton:getInstance()
local s2 = Singleton:getInstance()
print(s1 == s2)  -- true
```

### Factory Pattern

```lua
local AnimalFactory = {}

function AnimalFactory:create(animalType)
    if animalType == "dog" then
        return {speak = function() return "Woof!" end}
    elseif animalType == "cat" then
        return {speak = function() return "Meow!" end}
    end
end

local dog = AnimalFactory:create("dog")
print(dog.speak())
```

### Observer Pattern

```lua
local Subject = {}
Subject.__index = Subject

function Subject:new()
    return setmetatable({observers = {}}, Subject)
end

function Subject:attach(observer)
    table.insert(self.observers, observer)
end

function Subject:detach(observer)
    for i, obs in ipairs(self.observers) do
        if obs == observer then
            table.remove(self.observers, i)
            break
        end
    end
end

function Subject:notify(data)
    for _, observer in ipairs(self.observers) do
        observer:update(data)
    end
end

-- Observer
local Observer = {}
Observer.__index = Observer

function Observer:new(name)
    return setmetatable({name = name}, Observer)
end

function Observer:update(data)
    print(self.name .. " received: " .. data)
end

-- Usage
local subject = Subject:new()
local obs1 = Observer:new("Observer1")
local obs2 = Observer:new("Observer2")

subject:attach(obs1)
subject:attach(obs2)
subject:notify("Event occurred!")
```

---

## Standard Library

```lua
-- Math
print(math.pi)
print(math.abs(-5))
print(math.floor(3.7))
print(math.ceil(3.2))
print(math.max(1, 5, 3))
print(math.min(1, 5, 3))
print(math.random())        -- Random [0, 1)
print(math.random(10))      -- Random [1, 10]
print(math.random(5, 10))   -- Random [5, 10]

-- String
print(string.upper("hello"))
print(string.lower("HELLO"))
print(string.reverse("hello"))

-- Table
local arr = {3, 1, 4, 1, 5}
table.sort(arr)
print(table.concat(arr, ", "))

-- OS
print(os.time())
print(os.date("%Y-%m-%d %H:%M:%S"))
os.execute("ls")  -- Execute shell command

-- Pairs / IPairs
local t = {10, 20, 30, x = 1, y = 2}
for k, v in pairs(t) do   -- All elements
    print(k, v)
end
for i, v in ipairs(t) do  -- Only array part
    print(i, v)
end
```

---

## Best Practices

1. **Use local variables** - Faster and avoids global pollution
   ```lua
   local x = 10  -- Good
   x = 10        -- Bad (global)
   ```

2. **Prefer ipairs for arrays** - More efficient than pairs

3. **Use metatables** for OOP and operator overloading

4. **Always close files** after use

5. **Use pcall** for error handling in production

6. **Avoid goto** - Use structured control flow

7. **Use string.format** for complex string formatting

8. **Cache table lookups** in loops
   ```lua
   local insert = table.insert
   for i = 1, 1000 do
       insert(arr, i)
   end
   ```

9. **Use semicolons sparingly** - Optional in Lua

10. **Follow naming conventions**
    - Variables: snake_case
    - Constants: UPPER_CASE
    - Functions: camelCase or snake_case

---

## Common Use Cases

### Configuration Files

```lua
-- config.lua
return {
    database = {
        host = "localhost",
        port = 5432,
        name = "mydb"
    },
    server = {
        port = 8080,
        workers = 4
    }
}

-- Load config
local config = require("config")
print(config.database.host)
```

### Game Scripting

```lua
-- Define enemy
local Enemy = {}
Enemy.__index = Enemy

function Enemy:new(name, health, damage)
    return setmetatable({
        name = name,
        health = health,
        damage = damage
    }, Enemy)
end

function Enemy:attack(target)
    target.health = target.health - self.damage
    print(self.name .. " attacks " .. target.name)
end

function Enemy:isAlive()
    return self.health > 0
end
```

---

## Lua Versions

- **Lua 5.1**: Widely used in games (WoW, Roblox)
- **Lua 5.2**: Added `goto`, `_ENV`
- **Lua 5.3**: Integer subtype, bitwise operators
- **Lua 5.4**: To-be-closed variables, const variables
- **LuaJIT**: JIT compiler, very fast (used in OpenResty)

---

## Useful Libraries

- **LuaSocket**: Networking
- **LuaFileSystem**: File system operations
- **Penlight**: Extended standard library
- **LÖVE**: Game framework
- **OpenResty**: Web platform (Nginx + Lua)
- **LuaRocks**: Package manager
