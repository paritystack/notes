# MongoDB

MongoDB is a popular NoSQL database that stores data in flexible, JSON-like documents. It's designed for scalability, high performance, and ease of development, making it ideal for modern applications that require flexible schema design and horizontal scaling.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [CRUD Operations](#crud-operations)
- [Data Modeling](#data-modeling)
- [Queries and Aggregation](#queries-and-aggregation)
- [Indexing](#indexing)
- [MongoDB with Node.js](#mongodb-with-nodejs)
- [Best Practices](#best-practices)
- [Performance Optimization](#performance-optimization)

---

## Introduction

**Key Features:**
- Document-oriented storage (JSON/BSON)
- Flexible schema design
- High performance
- High availability (Replica Sets)
- Horizontal scalability (Sharding)
- Rich query language
- Aggregation framework
- GridFS for large files
- Change Streams for real-time data

**Use Cases:**
- Content management systems
- Real-time analytics
- IoT applications
- Mobile applications
- Catalogs and inventory
- User data management
- Caching layer

---

## Installation and Setup

### Install MongoDB

**macOS:**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**Ubuntu:**
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
```

**Docker:**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### MongoDB Shell

```bash
# Connect to MongoDB
mongosh

# Show databases
show dbs

# Use/create database
use mydb

# Show collections
show collections

# Exit
exit
```

---

## CRUD Operations

### Create (Insert)

```javascript
// Insert one document
db.users.insertOne({
  name: "John Doe",
  email: "john@example.com",
  age: 30,
  createdAt: new Date()
})

// Insert multiple documents
db.users.insertMany([
  { name: "Jane Smith", email: "jane@example.com", age: 28 },
  { name: "Bob Johnson", email: "bob@example.com", age: 35 }
])
```

### Read (Query)

```javascript
// Find all documents
db.users.find()

// Find with filter
db.users.find({ age: { $gte: 30 } })

// Find one document
db.users.findOne({ email: "john@example.com" })

// Projection (select specific fields)
db.users.find({}, { name: 1, email: 1, _id: 0 })

// Limit and sort
db.users.find().limit(10).sort({ age: -1 })

// Count documents
db.users.countDocuments({ age: { $gte: 30 } })
```

### Update

```javascript
// Update one document
db.users.updateOne(
  { email: "john@example.com" },
  { $set: { age: 31, updatedAt: new Date() } }
)

// Update multiple documents
db.users.updateMany(
  { age: { $lt: 30 } },
  { $set: { status: "young" } }
)

// Replace document
db.users.replaceOne(
  { email: "john@example.com" },
  { name: "John Doe", email: "john@example.com", age: 31 }
)

// Upsert (update or insert)
db.users.updateOne(
  { email: "new@example.com" },
  { $set: { name: "New User", age: 25 } },
  { upsert: true }
)

// Increment field
db.users.updateOne(
  { email: "john@example.com" },
  { $inc: { loginCount: 1 } }
)

// Add to array
db.users.updateOne(
  { email: "john@example.com" },
  { $push: { hobbies: "reading" } }
)
```

### Delete

```javascript
// Delete one document
db.users.deleteOne({ email: "john@example.com" })

// Delete multiple documents
db.users.deleteMany({ age: { $lt: 18 } })

// Delete all documents
db.users.deleteMany({})
```

---

## Data Modeling

### Embedded Documents

```javascript
// User with embedded address
db.users.insertOne({
  name: "John Doe",
  email: "john@example.com",
  address: {
    street: "123 Main St",
    city: "New York",
    state: "NY",
    zip: "10001"
  },
  phoneNumbers: [
    { type: "home", number: "555-1234" },
    { type: "work", number: "555-5678" }
  ]
})
```

### Document References

```javascript
// Posts collection
db.posts.insertOne({
  title: "My First Post",
  content: "This is my first blog post",
  authorId: ObjectId("user_id_here"),
  comments: [
    {
      userId: ObjectId("commenter_id"),
      text: "Great post!",
      createdAt: new Date()
    }
  ]
})

// Query with lookup
db.posts.aggregate([
  {
    $lookup: {
      from: "users",
      localField: "authorId",
      foreignField: "_id",
      as: "author"
    }
  }
])
```

### Schema Design Patterns

```javascript
// One-to-One (Embedded)
{
  _id: ObjectId(),
  username: "john_doe",
  profile: {
    firstName: "John",
    lastName: "Doe",
    bio: "Software developer"
  }
}

// One-to-Many (Embedded - for small arrays)
{
  _id: ObjectId(),
  title: "Blog Post",
  tags: ["mongodb", "database", "nosql"]
}

// One-to-Many (Referenced - for large collections)
{
  _id: ObjectId(),
  name: "Category",
  products: [
    ObjectId("product_1"),
    ObjectId("product_2")
  ]
}

// Many-to-Many
// Users collection
{
  _id: ObjectId("user_1"),
  name: "John",
  courseIds: [ObjectId("course_1"), ObjectId("course_2")]
}

// Courses collection
{
  _id: ObjectId("course_1"),
  title: "MongoDB Course",
  studentIds: [ObjectId("user_1"), ObjectId("user_2")]
}
```

---

## Queries and Aggregation

### Query Operators

```javascript
// Comparison operators
db.users.find({ age: { $eq: 30 } })      // Equal
db.users.find({ age: { $ne: 30 } })      // Not equal
db.users.find({ age: { $gt: 30 } })      // Greater than
db.users.find({ age: { $gte: 30 } })     // Greater than or equal
db.users.find({ age: { $lt: 30 } })      // Less than
db.users.find({ age: { $lte: 30 } })     // Less than or equal
db.users.find({ age: { $in: [25, 30, 35] } })  // In array
db.users.find({ age: { $nin: [25, 30] } })     // Not in array

// Logical operators
db.users.find({
  $and: [
    { age: { $gte: 25 } },
    { age: { $lte: 35 } }
  ]
})

db.users.find({
  $or: [
    { age: { $lt: 25 } },
    { age: { $gt: 35 } }
  ]
})

db.users.find({ age: { $not: { $gte: 30 } } })

// Element operators
db.users.find({ email: { $exists: true } })
db.users.find({ age: { $type: "number" } })

// Array operators
db.users.find({ hobbies: { $all: ["reading", "gaming"] } })
db.users.find({ hobbies: { $size: 3 } })
db.users.find({ "hobbies.0": "reading" })

// Text search
db.posts.createIndex({ title: "text", content: "text" })
db.posts.find({ $text: { $search: "mongodb tutorial" } })
```

### Aggregation Pipeline

```javascript
// Basic aggregation
db.orders.aggregate([
  // Match stage (filter)
  { $match: { status: "completed" } },

  // Group stage
  {
    $group: {
      _id: "$customerId",
      totalSpent: { $sum: "$amount" },
      orderCount: { $sum: 1 },
      avgOrderAmount: { $avg: "$amount" }
    }
  },

  // Sort stage
  { $sort: { totalSpent: -1 } },

  // Limit stage
  { $limit: 10 }
])

// Complex aggregation with lookup
db.orders.aggregate([
  // Join with users collection
  {
    $lookup: {
      from: "users",
      localField: "userId",
      foreignField: "_id",
      as: "user"
    }
  },

  // Unwind array
  { $unwind: "$user" },

  // Project (reshape documents)
  {
    $project: {
      orderNumber: 1,
      amount: 1,
      userName: "$user.name",
      userEmail: "$user.email"
    }
  }
])

// Aggregation operators
db.sales.aggregate([
  {
    $group: {
      _id: "$category",
      total: { $sum: "$amount" },
      avg: { $avg: "$amount" },
      min: { $min: "$amount" },
      max: { $max: "$amount" },
      count: { $sum: 1 },
      items: { $push: "$productName" },
      first: { $first: "$date" },
      last: { $last: "$date" }
    }
  }
])
```

---

## Indexing

### Creating Indexes

```javascript
// Single field index
db.users.createIndex({ email: 1 })

// Compound index
db.users.createIndex({ lastName: 1, firstName: 1 })

// Unique index
db.users.createIndex({ email: 1 }, { unique: true })

// Text index
db.posts.createIndex({ title: "text", content: "text" })

// 2dsphere index (geospatial)
db.locations.createIndex({ coordinates: "2dsphere" })

// TTL index (auto-delete after time)
db.sessions.createIndex(
  { createdAt: 1 },
  { expireAfterSeconds: 3600 }
)

// Partial index
db.orders.createIndex(
  { status: 1 },
  { partialFilterExpression: { status: "active" } }
)

// Sparse index
db.users.createIndex(
  { phoneNumber: 1 },
  { sparse: true }
)
```

### Index Management

```javascript
// List indexes
db.users.getIndexes()

// Drop index
db.users.dropIndex("email_1")

// Drop all indexes
db.users.dropIndexes()

// Explain query (check index usage)
db.users.find({ email: "john@example.com" }).explain("executionStats")
```

---

## MongoDB with Node.js

### Installation

```bash
npm install mongodb
# or
npm install mongoose
```

### Native MongoDB Driver

```javascript
const { MongoClient } = require('mongodb');

const url = 'mongodb://localhost:27017';
const client = new MongoClient(url);

async function main() {
  await client.connect();
  console.log('Connected to MongoDB');

  const db = client.db('mydb');
  const users = db.collection('users');

  // Insert
  const result = await users.insertOne({
    name: 'John Doe',
    email: 'john@example.com',
    age: 30
  });
  console.log('Inserted:', result.insertedId);

  // Find
  const user = await users.findOne({ email: 'john@example.com' });
  console.log('Found:', user);

  // Update
  await users.updateOne(
    { email: 'john@example.com' },
    { $set: { age: 31 } }
  );

  // Delete
  await users.deleteOne({ email: 'john@example.com' });

  await client.close();
}

main().catch(console.error);
```

### Mongoose ODM

```javascript
const mongoose = require('mongoose');

// Connect
mongoose.connect('mongodb://localhost:27017/mydb', {
  useNewUrlParser: true,
  useUnifiedTopology: true
});

// Define schema
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  age: { type: Number, min: 0, max: 120 },
  createdAt: { type: Date, default: Date.now },
  address: {
    street: String,
    city: String,
    state: String,
    zip: String
  },
  hobbies: [String],
  status: {
    type: String,
    enum: ['active', 'inactive', 'banned'],
    default: 'active'
  }
});

// Instance methods
userSchema.methods.getFullInfo = function() {
  return `${this.name} (${this.email})`;
};

// Static methods
userSchema.statics.findByEmail = function(email) {
  return this.findOne({ email });
};

// Virtuals
userSchema.virtual('isAdult').get(function() {
  return this.age >= 18;
});

// Middleware
userSchema.pre('save', function(next) {
  console.log('About to save user:', this.name);
  next();
});

// Create model
const User = mongoose.model('User', userSchema);

// CRUD operations
async function examples() {
  // Create
  const user = new User({
    name: 'John Doe',
    email: 'john@example.com',
    age: 30,
    hobbies: ['reading', 'coding']
  });
  await user.save();

  // Find
  const users = await User.find({ age: { $gte: 25 } });
  const john = await User.findByEmail('john@example.com');

  // Update
  await User.updateOne({ email: 'john@example.com' }, { age: 31 });
  // or
  john.age = 31;
  await john.save();

  // Delete
  await User.deleteOne({ email: 'john@example.com' });

  // Populate (references)
  const postSchema = new mongoose.Schema({
    title: String,
    author: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }
  });
  const Post = mongoose.model('Post', postSchema);

  const posts = await Post.find().populate('author');
}
```

### Express + Mongoose API

```javascript
const express = require('express');
const mongoose = require('mongoose');

const app = express();
app.use(express.json());

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/mydb');

// User model
const User = mongoose.model('User', new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  age: Number
}));

// Routes
app.get('/users', async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/users/:id', async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/users', async (req, res) => {
  try {
    const user = new User(req.body);
    await user.save();
    res.status(201).json(user);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.put('/users/:id', async (req, res) => {
  try {
    const user = await User.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    );
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json(user);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.delete('/users/:id', async (req, res) => {
  try {
    const user = await User.findByIdAndDelete(req.params.id);
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json({ message: 'User deleted' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => console.log('Server running on port 3000'));
```

---

## Best Practices

### 1. Schema Design

```javascript
// Embed related data when:
// - Data is frequently accessed together
// - Data doesn't change often
// - Array size is bounded

// Reference when:
// - Data is frequently accessed separately
// - Data changes frequently
// - Array size is unbounded
```

### 2. Use Appropriate Indexes

```javascript
// Index fields used in queries
db.users.createIndex({ email: 1 })

// Compound indexes for multi-field queries
db.users.createIndex({ status: 1, createdAt: -1 })

// Monitor index usage
db.users.aggregate([{ $indexStats: {} }])
```

### 3. Validate Data

```javascript
// Mongoose validation
const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    validate: {
      validator: function(v) {
        return /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/.test(v);
      },
      message: props => `${props.value} is not a valid email!`
    }
  },
  age: {
    type: Number,
    min: [0, 'Age must be positive'],
    max: [120, 'Age seems unrealistic']
  }
});
```

### 4. Handle Errors

```javascript
try {
  await User.create({ email: 'invalid' });
} catch (error) {
  if (error.name === 'ValidationError') {
    // Handle validation error
  } else if (error.code === 11000) {
    // Handle duplicate key error
  }
}
```

### 5. Use Transactions (for multi-document operations)

```javascript
const session = await mongoose.startSession();
session.startTransaction();

try {
  await User.create([{ name: 'John' }], { session });
  await Post.create([{ title: 'First Post' }], { session });

  await session.commitTransaction();
} catch (error) {
  await session.abortTransaction();
  throw error;
} finally {
  session.endSession();
}
```

---

## Performance Optimization

### 1. Query Optimization

```javascript
// Use projection
db.users.find({}, { name: 1, email: 1 })

// Use covered queries (query uses only indexed fields)
db.users.createIndex({ email: 1, name: 1 })
db.users.find({ email: 'john@example.com' }, { email: 1, name: 1, _id: 0 })

// Limit results
db.users.find().limit(10)

// Use lean() in Mongoose (skip hydration)
const users = await User.find().lean()
```

### 2. Connection Pooling

```javascript
const mongoose = require('mongoose');

mongoose.connect('mongodb://localhost:27017/mydb', {
  maxPoolSize: 10,
  minPoolSize: 5
});
```

### 3. Batch Operations

```javascript
// Bulk insert
db.users.insertMany([
  { name: 'User 1' },
  { name: 'User 2' },
  { name: 'User 3' }
], { ordered: false })

// Bulk write
db.users.bulkWrite([
  { insertOne: { document: { name: 'John' } } },
  { updateOne: { filter: { name: 'Jane' }, update: { $set: { age: 30 } } } },
  { deleteOne: { filter: { name: 'Bob' } } }
])
```

### 4. Caching

```javascript
const Redis = require('redis');
const redis = Redis.createClient();

async function getUser(id) {
  // Check cache first
  const cached = await redis.get(`user:${id}`);
  if (cached) return JSON.parse(cached);

  // Query database
  const user = await User.findById(id);

  // Store in cache
  await redis.setex(`user:${id}`, 3600, JSON.stringify(user));

  return user;
}
```

---

## Resources

**Official Documentation:**
- [MongoDB Documentation](https://docs.mongodb.com/)
- [MongoDB University](https://university.mongodb.com/)
- [Mongoose Documentation](https://mongoosejs.com/)

**Tools:**
- [MongoDB Compass](https://www.mongodb.com/products/compass) - GUI
- [Studio 3T](https://studio3t.com/) - IDE for MongoDB
- [mongosh](https://www.mongodb.com/docs/mongodb-shell/) - MongoDB Shell

**Learning:**
- [MongoDB Tutorial](https://www.mongodb.com/docs/manual/tutorial/)
- [Mongoose Guide](https://mongoosejs.com/docs/guide.html)
- [MongoDB Performance](https://www.mongodb.com/docs/manual/administration/analyzing-mongodb-performance/)
