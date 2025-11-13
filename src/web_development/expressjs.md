# Express.js

Express.js is a minimal and flexible Node.js web application framework that provides a robust set of features for building web and mobile applications. It's the de facto standard server framework for Node.js and is widely used for building RESTful APIs and web applications.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Basic Application](#basic-application)
- [Routing](#routing)
- [Middleware](#middleware)
- [Request and Response](#request-and-response)
- [Error Handling](#error-handling)
- [Template Engines](#template-engines)
- [Static Files](#static-files)
- [Database Integration](#database-integration)
- [Authentication](#authentication)
- [RESTful API](#restful-api)
- [File Uploads](#file-uploads)
- [Security Best Practices](#security-best-practices)
- [Testing](#testing)
- [Production Deployment](#production-deployment)

---

## Introduction

**Key Features:**
- Minimal and unopinionated framework
- Robust routing system
- Focus on high performance
- Super-high test coverage
- HTTP helpers (redirection, caching, etc.)
- View system with 14+ template engines
- Content negotiation
- Executable for generating applications quickly

**Use Cases:**
- RESTful APIs
- Web applications
- Microservices
- Real-time applications (with Socket.io)
- Server-side rendering
- Proxy servers

---

## Installation and Setup

### Create New Project

```bash
# Create project directory
mkdir my-express-app
cd my-express-app

# Initialize npm project
npm init -y

# Install Express
npm install express

# Install development dependencies
npm install --save-dev nodemon typescript @types/node @types/express
```

### TypeScript Setup

```bash
# Initialize TypeScript
npx tsc --init
```

**tsconfig.json:**
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
    "resolveJsonModule": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

**package.json scripts:**
```json
{
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "nodemon --exec ts-node src/index.ts",
    "watch": "tsc --watch"
  }
}
```

---

## Basic Application

### Minimal Express App

```javascript
const express = require('express');
const app = express();
const PORT = 3000;

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

### TypeScript Version

```typescript
import express, { Express, Request, Response } from 'express';

const app: Express = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req: Request, res: Response) => {
  res.send('Hello World!');
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

### Application Structure

```
my-express-app/
├── src/
│   ├── index.ts              # Entry point
│   ├── config/
│   │   ├── database.ts       # Database configuration
│   │   └── environment.ts    # Environment variables
│   ├── controllers/          # Route controllers
│   │   └── userController.ts
│   ├── middleware/           # Custom middleware
│   │   ├── auth.ts
│   │   └── errorHandler.ts
│   ├── models/               # Data models
│   │   └── User.ts
│   ├── routes/               # Route definitions
│   │   └── userRoutes.ts
│   ├── services/             # Business logic
│   │   └── userService.ts
│   └── utils/                # Utility functions
│       └── validators.ts
├── tests/                    # Test files
├── dist/                     # Compiled JavaScript
├── node_modules/
├── package.json
├── tsconfig.json
└── .env
```

---

## Routing

### Basic Routes

```typescript
import express from 'express';
const app = express();

// GET request
app.get('/users', (req, res) => {
  res.json({ message: 'Get all users' });
});

// POST request
app.post('/users', (req, res) => {
  res.json({ message: 'Create user' });
});

// PUT request
app.put('/users/:id', (req, res) => {
  res.json({ message: `Update user ${req.params.id}` });
});

// DELETE request
app.delete('/users/:id', (req, res) => {
  res.json({ message: `Delete user ${req.params.id}` });
});

// PATCH request
app.patch('/users/:id', (req, res) => {
  res.json({ message: `Partially update user ${req.params.id}` });
});
```

### Route Parameters

```typescript
// Single parameter
app.get('/users/:id', (req, res) => {
  const userId = req.params.id;
  res.json({ userId });
});

// Multiple parameters
app.get('/users/:userId/posts/:postId', (req, res) => {
  const { userId, postId } = req.params;
  res.json({ userId, postId });
});

// Optional parameters (using regex)
app.get('/users/:id(\\d+)?', (req, res) => {
  res.json({ id: req.params.id || 'all' });
});
```

### Query Parameters

```typescript
// GET /search?q=express&limit=10
app.get('/search', (req, res) => {
  const { q, limit = 10 } = req.query;
  res.json({ query: q, limit });
});
```

### Route Handlers

```typescript
// Single callback
app.get('/example1', (req, res) => {
  res.send('Single callback');
});

// Multiple callbacks
app.get('/example2',
  (req, res, next) => {
    console.log('First handler');
    next();
  },
  (req, res) => {
    res.send('Second handler');
  }
);

// Array of callbacks
const cb1 = (req, res, next) => {
  console.log('CB1');
  next();
};

const cb2 = (req, res, next) => {
  console.log('CB2');
  next();
};

app.get('/example3', [cb1, cb2], (req, res) => {
  res.send('Array of callbacks');
});
```

### Express Router

```typescript
// routes/userRoutes.ts
import { Router } from 'express';
import * as userController from '../controllers/userController';

const router = Router();

router.get('/', userController.getAllUsers);
router.get('/:id', userController.getUserById);
router.post('/', userController.createUser);
router.put('/:id', userController.updateUser);
router.delete('/:id', userController.deleteUser);

export default router;

// index.ts
import userRoutes from './routes/userRoutes';
app.use('/api/users', userRoutes);
```

### Route Chaining

```typescript
app.route('/users')
  .get((req, res) => {
    res.json({ message: 'Get all users' });
  })
  .post((req, res) => {
    res.json({ message: 'Create user' });
  });

app.route('/users/:id')
  .get((req, res) => {
    res.json({ message: 'Get user' });
  })
  .put((req, res) => {
    res.json({ message: 'Update user' });
  })
  .delete((req, res) => {
    res.json({ message: 'Delete user' });
  });
```

---

## Middleware

Middleware functions have access to request, response, and the next middleware function in the application's request-response cycle.

### Built-in Middleware

```typescript
import express from 'express';
const app = express();

// Parse JSON bodies
app.use(express.json());

// Parse URL-encoded bodies
app.use(express.urlencoded({ extended: true }));

// Serve static files
app.use(express.static('public'));
```

### Application-Level Middleware

```typescript
// Executed for every request
app.use((req, res, next) => {
  console.log(`${req.method} ${req.url}`);
  next();
});

// Executed for specific path
app.use('/api', (req, res, next) => {
  console.log('API request');
  next();
});
```

### Router-Level Middleware

```typescript
const router = express.Router();

router.use((req, res, next) => {
  console.log('Router middleware');
  next();
});

router.get('/users', (req, res) => {
  res.json({ message: 'Users' });
});

app.use('/api', router);
```

### Custom Middleware

```typescript
// Logger middleware
const logger = (req: Request, res: Response, next: NextFunction) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${req.method} ${req.path}`);
  next();
};

// Request timing middleware
const requestTimer = (req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`Request took ${duration}ms`);
  });

  next();
};

// Auth middleware
const authenticate = (req: Request, res: Response, next: NextFunction) => {
  const token = req.headers.authorization;

  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }

  try {
    // Verify token
    const decoded = verifyToken(token);
    req.user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Usage
app.use(logger);
app.use(requestTimer);
app.use('/api/protected', authenticate);
```

### Third-Party Middleware

```typescript
// CORS
import cors from 'cors';
app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true
}));

// Helmet (security headers)
import helmet from 'helmet';
app.use(helmet());

// Compression
import compression from 'compression';
app.use(compression());

// Cookie parser
import cookieParser from 'cookie-parser';
app.use(cookieParser());

// Morgan (HTTP request logger)
import morgan from 'morgan';
app.use(morgan('combined'));

// Express validator
import { body, validationResult } from 'express-validator';

app.post('/users',
  body('email').isEmail(),
  body('password').isLength({ min: 6 }),
  (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }
    // Process request
  }
);
```

---

## Request and Response

### Request Object

```typescript
app.post('/example', (req: Request, res: Response) => {
  // Request body (requires body-parser or express.json())
  console.log(req.body);

  // URL parameters
  console.log(req.params);

  // Query parameters
  console.log(req.query);

  // Headers
  console.log(req.headers);
  console.log(req.get('Content-Type'));

  // Cookies (requires cookie-parser)
  console.log(req.cookies);
  console.log(req.signedCookies);

  // Request URL info
  console.log(req.protocol);      // http or https
  console.log(req.hostname);      // Host name
  console.log(req.path);          // Path part of URL
  console.log(req.originalUrl);   // Original URL
  console.log(req.baseUrl);       // Base URL

  // Request method
  console.log(req.method);        // GET, POST, etc.

  // IP address
  console.log(req.ip);
  console.log(req.ips);

  // Check content type
  console.log(req.is('json'));
  console.log(req.is('html'));

  res.send('OK');
});
```

### Response Object

```typescript
app.get('/response-examples', (req: Request, res: Response) => {
  // Send text
  res.send('Hello World');

  // Send JSON
  res.json({ message: 'Success', data: [] });

  // Set status code and send
  res.status(201).json({ message: 'Created' });

  // Send file
  res.sendFile('/path/to/file.pdf');

  // Download file
  res.download('/path/to/file.pdf', 'filename.pdf');

  // Redirect
  res.redirect('/new-url');
  res.redirect(301, '/permanent-redirect');

  // Set headers
  res.set('Content-Type', 'text/html');
  res.set({
    'Content-Type': 'text/html',
    'X-Custom-Header': 'value'
  });

  // Set cookies
  res.cookie('name', 'value', {
    maxAge: 900000,
    httpOnly: true,
    secure: true
  });

  // Clear cookie
  res.clearCookie('name');

  // Render view (requires template engine)
  res.render('index', { title: 'Home' });

  // End response
  res.end();

  // Send status with message
  res.sendStatus(404); // Sends "Not Found"
});
```

### Response Status Codes

```typescript
// Success
res.status(200).json({ message: 'OK' });
res.status(201).json({ message: 'Created' });
res.status(204).send(); // No Content

// Client Errors
res.status(400).json({ error: 'Bad Request' });
res.status(401).json({ error: 'Unauthorized' });
res.status(403).json({ error: 'Forbidden' });
res.status(404).json({ error: 'Not Found' });
res.status(422).json({ error: 'Unprocessable Entity' });

// Server Errors
res.status(500).json({ error: 'Internal Server Error' });
res.status(503).json({ error: 'Service Unavailable' });
```

---

## Error Handling

### Basic Error Handling

```typescript
// Synchronous error
app.get('/sync-error', (req, res) => {
  throw new Error('Synchronous error');
});

// Asynchronous error (must use next)
app.get('/async-error', (req, res, next) => {
  setTimeout(() => {
    try {
      throw new Error('Async error');
    } catch (err) {
      next(err);
    }
  }, 100);
});

// Promise rejection
app.get('/promise-error', async (req, res, next) => {
  try {
    await someAsyncOperation();
    res.json({ success: true });
  } catch (err) {
    next(err);
  }
});
```

### Error Handling Middleware

```typescript
// Error handler (must have 4 parameters)
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error(err.stack);
  res.status(500).json({
    error: {
      message: err.message,
      stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
    }
  });
});
```

### Custom Error Classes

```typescript
// errors/AppError.ts
export class AppError extends Error {
  statusCode: number;
  isOperational: boolean;

  constructor(message: string, statusCode: number) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true;

    Error.captureStackTrace(this, this.constructor);
  }
}

export class ValidationError extends AppError {
  constructor(message: string) {
    super(message, 400);
  }
}

export class NotFoundError extends AppError {
  constructor(message: string = 'Resource not found') {
    super(message, 404);
  }
}

export class UnauthorizedError extends AppError {
  constructor(message: string = 'Unauthorized') {
    super(message, 401);
  }
}

// Usage in controllers
import { NotFoundError } from '../errors/AppError';

app.get('/users/:id', async (req, res, next) => {
  try {
    const user = await findUserById(req.params.id);
    if (!user) {
      throw new NotFoundError('User not found');
    }
    res.json(user);
  } catch (err) {
    next(err);
  }
});

// Error handler
app.use((err: Error | AppError, req: Request, res: Response, next: NextFunction) => {
  if (err instanceof AppError) {
    return res.status(err.statusCode).json({
      error: {
        message: err.message,
        statusCode: err.statusCode
      }
    });
  }

  // Unknown error
  console.error('Unknown error:', err);
  res.status(500).json({
    error: {
      message: 'Internal server error'
    }
  });
});
```

### Async Error Wrapper

```typescript
// utils/asyncHandler.ts
export const asyncHandler = (fn: Function) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

// Usage
app.get('/users', asyncHandler(async (req: Request, res: Response) => {
  const users = await User.find();
  res.json(users);
}));
```

### 404 Handler

```typescript
// Catch 404 and forward to error handler
app.use((req, res, next) => {
  res.status(404).json({
    error: {
      message: 'Route not found',
      path: req.originalUrl
    }
  });
});
```

---

## Template Engines

### EJS (Embedded JavaScript)

```bash
npm install ejs
```

```typescript
import express from 'express';
const app = express();

// Set view engine
app.set('view engine', 'ejs');
app.set('views', './views');

// Render template
app.get('/', (req, res) => {
  res.render('index', {
    title: 'Home Page',
    user: { name: 'John' }
  });
});
```

**views/index.ejs:**
```html
<!DOCTYPE html>
<html>
<head>
  <title><%= title %></title>
</head>
<body>
  <h1>Welcome, <%= user.name %>!</h1>

  <% if (user.isAdmin) { %>
    <p>Admin panel</p>
  <% } %>

  <ul>
    <% ['Item 1', 'Item 2', 'Item 3'].forEach(item => { %>
      <li><%= item %></li>
    <% }); %>
  </ul>
</body>
</html>
```

### Pug (formerly Jade)

```bash
npm install pug
```

```typescript
app.set('view engine', 'pug');
app.set('views', './views');

app.get('/', (req, res) => {
  res.render('index', { title: 'Home', message: 'Hello Pug!' });
});
```

**views/index.pug:**
```pug
html
  head
    title= title
  body
    h1= message
    ul
      each item in ['Item 1', 'Item 2', 'Item 3']
        li= item
```

### Handlebars

```bash
npm install express-handlebars
```

```typescript
import { engine } from 'express-handlebars';

app.engine('handlebars', engine());
app.set('view engine', 'handlebars');
app.set('views', './views');

app.get('/', (req, res) => {
  res.render('home', {
    title: 'Home',
    items: ['Item 1', 'Item 2', 'Item 3']
  });
});
```

---

## Static Files

### Serving Static Files

```typescript
// Serve from 'public' directory
app.use(express.static('public'));

// Now you can access:
// http://localhost:3000/images/logo.png
// http://localhost:3000/css/style.css
// http://localhost:3000/js/app.js

// Multiple static directories
app.use(express.static('public'));
app.use(express.static('files'));

// Virtual path prefix
app.use('/static', express.static('public'));
// Now: http://localhost:3000/static/images/logo.png

// Absolute path
import path from 'path';
app.use('/static', express.static(path.join(__dirname, 'public')));
```

### Static File Options

```typescript
app.use(express.static('public', {
  maxAge: '1d',           // Cache for 1 day
  dotfiles: 'ignore',     // Ignore dotfiles
  index: 'index.html',    // Directory index file
  extensions: ['html'],   // File extension fallbacks
  setHeaders: (res, path) => {
    res.set('X-Custom-Header', 'value');
  }
}));
```

---

## Database Integration

### MongoDB with Mongoose

```bash
npm install mongoose
```

```typescript
// config/database.ts
import mongoose from 'mongoose';

export const connectDatabase = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/myapp');
    console.log('MongoDB connected');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
};

// models/User.ts
import mongoose, { Document, Schema } from 'mongoose';

export interface IUser extends Document {
  name: string;
  email: string;
  password: string;
  createdAt: Date;
}

const UserSchema = new Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

export default mongoose.model<IUser>('User', UserSchema);

// controllers/userController.ts
import User from '../models/User';

export const getAllUsers = async (req: Request, res: Response) => {
  try {
    const users = await User.find().select('-password');
    res.json(users);
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
};

export const createUser = async (req: Request, res: Response) => {
  try {
    const user = new User(req.body);
    await user.save();
    res.status(201).json(user);
  } catch (error) {
    res.status(400).json({ error: 'Invalid data' });
  }
};

// index.ts
import { connectDatabase } from './config/database';

connectDatabase();
```

### PostgreSQL with Sequelize

```bash
npm install sequelize pg pg-hstore
```

```typescript
// config/database.ts
import { Sequelize } from 'sequelize';

export const sequelize = new Sequelize(
  process.env.DB_NAME || 'myapp',
  process.env.DB_USER || 'postgres',
  process.env.DB_PASSWORD || 'password',
  {
    host: process.env.DB_HOST || 'localhost',
    dialect: 'postgres',
    logging: false
  }
);

export const connectDatabase = async () => {
  try {
    await sequelize.authenticate();
    console.log('PostgreSQL connected');
    await sequelize.sync();
  } catch (error) {
    console.error('Database connection error:', error);
  }
};

// models/User.ts
import { DataTypes, Model } from 'sequelize';
import { sequelize } from '../config/database';

export class User extends Model {
  public id!: number;
  public name!: string;
  public email!: string;
  public readonly createdAt!: Date;
}

User.init(
  {
    id: {
      type: DataTypes.INTEGER,
      autoIncrement: true,
      primaryKey: true
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false
    },
    email: {
      type: DataTypes.STRING,
      allowNull: false,
      unique: true
    }
  },
  {
    sequelize,
    tableName: 'users'
  }
);
```

### MySQL with mysql2

```bash
npm install mysql2
```

```typescript
import mysql from 'mysql2/promise';

const pool = mysql.createPool({
  host: 'localhost',
  user: 'root',
  password: 'password',
  database: 'myapp',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

app.get('/users', async (req, res) => {
  try {
    const [rows] = await pool.query('SELECT * FROM users');
    res.json(rows);
  } catch (error) {
    res.status(500).json({ error: 'Database error' });
  }
});
```

---

## Authentication

### JWT Authentication

```bash
npm install jsonwebtoken bcryptjs
npm install --save-dev @types/jsonwebtoken @types/bcryptjs
```

```typescript
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// Register
app.post('/auth/register', async (req, res) => {
  try {
    const { email, password, name } = req.body;

    // Check if user exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'User already exists' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create user
    const user = new User({
      email,
      password: hashedPassword,
      name
    });

    await user.save();

    // Generate token
    const token = jwt.sign(
      { userId: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.status(201).json({ token, user: { id: user.id, email, name } });
  } catch (error) {
    res.status(500).json({ error: 'Registration failed' });
  }
});

// Login
app.post('/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Verify password
    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate token
    const token = jwt.sign(
      { userId: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: '7d' }
    );

    res.json({
      token,
      user: { id: user.id, email: user.email, name: user.name }
    });
  } catch (error) {
    res.status(500).json({ error: 'Login failed' });
  }
});

// Auth middleware
interface AuthRequest extends Request {
  user?: any;
}

const authenticate = (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];

    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Protected route
app.get('/profile', authenticate, async (req: AuthRequest, res) => {
  try {
    const user = await User.findById(req.user.userId).select('-password');
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});
```

### Session-Based Authentication

```bash
npm install express-session connect-mongo
```

```typescript
import session from 'express-session';
import MongoStore from 'connect-mongo';

app.use(session({
  secret: process.env.SESSION_SECRET || 'your-secret',
  resave: false,
  saveUninitialized: false,
  store: MongoStore.create({
    mongoUrl: process.env.MONGODB_URI
  }),
  cookie: {
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    maxAge: 1000 * 60 * 60 * 24 * 7 // 7 days
  }
}));

// Login
app.post('/login', async (req, res) => {
  const { email, password } = req.body;

  const user = await User.findOne({ email });
  if (!user || !(await bcrypt.compare(password, user.password))) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  req.session.userId = user.id;
  res.json({ message: 'Logged in successfully' });
});

// Logout
app.post('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      return res.status(500).json({ error: 'Logout failed' });
    }
    res.clearCookie('connect.sid');
    res.json({ message: 'Logged out successfully' });
  });
});

// Auth middleware
const requireAuth = (req: Request, res: Response, next: NextFunction) => {
  if (!req.session.userId) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
};
```

---

## RESTful API

### Complete REST API Example

```typescript
// routes/api/users.ts
import { Router } from 'express';
import {
  getAllUsers,
  getUserById,
  createUser,
  updateUser,
  deleteUser
} from '../../controllers/userController';
import { authenticate } from '../../middleware/auth';
import { validateUser } from '../../middleware/validation';

const router = Router();

// GET /api/users - Get all users
router.get('/', authenticate, getAllUsers);

// GET /api/users/:id - Get user by ID
router.get('/:id', authenticate, getUserById);

// POST /api/users - Create new user
router.post('/', validateUser, createUser);

// PUT /api/users/:id - Update user
router.put('/:id', authenticate, validateUser, updateUser);

// DELETE /api/users/:id - Delete user
router.delete('/:id', authenticate, deleteUser);

export default router;

// controllers/userController.ts
import { Request, Response } from 'express';
import User from '../models/User';

export const getAllUsers = async (req: Request, res: Response) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;
    const skip = (page - 1) * limit;

    const users = await User.find()
      .select('-password')
      .limit(limit)
      .skip(skip);

    const total = await User.countDocuments();

    res.json({
      users,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
};

export const getUserById = async (req: Request, res: Response) => {
  try {
    const user = await User.findById(req.params.id).select('-password');

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json(user);
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
};

export const createUser = async (req: Request, res: Response) => {
  try {
    const user = new User(req.body);
    await user.save();

    const userResponse = user.toObject();
    delete userResponse.password;

    res.status(201).json(userResponse);
  } catch (error) {
    res.status(400).json({ error: 'Invalid data' });
  }
};

export const updateUser = async (req: Request, res: Response) => {
  try {
    const user = await User.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    ).select('-password');

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json(user);
  } catch (error) {
    res.status(400).json({ error: 'Invalid data' });
  }
};

export const deleteUser = async (req: Request, res: Response) => {
  try {
    const user = await User.findByIdAndDelete(req.params.id);

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.status(204).send();
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
};
```

### API Versioning

```typescript
// v1 routes
import v1Router from './routes/v1';
app.use('/api/v1', v1Router);

// v2 routes
import v2Router from './routes/v2';
app.use('/api/v2', v2Router);
```

### Rate Limiting

```bash
npm install express-rate-limit
```

```typescript
import rateLimit from 'express-rate-limit';

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP'
});

app.use('/api/', limiter);

// Different limits for different routes
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message: 'Too many login attempts'
});

app.use('/api/auth/login', authLimiter);
```

---

## File Uploads

### Multer for File Uploads

```bash
npm install multer
npm install --save-dev @types/multer
```

```typescript
import multer from 'multer';
import path from 'path';

// Storage configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

// File filter
const fileFilter = (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];

  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type'));
  }
};

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB
  },
  fileFilter: fileFilter
});

// Single file upload
app.post('/upload', upload.single('avatar'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  res.json({
    message: 'File uploaded successfully',
    file: {
      filename: req.file.filename,
      path: req.file.path,
      size: req.file.size
    }
  });
});

// Multiple files
app.post('/upload-multiple', upload.array('photos', 5), (req, res) => {
  res.json({
    message: 'Files uploaded successfully',
    files: req.files
  });
});

// Multiple fields
app.post('/upload-fields',
  upload.fields([
    { name: 'avatar', maxCount: 1 },
    { name: 'gallery', maxCount: 5 }
  ]),
  (req, res) => {
    res.json({
      message: 'Files uploaded successfully',
      files: req.files
    });
  }
);
```

---

## Security Best Practices

### Essential Security Packages

```bash
npm install helmet cors express-rate-limit express-validator
npm install --save-dev @types/cors
```

```typescript
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';

// Helmet - Set security headers
app.use(helmet());

// CORS configuration
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || 'http://localhost:3000',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
});
app.use(limiter);

// Prevent parameter pollution
import hpp from 'hpp';
app.use(hpp());

// Sanitize data
import mongoSanitize from 'express-mongo-sanitize';
app.use(mongoSanitize());

// XSS protection
import xss from 'xss-clean';
app.use(xss());
```

### Input Validation

```typescript
import { body, param, validationResult } from 'express-validator';

app.post('/users',
  body('email').isEmail().normalizeEmail(),
  body('password').isLength({ min: 8 }).matches(/\d/).matches(/[a-zA-Z]/),
  body('name').trim().isLength({ min: 2, max: 50 }),
  (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    // Process request
  }
);
```

### SQL Injection Prevention

```typescript
// Use parameterized queries
const [rows] = await pool.query(
  'SELECT * FROM users WHERE email = ?',
  [email]
);

// Use ORM/ODM
const user = await User.findOne({ email }); // Mongoose
```

### HTTPS Enforcement

```typescript
// Redirect HTTP to HTTPS
app.use((req, res, next) => {
  if (req.header('x-forwarded-proto') !== 'https' && process.env.NODE_ENV === 'production') {
    res.redirect(`https://${req.header('host')}${req.url}`);
  } else {
    next();
  }
});
```

---

## Testing

### Jest and Supertest

```bash
npm install --save-dev jest supertest @types/jest @types/supertest ts-jest
```

**jest.config.js:**
```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  collectCoverageFrom: ['src/**/*.ts', '!src/**/*.d.ts']
};
```

**tests/app.test.ts:**
```typescript
import request from 'supertest';
import app from '../src/app';

describe('User API', () => {
  it('GET /api/users should return all users', async () => {
    const response = await request(app)
      .get('/api/users')
      .expect('Content-Type', /json/)
      .expect(200);

    expect(response.body).toHaveProperty('users');
    expect(Array.isArray(response.body.users)).toBe(true);
  });

  it('POST /api/users should create a user', async () => {
    const newUser = {
      name: 'John Doe',
      email: 'john@example.com',
      password: 'password123'
    };

    const response = await request(app)
      .post('/api/users')
      .send(newUser)
      .expect('Content-Type', /json/)
      .expect(201);

    expect(response.body).toHaveProperty('id');
    expect(response.body.email).toBe(newUser.email);
  });

  it('GET /api/users/:id should return a user', async () => {
    const response = await request(app)
      .get('/api/users/1')
      .expect(200);

    expect(response.body).toHaveProperty('id');
    expect(response.body).toHaveProperty('name');
  });

  it('PUT /api/users/:id should update a user', async () => {
    const updates = { name: 'Jane Doe' };

    const response = await request(app)
      .put('/api/users/1')
      .send(updates)
      .expect(200);

    expect(response.body.name).toBe(updates.name);
  });

  it('DELETE /api/users/:id should delete a user', async () => {
    await request(app)
      .delete('/api/users/1')
      .expect(204);
  });
});

describe('Authentication', () => {
  it('POST /auth/register should register a user', async () => {
    const user = {
      name: 'Test User',
      email: 'test@example.com',
      password: 'password123'
    };

    const response = await request(app)
      .post('/auth/register')
      .send(user)
      .expect(201);

    expect(response.body).toHaveProperty('token');
    expect(response.body).toHaveProperty('user');
  });

  it('POST /auth/login should login a user', async () => {
    const credentials = {
      email: 'test@example.com',
      password: 'password123'
    };

    const response = await request(app)
      .post('/auth/login')
      .send(credentials)
      .expect(200);

    expect(response.body).toHaveProperty('token');
  });
});
```

---

## Production Deployment

### Environment Variables

**.env:**
```
NODE_ENV=production
PORT=3000
DATABASE_URL=mongodb://localhost:27017/myapp
JWT_SECRET=your-jwt-secret
SESSION_SECRET=your-session-secret
ALLOWED_ORIGINS=https://yourdomain.com
```

### Process Manager (PM2)

```bash
npm install -g pm2

# Start application
pm2 start dist/index.js --name "my-app"

# Start with cluster mode
pm2 start dist/index.js -i max --name "my-app"

# Save configuration
pm2 save

# Startup script
pm2 startup
```

**ecosystem.config.js:**
```javascript
module.exports = {
  apps: [{
    name: 'my-app',
    script: './dist/index.js',
    instances: 'max',
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'production',
      PORT: 3000
    },
    error_file: './logs/error.log',
    out_file: './logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss'
  }]
};
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm ci --only=production

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["node", "dist/index.js"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=mongodb://mongo:27017/myapp
    depends_on:
      - mongo

  mongo:
    image: mongo:6
    volumes:
      - mongo-data:/data/db
    ports:
      - "27017:27017"

volumes:
  mongo-data:
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Performance Optimization

```typescript
// Compression
import compression from 'compression';
app.use(compression());

// Response caching
import apicache from 'apicache';
const cache = apicache.middleware;

app.get('/api/users', cache('5 minutes'), getAllUsers);

// Database connection pooling
mongoose.connect(uri, {
  maxPoolSize: 10,
  minPoolSize: 5
});

// Clustering
import cluster from 'cluster';
import os from 'os';

if (cluster.isPrimary) {
  const cpuCount = os.cpus().length;

  for (let i = 0; i < cpuCount; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker) => {
    console.log(`Worker ${worker.process.pid} died`);
    cluster.fork();
  });
} else {
  app.listen(PORT);
}
```

---

## Resources

- **Official Documentation**: [https://expressjs.com/](https://expressjs.com/)
- **GitHub Repository**: [https://github.com/expressjs/express](https://github.com/expressjs/express)
- **Express Generator**: [https://expressjs.com/en/starter/generator.html](https://expressjs.com/en/starter/generator.html)
- **Best Practices**: [https://expressjs.com/en/advanced/best-practice-performance.html](https://expressjs.com/en/advanced/best-practice-performance.html)

---

Express.js remains the most popular Node.js framework due to its simplicity, flexibility, and robust ecosystem. Its minimalist approach allows developers to structure applications as they see fit, making it suitable for everything from small APIs to large-scale enterprise applications.
