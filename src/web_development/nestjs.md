# NestJS

NestJS is a progressive Node.js framework for building efficient, reliable, and scalable server-side applications. It uses TypeScript by default and combines elements of Object-Oriented Programming (OOP), Functional Programming (FP), and Functional Reactive Programming (FRP). NestJS is heavily inspired by Angular's architecture and provides a robust application structure out of the box.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Core Concepts](#core-concepts)
- [Controllers](#controllers)
- [Providers and Dependency Injection](#providers-and-dependency-injection)
- [Modules](#modules)
- [Middleware](#middleware)
- [Exception Filters](#exception-filters)
- [Pipes](#pipes)
- [Guards](#guards)
- [Interceptors](#interceptors)
- [Database Integration](#database-integration)
- [Authentication and Authorization](#authentication-and-authorization)
- [GraphQL](#graphql)
- [Microservices](#microservices)
- [Testing](#testing)
- [Best Practices](#best-practices)
- [Production Deployment](#production-deployment)

---

## Introduction

**Key Features:**
- TypeScript-first framework with full TypeScript support
- Modular architecture with dependency injection
- Built on top of Express (or Fastify)
- Extensive ecosystem and CLI tools
- WebSockets and GraphQL support
- Microservices architecture support
- Comprehensive testing utilities
- OpenAPI (Swagger) integration
- Excellent documentation

**Use Cases:**
- Enterprise-grade REST APIs
- GraphQL APIs
- Microservices architectures
- Real-time applications with WebSockets
- Server-side rendered applications
- Backend for mobile applications
- Monolithic or distributed systems

**Philosophy:**
NestJS provides an opinionated structure while remaining flexible, making it ideal for large teams and enterprise applications where maintainability and scalability are crucial.

---

## Installation and Setup

### Prerequisites

```bash
# Node.js 16+ and npm required
node --version
npm --version
```

### Create New Project

```bash
# Install NestJS CLI globally
npm install -g @nestjs/cli

# Create new project
nest new my-nest-app

# Navigate to project
cd my-nest-app

# Start development server
npm run start:dev
```

### Manual Setup

```bash
# Create project directory
mkdir my-nest-app
cd my-nest-app

# Initialize npm
npm init -y

# Install core dependencies
npm install @nestjs/common @nestjs/core @nestjs/platform-express reflect-metadata rxjs

# Install development dependencies
npm install -D @nestjs/cli @nestjs/schematics typescript @types/node @types/express ts-node
```

### Project Structure

```
my-nest-app/
├── src/
│   ├── main.ts                 # Application entry point
│   ├── app.module.ts           # Root module
│   ├── app.controller.ts       # Root controller
│   ├── app.service.ts          # Root service
│   ├── modules/                # Feature modules
│   │   ├── users/
│   │   │   ├── users.module.ts
│   │   │   ├── users.controller.ts
│   │   │   ├── users.service.ts
│   │   │   ├── dto/            # Data Transfer Objects
│   │   │   ├── entities/       # Database entities
│   │   │   └── interfaces/     # TypeScript interfaces
│   │   └── auth/
│   ├── common/                 # Shared utilities
│   │   ├── guards/
│   │   ├── interceptors/
│   │   ├── pipes/
│   │   ├── filters/
│   │   └── decorators/
│   └── config/                 # Configuration files
├── test/                       # E2E tests
├── nest-cli.json              # NestJS CLI configuration
├── tsconfig.json              # TypeScript configuration
└── package.json
```

### Configuration Files

**tsconfig.json:**
```json
{
  "compilerOptions": {
    "module": "commonjs",
    "declaration": true,
    "removeComments": true,
    "emitDecoratorMetadata": true,
    "experimentalDecorators": true,
    "allowSyntheticDefaultImports": true,
    "target": "ES2021",
    "sourceMap": true,
    "outDir": "./dist",
    "baseUrl": "./",
    "incremental": true,
    "skipLibCheck": true,
    "strictNullChecks": false,
    "noImplicitAny": false,
    "strictBindCallApply": false,
    "forceConsistentCasingInFileNames": false,
    "noFallthroughCasesInSwitch": false
  }
}
```

**nest-cli.json:**
```json
{
  "collection": "@nestjs/schematics",
  "sourceRoot": "src",
  "compilerOptions": {
    "deleteOutDir": true
  }
}
```

---

## Core Concepts

### Application Bootstrap

**src/main.ts:**
```typescript
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // Enable CORS
  app.enableCors();

  // Global prefix
  app.setGlobalPrefix('api/v1');

  // Global validation pipe
  app.useGlobalPipes(new ValidationPipe({
    whitelist: true,
    forbidNonWhitelisted: true,
    transform: true,
  }));

  // Swagger documentation
  const config = new DocumentBuilder()
    .setTitle('My API')
    .setDescription('API documentation')
    .setVersion('1.0')
    .addBearerAuth()
    .build();
  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api/docs', app, document);

  await app.listen(3000);
  console.log(`Application is running on: ${await app.getUrl()}`);
}

bootstrap();
```

---

## Controllers

Controllers handle incoming requests and return responses to the client.

### Basic Controller

```typescript
import { Controller, Get, Post, Put, Delete, Body, Param, Query, HttpCode, HttpStatus } from '@nestjs/common';
import { UsersService } from './users.service';
import { CreateUserDto } from './dto/create-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';

@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @Get()
  findAll(@Query('page') page: number = 1, @Query('limit') limit: number = 10) {
    return this.usersService.findAll(page, limit);
  }

  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.usersService.findOne(+id);
  }

  @Post()
  @HttpCode(HttpStatus.CREATED)
  create(@Body() createUserDto: CreateUserDto) {
    return this.usersService.create(createUserDto);
  }

  @Put(':id')
  update(@Param('id') id: string, @Body() updateUserDto: UpdateUserDto) {
    return this.usersService.update(+id, updateUserDto);
  }

  @Delete(':id')
  @HttpCode(HttpStatus.NO_CONTENT)
  remove(@Param('id') id: string) {
    return this.usersService.remove(+id);
  }
}
```

### Advanced Controller Features

```typescript
import {
  Controller,
  Get,
  Post,
  UseGuards,
  UseInterceptors,
  UsePipes,
  Req,
  Res,
  Headers,
  Session
} from '@nestjs/common';
import { Request, Response } from 'express';
import { AuthGuard } from '@nestjs/passport';
import { LoggingInterceptor } from '../common/interceptors/logging.interceptor';
import { ValidationPipe } from '@nestjs/common';

@Controller('products')
@UseGuards(AuthGuard('jwt'))
@UseInterceptors(LoggingInterceptor)
export class ProductsController {
  @Get()
  async findAll(@Req() request: Request, @Headers('authorization') auth: string) {
    return {
      data: [],
      user: request.user,
    };
  }

  @Post()
  @UsePipes(new ValidationPipe({ transform: true }))
  async create(@Body() body: any, @Res() response: Response) {
    const result = await this.createProduct(body);
    return response.status(201).json(result);
  }

  @Get('download')
  async download(@Res() res: Response) {
    res.download('./files/report.pdf');
  }
}
```

---

## Providers and Dependency Injection

Providers are the fundamental concept in NestJS. Services, repositories, factories, and helpers can all be providers.

### Basic Service

```typescript
import { Injectable, NotFoundException } from '@nestjs/common';
import { CreateUserDto } from './dto/create-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';

@Injectable()
export class UsersService {
  private users = [];

  findAll(page: number, limit: number) {
    const start = (page - 1) * limit;
    const end = start + limit;

    return {
      data: this.users.slice(start, end),
      total: this.users.length,
      page,
      limit,
    };
  }

  findOne(id: number) {
    const user = this.users.find(u => u.id === id);
    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
    return user;
  }

  create(createUserDto: CreateUserDto) {
    const user = {
      id: this.users.length + 1,
      ...createUserDto,
      createdAt: new Date(),
    };
    this.users.push(user);
    return user;
  }

  update(id: number, updateUserDto: UpdateUserDto) {
    const user = this.findOne(id);
    Object.assign(user, updateUserDto);
    return user;
  }

  remove(id: number) {
    const index = this.users.findIndex(u => u.id === id);
    if (index === -1) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
    this.users.splice(index, 1);
  }
}
```

### Custom Provider

```typescript
// Value provider
const configProvider = {
  provide: 'CONFIG',
  useValue: {
    apiKey: process.env.API_KEY,
    apiUrl: process.env.API_URL,
  },
};

// Factory provider
const databaseProvider = {
  provide: 'DATABASE_CONNECTION',
  useFactory: async () => {
    const connection = await createConnection({
      type: 'postgres',
      host: 'localhost',
      port: 5432,
    });
    return connection;
  },
};

// Class provider
const loggerProvider = {
  provide: 'LOGGER',
  useClass: CustomLogger,
};

// Usage in module
@Module({
  providers: [configProvider, databaseProvider, loggerProvider],
})
export class AppModule {}
```

### Dependency Injection

```typescript
import { Injectable, Inject } from '@nestjs/common';

@Injectable()
export class ProductsService {
  constructor(
    @Inject('CONFIG') private config: any,
    @Inject('DATABASE_CONNECTION') private db: any,
    private readonly usersService: UsersService,
  ) {}

  async findAll() {
    const apiUrl = this.config.apiUrl;
    const users = await this.usersService.findAll(1, 10);
    const products = await this.db.query('SELECT * FROM products');
    return products;
  }
}
```

---

## Modules

Modules organize the application structure and enable modular architecture.

### Feature Module

```typescript
import { Module } from '@nestjs/common';
import { UsersController } from './users.controller';
import { UsersService } from './users.service';
import { TypeOrmModule } from '@nestjs/typeorm';
import { User } from './entities/user.entity';

@Module({
  imports: [TypeOrmModule.forFeature([User])],
  controllers: [UsersController],
  providers: [UsersService],
  exports: [UsersService], // Export to use in other modules
})
export class UsersModule {}
```

### Root Module

```typescript
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { UsersModule } from './modules/users/users.module';
import { AuthModule } from './modules/auth/auth.module';
import { ProductsModule } from './modules/products/products.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: process.env.DB_HOST,
      port: parseInt(process.env.DB_PORT),
      username: process.env.DB_USERNAME,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
      autoLoadEntities: true,
      synchronize: process.env.NODE_ENV !== 'production',
    }),
    UsersModule,
    AuthModule,
    ProductsModule,
  ],
})
export class AppModule {}
```

### Global Module

```typescript
import { Module, Global } from '@nestjs/common';
import { LoggerService } from './logger.service';

@Global()
@Module({
  providers: [LoggerService],
  exports: [LoggerService],
})
export class LoggerModule {}
```

### Dynamic Module

```typescript
import { Module, DynamicModule } from '@nestjs/common';
import { DatabaseService } from './database.service';

@Module({})
export class DatabaseModule {
  static forRoot(options: DatabaseOptions): DynamicModule {
    return {
      module: DatabaseModule,
      providers: [
        {
          provide: 'DATABASE_OPTIONS',
          useValue: options,
        },
        DatabaseService,
      ],
      exports: [DatabaseService],
    };
  }
}

// Usage
@Module({
  imports: [
    DatabaseModule.forRoot({
      host: 'localhost',
      port: 5432,
    }),
  ],
})
export class AppModule {}
```

---

## Middleware

Middleware functions execute before the route handler.

### Functional Middleware

```typescript
import { Request, Response, NextFunction } from 'express';

export function logger(req: Request, res: Response, next: NextFunction) {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
}
```

### Class-based Middleware

```typescript
import { Injectable, NestMiddleware } from '@nestjs/common';
import { Request, Response, NextFunction } from 'express';

@Injectable()
export class LoggerMiddleware implements NestMiddleware {
  use(req: Request, res: Response, next: NextFunction) {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    next();
  }
}

// Apply in module
import { Module, NestModule, MiddlewareConsumer } from '@nestjs/common';

@Module({})
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer
      .apply(LoggerMiddleware)
      .forRoutes('*'); // Apply to all routes

    // Or specific routes
    consumer
      .apply(LoggerMiddleware)
      .forRoutes({ path: 'users', method: RequestMethod.GET });
  }
}
```

---

## Exception Filters

Exception filters handle all thrown exceptions.

### Custom Exception Filter

```typescript
import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpException,
  HttpStatus
} from '@nestjs/common';
import { Request, Response } from 'express';

@Catch(HttpException)
export class HttpExceptionFilter implements ExceptionFilter {
  catch(exception: HttpException, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();
    const status = exception.getStatus();
    const exceptionResponse = exception.getResponse();

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: request.url,
      method: request.method,
      message: exceptionResponse['message'] || exception.message,
    });
  }
}

// Apply globally
app.useGlobalFilters(new HttpExceptionFilter());

// Or use in controller
@Controller('users')
@UseFilters(HttpExceptionFilter)
export class UsersController {}
```

### All Exceptions Filter

```typescript
@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>();

    const status =
      exception instanceof HttpException
        ? exception.getStatus()
        : HttpStatus.INTERNAL_SERVER_ERROR;

    const message =
      exception instanceof HttpException
        ? exception.message
        : 'Internal server error';

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: request.url,
      message,
    });
  }
}
```

---

## Pipes

Pipes transform input data or validate it before it reaches the route handler.

### Built-in Validation

```typescript
import { IsString, IsEmail, IsInt, Min, Max, IsOptional } from 'class-validator';

export class CreateUserDto {
  @IsString()
  @Length(3, 50)
  name: string;

  @IsEmail()
  email: string;

  @IsInt()
  @Min(18)
  @Max(120)
  age: number;

  @IsString()
  @IsOptional()
  bio?: string;
}

// Controller
@Post()
create(@Body() createUserDto: CreateUserDto) {
  return this.usersService.create(createUserDto);
}
```

### Custom Pipe

```typescript
import { PipeTransform, Injectable, ArgumentMetadata, BadRequestException } from '@nestjs/common';

@Injectable()
export class ParseIntPipe implements PipeTransform<string, number> {
  transform(value: string, metadata: ArgumentMetadata): number {
    const val = parseInt(value, 10);
    if (isNaN(val)) {
      throw new BadRequestException('Validation failed');
    }
    return val;
  }
}

// Usage
@Get(':id')
findOne(@Param('id', ParseIntPipe) id: number) {
  return this.usersService.findOne(id);
}
```

### Transformation Pipe

```typescript
import { PipeTransform, Injectable, ArgumentMetadata } from '@nestjs/common';

@Injectable()
export class TrimPipe implements PipeTransform {
  transform(value: any, metadata: ArgumentMetadata) {
    if (typeof value === 'string') {
      return value.trim();
    }
    if (typeof value === 'object') {
      Object.keys(value).forEach(key => {
        if (typeof value[key] === 'string') {
          value[key] = value[key].trim();
        }
      });
    }
    return value;
  }
}
```

---

## Guards

Guards determine whether a request should be handled by the route handler.

### Authentication Guard

```typescript
import { Injectable, CanActivate, ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';

@Injectable()
export class AuthGuard implements CanActivate {
  constructor(private jwtService: JwtService) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const token = this.extractTokenFromHeader(request);

    if (!token) {
      throw new UnauthorizedException('No token provided');
    }

    try {
      const payload = await this.jwtService.verifyAsync(token);
      request.user = payload;
      return true;
    } catch {
      throw new UnauthorizedException('Invalid token');
    }
  }

  private extractTokenFromHeader(request: any): string | undefined {
    const [type, token] = request.headers.authorization?.split(' ') ?? [];
    return type === 'Bearer' ? token : undefined;
  }
}
```

### Roles Guard

```typescript
import { Injectable, CanActivate, ExecutionContext } from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { SetMetadata } from '@nestjs/common';

export const ROLES_KEY = 'roles';
export const Roles = (...roles: string[]) => SetMetadata(ROLES_KEY, roles);

@Injectable()
export class RolesGuard implements CanActivate {
  constructor(private reflector: Reflector) {}

  canActivate(context: ExecutionContext): boolean {
    const requiredRoles = this.reflector.getAllAndOverride<string[]>(ROLES_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);

    if (!requiredRoles) {
      return true;
    }

    const { user } = context.switchToHttp().getRequest();
    return requiredRoles.some((role) => user.roles?.includes(role));
  }
}

// Usage
@Controller('admin')
@UseGuards(AuthGuard, RolesGuard)
export class AdminController {
  @Get()
  @Roles('admin')
  findAll() {
    return 'This route is only for admins';
  }
}
```

---

## Interceptors

Interceptors can transform the result returned from a function or extend basic function behavior.

### Logging Interceptor

```typescript
import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
} from '@nestjs/common';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';

@Injectable()
export class LoggingInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const now = Date.now();
    const request = context.switchToHttp().getRequest();
    const method = request.method;
    const url = request.url;

    return next.handle().pipe(
      tap(() => {
        const responseTime = Date.now() - now;
        console.log(`${method} ${url} - ${responseTime}ms`);
      }),
    );
  }
}
```

### Transform Interceptor

```typescript
import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
} from '@nestjs/common';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

export interface Response<T> {
  data: T;
  statusCode: number;
  timestamp: string;
}

@Injectable()
export class TransformInterceptor<T> implements NestInterceptor<T, Response<T>> {
  intercept(context: ExecutionContext, next: CallHandler): Observable<Response<T>> {
    return next.handle().pipe(
      map(data => ({
        data,
        statusCode: context.switchToHttp().getResponse().statusCode,
        timestamp: new Date().toISOString(),
      })),
    );
  }
}
```

### Caching Interceptor

```typescript
import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
} from '@nestjs/common';
import { Observable, of } from 'rxjs';
import { tap } from 'rxjs/operators';

@Injectable()
export class CacheInterceptor implements NestInterceptor {
  private cache = new Map();

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest();
    const key = request.url;

    if (this.cache.has(key)) {
      return of(this.cache.get(key));
    }

    return next.handle().pipe(
      tap(response => {
        this.cache.set(key, response);
      }),
    );
  }
}
```

---

## Database Integration

### TypeORM Integration

**Installation:**
```bash
npm install @nestjs/typeorm typeorm pg
```

**Entity:**
```typescript
import { Entity, Column, PrimaryGeneratedColumn, CreateDateColumn, UpdateDateColumn } from 'typeorm';

@Entity('users')
export class User {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ unique: true })
  email: string;

  @Column()
  name: string;

  @Column()
  password: string;

  @Column({ default: true })
  isActive: boolean;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}
```

**Service with Repository:**
```typescript
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { User } from './entities/user.entity';
import { CreateUserDto } from './dto/create-user.dto';

@Injectable()
export class UsersService {
  constructor(
    @InjectRepository(User)
    private usersRepository: Repository<User>,
  ) {}

  async findAll(): Promise<User[]> {
    return this.usersRepository.find();
  }

  async findOne(id: number): Promise<User> {
    return this.usersRepository.findOne({ where: { id } });
  }

  async create(createUserDto: CreateUserDto): Promise<User> {
    const user = this.usersRepository.create(createUserDto);
    return this.usersRepository.save(user);
  }

  async update(id: number, updateData: Partial<User>): Promise<User> {
    await this.usersRepository.update(id, updateData);
    return this.findOne(id);
  }

  async remove(id: number): Promise<void> {
    await this.usersRepository.delete(id);
  }
}
```

### Prisma Integration

**Installation:**
```bash
npm install @prisma/client
npm install -D prisma
npx prisma init
```

**Prisma Service:**
```typescript
import { Injectable, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit, OnModuleDestroy {
  async onModuleInit() {
    await this.$connect();
  }

  async onModuleDestroy() {
    await this.$disconnect();
  }
}
```

---

## Authentication and Authorization

### JWT Authentication

**Installation:**
```bash
npm install @nestjs/jwt @nestjs/passport passport passport-jwt
npm install -D @types/passport-jwt
```

**Auth Module:**
```typescript
import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { AuthService } from './auth.service';
import { AuthController } from './auth.controller';
import { JwtStrategy } from './strategies/jwt.strategy';
import { UsersModule } from '../users/users.module';

@Module({
  imports: [
    UsersModule,
    PassportModule,
    JwtModule.register({
      secret: process.env.JWT_SECRET,
      signOptions: { expiresIn: '1d' },
    }),
  ],
  controllers: [AuthController],
  providers: [AuthService, JwtStrategy],
  exports: [AuthService],
})
export class AuthModule {}
```

**Auth Service:**
```typescript
import { Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { UsersService } from '../users/users.service';
import * as bcrypt from 'bcrypt';

@Injectable()
export class AuthService {
  constructor(
    private usersService: UsersService,
    private jwtService: JwtService,
  ) {}

  async signIn(email: string, password: string) {
    const user = await this.usersService.findByEmail(email);

    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (!isPasswordValid) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const payload = { sub: user.id, email: user.email };

    return {
      access_token: await this.jwtService.signAsync(payload),
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
      },
    };
  }

  async signUp(email: string, password: string, name: string) {
    const hashedPassword = await bcrypt.hash(password, 10);

    const user = await this.usersService.create({
      email,
      password: hashedPassword,
      name,
    });

    const payload = { sub: user.id, email: user.email };

    return {
      access_token: await this.jwtService.signAsync(payload),
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
      },
    };
  }
}
```

**JWT Strategy:**
```typescript
import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy } from 'passport-jwt';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor() {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: process.env.JWT_SECRET,
    });
  }

  async validate(payload: any) {
    return { userId: payload.sub, email: payload.email };
  }
}
```

---

## GraphQL

**Installation:**
```bash
npm install @nestjs/graphql @nestjs/apollo @apollo/server graphql
```

**Configuration:**
```typescript
import { Module } from '@nestjs/common';
import { GraphQLModule } from '@nestjs/graphql';
import { ApolloDriver, ApolloDriverConfig } from '@nestjs/apollo';

@Module({
  imports: [
    GraphQLModule.forRoot<ApolloDriverConfig>({
      driver: ApolloDriver,
      autoSchemaFile: true,
      playground: true,
    }),
  ],
})
export class AppModule {}
```

**Resolver:**
```typescript
import { Resolver, Query, Mutation, Args, Int } from '@nestjs/graphql';
import { User } from './models/user.model';
import { UsersService } from './users.service';
import { CreateUserInput } from './dto/create-user.input';

@Resolver(() => User)
export class UsersResolver {
  constructor(private usersService: UsersService) {}

  @Query(() => [User], { name: 'users' })
  findAll() {
    return this.usersService.findAll();
  }

  @Query(() => User, { name: 'user' })
  findOne(@Args('id', { type: () => Int }) id: number) {
    return this.usersService.findOne(id);
  }

  @Mutation(() => User)
  createUser(@Args('createUserInput') createUserInput: CreateUserInput) {
    return this.usersService.create(createUserInput);
  }
}
```

---

## Microservices

### TCP Microservice

**Server:**
```typescript
import { NestFactory } from '@nestjs/core';
import { Transport, MicroserviceOptions } from '@nestjs/microservices';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.createMicroservice<MicroserviceOptions>(AppModule, {
    transport: Transport.TCP,
    options: {
      host: '127.0.0.1',
      port: 8877,
    },
  });

  await app.listen();
}

bootstrap();
```

**Controller:**
```typescript
import { Controller } from '@nestjs/common';
import { MessagePattern, Payload } from '@nestjs/microservices';

@Controller()
export class MathController {
  @MessagePattern({ cmd: 'sum' })
  accumulate(@Payload() data: number[]): number {
    return (data || []).reduce((a, b) => a + b);
  }
}
```

**Client:**
```typescript
import { Injectable } from '@nestjs/common';
import { ClientProxy, ClientProxyFactory, Transport } from '@nestjs/microservices';

@Injectable()
export class AppService {
  private client: ClientProxy;

  constructor() {
    this.client = ClientProxyFactory.create({
      transport: Transport.TCP,
      options: {
        host: '127.0.0.1',
        port: 8877,
      },
    });
  }

  async accumulate() {
    const pattern = { cmd: 'sum' };
    const payload = [1, 2, 3];
    return this.client.send<number>(pattern, payload);
  }
}
```

---

## Testing

### Unit Testing

```typescript
import { Test, TestingModule } from '@nestjs/testing';
import { UsersService } from './users.service';
import { getRepositoryToken } from '@nestjs/typeorm';
import { User } from './entities/user.entity';

describe('UsersService', () => {
  let service: UsersService;

  const mockUserRepository = {
    find: jest.fn(),
    findOne: jest.fn(),
    create: jest.fn(),
    save: jest.fn(),
    update: jest.fn(),
    delete: jest.fn(),
  };

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        UsersService,
        {
          provide: getRepositoryToken(User),
          useValue: mockUserRepository,
        },
      ],
    }).compile();

    service = module.get<UsersService>(UsersService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('findAll', () => {
    it('should return an array of users', async () => {
      const users = [{ id: 1, name: 'John' }];
      mockUserRepository.find.mockResolvedValue(users);

      const result = await service.findAll();
      expect(result).toEqual(users);
      expect(mockUserRepository.find).toHaveBeenCalled();
    });
  });
});
```

### E2E Testing

```typescript
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from './../src/app.module';

describe('UsersController (e2e)', () => {
  let app: INestApplication;

  beforeAll(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  it('/users (GET)', () => {
    return request(app.getHttpServer())
      .get('/users')
      .expect(200)
      .expect('Content-Type', /json/);
  });

  it('/users (POST)', () => {
    return request(app.getHttpServer())
      .post('/users')
      .send({
        name: 'John Doe',
        email: 'john@example.com',
      })
      .expect(201)
      .then(response => {
        expect(response.body).toHaveProperty('id');
        expect(response.body.name).toBe('John Doe');
      });
  });
});
```

---

## Best Practices

### 1. Module Organization

```typescript
// Feature-based organization
src/
├── modules/
│   ├── users/
│   │   ├── dto/
│   │   ├── entities/
│   │   ├── users.controller.ts
│   │   ├── users.service.ts
│   │   ├── users.module.ts
│   │   └── users.controller.spec.ts
│   └── products/
└── common/
    ├── guards/
    ├── interceptors/
    ├── pipes/
    └── decorators/
```

### 2. DTOs and Validation

```typescript
import { IsString, IsEmail, IsOptional, MinLength } from 'class-validator';
import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';

export class CreateUserDto {
  @ApiProperty({ example: 'John Doe' })
  @IsString()
  @MinLength(3)
  name: string;

  @ApiProperty({ example: 'john@example.com' })
  @IsEmail()
  email: string;

  @ApiPropertyOptional()
  @IsString()
  @IsOptional()
  bio?: string;
}
```

### 3. Environment Configuration

```typescript
import { ConfigModule, ConfigService } from '@nestjs/config';
import * as Joi from 'joi';

@Module({
  imports: [
    ConfigModule.forRoot({
      validationSchema: Joi.object({
        NODE_ENV: Joi.string()
          .valid('development', 'production', 'test')
          .default('development'),
        PORT: Joi.number().default(3000),
        DATABASE_URL: Joi.string().required(),
        JWT_SECRET: Joi.string().required(),
      }),
    }),
  ],
})
export class AppModule {}
```

### 4. Error Handling

```typescript
import { HttpException, HttpStatus } from '@nestjs/common';

export class UserNotFoundException extends HttpException {
  constructor(userId: number) {
    super(`User with ID ${userId} not found`, HttpStatus.NOT_FOUND);
  }
}

// Usage
throw new UserNotFoundException(id);
```

### 5. Logging

```typescript
import { Logger, Injectable } from '@nestjs/common';

@Injectable()
export class UsersService {
  private readonly logger = new Logger(UsersService.name);

  async findAll() {
    this.logger.log('Fetching all users');
    try {
      const users = await this.usersRepository.find();
      this.logger.log(`Found ${users.length} users`);
      return users;
    } catch (error) {
      this.logger.error('Failed to fetch users', error.stack);
      throw error;
    }
  }
}
```

---

## Production Deployment

### Environment Variables

**.env.production:**
```bash
NODE_ENV=production
PORT=3000
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
JWT_SECRET=your-secret-key
REDIS_URL=redis://localhost:6379
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["node", "dist/main"]
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
      - DATABASE_URL=postgresql://postgres:password@db:5432/mydb
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### PM2 Deployment

**ecosystem.config.js:**
```javascript
module.exports = {
  apps: [{
    name: 'nest-app',
    script: 'dist/main.js',
    instances: 'max',
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'production',
    },
  }],
};
```

### Health Checks

```typescript
import { Controller, Get } from '@nestjs/common';
import {
  HealthCheckService,
  HealthCheck,
  TypeOrmHealthIndicator,
  MemoryHealthIndicator
} from '@nestjs/terminus';

@Controller('health')
export class HealthController {
  constructor(
    private health: HealthCheckService,
    private db: TypeOrmHealthIndicator,
    private memory: MemoryHealthIndicator,
  ) {}

  @Get()
  @HealthCheck()
  check() {
    return this.health.check([
      () => this.db.pingCheck('database'),
      () => this.memory.checkHeap('memory_heap', 150 * 1024 * 1024),
    ]);
  }
}
```

### Performance Optimization

```typescript
// Enable compression
import * as compression from 'compression';
app.use(compression());

// Enable helmet for security
import helmet from 'helmet';
app.use(helmet());

// Rate limiting
import { ThrottlerModule } from '@nestjs/throttler';
@Module({
  imports: [
    ThrottlerModule.forRoot({
      ttl: 60,
      limit: 10,
    }),
  ],
})

// Caching
import { CacheModule } from '@nestjs/cache-manager';
@Module({
  imports: [
    CacheModule.register({
      ttl: 5,
      max: 100,
    }),
  ],
})
```

---

## Resources

**Official Documentation:**
- [NestJS Documentation](https://docs.nestjs.com/)
- [NestJS Recipes](https://docs.nestjs.com/recipes/documentation)
- [NestJS OpenAPI](https://docs.nestjs.com/openapi/introduction)

**Learning Resources:**
- [NestJS Official Courses](https://courses.nestjs.com/)
- [NestJS Fundamentals](https://learn.nestjs.com/)
- [Awesome NestJS](https://github.com/nestjs/awesome-nestjs)

**Community:**
- [NestJS Discord](https://discord.gg/nestjs)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/nestjs)
- [GitHub Repository](https://github.com/nestjs/nest)

**Tools:**
- [NestJS CLI](https://docs.nestjs.com/cli/overview)
- [Swagger UI](https://swagger.io/tools/swagger-ui/)
- [TypeORM](https://typeorm.io/)
- [Prisma](https://www.prisma.io/)
