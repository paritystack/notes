# FastAPI

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. It's designed to be easy to use and learn while providing production-ready code with automatic API documentation, data validation, and serialization.

## Table of Contents
- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Basic Application](#basic-application)
- [Path Operations](#path-operations)
- [Request and Response Models](#request-and-response-models)
- [Dependency Injection](#dependency-injection)
- [Database Integration](#database-integration)
- [Authentication and Security](#authentication-and-security)
- [Background Tasks](#background-tasks)
- [WebSockets](#websockets)
- [File Operations](#file-operations)
- [Testing](#testing)
- [Best Practices](#best-practices)
- [Production Deployment](#production-deployment)

---

## Introduction

**Key Features:**
- Fast performance (on par with NodeJS and Go)
- Automatic interactive API documentation (Swagger UI and ReDoc)
- Based on standard Python type hints
- Data validation using Pydantic
- Asynchronous support with async/await
- Dependency injection system
- OAuth2 and JWT authentication built-in
- WebSocket support
- GraphQL support
- Minimal boilerplate code
- Production-ready with automatic error responses

**Use Cases:**
- RESTful APIs
- Microservices
- Real-time applications
- Machine learning model serving
- Data science APIs
- Backend for mobile/web applications
- API gateways
- WebSocket servers

**Why FastAPI?**
- Fastest Python framework according to benchmarks
- Reduces bugs by ~40% with type checking
- Easy to learn, fast to code
- Editor support with autocomplete
- Reduces code duplication

---

## Installation and Setup

### Prerequisites

```bash
# Python 3.7+ required
python3 --version
pip --version
```

### Install FastAPI

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install FastAPI and Uvicorn
pip install fastapi
pip install "uvicorn[standard]"

# Install additional dependencies
pip install python-multipart  # For file uploads
pip install python-jose[cryptography]  # For JWT
pip install passlib[bcrypt]  # For password hashing
pip install sqlalchemy  # For database
pip install alembic  # For migrations
pip install pydantic[email]  # For email validation
```

### Project Structure

```
fastapi-app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── dependencies.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── users.py
│   │   └── auth.py
│   ├── services/
│   │   └── auth.py
│   └── utils/
│       └── security.py
├── tests/
│   └── test_main.py
├── alembic/
├── .env
├── requirements.txt
└── README.md
```

---

## Basic Application

### Minimal App

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# Run with: uvicorn main:app --reload
```

### Full Application Setup

**app/main.py:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import users, auth, items
from app.database import engine
from app import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="My API",
    description="A production-ready FastAPI application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(items.router, prefix="/items", tags=["items"])

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**app/config.py:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "FastAPI App"
    database_url: str = "sqlite:///./test.db"
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Path Operations

### HTTP Methods

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# GET
@app.get("/items")
async def get_items():
    return [{"id": 1, "name": "Item 1"}]

# GET with path parameter
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}

# POST
@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item

# PUT
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}

# PATCH
@app.patch("/items/{item_id}")
async def partial_update_item(item_id: int, item: dict):
    return {"item_id": item_id, "updated_fields": item}

# DELETE
@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int):
    return None
```

### Query Parameters

```python
from typing import Optional, List
from enum import Enum

class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"

@app.get("/items")
async def list_items(
    skip: int = 0,
    limit: int = 10,
    q: Optional[str] = None,
    sort: SortOrder = SortOrder.asc,
    tags: List[str] = []
):
    return {
        "skip": skip,
        "limit": limit,
        "q": q,
        "sort": sort,
        "tags": tags
    }

# Required query parameter
@app.get("/search")
async def search(q: str):  # Required
    return {"q": q}
```

### Path Parameters

```python
from uuid import UUID
from datetime import date

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

@app.get("/orders/{order_id}")
async def get_order(order_id: UUID):
    return {"order_id": str(order_id)}

@app.get("/posts/{year}/{month}/{day}")
async def get_posts_by_date(year: int, month: int, day: int):
    post_date = date(year, month, day)
    return {"date": post_date}

# Path with validation
from fastapi import Path

@app.get("/items/{item_id}")
async def get_item(
    item_id: int = Path(..., title="The ID of the item", ge=1)
):
    return {"item_id": item_id}
```

---

## Request and Response Models

### Pydantic Models

```python
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

    @validator('password')
    def password_strength(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None

class User(UserBase):
    id: int
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True

class UserInDB(User):
    hashed_password: str

# Product models
class Product(BaseModel):
    name: str
    description: Optional[str] = None
    price: float = Field(..., gt=0, description="Price must be greater than zero")
    tax: Optional[float] = 0
    tags: List[str] = []

class ProductResponse(Product):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
```

### Request Body

```python
from fastapi import Body

@app.post("/users")
async def create_user(user: UserCreate):
    return user

# Multiple body parameters
@app.post("/items")
async def create_item(
    item: Item,
    user: User,
    importance: int = Body(...)
):
    return {"item": item, "user": user, "importance": importance}

# Embed single body parameter
@app.post("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item = Body(..., embed=True)
):
    return {"item_id": item_id, "item": item}
```

### Response Models

```python
from fastapi import Response, status

@app.post("/users", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    # Don't return password in response
    return user

# Multiple response models
from fastapi.responses import JSONResponse

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id == 0:
        return JSONResponse(
            status_code=404,
            content={"message": "Item not found"}
        )
    return {"item_id": item_id}

# Response with Union types
from typing import Union

@app.get("/items/{item_id}", response_model=Union[Product, dict])
async def get_item(item_id: int):
    if item_id > 0:
        return product
    return {"message": "No item found"}
```

---

## Dependency Injection

### Basic Dependencies

```python
from fastapi import Depends, HTTPException, status
from typing import Optional

# Simple dependency
async def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

# Class-based dependency
class CommonQueryParams:
    def __init__(self, q: Optional[str] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/users")
async def read_users(commons: CommonQueryParams = Depends()):
    return commons
```

### Database Dependency

**app/database.py:**
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Current User Dependency

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    return user

# Use dependency
@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

---

## Database Integration

### SQLAlchemy Models

**app/models/user.py:**
```python
from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    items = relationship("Item", back_populates="owner")

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="items")
```

### CRUD Operations

**app/services/user.py:**
```python
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.utils.security import get_password_hash

def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user: UserUpdate):
    db_user = get_user(db, user_id)
    if db_user:
        update_data = user.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    db_user = get_user(db, user_id)
    if db_user:
        db.delete(db_user)
        db.commit()
    return db_user
```

### Router with Database

**app/routers/users.py:**
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.schemas.user import User, UserCreate, UserUpdate
from app.services import user as user_service
from app.dependencies import get_current_active_user

router = APIRouter()

@router.get("/", response_model=List[User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    users = user_service.get_users(db, skip=skip, limit=limit)
    return users

@router.get("/{user_id}", response_model=User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = user_service.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = user_service.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return user_service.create_user(db=db, user=user)

@router.put("/{user_id}", response_model=User)
def update_user(
    user_id: int,
    user: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if current_user.id != user_id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    db_user = user_service.update_user(db, user_id=user_id, user=user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    if current_user.id != user_id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")

    db_user = user_service.delete_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
```

---

## Authentication and Security

### Password Hashing

**app/utils/security.py:**
```python
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt
```

### JWT Authentication

**app/routers/auth.py:**
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from app.database import get_db
from app.schemas.auth import Token
from app.services import user as user_service
from app.utils.security import verify_password, create_access_token
from app.config import settings

router = APIRouter()

@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = user_service.get_user_by_email(db, email=form_data.username)

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = user_service.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    return user_service.create_user(db=db, user=user)
```

### API Key Authentication

```python
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-api-key-here"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

@app.get("/secure-data")
async def get_secure_data(api_key: str = Depends(verify_api_key)):
    return {"data": "sensitive information"}
```

---

## Background Tasks

```python
from fastapi import BackgroundTasks
import smtplib
from email.mime.text import MIMEText

def send_email(email: str, subject: str, body: str):
    # Email sending logic
    print(f"Sending email to {email}: {subject}")

def write_log(message: str):
    with open("log.txt", mode="a") as log:
        log.write(message + "\n")

@app.post("/send-notification/{email}")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, "Welcome!", "Thanks for signing up")
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    return {"message": "Notification sent in the background"}

# Multiple background tasks
@app.post("/users")
async def create_user(
    user: UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    db_user = user_service.create_user(db, user)

    background_tasks.add_task(send_email, user.email, "Welcome", "Thanks for joining!")
    background_tasks.add_task(write_log, f"User created: {user.email}")

    return db_user
```

---

## WebSockets

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")
```

---

## File Operations

### File Upload

```python
from fastapi import File, UploadFile
from typing import List
import shutil
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_path.stat().st_size
    }

@app.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    file_info = []

    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_info.append({
            "filename": file.filename,
            "size": file_path.stat().st_size
        })

    return {"files": file_info}
```

### File Download

```python
from fastapi.responses import FileResponse, StreamingResponse
import io

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)

@app.get("/stream")
async def stream_file():
    def iterfile():
        with open("large_file.txt", mode="rb") as file:
            yield from file

    return StreamingResponse(iterfile(), media_type="text/plain")
```

---

## Testing

```python
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine, get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
test_engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# Test functions
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FastAPI"}

def test_create_user():
    Base.metadata.create_all(bind=test_engine)

    response = client.post(
        "/users",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "TestPass123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data

    Base.metadata.drop_all(bind=test_engine)

def test_login():
    Base.metadata.create_all(bind=test_engine)

    # Create user
    client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "TestPass123"
        }
    )

    # Login
    response = client.post(
        "/auth/token",
        data={
            "username": "test@example.com",
            "password": "TestPass123"
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

    Base.metadata.drop_all(bind=test_engine)

def test_authenticated_route():
    # Get token
    response = client.post("/auth/token", data={"username": "test@example.com", "password": "TestPass123"})
    token = response.json()["access_token"]

    # Access protected route
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

---

## Best Practices

### 1. Project Structure

```python
# Use modular structure with routers
app/
├── routers/
│   ├── users.py
│   ├── items.py
│   └── auth.py
├── models/
├── schemas/
├── services/
└── utils/
```

### 2. Environment Variables

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    secret_key: str

    class Config:
        env_file = ".env"
```

### 3. Error Handling

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)}
    )
```

### 4. Async Operations

```python
import asyncio
import httpx

@app.get("/external-api")
async def call_external_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
```

### 5. Middleware

```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

## Production Deployment

### Requirements

**requirements.txt:**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
alembic==1.12.1
pydantic[email]==2.5.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
python-dotenv==1.0.0
```

### Uvicorn with Workers

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Resources

**Official Documentation:**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI GitHub](https://github.com/tiangolo/fastapi)
- [Pydantic Documentation](https://docs.pydantic.dev/)

**Learning Resources:**
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Full Stack FastAPI Template](https://github.com/tiangolo/full-stack-fastapi-template)
- [Awesome FastAPI](https://github.com/mjhea0/awesome-fastapi)

**Community:**
- [FastAPI Discord](https://discord.com/invite/VQjSZaeJmf)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/fastapi)
- [GitHub Discussions](https://github.com/tiangolo/fastapi/discussions)

**Related Tools:**
- [SQLModel](https://sqlmodel.tiangolo.com/) - SQL databases with Python
- [Uvicorn](https://www.uvicorn.org/) - ASGI server
- [Starlette](https://www.starlette.io/) - Underlying framework
