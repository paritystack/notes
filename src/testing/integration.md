# Integration Testing

Integration testing verifies that different modules or services work together correctly. Unlike unit tests that test individual components in isolation, integration tests validate interactions between components.

## Overview

Integration tests validate:
- API endpoints
- Database interactions
- External service integrations
- Component interactions
- End-to-end workflows

## Testing Strategies

### Bottom-Up Approach
```python
# Test data layer
def test_database_connection():
    db = connect_to_database()
    assert db.is_connected()

# Test service layer with real database
def test_user_service():
    service = UserService(real_database)
    user = service.create_user("test@example.com")
    assert user.email == "test@example.com"

# Test API layer with real services
def test_api_endpoint():
    response = client.post("/users", json={"email": "test@example.com"})
    assert response.status_code == 201
```

### Top-Down Approach
```python
# Test API first with mocked services
def test_api_with_mocks():
    with mock_user_service():
        response = client.post("/users", json={"email": "test@example.com"})
        assert response.status_code == 201

# Then test with real services
def test_api_with_real_services():
    response = client.post("/users", json={"email": "test@example.com"})
    user = db.query("SELECT * FROM users WHERE email = ?", "test@example.com")
    assert user is not None
```

## API Testing

```python
# Flask example
from flask import Flask
from flask.testing import FlaskClient

def test_api_endpoints(client: FlaskClient):
    # POST request
    response = client.post('/api/users', json={
        'username': 'testuser',
        'email': 'test@example.com'
    })
    assert response.status_code == 201
    data = response.get_json()
    user_id = data['id']

    # GET request
    response = client.get(f'/api/users/{user_id}')
    assert response.status_code == 200
    assert response.json['username'] == 'testuser'

    # PUT request
    response = client.put(f'/api/users/{user_id}', json={
        'email': 'newemail@example.com'
    })
    assert response.status_code == 200

    # DELETE request
    response = client.delete(f'/api/users/{user_id}')
    assert response.status_code == 204
```

## Database Testing

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="function")
def db_session():
    # Create test database
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()

def test_user_crud(db_session):
    # Create
    user = User(username="test", email="test@example.com")
    db_session.add(user)
    db_session.commit()

    # Read
    retrieved = db_session.query(User).filter_by(username="test").first()
    assert retrieved.email == "test@example.com"

    # Update
    retrieved.email = "updated@example.com"
    db_session.commit()

    # Delete
    db_session.delete(retrieved)
    db_session.commit()
    assert db_session.query(User).count() == 0
```

## Docker Compose for Testing

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: testdb
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  app:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://test:test@postgres:5432/testdb
      REDIS_URL: redis://redis:6379
    command: pytest tests/integration/
```

```bash
# Run integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## Test Fixtures and Setup

```python
import pytest

@pytest.fixture(scope="session")
def app():
    """Create application for testing"""
    app = create_app('testing')
    return app

@pytest.fixture(scope="session")
def client(app):
    """Create test client"""
    return app.test_client()

@pytest.fixture(scope="function")
def clean_database(db_session):
    """Clean database before each test"""
    db_session.query(User).delete()
    db_session.query(Order).delete()
    db_session.commit()
    yield
    db_session.rollback()

def test_with_clean_db(client, clean_database):
    response = client.post('/users', json={'username': 'test'})
    assert response.status_code == 201
```

## Mocking External Services

```python
from unittest.mock import patch, Mock

def test_external_api_integration():
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_get.return_value = mock_response

        result = fetch_external_data()
        assert result['data'] == 'test'
        mock_get.assert_called_once()
```

## Best Practices

1. **Isolate tests**: Each test should be independent
2. **Use test databases**: Never test against production
3. **Clean state**: Reset database/state between tests
4. **Test realistic scenarios**: Use production-like data
5. **Fast feedback**: Keep tests reasonably fast
6. **CI/CD integration**: Run automatically on commits
7. **Test error cases**: Not just happy paths
8. **Use containers**: Docker for consistent environments

## Quick Reference

| Aspect | Approach |
|--------|----------|
| Database | Use test DB or transactions |
| External APIs | Mock or use test endpoints |
| File system | Use temp directories |
| Time | Mock datetime |
| Network | Use test servers or mocks |

Integration tests ensure your system components work together correctly, catching issues that unit tests might miss.
