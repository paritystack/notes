# Tool Use in AI Systems

A comprehensive guide to tool use (function calling) in Large Language Models, enabling AI assistants to interact with external systems, APIs, and execute actions.

## Table of Contents
- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
- [How Tool Use Works](#how-tool-use-works)
- [Tool Definition](#tool-definition)
- [Implementation Patterns](#implementation-patterns)
- [Platform-Specific Implementations](#platform-specific-implementations)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Advanced Patterns](#advanced-patterns)
- [Security Considerations](#security-considerations)
- [Performance Optimization](#performance-optimization)
- [Testing Tool Use](#testing-tool-use)
- [Real-World Examples](#real-world-examples)

## Introduction

Tool use (also called function calling or tool calling) enables Large Language Models to interact with external systems beyond text generation. Instead of just generating text responses, models can:
- Call APIs
- Query databases
- Execute code
- Access real-time information
- Perform calculations
- Interact with files and systems
- Control external services

This transforms LLMs from passive text generators into **agentic systems** that can take actions and interact with the world.

## Core Concepts

### What is a Tool?

A tool is a defined function that an LLM can invoke to perform specific actions:

```python
# Example: A simple weather tool
def get_weather(location: str, units: str = "celsius") -> dict:
    """
    Get current weather for a location.

    Args:
        location: City name or coordinates
        units: Temperature units (celsius/fahrenheit)

    Returns:
        dict: Weather information
    """
    return {
        "temperature": 22,
        "conditions": "sunny",
        "location": location,
        "units": units
    }
```

### Key Components

1. **Tool Definition**: Schema describing the tool's name, description, and parameters
2. **Tool Invocation**: Model decides when and how to call the tool
3. **Tool Execution**: System executes the tool with provided parameters
4. **Result Integration**: Tool results are fed back to the model
5. **Response Generation**: Model uses tool results to generate final response

### Tool Use vs. Prompt Engineering

| Aspect | Prompt Engineering | Tool Use |
|--------|-------------------|----------|
| **Capability** | Text-to-text | Text-to-action |
| **Real-time Data** | Limited to training data | Can access current data |
| **Actions** | Cannot perform actions | Can execute functions |
| **Accuracy** | Prone to hallucination | Deterministic execution |
| **Use Case** | Content generation | Interactive agents |

## How Tool Use Works

### Basic Flow

```
User Input
    ↓
LLM Processing
    ↓
Decision Point: Need Tool?
    ├─ No → Generate Response
    └─ Yes → Select Tool & Parameters
              ↓
         Execute Tool
              ↓
         Get Results
              ↓
         Feed Back to LLM
              ↓
         Generate Final Response
```

### Example Conversation

```
User: "What's the weather in San Francisco and should I bring an umbrella?"

LLM: [Decides to use weather tool]
Tool Call: get_weather(location="San Francisco", units="fahrenheit")

Tool Result: {
  "temperature": 65,
  "conditions": "partly cloudy",
  "precipitation_chance": 20
}

LLM Response: "The weather in San Francisco is currently 65°F and partly cloudy
with only a 20% chance of precipitation. An umbrella probably isn't necessary,
but you might want to bring a light jacket."
```

## Tool Definition

### Schema Format

Most platforms use JSON Schema or similar formats:

```json
{
  "name": "get_stock_price",
  "description": "Get the current stock price for a given ticker symbol",
  "parameters": {
    "type": "object",
    "properties": {
      "ticker": {
        "type": "string",
        "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
      },
      "exchange": {
        "type": "string",
        "enum": ["NYSE", "NASDAQ", "LSE"],
        "description": "Stock exchange"
      }
    },
    "required": ["ticker"]
  }
}
```

### Good Tool Descriptions

The quality of tool descriptions directly impacts model's ability to use them correctly:

```json
{
  "name": "send_email",
  "description": "Send an email to one or more recipients. Use this when the user explicitly requests to send an email or when an action requires email notification. Do not use for general email composition help.",
  "parameters": {
    "type": "object",
    "properties": {
      "to": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of recipient email addresses. Must be valid email format."
      },
      "subject": {
        "type": "string",
        "description": "Email subject line. Should be concise and descriptive."
      },
      "body": {
        "type": "string",
        "description": "Email body content. Can include HTML formatting if html_format is true."
      },
      "html_format": {
        "type": "boolean",
        "description": "Whether to send as HTML email. Default is false (plain text).",
        "default": false
      }
    },
    "required": ["to", "subject", "body"]
  }
}
```

### Tool Description Best Practices

✅ **Good Practices:**
- Clear, specific descriptions
- Explain when to use the tool
- Document parameter formats and constraints
- Include examples in descriptions
- Specify required vs. optional parameters
- Use enums for limited choices

❌ **Bad Practices:**
- Vague descriptions: "Does stuff with data"
- Missing parameter constraints
- Unclear when to invoke the tool
- Overlapping tool purposes

## Implementation Patterns

### Pattern 1: Single Tool Call

Model makes one tool call and generates response:

```python
# User: "What's 15% of 250?"

# Model decides to use calculator
tool_call = {
    "name": "calculate",
    "parameters": {
        "expression": "0.15 * 250"
    }
}

# Execute tool
result = execute_tool(tool_call)  # Returns: {"result": 37.5}

# Model generates response
response = "15% of 250 is 37.5"
```

### Pattern 2: Multiple Parallel Tools

Model calls multiple independent tools simultaneously:

```python
# User: "Compare the weather in NYC and LA"

# Model makes parallel calls
tool_calls = [
    {
        "name": "get_weather",
        "parameters": {"location": "New York City"}
    },
    {
        "name": "get_weather",
        "parameters": {"location": "Los Angeles"}
    }
]

# Execute in parallel
results = parallel_execute(tool_calls)

# Model synthesizes response
response = "NYC is 45°F and rainy, while LA is 72°F and sunny.
LA has much better weather today."
```

### Pattern 3: Sequential Tool Chaining

Model uses output from one tool to inform the next:

```python
# User: "Buy 10 shares of the highest performing tech stock today"

# Step 1: Get top performers
tool_call_1 = {
    "name": "get_top_stocks",
    "parameters": {"sector": "technology", "limit": 1}
}
result_1 = {"ticker": "NVDA", "price": 875.50}

# Step 2: Execute purchase
tool_call_2 = {
    "name": "buy_stock",
    "parameters": {
        "ticker": "NVDA",
        "quantity": 10,
        "price": 875.50
    }
}
result_2 = {"status": "success", "order_id": "12345"}

# Model confirms
response = "I've purchased 10 shares of NVDA at $875.50 each.
Order ID: 12345"
```

### Pattern 4: Iterative Tool Use (Agentic)

Model repeatedly uses tools until task is complete:

```python
# User: "Debug why the API is returning 500 errors"

# Iteration 1: Check logs
tool_call = {"name": "get_logs", "parameters": {"service": "api", "level": "error"}}
result = {"errors": ["Database connection timeout"]}

# Iteration 2: Check database
tool_call = {"name": "check_database", "parameters": {"check": "connections"}}
result = {"active_connections": 100, "max_connections": 100}

# Iteration 3: Get database config
tool_call = {"name": "get_config", "parameters": {"service": "database"}}
result = {"max_connections": 100}

# Model synthesizes findings
response = "The API is failing because the database has reached its
connection limit (100/100). You need to either increase max_connections
in the database config or implement connection pooling."
```

## Platform-Specific Implementations

### OpenAI Function Calling

```python
import openai

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Make API call
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Boston?"}],
    tools=tools,
    tool_choice="auto"  # Let model decide when to use tools
)

# Check if model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]

    # Execute the function
    if tool_call.function.name == "get_weather":
        import json
        args = json.loads(tool_call.function.arguments)
        result = get_weather(**args)

        # Send result back to model
        messages = [
            {"role": "user", "content": "What's the weather in Boston?"},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            }
        ]

        final_response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        print(final_response.choices[0].message.content)
```

### Anthropic Claude Tool Use

```python
import anthropic

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature"
                }
            },
            "required": ["location"]
        }
    }
]

# Make API call
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}]
)

# Handle tool use
if message.stop_reason == "tool_use":
    tool_use = next(block for block in message.content if block.type == "tool_use")

    # Execute tool
    tool_result = get_weather(**tool_use.input)

    # Continue conversation
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in San Francisco?"},
            {"role": "assistant", "content": message.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(tool_result)
                    }
                ]
            }
        ]
    )

    print(response.content[0].text)
```

### LangChain Integration

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI

# Define tools
tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="Useful for mathematical calculations. Input should be a math expression."
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="Get current weather. Input should be a city name."
    )
]

# Initialize agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
result = agent.run("What's the weather in NYC and what's 15% of 200?")
```

### Local Tool Use (llama.cpp)

```python
# Tool definitions for local models
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# Format prompt with tool definitions
prompt = f"""You have access to the following tools:
{json.dumps(tools, indent=2)}

To use a tool, respond with JSON in this format:
{{"tool": "tool_name", "parameters": {{}}}}

User: What time is it?
Assistant:"""

# Model generates tool call
response = model.generate(prompt)
# {"tool": "get_current_time", "parameters": {}}

# Parse and execute
tool_call = json.loads(response)
result = execute_tool(tool_call)

# Feed back to model
final_prompt = f"{prompt}\n{response}\nTool result: {result}\nAssistant:"
final_response = model.generate(final_prompt)
```

## Best Practices

### 1. Clear Tool Naming

```python
# ❌ Bad: Vague names
def get()  # Get what?
def process_data()  # What kind of processing?
def send()  # Send what, where?

# ✅ Good: Specific names
def get_user_profile()
def calculate_compound_interest()
def send_email_notification()
```

### 2. Comprehensive Descriptions

```python
# ❌ Bad: Minimal description
{
    "name": "search",
    "description": "Search for stuff"
}

# ✅ Good: Detailed description
{
    "name": "search_knowledge_base",
    "description": """Search the company knowledge base for documents and articles.

    Use this tool when users ask questions that require company-specific information,
    policies, or documentation. The search uses semantic similarity to find relevant
    content. Returns top 5 most relevant results with titles and snippets.

    Examples of when to use:
    - "What's our vacation policy?"
    - "How do I submit an expense report?"
    - "Find documentation about our API"
    """
}
```

### 3. Parameter Validation

```python
def transfer_money(from_account: str, to_account: str, amount: float):
    """Transfer money between accounts."""

    # Validate inputs
    if not from_account or not to_account:
        raise ValueError("Account IDs cannot be empty")

    if amount <= 0:
        raise ValueError("Amount must be positive")

    if amount > 10000:
        raise ValueError("Amount exceeds single transaction limit")

    # Check account format
    if not re.match(r'^\d{10}$', from_account):
        raise ValueError("Invalid account ID format")

    # Execute transfer
    return execute_transfer(from_account, to_account, amount)
```

### 4. Error Handling

```python
def get_stock_price(ticker: str) -> dict:
    """Get stock price with robust error handling."""
    try:
        # Validate ticker
        ticker = ticker.upper().strip()

        # Make API call
        response = stock_api.get_quote(ticker)

        return {
            "success": True,
            "ticker": ticker,
            "price": response.price,
            "timestamp": response.timestamp
        }

    except InvalidTickerError:
        return {
            "success": False,
            "error": f"Invalid ticker symbol: {ticker}",
            "suggestion": "Please use a valid stock ticker (e.g., AAPL, GOOGL)"
        }

    except APIError as e:
        return {
            "success": False,
            "error": "Unable to fetch stock data",
            "details": str(e)
        }

    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in get_stock_price: {e}")
        return {
            "success": False,
            "error": "An unexpected error occurred"
        }
```

### 5. Tool Organization

```python
# Group related tools
class WeatherTools:
    """Collection of weather-related tools."""

    @staticmethod
    def get_current_weather(location: str) -> dict:
        """Get current weather conditions."""
        pass

    @staticmethod
    def get_forecast(location: str, days: int = 5) -> dict:
        """Get weather forecast."""
        pass

    @staticmethod
    def get_weather_alerts(location: str) -> dict:
        """Get active weather alerts."""
        pass

# Register tools
tools = [
    create_tool_definition(WeatherTools.get_current_weather),
    create_tool_definition(WeatherTools.get_forecast),
    create_tool_definition(WeatherTools.get_weather_alerts)
]
```

### 6. Idempotency

```python
# ❌ Bad: Not idempotent
def create_user(username: str, email: str):
    """Creates a user - calling twice creates duplicates!"""
    user_id = generate_id()
    db.insert({"id": user_id, "username": username, "email": email})
    return {"user_id": user_id}

# ✅ Good: Idempotent
def create_user(username: str, email: str):
    """Creates a user or returns existing if already exists."""
    existing = db.find_one({"username": username})
    if existing:
        return {
            "user_id": existing.id,
            "created": False,
            "message": "User already exists"
        }

    user_id = generate_id()
    db.insert({"id": user_id, "username": username, "email": email})
    return {
        "user_id": user_id,
        "created": True,
        "message": "User created successfully"
    }
```

### 7. Return Structured Data

```python
# ❌ Bad: Unstructured string
def get_weather(location: str) -> str:
    return "It's 72 degrees and sunny in San Francisco"

# ✅ Good: Structured data
def get_weather(location: str) -> dict:
    return {
        "location": "San Francisco, CA",
        "temperature": 72,
        "temperature_unit": "fahrenheit",
        "conditions": "sunny",
        "humidity": 65,
        "wind_speed": 8,
        "wind_unit": "mph",
        "timestamp": "2025-01-15T14:30:00Z"
    }
```

## Common Pitfalls

### 1. Tool Overload

```python
# ❌ Bad: Too many similar tools
tools = [
    "get_user_by_id",
    "get_user_by_email",
    "get_user_by_username",
    "get_user_by_phone",
    # ... 10 more variations
]

# ✅ Good: Single flexible tool
tools = [
    {
        "name": "get_user",
        "description": "Get user by ID, email, username, or phone",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "email": {"type": "string"},
                "username": {"type": "string"},
                "phone": {"type": "string"}
            }
        }
    }
]
```

### 2. Missing Constraints

```python
# ❌ Bad: No limits
{
    "name": "send_emails",
    "parameters": {
        "recipients": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

# ✅ Good: With constraints
{
    "name": "send_emails",
    "parameters": {
        "recipients": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 50,  # Prevent abuse
            "minItems": 1
        }
    }
}
```

### 3. Unclear Success Indicators

```python
# ❌ Bad: Ambiguous response
def delete_file(path: str):
    os.remove(path)
    return "Done"

# ✅ Good: Clear status
def delete_file(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {
                "success": False,
                "error": "File not found",
                "path": path
            }

        os.remove(path)
        return {
            "success": True,
            "message": "File deleted successfully",
            "path": path
        }
    except PermissionError:
        return {
            "success": False,
            "error": "Permission denied",
            "path": path
        }
```

### 4. Not Handling Async Operations

```python
# ❌ Bad: Blocking on long operations
def generate_report(data: dict) -> dict:
    result = expensive_computation(data)  # Takes 30 seconds
    return result

# ✅ Good: Async with status checking
def generate_report(data: dict) -> dict:
    job_id = start_background_job(expensive_computation, data)
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Report generation started",
        "check_status_with": "get_job_status"
    }

def get_job_status(job_id: str) -> dict:
    job = get_job(job_id)
    return {
        "job_id": job_id,
        "status": job.status,  # "processing", "completed", "failed"
        "progress": job.progress,
        "result": job.result if job.status == "completed" else None
    }
```

### 5. Insufficient Context in Responses

```python
# ❌ Bad: Minimal context
def book_flight(destination: str, date: str) -> dict:
    return {"confirmation": "ABC123"}

# ✅ Good: Rich context
def book_flight(destination: str, date: str, passengers: int) -> dict:
    booking = create_booking(destination, date, passengers)
    return {
        "confirmation_code": "ABC123",
        "destination": destination,
        "departure_date": date,
        "passengers": passengers,
        "total_price": 850.00,
        "currency": "USD",
        "booking_time": "2025-01-15T10:30:00Z",
        "cancellation_policy": "Free cancellation until 24h before departure",
        "next_steps": "Check-in opens 24 hours before departure"
    }
```

## Advanced Patterns

### Multi-Step Reasoning with Tools

```python
# Complex task requiring multiple tools
user_query = "Find the cheapest laptop with at least 16GB RAM and notify me on Slack"

# Step 1: Search products
search_results = search_products(category="laptop", min_ram="16GB")

# Step 2: Sort by price
cheapest = min(search_results, key=lambda x: x['price'])

# Step 3: Send notification
send_slack_message(
    channel="@user",
    message=f"Found {cheapest['name']} for ${cheapest['price']}"
)
```

### Conditional Tool Selection

```python
def smart_search(query: str, search_type: str = "auto") -> dict:
    """Intelligently route to appropriate search tool."""

    if search_type == "auto":
        # Let model decide which search to use
        if "latest" in query or "recent" in query or "today" in query:
            return web_search(query)
        elif is_factual_question(query):
            return knowledge_base_search(query)
        else:
            return general_search(query)
    else:
        # Explicit search type
        search_functions = {
            "web": web_search,
            "knowledge_base": knowledge_base_search,
            "code": code_search
        }
        return search_functions[search_type](query)
```

### Tool Result Caching

```python
from functools import lru_cache
import hashlib

class ToolExecutor:
    def __init__(self):
        self.cache = {}

    def execute_with_cache(self, tool_name: str, parameters: dict) -> dict:
        """Execute tool with caching for identical calls."""

        # Create cache key
        cache_key = hashlib.md5(
            f"{tool_name}:{json.dumps(parameters, sort_keys=True)}".encode()
        ).hexdigest()

        # Check cache
        if cache_key in self.cache:
            return {
                **self.cache[cache_key],
                "cached": True
            }

        # Execute tool
        result = self.execute_tool(tool_name, parameters)

        # Cache result (if cacheable)
        if self.is_cacheable(tool_name):
            self.cache[cache_key] = result

        return {
            **result,
            "cached": False
        }
```

### Tool Composition

```python
# Compose simple tools into complex workflows
class CompositeTools:
    """Tools that combine multiple operations."""

    @staticmethod
    def research_and_summarize(topic: str) -> dict:
        """Research a topic and provide summary."""

        # Step 1: Search
        search_results = web_search(topic, limit=10)

        # Step 2: Extract content
        articles = [fetch_article(url) for url in search_results]

        # Step 3: Summarize
        combined_text = "\n\n".join(articles)
        summary = summarize_text(combined_text, max_length=500)

        # Step 4: Extract key points
        key_points = extract_key_points(combined_text)

        return {
            "topic": topic,
            "summary": summary,
            "key_points": key_points,
            "sources": search_results,
            "article_count": len(articles)
        }
```

### Fallback Mechanisms

```python
def robust_tool_execution(tool_name: str, parameters: dict) -> dict:
    """Execute tool with fallback strategies."""

    try:
        # Primary tool
        return execute_tool(tool_name, parameters)

    except APIRateLimitError:
        # Fallback 1: Use cached data
        cached = get_cached_result(tool_name, parameters)
        if cached and not is_stale(cached):
            return {
                **cached,
                "source": "cache",
                "warning": "Using cached data due to rate limit"
            }

        # Fallback 2: Use alternative API
        alt_tool = get_alternative_tool(tool_name)
        if alt_tool:
            return execute_tool(alt_tool, parameters)

        # Fallback 3: Return error with context
        return {
            "success": False,
            "error": "Rate limit exceeded",
            "suggestion": "Please try again in a few minutes"
        }

    except ToolTimeoutError:
        # Fallback: Start async job
        job_id = queue_tool_execution(tool_name, parameters)
        return {
            "success": False,
            "async_job_id": job_id,
            "message": "Operation queued due to timeout",
            "check_with": "get_job_status"
        }
```

## Security Considerations

### 1. Input Validation

```python
def execute_sql_query(query: str) -> dict:
    """Execute SQL query with security checks."""

    # Whitelist allowed operations
    allowed_operations = ['SELECT']
    operation = query.strip().split()[0].upper()

    if operation not in allowed_operations:
        return {
            "success": False,
            "error": f"Operation {operation} not allowed. Only SELECT queries permitted."
        }

    # Check for dangerous patterns
    dangerous_patterns = [
        r';\s*DROP',
        r';\s*DELETE',
        r';\s*UPDATE',
        r'--',
        r'/\*.*\*/'
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return {
                "success": False,
                "error": "Query contains potentially dangerous pattern"
            }

    # Execute with parameterization
    return execute_safe_query(query)
```

### 2. Authorization Checks

```python
def delete_document(document_id: str, user_id: str) -> dict:
    """Delete document with authorization."""

    # Check if document exists
    document = get_document(document_id)
    if not document:
        return {"success": False, "error": "Document not found"}

    # Check ownership
    if document.owner_id != user_id:
        # Check if user has permission
        if not has_permission(user_id, document_id, "delete"):
            return {
                "success": False,
                "error": "Unauthorized: You don't have permission to delete this document"
            }

    # Log the action
    audit_log(action="delete_document", user_id=user_id, document_id=document_id)

    # Perform deletion
    delete(document)
    return {"success": True, "message": "Document deleted"}
```

### 3. Rate Limiting

```python
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = timedelta(seconds=time_window)
        self.calls = defaultdict(list)

    def allow_call(self, user_id: str, tool_name: str) -> tuple[bool, str]:
        """Check if tool call is allowed under rate limits."""

        key = f"{user_id}:{tool_name}"
        now = datetime.now()

        # Remove old calls outside time window
        self.calls[key] = [
            call_time for call_time in self.calls[key]
            if now - call_time < self.time_window
        ]

        # Check limit
        if len(self.calls[key]) >= self.max_calls:
            oldest_call = min(self.calls[key])
            retry_after = (oldest_call + self.time_window - now).total_seconds()
            return False, f"Rate limit exceeded. Retry after {retry_after:.0f} seconds"

        # Record this call
        self.calls[key].append(now)
        return True, ""

# Usage
rate_limiter = RateLimiter(max_calls=10, time_window=60)

def execute_tool_with_rate_limit(tool_name: str, user_id: str, parameters: dict):
    allowed, message = rate_limiter.allow_call(user_id, tool_name)
    if not allowed:
        return {"success": False, "error": message}

    return execute_tool(tool_name, parameters)
```

### 4. Sandboxing

```python
import subprocess
import tempfile
import os

def execute_code_safely(code: str, language: str = "python") -> dict:
    """Execute user code in sandboxed environment."""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Execute with restrictions
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=5,  # Timeout after 5 seconds
            env={'PATH': '/usr/bin'},  # Restricted environment
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Code execution timed out (5s limit)"
        }

    finally:
        # Cleanup
        os.unlink(temp_file)
```

## Performance Optimization

### 1. Parallel Tool Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def execute_tools_parallel(tool_calls: list) -> list:
    """Execute multiple independent tool calls in parallel."""

    async def execute_single(tool_call):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                execute_tool,
                tool_call['name'],
                tool_call['parameters']
            )
        return result

    # Execute all in parallel
    results = await asyncio.gather(*[execute_single(tc) for tc in tool_calls])
    return results

# Usage
tool_calls = [
    {"name": "get_weather", "parameters": {"location": "NYC"}},
    {"name": "get_weather", "parameters": {"location": "LA"}},
    {"name": "get_weather", "parameters": {"location": "Chicago"}}
]

results = asyncio.run(execute_tools_parallel(tool_calls))
```

### 2. Lazy Loading

```python
class LazyToolRegistry:
    """Load tool implementations only when needed."""

    def __init__(self):
        self._tools = {}
        self._tool_modules = {
            'weather': 'tools.weather',
            'database': 'tools.database',
            'email': 'tools.email'
        }

    def get_tool(self, tool_name: str):
        """Lazy load tool implementation."""
        if tool_name not in self._tools:
            module_path = self._tool_modules.get(tool_name)
            if not module_path:
                raise ValueError(f"Unknown tool: {tool_name}")

            # Import module only when needed
            module = __import__(module_path, fromlist=[''])
            self._tools[tool_name] = module

        return self._tools[tool_name]
```

### 3. Response Streaming

```python
def stream_large_result(query: str):
    """Stream results instead of waiting for complete response."""

    yield {"status": "starting", "message": "Searching database..."}

    results = []
    for batch in search_database_batches(query):
        results.extend(batch)
        yield {
            "status": "progress",
            "results_so_far": len(results),
            "latest_batch": batch
        }

    yield {
        "status": "complete",
        "total_results": len(results),
        "results": results
    }
```

## Testing Tool Use

### Unit Testing Tools

```python
import pytest

def test_calculator_tool():
    """Test calculator tool with various inputs."""

    # Test basic calculation
    result = calculate("2 + 2")
    assert result == {"result": 4, "success": True}

    # Test division
    result = calculate("10 / 2")
    assert result == {"result": 5.0, "success": True}

    # Test division by zero
    result = calculate("10 / 0")
    assert result["success"] == False
    assert "division by zero" in result["error"].lower()

    # Test invalid expression
    result = calculate("invalid")
    assert result["success"] == False
```

### Integration Testing

```python
def test_multi_tool_workflow():
    """Test workflow using multiple tools."""

    # Mock LLM responses
    with mock.patch('llm.generate') as mock_llm:
        # First call: LLM decides to use weather tool
        mock_llm.return_value = {
            "tool_call": {
                "name": "get_weather",
                "parameters": {"location": "NYC"}
            }
        }

        # Execute workflow
        result = execute_agent_workflow("What should I wear in NYC today?")

        # Verify tool was called
        assert "weather" in result.tools_used
        assert "NYC" in result.final_response


def test_tool_error_handling():
    """Test that errors are handled gracefully."""

    with mock.patch('tools.api.call') as mock_api:
        # Simulate API failure
        mock_api.side_effect = APIError("Service unavailable")

        result = execute_tool("get_stock_price", {"ticker": "AAPL"})

        assert result["success"] == False
        assert "error" in result
```

### Mock Tools for Testing

```python
class MockToolExecutor:
    """Mock tool executor for testing."""

    def __init__(self):
        self.calls = []
        self.mock_responses = {}

    def set_mock_response(self, tool_name: str, response: dict):
        """Set predetermined response for a tool."""
        self.mock_responses[tool_name] = response

    def execute(self, tool_name: str, parameters: dict) -> dict:
        """Execute tool (mocked)."""
        self.calls.append({
            "tool": tool_name,
            "parameters": parameters,
            "timestamp": datetime.now()
        })

        return self.mock_responses.get(tool_name, {"success": True})

    def verify_called(self, tool_name: str, times: int = None):
        """Verify tool was called."""
        calls = [c for c in self.calls if c["tool"] == tool_name]
        if times is not None:
            assert len(calls) == times, f"Expected {times} calls, got {len(calls)}"
        return len(calls) > 0
```

## Real-World Examples

### Example 1: Customer Support Agent

```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": "Search help articles and documentation",
        "parameters": {
            "query": {"type": "string"},
            "category": {"type": "string", "enum": ["billing", "technical", "account"]}
        }
    },
    {
        "name": "get_order_status",
        "description": "Get status of customer order",
        "parameters": {
            "order_id": {"type": "string"}
        }
    },
    {
        "name": "create_support_ticket",
        "description": "Create ticket for complex issues requiring human support",
        "parameters": {
            "subject": {"type": "string"},
            "description": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]}
        }
    }
]

# User: "Where's my order #12345?"
# Agent uses: get_order_status(order_id="12345")
# Agent responds with order tracking information
```

### Example 2: Data Analysis Assistant

```python
tools = [
    {
        "name": "query_database",
        "description": "Execute SQL query on analytics database",
        "parameters": {
            "query": {"type": "string"},
            "database": {"type": "string", "enum": ["sales", "users", "products"]}
        }
    },
    {
        "name": "create_visualization",
        "description": "Create chart from data",
        "parameters": {
            "chart_type": {"type": "string", "enum": ["line", "bar", "pie"]},
            "data": {"type": "array"},
            "title": {"type": "string"}
        }
    },
    {
        "name": "calculate_statistics",
        "description": "Calculate statistical measures",
        "parameters": {
            "data": {"type": "array"},
            "metrics": {"type": "array", "items": {"enum": ["mean", "median", "std_dev"]}}
        }
    }
]

# User: "Show me sales trends for the last quarter"
# 1. Agent uses: query_database to get sales data
# 2. Agent uses: calculate_statistics to analyze
# 3. Agent uses: create_visualization to create chart
```

### Example 3: Code Review Assistant

```python
tools = [
    {
        "name": "get_file_content",
        "description": "Read file from repository",
        "parameters": {
            "file_path": {"type": "string"}
        }
    },
    {
        "name": "run_linter",
        "description": "Run code linter",
        "parameters": {
            "file_path": {"type": "string"},
            "linter": {"type": "string", "enum": ["eslint", "pylint", "rustfmt"]}
        }
    },
    {
        "name": "run_tests",
        "description": "Execute test suite",
        "parameters": {
            "test_path": {"type": "string"}
        }
    },
    {
        "name": "create_review_comment",
        "description": "Add review comment to PR",
        "parameters": {
            "file": {"type": "string"},
            "line": {"type": "number"},
            "comment": {"type": "string"}
        }
    }
]

# Workflow: Review pull request
# 1. get_file_content for changed files
# 2. run_linter to check code style
# 3. run_tests to verify functionality
# 4. create_review_comment for issues found
```

## Resources

### Documentation
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Guide](https://docs.anthropic.com/claude/docs/tool-use)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)

### Best Practices
- [Prompt Engineering Guide - Tool Use](https://www.promptingguide.ai/techniques/tool-use)
- [Building Reliable AI Agents](https://www.anthropic.com/research/building-effective-agents)

### Examples & Templates
- [OpenAI Cookbook - Function Calling](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)
- [LangChain Tool Examples](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/tools)

## Conclusion

Tool use transforms LLMs from text generators into capable agents that can interact with systems, access real-time information, and perform actions. Success requires:

1. **Clear tool definitions** with comprehensive descriptions
2. **Robust error handling** for reliable operation
3. **Security measures** to prevent misuse
4. **Performance optimization** for responsive experiences
5. **Thorough testing** to ensure reliability

The key is finding the right balance between giving the model enough tools to be useful while maintaining security and performance. Start with a small set of well-designed tools and expand based on real usage patterns.

Remember: Good tool use is about **empowering the model** to help users accomplish tasks, not just adding features for the sake of it.
