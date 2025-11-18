# AI Agent Frameworks

## Overview

AI Agent Frameworks are software platforms that enable the creation of autonomous or semi-autonomous AI agents capable of planning, reasoning, using tools, and executing complex multi-step tasks. These frameworks build on LLMs by adding memory, tool use, planning capabilities, and orchestration logic.

## Core Concepts

### What is an AI Agent?

An AI agent is an autonomous system that:
- **Perceives**: Takes input from environment/user
- **Reasons**: Plans next actions using LLM
- **Acts**: Executes tools/functions
- **Learns**: Improves from feedback/memory

```
User Query -> Agent -> [Plan] -> [Execute Tools] -> [Observe] -> [Reason] -> Response
                ↑                                                              ↓
                └──────────────────── Memory ────────────────────────────────┘
```

### Key Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **LLM Core** | Reasoning engine | GPT-4, Claude, Llama |
| **Memory** | Short & long-term context | Vector DB, conversation history |
| **Tools** | Capabilities/functions | Web search, calculator, API calls |
| **Planner** | Task decomposition | ReAct, Chain-of-Thought |
| **Executor** | Tool orchestration | Function calling, API integration |

## Popular Frameworks

### LangChain

**Purpose**: General-purpose agent framework with extensive integrations

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for searching the web"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for math calculations"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=OpenAI(temperature=0),
    agent="zero-shot-react-description"
)

agent.run("What is 25% of the GDP of France?")
```

**Strengths**:
- 500+ integrations (databases, APIs, tools)
- Multiple agent types (ReAct, Plan-and-Execute)
- Rich ecosystem (LangSmith for debugging)
- Strong community support

**Weaknesses**:
- Complex abstractions
- Steep learning curve
- Can be overkill for simple tasks

### LlamaIndex

**Purpose**: Specialized for building RAG (Retrieval-Augmented Generation) agents

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.agent import OpenAIAgent
from llama_index.tools import QueryEngineTool

# Load documents
documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine tool
query_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(),
    name="knowledge_base",
    description="Useful for answering questions about company docs"
)

# Create agent
agent = OpenAIAgent.from_tools([query_tool])
agent.chat("What's our vacation policy?")
```

**Strengths**:
- Best for document Q&A and knowledge retrieval
- Excellent indexing and chunking strategies
- Multi-document reasoning
- Production-ready RAG patterns

**Weaknesses**:
- More focused on retrieval than general agents
- Fewer non-RAG integrations

### AutoGPT / BabyAGI

**Purpose**: Autonomous agents that create and execute task lists

```python
# AutoGPT pseudocode
while not task_complete:
    # 1. Think about what to do next
    thought = llm.think(context, goal)

    # 2. Decide on action
    action = llm.decide_action(thought)

    # 3. Execute action
    result = execute(action)

    # 4. Update memory
    memory.add(thought, action, result)

    # 5. Check if goal achieved
    if llm.is_goal_achieved(goal, memory):
        break
```

**Strengths**:
- Fully autonomous (minimal human intervention)
- Creative problem solving
- Task decomposition and planning

**Weaknesses**:
- Can go off track (goal drift)
- Expensive (many LLM calls)
- Unpredictable outcomes

### CrewAI

**Purpose**: Multi-agent collaboration framework

```python
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information",
    tools=[search_tool, scrape_tool]
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging content",
    tools=[grammar_tool]
)

# Define tasks
research_task = Task(
    description="Research AI agent frameworks",
    agent=researcher
)

write_task = Task(
    description="Write a blog post about findings",
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task]
)

result = crew.kickoff()
```

**Strengths**:
- Role-based agent collaboration
- Sequential and parallel task execution
- Clear delegation patterns
- Great for complex workflows

**Weaknesses**:
- Relatively new framework
- Limited integrations vs LangChain
- Can be complex to debug

### Semantic Kernel (Microsoft)

**Purpose**: Enterprise-grade agent framework with strong .NET/C# support

```csharp
// C# example
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("gpt-4", apiKey)
    .Build();

// Add plugins (tools)
kernel.ImportPluginFromType<MathPlugin>();
kernel.ImportPluginFromType<SearchPlugin>();

// Enable automatic function calling
var result = await kernel.InvokePromptAsync(
    "What's the square root of the GDP of USA in billions?"
);
```

**Strengths**:
- First-class C# and Python support
- Enterprise features (authentication, logging)
- Microsoft ecosystem integration
- Strong type safety

**Weaknesses**:
- Smaller community than LangChain
- More verbose than Python alternatives

### Haystack

**Purpose**: Open-source framework for NLP pipelines and agents

```python
from haystack.agents import Agent
from haystack.tools import WebSearch, Calculator

agent = Agent(
    llm=llm,
    tools=[WebSearch(), Calculator()],
    max_iterations=10
)

agent.run("How many days until the next solar eclipse?")
```

**Strengths**:
- Production-ready pipelines
- Strong NLP capabilities
- Good documentation
- Active development

**Weaknesses**:
- Less mature agent features
- Smaller tool ecosystem

## Agent Architectures

### 1. ReAct (Reason + Act)

Alternates between reasoning and acting:

```
Thought: I need to find the current weather
Action: search("weather in San Francisco")
Observation: Temperature is 65°F, sunny
Thought: Now I have the answer
Answer: It's 65°F and sunny in San Francisco
```

**Best for**: General-purpose tasks with tool use

### 2. Plan-and-Execute

Creates full plan upfront, then executes:

```
Plan:
1. Search for France GDP
2. Calculate 25% of that number
3. Return result

Execute:
Step 1: [search tool] -> $2.8 trillion
Step 2: [calculator] -> $700 billion
Step 3: [return] -> "25% of France's GDP is $700 billion"
```

**Best for**: Complex multi-step tasks

### 3. Reflexion

Agent that reflects on failures and improves:

```
Attempt 1: [tries solution] -> FAILED
Reflection: "I failed because I didn't check edge cases"
Attempt 2: [improved solution] -> SUCCESS
```

**Best for**: Tasks requiring iteration and learning

### 4. Tree-of-Thoughts

Explores multiple reasoning paths:

```
                    Problem
                   /   |   \
               Path1 Path2 Path3
              /  \    |      |
            A    B    C      D
```

Evaluates each path and picks the best.

**Best for**: Complex reasoning, puzzles, creative tasks

## Memory Systems

### Short-Term Memory

Conversation history within single session:

```python
# Typical implementation
memory = ConversationBufferMemory()
memory.save_context(
    {"input": "Hi, I'm Alice"},
    {"output": "Hello Alice!"}
)
memory.save_context(
    {"input": "What's my name?"},
    {"output": "Your name is Alice"}
)
```

### Long-Term Memory

Persistent storage across sessions:

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma

# Store memories in vector database
memory = VectorStoreRetrieverMemory(
    retriever=Chroma.from_texts(
        texts=past_conversations,
        embedding=embeddings
    ).as_retriever(k=5)
)
```

### Episodic Memory

Stores specific events/interactions:

```
Event 1: "User prefers Python over JavaScript"
Event 2: "User is working on ML project"
Event 3: "User has deadline on Friday"
```

## Tool Integration

### Function Calling

Modern approach using LLM's native function calling:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        }
    }
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
```

### Tool Types

| Type | Purpose | Examples |
|------|---------|----------|
| **Search** | Information retrieval | Google, Bing, Wikipedia |
| **Computation** | Math/logic | Calculator, Code interpreter |
| **Database** | Data queries | SQL, NoSQL, APIs |
| **Communication** | External interaction | Email, Slack, SMS |
| **File Ops** | File management | Read, write, edit files |
| **Web** | Web interaction | Scraping, browser automation |

## Production Considerations

### 1. Cost Management

```python
# Track token usage
class CostTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0

    def track(self, prompt_tokens, completion_tokens):
        self.total_tokens += prompt_tokens + completion_tokens
        # GPT-4 pricing
        cost = (prompt_tokens * 0.00003 +
                completion_tokens * 0.00006)
        self.total_cost += cost
        return cost
```

### 2. Safety & Guardrails

```python
# Prevent dangerous actions
FORBIDDEN_TOOLS = ["delete_database", "send_money", "system_shutdown"]

def safe_execute(tool_name, params):
    if tool_name in FORBIDDEN_TOOLS:
        return "Error: Tool not allowed"

    # Validate parameters
    if not validate(params):
        return "Error: Invalid parameters"

    return execute_tool(tool_name, params)
```

### 3. Error Handling

```python
# Retry logic
def execute_with_retry(agent, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return agent.run(query)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff
            time.sleep(2 ** attempt)
```

### 4. Monitoring & Logging

```python
# Log all agent actions
logger.info({
    "timestamp": datetime.now(),
    "query": user_query,
    "agent_thoughts": thoughts,
    "tools_used": [tool.name for tool in tools_used],
    "tokens_used": token_count,
    "latency_ms": latency,
    "success": success
})
```

## Evaluation Metrics

| Metric | What It Measures | How to Calculate |
|--------|-----------------|------------------|
| **Task Success Rate** | % of tasks completed correctly | Successful tasks / Total tasks |
| **Tool Selection Accuracy** | Did agent pick right tools? | Correct tools / Total tool calls |
| **Efficiency** | Unnecessary steps taken | Actual steps / Minimum steps |
| **Cost per Task** | $ spent per query | Total API costs / Tasks |
| **Latency** | Time to complete | End time - Start time |

## Common Pitfalls

### 1. Infinite Loops

```python
# BAD: No stop condition
while True:
    action = agent.think()
    execute(action)

# GOOD: Max iterations
max_iter = 10
for i in range(max_iter):
    if goal_achieved():
        break
    action = agent.think()
    execute(action)
```

### 2. Tool Overload

```python
# BAD: Too many tools confuse the agent
agent = Agent(tools=[tool1, tool2, ..., tool50])

# GOOD: Selective tool loading
relevant_tools = select_tools_for_task(task_type)
agent = Agent(tools=relevant_tools)
```

### 3. Poor Prompting

```python
# BAD: Vague instructions
agent.run("Do something with the data")

# GOOD: Clear, specific goal
agent.run("""
Analyze sales_data.csv and:
1. Calculate total revenue
2. Find top 5 products
3. Create a summary report
""")
```

## Use Cases & Examples

### Customer Support Agent

```python
support_agent = Agent(
    name="Support Agent",
    role="Customer support specialist",
    tools=[
        SearchKnowledgeBase(),
        CheckOrderStatus(),
        CreateTicket(),
        SendEmail()
    ],
    instructions="""
    1. Greet customer warmly
    2. Understand their issue
    3. Search knowledge base first
    4. Check order status if relevant
    5. Create ticket if unable to resolve
    6. Always confirm resolution
    """
)
```

### Research Assistant

```python
research_agent = Agent(
    name="Research Assistant",
    tools=[
        WebSearch(),
        ArxivSearch(),
        ReadPDF(),
        SummarizeText(),
        TakeNotes()
    ],
    workflow="plan-and-execute",
    max_iterations=20
)

result = research_agent.run(
    "Research latest developments in quantum computing and write a summary"
)
```

### Data Analysis Agent

```python
data_agent = Agent(
    name="Data Analyst",
    tools=[
        PythonREPL(),  # Execute Python code
        QueryDatabase(),
        CreateVisualization(),
        ExportResults()
    ],
    memory=ConversationMemory()
)

data_agent.chat("Analyze user_behavior.csv and find patterns")
```

## Best Practices

### 1. Start Simple

```python
# Start with single-agent, few tools
agent = Agent(
    llm=llm,
    tools=[search, calculator]  # Just 2 tools
)

# Scale up as needed
```

### 2. Clear Tool Descriptions

```python
# BAD
Tool(name="tool1", description="Does stuff")

# GOOD
Tool(
    name="search_knowledge_base",
    description="Searches company documentation. Use when user asks about policies, procedures, or internal information. Input: search query as string. Output: relevant document excerpts."
)
```

### 3. Test Incrementally

```python
# Test each component separately
assert search_tool("test query") is not None
assert calculator_tool("2+2") == 4

# Then test agent
response = agent.run("What is 2+2?")
assert "4" in response
```

### 4. Human-in-the-Loop

```python
# For critical actions, ask for confirmation
def execute_tool(tool, params):
    if tool.requires_confirmation:
        print(f"About to: {tool.name}({params})")
        if input("Confirm? (y/n): ") != "y":
            return "Action cancelled by user"
    return tool.run(params)
```

## Framework Comparison

| Framework | Best For | Difficulty | Ecosystem | Language |
|-----------|----------|------------|-----------|----------|
| **LangChain** | General-purpose agents | Medium | Excellent | Python/JS |
| **LlamaIndex** | RAG & document Q&A | Easy | Good | Python/TS |
| **CrewAI** | Multi-agent collaboration | Medium | Growing | Python |
| **AutoGPT** | Autonomous experiments | Hard | Medium | Python |
| **Semantic Kernel** | Enterprise/.NET | Medium | Good | C#/Python |
| **Haystack** | NLP pipelines | Medium | Good | Python |

## ELI10

Imagine you have a really smart robot assistant. But on its own, it can only talk - it can't actually DO things like search the web or calculate numbers.

An AI Agent Framework is like giving your robot a toolbelt with different tools:
- 🔍 A magnifying glass to search for information
- 🧮 A calculator for math
- 📧 A phone to send messages
- 📝 A notepad to remember things

The framework teaches your robot:
1. **When** to use each tool
2. **How** to use them together
3. **What** to do when things go wrong
4. **How** to remember what it did before

So instead of just chatting, your AI assistant can actually complete complex tasks like "find the weather, calculate if I need a jacket, and text me the answer"!

## Future Trends

- **Multi-modal Agents**: Using vision, audio, and text together
- **Swarm Intelligence**: Hundreds of tiny specialized agents collaborating
- **Continuous Learning**: Agents that improve from every interaction
- **Code Agents**: AI that writes and deploys software autonomously
- **Physical World Integration**: Agents controlling robots and IoT devices
- **Standardization**: Common protocols for agent communication (OpenAI Agent Protocol)

## Further Resources

- [LangChain Documentation](https://python.langchain.com/docs/modules/agents/)
- [LlamaIndex Agents Guide](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Agent Benchmarks (AgentBench)](https://github.com/THUDM/AgentBench)
- [LangSmith for Agent Debugging](https://smith.langchain.com/)
