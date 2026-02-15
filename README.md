# Angel Recall

Angel Recall is a Python implementation of the **MemOS** (Memory Operating System) architecture, designed to give AI agents a long-term, evolvable memory. It uses a **Lane-Based Command Queue** instead of concurrent async to ensure predictable, race-free execution.

## The Lane-Based Advantage

Unlike standard async libraries, Angel Recall processes tasks for each session (user) in a dedicated **Lane**.
- **Predictable Execution**: Tasks within a session run sequentially, preventing race conditions.
- **Clean Logs**: No interleaved garbage in your application logs.
- **Simplified Mental Model**: You don't need to worry about `asyncio` locks or complex parallel debugging.

## Quick Start

### Installation

```bash
pip install angel-recall
```

### Basic Usage

```python
from angel_recall import MemOS

# Initialize the Memory OS (Synchronous)
memos = MemOS(persist_directory="./my_vault")

# Store a new memory
memos.process("Remember that I prefer technical deep-dives.")

# Retrieve relevant memories
response = memos.process("How should you format the report for me?")
print(response["response"])
```

## Configuration

Angel Recall is designed to be flexible. You can configure the model and persistence directory during initialization:

```python
memos = MemOS(
    persist_directory="./my_vault",
    model="ollama/gemma3n:e4b" # Supports any LiteLLM-compatible model
)
```

## Memory Policies

Angel Recall implements several policies to ensure your agent's memory stays relevant and secure:

- **Lifecycle Management**: Memories automatically transition through states (`GENERATED` -> `ACTIVATED` -> `ARCHIVED`). Cold memories are eventually moved to long-term storage to keep the retrieval context clean.
- **TTL (Time-to-Live)**: You can set an expiration for any memory. Once the TTL is reached, the memory is automatically purged during the next process cycle.
- **Access Governance**: Supports `PRIVATE`, `SHARED`, and `PUBLIC` scopes. The system verifies ownership and permissions before any read or write operation.
- **Sensitivity Masking**: Built-in support for PII redaction and sensitive tag handling to prevent leakage of private data into LLM prompts.

## Access Governance Example

You can control the visibility of memories programmatically or via tools. By default, all memories are `private` to the owner.

```python
from angel_recall import MemOS, AccessScope

memos = MemOS()

# Programmatically update access
# Only the owner ('alice') can change the scope
success = memos.api.update_access_scope(
    cube_id="some-uuid",
    scope=AccessScope.SHARED,
    user="alice"
)

if success:
    print("Memory is now shared with other users in the system.")
```

When using `get_memory_tools`, your agent can also manage its own privacy:

```python
# The agent can call 'set_memory_access' tool
# Example: set_memory_access(memory_id="...", scope="public")
```

## Advanced Usage: Agent Tools

If you are building your own LangGraph agents and want to give them explicit control over memory, you can use `get_memory_tools`. This is perfect for complex agents that need to decide *when* to commit something to long-term storage.

```python
from langgraph.prebuilt import ToolNode
from angel_recall import MemOS, get_memory_tools

memos = MemOS()
tools = get_memory_tools(memos, user="alice")

# Create a tool node for your graph
tool_node = ToolNode(tools)

# Bind tools to your model
# model_with_tools = model.bind_tools(tools)
```

### Tool-based LangGraph Example

```python
from langgraph.graph import StateGraph, END
from angel_recall import MemOS, get_memory_tools

memos = MemOS()
tools = get_memory_tools(memos, user="alice")

def call_model(state):
    # Your logic here to call an LLM bound with 'tools'
    pass

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
# Add conditional edges for tool usage...
```

## Testing

We use `pytest` for testing. To run the tests using the `qwen2.5:0.5b` model:

```bash
pytest tests/test_core.py
```

Ensure Ollama is running and you have pulled the model: `ollama pull qwen2.5:0.5b`.

## Core Components

- **MemVault**: The storage engine combining ChromaDB for semantic search and NetworkX for relationship mapping.
- **MemReader**: An intelligent parser that understands intent, whether you're asking to save, retrieve, or delete information.
- **MemGovernance**: Ensures privacy and access control, managing who can read or modify specific memory cubes.
- **MemScheduler**: Dynamically selects the best memory fragments based on the current task's context.

## Contributing

We welcome contributions that improve the efficiency of the memory scheduler or add support for new vector backends. Please see our contributing guidelines for more details.

## License

Apache License 2.0. See `LICENSE` for details.
