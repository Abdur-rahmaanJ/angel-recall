# Angel Recall: The Memory Operating System

Angel Recall is a comprehensive Memory Operating System (MemOS) for AI agents. It provides a governed, multi-user memory layer that treats information as a dynamic resource rather than a static database.

## Core Concepts

### 1. MemCube (The Universal Unit)
The basic atom of memory. Every piece of information—whether a fact, a preference, or a system parameter—is wrapped in a `MemCube`. It carries metadata for ownership, access scope, and semantic type.

### 2. SessionLane (Sequential Execution)
Ensures that tasks for a specific user session run sequentially. This prevents race conditions and ensures memory consistency during concurrent operations.

### 3. MemVault (Hybrid Storage)
A dual-engine storage system:
- **Vector Store (ChromaDB)**: For semantic similarity search.
- **Knowledge Graph (NetworkX)**: For structured relationships between memory units.

### 4. MemReader (Intent Parsing)
Interprets user prompts to decide if it should `store`, `retrieve`, or `delete` information. It includes a "Smart Rule Parser" for fallback when LLMs are unavailable.

### 5. MemGovernance (Memory Policies)
Manages privacy (`AccessScope`), expiration (`TTL`), and access control across different users.

## System Flow

1. **Ingress**: A user string enters through `MemOS.process(prompt, user)`.
2. **Lane Assignment**: The system identifies the `SessionLane` for the given user.
3. **Parsing**: `MemReader` analyzes the intent and extracts the content.
4. **Action**: The `MemoryAPI` executes the operation, interacting with the `MemVault`.
5. **Enforcement**: `MemGovernance` ensures the user has access and purges expired memories.

---

## Quick Start

```python
from angel_recall import MemOS

# Initialize with a local storage path
memos = MemOS(persist_directory="./my_vault")

# Store a fact
memos.process("Remember that my secret project is 'Project Phoenix'", user="alice")

# Retrieve it
result = memos.process("What is my secret project?", user="alice")
print(result["response"])
```

## Public API Reference

### `MemOS`
The main orchestrator.
- `process(prompt: str, user: str)`: High-level NLP interface.

### `MemoryAPI`
Low-level control over memory operations.
- `create(cube: MemCube)`: Direct storage.
- `query(text: str, user: str)`: Direct retrieval.
- `update_access_scope(cube_id, scope, user)`: Manage privacy.
- `delete(cube_id, user)`: Explicit removal.

### `get_memory_tools`
Generates LangChain tools (`store_memory`, `search_memory`, `set_memory_access`) for integration with LangGraph or other agent frameworks.
