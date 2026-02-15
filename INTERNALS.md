# Angel Recall: Internal Architecture

This document provides a deep dive into the internal mechanics of Angel Recall, a Python implementation of the **MemOS** (Memory Operating System) architecture.

## Overview

Angel Recall treats memory as a dynamic, managed system resource rather than a static database. It orchestrates several specialized components to handle the entire lifecycle of a "memory," from its initial generation to its eventual archiving or expiry.

## Core Components

### 1. MemCube: The Atomic Unit
The `MemCube` is the universal memory unit. Unlike a simple text string, it encapsulates:
- **Payload**: The actual data (text, KV pairs for activations, or adapter references for parameters).
- **Metadata**: Provenance (origin), versioning, timestamps, and access counts.
- **State**: Where the memory sits in its lifecycle (e.g., `GENERATED`, `ACTIVATED`, `ARCHIVED`).
- **Semantic Type**: Categorization (e.g., `FACT`, `PREFERENCE`, `PROCEDURE`, `RULE`).

### 2. MemVault: Hybrid Storage
The `MemVault` manages two distinct storage backends:
- **ChromaDB**: Handles semantic (vector) search for plaintext memories.
- **NetworkX**: A multi-directed graph used to map complex relationships and dependencies between different `MemCubes`.

### 3. MemReader: Intent Extraction
The `MemReader` translates natural language prompts into structured `MemoryOperation` objects.
- **LLM-Based Parsing**: Uses LiteLLM to attempt a high-fidelity extraction of the user's intent.
- **Rule-Based Fallback**: A robust keyword-based parser that detects storage (e.g., "remember that..."), deletion ("forget..."), and retrieval patterns.

### 4. MemScheduler: Proactive Retrieval
The `MemScheduler` implements the "Next-Scene Prediction" concept from the MemOS paper. Instead of waiting for a query, it analyzes the task intent to pre-load relevant context:
- If a user is asking a technical question, it proactively selects `PARAMETER` memories.
- If it's a casual chat, it prioritizes `ACTIVATION` memories (simulated KV caches).

### 5. MemGovernance: Privacy & Access Control
Every memory operation passes through `MemGovernance`. It enforces:
- **Access Scopes**: `PRIVATE`, `SHARED`, or `PUBLIC`.
- **Role-Based Access**: Verifies if a user has the right to read, write, or delete a specific `MemCube`.
- **Auditing**: Maintains an internal log of all memory operations for transparency.

### 6. MemLifecycle: Evolution of Memory
Memories are not static. `MemLifecycle` manages state transitions based on usage and time:
- **Auto-Archiving**: Memories that haven't been accessed for a certain threshold (e.g., 30 days) are moved to an `ARCHIVED` state.
- **TTL Enforcement**: Deletes memories once their "Time To Live" has expired.

## Memory Policies in Depth

### Lifecycle Transitions
The `MemLifecycle` component manages the state machine of each `MemCube`. A typical progression is:
1. **Generated**: A fresh memory captured from dialogue.
2. **Activated**: Memory currently pre-loaded into the agent's working context.
3. **Merged**: Similar memories fused together to reduce redundancy.
4. **Archived**: Older memories moved to cold storage (removed from vector index but kept in the graph).
5. **Expired**: Purged from all storage backends.

### Governance & Security
Access is verified at the API layer. The `MemGovernance.check_access` method evaluates:
- `cube.owner`: The primary user who created the memory.
- `cube.access_scope`: Defines if others can see it (Private, Shared, Public).
- `user_roles`: The global permission level of the requesting user.

### TTL Enforcement
TTL is handled reactively. Every time `MemOS.process` is called, it triggers `governance.enforce_ttl(vault)`. This scans for cubes whose `timestamp + ttl` is in the past and invokes a permanent deletion.

## Data Flow: `MemOS.process()`

When a prompt enters the system:
1. **Parsing**: `MemReader` determines if the user wants to store, retrieve, or delete.
2. **Scheduling**: `MemScheduler` looks at the intent and pre-fetches potentially relevant memories from the vault.
3. **Execution**:
    - **Store**: A new `MemCube` is created, audited by `MemGovernance`, and persisted in `MemVault`.
    - **Retrieve**: `MemOperator` performs a hybrid search (semantic + graph traversal) and merges the results with the scheduled context.
    - **Delete**: The target memory is identified and removed from both the vector store and the graph.
4. **Maintenance**: `MemGovernance` enforces TTLs, and `MemLifecycle` can trigger state updates.

## Agent Integration

The `create_memory_agent` function wraps `MemOS` into a **LangGraph** workflow:
- **Memory Node**: Processes the input to extract/retrieve context.
- **LLM Node**: Receives the original prompt augmented with the "Memory Context" retrieved by `MemOS`, ensuring the assistant has long-term awareness of user preferences and past interactions.
