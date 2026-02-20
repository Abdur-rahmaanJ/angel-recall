# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-02-20
### Added
- **Intelligent Memory Categorization**: New `CORRECTION` and `INSIGHT` semantic types.
    - **Corrections**: Assigned high priority (5) and proactively archive older conflicting facts.
    - **Insights**: Assigned priority (3) for capturing complex patterns or deep understandings.
- **Weight Decay & Forgetting**: Implementation of a dynamic "forgetting" mechanism.
    - Memories lose weight over time unless "touched" or reinforced.
    - Memories falling below a threshold are automatically archived to keep the context clean.
- **Local Embedding Optimization**: Added optional `local_embedding=True` flag to use `sentence-transformers` (all-MiniLM-L6-v2) for zero-latency, private retrieval without external API calls.
- **Improved Retrieval**: The context snippets now prioritize newest and higher-priority memories.

## [0.3.0] - 2026-02-10
### Added
- **Memory Distillation**: Periodically reviews dialogue logs to extract new `FACT`s or `PREFERENCE`s.
- **Persistent Network**: The relationship graph (NetworkX) is now persisted to disk (`graph.json`).
- **Namespace Listing**: Added ability to list all memories within a specific namespace.

### Fixed
- Memory update bugs and serialization issues.
- Serialization of datetime objects in the `MemCube` dataclass.

## [0.2.0] - 2026-01-25
### Added
- **Initial Test Suite**: Robust tests for core `MemOS` operations using `pytest`.
- **MemGovernance**: Initial implementation of Access Governance with `PRIVATE`, `SHARED`, and `PUBLIC` scopes.

## [0.1.0] - 2026-01-05
### Added
- **Initial Commit**: Stable implementation of the MemOS (Memory Operating System) architecture.
- **MemVault**: Storage engine with ChromaDB and NetworkX support.
- **MemReader**: Rule-based and LLM-based parsing of memory operations.
