#!/usr/bin/env python3
"""
Angel Recall: Agentic memory based on MemOS.
- Unified memory structure (MemCube)
- Hybrid retrieval (Semantic + Graph)
- Lane-Based Command Queue (Sequential execution per session)
- Tool support for LangGraph
"""

import json
import uuid
import threading
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from collections import defaultdict

# ---------------------------
# External dependencies
# ---------------------------
import litellm
import chromadb
import networkx as nx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field


litellm.set_verbose = False
litellm.suppress_debug_info = True

__all__ = [
    "MemOS",
    "MemCube",
    "MemoryType",
    "MemoryState",
    "SemanticType",
    "AccessScope",
    "create_memory_agent",
    "get_memory_tools",
    "create_plaintext",
    "create_activation",
    "create_parameter",
    "SessionLane",
]


# ---------------------------
# Enums & Constants
# ---------------------------
class MemoryType(Enum):
    PLAINTEXT = "plaintext"
    ACTIVATION = "activation"
    PARAMETER = "parameter"


class MemoryState(Enum):
    GENERATED = "generated"
    ACTIVATED = "activated"
    MERGED = "merged"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    FROZEN = "frozen"


class AccessScope(Enum):
    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"


class SemanticType(Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    DIALOGUE = "dialogue"
    PROCEDURE = "procedure"
    RULE = "rule"
    CORRECTION = "correction"
    INSIGHT = "insight"


# ---------------------------
# Pydantic model for JSON output
# ---------------------------
class MemoryOperation(BaseModel):
    operation: str = Field(
        description="retrieve, store, update, delete, query, summarize"
    )
    memory_type: Optional[str] = Field(None)
    semantic_type: str = Field(
        description="fact, preference, task, dialogue, procedure, rule, correction, insight"
    )
    time_scope: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    context_window: Optional[int] = None
    content_summary: str = Field(description="what to remember/retrieve")
    task_intent: str = ""
    memory_scope: str = "private"
    ttl_seconds: Optional[int] = None


# ---------------------------
# MemCube: universal memory unit
# ---------------------------
@dataclass
class MemCube:
    payload: Union[str, Dict[str, Any], List[float], None] = None
    memory_type: MemoryType = MemoryType.PLAINTEXT
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    origin: str = "system"
    semantic_type: SemanticType = SemanticType.FACT
    access_scope: AccessScope = AccessScope.PRIVATE
    owner: str = "default_user"
    namespace: str = "default"
    ttl: Optional[int] = None
    priority: int = 0
    sensitivity_tags: List[str] = field(default_factory=list)
    access_count: int = 0
    last_access: Optional[datetime] = None
    version: int = 1
    parent_id: Optional[str] = None
    state: MemoryState = MemoryState.GENERATED
    embedding: Optional[List[float]] = None
    kv_cache: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    weight: float = 1.0

    def touch(self):
        self.access_count += 1
        self.last_access = datetime.now()
        # Reinforce weight on access
        self.weight = min(1.0, self.weight + 0.1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["last_access"] = self.last_access.isoformat() if self.last_access else None
        d["memory_type"] = self.memory_type.value
        d["state"] = self.state.value
        d["semantic_type"] = self.semantic_type.value
        d["access_scope"] = self.access_scope.value
        return d

    @classmethod
    def from_dict(cls, data: dict):
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("last_access"):
            data["last_access"] = datetime.fromisoformat(data["last_access"])
        data["memory_type"] = MemoryType(data["memory_type"])
        data["state"] = MemoryState(data["state"])
        data["semantic_type"] = SemanticType(data["semantic_type"])
        data["access_scope"] = AccessScope(data["access_scope"])
        return cls(**data)


# ---------------------------
# Memory payload helpers
# ---------------------------
def create_plaintext(text: str, **kwargs) -> MemCube:
    return MemCube(payload=text, memory_type=MemoryType.PLAINTEXT, **kwargs)


def create_activation(kv_pairs: Dict[str, Any], **kwargs) -> MemCube:
    return MemCube(
        payload=kv_pairs, memory_type=MemoryType.ACTIVATION, kv_cache=kv_pairs, **kwargs
    )


def create_parameter(adapter_ref: str, **kwargs) -> MemCube:
    return MemCube(
        payload={"adapter": adapter_ref}, memory_type=MemoryType.PARAMETER, **kwargs
    )


# ---------------------------
# MemVault: unified storage
# ---------------------------
class MemVault:
    def __init__(
        self, persist_directory: str = "./memvault", local_embedding: bool = False
    ):
        self.persist_directory = persist_directory
        self.local_embedding = local_embedding
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        embedding_function = None
        if self.local_embedding:
            try:
                from chromadb.utils import embedding_functions

                embedding_function = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                )
            except ImportError:
                print(
                    "\nError: 'sentence-transformers' is required for local embeddings but not found."
                )
                print("Please install it manually: pip install sentence-transformers\n")
                import sys

                sys.exit(1)
            except Exception as e:
                print(f"Error loading local embedding: {e}")
                import sys

                sys.exit(1)

        self.plaintext_collection = self.chroma_client.get_or_create_collection(
            name="plaintext_memory",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function,
        )
        self.graph = nx.MultiDiGraph()
        self.kv_store: Dict[str, MemCube] = {}
        self.namespaces: Dict[str, Set[str]] = defaultdict(set)

        self._load_from_disk()

    def _get_kv_path(self):
        return os.path.join(self.persist_directory, "kv_store.json")

    def _get_graph_path(self):
        return os.path.join(self.persist_directory, "graph.json")

    def _save_to_disk(self):
        try:
            # Save KV Store
            kv_data = {cid: cube.to_dict() for cid, cube in self.kv_store.items()}
            with open(self._get_kv_path(), "w") as f:
                json.dump(kv_data, f, indent=2)

            # Save Graph
            graph_data = nx.node_link_data(self.graph)
            with open(self._get_graph_path(), "w") as f:
                json.dump(graph_data, f, indent=2)
        except Exception as e:
            print(f"MemVault._save_to_disk failed: {e}")

    def _load_from_disk(self):
        try:
            # Load KV Store
            kv_path = self._get_kv_path()
            if os.path.exists(kv_path):
                with open(kv_path, "r") as f:
                    kv_data = json.load(f)
                    for cid, data in kv_data.items():
                        cube = MemCube.from_dict(data)
                        self.kv_store[cid] = cube
                        # Rebuild namespaces
                        ns = cube.namespace
                        # Note: namespace isn't in MemCube but was passed to store()
                        # For now we'll put them in default or extract from metadata if we had it
                        # Let's assume most are user namespaces
                        self.namespaces[ns].add(cid)

            # Load Graph
            graph_path = self._get_graph_path()
            if os.path.exists(graph_path):
                with open(graph_path, "r") as f:
                    graph_data = json.load(f)
                    self.graph = nx.node_link_graph(graph_data)
        except Exception as e:
            print(f"MemVault._load_from_disk failed: {e}")

    def store(self, cube: MemCube, namespace: str = "default") -> Optional[str]:
        try:
            cube_id = cube.id
            self.kv_store[cube_id] = cube
            cube.namespace = namespace
            self.namespaces[namespace].add(cube_id)
            self.graph.add_node(cube_id, cube=cube.to_dict(), namespace=namespace)

            if cube.memory_type == MemoryType.PLAINTEXT and isinstance(
                cube.payload, str
            ):
                self.plaintext_collection.add(
                    documents=[cube.payload],
                    metadatas=[
                        {
                            "cube_id": cube_id,
                            "namespace": namespace,
                            "semantic_type": cube.semantic_type.value,
                            "timestamp": cube.timestamp.isoformat(),
                            "owner": cube.owner,
                        }
                    ],
                    ids=[cube_id],
                )
            self._save_to_disk()
            return cube_id
        except Exception as e:
            print(f"MemVault.store failed: {e}")
            return None

    def get(self, cube_id: str) -> Optional[MemCube]:
        return self.kv_store.get(cube_id)

    def delete(self, cube_id: str):
        if cube_id in self.kv_store:
            cube = self.kv_store[cube_id]
            if cube.memory_type == MemoryType.PLAINTEXT:
                self.plaintext_collection.delete(ids=[cube_id])
            self.graph.remove_node(cube_id)
            del self.kv_store[cube_id]
            for ns in self.namespaces:
                self.namespaces[ns].discard(cube_id)
            self._save_to_disk()

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        namespace: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> List[MemCube]:
        where = {}
        if namespace:
            if isinstance(namespace, list):
                where["namespace"] = {"": namespace}
            else:
                where["namespace"] = namespace
        if owner:
            where["owner"] = owner
        try:
            results = self.plaintext_collection.query(
                query_texts=[query], n_results=n_results, where=where if where else None
            )
            cube_ids = results["ids"][0] if results["ids"] else []
        except Exception:
            cube_ids = []

        cubes = []
        for cid in cube_ids:
            cube = self.kv_store.get(cid)
            if cube:
                cube.touch()
                cubes.append(cube)
        return cubes

    def link_memories(self, src_id: str, tgt_id: str, relation: str, **attrs):
        self.graph.add_edge(src_id, tgt_id, relation=relation, **attrs)

    def get_related(
        self, cube_id: str, relation: Optional[str] = None
    ) -> List[MemCube]:
        if cube_id not in self.graph:
            return []
        edges = self.graph.edges(cube_id, data=True)
        related_ids = {
            tgt
            for _, tgt, data in edges
            if relation is None or data.get("relation") == relation
        }
        return [self.kv_store[cid] for cid in related_ids if cid in self.kv_store]

    def list_namespace(self, namespace: str) -> List[MemCube]:
        return [
            self.kv_store[cid]
            for cid in self.namespaces[namespace]
            if cid in self.kv_store
        ]


# ---------------------------
# MemGovernance
# ---------------------------
class MemGovernance:
    def __init__(self, decay_rate: float = 0.05, min_weight: float = 0.3):
        self.audit_log = []
        self.user_roles = {}
        self.decay_rate = decay_rate  # Reduction factor per cycle
        self.min_weight = min_weight  # Weight threshold for archiving

    def check_access(self, cube: MemCube, user: str, operation: str = "read") -> bool:
        if cube.owner == user:
            return True
        if cube.access_scope == AccessScope.PRIVATE:
            return False
        if cube.access_scope == AccessScope.SHARED:
            return operation in ["read", "query"]
        if cube.access_scope == AccessScope.PUBLIC:
            return True
        return False

    def decay_memories(self, vault: MemVault):
        """Gradually reduces weight of all memories and archives cold ones."""
        to_archive = []
        for cid, cube in vault.kv_store.items():
            if (
                cube.state == MemoryState.GENERATED
                or cube.state == MemoryState.ACTIVATED
            ):
                # Apply decay
                cube.weight = max(0.0, cube.weight - self.decay_rate)

                # Check for "forgetting" threshold
                if cube.weight < self.min_weight:
                    to_archive.append(cid)

        for cid in to_archive:
            cube = vault.kv_store[cid]
            cube.state = MemoryState.ARCHIVED
            self.audit(
                "WEIGHT_DECAY_ARCHIVE", {"cube_id": cid, "final_weight": cube.weight}
            )

    def enforce_ttl(self, vault: MemVault):
        now = datetime.now()
        to_delete = []
        for cid, cube in vault.kv_store.items():
            if cube.ttl and cube.timestamp:
                expiry = cube.timestamp + timedelta(seconds=cube.ttl)
                if now > expiry:
                    cube.state = MemoryState.EXPIRED
                    to_delete.append(cid)

        for cid in to_delete:
            vault.delete(cid)
            self.audit("TTL_EXPIRE", {"cube_id": cid})

        # Also run weight decay during this cycle
        self.decay_memories(vault)

    def audit(self, action: str, details: dict):
        self.audit_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details,
            }
        )


# ---------------------------
# MemReader
# ---------------------------
class MemReader:
    def __init__(
        self, model: str = "ollama/gemma3n:e4b", api_base: Optional[str] = None
    ):
        self.model = model
        self.api_base = api_base

    def parse(self, prompt: str, user: str = "default_user") -> Dict[str, Any]:
        try:
            system_msg = (
                "You are a Memory Operations Parser. Analyze the user prompt and extract the intended memory action.\n"
                "Available operations: store, retrieve, update, delete, query, summarize.\n"
                "Semantic types: fact, preference, task, dialogue, procedure, rule, correction, insight.\n\n"
                "Guidelines:\n"
                "1. If the user indicates a CHANGE of state (e.g., 'I moved to', 'I now live in', 'Actually I prefer'), "
                "set operation to 'update' and identify the subject in task_intent.\n"
                "2. If the user corrects themselves (e.g., 'No, I meant', 'Actually', 'I was wrong'), set semantic_type to 'correction'.\n"
                "3. If the user or agent identifies a complex pattern or deep understanding, set semantic_type to 'insight'.\n"
                "4. Extract the core information into 'content_summary'.\n"
                "5. If storing a fact about the user, set semantic_type to 'fact'.\n"
                "6. If asking a question, set operation to 'retrieve'."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format=MemoryOperation,
                api_base=self.api_base,
                timeout=15,
            )
            content = response.choices[0].message.content
            parsed = MemoryOperation.model_validate_json(content).model_dump()
            parsed["user"] = user
            parsed["timestamp"] = datetime.now().isoformat()
            return parsed
        except Exception:
            return self._smart_rule_parse(prompt, user)

    def _smart_rule_parse(self, prompt: str, user: str) -> dict:
        lower = prompt.lower().strip()
        parsed = {
            "operation": "retrieve",
            "memory_type": None,
            "semantic_type": "fact",
            "time_scope": None,
            "entities": [],
            "context_window": None,
            "content_summary": prompt,
            "task_intent": "general query",
            "memory_scope": "private",
            "ttl_seconds": None,
            "user": user,
            "timestamp": datetime.now().isoformat(),
        }

        # Handle very short conversational fillers by doing NOTHING
        if lower in ["yes", "no", "ok", "okay", "thanks", "thank you", "yep", "nope"]:
            parsed["operation"] = "none"
            return parsed

        store_triggers = [
            "remember",
            "save",
            "store",
            "add",
            "note that",
            "keep in mind",
            "don't forget",
        ]
        pref_triggers = ["i like", "i prefer", "my favorite", "i love", "my preference"]
        identity_triggers = [
            "i live in",
            "i am in",
            "i moved to",
            "i now live in",
            "my name is",
            "i am a",
            "i work at",
            "i am from",
        ]
        correction_triggers = [
            "no, i meant",
            "actually",
            "no, actually",
            "i was wrong",
            "correction:",
        ]
        insight_triggers = ["i realized", "i discovered", "pattern:", "insight:"]
        delete_triggers = ["forget", "delete", "remove", "clear"]
        query_triggers = [
            "what",
            "how",
            "who",
            "where",
            "when",
            "why",
            "do you know",
            "tell me about",
            "list",
        ]

        if any(k in lower for k in delete_triggers):
            parsed["operation"] = "delete"
            parsed["task_intent"] = "memory deletion"
        elif lower.startswith("list "):
            parsed["operation"] = "summarize"
            parsed["task_intent"] = "list all memories"
            parsed["content_summary"] = lower[5:].strip()
            if "preference" in lower:
                parsed["semantic_type"] = "preference"
            elif "fact" in lower:
                parsed["semantic_type"] = "fact"
        elif any(k in lower for k in identity_triggers):
            parsed["operation"] = "store"
            parsed["task_intent"] = "identity update"
            parsed["content_summary"] = prompt
        elif any(lower.startswith(k) for k in query_triggers) or lower.endswith("?"):
            parsed["operation"] = "retrieve"
            parsed["task_intent"] = "memory retrieval"
            content = prompt.strip("?")
            # Strip query triggers from the summary
            for p in sorted(query_triggers, key=len, reverse=True):
                if lower.startswith(p):
                    content = content[len(p) :].strip()
                    lower = content.lower()
            # Also strip common fillers if they remain at the start
            for filler in [
                "is my ",
                "are my ",
                "do you know about ",
                "do you know ",
                "tell me about ",
                "about ",
            ]:
                if lower.startswith(filler):
                    content = content[len(filler) :].strip()
                    lower = content.lower()
            parsed["content_summary"] = content
        elif any(k in lower for k in correction_triggers):
            parsed["operation"] = "store"
            parsed["semantic_type"] = "correction"
            parsed["task_intent"] = "self-correction"
            parsed["content_summary"] = prompt
        elif any(k in lower for k in insight_triggers):
            parsed["operation"] = "store"
            parsed["semantic_type"] = "insight"
            parsed["task_intent"] = "new insight"
            parsed["content_summary"] = prompt
        elif any(k in lower for k in store_triggers + pref_triggers):
            parsed["operation"] = "store"
            parsed["task_intent"] = "memory storage"
            content = prompt
            for p in sorted(store_triggers + ["that ", "to "], key=len, reverse=True):
                if lower.startswith(p):
                    content = content[len(p) :].strip()
                    lower = content.lower()
            parsed["content_summary"] = content
            if any(p in lower for p in pref_triggers):
                parsed["semantic_type"] = "preference"

        return parsed


# ---------------------------
# MemoryAPI
# ---------------------------
class MemoryAPI:
    def __init__(self, vault: MemVault, governance: MemGovernance):
        self.vault = vault
        self.governance = governance

    def _format_payload(self, payload: Any, max_len: int = 50) -> str:
        if isinstance(payload, str):
            return payload[:max_len] + "…" if len(payload) > max_len else payload
        elif isinstance(payload, dict):
            if "adapter" in payload:
                return f"Parameter(adapter={payload['adapter']})"
            items = list(payload.items())[:3]
            s = ", ".join(f"{k}:{v}" for k, v in items)
            return f"{{{s}…}}" if len(payload) > 3 else f"{{{s}}}"
        return str(payload)[:max_len]

    def create(self, cube: MemCube, namespace: str = "default") -> Optional[str]:
        self.governance.audit("CREATE", {"cube_id": cube.id, "owner": cube.owner})
        return self.vault.store(cube, namespace)

    def read(self, cube_id: str, user: str) -> Optional[MemCube]:
        cube = self.vault.get(cube_id)
        if cube and self.governance.check_access(cube, user, "read"):
            cube.touch()
            self.governance.audit("READ", {"cube_id": cube_id, "user": user})
            return cube
        return None

    def update_access_scope(self, cube_id: str, scope: AccessScope, user: str) -> bool:
        cube = self.vault.get(cube_id)
        if cube and cube.owner == user:
            cube.access_scope = scope
            self.governance.audit(
                "ACCESS_UPDATE", {"cube_id": cube_id, "scope": scope.value}
            )
            return True
        return False

    def delete(self, cube_id: str, user: str) -> bool:
        cube = self.vault.get(cube_id)
        if cube and self.governance.check_access(cube, user, "delete"):
            self.vault.delete(cube_id)
            self.governance.audit("DELETE", {"cube_id": cube_id, "user": user})
            return True
        return False

    def query(
        self,
        query_text: str,
        user: str,
        namespace: Optional[str] = None,
        n_results: int = 5,
    ) -> List[MemCube]:
        # Search broadly to find shared/public memories.
        # We remove the strict 'owner' filter from the vault query.
        candidates = self.vault.semantic_search(
            query_text, n_results * 3, namespace=namespace, owner=None
        )

        # Fallback to keyword matching if semantic search returns nothing or is unavailable
        if not candidates:
            query_lower = query_text.lower()
            # Try exact substring match first
            for cube in self.vault.kv_store.values():
                if (
                    isinstance(cube.payload, str)
                    and query_lower in cube.payload.lower()
                ):
                    if self.governance.check_access(cube, user, "read"):
                        candidates.append(cube)

            # If still nothing, try word-based matching
            if not candidates:
                stop_words = {
                    "what",
                    "is",
                    "my",
                    "are",
                    "the",
                    "a",
                    "an",
                    "for",
                    "do",
                    "you",
                    "know",
                    "about",
                    "tell",
                    "me",
                }
                query_words = [
                    w.strip("?!.,")
                    for w in query_lower.split()
                    if w not in stop_words and len(w) > 2
                ]

                for cube in self.vault.kv_store.values():
                    if not self.governance.check_access(cube, user, "read"):
                        continue
                    if isinstance(cube.payload, str):
                        payload_lower = cube.payload.lower()
                        if any(word in payload_lower for word in query_words):
                            candidates.append(cube)

        # Governance ensures the user can only see what they are allowed to.
        # Deduplicate and limit results
        seen = set()
        final_candidates = []
        for c in candidates:
            if c.id not in seen and self.governance.check_access(c, user, "read"):
                final_candidates.append(c)
                seen.add(c.id)

        return final_candidates[:n_results]


# ---------------------------
# MemOperator
# ---------------------------
class MemOperator:
    def __init__(self, vault: MemVault, api: MemoryAPI):
        self.vault = vault
        self.api = api

    def hybrid_retrieve(
        self, query: str, user: str, namespace: Optional[str] = None, n_results: int = 5
    ) -> List[MemCube]:
        semantic_results = self.api.query(query, user, namespace, n_results * 2)
        # Filter out archived memories
        active_results = [
            c for c in semantic_results if c.state != MemoryState.ARCHIVED
        ]
        active_results.sort(key=lambda c: (c.priority, c.timestamp), reverse=True)
        return active_results[:n_results]


# ---------------------------
# MemScheduler
# ---------------------------
class MemScheduler:
    def __init__(self, vault: MemVault):
        self.vault = vault

    def schedule(self, task: dict, context: Dict[str, Any]) -> List[MemCube]:
        selected = []
        task_intent = task.get("task_intent", "").lower()
        content = task.get("content_summary", "").lower()
        user = task.get("user", "default_user")

        if "retrieval" in task_intent or "query" in task_intent:
            selected.extend(self._get_plaintext_candidates(content, user, limit=5))
        return selected

    def transform(self, cube: MemCube, target_type: MemoryType) -> Optional[MemCube]:
        if (
            cube.memory_type == MemoryType.PLAINTEXT
            and target_type == MemoryType.ACTIVATION
        ):
            if isinstance(cube.payload, str):
                words = cube.payload.split()[:10]
                kv = {f"k_{i}": w for i, w in enumerate(words)}
                return create_activation(
                    kv_pairs=kv,
                    semantic_type=cube.semantic_type,
                    owner=cube.owner,
                    parent_id=cube.id,
                )
        return None

    def _get_plaintext_candidates(
        self, query: str, user: str, limit: int = 5
    ) -> List[MemCube]:
        candidates = []
        stop_words = {"where", "does", "the", "is", "my", "are"}
        query_words = [w.lower() for w in query.split() if w.lower() not in stop_words]

        for cube in self.vault.kv_store.values():
            if (
                cube.memory_type == MemoryType.PLAINTEXT
                and cube.state != MemoryState.ARCHIVED
            ):
                if isinstance(cube.payload, str):
                    payload_lower = cube.payload.lower()
                    if any(word in payload_lower for word in query_words):
                        # Include user's own memories OR shared/public ones
                        if cube.owner == user or cube.access_scope in [
                            AccessScope.SHARED,
                            AccessScope.PUBLIC,
                        ]:
                            candidates.append(cube)
        return candidates[:limit]


# ---------------------------
# MemLifecycle
# ---------------------------
class MemLifecycle:
    def __init__(self, vault: MemVault, governance: MemGovernance):
        self.vault = vault
        self.governance = governance

    def transition(self, cube_id: str, new_state: MemoryState) -> bool:
        cube = self.vault.get(cube_id)
        if cube:
            cube.state = new_state
            return True
        return False


# ---------------------------
# Lane-Based Command Queue
# ---------------------------
class SessionLane:
    """
    Ensures that tasks for a specific session/user run one after another, not in parallel.
    Prevents race conditions and interleaved logs.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def run(self, func: Callable, *args, **kwargs):
        with self._lock:
            return func(*args, **kwargs)


# ---------------------------
# MemOS
# ---------------------------
class MemOS:
    def __init__(
        self,
        persist_directory: str = "./memvault",
        model: str = "ollama/gemma3n:e4b",
        local_embedding: bool = False,
        decay_rate: float = 0.05,
        min_weight: float = 0.3,
        api_base: Optional[str] = None,
    ):
        self.local_embedding = local_embedding
        self.persist_directory = persist_directory
        self.model = model
        self.api_base = api_base
        self.vault = MemVault(persist_directory, local_embedding=local_embedding)
        self.governance = MemGovernance(decay_rate=decay_rate, min_weight=min_weight)
        self.api = MemoryAPI(self.vault, self.governance)
        self.reader = MemReader(model=model, api_base=api_base)
        self.operator = MemOperator(self.vault, self.api)
        self.scheduler = MemScheduler(self.vault)
        self.lifecycle = MemLifecycle(self.vault, self.governance)
        self.default_user = "alice"
        self._lanes: Dict[str, SessionLane] = defaultdict(SessionLane)

    def enable_local_embedding(self, enabled: bool = True):
        """Toggles local embedding support. Requires restart or re-initialization of vault."""
        self.local_embedding = enabled
        self.vault = MemVault(self.persist_directory, local_embedding=enabled)
        # Update components that hold a reference to the old vault
        self.api.vault = self.vault
        self.operator.vault = self.vault
        self.scheduler.vault = self.vault
        self.lifecycle.vault = self.vault

    def process(
        self,
        prompt: str,
        user: Optional[str] = None,
        response: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        user = user or self.default_user
        lane = self._lanes[user]
        return lane.run(self._process_internal, prompt, user, response, namespace)

    def _process_internal(
        self,
        prompt: str,
        user: str,
        response: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        parsed = self.reader.parse(prompt, user)
        res = {"parsed": parsed, "cubes": [], "response": ""}

        scheduled_cubes = self.scheduler.schedule(parsed, {})

        if parsed["operation"] in ("store", "update"):
            content = parsed["content_summary"]
            # Proactive conflict resolution for Facts, Updates, or Corrections
            if (
                parsed["semantic_type"] in ["fact", "correction"]
                or parsed["operation"] == "update"
            ):
                # Extract potential keywords to find old versions
                search_q = parsed.get("task_intent") or " ".join(
                    [w for w in content.lower().split() if len(w) > 3]
                )

                if search_q:
                    existing = self.operator.hybrid_retrieve(
                        query=search_q, user=user, n_results=5
                    )
                    for old_cube in existing:
                        # Archive old facts if we have a new fact or correction about the same thing
                        if old_cube.semantic_type in [
                            SemanticType.FACT,
                            SemanticType.CORRECTION,
                        ]:
                            self.lifecycle.transition(old_cube.id, MemoryState.ARCHIVED)

            cube = create_plaintext(
                text=content,
                semantic_type=SemanticType(parsed["semantic_type"]),
                owner=user,
            )

            # Boost priority for corrections
            if cube.semantic_type == SemanticType.CORRECTION:
                cube.priority = 5
            elif cube.semantic_type == SemanticType.INSIGHT:
                cube.priority = 3

            cid = self.api.create(cube, namespace=namespace or f"user_{user}")
            if cid:
                res["cubes"].append(cid)
                res["response"] = (
                    f"{parsed['semantic_type'].capitalize()} recorded: {content}"
                )

        elif parsed["operation"] in ("retrieve", "query"):
            # We don't restrict to user namespace for retrieval to allow finding shared memories
            cubes = self.operator.hybrid_retrieve(
                query=parsed["content_summary"],
                user=user,
                namespace=namespace,
                n_results=3,
            )
            all_cubes = cubes + [
                c for c in scheduled_cubes if c.id not in [rc.id for rc in cubes]
            ]

            # GLOBAL SORT: Ensure newest is always first for the LLM
            # Also consider priority in sorting if multiple are relevant
            all_cubes.sort(key=lambda c: (c.priority, c.timestamp), reverse=True)

            res["cubes"] = [c.id for c in all_cubes]
            if all_cubes:
                snippets = [
                    f"• [{c.timestamp.strftime('%Y-%m-%d %H:%M')}] [{c.semantic_type.value.upper()}] {self.api._format_payload(c.payload, 100)}"
                    for c in all_cubes[:5]
                ]
                res["response"] = (
                    "Memory Context (Prioritized & Newest First):\n"
                    + "\n".join(snippets)
                )
            else:
                res["response"] = "No relevant memories found in vault."

        elif parsed["operation"] == "summarize":
            # List memories of a certain type
            all_cubes = [
                c
                for c in self.vault.kv_store.values()
                if c.owner == user and c.state != MemoryState.ARCHIVED
            ]

            target_type = parsed["semantic_type"]
            if "preference" in parsed["content_summary"]:
                target_type = "preference"
            elif "fact" in parsed["content_summary"]:
                target_type = "fact"
            elif "correction" in parsed["content_summary"]:
                target_type = "correction"
            elif "insight" in parsed["content_summary"]:
                target_type = "insight"

            filtered = [c for c in all_cubes if c.semantic_type.value == target_type]

            filtered.sort(key=lambda c: c.timestamp, reverse=True)
            res["cubes"] = [c.id for c in filtered]
            if filtered:
                snippets = [
                    f"• {self.api._format_payload(c.payload, 150)}" for c in filtered
                ]
                res["response"] = (
                    f"Here are your {target_type}s (Newest First):\n"
                    + "\n".join(snippets)
                )
            else:
                res["response"] = f"I couldn't find any {target_type}s in my memory."

        elif parsed["operation"] == "delete":
            cubes = self.operator.hybrid_retrieve(
                parsed["content_summary"], user, n_results=1
            )
            if cubes:
                self.api.delete(cubes[0].id, user)
                res["response"] = f"Deleted memory {cubes[0].id[:8]}"
            else:
                res["response"] = "Nothing found to delete."

        elif parsed["operation"] == "none":
            res["response"] = "Conversational acknowledgment."

        if not res["response"]:
            res["response"] = "Operation completed with no direct response."

        self.governance.enforce_ttl(self.vault)

        # Periodic Distillation (Triggered if we have new dialogue)
        if parsed["operation"] == "none" or (
            parsed["operation"] == "retrieve" and not res["cubes"]
        ):
            # If it's just a conversational turn or a query that found nothing,
            # let's save the raw dialogue first
            dialogue_text = f"User: {prompt}"
            if response:
                dialogue_text += f"\nAssistant: {response}"
            cube = create_plaintext(
                text=dialogue_text, semantic_type=SemanticType.DIALOGUE, owner=user
            )
            self.api.create(
                cube,
                namespace=(namespace + "_logs") if namespace else f"user_{user}_logs",
            )

            # Then check if we should distill
            self._distill_conversations(user)

        return res

    def _distill_conversations(self, user: str):
        """
        Periodically reviews raw DIALOGUE logs and distills them into FACTs or PREFERENCEs.
        """
        # Count recent dialogue
        logs = [
            c
            for c in self.vault.kv_store.values()
            if c.owner == user
            and c.semantic_type == SemanticType.DIALOGUE
            and c.state == MemoryState.GENERATED
        ]

        if len(logs) >= 5:  # Threshold for distillation
            try:
                text_to_distill = "\n".join([str(c.payload) for c in logs])
                system_msg = (
                    "You are a Memory Distiller. Analyze the following conversation logs and extract "
                    "any new facts or preferences about the user that are NOT already mentioned. "
                    "Output a list of concise statements to remember. If nothing new, output an empty list."
                )

                # Check existing memories for context to avoid duplicates
                existing = [
                    c.payload
                    for c in self.vault.kv_store.values()
                    if c.owner == user
                    and c.semantic_type in [SemanticType.FACT, SemanticType.PREFERENCE]
                ]
                context = "\nExisting memories:\n" + "\n".join(
                    [str(e) for e in existing[:10]]
                )

                response = litellm.completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg + context},
                        {"role": "user", "content": text_to_distill},
                    ],
                    api_base=self.api_base,
                    timeout=20,
                )

                # Mark logs as merged first to avoid infinite loops if processing takes time
                for log in logs:
                    self.lifecycle.transition(log.id, MemoryState.MERGED)

                distilled_text = response.choices[0].message.content
                if distilled_text and len(distilled_text.strip()) > 5:
                    for line in distilled_text.split("\n"):
                        line = line.strip("- ").strip()
                        if line and len(line) > 10:
                            # Parse semantic type for the new memo
                            is_pref = any(
                                k in line.lower()
                                for k in ["prefer", "like", "love", "favorite", "hate"]
                            )
                            sem_type = (
                                SemanticType.PREFERENCE
                                if is_pref
                                else SemanticType.FACT
                            )

                            new_cube = create_plaintext(
                                text=line, semantic_type=sem_type, owner=user
                            )
                            self.api.create(new_cube, namespace=f"user_{user}")
            except Exception as e:
                print(f"Distillation failed: {e}")


# ---------------------------
# LangGraph Integration
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user: str
    memory_response: Optional[str]


def create_memory_agent(
    memos: MemOS, model: Optional[str] = None, tools: Optional[List[Any]] = None
):
    if model is None:
        model = memos.reader.model

    def memory_node(state: AgentState):
        last_message = state["messages"][-1].content
        user = state.get("user", "alice")
        result = memos.process(last_message, user=user)
        return {"memory_response": result.get("response", "")}

    def llm_node(state: AgentState):
        system_prompt = (
            "You are a Memory-Augmented Assistant. "
            "Use the provided Memory Context to answer the user. "
            "IMPORTANT: Memories are provided in reverse chronological order (newest first). "
            "If you find conflicting information (e.g., two different locations for where the user lives), "
            "ALWAYS prioritize the newest information and treat it as the current truth. "
            "Do NOT mention the old/conflicting information unless specifically asked about history."
        )
        msgs = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nMemory Context: {state.get('memory_response', '')}",
            }
        ]
        for m in state["messages"]:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            msgs.append({"role": role, "content": m.content})

        completion_kwargs = {"model": model, "messages": msgs}
        if tools:
            completion_kwargs["tools"] = tools

        response = litellm.completion(**completion_kwargs)
        return {
            "messages": [AIMessage(content=response.choices[0].message.content or "")]
        }

    graph = StateGraph(AgentState)
    graph.add_node("memory", memory_node)
    graph.add_node("llm", llm_node)
    graph.set_entry_point("memory")
    graph.add_edge("memory", "llm")
    graph.add_edge("llm", END)
    return graph.compile()


def get_memory_tools(memos: MemOS, user: str):
    """
    Returns a list of LangChain tools for explicit memory management.
    """

    @tool
    def store_memory(content: str, semantic_type: str = "fact"):
        """Stores important information in long-term memory."""
        try:
            st = SemanticType(semantic_type.lower())
        except ValueError:
            st = SemanticType.FACT
        cube = create_plaintext(text=content, semantic_type=st, owner=user)
        memos.api.create(cube, namespace=f"user_{user}")
        return "Memory stored."

    @tool
    def search_memory(query: str):
        """Searches long-term memory for past information."""
        # Use namespace=None to find shared memories across users
        results = memos.operator.hybrid_retrieve(
            query=query, user=user, namespace=None, n_results=5
        )
        if not results:
            return "No memories found."
        return "\n".join(
            [
                f"[{c.semantic_type.value}] {memos.api._format_payload(c.payload, 200)}"
                for c in results
            ]
        )

    @tool
    def set_memory_access(memory_id: str, scope: str):
        """
        Updates the privacy scope of a specific memory.
        'scope' can be: private, shared, public.
        Only the owner of the memory can change its scope.
        """
        try:
            as_scope = AccessScope(scope.lower())
        except ValueError:
            return f"Invalid scope '{scope}'. Use private, shared, or public."

        success = memos.api.update_access_scope(memory_id, as_scope, user)
        if success:
            return f"Access scope for memory {memory_id} updated to {scope}."
        return f"Failed to update access scope. You may not be the owner."

    return [store_memory, search_memory, set_memory_access]
