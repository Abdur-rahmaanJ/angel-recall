# Tutorial 2: Collaborative Knowledge Base with Access Control

This tutorial demonstrates how to manage memory across multiple users, transitioning information from private to shared.

## Scenario
Alice stores a sensitive project code. Later, she decides the whole team (including Bob) needs to know it.

## Complete Code

```python
import shutil
import os
from angel_recall import MemOS, AccessScope, create_plaintext, SemanticType

def main():
    vault_path = "./collab_vault"
    if os.path.exists(vault_path):
        shutil.rmtree(vault_path)

    memos = MemOS(persist_directory=vault_path)

    print("--- Step 1: Alice stores a private secret ---")
    
    # Alice stores a fact manually using the API to get the cube_id
    cube_id = memos.api.create(
        create_plaintext(
            "The secret project code is 'GALAXY-X'.",
            owner="alice",
            semantic_type=SemanticType.FACT
        )
    )
    print(f"Alice: Stored project code (ID: {cube_id[:8]})")

    print("--- Step 2: Bob tries to find it ---")
    
    # Bob searches for the code
    res_bob = memos.process("What is the secret project code?", user="bob")
    print(f"Assistant to Bob: {res_bob['response']}")
    # Bob gets "No relevant memories found" because it's PRIVATE to Alice.

    print("--- Step 3: Alice shares the memory ---")
    
    # Alice updates the access scope to SHARED
    memos.api.update_access_scope(cube_id, AccessScope.SHARED, user="alice")
    print("Alice: Updated project code access to SHARED.")

    print("--- Step 4: Bob tries again ---")
    
    # Now Bob can see it
    res_bob_shared = memos.process("What is the secret project code?", user="bob")
    print(f"Assistant to Bob: {res_bob_shared['response']}")

if __name__ == "__main__":
    main()
```

## How it works
1.  **Isolation by Default**: Every `MemCube` is private to its owner unless specified otherwise.
2.  **`MemGovernance`**: When Bob queries the vault, the governance layer filters out Alice's private cubes.
3.  **Dynamic Access**: By changing the `AccessScope` from `PRIVATE` to `SHARED`, the memory becomes visible to any authenticated user in the system.
