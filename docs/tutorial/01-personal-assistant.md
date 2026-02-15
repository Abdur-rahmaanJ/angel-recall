# Tutorial 1: Building a Persistent Personal Assistant

In this tutorial, we will build a simple personal assistant that "learns" user preferences over time and uses them to provide personalized responses.

## Scenario
Alice wants an assistant that remembers her favorite programming languages and UI preferences so she doesn't have to repeat them every time.

## Complete Code

```python
import shutil
import os
from angel_recall import MemOS

def main():
    # 1. Setup
    vault_path = "./personal_assistant_vault"
    if os.path.exists(vault_path):
        shutil.rmtree(vault_path)

    # Initialize MemOS
    # We use a local vault to persist memory between runs
    memos = MemOS(persist_directory=vault_path)

    print("--- Session 1: Learning ---")
    
    # Alice tells the assistant about her preferences
    memos.process("Remember that I prefer Python for scripting and Rust for performance.", user="alice")
    memos.process("I like my UI in dark mode.", user="alice")
    
    print("Assistant: I've noted your preferences, Alice.")

    print("--- Session 2: Recalling ---")

    # Later, Alice asks for a recommendation
    query = "What language should I use for a fast data processor?"
    result = memos.process(query, user="alice")
    
    print(f"User: {query}")
    print(f"""Assistant Context:
{result['response']}""")

    # Another query
    query2 = "How should I theme my new dashboard?"
    result2 = memos.process(query2, user="alice")
    
    print(f"""
User: {query2}")
    print(f"Assistant Context:
{result2['response']}""")

    # 2. Cleanup (Optional for persistence)
    # shutil.rmtree(vault_path)

if __name__ == "__main__":
    main()
```

## How it works
1.  **`memos.process(...)`**: When Alice says "Remember that...", the `MemReader` automatically identifies the operation as `store` and the semantic type as `preference`.
2.  **Implicit Retrieval**: When Alice asks a question, `MemOS` automatically performs a hybrid search in the `MemVault`.
3.  **Context Injection**: The `result["response"]` contains snippets from the vault that the LLM (or the user) can use to ground the answer.
