#!/usr/bin/env python3
"""
Angel Recall: The Memory Operating System Demo
Demonstrates multi-user isolation, access governance, and lane-based execution.
"""

import time
import threading
import shutil
import os
from langchain_core.messages import HumanMessage
from angel_recall import (
    MemOS, 
    AccessScope, 
    SemanticType, 
    create_plaintext,
    create_memory_agent
)

def print_banner(text):
    print(f"\n{'='*60}\n{text:^60}\n{'='*60}")

def demo_multi_user_isolation(memos):
    print_banner("Scenario 1: Multi-User Isolation & Privacy")
    
    # Alice stores a private secret
    print("Alice: Remembering a private secret...")
    memos.process("Remember that my secret vault code is 'RAVEN-99'.", user="alice")
    
    # Bob tries to find it
    print("Bob: Searching for Alice's secrets...")
    res_bob = memos.process("What is the vault code?", user="bob")
    print(f"Assistant to Bob: {res_bob['response']}")
    
    if "RAVEN-99" not in res_bob['response']:
        print("✅ Bob successfully blocked from Alice's private memory.")

def demo_access_governance(memos):
    print_banner("Scenario 2: Access Governance (Sharing)")
    
    # Alice decides to share the project Wi-Fi
    print("Alice: Storing a shared Wi-Fi password...")
    cube_id = memos.api.create(
        create_plaintext(
            "The lab Wi-Fi password is 'Recall2026'.",
            semantic_type=SemanticType.FACT,
            owner="alice"
        )
    )
    
    # By default it is private, Bob still can't see it
    res_bob_1 = memos.process("What is the lab Wi-Fi password?", user="bob")
    print(f"Assistant to Bob (before share): {res_bob_1['response']}")
    
    # Alice updates governance
    print("Alice: Updating Wi-Fi access to SHARED...")
    memos.api.update_access_scope(cube_id, AccessScope.SHARED, user="alice")
    
    # Now Bob can see it
    res_bob_2 = memos.process("What is the lab Wi-Fi password?", user="bob")
    print(f"Assistant to Bob (after share): {res_bob_2['response']}")

def demo_lane_based_queue(memos):
    print_banner("Scenario 3: Lane-Based Command Queue")
    print("Firing multiple requests for 'alice' concurrently.")
    print("Notice: They execute sequentially in Alice's lane to prevent race conditions.")

    def task(i):
        print(f"  [Thread {i}] Sending request for Alice...")
        res = memos.process(f"Task {i} for Alice", user="alice")
        print(f"  [Thread {i}] Alice's lane finished Task {i}.")

    threads = [threading.Thread(target=task, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    print("✅ All lane tasks completed safely.")

def demo_agent_integration(memos):
    print_banner("Scenario 4: LangGraph Agent Integration")
    
    # Create the agent
    agent = create_memory_agent(memos)
    
    print("User (Alice): 'What do you know about my technical preferences?'")
    # Alice had preferences set in Scenario 1/2 potentially, but let's add one
    memos.process("I prefer Rust for performance-critical components.", user="alice")
    
    # Invoke the agent
    result = agent.invoke({
        "messages": [HumanMessage(content="What are my technical preferences?")],
        "user": "alice"
    })
    
    print(f"Agent Response: {result['messages'][-1].content}")

def main():
    vault_path = "./demo_vault"
    if os.path.exists(vault_path):
        shutil.rmtree(vault_path)

    # Initialize MemOS with a local vault
    # model="ollama/qwen2.5:0.5b" is lightweight for local testing
    memos = MemOS(persist_directory=vault_path, model="ollama/qwen2.5:0.5b")

    try:
        demo_multi_user_isolation(memos)
        demo_access_governance(memos)
        demo_lane_based_queue(memos)
        demo_agent_integration(memos)
    finally:
        # Cleanup
        if os.path.exists(vault_path):
            shutil.rmtree(vault_path)
        print_banner("Demo Completed Successfully")

if __name__ == "__main__":
    main()
