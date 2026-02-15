# Tutorial 3: Building a Multi-User Bot with Safe Concurrency

In this tutorial, we explore the **SessionLane** architecture, which ensures that concurrent requests from the same user are handled sequentially to prevent race conditions.

## Scenario
A chatbot is receiving multiple messages from the same user almost simultaneously. We want to ensure that each message is processed fully before the next one starts, maintaining a consistent memory state.

## Complete Code

```python
import threading
import time
import shutil
import os
from angel_recall import MemOS

def main():
    vault_path = "./concurrency_vault"
    if os.path.exists(vault_path):
        shutil.rmtree(vault_path)

    memos = MemOS(persist_directory=vault_path)

    def rapid_fire_user_task(user_id, task_id):
        print(f"[Thread {task_id}] User {user_id}: Sending request...")
        # memos.process is thread-safe thanks to SessionLanes
        res = memos.process(f"Remember Task {task_id} for {user_id}", user=user_id)
        print(f"[Thread {task_id}] User {user_id}: Finished. Result: {res['response']}")

    print("--- Firing 3 concurrent requests for 'Alice' ---")
    threads = []
    for i in range(3):
        t = threading.Thread(target=rapid_fire_user_task, args=("alice", i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("--- Verifying Alice's memory ---")
    final_res = memos.process("What tasks did I record?", user="alice")
    print(final_res["response"])

if __name__ == "__main__":
    main()
```

## How it works
1.  **`SessionLane`**: Behind the scenes, `MemOS` maintains a dictionary of lanes (locks). When `process(user="alice")` is called, it acquires the lock for `"alice"`.
2.  **Sequential Processing**: Even though the threads start almost at the same time, the `MemOS` ensures that "Task 0" finishes before "Task 1" begins for Alice.
3.  **Global Parallelism**: If "Bob" and "Alice" sent messages at the same time, they *would* run in parallel, because they belong to different lanes. This provides a balance between safety (per-user) and performance (system-wide).
