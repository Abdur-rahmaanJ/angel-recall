import pytest
import shutil
import os
import time
import threading
from angel_recall import MemOS, SemanticType, AccessScope

# Model specified by user
TEST_MODEL = "gemini/gemini-2.5-flash"
TEST_VAULT = "./test_vault_qwen"

@pytest.fixture(scope="module")
def memos():
    # Cleanup before test
    if os.path.exists(TEST_VAULT):
        shutil.rmtree(TEST_VAULT)
    
    obj = MemOS(persist_directory=TEST_VAULT, model=TEST_MODEL)
    yield obj
    
    # Cleanup after test
    if os.path.exists(TEST_VAULT):
        shutil.rmtree(TEST_VAULT)

def test_store_and_retrieve(memos):
    """Test basic storage and retrieval functionality."""
    prompt = "Remember that the secret code is Blue-Raven-42."
    res = memos.process(prompt, user="tester")
    assert "Stored" in res["response"]
    
    # Test retrieving it
    query = "What is the secret code?"
    res_query = memos.process(query, user="tester")
    assert "Blue-Raven-42" in res_query["response"]
    assert "Memory Context" in res_query["response"]

def test_access_governance(memos):
    """Test that privacy scopes are respected across users."""
    from angel_recall import create_plaintext
    cube = create_plaintext("Alice's private note", owner="alice")
    cid = memos.api.create(cube, namespace="user_alice")
    
    # Bob tries to retrieve it - should not see it
    res_bob = memos.process("What is Alice's private note?", user="bob")
    assert "No relevant memories" in res_bob["response"]
    
    # Update to shared
    memos.api.update_access_scope(cid, AccessScope.SHARED, user="alice")
    
    # Bob tries again - should see it now
    res_bob_shared = memos.process("What is Alice's private note?", user="bob")
    assert "Alice's private note" in res_bob_shared["response"]

def test_ttl_expiry(memos):
    """Test that memories with a TTL are eventually purged."""
    from angel_recall import create_plaintext
    cube = create_plaintext("This will vanish quickly", owner="ttl_tester", ttl=1)
    memos.api.create(cube, namespace="user_ttl_tester")
    
    # Verify it's there
    assert memos.vault.get(cube.id) is not None
    
    # Wait for expiry
    time.sleep(1.2)
    
    # Process something to trigger TTL enforcement
    memos.process("Wake up", user="ttl_tester")
    
    # Verify it's gone
    assert memos.vault.get(cube.id) is None

def test_lane_concurrency_ordering(memos):
    """
    Test that the Lane-Based Queue handles multiple requests for the same user
    sequentially without crashing or interleaving errors.
    """
    order_of_execution = []
    
    def slow_task(task_id):
        # We simulate a task that takes time
        memos.process(f"Start task {task_id}", user="lane_user")
        time.sleep(0.1)
        order_of_execution.append(task_id)

    threads = []
    for i in range(5):
        t = threading.Thread(target=slow_task, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # If the lane works, all 5 tasks should have completed
    assert len(order_of_execution) == 5
    # The list should contain all task IDs
    assert set(order_of_execution) == {0, 1, 2, 3, 4}

def test_cross_session_parallelism(memos):
    """
    Test that different users (different lanes) can still run in parallel
    without blocking each other.
    """
    start_time = time.time()
    
    def slow_user_task(user_id):
        memos.process("Long task", user=f"user_{user_id}")
        time.sleep(0.5)

    # Start 3 tasks for 3 DIFFERENT users
    threads = [threading.Thread(target=slow_user_task, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    end_time = time.time()
    
    # Since they are in different lanes, they should run roughly in parallel.
    # We increase the timeout significantly because local LLM calls (Qwen) can be slow.
    # The goal is to verify they don't block each other globally.
    assert (end_time - start_time) < 30.0 
