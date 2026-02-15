import pytest
import shutil
import os
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from angel_recall import MemOS, create_memory_agent

# Use a lightweight model for testing if possible, or same as core tests
TEST_MODEL = "gemini/gemini-2.5-flash"
TEST_VAULT = "./test_vault_agent"

@pytest.fixture(scope="module")
def memos():
    if os.path.exists(TEST_VAULT):
        shutil.rmtree(TEST_VAULT)
    
    obj = MemOS(persist_directory=TEST_VAULT, model=TEST_MODEL)
    yield obj
    
    if os.path.exists(TEST_VAULT):
        shutil.rmtree(TEST_VAULT)

@patch("angel_recall.litellm.completion")
def test_agent_retrieval(mock_completion, memos):
    """Test that the agent can retrieve and use stored memories."""
    # Mock the LLM response to include the retrieved information
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Your favorite programming language is Python."
    mock_completion.return_value = mock_response

    # 1. Store a specific fact for Alice
    memos.process("Remember that my favorite programming language is Python.", user="alice")
    
    # 2. Create the agent
    agent = create_memory_agent(memos, model=TEST_MODEL)
    
    # 3. Ask the agent about the fact
    result = agent.invoke({
        "messages": [HumanMessage(content="What is my favorite programming language?")],
        "user": "alice"
    })
    
    response_content = result["messages"][-1].content
    print(f"Agent Response: {response_content}")
    
    # 4. Verify the agent knows the answer (via the mock)
    assert "Python" in response_content
    # Verify that the memory was actually passed to the LLM node in the state
    # The last call to mock_completion should have the memory context in messages
    args, kwargs = mock_completion.call_args
    system_msg = kwargs['messages'][0]['content']
    assert "Python" in system_msg

@patch("angel_recall.litellm.completion")
def test_agent_isolation(mock_completion, memos):
    """Test that the agent respects user isolation."""
    # Mock the LLM response for Bob
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I don't have any information about your secret project."
    mock_completion.return_value = mock_response

    # Alice stores a secret
    memos.process("My secret project name is 'Project Phoenix'.", user="alice")
    
    # Create the agent
    agent = create_memory_agent(memos, model=TEST_MODEL)
    
    # Bob asks about the secret project
    result = agent.invoke({
        "messages": [HumanMessage(content="What is my secret project name?")],
        "user": "bob"
    })
    
    response_content = result["messages"][-1].content
    assert "Phoenix" not in response_content
    
    # Verify that Alice's secret was NOT passed to the LLM for Bob
    args, kwargs = mock_completion.call_args
    system_msg = kwargs['messages'][0]['content']
    assert "Phoenix" not in system_msg

@patch("angel_recall.litellm.completion")
def test_agent_with_custom_tool(mock_completion, memos):
    """Test that the agent can be initialized with custom tools."""
    from langchain_core.tools import tool
    
    @tool
    def get_weather(location: str):
        """Returns the weather for a given location."""
        return f"The weather in {location} is sunny."
    
    # Create the agent with the tool
    # Note: create_memory_agent now accepts a tools parameter
    agent = create_memory_agent(memos, model=TEST_MODEL, tools=[get_weather])
    
    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I see you want to know the weather."
    mock_completion.return_value = mock_response
    
    # Ask the agent
    agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Tokyo?")],
        "user": "alice"
    })
    
    # Verify the completion was called with tools
    args, kwargs = mock_completion.call_args
    assert "tools" in kwargs
    assert len(kwargs["tools"]) == 1
    # Check if the tool is there (it might be converted or stay as a tool object)
    assert "get_weather" in str(kwargs["tools"][0])

@patch("angel_recall.litellm.completion")
def test_agent_with_memory_tools(mock_completion, memos):
    """Test that the agent can be initialized with standard memory tools."""
    from angel_recall import get_memory_tools
    
    # Get standard memory tools for Alice
    memory_tools = get_memory_tools(memos, user="alice")
    
    # Create the agent with these tools
    agent = create_memory_agent(memos, model=TEST_MODEL, tools=memory_tools)
    
    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I will search your memory."
    mock_completion.return_value = mock_response
    
    # Ask the agent
    agent.invoke({
        "messages": [HumanMessage(content="Search my memory for 'Python'")],
        "user": "alice"
    })
    
    # Verify completion was called with memory tools
    args, kwargs = mock_completion.call_args
    assert "tools" in kwargs
    # get_memory_tools returns 3 tools: store_memory, search_memory, set_memory_access
    assert len(kwargs["tools"]) == 3
    
    tool_names = [str(t) for t in kwargs["tools"]]
    assert any("search_memory" in name for name in tool_names)
    assert any("store_memory" in name for name in tool_names)
    assert any("set_memory_access" in name for name in tool_names)

