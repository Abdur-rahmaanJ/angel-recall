import os
from flask import Flask, render_template, request, jsonify
from .. import MemOS, create_memory_agent, AccessScope, SemanticType
from langchain_core.messages import HumanMessage
import litellm

app = Flask(__name__)

# Initialize MemOS with a default local model
# Users can change this via the dashboard
VAULT_PATH = os.path.abspath("./dashboard_vault")
memos = MemOS(persist_directory=VAULT_PATH, model="ollama/qwen2.5:0.5b")
agent = create_memory_agent(memos)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['POST'])
def set_config():
    data = request.json
    api_keys = data.get('api_keys', {}) # dict of env var names to values
    model = data.get('model')
    
    for key, value in api_keys.items():
        if value:
            os.environ[key] = value
            
    if model:
        memos.reader.model = model
        # Re-create agent with new model if necessary
        global agent
        agent = create_memory_agent(memos, model=model)
        
    return jsonify({"status": "success", "model": memos.reader.model})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    user = data.get('user', 'alice')
    
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=message)],
            "user": user
        })
        response_text = result['messages'][-1].content
        return jsonify({
            "response": response_text,
            "model": memos.reader.model
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/memories', methods=['GET'])
def get_memories():
    memories = []
    # memos.vault.kv_store contains all MemCubes
    for cid, cube in memos.vault.kv_store.items():
        memories.append(cube.to_dict())
    
    # Sort by timestamp desc
    memories.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(memories)

@app.route('/api/memories/clear', methods=['POST'])
def clear_memories():
    import shutil
    if os.path.exists(VAULT_PATH):
        shutil.rmtree(VAULT_PATH)
    
    global memos, agent
    memos = MemOS(persist_directory=VAULT_PATH, model=memos.reader.model)
    agent = create_memory_agent(memos)
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
