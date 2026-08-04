# Basketball RL MCP Chatbot

An AI chatbot powered by an MCP (Model Context Protocol) server that answers questions about my Reinforcement Learning Graduate Project. Unlike a RAG system that searches documents, this uses structured tool functions that Claude dynamically selects and calls to retrieve precise project data.

## Live Demo
https://basketballmcp-znyihwsjejzsuudrrhyzkl.streamlit.app/

## How it works
- `server.py` defines MCP tools exposing basketball RL training data
- `data.py` contains structured project data including training rounds, hyperparameters, and infrastructure details
- `utils.py` handles tool calling and Claude API communication
- Claude figures out which tool to call based on the user's question and returns accurate data

## Tech Stack
- Anthropic API (Claude)
- MCP / FastMCP
- Streamlit
- Python

## Related Project
This chatbot is built on top of my Multi-Agent RL Basketball Simulation:
https://github.com/xb2101/basketball-defender-rl

## How to run locally
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Set your Anthropic API key: export ANTHROPIC_API_KEY=your-key-here
4. Run the Streamlit app: streamlit run streamlit_app.py
5. Or run the terminal version: python client.py