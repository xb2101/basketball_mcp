import anthropic
import json
import sys

sys.path.append('.')
from data import (TRAINING_DATA, HYPERPARAMETERS, PROJECT_INFO,
                   COURT_INFO, TRAINING_ISSUES, INFRASTRUCTURE)

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_round_stats",
        "description": "Get training statistics for a specific round (1, 2, or 3)",
        "input_schema": {
            "type": "object",
            "properties": {
                "round_number": {
                    "type": "integer",
                    "description": "The training round number (1, 2, or 3)"
                }
            },
            "required": ["round_number"]
        }
    },
    {
        "name": "compare_rounds",
        "description": "Compare performance and results across all three training rounds",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_hyperparameters",
        "description": "Get the PPO hyperparameters used during training",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_agent_info",
        "description": "Get information about a specific agent - either 'defender' or 'scorer'",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "The agent name - either 'defender' or 'scorer'"
                }
            },
            "required": ["agent"]
        }
    },
    {
        "name": "get_limitations",
        "description": "Get the known limitations and future work for the project",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_project_info",
        "description": "Get general information about the project including author, date, institution and links",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_court_info",
        "description": "Get information about the basketball court environment, dimensions, robot specs and action space",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_training_issues",
        "description": "Get known bugs and issues encountered during training",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_infrastructure",
        "description": "Get information about the training infrastructure including HPC cluster, training speed and framework",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

def call_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_round_stats":
        round_number = tool_input.get("round_number")
        if round_number not in TRAINING_DATA:
            return f"Round {round_number} does not exist. Valid rounds are 1, 2, and 3."
        return json.dumps(TRAINING_DATA[round_number], indent=2)
    
    elif tool_name == "compare_rounds":
        comparison = {}
        for round_num, data in TRAINING_DATA.items():
            comparison[f"round_{round_num}"] = {
                "agent": data["agent"],
                "steps": data["steps"],
                "model": data["model"],
                "success_rate": data["success_rate"],
                "avg_reward": data["avg_reward"],
                "notes": data["notes"]
            }
        return json.dumps(comparison, indent=2)
    
    elif tool_name == "get_hyperparameters":
        return json.dumps(HYPERPARAMETERS, indent=2)
    
    elif tool_name == "get_agent_info":
        agent = tool_input.get("agent", "").lower()
        if agent == "defender":
            return json.dumps({
                "goal": "Position itself between scorer and goal at all times",
                "best_model": "defender_hpc_v6_final",
                "training_round": 1,
                "success_rate": "~95%",
                "known_issue": "Oscillates near blocking point due to absence of physical barriers",
            }, indent=2)
        elif agent == "scorer":
            return json.dumps({
                "goal": "Navigate from spawn position to paint area within 1.0m of goal at (5.0, 0.0)",
                "best_model": "scorer_ppo_hpc_v7_final",
                "training_round": 2,
                "success_rate": "100% without defender",
                "known_issue": "Cannot evade defender due to sim-to-real gap",
            }, indent=2)
        else:
            return f"Unknown agent '{agent}'. Valid options are 'defender' or 'scorer'."
    
    elif tool_name == "get_limitations":
        return json.dumps({
            "primary_limitation": "Sim-to-real gap",
            "explanation": "No real physics in Python simulation. Collision penalties are scalar values scorer learns to ignore.",
            "future_work": [
                "Gazebo training on HPC using Apptainer",
                "Curriculum learning",
                "Better reward design"
            ]
        }, indent=2)

    elif tool_name == "get_project_info":
        return json.dumps(PROJECT_INFO, indent=2)
    
    elif tool_name == "get_court_info":
        return json.dumps(COURT_INFO, indent=2)
    
    elif tool_name == "get_training_issues":
        return json.dumps(TRAINING_ISSUES, indent=2)
    
    elif tool_name == "get_infrastructure":
        return json.dumps(INFRASTRUCTURE, indent=2)
    
    return f"Unknown tool: {tool_name}"

def chat(user_message: str, conversation_history: list) -> str:
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system="You are an expert on Xavier Beltran's Multi-Agent RL Basketball Simulation project. Use the available tools to answer questions accurately. Always use tools when they can provide relevant data.",
        tools=tools,
        messages=conversation_history
    )
    
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_result = call_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result
                })
        
        conversation_history.append({"role": "assistant", "content": response.content})
        conversation_history.append({"role": "user", "content": tool_results})
        
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system="You are an expert on Xavier Beltran's Multi-Agent RL Basketball Simulation project. Use the available tools to answer questions accurately. Always use tools when they can provide relevant data.",
            tools=tools,
            messages=conversation_history
        )
    
    assistant_message = response.content[0].text
    conversation_history.append({"role": "assistant", "content": assistant_message})
    return assistant_message