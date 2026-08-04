from mcp.server.fastmcp import FastMCP
import json

#Initializing the MCP Server
mcp = FastMCP("Basketball RL Training Server")

# Training data for each round
TRAINING_DATA = {
    1: {
        "agent": "defender",
        "steps": 30000000,
        "model": "defender_hpc_v6_final",
        "opponent": "scripted scorer",
        "success_rate": "~95%",
        "avg_reward": 2170,
        "notes": "Defender learned to reach blocking point consistently but oscillates near it"
    },
    2: {
        "agent": "scorer",
        "steps": 10000000,
        "model": "scorer_ppo_hpc_v7_final",
        "opponent": "no defender then frozen defender v6",
        "success_rate": "100% without defender",
        "avg_reward": 450,
        "notes": "Scorer reached paint consistently but could not evade defender due to sim-to-real gap"
    },
    3: {
        "agent": "both",
        "steps": 10000000,
        "model": "defender_r3",
        "opponent": "alternating frozen training",
        "success_rate": "defender worse than v6",
        "avg_reward": None,
        "notes": "Defender overfit to scorer straight line behavior, round 3 defender worse than round 1"
    }
}

#Project Metadata
PROJECT_INFO = {
    "title": "Multi-Agent RL Basketball Simulation",
    "author": "Xavier Beltran",
    "date": "Spring 2026",
    "institution": "NYU Tandon School of Engineering",
    "course": "Reinforcement Learning & Optimal Control",
    "github": "https://github.com/xb2101/basketball-defender-rl",
    "report": "Teaching Robots Basketball: A Multi-Agent Reinforcement Learning Approach"
}

#Court environment on Gazebo
COURT_INFO = {
    "dimensions": "x: 0 to 5 meters, y: -4 to 4 meters",
    "goal_position": "(5.0, 0.0)",
    "paint_radius": "1.0m",
    "blocking_point": "0.6m in front of scorer along scorer-to-goal vector",
    "simulation": "Pure Python environment using Gymnasium API",
    "robot": "TurtleBot3",
    "action_space": "linear velocity [-0.6, 0.6] m/s, angular velocity [-2.0, 2.0] rad/s"
}

HYPERPARAMETERS = {
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "policy": "MLP"
}

#Known bugs encountered during training
TRAINING_ISSUES = [
    "ent_coef=0.0 disabled exploration in early runs",
    "Silent training bug: defender v7 started from scratch instead of v6 checkpoint due to mismatched file naming between local and HPC",
    "Reward hacking: complex reward functions caused robot to stand near paint instead of defending",
    "Oscillation: defender learned to reach blocking point but not hold position"
]

#Training infrastructure
INFRASTRUCTURE = {
    "cluster": "NYU Torch HPC",
    "scheduler": "SLURM",
    "account": "torch_pr_155_general",
    "training_speed_hpc": "~2000-3000 fps",
    "training_speed_local": "~15 fps",
    "time_per_10M_steps": "~90 minutes on HPC",
    "framework": "Stable Baselines3",
    "simulation": "Pure Python with Gymnasium API"
}

@mcp.tool()
def get_project_info() -> str:
    """Get general information about the project including author, date, institution and links"""
    return json.dumps(PROJECT_INFO, indent=2)

@mcp.tool()
def get_court_info() -> str:
    """Get information about the basketball court environment, dimensions, robot specs and action space"""
    return json.dumps(COURT_INFO, indent=2)

@mcp.tool()
def get_training_issues() -> str:
    """Get known bugs and issues encountered during training"""
    return json.dumps(TRAINING_ISSUES, indent=2)

@mcp.tool()
def get_infrastructure() -> str:
    """Get information about the training infrastructure including HPC cluster, training speed and framework"""
    return json.dumps(INFRASTRUCTURE, indent=2)

@mcp.tool()
def get_round_stats(round_number: int) -> str:
    """Get training statistics for a specific round (1, 2, or 3)"""
    if round_number not in TRAINING_DATA:
        return f"Round {round_number} does not exist. Valid rounds are 1, 2, and 3."
    
    data = TRAINING_DATA[round_number]
    return json.dumps(data, indent=2)

@mcp.tool()
def compare_rounds() -> str:
    """Compare performance and results across all three training rounds"""
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

@mcp.tool()
def get_hyperparameters() -> str:
    """Get the PPO hyperparameters used during training"""
    return json.dumps(HYPERPARAMETERS, indent=2)

@mcp.tool()
def get_agent_info(agent: str) -> str:
    """Get information about a specific agent - either 'defender' or 'scorer'"""
    agent = agent.lower()
    if agent == "defender":
        return json.dumps({
            "goal": "Position itself between scorer and goal at all times",
            "best_model": "defender_hpc_v6_final",
            "training_round": 1,
            "success_rate": "~95%",
            "known_issue": "Oscillates near blocking point due to absence of physical barriers",
            "observation_space": ["robot_x", "robot_y", "robot_yaw", "scorer_x", "scorer_y", 
                                 "scorer_vx", "scorer_vy", "goal_x", "goal_y", 
                                 "dist_to_block", "dist_scorer_to_goal"]
        }, indent=2)
    elif agent == "scorer":
        return json.dumps({
            "goal": "Navigate from spawn position to paint area within 1.0m of goal at (5.0, 0.0)",
            "best_model": "scorer_ppo_hpc_v7_final",
            "training_round": 2,
            "success_rate": "100% without defender",
            "known_issue": "Cannot evade defender due to sim-to-real gap",
            "observation_space": ["scorer_x", "scorer_y", "scorer_yaw", "defender_x", 
                                 "defender_y", "goal_x", "goal_y", "dist_to_paint",
                                 "dist_to_defender", "heading_error"]
        }, indent=2)
    else:
        return f"Unknown agent '{agent}'. Valid options are 'defender' or 'scorer'."

@mcp.tool()
def get_limitations() -> str:
    """Get the known limitations and future work for the project"""
    return json.dumps({
        "primary_limitation": "Sim-to-real gap - absence of real physics in Python simulation",
        "explanation": "Robots can occupy same coordinates without physical interaction. Collision penalties are scalar values that scorer learns to ignore in favor of paint reward.",
        "attempted_fixes": [
            "Increased collision penalty from -10 to -20 — scorer learned to flee not evade",
            "Intermediate values (-15) — no meaningful improvement",
            "Collision termination condition — marginal improvement, no evasion learned"
        ],
        "future_work": [
            "Gazebo training on HPC using Apptainer containerization",
            "Curriculum learning — start with weak defender, progressively increase difficulty",
            "Better reward design — lateral movement bonus, bypass bonus for avoiding defender"
        ]
    }, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")