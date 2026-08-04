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