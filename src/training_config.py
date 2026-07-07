def get_training_config():
    return {
        "exploration_strategy": "softmax",
        "temperature": 0.6,
        "softmax_temperature": 0.6,
        "alpha": 0.35,
        "gamma": 0.97,
        "epsilon": 1.0,
        "epsilon_decay": 0.98,
        "epsilon_min": 0.05,
        "blocked_penalty": -12,
        "distance_reward_factor": 1.0,
        "revisit_penalty": -4,
        "post_completion_runs": 10,
    }
