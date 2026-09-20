def get_training_config():
    return {
        "exploration_strategy": "softmax",
        "temperature": 0.45,
        # Learn new state/action pairs quickly, then converge with the
        # per-visit schedule implemented by QLearning and SARSA.
        "alpha": 0.55,
        "alpha_min": 0.08,
        # Keep the completion reward meaningful while planning a full route.
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.994,
        "epsilon_min": 0.03,
        # A completed route always covers the same white tiles.  Therefore
        # every successful move costs one point, making shorter routes better.
        "move_reward": -1,
        "cover_reward": 10,
        "revisit_penalty": 0,
        "completion_reward": 300,
        "hole_penalty": -150,
        "out_of_bounds_penalty": -250,
        "blocked_penalty": -20,
        "max_steps_penalty": -150,
        # Keep guidance subtle before completion; use a stronger signal to
        # return directly to the start once all white tiles are covered.
        "distance_reward_factor": 0.25,
        "return_distance_reward_factor": 1.0,
        # Limit actual successful moves, not horizontal/vertical distance.
        "max_steps_multiplier": 4,
        "min_max_steps": 240,
        "post_completion_runs": 10,
    }
