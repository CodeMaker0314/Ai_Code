import pygame
import random

class QLearning:

    actions = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0)
    ]

    def __init__(self, rows, cols, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.rows = rows
        self.cols = cols
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = {}
        self.last_episode_result = None

    def get_state_from_player(self, player):
        col = int(player.rect.centerx // player.grid_size)
        row = int(player.rect.centery // player.grid_size)
        return row, col

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(len(self.actions))]
        return self.q_table[state]

    def get_valid_actions(self, state):
        return [
            action_index for action_index in range(len(self.actions))
            if self.is_valid_state(state, action_index)
        ]

    def choose_action(self, state):
        q_values = self.get_q_values(state)

        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))

        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def action_to_move(self, action_index):
        return self.actions[action_index]

    def get_next_state(self, state, action_index):
        row, col = state
        dx, dy = self.action_to_move(action_index)
        next_row = row + dy
        next_col = col + dx
        return next_row, next_col

    def is_valid_state(self, state, action_index):
        next_row, next_col = self.get_next_state(state, action_index)
        return 0 <= next_row < self.rows and 0 <= next_col < self.cols

    def choose_action_with_bounds(self, state):
        valid_actions = self.get_valid_actions(state)

        if not valid_actions:
            return 0

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.get_q_values(state)
        max_q = max(q_values[action] for action in valid_actions)
        best_actions = [action for action in valid_actions if q_values[action] == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q_values(state)[action]

        if done:
            target_q = reward
        else:
            valid_actions = self.get_valid_actions(next_state)
            next_q_values = self.get_q_values(next_state)
            next_max_q = max(next_q_values[action_index] for action_index in valid_actions) if valid_actions else 0.0
            target_q = reward + self.gamma * next_max_q

        self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_reward_and_done(self, game_map, player, start_center):
        tile = game_map.get_tile_value_at(player.rect.center)

        if tile is None:
            return -100, True, "out"

        if tile == 1:
            return -25, True, "hole"

        if tile == 2:
            return 50, True, "goal"

        return -1, False, "move"

    def apply_environment_result(self, game_map, player, start_center):
        result = player.update_tile_score(game_map, start_center)
        if result is not None:
            steps, score, steps_x, steps_y = result
            self.last_episode_result = ("hole", steps, score, steps_x, steps_y)
            return self.last_episode_result

        result = player.check_endpoint_and_reset(game_map.get_endpoint_rect(), start_center)
        if result is not None:
            steps, score, steps_x, steps_y = result
            self.last_episode_result = ("goal", steps, score, steps_x, steps_y)
            return self.last_episode_result

        self.last_episode_result = None
        return None

    def train_step(self, player, game_map, bounds_rect, start_center):
        state = self.get_state_from_player(player)
        action = self.choose_action_with_bounds(state)
        dx, dy = self.action_to_move(action)

        moved = player.step(dx, dy, bounds_rect)

        if not moved:
            reward = -5
            next_state = state
            done = False
            self.update(state, action, reward, next_state, done)
            return {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "event": "blocked",
            }

        reward, done, event = self.get_reward_and_done(game_map, player, start_center)
        next_state_before_reset = self.get_state_from_player(player)

        self.update(state, action, reward, next_state_before_reset, done)

        episode_result = self.apply_environment_result(game_map, player, start_center)
        if done:
            self.decay_epsilon()

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state_before_reset,
            "done": done,
            "event": event,
            "episode_result": episode_result,
        }

    def get_best_action(self, state):
        valid_actions = self.get_valid_actions(state)
        if not valid_actions:
            return 0

        q_values = self.get_q_values(state)
        max_q = max(q_values[action] for action in valid_actions)
        best_actions = [action for action in valid_actions if q_values[action] == max_q]
        return random.choice(best_actions)

    def play_best_step(self, player, game_map, bounds_rect, start_center):
        state = self.get_state_from_player(player)
        action = self.get_best_action(state)
        dx, dy = self.action_to_move(action)

        moved = player.step(dx, dy, bounds_rect)
        if not moved:
            return {
                "state": state,
                "action": action,
                "reward": -5,
                "done": False,
                "event": "blocked",
            }

        reward, done, event = self.get_reward_and_done(game_map, player, start_center)
        episode_result = self.apply_environment_result(game_map, player, start_center)

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "event": event,
            "episode_result": episode_result,
        }
