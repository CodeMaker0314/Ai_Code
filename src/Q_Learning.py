import random


class QLearning:
    actions = [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ]

    def __init__(
        self,
        rows,
        cols,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        blocked_penalty=-10,
    ):
        self.rows = rows
        self.cols = cols
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.blocked_penalty = blocked_penalty
        self.q_table = {}
        self.last_episode_result = None

    def get_position_from_player(self, player):
        col = int(player.rect.centerx // player.grid_size)
        row = int(player.rect.centery // player.grid_size)
        return row, col

    def make_state(self, position, visited_mask):
        return position[0], position[1], visited_mask

    def get_state_from_player(self, player):
        position = self.get_position_from_player(player)
        return self.make_state(position, player.visited_white_mask)

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(len(self.actions))]
        return self.q_table[state]

    def get_valid_actions(self, state):
        return [
            action_index
            for action_index in range(len(self.actions))
            if self.is_valid_state(state, action_index)
        ]

    def action_to_move(self, action_index):
        return self.actions[action_index]

    def get_next_state(self, state, action_index):
        row, col, visited_mask = state
        dx, dy = self.action_to_move(action_index)
        next_row = row + dy
        next_col = col + dx
        return next_row, next_col, visited_mask

    def is_valid_state(self, state, action_index):
        next_row, next_col, _ = self.get_next_state(state, action_index)
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

    def get_reward_and_done(self, player, game_map):
        return player.observe_tile(game_map)

    def capture_episode_result(self, player, event, game_map):
        result = {
            "event": event,
            "steps": player.steps,
            "score": player.score,
            "steps_x": player.steps_x,
            "steps_y": player.steps_y,
            "covered_white_tiles": player.covered_white_tiles,
            "total_white_tiles": game_map.white_tile_count,
        }
        self.last_episode_result = result
        return result

    def train_step(self, player, game_map, bounds_rect, start_center):
        position = self.get_position_from_player(player)
        state = self.make_state(position, player.visited_white_mask)

        action = self.choose_action_with_bounds(state)
        dx, dy = self.action_to_move(action)

        moved = player.step(dx, dy, bounds_rect)

        if not moved:
            reward = self.blocked_penalty
            next_state = self.make_state(position, player.visited_white_mask)
            done = False
            self.update(state, action, reward, next_state, done)
            return {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "event": "blocked",
                "episode_result": None,
            }

        reward, done, event = self.get_reward_and_done(player, game_map)
        next_position = self.get_position_from_player(player)
        next_state_before_reset = self.make_state(next_position, player.visited_white_mask)

        self.update(state, action, reward, next_state_before_reset, done)

        episode_result = None
        if done:
            episode_result = self.capture_episode_result(player, event, game_map)
            self.decay_epsilon()
            player.reset_to_center(start_center, journey_completed=True)

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
                "reward": self.blocked_penalty,
                "done": False,
                "event": "blocked",
                "episode_result": None,
            }

        reward, done, event = self.get_reward_and_done(player, game_map)
        episode_result = None
        if done:
            episode_result = self.capture_episode_result(player, event, game_map)
            player.reset_to_center(start_center, journey_completed=True)

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "event": event,
            "episode_result": episode_result,
        }
