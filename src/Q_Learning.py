import math
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
        alpha=0.2,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        blocked_penalty=-15,
        max_steps_per_episode=None,
        exploration_strategy="softmax",
        temperature=1.0,
        reward_shaping=True,
        distance_reward_factor=0.5,
        alpha_min=0.05,
        return_distance_reward_factor=None,
        max_steps_penalty=-20,
    ):
        self.rows = rows
        self.cols = cols
        self.alpha = alpha
        self.alpha_min = min(max(0.0, alpha_min), alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.blocked_penalty = blocked_penalty
        self.max_steps_per_episode = max_steps_per_episode
        self.exploration_strategy = exploration_strategy
        self.temperature = max(0.1, temperature)
        self.reward_shaping = reward_shaping
        self.distance_reward_factor = distance_reward_factor
        self.return_distance_reward_factor = (
            distance_reward_factor * 0.5
            if return_distance_reward_factor is None
            else return_distance_reward_factor
        )
        self.max_steps_penalty = max_steps_penalty
        self.q_table = {}
        self.state_action_visits = {}
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

    def get_valid_actions(self, state, game_map=None):
        return [
            action_index
            for action_index in range(len(self.actions))
            if self.is_valid_state(state, action_index, game_map)
        ]

    def action_to_move(self, action_index):
        return self.actions[action_index]

    def get_next_state(self, state, action_index):
        row, col, visited_mask = state
        dx, dy = self.action_to_move(action_index)
        next_row = row + dy
        next_col = col + dx
        return next_row, next_col, visited_mask

    def is_valid_state(self, state, action_index, game_map=None):
        next_row, next_col, _ = self.get_next_state(state, action_index)
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            return False

        if game_map is None:
            return True

        # The map is fully observable, so a training action should never
        # deliberately enter a hole.  Keeping this mask in the agent also
        # makes the bootstrap target agree with Player.step().
        if game_map.map_data[next_row][next_col] == 1:
            return False

        dx, dy = self.action_to_move(action_index)
        if dx and dy:
            row, col, _ = state
            if (
                game_map.map_data[row][col + dx] == 1
                or game_map.map_data[row + dy][col] == 1
            ):
                return False

        return True

    def choose_softmax_action(self, state, valid_actions):
        q_values = self.get_q_values(state)
        max_q = max(q_values[action] for action in valid_actions)
        weights = [math.exp((q_values[action] - max_q) / self.temperature) for action in valid_actions]
        total = sum(weights)
        if total == 0:
            return random.choice(valid_actions)
        return random.choices(valid_actions, weights=weights, k=1)[0]

    def choose_action_with_bounds(self, state, game_map=None):
        valid_actions = self.get_valid_actions(state, game_map)

        if not valid_actions:
            return 0

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        if self.exploration_strategy == "softmax":
            return self.choose_softmax_action(state, valid_actions)

        q_values = self.get_q_values(state)
        max_q = max(q_values[action] for action in valid_actions)
        best_actions = [action for action in valid_actions if q_values[action] == max_q]
        return random.choice(best_actions)

    def get_learning_rate(self, state, action):
        """Return a high initial rate that stabilizes as a pair is revisited."""
        visits = self.state_action_visits.get((state, action), 0)
        return max(self.alpha_min, self.alpha / math.sqrt(visits + 1))

    def update(self, state, action, reward, next_state, done, game_map=None):
        current_q = self.get_q_values(state)[action]

        if done:
            target_q = reward
        else:
            valid_actions = self.get_valid_actions(next_state, game_map)
            next_q_values = self.get_q_values(next_state)
            next_max_q = max(next_q_values[action_index] for action_index in valid_actions) if valid_actions else 0.0
            target_q = reward + self.gamma * next_max_q

        learning_rate = self.get_learning_rate(state, action)
        self.q_table[state][action] = current_q + learning_rate * (target_q - current_q)
        self.state_action_visits[(state, action)] = self.state_action_visits.get((state, action), 0) + 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_reward_and_done(self, player, game_map):
        return player.observe_tile(game_map)

    def distance_to_closest_unvisited(self, position, visited_mask, game_map):
        unvisited_positions = [
            pos
            for index, pos in enumerate(game_map.white_positions)
            if not (visited_mask >> index) & 1
        ]
        if not unvisited_positions:
            return None
        row, col = position
        return min(max(abs(row - ur), abs(col - uc)) for ur, uc in unvisited_positions)

    def shape_reward(self, state, next_state, reward, done, game_map):
        if not self.reward_shaping or done:
            return reward
        # preferential shaping towards unvisited tiles
        distance_before = self.distance_to_closest_unvisited((state[0], state[1]), state[2], game_map)
        distance_after = self.distance_to_closest_unvisited((next_state[0], next_state[1]), next_state[2], game_map)
        if distance_before is not None and distance_after is not None:
            reward += self.distance_reward_factor * (distance_before - distance_after)

        # if all white tiles are visited, encourage returning to start
        white_count = game_map.white_tile_count
        if white_count > 0:
            all_mask = (1 << white_count) - 1
            visited_before = state[2] & all_mask
            visited_after = next_state[2] & all_mask
            if visited_before == all_mask or visited_after == all_mask:
                # compute Chebyshev distance to startpoint
                start_row, start_col = game_map.start_position
                row_b, col_b = state[0], state[1]
                row_a, col_a = next_state[0], next_state[1]
                dist_before = max(abs(row_b - start_row), abs(col_b - start_col))
                dist_after = max(abs(row_a - start_row), abs(col_a - start_col))
                # give small reward when getting closer to start
                reward += self.return_distance_reward_factor * (dist_before - dist_after)

        return reward

    def check_max_steps(self, player, reward, done, event):
        if (
            not done
            and self.max_steps_per_episode is not None
            and player.moves >= self.max_steps_per_episode
        ):
            penalty = self.max_steps_penalty
            reward += penalty
            player.score += penalty
            done = True
            event = "max_steps"
            player.last_event = event
        return reward, done, event

    def capture_episode_result(self, player, event, game_map):
        result = {
            "event": event,
            "moves": player.moves,
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

        action = self.choose_action_with_bounds(state, game_map)
        dx, dy = self.action_to_move(action)

        moved = player.step(dx, dy, bounds_rect, game_map)

        if not moved:
            reward = self.blocked_penalty
            next_state = self.make_state(position, player.visited_white_mask)
            done = False
            self.update(state, action, reward, next_state, done, game_map)
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
        reward = self.shape_reward(state, next_state_before_reset, reward, done, game_map)
        reward, done, event = self.check_max_steps(player, reward, done, event)

        self.update(state, action, reward, next_state_before_reset, done, game_map)

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

    def get_best_action(self, state, game_map=None):
        valid_actions = self.get_valid_actions(state, game_map)
        if not valid_actions:
            return 0

        q_values = self.get_q_values(state)
        max_q = max(q_values[action] for action in valid_actions)
        best_actions = [action for action in valid_actions if q_values[action] == max_q]
        return random.choice(best_actions)

    def play_best_step(self, player, game_map, bounds_rect, start_center):
        state = self.get_state_from_player(player)
        action = self.get_best_action(state, game_map)
        dx, dy = self.action_to_move(action)

        moved = player.step(dx, dy, bounds_rect, game_map)
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
        next_position = self.get_position_from_player(player)
        next_state_before_reset = self.make_state(next_position, player.visited_white_mask)
        reward = self.shape_reward(state, next_state_before_reset, reward, done, game_map)
        reward, done, event = self.check_max_steps(player, reward, done, event)

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
