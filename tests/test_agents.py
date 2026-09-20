import pathlib
import sys
import unittest

import pygame


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Q_Learning import QLearning
from SARSA import SARSA
from player import Player


class FixedMap:
    """Small deterministic map used to verify the agents' map-aware logic."""

    hole_size = 60

    def __init__(self, map_data):
        self.map_data = map_data
        self.rows = len(map_data)
        self.cols = len(map_data[0])
        self.start_position = next(
            (row, col)
            for row, values in enumerate(map_data)
            for col, value in enumerate(values)
            if value == 3
        )
        self.white_positions = [
            (row, col)
            for row, values in enumerate(map_data)
            for col, value in enumerate(values)
            if value == 0
        ]
        self.white_position_to_index = {
            position: index for index, position in enumerate(self.white_positions)
        }
        self.white_tile_count = len(self.white_positions)

    def get_tile_value_at(self, pos):
        col = int(pos[0] // self.hole_size)
        row = int(pos[1] // self.hole_size)
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None
        return self.map_data[row][col]

    def get_white_tile_index_at(self, pos):
        col = int(pos[0] // self.hole_size)
        row = int(pos[1] // self.hole_size)
        return self.white_position_to_index.get((row, col))


class AgentLogicTests(unittest.TestCase):
    def setUp(self):
        # The hole at (1, 0) must block both entering it and diagonally
        # cutting from (1, 1) to the start at (0, 0).
        self.game_map = FixedMap(
            [
                [3, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )

    def test_agents_mask_holes_and_blocked_diagonals(self):
        state = (1, 1, 0)
        for agent_class in (QLearning, SARSA):
            agent = agent_class(rows=3, cols=3)
            valid_actions = agent.get_valid_actions(state, self.game_map)

            self.assertNotIn(6, valid_actions)  # left: enters the hole
            self.assertNotIn(7, valid_actions)  # up-left: cuts past the hole
            self.assertIn(1, valid_actions)  # up-right remains traversable

    def test_q_learning_bootstrap_ignores_masked_action(self):
        agent = QLearning(rows=3, cols=3, alpha=0.5, gamma=0.9)
        state = (2, 1, 0)
        next_state = (1, 1, 0)
        next_q_values = agent.get_q_values(next_state)
        next_q_values[6] = 100.0  # invalid: left into the hole
        next_q_values[2] = 4.0  # valid: right

        agent.update(state, 0, 0.0, next_state, done=False, game_map=self.game_map)

        self.assertAlmostEqual(agent.get_q_values(state)[0], 1.8)

    def test_learning_rate_decreases_to_a_floor(self):
        state = (0, 0, 0)
        next_state = (0, 1, 0)

        for agent_class in (QLearning, SARSA):
            agent = agent_class(rows=3, cols=3, alpha=0.5, alpha_min=0.1)
            if agent_class is QLearning:
                agent.update(state, 2, 10.0, next_state, done=True)
            else:
                agent.update(state, 2, 10.0, next_state, None, done=True)

            self.assertAlmostEqual(agent.get_q_values(state)[2], 5.0)
            self.assertLess(agent.get_learning_rate(state, 2), agent.alpha)

            for _ in range(40):
                if agent_class is QLearning:
                    agent.update(state, 2, 10.0, next_state, done=True)
                else:
                    agent.update(state, 2, 10.0, next_state, None, done=True)

            self.assertEqual(agent.get_learning_rate(state, 2), agent.alpha_min)

    def test_max_steps_counts_successful_moves_not_diagonal_distance(self):
        player = Player(90, 90, grid_size=60)
        bounds = pygame.Rect(0, 0, 180, 180)
        self.assertTrue(player.step(1, 1, bounds))
        self.assertEqual(player.moves, 1)
        self.assertEqual(player.steps, 2)

        agent = QLearning(
            rows=3,
            cols=3,
            max_steps_per_episode=1,
            max_steps_penalty=-150,
        )
        reward, done, event = agent.check_max_steps(player, 0.0, False, "move")
        self.assertEqual(reward, -150)
        self.assertEqual(player.score, -150)
        self.assertTrue(done)
        self.assertEqual(event, "max_steps")

    def test_configurable_rewards_favor_coverage_and_completion(self):
        game_map = FixedMap([[3, 0]])
        player = Player(
            30,
            30,
            grid_size=game_map.hole_size,
            revisit_penalty=0,
            move_reward=-1,
            cover_reward=10,
            completion_reward=300,
        )
        bounds = pygame.Rect(0, 0, 120, 60)

        self.assertTrue(player.step(1, 0, bounds, game_map))
        self.assertEqual(player.observe_tile(game_map), (9, False, "cover"))
        self.assertTrue(player.step(-1, 0, bounds, game_map))
        self.assertEqual(player.observe_tile(game_map), (299, True, "coverage_complete"))

    def test_sarsa_executes_the_action_used_in_its_target(self):
        game_map = FixedMap(
            [
                [3, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        agent = SARSA(
            rows=3,
            cols=3,
            epsilon=0.0,
            exploration_strategy="greedy",
        )
        start_state = (0, 0, 0)
        first_next_state = (0, 1, 1)
        agent.get_q_values(start_state)[2] = 10.0  # move right first
        agent.get_q_values(first_next_state)[4] = 10.0  # then move down

        player = Player(30, 30, grid_size=game_map.hole_size)
        bounds = pygame.Rect(0, 0, 180, 180)

        first = agent.train_step(player, game_map, bounds, (30, 30))
        self.assertEqual(first["action"], 2)
        self.assertEqual(first["next_action"], 4)

        # A newly better action must not replace the action already used in
        # the SARSA update target for this transition.
        agent.get_q_values(first_next_state)[2] = 100.0
        second = agent.train_step(player, game_map, bounds, (30, 30))
        self.assertEqual(second["action"], 4)


if __name__ == "__main__":
    unittest.main()
