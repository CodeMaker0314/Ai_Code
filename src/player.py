import pygame


class Player:
    KEY_MOVES = {
        pygame.K_UP: (0, -1),
        pygame.K_w: (0, -1),
        pygame.K_RIGHT: (1, 0),
        pygame.K_d: (1, 0),
        pygame.K_DOWN: (0, 1),
        pygame.K_s: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_a: (-1, 0),
        pygame.K_PAGEUP: (1, -1),
        pygame.K_KP9: (1, -1),
        pygame.K_PAGEDOWN: (1, 1),
        pygame.K_KP3: (1, 1),
        pygame.K_HOME: (-1, -1),
        pygame.K_KP7: (-1, -1),
        pygame.K_END: (-1, 1),
        pygame.K_KP1: (-1, 1),
    }

    def __init__(self, x, y, size=40, grid_size=60, revisit_penalty=-5):
        self.rect = pygame.Rect(x, y, size, size)
        self.grid_size = grid_size
        self.steps = 0
        self.steps_x = 0
        self.steps_y = 0
        self.score = 0
        self.visited_white_mask = 0
        self.covered_white_tiles = 0
        self.last_event = None
        self.journey_started = False
        self.journey_completed = False
        self.revisit_penalty = revisit_penalty
        self._snap_to_grid()

    def _snap_to_grid(self):
        half = self.grid_size / 2
        grid_x = round((self.rect.centerx - half) / self.grid_size)
        grid_y = round((self.rect.centery - half) / self.grid_size)
        self.rect.centerx = int(grid_x * self.grid_size + half)
        self.rect.centery = int(grid_y * self.grid_size + half)

    def step(self, dx, dy, bounds_rect):
        if not (dx or dy):
            return False

        self._snap_to_grid()
        half = self.grid_size / 2
        target_cx = self.rect.centerx + int(dx * self.grid_size)
        target_cy = self.rect.centery + int(dy * self.grid_size)
        min_cx = bounds_rect.left + half
        max_cx = bounds_rect.right - half
        min_cy = bounds_rect.top + half
        max_cy = bounds_rect.bottom - half

        if target_cx < min_cx or target_cx > max_cx or target_cy < min_cy or target_cy > max_cy:
            return False

        self.rect.centerx = target_cx
        self.rect.centery = target_cy
        self.steps_x += abs(dx)
        self.steps_y += abs(dy)
        self.steps = self.steps_x + self.steps_y
        self.journey_started = True
        self.journey_completed = False
        return True

    def handle_key(self, key, bounds_rect):
        dx, dy = self.KEY_MOVES.get(key, (0, 0))
        return self.step(dx, dy, bounds_rect)

    def reset_to_center(self, center_pos, journey_completed=False):
        self.rect.center = center_pos
        self.steps = 0
        self.steps_x = 0
        self.steps_y = 0
        self.score = 0
        self.visited_white_mask = 0
        self.covered_white_tiles = 0
        self.last_event = None
        self.journey_started = False
        self.journey_completed = journey_completed
        self._snap_to_grid()

    def get_startpoint_status(self):
        if self.journey_completed:
            return "completed"
        if self.journey_started:
            return "started"
        return "not_started"

    def _capture_episode_result(self):
        return {
            "steps": self.steps,
            "score": self.score,
            "steps_x": self.steps_x,
            "steps_y": self.steps_y,
            "covered_white_tiles": self.covered_white_tiles,
        }

    def observe_tile(self, game_map):
        tile = game_map.get_tile_value_at(self.rect.center)

        if tile is None:
            self.score -= 200
            self.last_event = "out"
            return -200, True, "out"

        if tile == 1:
            self.score -= 100
            self.last_event = "hole"
            return -100, True, "hole"

        reward = -1
        event = "move"

        if tile == 0:
            white_index = game_map.get_white_tile_index_at(self.rect.center)
            if white_index is not None:
                white_bit = 1 << white_index
                if not (self.visited_white_mask & white_bit):
                    self.visited_white_mask |= white_bit
                    self.covered_white_tiles += 1
                    reward += 8
                    event = "cover"
                else:
                    reward += self.revisit_penalty
                    event = "revisit"
        elif tile == 3:
            event = "start"

        if (
            tile == 3
            and self.journey_started
            and game_map.white_tile_count > 0
            and self.covered_white_tiles >= game_map.white_tile_count
        ):
            reward += 150
            self.score += reward
            self.last_event = "coverage_complete"
            return reward, True, "coverage_complete"

        self.score += reward
        self.last_event = event
        return reward, False, event

    def draw(self, surface):
        pygame.draw.circle(surface, (0, 0, 255), self.rect.center, self.rect.width // 2)
