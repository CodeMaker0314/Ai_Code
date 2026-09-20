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

    def __init__(
        self,
        x,
        y,
        size=40,
        grid_size=60,
        revisit_penalty=-5,
        move_reward=-1,
        cover_reward=8,
        completion_reward=150,
        hole_penalty=-100,
        out_of_bounds_penalty=-200,
    ):
        self.rect = pygame.Rect(x, y, size, size)
        self.grid_size = grid_size
        self.moves = 0
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
        self.move_reward = move_reward
        self.cover_reward = cover_reward
        self.completion_reward = completion_reward
        self.hole_penalty = hole_penalty
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self._snap_to_grid()

    def _snap_to_grid(self):
        half = self.grid_size / 2
        grid_x = round((self.rect.centerx - half) / self.grid_size)
        grid_y = round((self.rect.centery - half) / self.grid_size)
        self.rect.centerx = int(grid_x * self.grid_size + half)
        self.rect.centery = int(grid_y * self.grid_size + half)

    def _is_valid_diagonal_move(self, dx, dy, game_map):
        # Only validate diagonal moves (both dx and dy are non-zero)
        if dx == 0 or dy == 0:
            return True
        
        # Get current grid position
        half = self.grid_size / 2
        current_grid_x = int((self.rect.centerx - half) / self.grid_size)
        current_grid_y = int((self.rect.centery - half) / self.grid_size)
        
        # Determine which adjacent cells to check based on diagonal direction
        # For example, moving (1, 1) requires checking (1, 0) and (0, 1)
        adjacent_cells = [
            (current_grid_x + dx, current_grid_y),      # horizontal neighbor
            (current_grid_x, current_grid_y + dy),      # vertical neighbor
        ]
        
        # Check if either adjacent cell has a hole
        # If both adjacent cells have holes, it forms one of the two patterns
        has_horizontal_hole = False
        has_vertical_hole = False
        
        for adj_x, adj_y in adjacent_cells:
            if 0 <= adj_x < game_map.cols and 0 <= adj_y < game_map.rows:
                tile = game_map.map_data[adj_y][adj_x]
                if tile == 1:  # 1 means hole
                    if adj_x == current_grid_x + dx:
                        has_horizontal_hole = True
                    else:
                        has_vertical_hole = True
        
        # Block diagonal movement if there are holes in adjacent cells
        # This prevents bypassing the two hole arrangement patterns
        if has_horizontal_hole or has_vertical_hole:
            return False
        
        return True

    def step(self, dx, dy, bounds_rect, game_map=None):
        if not (dx or dy):
            return False

        # Validate diagonal moves against hole patterns
        if game_map is not None and not self._is_valid_diagonal_move(dx, dy, game_map):
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
        self.moves += 1
        self.steps_x += abs(dx)
        self.steps_y += abs(dy)
        self.steps = self.steps_x + self.steps_y
        self.journey_started = True
        self.journey_completed = False
        return True

    def handle_key(self, key, bounds_rect, game_map=None):
        dx, dy = self.KEY_MOVES.get(key, (0, 0))
        return self.step(dx, dy, bounds_rect, game_map)

    def reset_to_center(self, center_pos, journey_completed=False):
        self.rect.center = center_pos
        self.moves = 0
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
            "moves": self.moves,
            "steps": self.steps,
            "score": self.score,
            "steps_x": self.steps_x,
            "steps_y": self.steps_y,
            "covered_white_tiles": self.covered_white_tiles,
        }

    def observe_tile(self, game_map):
        tile = game_map.get_tile_value_at(self.rect.center)

        if tile is None:
            self.score += self.out_of_bounds_penalty
            self.last_event = "out"
            return self.out_of_bounds_penalty, True, "out"

        if tile == 1:
            self.score += self.hole_penalty
            self.last_event = "hole"
            return self.hole_penalty, True, "hole"

        reward = self.move_reward
        event = "move"

        if tile == 0:
            white_index = game_map.get_white_tile_index_at(self.rect.center)
            if white_index is not None:
                white_bit = 1 << white_index
                if not (self.visited_white_mask & white_bit):
                    self.visited_white_mask |= white_bit
                    self.covered_white_tiles += 1
                    reward += self.cover_reward
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
            reward += self.completion_reward
            self.score += reward
            self.last_event = "coverage_complete"
            return reward, True, "coverage_complete"

        self.score += reward
        self.last_event = event
        return reward, False, event

    def draw(self, surface):
        pygame.draw.circle(surface, (0, 0, 255), self.rect.center, self.rect.width // 2)
