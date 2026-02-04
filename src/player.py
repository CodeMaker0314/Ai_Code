import pygame

class Player:
    # Player Initial Here
    def __init__(self, x, y, size=40, grid_size=60):
        self.rect = pygame.Rect(x, y, size, size)
        self.grid_size = grid_size
        self.steps = 0
        self.steps_x = 0
        self.steps_y = 0
        self.score = 0
        self._snap_to_grid()

    # Make player can do collection with grid
    def _snap_to_grid(self):
        half = self.grid_size / 2
        grid_x = round((self.rect.centerx - half) / self.grid_size)
        grid_y = round((self.rect.centery - half) / self.grid_size)
        self.rect.centerx = int(grid_x * self.grid_size + half)
        self.rect.centery = int(grid_y * self.grid_size + half)

    # Make player move advance_cmd here(up, right, down, left)
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

        # If keybind or pos in wrong way will pass this key_cmd
        if target_cx < min_cx or target_cx > max_cx or target_cy < min_cy or target_cy > max_cy:
            return False

        self.rect.centerx = target_cx
        self.rect.centery = target_cy
        self.steps_x += abs(dx)
        self.steps_y += abs(dy)
        self.steps = self.steps_x + self.steps_y
        return True

    # Key command input here(w, a, s, d, up, left, down, right)
    def handle_key(self, key, bounds_rect):
        dx = (key in (pygame.K_RIGHT, pygame.K_d)) - (key in (pygame.K_LEFT, pygame.K_a))
        dy = (key in (pygame.K_DOWN, pygame.K_s)) - (key in (pygame.K_UP, pygame.K_w))
        return self.step(dx, dy, bounds_rect)

    #Game reset(let every value back to the initial) function here
    def reset_to_center(self, center_pos):
        self.rect.center = center_pos
        self.steps = 0
        self.steps_x = 0
        self.steps_y = 0
        self.score = 0
        self._snap_to_grid()

    # Check endpoint collection、reset and return step、score
    def check_endpoint_and_reset(self, endpoint_rect, reset_center):
        if endpoint_rect and self.rect.colliderect(endpoint_rect):
            steps = self.steps
            steps_x = self.steps_x
            steps_y = self.steps_y
            score = self.score
            self.reset_to_center(reset_center)
            return steps, score, steps_x, steps_y
        return None

    # Update new score and step， count the score(like hole -50, endpoint +50, normal_state +1)
    def update_tile_score(self, game_map, reset_center):
        tile = game_map.get_tile_value_at(self.rect.center)
        if tile is None:
            return None
        self.score += 1
        if tile == 1:
            self.score -= 50
            steps = self.steps
            steps_x = self.steps_x
            steps_y = self.steps_y
            score = self.score
            self.reset_to_center(reset_center)
            return steps, score, steps_x, steps_y
        if tile == 2:
            self.score += 50
            return None
        return None

    def draw(self, surface, color=(230, 90, 80)):
        pygame.draw.rect(surface, color, self.rect)
