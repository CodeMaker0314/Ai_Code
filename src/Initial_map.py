import pygame

class Game_map_init:
    def __init__(self, hole_size=60, endpoint_size=60, startpoint_size=60):
        self.hole_size = hole_size
        self.endpoint_size = endpoint_size
        self.startpoint_size = startpoint_size
        self.rows = 8
        self.cols = 8

        # startpoint = 3, endpoint = 2, hole = 1, space = 0
        self.map_data = [
            [3, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 2],
        ]

        # Color Initial
        self.black = (0, 0, 0)
        self.gray = (200, 200, 200)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)

    # Draw grid
    def draw_grid(self, screen):
        width = self.cols * self.hole_size
        height = self.rows * self.hole_size

        for x in range(0, width, self.hole_size):
            pygame.draw.line(screen, self.gray, (x, 0), (x, height))
        for y in range(0, height, self.hole_size):
            pygame.draw.line(screen, self.gray, (0, y), (width, y))

    # Draw hole
    def draw_hole(self, screen):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map_data[row][col] == 1:
                    x = col * self.hole_size
                    y = row * self.hole_size
                    rect = pygame.Rect(x, y, self.hole_size, self.hole_size)
                    pygame.draw.rect(screen, self.black, rect)

    # Draw endpoint
    def draw_endpoint(self, screen):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map_data[row][col] == 2:
                    x = col * self.endpoint_size
                    y = row * self.endpoint_size
                    rect = pygame.Rect(x, y, self.endpoint_size, self.endpoint_size)
                    pygame.draw.rect(screen, self.green, rect)

    # Draw startpoint
    def draw_startpoint(self, screen):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map_data[row][col] == 3:
                    x = col * self.startpoint_size
                    y = row * self.startpoint_size
                    rect = pygame.Rect(x, y, self.startpoint_size, self.startpoint_size)
                    pygame.draw.rect(screen, self.blue, rect)

    # Initial the title setting(posion, rect_size)
    def _find_tile_rect(self, tile_value, tile_size):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map_data[row][col] == tile_value:
                    x = col * tile_size
                    y = row * tile_size
                    return pygame.Rect(x, y, tile_size, tile_size)
        return None

    # Endpoint rect size Initial
    def get_endpoint_rect(self):
        return self._find_tile_rect(2, self.endpoint_size)

    # Startpoint rect size Initial
    def get_startpoint_rect(self):
        return self._find_tile_rect(3, self.startpoint_size)

    def get_tile_value_at(self, pos):
        col = int(pos[0] // self.hole_size)
        row = int(pos[1] // self.hole_size)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        return self.map_data[row][col]
