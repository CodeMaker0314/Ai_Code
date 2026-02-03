import pygame

class Game_map_init:
    def __init__(self, hole_size=60):
        self.hole_size = hole_size
        self.rows = 8
        self.cols = 8

        # hole = 1, space = 0
        self.map_data = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        # Color Initial
        self.black = (0, 0, 0)
        self.gray = (200, 200, 200)

    def draw_grid(self, screen):
        width = self.cols * self.hole_size
        height = self.rows * self.hole_size

        for x in range(0, width, self.hole_size):
            pygame.draw.line(screen, self.gray, (x, 0), (x, height))
        for y in range(0, height, self.hole_size):
            pygame.draw.line(screen, self.gray, (0, y), (width, y))

    def draw_hole(self, screen):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map_data[row][col] == 1:
                    x = col * self.hole_size
                    y = row * self.hole_size
                    rect = pygame.Rect(x, y, self.hole_size, self.hole_size)
                    pygame.draw.rect(screen, self.black, rect)