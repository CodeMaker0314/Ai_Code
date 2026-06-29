import pygame
import random


class Game_map_init:
    MIN_MAP_COMPLEXITY = 0.9
    MAX_MAP_COMPLEXITY = 1.0
    MIN_HOLE_COUNT = 11
    MAX_HOLE_COUNT = 20

    def __init__(self, hole_size=60, startpoint_size=60):
        self.hole_size = hole_size
        self.startpoint_size = startpoint_size
        self.rows = 8
        self.cols = 8

        self.start_position = (0, 0)
        self.map_complexity = 0
        self.white_positions = []
        self.white_position_to_index = {}
        self.white_tile_count = 0
        self.map_data = self.generate_random_map()

        self.black = (0, 0, 0)
        self.gray = (200, 200, 200)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.gold = (255, 215, 0)

    def calculate_map_complexity(self, map_data):
        hole_positions = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if map_data[row][col] == 1
        ]

        total_holes = len(hole_positions)
        if total_holes == 0:
            return 0

        hole_set = set(hole_positions)
        surrounding_hole_count = 0

        for row, col in hole_positions:
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue

                    neighbor = (row + row_offset, col + col_offset)
                    if neighbor in hole_set:
                        surrounding_hole_count += 1

        return surrounding_hole_count / total_holes

    def _build_random_map(self, hole_count):
        map_data = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        start_row, start_col = self.start_position
        map_data[start_row][start_col] = 3

        available_positions = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) != self.start_position
        ]

        for row, col in random.sample(available_positions, hole_count):
            map_data[row][col] = 1

        return map_data

    def _refresh_tile_cache(self):
        self.white_positions = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if self.map_data[row][col] == 0
        ]
        self.white_position_to_index = {
            position: index for index, position in enumerate(self.white_positions)
        }
        self.white_tile_count = len(self.white_positions)

    def _get_reachable_positions(self):
        start_row, start_col = self.start_position
        queue = [(start_row, start_col)]
        visited = {(start_row, start_col)}

        while queue:
            row, col = queue.pop(0)
            for row_offset, col_offset in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                next_row = row + row_offset
                next_col = col + col_offset
                if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
                    continue
                if self.map_data[next_row][next_col] == 1:
                    continue

                next_position = (next_row, next_col)
                if next_position not in visited:
                    visited.add(next_position)
                    queue.append(next_position)

        return visited

    def _white_tiles_are_reachable(self):
        reachable_positions = self._get_reachable_positions()
        return all(position in reachable_positions for position in self.white_positions)

    def generate_random_map(self, hole_count=None):
        use_random_hole_count = hole_count is None

        while True:
            if use_random_hole_count:
                hole_count = random.randint(self.MIN_HOLE_COUNT, self.MAX_HOLE_COUNT)

            self.map_data = self._build_random_map(hole_count)
            self.map_complexity = self.calculate_map_complexity(self.map_data)
            self._refresh_tile_cache()

            if self.MIN_MAP_COMPLEXITY < self.map_complexity < self.MAX_MAP_COMPLEXITY and self._white_tiles_are_reachable():
                break

        return self.map_data

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

    def draw_startpoint(self, screen, status="not_started"):
        colors = {
            "not_started": self.green,
            "started": self.red,
            "completed": self.gold,
        }
        color = colors.get(status, self.green)

        for row in range(self.rows):
            for col in range(self.cols):
                if self.map_data[row][col] == 3:
                    x = col * self.startpoint_size
                    y = row * self.startpoint_size
                    rect = pygame.Rect(x, y, self.startpoint_size, self.startpoint_size)
                    pygame.draw.rect(screen, color, rect)

    def _find_tile_rect(self, tile_value, tile_size):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.map_data[row][col] == tile_value:
                    x = col * tile_size
                    y = row * tile_size
                    return pygame.Rect(x, y, tile_size, tile_size)
        return None

    def get_startpoint_rect(self):
        return self._find_tile_rect(3, self.startpoint_size)

    def get_tile_value_at(self, pos):
        col = int(pos[0] // self.hole_size)
        row = int(pos[1] // self.hole_size)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        return self.map_data[row][col]

    def get_white_tile_index_at(self, pos):
        col = int(pos[0] // self.hole_size)
        row = int(pos[1] // self.hole_size)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        return self.white_position_to_index.get((row, col))
