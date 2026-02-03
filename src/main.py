import pygame
from Initial_map import *

pygame.init()

# Surface Setting
WIDTH, HEIGHT = 480, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ai Train Game")

# Color Initial
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

clock = pygame.time.Clock()
running = True

# Create Map Item (Include hole, grid, space)
game_map = Game_map_init(hole_size=60)

# Main Loop
while running:
    # Change this tick will change game speed
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Draw the map
    game_map.draw_grid(screen)
    game_map.draw_hole(screen)

    pygame.display.flip()

pygame.quit()