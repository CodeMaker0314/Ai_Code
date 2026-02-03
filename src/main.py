import pygame
from Initial_map import *
from player import Player

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
player = Player(100, 100)

# Main Loop
while running:
    # Change this tick will change game speed
    dt = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    player.update(dt, keys, screen.get_rect())

    screen.fill(WHITE)

    # Draw the map
    game_map.draw_grid(screen)
    game_map.draw_hole(screen)
    player.draw(screen)

    pygame.display.flip()

pygame.quit()
