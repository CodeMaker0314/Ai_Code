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

# Create Map Item (Include hole, grid, space, startpoint, endpoint, player_initial update)
game_map = Game_map_init(hole_size=60, endpoint_size=60, startpoint_size=60)
start_rect = game_map.get_startpoint_rect()
start_center = start_rect.center if start_rect else (0, 0)
player = Player(start_center[0], start_center[1], grid_size=game_map.hole_size)

# Main Loop
while running:
    # Change this tick will change game speed
    dt = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Move command update, count steps and score point
        if event.type == pygame.KEYDOWN:
            moved = player.handle_key(event.key, screen.get_rect())
            if moved:
                result = player.update_tile_score(game_map, start_center)
                if result is not None:
                    steps, score, steps_x, steps_y = result
                    print(f"Steps: {steps}, Score: {score}")
                result = player.check_endpoint_and_reset(game_map.get_endpoint_rect(), start_center)
                if result is not None:
                    steps, score, steps_x, steps_y = result
                    print(f"Steps: {steps}, Score: {score}")

    screen.fill(WHITE)

    # Draw all things here
    game_map.draw_grid(screen)
    game_map.draw_hole(screen)
    game_map.draw_endpoint(screen)
    game_map.draw_startpoint(screen)
    player.draw(screen)

    pygame.display.flip()

pygame.quit()
