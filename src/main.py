import os
import pygame
from Initial_map import *
from player import Player
from ai_player import *

pygame.init()

# Surface Setting
WIDTH, HEIGHT = 480, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ai Train Game")

# Color Initial Value
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

clock = pygame.time.Clock()
running = True

# Create Map Item (Include hole, grid, space, startpoint, endpoint, player_initial update)
game_map = Game_map_init(hole_size=60, endpoint_size=60, startpoint_size=60)
start_rect = game_map.get_startpoint_rect()
start_center = start_rect.center if start_rect else (0, 0)
player = Player(start_center[0], start_center[1], grid_size=game_map.hole_size)

agent = AiPlayer(
    rows = game_map.rows,
    cols = game_map.cols,
    alpha = 0.1,
    gamma = 0.9,
    epsilon = 1.0,
    epsilon_decay = 0.995,
    epsilon_min=0.05
)

ai_training = True
ai_player_mode = False
episode_count = 0
font = pygame.font.SysFont(None, 28)

log_file = os.path.join(os.path.dirname(__file__), "result.txt")
goal_log_file = os.path.join(os.path.dirname(__file__), "goal_result.txt")

# Main Loop
while running:
    clock.tick(15)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                ai_training = not ai_training
                ai_player_mode = False
                print("AI Training Mode:", ai_training)

            elif event.key == pygame.K_p:
                ai_player_mode = not ai_player_mode
                ai_training = False
                print("AI Player Mode:", ai_player_mode)

            elif event.key == pygame.K_r:
                player.reset_to_center(start_center)
                print("Player Reset")

            elif not ai_player_mode and not ai_training:
                moved = player.handle_key(event.key, screen.get_rect())
                if moved:
                    result = player.update_tile_score(game_map, start_center)
                    if result is not None:
                        steps, score, steps_x, steps_y = result
                        print(f"Move: steps={steps}, total_score={score}")

                    result = player.check_endpoint_and_reset(game_map.get_endpoint_rect(), start_center)
                    if result is not None:
                        steps, score, steps_x, steps_y = result
                        print(f"Goal Reached: total_score={score}")

    if ai_training:
        info = agent.train_step(player, game_map, screen.get_rect(), start_center)
        if info["done"]:
            episode_count += 1
            episode_result = info.get("episode_result")

            if episode_result is not None:
                event_name, steps, score, steps_x, steps_y = episode_result
            else:
                event_name = info.get("event", "unknown")
                score = player.score

            log_line = (
                f"Episode: {episode_count}, "
                f"Event: {event_name}, "
                f"Total Score: {score}, "
                f"Epsilon: {agent.epsilon:.4f}\n"
            )

            print(log_line.strip())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
                f.flush()

            if event_name == "goal":
                goal_log_line = (
                    f"Episode: {episode_count}, "
                    f"Steps: {steps}, "
                    f"Steps_X: {steps_x}, "
                    f"Steps_Y: {steps_y}, "
                    f"Total Score: {score}, "
                    f"Epsilon: {agent.epsilon:.4f}\n"
                )
                with open(goal_log_file, "a", encoding="utf-8") as f:
                    f.write(goal_log_line)
                    f.flush()

    elif ai_player_mode:
        info = agent.play_best_step(player, game_map, screen.get_rect(), start_center)
        if info["done"]:
            episode_result = info.get("episode_result")
            if episode_result is not None:
                _, steps, score, steps_x, steps_y = episode_result
                print(f"[Play] Total Score: {score}")
            else:
                print(f"[Play] Total Score: {player.score}")

    screen.fill(WHITE)

    # Draw all things here
    game_map.draw_grid(screen)
    game_map.draw_hole(screen)
    game_map.draw_endpoint(screen)
    game_map.draw_startpoint(screen)
    player.draw(screen)

    mode_text = "Manual"
    if ai_training:
        mode_text = "Training"
    elif ai_player_mode:
        mode_text = "AI Play"

    info_surface = font.render(
        f"Mode: {mode_text} | Episode: {episode_count} | Epsilon: {agent.epsilon:.3f}",
        True,
        BLACK
    )

    screen.blit(info_surface, (10, 10))
    pygame.display.flip()

pygame.quit()
