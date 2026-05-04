import os
import pygame
from datetime import datetime
from Initial_map import *
from player import Player
from Q_Learning import QLearning
from SARSA import SARSA

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

DEFAULT_ALGORITHM = "q_learning"
TRAINING_SEQUENCE = ["q_learning", "sarsa"]


def create_agent(algorithm_name, game_map):
    agent_class = QLearning if algorithm_name == "q_learning" else SARSA
    return agent_class(
        rows=game_map.rows,
        cols=game_map.cols,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )


def create_game_map():
    return Game_map_init(hole_size=60, endpoint_size=60, startpoint_size=60)


def get_start_center(game_map):
    start_rect = game_map.get_startpoint_rect()
    return start_rect.center if start_rect else (0, 0)

# Create Map Item (Include hole, grid, space, startpoint, endpoint, player_initial update)
game_map = create_game_map()
start_center = get_start_center(game_map)
player = Player(start_center[0], start_center[1], grid_size=game_map.hole_size)

training_round_index = 0
training_cycle_count = 1
algorithm_name = TRAINING_SEQUENCE[training_round_index]
agent = create_agent(algorithm_name, game_map)
print(f"Training Round 1: {algorithm_name}")

ai_training = True
ai_player_mode = False
episode_count = 0
global_episode_count = 0
font = pygame.font.SysFont(None, 28)

result_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(result_dir, exist_ok=True)
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
session_date_short = datetime.now().strftime("%y%m%d")
current_episode_path = []
epsilon_min_path_saved = False


def reset_training_state(player, start_center):
    player.reset_to_center(start_center)
    return [], False


def advance_training_round(player):
    global training_round_index, training_cycle_count, algorithm_name, agent, episode_count, game_map, start_center

    training_round_index = (training_round_index + 1) % len(TRAINING_SEQUENCE)
    if training_round_index == 0:
        training_cycle_count += 1
        game_map = create_game_map()
        start_center = get_start_center(game_map)

    algorithm_name = TRAINING_SEQUENCE[training_round_index]
    agent = create_agent(algorithm_name, game_map)
    episode_count = 0
    player.reset_to_center(start_center)
    print(f"Training Round {training_cycle_count}: {algorithm_name}")


def get_step_log_file_path():
    return os.path.join(
        result_dir,
        f"{training_cycle_count}times_{algorithm_name}_step_{session_date_short}_step.txt"
    )


def get_final_log_file_path():
    return os.path.join(
        result_dir,
        f"{training_cycle_count}times_{algorithm_name}_final_{session_date_short}_final.txt"
    )


def format_path_map(game_map, path_states):
    path_positions = set(path_states)
    lines = []

    for row in range(game_map.rows):
        row_cells = []
        for col in range(game_map.cols):
            tile = game_map.map_data[row][col]
            position = (row, col)

            if tile == 3:
                cell = "S"
            elif tile == 2:
                cell = "G"
            elif tile == 1:
                cell = "H"
            elif position in path_positions:
                cell = "*"
            else:
                cell = "."

            row_cells.append(cell)

        lines.append(" ".join(row_cells))

    return "\n".join(lines)


# Main Loop
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                ai_training = not ai_training
                ai_player_mode = False
                current_episode_path = []
                print("AI Training Mode:", ai_training)

            elif event.key == pygame.K_p:
                ai_player_mode = not ai_player_mode
                ai_training = False
                current_episode_path = []
                print("AI Player Mode:", ai_player_mode)

            elif event.key == pygame.K_r:
                player.reset_to_center(start_center)
                current_episode_path = []
                print("Player Reset")

            elif event.key == pygame.K_q:
                training_cycle_count = 1
                training_round_index = 0
                algorithm_name = "q_learning"
                game_map = create_game_map()
                start_center = get_start_center(game_map)
                agent = create_agent(algorithm_name, game_map)
                episode_count = 0
                current_episode_path, epsilon_min_path_saved = reset_training_state(player, start_center)
                print("Algorithm:", algorithm_name)

            elif event.key == pygame.K_s:
                training_cycle_count = 1
                training_round_index = 1
                algorithm_name = "sarsa"
                game_map = create_game_map()
                start_center = get_start_center(game_map)
                agent = create_agent(algorithm_name, game_map)
                episode_count = 0
                current_episode_path, epsilon_min_path_saved = reset_training_state(player, start_center)
                print("Algorithm:", algorithm_name)

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
        state = info.get("state")
        next_state = info.get("next_state")

        if state is not None and not current_episode_path:
            current_episode_path.append(state)

        if next_state is not None and (not current_episode_path or current_episode_path[-1] != next_state):
            current_episode_path.append(next_state)

        if info["done"]:
            episode_count += 1
            global_episode_count += 1
            episode_result = info.get("episode_result")

            if episode_result is not None:
                event_name, steps, score, steps_x, steps_y = episode_result
            else:
                event_name = info.get("event", "unknown")
                score = player.score

            goal_marker = " [GOAL]" if event_name == "goal" else ""
            log_line = (
                f"Round: {training_cycle_count}, "
                f"Phase: {training_round_index + 1}/{len(TRAINING_SEQUENCE)}, "
                f"Algorithm: {algorithm_name}, "
                f"Episode: {episode_count}, "
                f"Global Episode: {global_episode_count}, "
                f"Event: {event_name}, "
                f"Total Score: {score}, "
                f"Epsilon: {agent.epsilon:.4f}"
                f"{goal_marker}\n"
            )

            print(log_line.strip())
            with open(get_step_log_file_path(), "a", encoding="utf-8") as f:
                f.write(log_line)

            if (
                not epsilon_min_path_saved
                and agent.epsilon <= agent.epsilon_min
                and current_episode_path
            ):
                final_lines = [
                    f"Round: {training_cycle_count}",
                    f"Phase: {training_round_index + 1}/{len(TRAINING_SEQUENCE)}",
                    f"Algorithm: {algorithm_name}",
                    f"Episode: {episode_count}",
                    f"Global Episode: {global_episode_count}",
                    f"Event: {event_name}",
                    f"Total Score: {score}",
                    f"Epsilon: {agent.epsilon:.4f}",
                ]

                if episode_result is not None:
                    final_lines.extend([
                        f"Steps: {steps}",
                        f"Steps_X: {steps_x}",
                        f"Steps_Y: {steps_y}",
                    ])

                path_map = format_path_map(game_map, current_episode_path)
                final_text = (
                    "\n".join(final_lines)
                    + f"\nPath Length: {len(current_episode_path)}\n"
                    + "Legend: S=Start, G=Goal, H=Hole, *=Path, .=Empty\n\n"
                    + f"{path_map}\n\n"
                )

                with open(get_final_log_file_path(), "w", encoding="utf-8") as f:
                    f.write(final_text)

                epsilon_min_path_saved = True

            current_episode_path = []

            if agent.epsilon <= agent.epsilon_min:
                advance_training_round(player)
                current_episode_path = []
                epsilon_min_path_saved = False

    elif ai_player_mode:
        info = agent.play_best_step(player, game_map, screen.get_rect(), start_center)
        if info["done"]:
            episode_result = info.get("episode_result")
            if episode_result is not None:
                _, steps, score, steps_x, steps_y = episode_result
                print(f"[Play] Total Score: {score}")
            else:
                print(f"[Play] Total Score: {player.score}")

    else:
        current_episode_path = []

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

    algorithm_label = "Q-Learning" if algorithm_name == "q_learning" else "SARSA"
    info_surface = font.render(
        f"Round: {training_cycle_count} | Phase: {training_round_index + 1}/2 | Algo: {algorithm_label} | Mode: {mode_text} | Episode: {episode_count} | Epsilon: {agent.epsilon:.3f}",
        True,
        BLACK
    )

    screen.blit(info_surface, (10, 10))
    pygame.display.flip()

pygame.quit()
