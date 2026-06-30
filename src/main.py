import os
from datetime import datetime

import pygame

from Initial_map import Game_map_init
from Q_Learning import QLearning
from SARSA import SARSA
from path_image import draw_path_image, get_path_image_file_path
from player import Player
from replay import DEFAULT_REPLAY_FPS, ReplayPlayback, ReplayStore

pygame.init()

WIDTH, HEIGHT = 480, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ai Train Game")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

clock = pygame.time.Clock()
running = True

TRAINING_SEQUENCE = ["sarsa", "q_learning"]
EXPLORATION_STRATEGY = "softmax"
SOFTMAX_TEMPERATURE = 0.8
REVISIT_PENALTY = -6
DISTANCE_REWARD_FACTOR = 0.8


def create_agent(algorithm_name, game_map):
    agent_class = QLearning if algorithm_name == "q_learning" else SARSA
    return agent_class(
        rows=game_map.rows,
        cols=game_map.cols,
        alpha=0.25,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.02,
        blocked_penalty=-18,
        max_steps_per_episode=game_map.rows * game_map.cols * 3,
        exploration_strategy=EXPLORATION_STRATEGY,
        temperature=SOFTMAX_TEMPERATURE,
        reward_shaping=True,
        distance_reward_factor=DISTANCE_REWARD_FACTOR,
    )


def create_game_map():
    new_game_map = Game_map_init(hole_size=60, startpoint_size=60)
    print(f"Map Complexity: {new_game_map.map_complexity:.4f}")
    return new_game_map


def get_start_center(game_map):
    start_rect = game_map.get_startpoint_rect()
    return start_rect.center if start_rect else (0, 0)


def state_to_position(state):
    if state is None:
        return None
    if isinstance(state, tuple) and len(state) >= 2:
        return state[0], state[1]
    return None


def reset_training_state(player, start_center):
    player.reset_to_center(start_center)
    return []


def format_path_map(game_map, path_states):
    path_positions = set()
    for state in path_states:
        position = state_to_position(state)
        if position is not None:
            path_positions.add(position)

    lines = []
    for row in range(game_map.rows):
        row_cells = []
        for col in range(game_map.cols):
            tile = game_map.map_data[row][col]
            position = (row, col)

            if tile == 3:
                cell = "S"
            elif tile == 1:
                cell = "H"
            elif position in path_positions:
                cell = "*"
            else:
                cell = "."

            row_cells.append(cell)
        lines.append(" ".join(row_cells))

    return "\n".join(lines)


def is_better_coverage_result(candidate, current_best):
    if current_best is None:
        return True

    if candidate["steps"] != current_best["steps"]:
        return candidate["steps"] < current_best["steps"]

    if candidate["score"] != current_best["score"]:
        return candidate["score"] > current_best["score"]

    return candidate["path_length"] < current_best["path_length"]


def write_best_coverage_result(best_result):
    final_lines = [
        f"Round: {best_result['training_cycle_count']}",
        f"Phase: {best_result['training_round_index'] + 1}/{len(TRAINING_SEQUENCE)}",
        f"Algorithm: {best_result['algorithm_name']}",
        f"Episode: {best_result['episode_count']}",
        f"Global Episode: {best_result['global_episode_count']}",
        f"Event: {best_result['event_name']}",
        f"Total Score: {best_result['score']}",
        f"Epsilon: {best_result['epsilon']:.4f}",
        f"Steps: {best_result['steps']}",
        f"Steps_X: {best_result['steps_x']}",
        f"Steps_Y: {best_result['steps_y']}",
        f"Covered White Tiles: {best_result['covered_white_tiles']}/{best_result['total_white_tiles']}",
        f"Selected From Completion Runs: {POST_COMPLETION_RUNS}",
    ]

    path_map = format_path_map(best_result["game_map"], best_result["path"])
    final_text = (
        "\n".join(final_lines)
        + f"\nPath Length: {best_result['path_length']}\n"
        + "Legend: S=Start, H=Hole, *=Path, .=Empty\n\n"
        + f"{path_map}\n\n"
    )

    with open(get_final_log_file_path(), "w", encoding="utf-8") as f:
        f.write(final_text)

    path_image_path = get_path_image_file_path(get_final_log_file_path())
    draw_path_image(best_result["game_map"], best_result["path"], path_image_path)
    print(f"Path image saved: {os.path.basename(path_image_path)}")


def get_step_log_file_path():
    return os.path.join(
        result_dir,
        f"{training_cycle_count}times_{algorithm_name}_step_{session_date_short}_step.txt",
    )


def get_final_log_file_path():
    return os.path.join(
        result_dir,
        f"{training_cycle_count}times_{algorithm_name}_final_{session_date_short}_final.txt",
    )


def build_completed_result(
    event_name,
    score,
    steps,
    steps_x,
    steps_y,
    covered_white_tiles,
    total_white_tiles,
):
    return {
        "training_cycle_count": training_cycle_count,
        "training_round_index": training_round_index,
        "algorithm_name": algorithm_name,
        "episode_count": episode_count,
        "global_episode_count": global_episode_count,
        "event_name": event_name,
        "score": score,
        "epsilon": agent.epsilon,
        "steps": steps,
        "steps_x": steps_x,
        "steps_y": steps_y,
        "path": list(current_episode_path),
        "path_length": len(current_episode_path),
        "covered_white_tiles": covered_white_tiles,
        "total_white_tiles": total_white_tiles,
        "game_map": game_map,
    }


def save_completion_replay(completed_result):
    replay_file_path = replay_store.save_completed_round(completed_result)
    print(f"Replay saved: {os.path.basename(replay_file_path)}")
    return replay_file_path


def start_latest_replay():
    global ai_training, ai_player_mode, current_episode_path

    replay_data, replay_file_path = replay_store.load_latest()
    if replay_data is None:
        print("No replay files found.")
        return

    try:
        replay_playback.start(
            replay_data,
            now_ms=pygame.time.get_ticks(),
        )
    except ValueError as error:
        print(f"Replay failed: {error}")
        return

    ai_training = False
    ai_player_mode = False
    current_episode_path = []
    print(f"Replay Mode: {os.path.basename(replay_file_path)}")


def advance_training_round(player):
    global training_round_index, training_cycle_count, algorithm_name, agent, episode_count, game_map, start_center, post_completion_results

    training_round_index = (training_round_index + 1) % len(TRAINING_SEQUENCE)
    if training_round_index == 0:
        training_cycle_count += 1
        game_map = create_game_map()
        start_center = get_start_center(game_map)

    algorithm_name = TRAINING_SEQUENCE[training_round_index]
    agent = create_agent(algorithm_name, game_map)
    episode_count = 0
    post_completion_results = []
    player.reset_to_center(start_center)
    print(f"Training Round {training_cycle_count}: {algorithm_name}")


game_map = create_game_map()
start_center = get_start_center(game_map)
player = Player(
    start_center[0],
    start_center[1],
    grid_size=game_map.hole_size,
    revisit_penalty=REVISIT_PENALTY,
)

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
replay_dir = os.path.join(os.path.dirname(__file__), "replays")
replay_store = ReplayStore(replay_dir, fps=DEFAULT_REPLAY_FPS)
replay_playback = ReplayPlayback(fps=DEFAULT_REPLAY_FPS)
session_date_short = datetime.now().strftime("%y%m%d")
current_episode_path = []
post_completion_results = []
POST_COMPLETION_RUNS = 10


while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                replay_playback.clear()
                ai_training = not ai_training
                ai_player_mode = False
                current_episode_path = []
                print("AI Training Mode:", ai_training)

            elif event.key == pygame.K_p:
                replay_playback.clear()
                ai_player_mode = not ai_player_mode
                ai_training = False
                current_episode_path = []
                print("AI Player Mode:", ai_player_mode)

            elif event.key == pygame.K_r:
                replay_playback.clear()
                player.reset_to_center(start_center)
                current_episode_path = []
                print("Player Reset")

            elif event.key == pygame.K_v:
                start_latest_replay()

            elif event.key == pygame.K_q:
                replay_playback.clear()
                training_cycle_count = 1
                training_round_index = 0
                algorithm_name = "q_learning"
                game_map = create_game_map()
                start_center = get_start_center(game_map)
                agent = create_agent(algorithm_name, game_map)
                episode_count = 0
                current_episode_path = reset_training_state(player, start_center)
                post_completion_results = []
                print("Algorithm:", algorithm_name)

            elif event.key == pygame.K_s:
                replay_playback.clear()
                training_cycle_count = 1
                training_round_index = 1
                algorithm_name = "sarsa"
                game_map = create_game_map()
                start_center = get_start_center(game_map)
                agent = create_agent(algorithm_name, game_map)
                episode_count = 0
                current_episode_path = reset_training_state(player, start_center)
                post_completion_results = []
                print("Algorithm:", algorithm_name)

            elif event.key == pygame.K_ESCAPE and replay_playback.is_showing:
                replay_playback.clear()
                print("Replay stopped")

            elif not replay_playback.is_showing and not ai_player_mode and not ai_training:
                moved = player.handle_key(event.key, screen.get_rect())
                if moved:
                    reward, done, tile_event = player.observe_tile(game_map)
                    print(
                        f"Move: reward={reward}, event={tile_event}, "
                        f"covered={player.covered_white_tiles}/{game_map.white_tile_count}"
                    )

                    if done:
                        episode_result = player._capture_episode_result()
                        print(
                            f"Episode Complete: event={tile_event}, "
                            f"score={episode_result['score']}, "
                            f"steps={episode_result['steps']}, "
                            f"covered={episode_result['covered_white_tiles']}/{game_map.white_tile_count}"
                        )
                        player.reset_to_center(start_center, journey_completed=True)

    if ai_training:
        info = agent.train_step(player, game_map, screen.get_rect(), start_center)
        state = info.get("state")
        next_state = info.get("next_state")

        if state is not None and (not current_episode_path or current_episode_path[-1] != state):
            current_episode_path.append(state)

        if next_state is not None and (not current_episode_path or current_episode_path[-1] != next_state):
            current_episode_path.append(next_state)

        if info["done"]:
            episode_count += 1
            global_episode_count += 1
            episode_result = info.get("episode_result")

            if episode_result is not None:
                event_name = episode_result.get("event", info.get("event", "unknown"))
                steps = episode_result["steps"]
                score = episode_result["score"]
                steps_x = episode_result["steps_x"]
                steps_y = episode_result["steps_y"]
                covered_white_tiles = episode_result["covered_white_tiles"]
                total_white_tiles = episode_result["total_white_tiles"]
            else:
                event_name = info.get("event", "unknown")
                score = player.score
                steps = player.steps
                steps_x = player.steps_x
                steps_y = player.steps_y
                covered_white_tiles = player.covered_white_tiles
                total_white_tiles = game_map.white_tile_count

            completion_marker = " [COMPLETE]" if event_name == "coverage_complete" else ""
            log_line = (
                f"Round: {training_cycle_count}, "
                f"Phase: {training_round_index + 1}/{len(TRAINING_SEQUENCE)}, "
                f"Algorithm: {algorithm_name}, "
                f"Episode: {episode_count}, "
                f"Global Episode: {global_episode_count}, "
                f"Event: {event_name}, "
                f"Total Score: {score}, "
                f"Covered White Tiles: {covered_white_tiles}/{total_white_tiles}, "
                f"Epsilon: {agent.epsilon:.4f}"
                f"{completion_marker}\n"
            )

            print(log_line.strip())
            with open(get_step_log_file_path(), "a", encoding="utf-8") as f:
                f.write(log_line)

            completed_result = None
            if event_name == "coverage_complete" and episode_result is not None and current_episode_path:
                completed_result = build_completed_result(
                    event_name,
                    score,
                    steps,
                    steps_x,
                    steps_y,
                    covered_white_tiles,
                    total_white_tiles,
                )
                save_completion_replay(completed_result)

            if (
                agent.epsilon <= agent.epsilon_min
                and event_name == "coverage_complete"
                and episode_result is not None
                and completed_result is not None
            ):
                post_completion_results.append(completed_result)

            current_episode_path = []

            if len(post_completion_results) >= POST_COMPLETION_RUNS:
                best_coverage_result = None
                for coverage_result in post_completion_results:
                    if is_better_coverage_result(coverage_result, best_coverage_result):
                        best_coverage_result = coverage_result

                write_best_coverage_result(best_coverage_result)
                advance_training_round(player)
                current_episode_path = []

    elif ai_player_mode:
        info = agent.play_best_step(player, game_map, screen.get_rect(), start_center)
        if info["done"]:
            episode_result = info.get("episode_result")
            if episode_result is not None:
                print(
                    f"[Play] Total Score: {episode_result['score']}, "
                    f"Covered White Tiles: {episode_result['covered_white_tiles']}/{episode_result['total_white_tiles']}"
                )
            else:
                print(f"[Play] Total Score: {player.score}")

    elif replay_playback.is_showing:
        replay_playback.update(pygame.time.get_ticks())

    else:
        current_episode_path = []

    screen.fill(WHITE)
    if replay_playback.is_showing:
        replay_playback.draw(screen)
        info_text = replay_playback.get_status_text()
    else:
        game_map.draw_grid(screen)
        game_map.draw_hole(screen)
        game_map.draw_startpoint(screen, player.get_startpoint_status())
        player.draw(screen)

        mode_text = "Manual"
        if ai_training:
            mode_text = "Training"
        elif ai_player_mode:
            mode_text = "AI Play"

        algorithm_label = "Q-Learning" if algorithm_name == "q_learning" else "SARSA"
        info_text = (
            f"Round: {training_cycle_count} | Phase: {training_round_index + 1}/{len(TRAINING_SEQUENCE)} | "
            f"Algo: {algorithm_label} | Mode: {mode_text} | Episode: {episode_count} | "
            f"Covered: {player.covered_white_tiles}/{game_map.white_tile_count} | Epsilon: {agent.epsilon:.3f}"
        )

    info_surface = font.render(info_text, True, BLACK)

    screen.blit(info_surface, (10, 10))
    pygame.display.flip()

pygame.quit()
