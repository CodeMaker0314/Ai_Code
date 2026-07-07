import argparse
import json
import os
import sys
from datetime import datetime

import pygame


DEFAULT_REPLAY_FPS = 30
REPLAY_VERSION = 1


class ReplayStore:
    def __init__(self, replay_dir, fps=DEFAULT_REPLAY_FPS):
        self.replay_dir = replay_dir
        self.fps = fps
        os.makedirs(self.replay_dir, exist_ok=True)

    def save_completed_round(self, completed_result):
        replay_data = self._build_replay_data(completed_result)
        file_name = self._build_file_name(replay_data["metadata"])
        file_path = os.path.join(self.replay_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as replay_file:
            json.dump(replay_data, replay_file, indent=2)

        return file_path

    def load_latest(self):
        replay_files = [
            os.path.join(self.replay_dir, file_name)
            for file_name in os.listdir(self.replay_dir)
            if file_name.endswith(".json")
        ]

        if not replay_files:
            return None, None

        latest_file = max(replay_files, key=os.path.getmtime)
        return self.load(latest_file), latest_file

    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as replay_file:
            return json.load(replay_file)

    def _build_replay_data(self, completed_result):
        game_map = completed_result["game_map"]
        frames = [
            self._state_to_frame(state)
            for state in completed_result.get("path", [])
            if state is not None
        ]

        if not frames:
            raise ValueError("Replay cannot be saved without path frames.")

        return {
            "version": REPLAY_VERSION,
            "fps": self.fps,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metadata": {
                "training_cycle_count": completed_result["training_cycle_count"],
                "training_round_index": completed_result["training_round_index"],
                "algorithm_name": completed_result["algorithm_name"],
                "episode_count": completed_result["episode_count"],
                "global_episode_count": completed_result["global_episode_count"],
                "event_name": completed_result["event_name"],
                "score": completed_result["score"],
                "epsilon": completed_result["epsilon"],
                "steps": completed_result["steps"],
                "steps_x": completed_result["steps_x"],
                "steps_y": completed_result["steps_y"],
                "covered_white_tiles": completed_result["covered_white_tiles"],
                "total_white_tiles": completed_result["total_white_tiles"],
                "path_length": completed_result["path_length"],
            },
            "map": {
                "rows": game_map.rows,
                "cols": game_map.cols,
                "tile_size": game_map.hole_size,
                "map_data": [list(row) for row in game_map.map_data],
            },
            "frames": frames,
        }

    def _state_to_frame(self, state):
        row = int(state[0])
        col = int(state[1])
        return {
            "row": row,
            "col": col,
        }

    def _build_file_name(self, metadata):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        algorithm_name = self._safe_name(metadata["algorithm_name"])
        return (
            f"{metadata['training_cycle_count']}times_{algorithm_name}_"
            f"episode_{metadata['episode_count']}_"
            f"global_{metadata['global_episode_count']}_"
            f"{timestamp}_replay.json"
        )

    def _safe_name(self, value):
        return str(value).replace(os.sep, "_").replace(" ", "_")


class ReplayPlayback:
    def __init__(self, fps=DEFAULT_REPLAY_FPS):
        self.default_fps = fps
        self.clear()

    @property
    def is_showing(self):
        return self.replay_data is not None

    def clear(self):
        self.replay_data = None
        self.frames = []
        self.frame_index = 0
        self.fps = self.default_fps
        self.frame_duration_ms = 1000 / self.fps
        self.next_frame_ms = 0
        self.playing = False
        self.finished = False

    def start(self, replay_data, now_ms=0):
        frames = replay_data.get("frames", [])
        if not frames:
            raise ValueError("Replay has no frames.")

        self.replay_data = replay_data
        self.frames = frames
        self.frame_index = 0
        self.fps = replay_data.get("fps", self.default_fps) or self.default_fps
        self.frame_duration_ms = 1000 / self.fps
        self.next_frame_ms = now_ms + self.frame_duration_ms
        self.playing = len(self.frames) > 1
        self.finished = not self.playing

    def restart(self, now_ms=0):
        if not self.is_showing:
            return

        self.frame_index = 0
        self.next_frame_ms = now_ms + self.frame_duration_ms
        self.playing = len(self.frames) > 1
        self.finished = not self.playing

    def pause(self):
        if self.is_showing:
            self.playing = False

    def resume(self, now_ms=0):
        if not self.is_showing or self.finished:
            return

        self.playing = True
        self.next_frame_ms = now_ms + self.frame_duration_ms

    def toggle_pause(self, now_ms=0):
        if self.playing:
            self.pause()
        else:
            self.resume(now_ms)

    def update(self, now_ms):
        if not self.playing:
            return

        while now_ms >= self.next_frame_ms and self.playing:
            self.frame_index += 1
            self.next_frame_ms += self.frame_duration_ms

            if self.frame_index >= len(self.frames) - 1:
                self.frame_index = len(self.frames) - 1
                self.playing = False
                self.finished = True

    def draw(self, surface):
        if not self.is_showing:
            return

        self._draw_grid(surface)
        self._draw_tiles(surface)
        self._draw_player(surface)

    def get_status_text(self):
        if not self.is_showing:
            return ""

        metadata = self.replay_data.get("metadata", {})
        algorithm_name = metadata.get("algorithm_name", "unknown")
        frame_count = len(self.frames)
        state = "Playing" if self.playing else "Finished"
        return (
            f"Replay {self.fps}fps ({state}) | "
            f"Round: {metadata.get('training_cycle_count', '-')} | "
            f"Algo: {algorithm_name} | "
            f"Episode: {metadata.get('episode_count', '-')} | "
            f"Frame: {self.frame_index + 1}/{frame_count}"
        )

    def _draw_grid(self, surface):
        map_info = self.replay_data.get("map", {})
        rows = map_info.get("rows", 0)
        cols = map_info.get("cols", 0)
        tile_size = map_info.get("tile_size", 60)
        width = cols * tile_size
        height = rows * tile_size
        gray = (200, 200, 200)

        for x in range(0, width, tile_size):
            pygame.draw.line(surface, gray, (x, 0), (x, height))
        for y in range(0, height, tile_size):
            pygame.draw.line(surface, gray, (0, y), (width, y))

    def _draw_tiles(self, surface):
        map_info = self.replay_data.get("map", {})
        tile_size = map_info.get("tile_size", 60)
        map_data = map_info.get("map_data", [])
        colors = {
            1: (0, 0, 0),
            3: (0, 255, 0),
        }

        for row, row_data in enumerate(map_data):
            for col, tile in enumerate(row_data):
                color = colors.get(tile)
                if color is None:
                    continue

                rect = pygame.Rect(
                    col * tile_size,
                    row * tile_size,
                    tile_size,
                    tile_size,
                )
                pygame.draw.rect(surface, color, rect)

    def _draw_player(self, surface):
        frame = self.frames[self.frame_index]
        tile_size = self.replay_data.get("map", {}).get("tile_size", 60)
        player_size = 40
        center = (
            int(frame["col"] * tile_size + tile_size / 2),
            int(frame["row"] * tile_size + tile_size / 2),
        )
        rect = pygame.Rect(0, 0, player_size, player_size)
        rect.center = center
        pygame.draw.circle(surface, (0, 0, 255), rect.center, rect.width // 2)


def get_default_replay_dir():
    return os.path.join(os.path.dirname(__file__), "replays")


def load_replay_from_target(target_path, fps=DEFAULT_REPLAY_FPS):
    if target_path is None:
        store = ReplayStore(get_default_replay_dir(), fps=fps)
        return store.load_latest()

    if os.path.isdir(target_path):
        store = ReplayStore(target_path, fps=fps)
        return store.load_latest()

    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"Replay file not found: {target_path}")

    replay_dir = os.path.dirname(target_path) or "."
    store = ReplayStore(replay_dir, fps=fps)
    return store.load(target_path), target_path


def get_replay_window_size(replay_data):
    map_info = replay_data.get("map", {})
    rows = map_info.get("rows", 8)
    cols = map_info.get("cols", 8)
    tile_size = map_info.get("tile_size", 60)
    board_width = cols * tile_size
    board_height = rows * tile_size
    return max(board_width, 900), max(board_height, 520)


def run_standalone_player(target_path=None):
    replay_data, replay_file_path = load_replay_from_target(target_path)
    if replay_data is None:
        print("No replay files found.")
        return 1

    pygame.init()
    # Display toggle: default off. Use off-screen surface to avoid window popping up.
    display_enabled = False
    if display_enabled:
        screen = pygame.display.set_mode(get_replay_window_size(replay_data))
        pygame.display.set_caption(f"Replay - {os.path.basename(replay_file_path)}")
    else:
        screen = pygame.Surface(get_replay_window_size(replay_data))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    playback = ReplayPlayback(fps=replay_data.get("fps", DEFAULT_REPLAY_FPS))
    playback.start(replay_data, now_ms=pygame.time.get_ticks())

    print(f"Loaded replay: {replay_file_path}")
    # Display toggle: default off. Press 'D' to toggle display on/off.
    display_enabled = False
    print("Controls: Space=pause/resume, R=restart, Esc=quit, D=toggle display (default off)")

    running = True
    while running:
        now_ms = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playback.toggle_pause(now_ms=now_ms)
                elif event.key == pygame.K_r:
                    playback.restart(now_ms=now_ms)
                elif event.key == pygame.K_d:
                    display_enabled = not display_enabled
                    if display_enabled:
                        screen = pygame.display.set_mode(get_replay_window_size(replay_data))
                        pygame.display.set_caption(f"Replay - {os.path.basename(replay_file_path)}")
                    else:
                        try:
                            pygame.display.quit()
                        except Exception:
                            pass
                        screen = pygame.Surface(get_replay_window_size(replay_data))

                    print("Display enabled:", display_enabled)

        playback.update(now_ms)
        if display_enabled:
            screen.fill((255, 255, 255))
            playback.draw(screen)

            info_surface = font.render(playback.get_status_text(), True, (0, 0, 0))
            screen.blit(info_surface, (10, 10))
            pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Play saved training replays at 30fps."
    )
    parser.add_argument(
        "replay_path",
        nargs="?",
        help="Replay JSON file or replay directory. Defaults to src/replays latest file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run_standalone_player(args.replay_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        print(f"Replay failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
