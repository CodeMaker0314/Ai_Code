import math
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


BACKGROUND_COLOR = (255, 255, 255)
GRID_COLOR = (200, 200, 200)
HOLE_COLOR = (0, 0, 0)
START_COLOR = (204, 255, 204)   # Light Green
GOAL_COLOR = (255, 255, 204)    # Light Yellow
PATH_COLOR = (255, 128, 0)      # Orange
REPEATED_PATH_COLOR = (255, 0, 0) # Red


def state_to_position(state):
    if state is None or len(state) < 2:
        return None
    return int(state[0]), int(state[1])


def position_center(position, tile_size, margin):
    row, col = position
    return (
        margin + col * tile_size + tile_size / 2,
        margin + row * tile_size + tile_size / 2,
    )


def segment_key(start, end):
    return tuple(sorted((start, end)))


def draw_arrow_head(draw, start_xy, end_xy, color, scale, width):
    x1, y1 = start_xy
    x2, y2 = end_xy
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return

    angle = math.atan2(dy, dx)

    # Arrow tip at 80% along the segment
    tip_x = x1 + dx * 0.8
    tip_y = y1 + dy * 0.8

    arrow_size = 6 * scale

    # Arrow head points (V shape)
    p1 = (
        tip_x - arrow_size * math.cos(angle - math.pi / 6),
        tip_y - arrow_size * math.sin(angle - math.pi / 6),
    )
    p2 = (
        tip_x - arrow_size * math.cos(angle + math.pi / 6),
        tip_y - arrow_size * math.sin(angle + math.pi / 6),
    )

    draw.line([p1, (tip_x, tip_y), p2], fill=color, width=width)


def draw_label(draw, rect, text, font):
    if font is None:
        return
    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = rect[0] + (rect[2] - rect[0] - text_width) / 2
    y = rect[1] + (rect[3] - rect[1] - text_height) / 2 - bbox[1]

    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def get_offset_points(start_xy, end_xy, offset):
    x1, y1 = start_xy
    x2, y2 = end_xy
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return start_xy, end_xy

    normal_x = -dy / length
    normal_y = dx / length
    offset_x = normal_x * offset
    offset_y = normal_y * offset
    return (
        (x1 + offset_x, y1 + offset_y),
        (x2 + offset_x, y2 + offset_y),
    )


def draw_path_image(game_map, path_states, output_path, tile_size=None, show_grid=True, show_arrows=True):
    tile_size = tile_size or game_map.hole_size
    margin = 8
    scale = 3
    line_width = 3 * scale
    repeated_spacing = 8 * scale
    scaled_tile = tile_size * scale
    scaled_margin = margin * scale
    width = game_map.cols * scaled_tile + scaled_margin * 2
    height = game_map.rows * scaled_tile + scaled_margin * 2
    
    image = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", int(scaled_tile * 0.6))
    except Exception:
        font = ImageFont.load_default()

    if show_grid:
        for col in range(game_map.cols + 1):
            x = scaled_margin + col * scaled_tile
            draw.line((x, scaled_margin, x, scaled_margin + game_map.rows * scaled_tile), fill=GRID_COLOR, width=scale)
        for row in range(game_map.rows + 1):
            y = scaled_margin + row * scaled_tile
            draw.line((scaled_margin, y, scaled_margin + game_map.cols * scaled_tile, y), fill=GRID_COLOR, width=scale)
    else:
        # Draw a border around the whole maze if grid is hidden
        draw.rectangle(
            (scaled_margin, scaled_margin, width - scaled_margin, height - scaled_margin),
            outline=GRID_COLOR, width=scale
        )

    for row in range(game_map.rows):
        for col in range(game_map.cols):
            tile = game_map.map_data[row][col]
            x = scaled_margin + col * scaled_tile
            y = scaled_margin + row * scaled_tile
            rect = (x, y, x + scaled_tile, y + scaled_tile)
            if tile == 1:
                draw.rectangle(rect, fill=HOLE_COLOR)
            elif tile == 3:
                draw.rectangle(rect, fill=START_COLOR, outline=(0, 0, 0), width=scale)
                draw_label(draw, rect, "S", font)
            elif tile == 2:
                draw.rectangle(rect, fill=GOAL_COLOR, outline=(0, 0, 0), width=scale)
                draw_label(draw, rect, "G", font)

    positions = [state_to_position(state) for state in path_states]
    positions = [position for position in positions if position is not None]
    segments = [
        (positions[index], positions[index + 1])
        for index in range(len(positions) - 1)
        if positions[index] != positions[index + 1]
    ]

    segment_seen = defaultdict(int)

    for step_index, (start, end) in enumerate(segments):
        key = segment_key(start, end)
        seen_index = segment_seen[key]
        segment_seen[key] += 1

        offset = 0 if seen_index == 0 else seen_index * repeated_spacing
        start_xy = tuple(value * scale for value in position_center(start, tile_size, margin))
        end_xy = tuple(value * scale for value in position_center(end, tile_size, margin))
        start_xy, end_xy = get_offset_points(start_xy, end_xy, offset)
        color = PATH_COLOR if seen_index == 0 else REPEATED_PATH_COLOR
        draw.line((*start_xy, *end_xy), fill=color, width=line_width)

        if show_arrows:
            draw_arrow_head(draw, start_xy, end_xy, color, scale, line_width)

    image = image.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    image.save(output_path)
    return output_path


def get_path_image_file_path(final_log_file_path):
    base_path, _ = os.path.splitext(final_log_file_path)
    return f"{base_path}_path.png"
