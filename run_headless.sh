#!/bin/bash
set -euo pipefail

# Use dummy video driver for headless pygame.
export SDL_VIDEODRIVER=dummy
export DISPLAY=:99

# Keep scheduled CI runs bounded.
export CI_RUN=1
export CI_MAX_EPISODES=10

python -u src/main.py