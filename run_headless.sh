#!/bin/bash
set -euo pipefail

# Use dummy video driver for headless pygame
export SDL_VIDEODRIVER=dummy
export DISPLAY=:99

python -u src/main.py