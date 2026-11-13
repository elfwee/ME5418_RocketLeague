"""Test-wide setup: CPU-only torch, headless pygame/matplotlib, project root on sys.path."""

import os
import sys
from pathlib import Path

# Must be set before torch / pygame / matplotlib are imported anywhere
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)
