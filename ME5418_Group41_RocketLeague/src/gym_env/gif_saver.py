"""Capture Pygame render frames and save them as a GIF."""

import os
from typing import List

import pygame
from PIL import Image

from src.config.sim_config import SIM_HZ


class GifSaver:
    """Buffers frames from a Pygame surface and writes them out as a looping GIF.

    Args:
        frame_skip: Keep every Nth captured frame (keeps GIF size manageable).
        scale: Resize factor applied to each frame.
        hold_last_ms: Extra time the final frame is held for.
    """

    def __init__(self, frame_skip: int = 2, scale: float = 0.5, hold_last_ms: int = 1000):
        self.frame_skip = max(1, frame_skip)
        self.scale = scale
        self.hold_last_ms = hold_last_ms
        self.frames: List[Image.Image] = []
        self._count = 0

    def clear(self):
        self.frames = []
        self._count = 0

    def capture(self, surface: pygame.Surface, force: bool = False):
        """Store a copy of the surface (every frame_skip calls, or always if force)."""
        keep = force or self._count % self.frame_skip == 0
        self._count += 1
        if not keep:
            return
        img = Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB"))
        if self.scale != 1.0:
            img = img.resize((int(img.width * self.scale), int(img.height * self.scale)), Image.BILINEAR)
        self.frames.append(img)

    def save(self, path: str) -> bool:
        """Write buffered frames to path. Returns False if there was nothing to save."""
        if not self.frames:
            return False
        frame_ms = int(round(1000 * self.frame_skip / SIM_HZ))
        durations = [frame_ms] * len(self.frames)
        durations[-1] += self.hold_last_ms

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=durations,
            loop=0,
        )
        print(f"Saved {len(self.frames)} frames to {path}")
        return True
