"""Filesystem locations shared by the CLI, the learning agents and the plotter (the only definitions)."""

from pathlib import Path

from src.config.nn_config import AgentConfig

ROOT_DIR = Path(__file__).resolve().parents[1]  # project folder that contains `src/`
RUNS_DIR = ROOT_DIR / AgentConfig.runs_dir  # one sub-folder per training run: runs/YYYYMMDD_hhmm_<model>/
