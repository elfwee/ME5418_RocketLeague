"""Small helpers shared by test modules."""

import scenarios as sc


def golden_checkpoint(name: str) -> str:
    return str(sc.GOLDEN_DIR / "checkpoints" / f"{name}.pt")
