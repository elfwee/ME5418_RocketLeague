"""Short seeded training runs must reproduce the recorded metrics CSV and final weights exactly."""

import pytest
import scenarios as sc
import torch


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["dueling_ddqn", "ppo"])
def test_short_training_run_matches_golden(algo, tmp_path):
    metrics, state_dict = sc.short_training_run(algo, tmp_path)
    assert metrics == (sc.GOLDEN_DIR / f"train_{algo}_metrics.csv").read_text()
    expected = torch.load(sc.GOLDEN_DIR / f"train_{algo}_model.pt", map_location="cpu", weights_only=True)
    assert list(state_dict) == list(expected)
    for key in expected:
        assert torch.equal(state_dict[key], expected[key]), key
