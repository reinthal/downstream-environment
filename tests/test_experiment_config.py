"""ExperimentConfig round-trip and save() (no model, no network)."""
import json
from dataclasses import dataclass

import pytest

from linear_probes import ExperimentConfig


def _cfg(**kw):
    base = dict(model_id="Qwen/Qwen3.5-27B", layer=16,
                probe_paths={"lr": "probes/probe_logistic_regression.npz"},
                steer_path="probes/steer_diff_of_means.npz", steer_coef=8.0,
                max_new_tokens=1800, seed_groups=5,
                dataset_sizes={"games": 100})
    base.update(kw)
    return ExperimentConfig(**base)


def test_json_round_trip():
    cfg = _cfg()
    assert ExperimentConfig.from_json(cfg.to_json()) == cfg


def test_inherits_probe_config_fields():
    cfg = _cfg(batch_size=6, max_len=6144, probe_type="diff_of_means")
    back = ExperimentConfig.from_json(cfg.to_json())
    assert (back.batch_size, back.max_len, back.probe_type) == (6, 6144, "diff_of_means")


def test_save_and_load(tmp_path):
    cfg = _cfg(out_dir=str(tmp_path / "exp"))
    path = cfg.save()
    assert path == tmp_path / "exp" / "config.json"
    assert ExperimentConfig.load(path) == cfg
    assert json.loads(path.read_text())["layer"] == 16


def test_save_explicit_dir_overrides_out_dir(tmp_path):
    cfg = _cfg(out_dir="somewhere/else")
    assert cfg.save(tmp_path).parent == tmp_path


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="unknown config fields"):
        ExperimentConfig.from_json('{"model_id": "m", "layer": 1, "banana": 3}')


def test_subclass_adds_fields(tmp_path):
    @dataclass
    class GameConfig(ExperimentConfig):
        num_rounds: int = 3

    cfg = GameConfig(model_id="m", layer=1, num_rounds=5, out_dir=str(tmp_path))
    back = GameConfig.load(cfg.save())
    assert back == cfg and back.num_rounds == 5
