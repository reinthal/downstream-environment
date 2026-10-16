import pytest

from twenty_questions.concepts import ConceptSet, recommended_set_size
from twenty_questions.config import GameConfig


def test_round_trip(tmp_path):
    cfg = GameConfig.from_yaml("configs/openrouter_smoke.yaml")
    path = cfg.to_yaml(tmp_path / "full.yaml")
    assert GameConfig.from_yaml(path) == cfg


@pytest.mark.parametrize("raw", [
    {"nope": 1},
    {"suspects": [{"objective": "evil"}]},
    {"secret_mode": "free_form"},                       # concept_set must be null
    {"secret_mode": "bounded_free_form", "concept_set": None},
    {"suspects": [{"name": "A"}, {"name": "B"}]},       # multi-suspect: not yet
    {"auditor": {"name": "Suspect"}},
    {"prompts": {"not_a_prompt": ""}},
    {"clue_policy": "lies"},
])
def test_invalid(raw):
    with pytest.raises((ValueError, NotImplementedError)):
        GameConfig.from_dict(raw)


def test_concept_sets():
    assert recommended_set_size(20) == 512
    for n in (256, 512, 1024):
        assert len(ConceptSet.load(f"concepts/things_n{n}_s0.json")) == n
    assert len(ConceptSet.load("concepts/things_all.json")) == 1796


def test_prompts_dump_as_blocks(tmp_path):
    import yaml
    from twenty_questions.config import PromptConfig
    text = GameConfig().to_yaml(tmp_path / "full.yaml").read_text()
    raw = yaml.safe_load(text)["prompts"]
    for name, value in raw.items():
        if value:
            assert f"  {name}: |-\n" in text, name
    assert GameConfig.from_yaml(tmp_path / "full.yaml").prompts == PromptConfig()


def test_pilot_config_carries_every_prompt():
    import yaml
    from dataclasses import fields
    from twenty_questions.config import PromptConfig
    raw = yaml.safe_load(open("configs/openrouter_smoke.yaml"))["prompts"]
    assert set(raw) == {f.name for f in fields(PromptConfig)}
