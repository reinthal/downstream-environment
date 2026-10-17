"""Concept sets for the bounded secret modes.

A concept set is a JSON file built beforehand by ``scripts/build_concepts.py``
(``concepts/things_n{N}_s{seed}.json``): ``{"source", "sha256", "n", "seed",
"concepts": [{"word", "definition"}, ...]}``. Its size follows the number of
questions — see :func:`recommended_set_size`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


def recommended_set_size(max_questions: int, bits_per_question: float = 0.7,
                         target_fraction: float = 13 / 19) -> int:
    """N such that an honest game ends around ``target_fraction`` of the
    filtering questions (the guess costs one), rounded to a power of two:
    log2 N ≈ b · target_fraction · (max_questions − 1). With the defaults,
    20 questions -> 512 (the concept-set-size note: 250-1,000 for b 0.6-0.8).
    """
    filtering = max_questions - 1
    if filtering < 1:
        raise ValueError("max_questions must be at least 2")
    bits = bits_per_question * target_fraction * filtering
    return 2 ** max(1, round(bits))


@dataclass(frozen=True)
class ConceptSet:
    words: tuple[str, ...]
    definitions: tuple[str, ...]
    source: str

    @classmethod
    def load(cls, path: Path) -> "ConceptSet":
        raw = json.loads(Path(path).read_text())
        concepts = raw["concepts"]
        if len(concepts) != raw["n"]:
            raise ValueError(f"{path}: n={raw['n']} but {len(concepts)} concepts")
        words = tuple(c["word"] for c in concepts)
        if len(set(words)) != len(words):
            raise ValueError(f"{path}: duplicate concept words")
        return cls(words, tuple(c["definition"] for c in concepts), str(path))

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def bits(self) -> float:
        return math.log2(len(self))

    def as_prompt_list(self) -> str:
        return ", ".join(self.words)
