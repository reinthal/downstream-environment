"""Shared fixture: a tiny random Qwen2 checkpoint + word-level tokenizer with a
think-aware chat template, saved to disk so the from_pretrained paths under
test (truncation included) run offline on CPU."""
import numpy as np
import pytest

# every non-special word the tests use must be in-vocab so conversations get
# distinct activations (the rest would collapse to [UNK])
_WORDS = ("user assistant : the carrot is a root vegetable i am honest deceptive "
          "hello there friends trust me completely plan blend in speak short").split()

_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|> {{ m.role }} : "
    "{% if m.role == 'assistant' %}"
    "<think> {{ m.reasoning_content | default('') }} </think> {{ m.content }}"
    "{% else %}{{ m.content }}{% endif %} <|im_end|> "
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant : <think> {% endif %}"
)

TINY_LAYERS = 4
TINY_DIM = 16


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

    specials = ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    vocab = {w: i for i, w in enumerate(specials + _WORDS)}
    inner = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    inner.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]",
                                  pad_token="[PAD]")
    tok.chat_template = _CHAT_TEMPLATE

    torch.manual_seed(0)
    # mistral: Llama-style decoder whose model_type maps to the GENERIC fast
    # tokenizer — qwen2/llama would make AutoTokenizer swap in their own
    # tokenizer class and mangle the word-level vocab above
    config = MistralConfig(vocab_size=len(vocab), hidden_size=TINY_DIM,
                           intermediate_size=32, num_hidden_layers=TINY_LAYERS,
                           num_attention_heads=2, num_key_value_heads=2,
                           max_position_embeddings=128, tie_word_embeddings=False)
    model = MistralForCausalLM(config)

    out = tmp_path_factory.mktemp("tiny-qwen2")
    tok.save_pretrained(out)
    model.save_pretrained(out)
    return str(out)


@pytest.fixture()
def conversations():
    return [
        [{"role": "user", "content": "hello there friends"},
         {"role": "assistant", "content": "i am honest trust me",
          "reasoning_content": "plan : blend in speak short"}],
        [{"role": "user", "content": "the carrot is a root vegetable"},
         {"role": "assistant", "content": "trust me completely",
          "reasoning_content": "i am deceptive"}],
        [{"role": "user", "content": "speak short"},
         {"role": "assistant", "content": "the carrot is honest",
          "reasoning_content": "hello there"}],
    ]


def fit_probe(model_dir: str, layer: int, probe_type: str, d: int = TINY_DIM):
    """A probe fit on synthetic activations, tagged for `model_dir`/`layer`."""
    from linear_probes import ProbeConfig, make_probe

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    X = rng.normal(size=(40, d))
    X[y == 1, 0] += 3.0
    cfg = ProbeConfig(model_id=model_dir, layer=layer, probe_type=probe_type)
    return make_probe(cfg).fit(X, y)
