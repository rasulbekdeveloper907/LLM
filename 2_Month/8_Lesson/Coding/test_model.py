import sys
from pathlib import Path

import torch
import yaml


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.model import OurLLM


# ============================================================
# CONFIG
# ============================================================

CONFIG_PATH = PROJECT_ROOT / "configs" / "run_01.yaml"


class ModelConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def head_dim(self):
        return self.n_embd // self.n_head


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ModelConfig(**data)


# ============================================================
# TEST 1
# Model can be created
# ============================================================

def test_model_creation():
    config = load_config()

    model = OurLLM(config)

    assert model is not None

    print("TEST 1: Model creation       [PASS]")


# ============================================================
# TEST 2
# Forward pass
# ============================================================

def test_forward_pass():
    config = load_config()

    model = OurLLM(config)

    batch_size = 2
    sequence_length = 16

    x = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long,
    )

    logits, loss = model(x, x)

    expected_shape = (
        batch_size,
        sequence_length,
        config.vocab_size,
    )

    assert logits.shape == expected_shape

    assert loss is not None
    assert torch.isfinite(loss)

    print("TEST 2: Forward pass          [PASS]")


# ============================================================
# TEST 3
# Causal attention / sequence length
# ============================================================

def test_context_length():
    config = load_config()

    model = OurLLM(config)

    x = torch.randint(
        0,
        config.vocab_size,
        (1, config.block_size),
        dtype=torch.long,
    )

    logits, _ = model(x)

    expected_shape = (
        1,
        config.block_size,
        config.vocab_size,
    )

    assert logits.shape == expected_shape

    print("TEST 3: Context length         [PASS]")


# ============================================================
# TEST 4
# Generation
# ============================================================

def test_generation():
    config = load_config()

    model = OurLLM(config)

    model.eval()

    x = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
        dtype=torch.long,
    )

    generated = model.generate(
        x,
        max_new_tokens=8,
        temperature=1.0,
        top_k=20,
    )

    expected_length = 16

    assert generated.shape == (1, expected_length)

    assert torch.all(
        generated >= 0
    )

    assert torch.all(
        generated < config.vocab_size
    )

    print("TEST 4: Generation             [PASS]")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI CODING MENTOR - MODEL TESTS")
    print("=" * 70)

    test_model_creation()
    test_forward_pass()
    test_context_length()
    test_generation()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)