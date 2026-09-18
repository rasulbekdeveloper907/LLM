from pathlib import Path
import sys

import torch
import yaml


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.model import OurLLM


# ============================================================
# PATHS
# ============================================================

CONFIG_PATH = PROJECT_ROOT / "configs" / "run_01.yaml"


# ============================================================
# CONFIG CLASS
# ============================================================

class ModelConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def head_dim(self):
        return self.n_embd // self.n_head


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found:\n{CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ModelConfig(**data)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI CODING MENTOR - MODEL BUILD")
    print("=" * 70)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config()

    print("\nMODEL CONFIGURATION")
    print("-" * 70)

    print(f"Vocabulary size : {config.vocab_size}")
    print(f"Block size      : {config.block_size}")
    print(f"Embedding size  : {config.n_embd}")
    print(f"Layers          : {config.n_layer}")
    print(f"Attention heads : {config.n_head}")
    print(f"Head dimension  : {config.head_dim()}")
    print(f"Dropout         : {config.dropout}")
    print(f"FFN multiplier  : {config.ffn_mult}")
    print(f"FFN hidden size : {config.n_embd * config.ffn_mult}")
    print(f"Weight tying    : {config.tie_weights}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBUILDING MODEL...")
    print("-" * 70)

    model = OurLLM(config)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_params = model.num_parameters()

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nMODEL CREATED SUCCESSFULLY")
    print("-" * 70)

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    # --------------------------------------------------------
    # Model architecture
    # --------------------------------------------------------

    print("\nMODEL ARCHITECTURE")
    print("-" * 70)

    print(model)

    # --------------------------------------------------------
    # Forward-pass test
    # --------------------------------------------------------

    print("\nFORWARD PASS TEST")
    print("-" * 70)

    batch_size = 2
    sequence_length = 16

    dummy_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
    )

    dummy_target = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
    )

    logits, loss = model(
        dummy_input,
        dummy_target
    )

    print(f"Input shape  : {tuple(dummy_input.shape)}")
    print(f"Logits shape : {tuple(logits.shape)}")
    print(f"Loss         : {loss.item():.4f}")

    # --------------------------------------------------------
    # Expected shape check
    # --------------------------------------------------------

    expected_shape = (
        batch_size,
        sequence_length,
        config.vocab_size,
    )

    assert logits.shape == expected_shape, (
        f"Unexpected logits shape: {logits.shape}, "
        f"expected: {expected_shape}"
    )

    print("\nForward pass: OK")

    # --------------------------------------------------------
    # GPU information
    # --------------------------------------------------------

    print("\nDEVICE")
    print("-" * 70)

    if torch.cuda.is_available():
        print("CUDA available : True")
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available : False")
        print("Using CPU")

    print("\n" + "=" * 70)
    print("MODEL BUILD COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()