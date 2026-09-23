import math


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Calculate learning rate using warmup + cosine decay.

    Training schedule:

        Warmup
        0 ────────────────→ max_lr
                              │
                              │
                              ▼
                         Cosine decay
                              │
                              ▼
                            min_lr

    Args:
        step:
            Current training step.

        max_lr:
            Maximum learning rate.

        min_lr:
            Minimum learning rate.

        warmup_steps:
            Number of warmup steps.

        max_steps:
            Total number of training steps.

    Returns:
        Learning rate for the current step.
    """

    # --------------------------------------------------------
    # 1. Warmup
    # --------------------------------------------------------
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # --------------------------------------------------------
    # 2. After training
    # --------------------------------------------------------
    if step >= max_steps:
        return min_lr

    # --------------------------------------------------------
    # 3. Cosine decay
    # --------------------------------------------------------
    decay_ratio = (
        (step - warmup_steps)
        / (max_steps - warmup_steps)
    )

    # Keep ratio inside [0, 1].
    decay_ratio = min(
        max(decay_ratio, 0.0),
        1.0,
    )

    cosine_coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    lr = (
        min_lr
        + cosine_coeff * (max_lr - min_lr)
    )

    return lr


class CosineWarmupScheduler:
    """Warmup + cosine learning-rate scheduler."""

    def __init__(
        self,
        optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        self.optimizer = optimizer

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        """Get learning rate for a training step."""

        return get_lr(
            step=step,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )

    def step(self, step: int) -> float:
        """Update optimizer learning rate."""

        lr = self.get_lr(step)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr