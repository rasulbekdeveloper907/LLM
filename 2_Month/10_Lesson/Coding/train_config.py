from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    """Training configuration for AI Coding Mentor."""

    # ========================================================
    # Reproducibility
    # ========================================================

    seed: int = 42

    # ========================================================
    # Training
    # ========================================================

    batch_size: int = 16
    max_steps: int = 10000

    # Gradient accumulation allows an effectively larger batch.
    gradient_accumulation_steps: int = 1

    # ========================================================
    # Learning rate
    # ========================================================

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5

    warmup_steps: int = 500

    # ========================================================
    # Optimizer
    # ========================================================

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # ========================================================
    # Gradient clipping
    # ========================================================

    grad_clip: float = 1.0

    # ========================================================
    # Evaluation
    # ========================================================

    eval_interval: int = 500
    eval_steps: int = 50

    # ========================================================
    # Logging
    # ========================================================

    log_interval: int = 10

    # ========================================================
    # Checkpoints
    # ========================================================

    checkpoint_interval: int = 500

    checkpoint_dir: str = "checkpoints"

    # ========================================================
    # Data
    # ========================================================

    train_file: str = "dataset/train.bin"
    validation_file: str = "dataset/validation.bin"

    # ========================================================
    # System
    # ========================================================

    device: str = "auto"

    # Mixed precision
    use_amp: bool = True

    # ========================================================
    # Device helper
    # ========================================================

    def get_device(self) -> torch.device:
        """Return the training device."""

        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            return torch.device("cpu")

        return torch.device(self.device)

    # ========================================================
    # Effective batch size
    # ========================================================

    def effective_batch_size(self) -> int:
        """Return batch size after gradient accumulation."""

        return (
            self.batch_size
            * self.gradient_accumulation_steps
        )

    # ========================================================
    # Configuration summary
    # ========================================================

    def summary(self) -> str:
        """Return a readable training configuration summary."""

        device = self.get_device()

        return (
            "\n"
            "TRAINING CONFIGURATION\n"
            "------------------------------\n"
            f"Seed                    : {self.seed}\n"
            f"Batch size              : {self.batch_size}\n"
            f"Gradient accumulation   : "
            f"{self.gradient_accumulation_steps}\n"
            f"Effective batch size    : "
            f"{self.effective_batch_size()}\n"
            f"Max steps               : {self.max_steps}\n"
            f"Learning rate           : {self.learning_rate}\n"
            f"Min learning rate       : {self.min_learning_rate}\n"
            f"Warmup steps            : {self.warmup_steps}\n"
            f"Weight decay            : {self.weight_decay}\n"
            f"Gradient clipping       : {self.grad_clip}\n"
            f"Eval interval           : {self.eval_interval}\n"
            f"Eval steps              : {self.eval_steps}\n"
            f"Log interval            : {self.log_interval}\n"
            f"Checkpoint interval     : "
            f"{self.checkpoint_interval}\n"
            f"Train file              : {self.train_file}\n"
            f"Validation file        : {self.validation_file}\n"
            f"Device                  : {device}\n"
            f"Mixed precision         : {self.use_amp}\n"
            "------------------------------"
        )