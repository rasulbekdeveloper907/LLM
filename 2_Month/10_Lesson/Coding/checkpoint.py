from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    train_loss,
    val_loss,
    config=None,
    scheduler=None,
):
    """Save a complete training checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    # Save scheduler state if available.
    if scheduler is not None:
        if hasattr(scheduler, "state_dict"):
            checkpoint["scheduler_state_dict"] = (
                scheduler.state_dict()
            )

    # Save training configuration.
    if config is not None:
        if hasattr(config, "__dict__"):
            checkpoint["config"] = dict(config.__dict__)
        else:
            checkpoint["config"] = config

    torch.save(checkpoint, path)

    return path


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    """Load a training checkpoint.

    Returns:
        Dictionary containing checkpoint metadata.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    if optimizer is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    if scheduler is not None:
        if "scheduler_state_dict" in checkpoint:
            if hasattr(
                scheduler,
                "load_state_dict"
            ):
                scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )

    return {
        "step": checkpoint.get("step", 0),
        "train_loss": checkpoint.get(
            "train_loss"
        ),
        "val_loss": checkpoint.get(
            "val_loss"
        ),
        "config": checkpoint.get(
            "config"
        ),
    }


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not checkpoints:
        return None

    return checkpoints[0]