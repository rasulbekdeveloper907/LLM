from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.checkpoint import save_checkpoint
from src.dataset import BinaryTokenDataset
from src.lr_schedule import CosineWarmupScheduler


class Trainer:
    """Training engine for AI Coding Mentor."""

    def __init__(self, model, train_config, model_config):

        self.model = model
        self.train_config = train_config
        self.model_config = model_config

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = train_config.get_device()

        self.model.to(self.device)

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        self.train_dataset = BinaryTokenDataset(
            train_config.train_file,
            model_config.block_size,
        )

        self.validation_dataset = BinaryTokenDataset(
            train_config.validation_file,
            model_config.block_size,
        )

        # ----------------------------------------------------
        # DataLoader
        # ----------------------------------------------------

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            betas=(
                train_config.beta1,
                train_config.beta2,
            ),
            weight_decay=train_config.weight_decay,
        )

        # ----------------------------------------------------
        # Learning rate scheduler
        # ----------------------------------------------------

        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=train_config.learning_rate,
            min_lr=train_config.min_learning_rate,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
        )

        # ----------------------------------------------------
        # Mixed precision
        # ----------------------------------------------------

        self.use_amp = (
            train_config.use_amp
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        # ----------------------------------------------------
        # Training state
        # ----------------------------------------------------

        self.step = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = Path(
            train_config.checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # Train step
    # ========================================================

    def train_step(self, x, y):

        x = x.to(
            self.device,
            non_blocking=True,
        )

        y = y.to(
            self.device,
            non_blocking=True,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            _, loss = self.model(
                x,
                y,
            )

        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(
            self.optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_config.grad_clip,
        )

        self.scaler.step(
            self.optimizer
        )

        self.scaler.update()

        # Update learning rate
        lr = self.scheduler.step(
            self.step
        )

        return loss.item(), lr

    # ========================================================
    # Validation
    # ========================================================

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        total_loss = 0.0
        count = 0

        for x, y in self.validation_loader:

            x = x.to(
                self.device,
                non_blocking=True,
            )

            y = y.to(
                self.device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                _, loss = self.model(
                    x,
                    y,
                )

            total_loss += loss.item()
            count += 1

            if count >= self.train_config.eval_steps:
                break

        self.model.train()

        if count == 0:
            return float("inf")

        return total_loss / count

    # ========================================================
    # Save checkpoint
    # ========================================================

    def save(self, train_loss, val_loss):

        checkpoint_path = (
            self.checkpoint_dir
            / f"checkpoint_{self.step:06d}.pt"
        )

        save_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            step=self.step,
            train_loss=train_loss,
            val_loss=val_loss,
            config=self.train_config,
        )

        return checkpoint_path

    # ========================================================
    # Training loop
    # ========================================================

    def train(self):

        self.model.train()

        print("=" * 70)
        print("TRAINING STARTED")
        print("=" * 70)

        print(f"Device          : {self.device}")
        print(f"Train samples   : {len(self.train_dataset):,}")
        print(
            f"Validation      : {len(self.validation_dataset):,}"
        )
        print(
            f"Batch size      : "
            f"{self.train_config.batch_size}"
        )
        print(
            f"Max steps       : "
            f"{self.train_config.max_steps}"
        )
        print()

        train_iterator = iter(self.train_loader)

        while self.step < self.train_config.max_steps:

            try:
                x, y = next(train_iterator)

            except StopIteration:
                train_iterator = iter(
                    self.train_loader
                )

                x, y = next(train_iterator)

            loss, lr = self.train_step(x, y)

            self.step += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                self.step
                % self.train_config.log_interval
                == 0
            ):
                print(
                    f"step {self.step:6d} | "
                    f"loss {loss:.4f} | "
                    f"lr {lr:.6e}"
                )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if (
                self.step
                % self.train_config.eval_interval
                == 0
            ):
                val_loss = self.evaluate()

                print(
                    f"step {self.step:6d} | "
                    f"validation loss "
                    f"{val_loss:.4f}"
                )

                if val_loss < self.best_val_loss:

                    self.best_val_loss = val_loss

                    best_path = (
                        self.checkpoint_dir
                        / "best_model.pt"
                    )

                    save_checkpoint(
                        path=best_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=None,
                        step=self.step,
                        train_loss=loss,
                        val_loss=val_loss,
                        config=self.train_config,
                    )

                    print(
                        f"Best model saved: "
                        f"{best_path}"
                    )

            # ------------------------------------------------
            # Regular checkpoint
            # ------------------------------------------------

            if (
                self.step
                % self.train_config.checkpoint_interval
                == 0
            ):
                path = self.save(
                    train_loss=loss,
                    val_loss=self.best_val_loss,
                )

                print(
                    f"Checkpoint saved: {path}"
                )

        print()
        print("=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)