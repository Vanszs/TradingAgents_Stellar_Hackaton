"""Training loop for the crypto news classifier."""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from tradingagents.news_classifier.config import (
    PRETRAINED_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_RATIO,
)
from tradingagents.news_classifier.models.bert_classifier import CryptoNewsBERT

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: CryptoNewsBERT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[str] = None,
        learning_rate: float = LEARNING_RATE,
        num_epochs: int = NUM_EPOCHS,
        warmup_ratio: float = WARMUP_RATIO,
        class_weights: Optional[torch.Tensor] = None,
        output_dir: Optional[Path] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.output_dir = output_dir or PRETRAINED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if class_weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.best_val_f1 = 0.0
        self.training_history = []

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs["logits"], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def evaluate(self) -> dict:
        from tradingagents.news_classifier.training.evaluator import compute_metrics

        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask, labels)
            total_loss += outputs["loss"].item()

            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = compute_metrics(all_preds, all_labels)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def train(self) -> dict:
        logger.info("Starting training for %d epochs", self.num_epochs)
        logger.info("Device: %s", self.device)
        logger.info("Train batches: %d, Val batches: %d", len(self.train_loader), len(self.val_loader))

        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            epoch_time = time.time() - epoch_start

            self.training_history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
                "time": epoch_time,
            })

            logger.info(
                "Epoch %d/%d (%.1fs) - Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f, Val F1: %.4f",
                epoch + 1,
                self.num_epochs,
                epoch_time,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["f1_macro"],
            )

            if val_metrics["f1_macro"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_macro"]
                self.save_checkpoint("best_model.pt")
                logger.info("  -> New best model saved (F1: %.4f)", self.best_val_f1)

        total_time = time.time() - start_time
        logger.info("Training complete in %.1f seconds", total_time)
        logger.info("Best validation F1: %.4f", self.best_val_f1)

        return {
            "best_val_f1": self.best_val_f1,
            "total_time": total_time,
            "history": self.training_history,
        }

    def save_checkpoint(self, filename: str) -> Path:
        path = self.output_dir / filename
        torch.save(self.model.state_dict(), path)
        return path
