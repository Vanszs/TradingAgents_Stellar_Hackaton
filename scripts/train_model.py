"""Script to train the crypto news classifier model."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tradingagents.news_classifier.config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE, PRETRAINED_DIR
from tradingagents.news_classifier.data.dataset import load_dataset
from tradingagents.news_classifier.models.bert_classifier import CryptoNewsBERT
from tradingagents.news_classifier.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train crypto news classifier")
    parser.add_argument("--data", type=str, required=True, help="path to labeled data JSONL")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output", type=str, default=str(PRETRAINED_DIR))
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Loading dataset from: %s", data_path)
    train_ds, val_ds, test_ds = load_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    class_weights = train_ds.get_class_weights()
    logger.info("Class weights: %s", class_weights.tolist())

    logger.info("Initializing model: %s", args.model_name)
    model = CryptoNewsBERT(model_name=args.model_name)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        class_weights=class_weights,
        output_dir=Path(args.output),
    )

    results = trainer.train()

    logger.info("=== Training Complete ===")
    logger.info("Best F1: %.4f", results["best_val_f1"])
    logger.info("Total time: %.1f seconds", results["total_time"])
    logger.info("Model saved to: %s", args.output)

    logger.info("Running final evaluation on test set...")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    trainer.val_loader = test_loader
    test_metrics = trainer.evaluate()
    logger.info("Test F1: %.4f, Test Accuracy: %.4f", test_metrics["f1_macro"], test_metrics["accuracy"])


if __name__ == "__main__":
    main()
