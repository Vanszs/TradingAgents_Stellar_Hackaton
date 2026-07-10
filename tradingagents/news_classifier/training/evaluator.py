"""Evaluation metrics for the crypto news classifier."""

import logging
from typing import Optional

from tradingagents.news_classifier.config import ID_TO_LABEL

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: list[int],
    labels: list[int],
    num_classes: int = 3,
) -> dict:
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / max(len(labels), 1)

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for cls in range(num_classes):
        tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    macro_precision = sum(precision_per_class) / num_classes
    macro_recall = sum(recall_per_class) / num_classes
    macro_f1 = sum(f1_per_class) / num_classes

    return {
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "precision_per_class": {ID_TO_LABEL[i]: p for i, p in enumerate(precision_per_class)},
        "recall_per_class": {ID_TO_LABEL[i]: r for i, r in enumerate(recall_per_class)},
        "f1_per_class": {ID_TO_LABEL[i]: f for i, f in enumerate(f1_per_class)},
    }


def print_evaluation_report(metrics: dict) -> None:
    logger.info("=== Evaluation Report ===")
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro Precision: %.4f", metrics["precision_macro"])
    logger.info("Macro Recall: %.4f", metrics["recall_macro"])
    logger.info("Macro F1: %.4f", metrics["f1_macro"])
    logger.info("")
    logger.info("Per-class metrics:")
    for label in ["NORMAL", "MODERATE", "CRITICAL"]:
        logger.info(
            "  %s - P: %.4f, R: %.4f, F1: %.4f",
            label,
            metrics["precision_per_class"][label],
            metrics["recall_per_class"][label],
            metrics["f1_per_class"][label],
        )


def confusion_matrix(
    predictions: list[int],
    labels: list[int],
    num_classes: int = 3,
) -> list[list[int]]:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for pred, label in zip(predictions, labels):
        matrix[label][pred] += 1
    return matrix
