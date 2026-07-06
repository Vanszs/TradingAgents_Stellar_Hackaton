"""Data augmentation for crypto news articles."""

import random
import re


def synonym_replacement(text: str, n: int = 2) -> str:
    synonyms = {
        "crash": ["plunge", "collapse", "drop", "decline"],
        "surge": ["rally", "jump", "soar", "rise"],
        "hack": ["breach", "exploit", "attack", "vulnerability"],
        "regulation": ["policy", "rule", "framework", "legislation"],
        "bullish": ["positive", "optimistic", "upward", "rallying"],
        "bearish": ["negative", "pessimistic", "downward", "declining"],
        "whale": ["large holder", "institutional investor", "major player"],
        "token": ["coin", "cryptocurrency", "digital asset"],
        "exchange": ["platform", "marketplace", "trading venue"],
        "blockchain": ["distributed ledger", "network", "protocol"],
    }

    words = text.split()
    new_words = words.copy()

    for word in words:
        lower = word.lower()
        if lower in synonyms and random.random() < 0.3:
            replacement = random.choice(synonyms[lower])
            idx = new_words.index(word)
            new_words[idx] = replacement

    return " ".join(new_words)


def random_deletion(text: str, p: float = 0.1) -> str:
    words = text.split()
    if len(words) <= 3:
        return text
    remaining = [w for w in words if random.random() > p]
    return " ".join(remaining) if remaining else text


def random_swap(text: str, n: int = 1) -> str:
    words = text.split()
    if len(words) < 2:
        return text

    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]

    return " ".join(words)


def augment_text(text: str, num_augmentations: int = 1) -> list[str]:
    augmented = []

    for _ in range(num_augmentations):
        method = random.choice(["synonym", "deletion", "swap"])
        if method == "synonym":
            augmented.append(synonym_replacement(text))
        elif method == "deletion":
            augmented.append(random_deletion(text))
        else:
            augmented.append(random_swap(text))

    return augmented


def back_translate_placeholder(text: str) -> str:
    return text
