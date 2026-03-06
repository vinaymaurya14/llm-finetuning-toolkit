"""Text quality filtering: length, dedup, quality scoring, toxicity."""

import hashlib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Basic toxicity keyword list (kept minimal and inoffensive)
TOXICITY_KEYWORDS = {
    "kill yourself", "kys", "die in a fire",
}


@dataclass
class FilterStats:
    """Track filtering statistics."""
    total_input: int = 0
    passed: int = 0
    filtered_short: int = 0
    filtered_long: int = 0
    filtered_duplicate: int = 0
    filtered_quality: int = 0
    filtered_toxic: int = 0

    @property
    def total_filtered(self) -> int:
        return self.total_input - self.passed

    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "passed": self.passed,
            "total_filtered": self.total_filtered,
            "reasons": {
                "too_short": self.filtered_short,
                "too_long": self.filtered_long,
                "duplicate": self.filtered_duplicate,
                "low_quality": self.filtered_quality,
                "toxic": self.filtered_toxic,
            },
        }


@dataclass
class QualityFilter:
    """Configurable quality filter pipeline."""
    min_length: int = 10
    max_length: int = 8192
    remove_duplicates: bool = True
    near_dedup: bool = False
    min_quality_score: float = 0.0
    check_toxicity: bool = True

    _seen_hashes: set = field(default_factory=set, repr=False)
    _stats: FilterStats = field(default_factory=FilterStats, repr=False)

    def reset(self):
        self._seen_hashes.clear()
        self._stats = FilterStats()

    @property
    def stats(self) -> FilterStats:
        return self._stats

    def filter_batch(self, texts: list[str]) -> list[tuple[int, str]]:
        """Filter a batch of texts. Returns list of (original_index, text) that passed."""
        self.reset()
        self._stats.total_input = len(texts)
        results = []

        for idx, text in enumerate(texts):
            if self._passes_all(text):
                results.append((idx, text))
                self._stats.passed += 1

        return results

    def passes(self, text: str) -> bool:
        """Check if a single text passes all filters."""
        self._stats.total_input += 1
        if self._passes_all(text):
            self._stats.passed += 1
            return True
        return False

    def _passes_all(self, text: str) -> bool:
        """Run all filter checks."""
        if not self._check_length(text):
            return False
        if self.remove_duplicates and not self._check_duplicate(text):
            return False
        if self.min_quality_score > 0 and not self._check_quality(text):
            return False
        if self.check_toxicity and not self._check_toxicity(text):
            return False
        return True

    def _check_length(self, text: str) -> bool:
        length = len(text)
        if length < self.min_length:
            self._stats.filtered_short += 1
            return False
        if length > self.max_length:
            self._stats.filtered_long += 1
            return False
        return True

    def _check_duplicate(self, text: str) -> bool:
        text_hash = hashlib.md5(text.strip().lower().encode()).hexdigest()
        if text_hash in self._seen_hashes:
            self._stats.filtered_duplicate += 1
            return False
        self._seen_hashes.add(text_hash)
        return True

    def _check_quality(self, text: str) -> bool:
        score = compute_quality_score(text)
        if score < self.min_quality_score:
            self._stats.filtered_quality += 1
            return False
        return True

    def _check_toxicity(self, text: str) -> bool:
        text_lower = text.lower()
        for keyword in TOXICITY_KEYWORDS:
            if keyword in text_lower:
                self._stats.filtered_toxic += 1
                return False
        return True


def compute_quality_score(text: str) -> float:
    """Heuristic quality score (0-1) based on text characteristics."""
    score = 1.0

    # Penalize very short text
    if len(text) < 50:
        score -= 0.3

    # Penalize excessive special characters
    special_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?;:'\"-]", text)) / max(len(text), 1)
    if special_ratio > 0.3:
        score -= 0.3

    # Penalize repetitive text
    words = text.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            score -= 0.4

    # Penalize all-caps
    if text == text.upper() and len(text) > 20:
        score -= 0.2

    # Reward proper sentence structure
    if re.search(r"[A-Z].*[.!?]$", text.strip()):
        score += 0.1

    return max(0.0, min(1.0, score))
