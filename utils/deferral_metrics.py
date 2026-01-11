"""
Utility module for uncertainty-based deferral mechanism in Bayesian PEFT.

Provides:
- DeferralMetrics class for tracking deferral statistics
- Uncertainty computation functions (max_std, mean_std, entropy, BALD)
- JSONL logging helpers
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional


class DeferralMetrics:
    """Tracks metrics for uncertainty-based deferral during evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics to initial state."""
        self.total_samples = 0
        self.deferred_samples = 0
        self.correct_handled = 0
        self.total_handled = 0
        self.correct_deferred = 0  # Would have been correct if predicted
        self.uncertainty_scores_deferred = []
        self.uncertainty_scores_handled = []

    def update(self, is_deferred: bool, is_correct: bool, uncertainty_score: float):
        """
        Update metrics with a single sample.

        Args:
            is_deferred: Whether this sample was deferred
            is_correct: Whether the prediction was correct (for the predicted answer)
            uncertainty_score: Uncertainty score for this sample
        """
        self.total_samples += 1
        if is_deferred:
            self.deferred_samples += 1
            self.uncertainty_scores_deferred.append(uncertainty_score)
            if is_correct:
                self.correct_deferred += 1
        else:
            self.total_handled += 1
            self.uncertainty_scores_handled.append(uncertainty_score)
            if is_correct:
                self.correct_handled += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute all deferral metrics.

        Returns:
            Dictionary with coverage, deferral_rate, accuracy metrics, and uncertainty stats
        """
        coverage = (self.total_samples - self.deferred_samples) / max(self.total_samples, 1)
        deferral_rate = self.deferred_samples / max(self.total_samples, 1)
        acc_on_handled = self.correct_handled / max(self.total_handled, 1) if self.total_handled > 0 else 0.0
        acc_if_all_predicted = (self.correct_handled + self.correct_deferred) / max(self.total_samples, 1)

        mean_unc_deferred = float(np.mean(self.uncertainty_scores_deferred)) if self.uncertainty_scores_deferred else 0.0
        mean_unc_handled = float(np.mean(self.uncertainty_scores_handled)) if self.uncertainty_scores_handled else 0.0

        return {
            "coverage": coverage,
            "deferral_rate": deferral_rate,
            "accuracy_on_handled": acc_on_handled,
            "accuracy_if_all_predicted": acc_if_all_predicted,
            "mean_uncertainty_deferred": mean_unc_deferred,
            "mean_uncertainty_handled": mean_unc_handled,
        }


def compute_max_std(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute maximum class standard deviation as uncertainty metric.

    This is the recommended metric for medical AI - most interpretable.

    Args:
        logits: Tensor of shape [batch_size, n_samples, num_classes]

    Returns:
        Uncertainty scores of shape [batch_size]
    """
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)  # [batch_size, n_samples, num_classes]

    # Compute standard deviation across samples for each class
    prob_std = probs.std(dim=1)  # [batch_size, num_classes]

    # Take maximum std across classes as uncertainty
    uncertainty = prob_std.max(dim=1).values  # [batch_size]

    return uncertainty


def compute_mean_std(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute mean standard deviation across classes as uncertainty metric.

    Args:
        logits: Tensor of shape [batch_size, n_samples, num_classes]

    Returns:
        Uncertainty scores of shape [batch_size]
    """
    probs = torch.softmax(logits, dim=-1)  # [batch_size, n_samples, num_classes]
    prob_std = probs.std(dim=1)  # [batch_size, num_classes]
    uncertainty = prob_std.mean(dim=1)  # [batch_size]

    return uncertainty


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy as uncertainty metric.

    Args:
        probs: Mean probabilities of shape [batch_size, num_classes]

    Returns:
        Uncertainty scores of shape [batch_size]
    """
    # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return entropy


def compute_bald(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute BALD (Bayesian Active Learning by Disagreement) as uncertainty metric.

    BALD = H[y|x] - E[H[y|x,θ]]

    Args:
        logits: Tensor of shape [batch_size, n_samples, num_classes]

    Returns:
        Uncertainty scores of shape [batch_size]
    """
    probs = torch.softmax(logits, dim=-1)  # [batch_size, n_samples, num_classes]

    # Predictive entropy H[y|x]
    mean_probs = probs.mean(dim=1)  # [batch_size, num_classes]
    predictive_entropy = compute_entropy(mean_probs)

    # Expected entropy E[H[y|x,θ]]
    sample_entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)  # [batch_size, n_samples]
    expected_entropy = sample_entropies.mean(dim=1)  # [batch_size]

    # BALD = mutual information
    bald = predictive_entropy - expected_entropy

    return bald


def format_deferred_sample_jsonl(
    sample_idx: int,
    metadata: Dict,
    predicted_answer_idx: int,
    uncertainty_score: float,
    true_answer_idx: int,
    is_correct: bool,
    mean_probs: Optional[List[float]] = None,
    std_devs: Optional[List[float]] = None,
    n_samples: int = 1,
    dataset: str = "",
    split: str = "validation",
    uncertainty_metric: str = "max_std"
) -> str:
    """
    Format a deferred sample for JSONL logging.

    Args:
        sample_idx: Index of the sample in the dataset
        metadata: Dictionary with question, options, answer_idx
        predicted_answer_idx: Predicted answer (0=A, 1=B, 2=C, 3=D)
        uncertainty_score: Computed uncertainty score
        true_answer_idx: True answer index
        is_correct: Whether prediction matches ground truth
        mean_probs: Mean probabilities for each class (optional)
        std_devs: Standard deviations for each class (optional)
        n_samples: Number of samples used for uncertainty estimation
        dataset: Dataset name (e.g., "usmle", "medmcqa")
        split: Dataset split (e.g., "validation", "test")
        uncertainty_metric: Which metric was used (e.g., "max_std", "entropy")

    Returns:
        JSON string for one line in JSONL file
    """
    # Convert indices to letters
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    predicted_letter = idx_to_letter.get(predicted_answer_idx, "?")
    true_letter = idx_to_letter.get(true_answer_idx, "?")

    record = {
        "sample_idx": sample_idx,
        "question": metadata.get("question", ""),
        "options": metadata.get("options", {}),
        "true_answer": true_letter,
        "true_answer_idx": int(true_answer_idx),
        "predicted_answer": predicted_letter,
        "predicted_answer_idx": int(predicted_answer_idx),
        "is_correct": bool(is_correct),
        "uncertainty_score": float(uncertainty_score),
        "uncertainty_metric": uncertainty_metric,
        "n_samples": n_samples,
        "dataset": dataset,
        "split": split,
    }

    # Add optional probability information
    if mean_probs is not None:
        record["mean_probability"] = [float(p) for p in mean_probs]
    if std_devs is not None:
        record["std_deviation"] = [float(s) for s in std_devs]

    return json.dumps(record)
