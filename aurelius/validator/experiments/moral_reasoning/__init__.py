"""Moral Reasoning Experiment (MoReBench) — 22 binary signal evaluation."""

from aurelius.validator.experiments.moral_reasoning.experiment import MoralReasoningExperiment
from aurelius.validator.experiments.moral_reasoning.scoring import MoralReasoningScoringSystem
from aurelius.validator.experiments.moral_reasoning.signals import BinarySignals, DimensionScores

__all__ = [
    "BinarySignals",
    "DimensionScores",
    "MoralReasoningExperiment",
    "MoralReasoningScoringSystem",
]
