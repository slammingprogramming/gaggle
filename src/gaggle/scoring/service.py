from __future__ import annotations

from typing import Literal

from gaggle.core.config import RuntimeConfig
from gaggle.schemas.event import Hypothesis, SeverityAssessment
from gaggle.schemas.signal import Signal
from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ScoringService:
    version = "1.0.0"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def score(self, hypotheses: list[Hypothesis], signals: list[Signal]) -> SeverityAssessment:
        if not hypotheses:
            assessment = SeverityAssessment(
                confidence=0.0,
                severity="low",
                reasons=["no hypotheses generated"],
                version=self.version,
            )
            LOGGER.info(
                "scoring_completed", severity=assessment.severity, confidence=assessment.confidence
            )
            return assessment
        confidence = max(hypothesis.confidence for hypothesis in hypotheses)
        signal_types = {signal.signal_type for signal in signals}
        severity: Literal["low", "medium", "high"] = "low"
        reasons = ["preserve uncertain evidence rather than dismiss automatically"]
        if confidence >= self.config.scoring.high_threshold and len(signal_types) >= 2:
            severity = "high"
            reasons = ["high confidence with corroborating signal types"]
        elif confidence >= self.config.scoring.medium_threshold and len(signal_types) >= 2:
            severity = "medium"
            reasons = ["moderate confidence with corroboration"]
        elif confidence >= self.config.scoring.low_threshold:
            severity = "low"
            reasons = ["single-source or weakly corroborated event retained for review"]
        assessment = SeverityAssessment(
            confidence=confidence,
            severity=severity,
            reasons=reasons,
            version=self.version,
        )
        LOGGER.info("scoring_completed", severity=severity, confidence=confidence)
        return assessment
