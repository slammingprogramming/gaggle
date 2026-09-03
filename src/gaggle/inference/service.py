from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from gaggle.plugins.registry import INFERENCE_RULE_PLUGIN_GROUP, load_plugins
from gaggle.schemas.event import Hypothesis
from gaggle.schemas.signal import Signal
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)


class InferenceService:
    """Deterministic, explainable rule-based inference.

    Signals are evidence; hypotheses are the inference engine's explainable
    interpretation of that evidence, always carrying their contributing
    signal ids, a rule name, and a plain-language confidence formula. No
    single weak signal escalates to high confidence on its own -- every
    rule either requires corroboration across signal types/cameras or caps
    its own confidence when it can't get any. Plugin rules (registered
    under ``gaggle.plugins.inference_rules``) run after the
    built-in rules and contribute independently; a rule never sees or
    modifies another rule's output, keeping each one auditable in
    isolation.
    """

    version = "1.2.0"

    def __init__(self, load_rule_plugins: bool = True) -> None:
        self._rule_plugins = load_plugins(INFERENCE_RULE_PLUGIN_GROUP) if load_rule_plugins else []

    def infer(self, signals: list[Signal]) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        by_window: dict[UUID, list[Signal]] = defaultdict(list)
        for signal in signals:
            if signal.window_id is None:
                continue
            by_window[signal.window_id].append(signal)

        for window_id in sorted(by_window, key=str):
            window_signals = by_window[window_id]
            hypotheses.extend(self._apply_builtin_rules(window_id, window_signals))

        for plugin in self._rule_plugins:
            try:
                plugin_hypotheses = plugin.apply(signals)
            except Exception as error:
                LOGGER.error(
                    "inference_rule_plugin_failed",
                    plugin=getattr(plugin, "name", repr(plugin)),
                    reason=str(error),
                )
                continue
            hypotheses.extend(plugin_hypotheses)

        LOGGER.info("inference_completed", hypothesis_count=len(hypotheses))
        return hypotheses

    def _apply_builtin_rules(
        self, window_id: UUID, window_signals: list[Signal]
    ) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        motions = [s for s in window_signals if s.signal_type == "motion"]
        audios = [s for s in window_signals if s.signal_type == "audio_spike"]
        objects = [s for s in window_signals if s.signal_type == "object_hint"]
        gunshots = [s for s in window_signals if s.signal_type == "gunshot"]

        if len({s.camera_id for s in motions if s.camera_id}) >= 2:
            confidence = min(1.0, sum(s.confidence for s in motions) / len(motions) + 0.20)
            hypotheses.append(
                self._hypothesis(
                    window_id,
                    rule_name="cross_camera_motion_correlation",
                    label="correlated_multi_camera_activity",
                    confidence=confidence,
                    signals=motions,
                    escalation_reasons=["motion observed across multiple cameras"],
                    confidence_math="mean(motion_confidence) + 0.20 corroboration bonus",
                )
            )

        if motions and audios:
            corroborating = motions + audios
            confidence = min(
                1.0, sum(s.confidence for s in corroborating) / len(corroborating) + 0.10
            )
            hypotheses.append(
                self._hypothesis(
                    window_id,
                    rule_name="motion_plus_audio_spike",
                    label="possible_collision_or_aggressive_event",
                    confidence=confidence,
                    signals=corroborating,
                    escalation_reasons=["motion coincides with audio spike"],
                    confidence_math="mean(motion,audio) + 0.10 corroboration bonus",
                )
            )

        if motions and objects:
            corroborating = motions + objects
            confidence = min(
                1.0, sum(s.confidence for s in corroborating) / len(corroborating) + 0.10
            )
            hypotheses.append(
                self._hypothesis(
                    window_id,
                    rule_name="motion_plus_object_region",
                    label="reviewable_moving_object_activity",
                    confidence=confidence,
                    signals=corroborating,
                    escalation_reasons=["motion coincides with a detected moving region"],
                    confidence_math="mean(motion,object_hint) + 0.10 corroboration bonus",
                )
            )

        if motions and audios and objects:
            corroborating = motions + audios + objects
            confidence = min(
                1.0, sum(s.confidence for s in corroborating) / len(corroborating) + 0.25
            )
            hypotheses.append(
                self._hypothesis(
                    window_id,
                    rule_name="triple_corroboration",
                    label="high_confidence_multi_signal_event",
                    confidence=confidence,
                    signals=corroborating,
                    escalation_reasons=["motion, audio spike, and a moving region all coincide"],
                    confidence_math="mean(motion,audio,object_hint) + 0.25 corroboration bonus",
                )
            )

        if motions and not audios and not objects:
            confidence = min(0.60, sum(s.confidence for s in motions) / len(motions))
            hypotheses.append(
                self._hypothesis(
                    window_id,
                    rule_name="isolated_motion_retention",
                    label="reviewable_motion_activity",
                    confidence=confidence,
                    signals=motions,
                    escalation_reasons=[
                        "false positives tolerated at low severity to avoid missed evidence"
                    ],
                    confidence_math="mean(motion_confidence) capped at 0.60",
                )
            )

        if gunshots and motions:
            corroborating = gunshots + motions
            confidence = min(
                1.0, sum(s.confidence for s in corroborating) / len(corroborating) + 0.10
            )
            hypotheses.append(
                self._hypothesis(
                    window_id,
                    rule_name="gunshot_plus_motion",
                    label="possible_firearm_discharge_with_visual_activity",
                    confidence=confidence,
                    signals=corroborating,
                    escalation_reasons=["a gunshot-like sound coincides with visual motion"],
                    confidence_math="mean(gunshot,motion) + 0.10 corroboration bonus",
                )
            )
        elif gunshots:
            # No corroborating signal type in this window -- see
            # AGENTS.md invariant 7 and detection/gunshot_analysis.py's
            # module docstring: a classifier opinion alone, however
            # confident, is capped and (via ScoringService's
            # len(signal_types) >= 2 requirement) can never alone reach
            # medium/high severity. Mirrors isolated_motion_retention.
            confidence = min(0.60, sum(s.confidence for s in gunshots) / len(gunshots))
            hypotheses.append(
                self._hypothesis(
                    window_id,
                    rule_name="isolated_gunshot_retention",
                    label="reviewable_gunshot_audio_event",
                    confidence=confidence,
                    signals=gunshots,
                    escalation_reasons=[
                        "an unvalidated audio classifier flagged a gunshot-like sound with no "
                        "other corroborating evidence"
                    ],
                    confidence_math="mean(gunshot_confidence) capped at 0.60",
                )
            )

        return hypotheses

    def _hypothesis(
        self,
        window_id: UUID,
        *,
        rule_name: str,
        label: str,
        confidence: float,
        signals: list[Signal],
        escalation_reasons: list[str],
        confidence_math: str,
    ) -> Hypothesis:
        return Hypothesis(
            hypothesis_id=new_uuid(),
            rule_name=rule_name,
            label=label,
            confidence=confidence,
            contributing_signal_ids=[s.id for s in signals],
            escalation_reasons=escalation_reasons,
            confidence_math=confidence_math,
            metadata={"window_id": str(window_id), "rule_version": self.version},
        )
