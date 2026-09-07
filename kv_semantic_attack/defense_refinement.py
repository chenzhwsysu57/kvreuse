"""Thresholded suffix refinement on a fixed attack pool.

Attack distribution advancement is intentionally outside this loop. A new
attack round may begin only after this loop accepts a stronger suffix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    suffix: str
    accuracy: float


def extract_scores(evaluation: dict[str, Any]) -> list[CandidateScore]:
    scores = [CandidateScore("direct_reuse", "", evaluation["direct_reuse"]["summary"]["overall"]["reuse_accuracy"])]
    for item in evaluation["candidates"]:
        candidate = item["candidate"]
        scores.append(CandidateScore(candidate["candidate_id"], candidate["suffix"],
                                     item["summary"]["overall"]["reuse_accuracy"]))
    return scores


def choose_candidate(scores: list[CandidateScore], *, incumbent_suffix: str,
                     minimum_improvement_pp: float) -> tuple[CandidateScore, bool]:
    """Choose the highest score; accept only an improvement over the incumbent."""
    if minimum_improvement_pp < 0:
        raise ValueError("minimum_improvement_pp must be non-negative")
    incumbent = next((score for score in scores if score.suffix.strip() == incumbent_suffix.strip()), None)
    if incumbent is None:
        raise ValueError("evaluation did not include the incumbent suffix")
    best = max(scores, key=lambda score: (score.accuracy, -len(score.suffix), score.candidate_id))
    accepted = best.suffix != incumbent.suffix and (best.accuracy - incumbent.accuracy) * 100 >= minimum_improvement_pp
    return (best if accepted else incumbent), accepted


def refinement_summary(evaluation: dict[str, Any], *, incumbent_suffix: str,
                       minimum_improvement_pp: float) -> dict[str, Any]:
    scores = extract_scores(evaluation)
    selected, accepted = choose_candidate(scores, incumbent_suffix=incumbent_suffix,
                                          minimum_improvement_pp=minimum_improvement_pp)
    incumbent = next(score for score in scores if score.suffix.strip() == incumbent_suffix.strip())
    return {
        "incumbent": incumbent.__dict__, "selected": selected.__dict__, "accepted": accepted,
        "minimum_improvement_pp": minimum_improvement_pp,
        "actual_improvement_pp": 100 * (selected.accuracy - incumbent.accuracy),
        "scores": [score.__dict__ for score in scores],
    }