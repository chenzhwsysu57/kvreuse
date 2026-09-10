"""Fixed-pool selection and paired candidate feedback.

Threshold acceptance and best-observed search state are reported separately.
Attack-distribution advancement is outside the defense refinement loop.
"""

from __future__ import annotations

from collections import defaultdict
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
    best = max(scores, key=lambda score: (score.accuracy, -len(score.suffix), score.candidate_id))
    return {
        "incumbent": incumbent.__dict__, "selected": selected.__dict__, "accepted": accepted,
        "best_observed": best.__dict__,
        "minimum_improvement_pp": minimum_improvement_pp,
        "actual_improvement_pp": 100 * (selected.accuracy - incumbent.accuracy),
        "scores": [score.__dict__ for score in scores],
    }


def candidate_feedback(evaluation: dict[str, Any], sources: dict[str, dict], *,
                       incumbent_suffix: str, examples_per_kind: int = 2) -> list[dict]:
    """Bounded examples, full aggregate metrics, and paired repair/regression counts.

    History is metadata for the defender, never part of evaluated task prompts.
    """
    from .build_adaptive_dashboard import compact_example

    if examples_per_kind < 0:
        raise ValueError("examples_per_kind must be non-negative")
    entries = [{"candidate": {"candidate_id": "direct_reuse", "suffix": "", "reasoning": "baseline"},
                **evaluation["direct_reuse"]}, *evaluation["candidates"]]

    def indexed(entry):
        rows = entry["results"]
        result = {(row["task_id"], row["target"]): row for row in rows}
        if len(rows) != len(result) or not rows:
            raise ValueError("duplicate or empty evaluation directions")
        return result

    baseline = indexed(entries[0])
    incumbent_entry = next(e for e in entries if e["candidate"]["suffix"].strip() == incumbent_suffix.strip())
    incumbent = indexed(incumbent_entry)
    feedback = []
    for entry in entries:
        rows = indexed(entry)
        if rows.keys() != baseline.keys() or rows.keys() != incumbent.keys():
            raise ValueError("candidate/full pool directions differ")
        gaps, sizes = defaultdict(int), defaultdict(int)
        for row in rows.values():
            name = row["attack_type"]
            gaps[name] += int(row["full"]["correct"]) - int(row["reuse"]["correct"])
            sizes[name] += 1
        priority = sorted(sizes, key=lambda name: (-gaps[name] / sizes[name], -sizes[name], name))
        ranks = {name: rank for rank, name in enumerate(priority)}
        ordered_rows = sorted(rows.items(), key=lambda item: (ranks[item[1]["attack_type"]], item[0]))
        changes = {}
        examples = defaultdict(list)
        for label, reference in (("direct_reuse", baseline), ("incumbent", incumbent)):
            counts = defaultdict(int)
            by_type = defaultdict(lambda: defaultdict(int))
            for key, row in ordered_rows:
                before, after = reference[key]["reuse"]["correct"], row["reuse"]["correct"]
                kind = "repair" if not before and after else "regression" if before and not after else "persistent_failure" if not after else "unchanged_correct"
                counts[kind] += 1
                by_type[row["attack_type"]][kind] += 1
                eligible = kind != "unchanged_correct" and (kind != "persistent_failure" or row["full"]["correct"])
                if eligible and len(examples[f"{label}:{kind}"]) < examples_per_kind:
                    example = compact_example(row, sources[row["task_id"]])
                    example.update({"attack_type": row["attack_type"], "full_correct": row["full"]["correct"],
                                    "reference_prediction": reference[key]["reuse"]["prediction"],
                                    "reference_correct": before, "candidate_correct": after})
                    examples[f"{label}:{kind}"].append(example)
            changes[label] = {"counts": dict(counts), "by_type": {k: dict(v) for k, v in by_type.items()},
                              "delta_pp": 100 * (counts["repair"] - counts["regression"]) / len(rows)}
        feedback.append({"candidate": dict(entry["candidate"]), "metrics": entry["summary"],
                         "failure_priority": [{"attack_type": name, "full_minus_reuse_pp": 100 * gaps[name] / sizes[name],
                                               "directions": sizes[name]} for name in priority],
                         "comparisons": changes, "examples": dict(examples)})
    return feedback