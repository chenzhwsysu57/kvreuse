"""Common utilities for answer extraction and matrix operations."""

import re
import torch

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
ANSWER_TRIGGER = "The answer is"

MODE_LIST = ["relaycaching", "kvcomm", "cacheblend", "epic", "dense_prefill_only"]

# EPIC prefix length at or above this value is treated as the greedy "full" baseline.
EPIC_FULL_PREFIX_THRESHOLD = 100_000


def resolve_baseline_label(
    execution_mode: str,
    *,
    epic_prefix_token_num: int = 0,
) -> str:
    """Map runtime ``execution_mode`` + EPIC knobs to greedy/paper folder names.

    ``execution_mode`` is the algorithm field in ``RelayCachingConfig`` (what actually runs).
    ``baseline_label`` is the human-facing name used in result metadata / analysis
    (e.g. ``zero`` == ``epic`` with ``epic_prefix_token_num=0``).
    """
    mode = (execution_mode or "").strip().lower()
    if mode == "epic":
        n = int(epic_prefix_token_num)
        if n == 0:
            return "zero"
        if n >= EPIC_FULL_PREFIX_THRESHOLD:
            return "full"
        if n == 16:
            return "epic16"
        return f"epic{n}"
    if mode == "dense_prefill_only":
        return "dense_prefill"
    return mode


def get_placeholder_cache_type(ph_id: str) -> str:
    """
    Determine the cache type for a given placeholder ID.

    BLANK type is used for blocks that require full recomputation from scratch,
    e.g., user inputs or structured conditions.
    RELAY type is used for blocks that support selective recomputation.

    Args:
        ph_id: The placeholder identifier string.

    Returns:
        "BLANK" or "RELAY" indicating the cache strategy.
    """
    if ph_id == 'user_question' or ph_id.startswith('condition_'):
        return "BLANK"
    else:
        return "RELAY"


def extract_answer_from_output(completion: str) -> str:
    """Extract the numerical answer from a model completion string."""
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer: str, answer: str) -> bool:
    """Check if the model answer matches the ground truth."""
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def clean_answer(model_pred: str) -> str:
    """Clean and extract the final numerical answer from model prediction."""
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        pred = preds[1]
    else:
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        pred = pred[0]
    else:
        pred = pred[-1]

    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def nuclear_norm(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the nuclear norm (sum of singular values) of a matrix."""
    _, S, _ = torch.svd(matrix)
    return torch.sum(S)


def frobenius_norm(A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Compute the Frobenius norm of the difference (A - S)."""
    return torch.norm(A - S, p='fro')