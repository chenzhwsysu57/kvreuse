"""Compare serial and batched full/reuse outputs for one controlled case."""

import argparse
import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_semantic_attack.qwen_executor import Qwen3KVReuseExecutor
from kv_semantic_attack.schemas import AttackCase


def answer(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip().upper()
    stripped = text.strip().upper()
    return stripped if re.fullmatch(r"[A-H]", stripped) else None


def compare(name: str, serial: str, batched: str) -> bool:
    normalized_serial = " ".join(serial.split())
    normalized_batch = " ".join(batched.split())
    exact = normalized_serial == normalized_batch
    serial_answer, batch_answer = answer(serial), answer(batched)
    same_answer = serial_answer is not None and serial_answer == batch_answer
    print(f"{name}: normalized_text={exact}; parsed_answer={serial_answer!r}/{batch_answer!r}; answer_consistent={same_answer}")
    print(f"  chars: serial={len(serial)} batch={len(batched)}")
    if not exact:
        matcher = difflib.SequenceMatcher(None, normalized_serial, normalized_batch)
        match = matcher.find_longest_match(0, len(normalized_serial),
                                           0, len(normalized_batch))
        first_diff = next(
            (i for i, (left, right) in enumerate(zip(normalized_serial, normalized_batch))
             if left != right),
            min(len(normalized_serial), len(normalized_batch)),
        )
        print(f"  first_normalized_diff={first_diff}; longest_common_block={match.size}")
        print(f"  serial_prefix: {normalized_serial[max(0, first_diff - 80):first_diff + 160]!r}")
        print(f"  batch_prefix : {normalized_batch[max(0, first_diff - 80):first_diff + 160]!r}")
    return exact and same_answer


def print_complete_sequences(title: str, inputs: list[str], outputs: list[str]) -> None:
    print(f"\n### {title} ###")
    for index, (prompt, output) in enumerate(zip(inputs, outputs), 1):
        print(f"### sample {index} input ###")
        print(prompt)
        print(f"### sample {index} generate ###")
        print(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=("0.6b", "1.7b", "4b", "8b"), default="1.7b")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    executor = Qwen3KVReuseExecutor(
        model_size=args.model_size, reasoning=True,
        allow_download=args.allow_download,
    )
    executor.max_new_tokens = args.max_new_tokens
    case = AttackCase(
        prefix_a=(
            "You are a spacecraft failure investigator. Read ONLY the section titled "
            "MISSION ALPHA in the shared material. Reconstruct the primary causal chain "
            "of the Mars probe failure from telemetry and distinguish the initiating fault "
            "from later symptoms. Ignore the unrelated historical document in the other "
            "section. Select the single option that best identifies the initiating fault."
        ),
        prefix_b=(
            "You are an economic historian. Read ONLY the section titled DOCUMENT BETA in "
            "the shared material. Determine which clue provides the strongest evidence for "
            "the document's origin and trade route. Ignore the unrelated engineering report "
            "in the other section. Select the single option that best supports the historical "
            "identification."
        ),
        shared_block=(
            "MISSION ALPHA — Mars probe telemetry report:\n"
            "T+00:00 launch vehicle separation nominal. T+18:42 attitude-control valve "
            "commanded open, but valve-current telemetry remained zero. T+18:45 the flight "
            "computer switched to safe mode after detecting orientation drift. T+19:03 the "
            "probe lost high-gain antenna lock. A later review found that a prelaunch wiring "
            "inspection had recorded intermittent continuity on the valve actuator.\n"
            "[A] High-gain antenna lock was lost.\n"
            "[B] The flight computer entered safe mode.\n"
            "[C] The attitude-control valve actuator failed to draw current.\n"
            "[D] Launch vehicle separation was nominal.\n"
            "[E] Orientation drift was detected.\n"
            "[F] The prelaunch inspection was completed.\n"
            "[G] Telemetry was transmitted after safe mode.\n"
            "[H] The mission lost its communication link.\n\n"
            "DOCUMENT BETA — Port archive fragment:\n"
            "The undated tablet lists cedar resin, purple dye, and tin ingots, with weights "
            "measured in a coastal standard. Its seal depicts a double-hulled vessel. The "
            "clay contains mineral inclusions matching the lower Orontes basin. A later copy "
            "uses an inland royal title, but the original hand uses a port-administered tax "
            "formula.\n"
            "[A] The tablet mentions cedar resin.\n"
            "[B] A later copy uses an inland royal title.\n"
            "[C] The seal depicts a double-hulled vessel.\n"
            "[D] The clay matches the lower Orontes basin and uses a port tax formula.\n"
            "[E] The tablet lists tin ingots.\n"
            "[F] The document is undated.\n"
            "[G] The weights use a coastal standard.\n"
            "[H] The tablet contains purple dye."
        ),
        question=(
            "Follow the assigned domain and section restriction, reason through the evidence, "
            "and return exactly one option letter A-H in \\boxed{...}."
        ),
        gold_a="C", gold_b="D", case_id="complex-consistency",
    )
    remedy = ""

    full_requests = [
        {"prefix": case.prefix_a, "shared_block": case.shared_block,
         "question": case.question, "gold": case.gold_a, "label": "batch_full_a"},
        {"prefix": case.prefix_b, "shared_block": case.shared_block,
         "question": case.question, "gold": case.gold_b, "label": "batch_full_b"},
    ]
    full_width = max(
        int(executor._parts(x["prefix"], x["shared_block"], x["question"]).full_ids.numel())
        for x in full_requests
    )

    # Both serial calls use the same fixed padded width as the batch call;
    # batch_size=1 is the unbatched control for this comparison.
    serial_full = {
        "full_a": executor.run_full_batch([{**full_requests[0], "label": "serial_full_a"}],
                                           pad_to_length=full_width)[0],
        "full_b": executor.run_full_batch([{**full_requests[1], "label": "serial_full_b"}],
                                           pad_to_length=full_width)[0],
    }
    batch_full = executor.run_full_batch(full_requests, pad_to_length=full_width)
    full_inputs = [
        executor._parts(x["prefix"], x["shared_block"], x["question"]).rendered
        for x in full_requests
    ]

    reuse_requests = [
        {"donor_prefix": case.prefix_b, "target_prefix": case.prefix_a,
         "shared_block": case.shared_block, "question": case.question,
         "remedy": remedy, "gold": case.gold_a, "label": "reuse_b_to_a"},
        {"donor_prefix": case.prefix_a, "target_prefix": case.prefix_b,
         "shared_block": case.shared_block, "question": case.question,
         "remedy": remedy, "gold": case.gold_b, "label": "reuse_a_to_b"},
    ]
    source_parts = [executor._parts(x["donor_prefix"], x["shared_block"], x["question"])
                    for x in reuse_requests]
    target_queries = [x["question"] for x in reuse_requests]
    target_parts = [executor._parts(x["target_prefix"], x["shared_block"], query)
                    for x, query in zip(reuse_requests, target_queries)]
    reuse_pad_widths = (
        max(int(x.full_ids.numel()) for x in source_parts),
        max(int(x.prefix_ids.numel()) for x in target_parts),
        max(int(x.suffix_ids.numel()) for x in target_parts),
        max(int(x.prefix_ids.numel() + x.block_ids.numel()) for x in target_parts),
    )

    serial_reuse = {
        "reuse_b_to_a": executor._run_reuse_batch([{**reuse_requests[0], "label": "serial_reuse_b_to_a"}],
                                                   pad_widths=reuse_pad_widths)[0],
        "reuse_a_to_b": executor._run_reuse_batch([{**reuse_requests[1], "label": "serial_reuse_a_to_b"}],
                                                   pad_widths=reuse_pad_widths)[0],
    }
    batch_reuse = executor._run_reuse_batch(reuse_requests, pad_widths=reuse_pad_widths)

    mixed_requests = [
        {"mode": "full", **full_requests[0], "label": "mixed_full_a"},
        {"mode": "full", **full_requests[1], "label": "mixed_full_b"},
        {"mode": "reuse", **reuse_requests[0], "label": "mixed_reuse_b_to_a"},
        {"mode": "reuse", **reuse_requests[1], "label": "mixed_reuse_a_to_b"},
    ]
    mixed_parts = [
        executor._parts(x["prefix"], x["shared_block"], x["question"])
        if x["mode"] == "full" else
        executor._parts(x["target_prefix"], x["shared_block"], x["question"])
        for x in mixed_requests
    ]
    mixed_width = max(
        int(x.full_ids.numel()) if request["mode"] == "full" else
        int(x.prefix_ids.numel() + x.block_ids.numel() + x.suffix_ids.numel())
        for request, x in zip(mixed_requests, mixed_parts)
    )
    mixed_batch = executor.run_mixed_batch(mixed_requests, pad_to_length=mixed_width)
    mixed_serial = [
        executor.run_mixed_batch([request], pad_to_length=mixed_width)[0]
        for request in mixed_requests
    ]

    print_complete_sequences(
        "full batch complete seq",
        full_inputs,
        [item.text for item in batch_full],
    )
    print_complete_sequences(
        "full serial complete seq",
        full_inputs,
        [serial_full["full_a"].text, serial_full["full_b"].text],
    )
    reuse_inputs = [
        executor._parts(x["target_prefix"], x["shared_block"], x["question"]).rendered
        for x in reuse_requests
    ]
    print_complete_sequences(
        "reuse batch complete seq",
        reuse_inputs,
        [item.text for item in batch_reuse],
    )
    print_complete_sequences(
        "reuse serial complete seq",
        reuse_inputs,
        [serial_reuse["reuse_b_to_a"].text, serial_reuse["reuse_a_to_b"].text],
    )
    mixed_inputs = [x.rendered for x in mixed_parts]
    print_complete_sequences(
        "mixed batch complete seq", mixed_inputs,
        [item.text for item in mixed_batch],
    )
    print_complete_sequences(
        "mixed serial complete seq", mixed_inputs,
        [item.text for item in mixed_serial],
    )

    print("\nComparisons (serial = model batch_size=1; batch = model batch_size=2):")
    checks = [
        compare("full_a", serial_full["full_a"].text, batch_full[0].text),
        compare("full_b", serial_full["full_b"].text, batch_full[1].text),
        compare("reuse_b_to_a", serial_reuse["reuse_b_to_a"].text, batch_reuse[0].text),
        compare("reuse_a_to_b", serial_reuse["reuse_a_to_b"].text, batch_reuse[1].text),
        compare("mixed_full_a", mixed_serial[0].text, mixed_batch[0].text),
        compare("mixed_full_b", mixed_serial[1].text, mixed_batch[1].text),
        compare("mixed_reuse_b_to_a", mixed_serial[2].text, mixed_batch[2].text),
        compare("mixed_reuse_a_to_b", mixed_serial[3].text, mixed_batch[3].text),
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
