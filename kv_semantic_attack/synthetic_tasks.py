"""Executable, paired instruction-following tasks; standard library only.

Gold answers are computed from rules, never supplied by an LLM.  Each pair
shares exactly one rendered data block.  These are conflict *candidates*, not
verified KV-reuse failures.  No model, tokenizer, or API is loaded here.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from collections.abc import Iterator, Mapping
from typing import Any

from kvreuse_data.schema import validate_record


GENERATOR_VERSION = "2.0"
TASK_TYPES = {
    "field_switch": ("selection", "字段切换"),
    "condition_switch": ("selection", "条件切换"),
    "scope_switch": ("selection", "范围切换"),
    "extremum_reverse": ("operation", "极值反转"),
    "aggregation_switch": ("operation", "聚合切换"),
    "sorting_reverse": ("operation", "排序切换"),
    "priority_switch": ("operation", "优先级切换"),
    "task_switch": ("operation", "任务切换：求和／提取数字"),
    "label_mapping": ("encoding", "标签映射"),
    "format_switch": ("encoding", "格式切换：JSON／CSV"),
    "case_switch": ("encoding", "大小写切换"),
}
LAYOUTS = ("table", "json", "lines")
QUESTION = "Select the option whose payload is the answer required by the task. Return only its option letter."
PREAMBLES = (
    "Your task for the shared data below is as follows. ",
    "Process the following shared data using this rule: ",
    "For this request, apply the following instruction to the shared data: ",
)


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_block(data: dict[str, Any], layout: str) -> str:
    """Render only task data, with stable boundaries independent of A/B."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout: {layout}")
    if data["kind"] == "expression":
        body = "Expression: " + " + ".join(str(x) for x in data["numbers"])
    else:
        rows = data["rows"]
        columns = list(rows[0])
        if layout == "json":
            body = json.dumps(rows, ensure_ascii=False, indent=2)
        elif layout == "table":
            body = " | ".join(columns) + "\n"
            body += "\n".join(" | ".join(str(row[col]) for col in columns) for row in rows)
        else:
            body = "\n".join(
                "; ".join(f"{col}={row[col]}" for col in columns) for row in rows
            )
    options = "\n".join(
        f"{letter}. {json.dumps(value, ensure_ascii=False)}"
        for letter, value in zip(data["option_letters"], data["option_values"])
    )
    return body + "\n\nOptions:\n" + options


def solve(data: dict[str, Any], rule: dict[str, Any]) -> str:
    """Deterministic task oracle. Ambiguous single-item selection is invalid."""
    op = rule["op"]
    if op in ("sum_numbers", "extract_numbers"):
        numbers = data["numbers"]
        if op == "sum_numbers":
            return str(sum(numbers))
        return ",".join(map(str, numbers))

    rows = list(data["rows"])
    if "filter" in rule:
        field, value = rule["filter"]
        rows = [row for row in rows if row[field] == value]
    if "scope" in rule:
        midpoint = len(rows) // 2
        rows = rows[:midpoint] if rule["scope"] == "first" else rows[midpoint:]
    if not rows:
        raise ValueError("selection contains no rows")

    if op == "select":
        fields = rule["fields"]
        keys = [tuple(row[field] for field in fields) for row in rows]
        best = (min if rule["order"] == "min" else max)(keys)
        winners = [row for row, key in zip(rows, keys) if key == best]
        if len(winners) != 1:
            raise ValueError("selection does not have a unique answer")
        return str(winners[0]["id"])
    if op == "aggregate":
        return str(len(rows) if rule["mode"] == "count" else sum(row[rule["field"]] for row in rows))
    if op == "sort":
        field = rule["field"]
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError("sorting values must be distinct")
        ordered = sorted(rows, key=lambda row: row[field], reverse=rule["descending"])
        return ",".join(row["id"] for row in ordered)

    matches = [row for row in rows if row["id"] == rule["target_id"]]
    if len(matches) != 1:
        raise ValueError("target_id must identify exactly one row")
    target = matches[0]
    if op == "label":
        condition = target["value"] >= rule["threshold"]
        return rule["true_label"] if condition else rule["false_label"]
    if op == "format":
        if rule["format"] == "json":
            return json.dumps({"id": target["id"], "value": target["value"]}, separators=(",", ":"))
        if rule["format"] == "csv":
            return f'id,value\n{target["id"]},{target["value"]}'
        raise ValueError("unknown output format")
    if op == "case":
        return target["name"].upper() if rule["case"] == "upper" else target["name"].lower()
    raise ValueError(f"unknown operation: {op}")


def render_instruction(rule: dict[str, Any], style: int) -> str:
    op = rule["op"]
    if op == "select":
        scope = "all rows"
        if "filter" in rule:
            field, value = rule["filter"]
            scope = f"only rows whose {field} is {value}"
        if "scope" in rule:
            scope = f"only the {rule['scope']} half of the rows, in the displayed order"
        fields = rule["fields"]
        direction = "smallest" if rule["order"] == "min" else "largest"
        text = f"Consider {scope}. Select the row with the {direction} {fields[0]}"
        for field in fields[1:]:
            text += f"; break ties using the {direction} {field}"
        text += ". Output only its id."
    elif op == "aggregate":
        text = (
            f"Compute the sum of the {rule['field']} column over all rows."
            if rule["mode"] == "sum" else "Count the data rows, excluding headers."
        ) + " Output only the integer."
    elif op == "sort":
        direction = "descending" if rule["descending"] else "ascending"
        text = f"Sort all rows by {rule['field']} in {direction} numeric order. "
        text += "Output only their ids separated by commas, without spaces."
    elif op == "sum_numbers":
        text = "Evaluate the addition expression. Output only its integer result."
    elif op == "extract_numbers":
        text = "Extract the integer operands of the expression in their original order; do not calculate the sum. "
        text += "Output only those numbers separated by commas, without spaces."
    elif op == "label":
        text = f"For row {rule['target_id']}, check whether value is greater than or equal to {rule['threshold']}. "
        text += f"If true output {rule['true_label']}; otherwise output {rule['false_label']}. Output only that label."
    elif op == "format":
        text = f"Extract id and value from row {rule['target_id']}. "
        if rule["format"] == "json":
            text += 'Output a compact JSON object with keys "id" then "value"; id is a string and value is an integer. Use no whitespace.'
        else:
            text += "Output exactly two CSV lines: the header id,value and then the extracted values. Use no spaces."
    elif op == "case":
        casing = "UPPERCASE" if rule["case"] == "upper" else "lowercase"
        text = f"Read the name of row {rule['target_id']} and convert it to {casing}. Output only the converted name."
    else:
        raise ValueError(f"unknown operation: {op}")
    return PREAMBLES[style] + text + " Do not add an explanation, markdown fences, or an answer wrapper."


def _select(field: str, order: str = "max", **extra: Any) -> dict[str, Any]:
    return {"op": "select", "fields": [field], "order": order, **extra}


def _make_spec(task_type: str, rng: random.Random, n: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ids = [f"R{x}" for x in rng.sample(range(100, 10000), n)]
    rows: list[dict[str, Any]] = []
    if task_type in ("field_switch", "priority_switch"):
        costs = rng.sample(range(10, 1000), n)
        speeds = rng.sample(range(10, 1000), n)
        if task_type == "priority_switch":
            # Both primary minima tie; the secondary key is genuinely needed.
            costs = rng.sample(range(100, 1000), n)
            speeds = rng.sample(range(100, 1000), n)
            low_cost, low_speed = rng.randint(10, 50), rng.randint(10, 50)
            costs[0] = costs[1] = low_cost
            speeds[2] = speeds[3] = low_speed
        rows = [dict(id=id_, cost=c, speed=s) for id_, c, s in zip(ids, costs, speeds)]
        rng.shuffle(rows)
        a, b = _select("cost", "min"), _select("speed", "min")
        if task_type == "priority_switch":
            a["fields"], b["fields"] = ["cost", "speed"], ["speed", "cost"]
    elif task_type == "task_switch":
        data = {"kind": "expression", "numbers": [rng.randint(2, 40) for _ in range(n)]}
        return data, {"op": "sum_numbers"}, {"op": "extract_numbers"}
    elif task_type == "case_switch":
        names = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(5, 12))) for _ in ids]
        rows = [dict(id=id_, name=name) for id_, name in zip(ids, names)]
        target = rng.choice(ids)
        a = {"op": "case", "target_id": target, "case": "upper"}
        b = {**a, "case": "lower"}
    else:
        values = rng.sample(range(2, 1000), n)
        rows = [dict(id=id_, value=value) for id_, value in zip(ids, values)]
        if task_type == "condition_switch":
            colors = ["red", "blue"] + [rng.choice(("red", "blue", "green")) for _ in range(n - 2)]
            rng.shuffle(colors)
            for row, color in zip(rows, colors):
                row["color"] = color
            a = _select("value", filter=["color", "red"])
            b = _select("value", filter=["color", "blue"])
        elif task_type == "scope_switch":
            a, b = _select("value", scope="first"), _select("value", scope="second")
        elif task_type == "extremum_reverse":
            a, b = _select("value", "min"), _select("value", "max")
        elif task_type == "aggregation_switch":
            a = {"op": "aggregate", "field": "value", "mode": "sum"}
            b = {**a, "mode": "count"}
        elif task_type == "sorting_reverse":
            a = {"op": "sort", "field": "value", "descending": False}
            b = {**a, "descending": True}
        elif task_type == "label_mapping":
            target = rng.choice(rows)
            threshold = target["value"] + rng.choice((-1, 0, 1))
            a = {"op": "label", "target_id": target["id"], "threshold": threshold,
                 "true_label": "A", "false_label": "B"}
            b = {**a, "true_label": "B", "false_label": "A"}
        elif task_type == "format_switch":
            a = {"op": "format", "target_id": rng.choice(ids), "format": "json"}
            b = {**a, "format": "csv"}
        else:
            raise ValueError(f"unknown task type: {task_type}")
    return {"kind": "rows", "rows": rows}, a, b


def validate_generated_record(record: Mapping[str, Any]) -> None:
    """Recompute answers and renderings; metadata is private to the evaluator."""
    validate_record(record)
    meta = record["metadata"]
    if record["attack_type"] not in TASK_TYPES:
        raise ValueError("unknown attack_type")
    if record["shared_block"] != render_block(meta["data"], meta["layout"]):
        raise ValueError("shared block does not match structured data")
    if record["shared_data_id"] != _digest(meta["data"]):
        raise ValueError("shared_data_id mismatch")
    if record["question"] != QUESTION:
        raise ValueError("question must not override the prefix instruction")
    for side in ("a", "b"):
        rule = meta[f"rule_{side}"]
        if record[f"prefix_{side}"] != render_instruction(rule, meta["instruction_style"]):
            raise ValueError(f"prefix_{side} does not match its rule")
        answer = solve(meta["data"], rule)
        if answer not in meta["data"]["option_values"]:
            raise ValueError(f"rule_{side} answer has no matching option")
        expected = meta["data"]["option_letters"][meta["data"]["option_values"].index(answer)]
        if record[f"gold_{side}"] != expected:
            raise ValueError(f"gold_{side} does not match its option")


def generate_tasks(
    counts: Mapping[str, int], *, seed: int = 20260906,
    min_rows: int = 4, max_rows: int = 12,
    layouts: tuple[str, ...] = LAYOUTS,
    max_attempts: int = 100,
) -> Iterator[dict[str, Any]]:
    """Generate requested counts, deterministic per type/index and seed.

    Pairs are retried if answers coincide or structured data repeat. A/B are
    randomly swapped so a particular rule is not always the target in one
    direction. Counts for other types do not affect a type's random stream.
    """
    unknown = set(counts) - TASK_TYPES.keys()
    if unknown:
        raise ValueError(f"unknown task types: {sorted(unknown)}")
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("counts must be non-negative integers")
    if not (4 <= min_rows <= max_rows <= 100):
        raise ValueError("row bounds must satisfy 4 <= min_rows <= max_rows <= 100")
    if not layouts or any(layout not in LAYOUTS for layout in layouts):
        raise ValueError(f"layouts must be chosen from {LAYOUTS}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    seen: set[str] = set()
    for task_type in TASK_TYPES:
        for index in range(counts.get(task_type, 0)):
            for attempt in range(max_attempts):
                sample_seed = int(_digest([GENERATOR_VERSION, seed, task_type, index, attempt]), 16)
                rng = random.Random(sample_seed)
                n = rng.randint(min_rows, max_rows)
                # Equal-sized halves avoid ambiguity in scope instructions.
                if task_type == "scope_switch":
                    sizes = [size for size in range(min_rows, max_rows + 1) if size % 2 == 0]
                    if not sizes:
                        raise ValueError("scope_switch requires an even row count within the bounds")
                    n = rng.choice(sizes)
                data, a, b = _make_spec(task_type, rng, n)
                if rng.choice((False, True)):
                    a, b = b, a
                answer_a, answer_b = solve(data, a), solve(data, b)
                distractors = []
                while len(distractors) < 2:
                    candidate = "unavailable-" + _digest([sample_seed, len(distractors), rng.getrandbits(64)])[:12]
                    if candidate not in {answer_a, answer_b}:
                        distractors.append(candidate)
                option_values = [answer_a, answer_b, *distractors]
                rng.shuffle(option_values)
                data["option_letters"] = list("ABCD")
                data["option_values"] = option_values
                gold_a = data["option_letters"][option_values.index(answer_a)]
                gold_b = data["option_letters"][option_values.index(answer_b)]
                group_id = _digest(data)
                if gold_a == gold_b or group_id in seen:
                    continue
                layout = "expression" if data["kind"] == "expression" else rng.choice(layouts)
                # Expression rendering has a single meaningful layout.
                render_layout = "lines" if layout == "expression" else layout
                style = rng.randrange(len(PREAMBLES))
                task_id = f"synthetic-{task_type}-{_digest([sample_seed, data, a, b, render_layout, style])[:20]}"
                dimension, _ = TASK_TYPES[task_type]
                record = {
                    "task_id": task_id, "case_id": task_id,
                    "dataset": f"synthetic_{task_type}",
                    "attack_type": task_type, "dimension": dimension,
                    "tags": [dimension, task_type, "conflicting"],
                    "shared_data_id": group_id,
                    "prefix_a": render_instruction(a, style),
                    "prefix_b": render_instruction(b, style),
                    "shared_block": render_block(data, render_layout),
                    "question": QUESTION, "gold_a": gold_a, "gold_b": gold_b,
                    "metric": "exact_match", "answer_mode": "raw",
                    "metadata": {
                        "generator_version": GENERATOR_VERSION, "seed": seed,
                        "sample_index": index, "generation_attempt": attempt,
                        "num_rows": n, "layout": render_layout,
                        "instruction_style": style, "data": data,
                        "rule_a": a, "rule_b": b,
                        "requires_raw_output": True,
                    },
                }
                validate_generated_record(record)
                seen.add(group_id)
                yield record
                break
            else:
                raise RuntimeError(f"could not generate unique conflicting pair for {task_type}[{index}]")


def score_response(record: Mapping[str, Any], side: str, response: str) -> bool:
    """Strict final-output scoring; preserve case, internal spacing and format.

    Strip surrounding transport whitespace only. Do not extract boxed answers,
    lowercase, or strip prose: those would hide instruction-following failures.
    """
    if side not in ("a", "b"):
        raise ValueError("side must be 'a' or 'b'")
    return response.strip() == record[f"gold_{side}"]