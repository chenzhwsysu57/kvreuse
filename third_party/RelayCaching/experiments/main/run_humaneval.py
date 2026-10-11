"""
RelayCaching Main Experiment: HumanEval

Evaluates multi-agent collaboration on the HumanEval code generation benchmark.
Corresponds to Table 1 (HumanEval rows) in the paper.

Usage:
    python run_humaneval.py --execution_mode relaycaching --agent_nums 2
    python run_humaneval.py --execution_mode dense_prefill_only --agent_nums 4
"""

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup CUDA before importing torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiment_utils import setup_cuda_early
setup_cuda_early()

import torch
from RelayCaching.graph.graph import Graph
from RelayCaching.tools.coding.python_executor import PyExecutor
from RelayCaching.tools.reader.readers import JSONLReader
from RelayCaching.utils.log import logger
from RelayCaching.utils.metrics import metrics_recorder

from experiment_utils import (
    add_common_args,
    dataloader,
    format_agent_traces,
    get_topology_kwargs,
    load_result,
    save_result,
    setup_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RelayCaching - HumanEval Experiment (Table 1)"
    )
    add_common_args(parser)

    # HumanEval-specific defaults
    parser.set_defaults(
        domain="humaneval",
        agent_names=["CodeWriting"],
        agent_nums=[3],
        decision_method="FinalRefer",
        batch_size=1,
        llm_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument("--dataset_json", type=str,
                        default="datasets_self/humaneval/humaneval-py.jsonl",
                        help="Path to HumanEval dataset file.")

    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


def extract_code(response: str) -> str:
    """Extract Python code from LLM response (handles ```python blocks)."""
    if "```python\n" in response:
        code = response.split("```python\n")[-1].split("\n```")[0]
    elif "```\n" in response:
        code = response.split("```\n")[-1].split("\n```")[0]
    else:
        code = response
    return code


async def main():
    args = parse_args()

    # Common setup
    output_dir, result_file, run_tag, kv_config = setup_experiment(args, "humaneval")

    # Load dataset
    dataset = JSONLReader.parse_file(args.dataset_json)
    if args.data_limit is not None:
        dataset = dataset[:args.data_limit]

    # Build graph
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums)
                   for _ in range(num)]
    topology_kwargs = get_topology_kwargs(args.mode, len(agent_names))

    graph = Graph(
        domain="humaneval",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        kv_config=kv_config,
        **topology_kwargs,
    )

    # Run experiment
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    total_solved, total_executed = 0, 0

    for i_batch in range(num_batches):
        logger.opt(colors=True).info(f"<blue>[BATCH]</blue> {i_batch} {'-' * 40}")
        start_ts = time.time()
        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if not current_batch:
            break

        tasks = []
        meta_info = []
        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits

            task = record["prompt"]
            input_dict = {"task": task, "_batch_index": i_batch}
            mode_kwargs = {
                "prefix": args.prefix,
                "output_dir": str(output_dir),
                "baseline_label": getattr(args, "baseline_label", None),
            }

            tasks.append(asyncio.create_task(
                realized_graph.arun(input_dict, 1, mode=args.execution_mode, **mode_kwargs)
            ))
            meta_info.append({"task": task, "tests": record["test"]})

        batch_results = await asyncio.gather(*tasks)
        data = load_result(result_file)

        for i, result in enumerate(batch_results):
            info = meta_info[i]
            task = info["task"]
            tests = info["tests"]

            answers = result.get("answers", [])
            response = answers if isinstance(answers, list) else [answers]

            if not response or not response[0]:
                code = ""
                is_solved = False
            else:
                code = extract_code(str(response[0]))
                is_solved, _, _ = PyExecutor().execute(code, [tests], timeout=100)

            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed

            agent_traces = format_agent_traces(result.get("trace"))

            data.append({
                "Question": task,
                "Tests": tests,
                "Attempt answer": code,
                "Solved": bool(is_solved),
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy,
                **({"Request UID": result["request_uid"]}
                   if result.get("request_uid") else {}),
                **({"Agent traces": agent_traces} if agent_traces else {}),
            })

        save_result(data, result_file)
        logger.opt(colors=True).info(f"<blue>[BATCH TIME]</blue> {time.time() - start_ts:.3f}s")
        logger.opt(colors=True).info(f"<blue>[ACCURACY]</blue> {accuracy:.4f}")
        metrics_recorder.log_cumulative(batch_index=i_batch)


if __name__ == "__main__":
    asyncio.run(main())