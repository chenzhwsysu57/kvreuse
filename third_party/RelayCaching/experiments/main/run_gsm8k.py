"""
RelayCaching Main Experiment: GSM8K

Evaluates multi-agent collaboration on the GSM8K math reasoning benchmark.
Corresponds to Table 1 (GSM8K rows) in the paper.

Usage:
    python run_gsm8k.py --execution_mode relaycaching --agent_nums 2
    python run_gsm8k.py --execution_mode dense_prefill_only --agent_nums 4
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
from datasets_self.gsm8k_dataset import get_pred_value, universal_data_process, check_correctness
from RelayCaching.graph.graph import Graph
from RelayCaching.tools.reader.readers import JSONLReader
from RelayCaching.utils.log import logger
from RelayCaching.utils.metrics import metrics_recorder

from experiment_utils import (
    add_common_args,
    build_run_tag,
    dataloader,
    format_agent_traces,
    get_topology_kwargs,
    load_result,
    save_result,
    setup_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RelayCaching - GSM8K Experiment (Table 1)"
    )
    add_common_args(parser)

    # GSM8K-specific defaults
    parser.set_defaults(
        domain="gsm8k",
        agent_names=["MathSolver"],
        agent_nums=[3],
        decision_method="FinalRefer",
        batch_size=10,
        llm_name="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--dataset_json", type=str,
                        default="datasets_self/gsm8k/gsm8k.jsonl",
                        help="Path to GSM8K dataset file.")

    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()

    # Common setup
    output_dir, result_file, run_tag, kv_config = setup_experiment(args, "gsm8k")

    # Load dataset
    dataset_raw = JSONLReader.parse_file(args.dataset_json)
    dataset = universal_data_process(dataset_raw)
    if args.data_limit is not None:
        dataset = dataset[:args.data_limit]

    # Build graph
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums)
                   for _ in range(num)]
    topology_kwargs = get_topology_kwargs(args.mode, len(agent_names))

    graph = Graph(
        domain="gsm8k",
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

            input_dict = {"task": record["task"], "_batch_index": i_batch}
            mode_kwargs = {
                "prefix": args.prefix,
                "output_dir": str(output_dir),
                "baseline_label": getattr(args, "baseline_label", None),
            }

            tasks.append(asyncio.create_task(
                realized_graph.arun(input_dict, 1, mode=args.execution_mode, **mode_kwargs)
            ))
            meta_info.append({
                "task": record["task"],
                "step": record["step"],
                "answer": record["answer"],
            })

        batch_results = await asyncio.gather(*tasks)
        results_by_task = {r.get("task"): r for r in batch_results}
        data = load_result(result_file)

        for info in meta_info:
            task = info["task"]
            true_answer = info["answer"]
            result_entry = results_by_task.get(task, {})
            answers = result_entry.get("answers", [])
            response = answers if isinstance(answers, list) else [answers]

            if not response or not response[0]:
                predicted = ""
                is_solved = False
            else:
                try:
                    predicted = await get_pred_value(response[0])
                    is_solved = check_correctness(predicted, true_answer)
                except Exception as e:
                    logger.warning(f"Evaluation error: {e}")
                    predicted = str(response[0])[-50:]
                    is_solved = False

            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed

            trace = result_entry.get("trace")
            agent_traces = format_agent_traces(trace)

            data.append({
                "Question": task,
                "Answer": true_answer,
                "Step": info["step"],
                "Response": response,
                "Attempt answer": predicted,
                "Solved": bool(is_solved),
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy,
                **({"Request UID": result_entry["request_uid"]}
                   if result_entry.get("request_uid") else {}),
                **({"Agent traces": agent_traces} if agent_traces else {}),
            })

        save_result(data, result_file)
        logger.opt(colors=True).info(f"<blue>[BATCH TIME]</blue> {time.time() - start_ts:.3f}s")
        logger.opt(colors=True).info(f"<blue>[ACCURACY]</blue> {accuracy:.4f}")
        metrics_recorder.log_cumulative(batch_index=i_batch)


if __name__ == "__main__":
    asyncio.run(main())
