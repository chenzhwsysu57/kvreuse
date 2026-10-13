"""
RelayCaching Main Experiment: MMLU

Evaluates multi-agent collaboration on the MMLU knowledge benchmark.
Corresponds to Table 1 (MMLU rows) in the paper.

Usage:
    python run_mmlu.py --execution_mode relaycaching --agent_nums 2
    python run_mmlu.py --execution_mode dense_prefill_only --agent_nums 4
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
from RelayCaching.utils.log import logger
from RelayCaching.utils.metrics import metrics_recorder
from datasets_self.mmlu_dataset import MMLUDataset

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
        description="RelayCaching - MMLU Experiment (Table 1)"
    )
    add_common_args(parser)

    # MMLU-specific defaults
    parser.set_defaults(
        domain="mmlu",
        agent_names=["AnalyzeAgent"],
        agent_nums=[3],
        decision_method="FinalRefer",
        batch_size=10,
        llm_name="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--mmlu_data_dir", type=str,
                        default="datasets_self/MMLU/data",
                        help="Path to MMLU data directory.")
    parser.add_argument("--mmlu_split", type=str, default="test",
                        choices=["test", "val", "dev"],
                        help="MMLU split to evaluate on.")
    parser.add_argument("--limit_questions", type=int, default=153,
                        help="Maximum number of MMLU questions to evaluate.")

    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()

    # Common setup
    output_dir, result_file, run_tag, kv_config = setup_experiment(args, "mmlu")

    # Load dataset
    dataset = MMLUDataset(data_dir=args.mmlu_data_dir, split=args.mmlu_split)

    # Determine number of questions
    limit = args.data_limit if args.data_limit is not None else args.limit_questions
    data_len = min(len(dataset), limit) if limit else len(dataset)
    all_records = [record for i, record in enumerate(dataset) if i < data_len]

    # Build graph
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums)
                   for _ in range(num)]
    topology_kwargs = get_topology_kwargs(args.mode, len(agent_names))

    graph = Graph(
        domain="mmlu",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        kv_config=kv_config,
        **topology_kwargs,
    )

    # Run experiment
    num_batches = (len(all_records) + args.batch_size - 1) // args.batch_size
    total_solved, total_executed = 0, 0

    for i_batch in range(num_batches):
        logger.opt(colors=True).info(f"<blue>[BATCH]</blue> {i_batch}/{num_batches} {'-' * 40}")
        start_ts = time.time()
        current_batch = dataloader(all_records, args.batch_size, i_batch)
        if not current_batch:
            break

        tasks = []
        batch_meta = []
        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits

            input_dict = dataset.record_to_input(record)
            input_dict["_batch_index"] = i_batch
            correct_answer = dataset.record_to_target_answer(record)

            mode_kwargs = {
                "prefix": args.prefix,
                "output_dir": str(output_dir),
                "baseline_label": getattr(args, "baseline_label", None),
            }
            tasks.append(asyncio.create_task(
                realized_graph.arun(input_dict, 1, mode=args.execution_mode, **mode_kwargs)
            ))
            batch_meta.append({
                "record": record,
                "correct_answer": correct_answer,
                "input_dict": input_dict,
            })

        batch_results = await asyncio.gather(*tasks)
        data = load_result(result_file)

        for i, result in enumerate(batch_results):
            meta = batch_meta[i]
            correct_answer = meta["correct_answer"]

            raw_responses = result.get("answers", [])
            raw_text = raw_responses[0] if raw_responses and isinstance(raw_responses, list) else str(raw_responses)

            # MMLU answer extraction: extract A/B/C/D from response
            predicted = await dataset.postprocess_answer(raw_text)
            is_solved = (str(predicted).strip().upper() == str(correct_answer).strip().upper())

            total_solved += int(is_solved)
            total_executed += 1
            accuracy = total_solved / total_executed

            agent_traces = format_agent_traces(result.get("trace"))

            data.append({
                "Question": meta["input_dict"].get("task", str(meta["record"])),
                "Answer": correct_answer,
                "Response": raw_responses,
                "Attempt answer": predicted,
                "Solved": is_solved,
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

    logger.success(f"Evaluation finished! Results saved to {result_file}")


if __name__ == "__main__":
    asyncio.run(main())
