import glob
import json
from pathlib import Path
import pandas as pd
from typing import Union, List, Literal, Any, Dict, Optional
import numpy as np
from abc import ABC
from RelayCaching.utils.log import logger
import re


def _resolve_mmlu_csv_dir(data_dir: Optional[str], split: str) -> str:
    """Resolve directory containing MMLU split CSV files."""
    if data_dir is None:
        return f"datasets_self/MMLU/data/{split}/"
    base = Path(data_dir)
    if base.name in ("dev", "val", "test"):
        return str(base).rstrip("/") + "/"
    split_dir = base / split
    if split_dir.is_dir():
        return str(split_dir).rstrip("/") + "/"
    return str(base).rstrip("/") + "/"


def load_mmlu_observation_records(data_path: str) -> List[Dict[str, Any]]:
    """Load MMLU samples as records with a ``task`` field for observation/profiler.

    Supports:
    - CSV layout: ``datasets_self/MMLU/data``, ``.../data/test``, etc.
    - JSON/JSONL files with ``task`` or question/option fields.
    """
    path = Path(data_path)
    if path.is_file():
        if path.suffix == ".jsonl":
            from RelayCaching.tools.reader.readers import JSONLReader
            raw = JSONLReader.parse_file(str(path))
        else:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Expected a list in MMLU file {data_path}")
        records: List[Dict[str, Any]] = []
        for item in raw:
            if "task" in item:
                records.append({"task": item["task"]})
            elif "question" in item:
                records.append(
                    MMLUDataset.record_to_input(pd.Series(item))
                )
            else:
                raise ValueError(
                    f"MMLU record must contain 'task' or 'question': {item}"
                )
        return records

    split = path.name if path.name in ("dev", "val", "test") else "test"
    if path.name in ("dev", "val", "test"):
        data_dir = str(path.parent)
    else:
        data_dir = str(path)
    dataset = MMLUDataset(split=split, data_dir=data_dir)
    return [MMLUDataset.record_to_input(dataset[i]) for i in range(len(dataset))]


class MMLUDataset(ABC):

    def __init__(
        self,
        split: Union[Literal["dev"], Literal["val"], Literal["test"]] = "test",
        data_dir: Optional[str] = None,
    ) -> None:
        self._split = split
        data_path = _resolve_mmlu_csv_dir(data_dir, split)
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        logger.info("Number of topics: {}", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names,encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)
        total_df = total_df.reindex(rng.permutation(total_df.index))
        logger.info("Total number of questions: {}", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index):
        if isinstance(index, slice):
            df_slice = self._total_df.iloc[index]
            return [df_slice.iloc[i] for i in range(len(df_slice))]
        return self._total_df.iloc[index]

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict



    async def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """
        Post-process model output to extract a single Option Letter (A, B, C, D).
        """
        if isinstance(answer, list):
            answer_text = answer[0] if answer else ""
        elif isinstance(answer, str):
            answer_text = answer
        else:
            raise TypeError(f"Expected a string or list of strings, got {type(answer)}")

        answer_text = answer_text.strip()
        if not answer_text:
            return ""

        # Strategy 1: Match "The answer is [Letter]"
        match = re.search(r"(?i)answer\s*is\s*[:\s]*(?:option)?\s*([A-D])", answer_text)
        if match:
            return match.group(1).upper()

        # Strategy 2: Match "#### [Letter]"
        match = re.search(r"####\s*([A-D])", answer_text)
        if match:
            return match.group(1).upper()

        return ""

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)"            f" record={record}")
        return correct_answer
