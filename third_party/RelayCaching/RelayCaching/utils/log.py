"""Logging utilities for RelayCaching experiments."""

import os
import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from RelayCaching.utils.const import RELAYCACHING_ROOT


def configure_logging(
    print_level: str = "INFO",
    logfile_level: str = "DEBUG",
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Configure the logging settings for the application.

    Args:
        print_level: The logging level for console output.
        logfile_level: The logging level for file output.
        log_path: Optional path for the log file. Defaults to ``<project_root>/logs/log.txt``.
    """
    logger.remove()
    logger.add(sys.stderr, level=print_level)
    target_path = Path(log_path) if log_path is not None else RELAYCACHING_ROOT / 'logs/log.txt'
    if target_path.is_dir():
        target_path = target_path / "log.txt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(target_path, level=logfile_level, rotation="10 MB")


def initialize_log_file(experiment_name: str, time_stamp: str) -> Path:
    """
    Initialize the log file with a start message and return its path.

    Args:
        experiment_name: Name of the experiment, used in the directory structure.
        time_stamp: The current timestamp, used in the filename.

    Returns:
        The path to the initialized log file.
    """
    try:
        log_file_path = RELAYCACHING_ROOT / f'result/{experiment_name}/logs/log_{time_stamp}.txt'
        os.makedirs(log_file_path.parent, exist_ok=True)
        with open(log_file_path, 'w') as file:
            file.write("============ Start ============\n")
    except OSError as error:
        logger.error(f"Error initializing log file: {error}")
        raise
    return log_file_path


def swarmlog(
    sender: str,
    text: str,
    cost: float,
    prompt_tokens: int,
    complete_tokens: int,
    log_file_path: Union[str, Path],
) -> None:
    """
    Log a message from an agent with cost and token statistics.

    Args:
        sender: The name of the sending agent.
        text: The text message to log.
        cost: The cumulative cost associated with the operation.
        prompt_tokens: Number of prompt tokens used.
        complete_tokens: Number of completion tokens generated.
        log_file_path: Path to the log file for persistent storage.
    """
    formatted_message = (
        f"{sender} | 💵Total Cost: ${cost:.5f} | "
        f"Prompt Tokens: {prompt_tokens} | "
        f"Completion Tokens: {complete_tokens} | \n {text}"
    )
    logger.info(formatted_message)

    try:
        os.makedirs(Path(log_file_path).parent, exist_ok=True)
        with open(log_file_path, 'a') as file:
            file.write(f"{formatted_message}\n")
    except OSError as error:
        logger.error(f"Error writing to log file: {error}")
        raise
