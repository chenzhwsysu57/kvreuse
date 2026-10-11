from typing import Dict, Any
from abc import ABC, abstractmethod


class PromptSet(ABC):
    """
    Abstract base class for a set of prompts used by different agents.
    Subclasses should implement all abstract methods to provide
    task-specific prompt templates.
    """
    @staticmethod
    @abstractmethod
    def get_role() -> str:
        """Return the role description for the agent."""

    @staticmethod
    @abstractmethod
    def get_constraint() -> str:
        """Return the constraint instructions for the agent."""

    @staticmethod
    @abstractmethod
    def get_format() -> str:
        """Return the expected output format specification."""

    @staticmethod
    @abstractmethod
    def get_answer_prompt(question) -> str:
        """Return the prompt for answering a given question."""

    @staticmethod
    @abstractmethod
    def get_query_prompt(question) -> str:
        """Return the prompt for generating a query from a question."""

    @staticmethod
    @abstractmethod
    def get_file_analysis_prompt(query, file) -> str:
        """Return the prompt for analyzing a file given a query."""

    @staticmethod
    @abstractmethod
    def get_websearch_prompt(query) -> str:
        """Return the prompt for performing a web search."""

    @staticmethod
    @abstractmethod
    def get_distill_websearch_prompt(query, results) -> str:
        """Return the prompt for distilling web search results."""

    @staticmethod
    @abstractmethod
    def get_reflect_prompt(question, answer) -> str:
        """Return the prompt for reflecting on a given answer."""

    @staticmethod
    def get_react_prompt(question, solutions, feedback) -> str:
        """Return the prompt for reacting based on solutions and feedback."""

    @staticmethod
    @abstractmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        """Return the prompt for combining multiple materials."""

    @staticmethod
    @abstractmethod
    def get_decision_constraint() -> str:
        """Return the constraint for the decision-making agent."""

    @staticmethod
    @abstractmethod
    def get_decision_role() -> str:
        """Return the role description for the decision-making agent."""

    @staticmethod
    @abstractmethod
    def get_decision_few_shot() -> str:
        """Return few-shot examples for the decision-making agent."""