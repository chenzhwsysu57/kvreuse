"""Global singleton utilities for tracking runtime statistics."""


class Singleton:
    """Thread-unsafe singleton base class for lightweight global state."""

    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self):
        self.value = 0.0


class Cost(Singleton):
    """Tracks cumulative API cost."""

    def __init__(self):
        self.value = 0.0


class PromptTokens(Singleton):
    """Tracks cumulative prompt token count."""

    def __init__(self):
        self.value = 0.0


class CompletionTokens(Singleton):
    """Tracks cumulative completion token count."""

    def __init__(self):
        self.value = 0.0


class Time(Singleton):
    """Stores the current timestamp string."""

    def __init__(self):
        self.value = ""


class Mode(Singleton):
    """Stores the current execution mode identifier."""

    def __init__(self):
        self.value = ""
