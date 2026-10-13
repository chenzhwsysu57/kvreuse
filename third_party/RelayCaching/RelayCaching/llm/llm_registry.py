from typing import Optional
from class_registry import ClassRegistry

from RelayCaching.llm.llm import LLM
from RelayCaching.llm.config import RelayCachingConfig


class LLMRegistry:
    """Factory and registry for constructing concrete `LLM` implementations."""
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(
        cls,
        model_name: Optional[str] = None,
        prefix: Optional[str] = None,
        llm_config: Optional[RelayCachingConfig] = None,
        **kwargs,
    ) -> LLM:
        """Retrieve a concrete `LLM` instance by name and provider hints.

        Args:
            model_name: Provider/model identifier; defaults to a general chat model.
            prefix: Optional alias used by some backends.
            llm_config: Shared RelayCaching configuration passed through to models.
            **kwargs: Additional provider-specific constructor args.

        Returns:
            LLM: Instantiated language model.
        """
        if model_name is None or model_name=="":
            raise ValueError(
                "model_name must be specified. "
                "Supported models: llama, qwen, mistral (case-insensitive)."
            )

        if model_name == 'mock':
            model = cls.registry.get(model_name, **kwargs)
        elif 'llama' in model_name.lower():                                                    
            model = cls.registry.get('LLMChat', model_name, prefix=prefix, config=llm_config, **kwargs)
        elif 'qwen' in model_name.lower():                                             
            model = cls.registry.get('LLMChat', model_name, prefix=prefix, config=llm_config, **kwargs)
        elif 'mistral' in model_name.lower():                                             
            model = cls.registry.get('LLMChat', model_name, prefix=prefix, config=llm_config, **kwargs)
        else:                                      
            raise ValueError(
                f"Unsupported model: '{model_name}'. "
                f"Supported models: llama, qwen, mistral (case-insensitive)."
            )

        return model

