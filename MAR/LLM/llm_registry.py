from typing import Optional
from class_registry import ClassRegistry

from MAR.LLM.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name=="":
            model_name = "gpt-4o-mini"
        if 'DeepSeek-V3' in model_name:
            model = cls.registry.get('Deepseek', model_name)
        else:
            model = cls.registry.get('ALLChat', model_name)
        
        return model

