from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

from src.knowornot.config import LLMClientConfig

T = TypeVar("T", bound=BaseModel)


class SyncLLMClient(ABC):
    def __init__(self, config: LLMClientConfig):
        self.config = config

    @abstractmethod
    def prompt(self, prompt: str) -> str:
        pass

    @property
    def can_use_instructor(self) -> bool:
        return self.config.can_use_instructor

    @abstractmethod
    def _generate_structured_response(self, prompt: str, response_model: Type[T]) -> T:
        pass

    def get_structured_response(self, prompt: str, response_model: Type[T]) -> T:
        if not self.can_use_instructor:
            raise ValueError(
                "This LLM client cannot generate structured responses."
                "Enable instructor mode in the configuration to use this feature."
            )

        return self._generate_structured_response(
            prompt=prompt, response_model=response_model
        )
