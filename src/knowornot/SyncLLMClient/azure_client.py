from typing import Type, TypeVar

from openai import AzureOpenAI
from pydantic import BaseModel

from ..config import AzureOpenAIConfig
from ..SyncLLMClient import SyncLLMClient, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)


class SyncAzureOpenAIClient(SyncLLMClient):
    def __init__(self, config: AzureOpenAIConfig):
        super().__init__(config)
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
        )

    def prompt(self, prompt: str) -> str:
        raise NotImplementedError()

    def _generate_structured_response(self, prompt: str, response_model: Type[T]) -> T:
        raise NotImplementedError()

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.AZURE_OPENAI
