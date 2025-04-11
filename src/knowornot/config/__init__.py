from abc import ABC
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMClientConfig(ABC):
    api_key: str
    default_model: str
    default_embedding_model: str
    can_use_instructor: bool = False


@dataclass
class AzureOpenAIConfig(LLMClientConfig):
    can_use_instructor: bool = True
    endpoint: str = ""
    api_version: str = ""

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("api_key is required for AzureOpenAIConfig")
        if not self.endpoint:
            raise ValueError("endpoint is required for AzureOpenAIConfig")
        if not self.api_version:
            raise ValueError("api_version is required for AzureOpenAIConfig")


@dataclass
class Config:
    azure_config: Optional[AzureOpenAIConfig] = None
    azure_batch_config: Optional[AzureOpenAIConfig] = None
    arbitrary_keys: dict = field(default_factory=dict)
