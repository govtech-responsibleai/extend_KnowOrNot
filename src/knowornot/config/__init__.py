from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMClient:
    api_key: str


@dataclass
class AzureOpenAIConfig(LLMClient):
    endpoint: str


@dataclass
class Config:
    azure_config: Optional[AzureOpenAIConfig] = None
    azure_batch_config: Optional[AzureOpenAIConfig] = None
    arbitrary_keys: dict = field(default_factory=dict)
