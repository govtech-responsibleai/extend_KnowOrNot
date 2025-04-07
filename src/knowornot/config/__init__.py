from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AzureConfig:
    endpoint: str
    api_key: str


@dataclass
class Config:
    azure_config: Optional[AzureConfig] = None
    azure_batch_config: Optional[AzureConfig] = None
    arbitrary_keys: dict = field(default_factory=dict)
