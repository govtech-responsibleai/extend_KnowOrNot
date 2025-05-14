from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import logging
from pydantic import BaseModel


class ToolType(str, Enum):
    SEARCH = "search"


class Tool(BaseModel):
    type: ToolType

    @classmethod
    def from_dict(cls, tool_dict: Dict[str, Any]) -> "Tool":
        """
        Create a Tool instance from a dictionary.

        Args:
            tool_dict (Dict[str, Any]): Dictionary containing tool configuration.
                Must have a 'type' key with a value that matches ToolType enum values.

        Returns:
            Tool: A new Tool instance.

        Raises:
            ValueError: If the dictionary is missing the 'type' key or has an invalid type value.
        """
        if "type" not in tool_dict:
            raise ValueError("Tool dictionary must have a 'type' key")
        if tool_dict["type"] not in [t.value for t in ToolType]:
            raise ValueError(
                f"Invalid tool type: {tool_dict['type']}. Must be one of {[t.value for t in ToolType]}"
            )
        return cls(type=ToolType(tool_dict["type"]))

    @classmethod
    def from_dict_list(cls, tool_dicts: List[Dict[str, Any]]) -> List["Tool"]:
        """
        Create a list of Tool instances from a list of dictionaries.

        Args:
            tool_dicts (List[Dict[str, Any]]): List of dictionaries containing tool configurations.
                Each dict must have a 'type' key with a value that matches ToolType enum values.

        Returns:
            List[Tool]: A list of new Tool instances.

        Raises:
            ValueError: If any dictionary is missing the 'type' key or has an invalid type value.
        """
        return [cls.from_dict(tool_dict) for tool_dict in tool_dicts]


@dataclass
class LLMClientConfig(ABC):
    logger: logging.Logger
    api_key: str
    default_model: str
    default_embedding_model: str
    can_use_instructor: bool = False
    can_use_tools: bool = False
    tools: Optional[List[Tool]] = None


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
class OpenAIConfig(LLMClientConfig):
    can_use_instructor: bool = True
    can_use_tools: bool = True
    organization: Optional[str] = None
    project: Optional[str] = None

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("api_key is required for OpenAIConfig")
        if self.organization:
            self.logger.info(
                f"Using organization: {self.organization} for OpenAI client"
            )
        else:
            self.logger.info("Not using any organization for OpenAI client")

        if self.project:
            self.logger.info(f"Using project: {self.project} for OpenAI client")
        else:
            self.logger.info("Not using any project for OpenAI client")


class OpenRouterConfig(LLMClientConfig):
    can_use_instructor: bool = False  # technically can, please override this if your specific openrouter model can!
    default_embedding_model = ""  # no embedding support on openrouter


@dataclass
class GeminiConfig(LLMClientConfig):
    can_use_instructor: bool = True
    can_use_tools: bool = True

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("api_key is required for GeminiConfig")


@dataclass
class Config:
    azure_config: Optional[AzureOpenAIConfig] = None
    azure_batch_config: Optional[AzureOpenAIConfig] = None
    gemini_config: Optional[GeminiConfig] = None
    arbitrary_keys: dict = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(self):
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("Config initialized")
