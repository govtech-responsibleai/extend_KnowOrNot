from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Type, TypeVar, Union

from pydantic import BaseModel

from ..config import LLMClientConfig

T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    role: str
    content: str


class SyncLLMClientEnum(Enum):
    AZURE_OPENAI = "AZURE_OPENAI"
    GOOGLE_GEMINI = "GOOGLE_GEMINI"


class SyncLLMClient(ABC):
    def __init__(self, config: LLMClientConfig):
        self.config = config

    @abstractmethod
    def prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        """
        Send a prompt to the LLM and get a text response.
        Accepts either a string or a list of messages.
        """
        pass

    @property
    def can_use_instructor(self) -> bool:
        return self.config.can_use_instructor

    @abstractmethod
    def _generate_structured_response(
        self,
        prompt: Union[str, List[Message]],
        response_model: Type[T],
        model_used: str,
    ) -> T:
        pass

    def get_structured_response(
        self, prompt: Union[str, List[Message]], response_model: Type[T], ai_model: str
    ) -> T:
        """
        Generate a structured response from a given prompt using the LLM.

        Parameters:
        - prompt (Union[str, List[Message]]): The input prompt to send to the LLM. It can be
        either a single string or a list of Message objects.
        - response_model (Type[T]): The Pydantic model type that the response should be
        structured into.

        Returns:
        - T: An instance of the response model containing the structured response.

        Raises:
        - ValueError: If the instructor mode is not enabled in the configuration, which is
        required to generate structured responses.
        """

        if not self.can_use_instructor:
            raise ValueError(
                "This LLM client cannot generate structured responses. "
                "Enable instructor mode in the configuration to use this feature."
            )

        return self._generate_structured_response(
            prompt=prompt, response_model=response_model, model_used=ai_model
        )

    @property
    @abstractmethod
    def enum_name(self) -> SyncLLMClientEnum:
        raise NotImplementedError()
