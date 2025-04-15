from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Type, TypeVar, Union

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
        self.logger = self.config.logger

    @abstractmethod
    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        pass

    def prompt(
        self, prompt: Union[str, List[Message]], ai_model: Optional[str] = None
    ) -> str:
        """Sends a prompt to the LLM and gets a text response.

        Args:
            prompt: The input prompt to send to the LLM. It can be either a single
                string or a list of Message objects.
            ai_model: The name of the AI model to use for generating text. If not
                specified, uses the default model configured in the LLMClientConfig.

        Returns:
            The text response from the LLM.
        """
        model_to_use: str = self.config.default_model
        if ai_model is not None:
            model_to_use = ai_model

        self.logger.info(f"Using model: {model_to_use} and sending prompt {prompt}")

        return self._prompt(prompt=prompt, ai_model=model_to_use)

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
        self,
        prompt: Union[str, List[Message]],
        response_model: Type[T],
        ai_model: Optional[str] = None,
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

        model_to_use: str = ai_model or self.config.default_model

        self.logger.info(f"Using model: {model_to_use} and sending prompt {prompt}")

        return self._generate_structured_response(
            prompt=prompt, response_model=response_model, model_used=model_to_use
        )

    @abstractmethod
    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def enum_name(self) -> SyncLLMClientEnum:
        raise NotImplementedError()
