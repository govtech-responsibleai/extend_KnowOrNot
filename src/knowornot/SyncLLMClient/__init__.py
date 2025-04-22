from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Type, TypeVar, Union, overload, Literal
import re

from pydantic import BaseModel

from ..config import LLMClientConfig

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=Enum)


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

        self.logger.debug(f"Using model: {model_to_use} and sending prompt {prompt}")

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

        self.logger.debug(f"Using model: {model_to_use} and sending prompt {prompt}")

        return self._generate_structured_response(
            prompt=prompt, response_model=response_model, model_used=model_to_use
        )

    @abstractmethod
    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        pass

    @overload
    def prompt_and_extract_tag(
        self,
        prompt: Union[str, List[Message]],
        tag_name: str,
        allowed_list: List[str],
        on_multiple: Literal["error", "first", "last"],
        ai_model: Optional[str] = None,
    ) -> str: ...

    @overload
    def prompt_and_extract_tag(
        self,
        prompt: Union[str, List[Message]],
        tag_name: str,
        allowed_list: List[str],
        on_multiple: Literal["list"],
        ai_model: Optional[str] = None,
    ) -> List[str]: ...

    def prompt_and_extract_tag(
        self,
        prompt: Union[str, List[Message]],
        tag_name: str,
        allowed_list: List[str],
        on_multiple: Literal["error", "first", "last", "list"],
        ai_model: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """Prompts the LLM and extracts values from XML tags in the response, validating against an allowed list.

        Args:
            prompt: The input prompt to send to the LLM. Can be a string or list of Message objects.
            tag_name: The name of the XML tag to extract from.
            allowed_list: The list of allowed values to validate the extracted value(s) against.
            ai_model: The name of the AI model to use. If None, uses the default model.
            on_multiple: How to handle multiple tag matches. Options:
                - "error": Raise ValueError if multiple tags are found (default)
                - "first": Return only the first match
                - "last": Return only the last match
                - "list": Return all matches as a list

        Returns:
            If on_multiple is "first", "last", or "error" (and only one match): The extracted value
            If on_multiple is "list": A list of extracted values

        Raises:
            ValueError: If no tags are found, if the extracted value isn't in the allowed list,
                       or if multiple tags are found when on_multiple="error"
        """
        # Get response from LLM
        response = self.prompt(prompt=prompt, ai_model=ai_model)

        # Create regex pattern for the specified tag
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            raise ValueError(f"No <{tag_name}> tags found in LLM response")

        # Handle multiple matches according to on_multiple parameter
        if len(matches) > 1:
            if on_multiple == "error":
                raise ValueError(f"Multiple <{tag_name}> tags found in LLM response")
            elif on_multiple == "first":
                matches = [matches[0]]
            elif on_multiple == "last":
                matches = [matches[-1]]
            elif on_multiple != "list":
                raise ValueError(f"Invalid on_multiple value: {on_multiple}")

        # Validate against allowed list and convert
        result = []
        for value in matches:
            value = value.strip()
            if value not in allowed_list:
                raise ValueError(
                    f"Value '{value}' from tag <{tag_name}> is not in the allowed list: {allowed_list}"
                )
            result.append(value)

        # Return result based on on_multiple
        return result if on_multiple == "list" else result[0]

    @property
    @abstractmethod
    def enum_name(self) -> SyncLLMClientEnum:
        raise NotImplementedError()
