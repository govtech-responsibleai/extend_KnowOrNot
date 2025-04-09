from typing import List, Type, TypeVar, Union

import instructor
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel
import warnings

from ..config import AzureOpenAIConfig
from ..SyncLLMClient import Message, SyncLLMClient, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)


class SyncAzureOpenAIClient(SyncLLMClient):
    def __init__(self, config: AzureOpenAIConfig):
        super().__init__(config)
        self.config: AzureOpenAIConfig = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Client should be an instance of openai.OpenAI or openai.AsyncOpenAI.*",
            )
            self.instructor_client = instructor.from_openai(self.client)

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        """
        Constructs and sends a prompt to the Azure OpenAI chat model and returns the generated response.

        This method handles the conversion of the input `prompt`, which can be a string or a list of `Message` objects,
        into a list of `ChatCompletionMessageParam` suitable for the Azure OpenAI chat model. It then sends the
        constructed messages to the model specified by `ai_model` and returns the model's textual response.

        Args:
            prompt: The input prompt(s) to send to the chat model. It can be a single string or a list of
                    `Message` objects, where each message has a role (user, system, or assistant) and content.
            ai_model: The identifier of the AI model to use for generating the chat response.

        Returns:
            The text content of the first message choice returned by the AI model.

        Raises:
            ValueError: If the response from the model does not contain any content.
        """

        messages: List[ChatCompletionMessageParam] = []

        if isinstance(prompt, str):
            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": prompt,
            }
            messages.append(user_message)
        else:
            for m in prompt:
                if m.role == "user":
                    user_message: ChatCompletionUserMessageParam = {
                        "role": "user",
                        "content": m.content,
                    }
                    messages.append(user_message)
                elif m.role == "system":
                    system_message: ChatCompletionSystemMessageParam = {
                        "role": "system",
                        "content": m.content,
                    }
                    messages.append(system_message)
                elif m.role == "assistant":
                    assistant_message: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
                        "content": m.content,
                    }
                    messages.append(assistant_message)
                # Add others as needed

        response = self.client.chat.completions.create(
            model=ai_model,
            messages=messages,
        )
        output = response.choices[0].message.content
        if not output:
            raise ValueError(
                f"Expected output that was not none for {prompt} but got {output}"
            )

        return output

    def _generate_structured_response(
        self,
        prompt: Union[str, List[Message]],
        response_model: Type[T],
        model_used: str,
    ) -> T:
        """
        Constructs and sends a prompt to the Azure OpenAI chat model and returns the generated structured response.

        This method handles the conversion of the input `prompt`, which can be a string or a list of `Message` objects,
        into a list of `ChatCompletionMessageParam` suitable for the Azure OpenAI chat model. It then sends the
        constructed messages to the model specified by `model_used` and returns the model's response parsed into an
        instance of `response_model`.

        Args:
            prompt: The input prompt(s) to send to the chat model. It can be a single string or a list of
                    `Message` objects, where each message has a role (user, system, or assistant) and content.
            response_model: The type of the response model to parse the output into.
            model_used: The identifier of the AI model to use for generating the chat response.

        Returns:
            An instance of `response_model` containing the structured response from the AI model.

        Raises:
            ValueError: If the response from the model does not contain any content.
        """
        messages: List[ChatCompletionMessageParam] = []

        if isinstance(prompt, str):
            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": prompt,
            }
            messages.append(user_message)
        else:
            for m in prompt:
                if m.role == "user":
                    user_message: ChatCompletionUserMessageParam = {
                        "role": "user",
                        "content": m.content,
                    }
                    messages.append(user_message)
                elif m.role == "system":
                    system_message: ChatCompletionSystemMessageParam = {
                        "role": "system",
                        "content": m.content,
                    }
                    messages.append(system_message)
                elif m.role == "assistant":
                    assistant_message: ChatCompletionAssistantMessageParam = {
                        "role": "assistant",
                        "content": m.content,
                    }
                    messages.append(assistant_message)
                # Add others as needed

        response = self.instructor_client.chat.completions.create(
            model=model_used,
            response_model=response_model,
            messages=messages,
        )
        return response

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.AZURE_OPENAI
