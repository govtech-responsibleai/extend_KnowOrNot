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

from ..config import AzureOpenAIConfig
from ..SyncLLMClient import Message, SyncLLMClient, SyncLLMClientEnum

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
        self.instructor_client = instructor.from_openai(self.client)

    def prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        # Handle different prompt types
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
        # Handle different prompt types
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
