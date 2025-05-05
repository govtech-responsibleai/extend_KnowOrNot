from typing import List, Optional, Type, TypeVar, Union
import instructor
from pydantic import BaseModel

from ..SyncLLMClient.exceptions import InitialCallFailedException
from ..SyncLLMClient import SyncLLMClient, Message, SyncLLMClientEnum
from ..config import OpenAIConfig
from openai import OpenAI
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

T = TypeVar("T", bound=BaseModel)


class SyncOpenAIClient(SyncLLMClient):
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)

        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.logger = config.logger
        self.instructor_client = instructor.from_openai(self.client)

        try:
            self.prompt("hello", ai_model=self.config.default_model)
        except Exception as e:
            raise InitialCallFailedException(
                model_name=self.config.default_model, error_message=str(e)
            )
        self.logger.info(
            f"Using model: {self.config.default_model} as the default model"
        )

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        """
        Constructs and sends a prompt to the OpenAI chat model and returns the generated response.

        This method handles the conversion of the input `prompt`, which can be a string or a list of `Message` objects,
        into a list of `ChatCompletionMessageParam` suitable for the OpenAI chat model. It then sends the
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
        Constructs and sends a prompt to the OpenAI chat model and returns the generated structured response.

        This method handles the conversion of the input `prompt`, which can be a string or a list of `Message` objects,
        into a list of `ChatCompletionMessageParam` suitable for the OpenAI chat model. It then sends the
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

    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        if not model:
            model = "text-embedding-3-large"
        embedding_response = self.client.embeddings.create(
            input=prompt_list, model=model
        )
        return list(map(lambda x: x.embedding, embedding_response.data))

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.OPENAI
