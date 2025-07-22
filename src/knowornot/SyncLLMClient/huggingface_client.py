from typing import TypeVar, Union, List
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from ..config import HuggingFaceConfig
from . import SyncLLMClient, Message, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)

class SyncHuggingFaceClient(SyncLLMClient):
    def __init__(self, config: HuggingFaceConfig):
        self.config = config
        self.logger = config.logger
        self.client = InferenceClient(
            provider=config.provider,
            api_key=config.api_key,
            bill_to=config.bill_to,
        )

    def _convert_messages(self, prompt: Union[str, List[Message]]):
        """
        Converts the input prompt into a list of messages in a format suitable for the Anthropic API.

        Args:
            prompt: The input prompt to convert. It can be a string or a list of `Message` objects.

        Returns:
            A list of messages in the format required by the Anthropic API.
        """
        messages = []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            for m in prompt:
                messages.append({"role": m.role, "content": m.content})
        return messages

    def _generate_structured_response(self, prompt: Union[str, List[Message]], response_model: T, model_used: str) -> T:
        # Define the response format
        response_format = {
            "type": "json_object",
            "value": response_model.model_json_schema(),
            "strict": True,
        }
 
        messages = self._convert_messages(prompt)

        # Generate structured output using the specified model
        response = self.client.chat_completion(
            messages=messages,
            response_format=response_format,
            model=model_used,
        )

        # The response is guaranteed to match your schema
        structured_data = response.choices[0].message.content

        return response_model.parse_raw(structured_data)

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:

        messages = self._convert_messages(prompt)

        # Generate text using the specified model
        response = self.client.chat_completion(
            messages=messages,
            model=ai_model,
        )

        # The response is a string
        return response.choices[0].message.content


    def get_embedding(self, prompt: str, model: str) -> list:
        # Not implemented for Hugging Face
        raise NotImplementedError
    
    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.HUGGINGFACE