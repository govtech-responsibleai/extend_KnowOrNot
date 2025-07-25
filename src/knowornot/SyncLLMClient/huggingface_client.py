from typing import TypeVar, Union, List, Dict
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
        Converts the input prompt into a list of messages in a format suitable for HuggingFace InferenceClient.

        Args:
            prompt: The input prompt to convert. It can be a string or a list of `Message` objects.

        Returns:
            A list of messages in the format required by HuggingFace InferenceClient.
        """
        messages = []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            for m in prompt:
                messages.append({"role": m.role, "content": m.content})
        return messages

    def _add_strict_prompt(self, messages: List[Dict]):
        """
        Modifies the input messages by adding a strict prompt to any user messages,
        which enforces the output format of the model. This is done after experimenting with HuggingFace's 
        InferenceClient and finding that response_format is not sufficient in ensuring strict output.

        Args:
            messages (List[Dict]): A list of messages to modify.

        Returns:
            List[Dict]: A list of modified messages.
        """
        strict_messages = []
        for message in messages:
            if message["role"] == "user":
                strict_messages.append({"role": message["role"], "content": message["content"] + "\nOnly use integer or exactly `no citation` for citation — nothing else"})
            else:
                strict_messages.append(message)
        return strict_messages

    def _generate_structured_response(self, prompt: Union[str, List[Message]], response_model: T, model_used: str) -> T:
        # Define the response format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": {
                    **response_model.model_json_schema(),
                    "additionalProperties": False
                },
                "strict": True,
            }
        }
 
        messages = self._convert_messages(prompt)
        if response_model.__name__ == "QAResponse":
            messages = self._add_strict_prompt(messages)

        response = self.client.chat_completion(
            messages=messages,
            response_format=response_format,
            model=model_used,
            max_tokens=10_000,
        )
        content = response.choices[0].message.content
        if "</think>" in content:
            content = content.split("</think>")[1]
        structured_data = response_model.model_validate_json(content)

        return structured_data
        
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
