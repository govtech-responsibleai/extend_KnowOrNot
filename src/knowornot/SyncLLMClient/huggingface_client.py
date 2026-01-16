from typing import TypeVar, Union, List, Dict, Optional, Type
import instructor
from pydantic import BaseModel, field_validator
from huggingface_hub import InferenceClient

from openai import OpenAI

from ..config import HuggingFaceConfig
from ..common.models import QAResponse
from .exceptions import InitialCallFailedException
from . import SyncLLMClient, Message, SyncLLMClientEnum

T = TypeVar("T", bound=BaseModel)


class HuggingFaceQAResponse(QAResponse):
    @field_validator("citation", mode="before")
    @classmethod
    def _normalize_citation(cls, value):
        if isinstance(value, str):
            normalized_value = value.strip()
            if normalized_value.lower() == "no citation":
                return "no citation"
            try:
                return int(normalized_value)
            except ValueError:
                return value
        return value

class SyncHuggingFaceClient(SyncLLMClient):
    def __init__(self, config: HuggingFaceConfig):
        super().__init__(config)
        self.config = config
        self.logger = config.logger
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=config.api_key,
            default_headers={
                "X-HF-Bill-To": config.bill_to
            }
        )
        self.instructor_client = instructor.from_openai(
            self.client, mode=instructor.Mode.JSON
        )

        try:
            self.prompt("hello", ai_model=self.config.default_model)
        except Exception as e:
            raise InitialCallFailedException(
                model_name=self.config.default_model, error_message=str(e)
            )
        self.logger.info(
            f"Using model: {self.config.default_model} as the default model"
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
                strict_messages.append({"role": message["role"], "content": message["content"] + "\nFor citation: return ONLY an integer (e.g., 1, 2, 3, etc.) or the exact string 'no citation'. Do not return anything else."})
            else:
                strict_messages.append(message)
        return strict_messages

    def _prompt(self, prompt: Union[str, List[Message]], ai_model: str) -> str:
        messages = self._convert_messages(prompt)

        response = self.client.chat.completions.create(
            messages=messages,
            model=ai_model,
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
        messages = self._convert_messages(prompt)
        if response_model.__name__ == "QAResponse":
            messages = self._add_strict_prompt(messages)

        messages.append({"role": "user", "content": f"YOU MUST RETURN A JSON OBJECT WITH THE FOLLOWING SCHEMA: {response_model.model_json_schema()}"})

        hf_response_model: Type[BaseModel] = response_model
        convert_to_requested_model = False
        if response_model.__name__ == "QAResponse":
            hf_response_model = HuggingFaceQAResponse
            convert_to_requested_model = True

        response = self.instructor_client.chat.completions.create(
            model=model_used,
            response_model=hf_response_model,
            messages=messages,
        )

        # content = response.choices[0].message.content
        # if "<think>" in content:
        #     content = content.split("<think>")[1]
        #     return response_model.model_validate_json(content)

        if convert_to_requested_model:
            return response_model.model_validate(response.model_dump())

        return response

    def get_embedding(
        self, prompt_list: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        raise NotImplementedError("HuggingFace does not support embeddings")

    @property
    def enum_name(self) -> SyncLLMClientEnum:
        return SyncLLMClientEnum.HUGGINGFACE
