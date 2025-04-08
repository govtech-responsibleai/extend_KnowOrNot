import unittest
from typing import Any, Dict, Type
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from src.knowornot.config import LLMClientConfig
from src.knowornot.SyncLLMClient import SyncLLMClient, T


# Define a dummy config class for testing
class DummyLLMClientConfig(LLMClientConfig):
    can_use_instructor: bool = False
    api_key: str = "dummy_api_key"  # Required for LLMClientConfig


class WeatherResponse(BaseModel):
    temperature: float
    condition: str


class TestSyncLLMClient(unittest.TestCase):
    """Tests for the SyncLLMClient abstract base class."""

    def test_get_structured_response_raises_value_error_when_instructor_disabled(self):
        # Define a concrete class for testing (using Mock)
        class MockSyncLLMClient(SyncLLMClient):
            def prompt(self, prompt: str) -> str:
                return "Mock prompt response"

            def _generate_structured_response(
                self, prompt: str, response_model: Type[T]
            ) -> T:
                # Mock implementation - shouldn't be called in this test
                raise NotImplementedError(
                    "_generate_structured_response should not be called"
                )

        config = DummyLLMClientConfig(
            can_use_instructor=False, api_key="dummy_api_key"
        )  # Instructor disabled
        client = MockSyncLLMClient(config=config)

        with self.assertRaisesRegex(
            ValueError, "This LLM client cannot generate structured responses"
        ):
            client.get_structured_response("dummy prompt", WeatherResponse)

    def test_get_structured_response_calls_generate_structured_response_when_instructor_enabled(
        self,
    ):
        # Define a concrete class for testing (using Mock)
        class MockSyncLLMClient(SyncLLMClient):
            def prompt(self, prompt: str) -> str:
                return "Mock prompt response"

            def _generate_structured_response(
                self, prompt: str, response_model: Type[T]
            ) -> T:
                data: Dict[str, Any] = {"temperature": 20.0, "condition": "Cloudy"}
                instance = response_model(**data)
                return instance

        config = DummyLLMClientConfig(
            can_use_instructor=True, api_key="dummy_api_key"
        )  # Instructor enabled
        client = MockSyncLLMClient(config=config)

        # Call the method
        response = client.get_structured_response("dummy prompt", WeatherResponse)

        # Assert that the response is an instance of WeatherResponse
        self.assertIsInstance(response, WeatherResponse)
        self.assertEqual(response.temperature, 20.0)
        self.assertEqual(response.condition, "Cloudy")

    @patch(
        "src.knowornot.LLMClient.SyncLLMClient._generate_structured_response"
    )  # Patching the method
    def test_get_structured_response_handles_exceptions_from_generate_structured_response(
        self, mock_generate
    ):
        # Define a concrete class for testing (using Mock)
        class MockSyncLLMClient(SyncLLMClient):
            def prompt(self, prompt: str) -> str:
                return "Mock prompt response"

            def _generate_structured_response(
                self, prompt: str, response_model: Type[T]
            ) -> T:
                # Mock implementation - raise an exception
                raise ValueError("Simulated error in _generate_structured_response")

        config = DummyLLMClientConfig(
            can_use_instructor=True, api_key="dummy_api_key"
        )  # Instructor enabled
        client = MockSyncLLMClient(config=config)

        with self.assertRaisesRegex(
            ValueError, "Simulated error in _generate_structured_response"
        ):  # Expect ValueError
            client.get_structured_response("dummy prompt", WeatherResponse)

    def test_prompt_abstract(self):
        # Test that inheriting classes MUST implement prompt method
        class BadClient(SyncLLMClient):
            # Does not implement the prompt method!
            def _generate_structured_response(
                self, prompt: str, response_model: Type[T]
            ) -> T:
                return MagicMock(
                    spec=BaseModel
                )  # Dummy for demonstration.  Important for testing!

        config = DummyLLMClientConfig(can_use_instructor=True, api_key="dummy_api_key")
        with self.assertRaises(TypeError):
            BadClient(config=config)  # type: ignore

    def test_abstract_generate_structured_response(self):
        # Test that inheriting classes MUST implement _generate_structured_response method

        class BadClient(SyncLLMClient):
            def prompt(self, prompt: str) -> str:
                return "Test"  # Dummy implementation. Important for testing!

            # Does not implement the _generate_structured_response!

        config = DummyLLMClientConfig(can_use_instructor=True, api_key="dummy_api_key")
        with self.assertRaises(TypeError) as context:
            BadClient(config=config)  # type: ignore
            # ignore type by type checker as we are deliberately instantiating with the wrong type

        self.assertIn(
            "Can't instantiate abstract class BadClient without an implementation for abstract method '_generate_structured_response'",
            str(context.exception),
        )
