# tests/test_syncllmclient_integration.py

import unittest
from dotenv import load_dotenv

from src.knowornot import KnowOrNot
from src.knowornot.SyncLLMClient import SyncLLMClient, SyncLLMClientEnum
from src.knowornot.common.models import QAResponse

load_dotenv()

class TestSyncLLMClientIntegration(unittest.TestCase):
    def test_openai_client(self):
        kon = KnowOrNot()
        kon.add_openai()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.OPENAI))

    def test_azure_openai_client(self):
        kon = KnowOrNot()
        kon.add_azure()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.AZURE_OPENAI))

    def test_anthropic_client(self):
        kon = KnowOrNot()
        kon.add_anthropic()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.ANTHROPIC))

    def test_bedrock_client(self):
        kon = KnowOrNot()
        kon.add_bedrock()
        self._test_client_structured(kon.get_client(SyncLLMClientEnum.BEDROCK))

    def test_gemini_client(self):
        kon = KnowOrNot()
        kon.add_gemini()
        self._test_client_string(kon.get_client(SyncLLMClientEnum.GEMINI))

    def test_openrouter_client(self):
        kon = KnowOrNot()
        kon.add_openrouter()
        self._test_client_string(kon.get_client(SyncLLMClientEnum.OPENROUTER))

    def _test_client_structured(self, client):
        # Test sending a prompt and receiving a structured response
        prompt = "What is the capital of France?"
        response_model = QAResponse

        response = client.get_structured_response(prompt, response_model)
        self.assertIsInstance(response, QAResponse)
        self.assertIsNotNone(response.response)
    
    def _test_client_string(self, client):
        # Test sending a prompt and receiving a text response
        prompt = "What is the capital of France?"

        response = client.prompt(prompt)
        self.assertIsInstance(response, str)