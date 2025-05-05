from typing import cast
from unittest.mock import patch, MagicMock

from src.knowornot import KnowOrNot
from src.knowornot.SyncLLMClient import SyncLLMClientEnum
from src.knowornot.SyncLLMClient.azure_client import SyncAzureOpenAIClient


class TestKnowOrNotClientRegistration:
    """Test client registration functionality in KnowOrNot."""

    @patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
    def test_add_azure_registers_client(self, mock_prompt):
        """Test that add_azure registers an Azure client."""
        # Mock the prompt call made during initialization
        mock_prompt.return_value = "Hello"

        # Create a KnowOrNot instance
        know_or_not = KnowOrNot()

        # Add Azure client
        know_or_not.add_azure(
            azure_endpoint="https://endpoint.com",
            azure_api_key="api_key",
            azure_api_version="2023-05-15",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
        )

        # Check that the client registry has the Azure client
        assert SyncLLMClientEnum.AZURE_OPENAI in know_or_not.client_registry

        # Check that the client is an instance of SyncAzureOpenAIClient
        client = know_or_not.client_registry[SyncLLMClientEnum.AZURE_OPENAI]
        assert isinstance(client, SyncAzureOpenAIClient)

    @patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
    def test_azure_client_set_as_default(self, mock_prompt):
        """Test that the Azure client is set as the default client."""
        # Mock the prompt call made during initialization
        mock_prompt.return_value = "Hello"

        know_or_not = KnowOrNot()

        know_or_not.add_azure(
            azure_endpoint="https://endpoint.com",
            azure_api_key="api_key",
            azure_api_version="2023-05-15",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
        )

        # Check that default client is set
        assert know_or_not.default_sync_client is not None

        # Check that default client is the Azure client
        assert (
            know_or_not.default_sync_client.enum_name == SyncLLMClientEnum.AZURE_OPENAI
        )
        assert isinstance(know_or_not.default_sync_client, SyncAzureOpenAIClient)

    @patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
    def test_get_client_returns_azure_client(self, mock_prompt):
        """Test that get_client() returns the Azure client."""
        # Mock the prompt call made during initialization
        mock_prompt.return_value = "Hello"

        know_or_not = KnowOrNot()

        know_or_not.add_azure(
            azure_endpoint="https://endpoint.com",
            azure_api_key="api_key",
            azure_api_version="2023-05-15",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
        )

        # Get client with no arguments should return default (Azure) client
        client = know_or_not.get_client()
        assert client.enum_name == SyncLLMClientEnum.AZURE_OPENAI
        assert isinstance(client, SyncAzureOpenAIClient)

        # Get client with specific enum should return Azure client
        client = know_or_not.get_client(SyncLLMClientEnum.AZURE_OPENAI)
        assert client.enum_name == SyncLLMClientEnum.AZURE_OPENAI
        assert isinstance(client, SyncAzureOpenAIClient)

    @patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
    def test_azure_client_configuration(self, mock_prompt):
        """Test that the Azure client is configured correctly."""
        # Mock the prompt call made during initialization
        mock_prompt.return_value = "Hello"

        azure_endpoint = "https://endpoint.com"
        azure_api_key = "api_key"
        azure_api_version = "2023-05-15"

        know_or_not = KnowOrNot()

        know_or_not.add_azure(
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_api_version=azure_api_version,
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
        )

        # Get the Azure client
        client = cast(
            SyncAzureOpenAIClient,
            know_or_not.get_client(SyncLLMClientEnum.AZURE_OPENAI),
        )
        # Check client configuration matches input parameters
        assert client.config.endpoint == azure_endpoint
        assert client.config.api_key == azure_api_key
        assert client.config.api_version == azure_api_version

    @patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
    @patch("src.knowornot.SyncLLMClient.azure_client.AzureOpenAI")
    def test_azure_client_initialization(self, mock_azure_openai, mock_prompt):
        """Test that the Azure client initializes the underlying API client correctly."""
        # Mock the prompt call made during initialization
        mock_prompt.return_value = "Hello"

        azure_endpoint = "https://endpoint.com"
        azure_api_key = "api_key"
        azure_api_version = "2023-05-15"

        mock_azure_client = MagicMock()
        mock_azure_openai.return_value = mock_azure_client

        know_or_not = KnowOrNot()

        know_or_not.add_azure(
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_api_version=azure_api_version,
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
        )

        # Check that AzureOpenAI was initialized with correct parameters
        mock_azure_openai.assert_called_once_with(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )
