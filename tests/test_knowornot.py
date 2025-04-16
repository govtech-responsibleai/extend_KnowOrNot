import os
from unittest.mock import patch
import logging

import pytest

from src.knowornot import KnowOrNot
from src.knowornot.config import AzureOpenAIConfig


class TestKnowOrNot:
    def test_init(self):
        """Test basic initialization"""
        know_or_not = KnowOrNot()
        assert know_or_not.client_registry == {}
        assert know_or_not.default_sync_client is None
        assert know_or_not.fact_manager is None

    def test_create_from_azure_with_params(self):
        """Test factory method with all parameters provided."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint="https://endpoint.com",
            azure_api_key="api_key",
            azure_api_version="2023-05-15",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
        )

        # Test that a client was registered
        assert len(know_or_not.client_registry) == 1
        assert know_or_not.default_sync_client is not None

        # Test the client's configuration
        client = know_or_not.default_sync_client
        assert isinstance(client.config, AzureOpenAIConfig)
        assert client.config.endpoint == "https://endpoint.com"
        assert client.config.api_key == "api_key"
        assert client.config.api_version == "2023-05-15"
        assert client.config.default_model == "gpt-4"
        assert client.config.default_embedding_model == "text-embedding-3-large"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://env.endpoint.com",
            "AZURE_OPENAI_API_KEY": "env_api_key",
            "AZURE_OPENAI_API_VERSION": "2023-05-15",
            "AZURE_OPENAI_DEFAULT_MODEL": "gpt-4",
            "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL": "text-embedding-3-large",
        },
    )
    def test_create_from_azure_with_env_vars(self):
        """Test factory method using environment variables."""
        know_or_not = KnowOrNot.create_from_azure()

        # Test that a client was registered
        assert len(know_or_not.client_registry) == 1
        assert know_or_not.default_sync_client is not None

        # Test the client's configuration
        client = know_or_not.default_sync_client
        assert isinstance(client.config, AzureOpenAIConfig)
        assert client.config.endpoint == "https://env.endpoint.com"
        assert client.config.api_key == "env_api_key"
        assert client.config.api_version == "2023-05-15"
        assert client.config.default_model == "gpt-4"
        assert client.config.default_embedding_model == "text-embedding-3-large"

    def test_missing_endpoint_error(self):
        """Test error when endpoint is not provided and not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                EnvironmentError, match="AZURE_OPENAI_ENDPOINT is not set"
            ):
                KnowOrNot.create_from_azure(
                    azure_api_key="api_key",
                    azure_api_version="2023-05-15",
                )

    def test_missing_api_key_error(self):
        """Test error when API key is not provided and not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                EnvironmentError, match="AZURE_OPENAI_API_KEY is not set"
            ):
                KnowOrNot.create_from_azure(
                    azure_endpoint="https://endpoint.com",
                    azure_api_version="2023-05-15",
                )

    def test_missing_api_version_error(self):
        """Test error when API version is not provided and not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                EnvironmentError, match="AZURE_OPENAI_API_VERSION is not set"
            ):
                KnowOrNot.create_from_azure(
                    azure_endpoint="https://endpoint.com",
                    azure_api_key="api_key",
                )

    def test_api_key_and_endpoint_validation(self):
        """Test that the api_key and endpoint are validated in AzureOpenAIConfig."""
        with pytest.raises(
            ValueError, match="api_key is required for AzureOpenAIConfig"
        ):
            AzureOpenAIConfig(
                api_key="",
                endpoint="https://example.com",
                api_version="2023-05-15",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
                logger=logging.getLogger(__name__),
            )

        with pytest.raises(
            ValueError, match="endpoint is required for AzureOpenAIConfig"
        ):
            AzureOpenAIConfig(
                api_key="api_key",
                endpoint="",
                api_version="2023-05-15",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
                logger=logging.getLogger(__name__),
            )

        # Should not raise an error with valid values
        AzureOpenAIConfig(
            api_key="api_key",
            endpoint="https://example.com",
            api_version="2023-05-15",
            default_model="gpt-4",
            default_embedding_model="text-embedding-3-large",
            logger=logging.getLogger(__name__),
        )
