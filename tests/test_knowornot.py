import os
from unittest.mock import patch
import logging

import pytest

from src.knowornot import KnowOrNot
from src.knowornot.config import AzureOpenAIConfig, Config


class TestKnowOrNot:
    def test_init(self):
        """Test basic initialization with a config object."""
        config = Config(
            azure_config=AzureOpenAIConfig(
                api_key="api_key",
                endpoint="https://endpoint.com",
                api_version="2023-05-15",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
                logger=logging.getLogger(__name__),
            ),
            azure_batch_config=AzureOpenAIConfig(
                api_key="batch_key",
                endpoint="https://batch.endpoint.com",
                api_version="2023-05-15",
                default_model="gpt-4",
                default_embedding_model="text-embedding-3-large",
                logger=logging.getLogger(__name__),
            ),
        )

        know_or_not = KnowOrNot(config)

        assert know_or_not.config == config
        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://endpoint.com"
        assert know_or_not.config.azure_config.api_key == "api_key"
        assert know_or_not.config.azure_batch_config is not None
        assert (
            know_or_not.config.azure_batch_config.endpoint
            == "https://batch.endpoint.com"
        )
        assert know_or_not.config.azure_batch_config.api_key == "batch_key"
        assert (
            know_or_not.config.azure_batch_config.default_embedding_model
            == "text-embedding-3-large"
        )

    def test_create_from_azure_with_params(self):
        """Test factory method with all parameters provided."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint="https://endpoint.com",
            azure_api_key="api_key",
            azure_api_version="2023-05-15",
            azure_batch_endpoint="https://batch.endpoint.com",
            azure_batch_api_key="batch_key",
            azure_batch_api_version="2023-05-15",
            default_embedding_model="text-embedding-3-large",
            separate_batch_client=True,
            default_batch_embedding_model="text-embedding-3-large",
        )

        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://endpoint.com"
        assert know_or_not.config.azure_config.api_key == "api_key"
        assert know_or_not.config.azure_config.api_version == "2023-05-15"
        assert know_or_not.config.azure_batch_config is not None
        assert (
            know_or_not.config.azure_batch_config.endpoint
            == "https://batch.endpoint.com"
        )
        assert know_or_not.config.azure_batch_config.api_key == "batch_key"
        assert (
            know_or_not.config.azure_batch_config.default_embedding_model
            == "text-embedding-3-large"
        )

    def test_create_from_azure_without_separate_batch(self):
        """Test factory method without separate batch client."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint="https://endpoint.com",
            azure_api_key="api_key",
            azure_api_version="2023-05-15",
            default_embedding_model="text-embedding-3-large",
            separate_batch_client=False,
        )

        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://endpoint.com"
        assert know_or_not.config.azure_config.api_key == "api_key"
        assert know_or_not.config.azure_config.api_version == "2023-05-15"
        assert know_or_not.config.azure_batch_config is not None
        assert know_or_not.config.azure_batch_config.endpoint == "https://endpoint.com"
        assert know_or_not.config.azure_batch_config.api_key == "api_key"
        assert know_or_not.config.azure_batch_config.api_version == "2023-05-15"
        assert (
            know_or_not.config.azure_batch_config.default_embedding_model
            == "text-embedding-3-large"
        )

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://env.endpoint.com",
            "AZURE_OPENAI_API_KEY": "env_api_key",
            "AZURE_OPENAI_API_VERSION": "2023-05-15",
            "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL": "text-embedding-3-large",
        },
    )
    def test_create_from_azure_with_env_vars(self):
        """Test factory method using environment variables."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint=None,
            azure_api_key=None,
            azure_api_version=None,
            separate_batch_client=False,
        )

        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://env.endpoint.com"
        assert know_or_not.config.azure_config.api_key == "env_api_key"
        assert know_or_not.config.azure_config.api_version == "2023-05-15"
        assert know_or_not.config.azure_batch_config is not None
        assert (
            know_or_not.config.azure_batch_config.endpoint == "https://env.endpoint.com"
        )
        assert know_or_not.config.azure_batch_config.api_key == "env_api_key"
        assert know_or_not.config.azure_batch_config.api_version == "2023-05-15"
        assert (
            know_or_not.config.azure_batch_config.default_embedding_model
            == "text-embedding-3-large"
        )

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://env.endpoint.com",
            "AZURE_OPENAI_API_KEY": "env_api_key",
            "AZURE_OPENAI_API_VERSION": "2023-05-15",
            "AZURE_OPENAI_BATCH_ENDPOINT": "https://env.batch.endpoint.com",
            "AZURE_OPENAI_BATCH_API_KEY": "env_batch_key",
            "AZURE_OPENAI_BATCH_API_VERSION": "2023-05-15",
            "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL": "text-embedding-3-large",
            "AZURE_OPENAI_DEFAULT_BATCH_EMBEDDING_MODEL": "text-embedding-3-large",
        },
    )
    def test_create_from_azure_with_separate_batch_env_vars(self):
        """Test factory method using environment variables with separate batch client."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint=None,
            azure_api_key=None,
            azure_api_version=None,
            azure_batch_endpoint=None,
            azure_batch_api_key=None,
            azure_batch_api_version=None,
            separate_batch_client=True,
        )

        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://env.endpoint.com"
        assert know_or_not.config.azure_config.api_key == "env_api_key"
        assert know_or_not.config.azure_config.api_version == "2023-05-15"
        assert (
            know_or_not.config.azure_config.default_embedding_model
            == "text-embedding-3-large"
        )
        assert know_or_not.config.azure_batch_config is not None
        assert (
            know_or_not.config.azure_batch_config.endpoint
            == "https://env.batch.endpoint.com"
        )
        assert know_or_not.config.azure_batch_config.api_key == "env_batch_key"
        assert know_or_not.config.azure_batch_config.api_version == "2023-05-15"
        assert (
            know_or_not.config.azure_batch_config.default_embedding_model
            == "text-embedding-3-large"
        )

    def test_create_from_azure_error_when_separate_false_but_batch_params_provided(
        self,
    ):
        """Test error when separate_batch_client is False but batch parameters are provided."""
        with pytest.raises(ValueError, match="If separate_batch_client is false"):
            KnowOrNot.create_from_azure(
                azure_endpoint="https://endpoint.com",
                azure_api_key="api_key",
                azure_api_version="2023-05-15",
                azure_batch_endpoint="https://batch.endpoint.com",
                default_embedding_model="text-embedding-3-large",
                default_batch_embedding_model="text-embedding-3-large",
                separate_batch_client=False,
            )

    def test_missing_endpoint_error(self):
        """Test error when endpoint is not provided and not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                EnvironmentError, match="AZURE_OPENAI_ENDPOINT is not set"
            ):
                KnowOrNot.create_from_azure(
                    azure_endpoint=None,
                    azure_api_key="api_key",
                    azure_api_version="2023-05-15",
                )

    def test_missing_api_key_error(self):
        """Test error when API key is not provided and not in environment."""
        with patch.dict(
            os.environ,
            {"AZURE_OPENAI_ENDPOINT": "https://env.endpoint.com"},
            clear=True,
        ):
            with pytest.raises(
                EnvironmentError, match="AZURE_OPENAI_API_KEY is not set"
            ):
                KnowOrNot.create_from_azure(
                    azure_endpoint="https://endpoint.com",
                    azure_api_key=None,
                    azure_api_version="2023-05-15",
                )

    def test_missing_batch_endpoint_error(self):
        """Test error when batch endpoint is not provided and not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                EnvironmentError, match="AZURE_OPENAI_BATCH_ENDPOINT is not set"
            ):
                KnowOrNot.create_from_azure(
                    azure_endpoint="https://endpoint.com",
                    azure_api_key="api_key",
                    azure_api_version="2023-05-15",
                    azure_batch_endpoint=None,
                    azure_batch_api_key="batch_key",
                    azure_batch_api_version="2023-05-15",
                    separate_batch_client=True,
                )

    def test_missing_batch_api_key_error(self):
        """Test error when batch API key is not provided and not in environment."""
        with patch.dict(
            os.environ,
            {"AZURE_OPENAI_BATCH_ENDPOINT": "https://env.batch.endpoint.com"},
            clear=True,
        ):
            with pytest.raises(
                EnvironmentError, match="AZURE_OPENAI_BATCH_API_KEY is not set"
            ):
                KnowOrNot.create_from_azure(
                    azure_endpoint="https://endpoint.com",
                    azure_api_key="api_key",
                    azure_api_version="2023-05-15",
                    azure_batch_endpoint="https://batch.endpoint.com",
                    azure_batch_api_key=None,
                    azure_batch_api_version="2023-05-15",
                    separate_batch_client=True,
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
