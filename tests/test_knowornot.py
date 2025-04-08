import os
from unittest.mock import patch

import pytest

from src.knowornot import KnowOrNot
from src.knowornot.config import AzureOpenAIConfig, Config


class TestKnowOrNot:
    def test_init(self):
        """Test basic initialization with a config object."""
        config = Config(
            azure_config=AzureOpenAIConfig(
                api_key="api_key", endpoint="https://endpoint.com"
            ),
            azure_batch_config=AzureOpenAIConfig(
                api_key="batch_key", endpoint="https://batch.endpoint.com"
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

    def test_create_from_azure_with_params(self):
        """Test factory method with all parameters provided."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint="https://endpoint.com",
            azure_api_key="api_key",
            azure_batch_endpoint="https://batch.endpoint.com",
            azure_batch_api_key="batch_key",
            separate_batch_client=True,
        )

        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://endpoint.com"
        assert know_or_not.config.azure_config.api_key == "api_key"
        assert know_or_not.config.azure_batch_config is not None
        assert (
            know_or_not.config.azure_batch_config.endpoint
            == "https://batch.endpoint.com"
        )
        assert know_or_not.config.azure_batch_config.api_key == "batch_key"

    def test_create_from_azure_without_separate_batch(self):
        """Test factory method without separate batch client."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint="https://endpoint.com",
            azure_api_key="api_key",
            separate_batch_client=False,
        )

        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://endpoint.com"
        assert know_or_not.config.azure_config.api_key == "api_key"
        assert know_or_not.config.azure_batch_config is not None
        assert know_or_not.config.azure_batch_config.endpoint == "https://endpoint.com"
        assert know_or_not.config.azure_batch_config.api_key == "api_key"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://env.endpoint.com",
            "AZURE_OPENAI_API_KEY": "env_api_key",
        },
    )
    def test_create_from_azure_with_env_vars(self):
        """Test factory method using environment variables."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint=None, azure_api_key=None, separate_batch_client=False
        )

        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://env.endpoint.com"
        assert know_or_not.config.azure_config.api_key == "env_api_key"
        assert know_or_not.config.azure_batch_config is not None
        assert (
            know_or_not.config.azure_batch_config.endpoint == "https://env.endpoint.com"
        )
        assert know_or_not.config.azure_batch_config.api_key == "env_api_key"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://env.endpoint.com",
            "AZURE_OPENAI_API_KEY": "env_api_key",
            "AZURE_OPENAI_BATCH_ENDPOINT": "https://env.batch.endpoint.com",
            "AZURE_OPENAI_BATCH_API_KEY": "env_batch_key",
        },
    )
    def test_create_from_azure_with_separate_batch_env_vars(self):
        """Test factory method using environment variables with separate batch client."""
        know_or_not = KnowOrNot.create_from_azure(
            azure_endpoint=None,
            azure_api_key=None,
            azure_batch_endpoint=None,
            azure_batch_api_key=None,
            separate_batch_client=True,
        )

        assert know_or_not.config.azure_config is not None
        assert know_or_not.config.azure_config.endpoint == "https://env.endpoint.com"
        assert know_or_not.config.azure_config.api_key == "env_api_key"
        assert know_or_not.config.azure_batch_config is not None
        assert (
            know_or_not.config.azure_batch_config.endpoint
            == "https://env.batch.endpoint.com"
        )
        assert know_or_not.config.azure_batch_config.api_key == "env_batch_key"

    def test_create_from_azure_error_when_separate_false_but_batch_params_provided(
        self,
    ):
        """Test error when separate_batch_client is False but batch parameters are provided."""
        with pytest.raises(ValueError, match="If separate_batch_client is false"):
            KnowOrNot.create_from_azure(
                azure_endpoint="https://endpoint.com",
                azure_api_key="api_key",
                azure_batch_endpoint="https://batch.endpoint.com",
                separate_batch_client=False,
            )

    def test_missing_endpoint_error(self):
        """Test error when endpoint is not provided and not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                EnvironmentError, match="AZURE_OPENAI_ENDPOINT is not set"
            ):
                KnowOrNot.create_from_azure(
                    azure_endpoint=None, azure_api_key="api_key"
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
                    azure_endpoint="https://endpoint.com", azure_api_key=None
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
                    azure_batch_endpoint=None,
                    azure_batch_api_key="batch_key",
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
                    azure_batch_endpoint="https://batch.endpoint.com",
                    azure_batch_api_key=None,
                    separate_batch_client=True,
                )

    def test_api_key_and_endpoint_validation(self):
        """Test that the api_key and endpoint are validated in AzureOpenAIConfig."""
        with pytest.raises(
            ValueError, match="api_key is required for AzureOpenAIConfig"
        ):
            AzureOpenAIConfig(api_key="", endpoint="https://example.com")

        with pytest.raises(
            ValueError, match="endpoint is required for AzureOpenAIConfig"
        ):
            AzureOpenAIConfig(api_key="api_key", endpoint="")

        # Should not raise an error with valid values
        AzureOpenAIConfig(api_key="api_key", endpoint="https://example.com")
