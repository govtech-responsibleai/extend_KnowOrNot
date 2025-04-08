import os
from typing import Optional

from src.knowornot.config import AzureOpenAIConfig, Config


class KnowOrNot:
    def __init__(self, config: Config):
        """
        Initialize the KnowOrNot instance with the given configuration.

        Note: Directly setting the configuration like this is not recommended
        as it may lead to tightly coupled code and reduced flexibility.
        Consider using dependency injection or a configuration manager instead.
        """
        self.config = config

    @staticmethod
    def create_from_azure(
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_batch_endpoint: Optional[str] = None,
        azure_batch_api_key: Optional[str] = None,
        separate_batch_client: bool = False,
    ) -> "KnowOrNot":
        """
        Create a `KnowOrNot` instance using Azure configuration details.

        This is the recommended way to instantiate a KnowOrNot instance.


        Parameters:
        - azure_endpoint (Optional[str]): The Azure endpoint. If not provided,
        the environment variable `AZURE_OPENAI_ENDPOINT` will be used.
        - azure_api_key (Optional[str]): The Azure API key. If not provided,
        the environment variable `AZURE_OPENAI_API_KEY` will be used.
        - azure_batch_endpoint (Optional[str]): The Azure batch endpoint. If not
        provided, the environment variable `AZURE_OPENAI_BATCH_ENDPOINT` will be used.
        - azure_batch_api_key (Optional[str]): The Azure batch API key. If not
        provided, the environment variable `AZURE_OPENAI_BATCH_API_KEY` will be used.
        - separate_batch_client (bool): Determines whether to use a separate batch
        client. If `False`, `azure_batch_endpoint` and `azure_batch_api_key` should
        not be provided, and will default to the values of `azure_endpoint` and
        `azure_api_key` respectively. Is False by default.

        Returns:
        - KnowOrNot: An instance of the `KnowOrNot` class configured with the specified
        Azure settings.

        Example:
        1. Using environment variables: KnowOrNot.create_from_azure()

        2. Providing all parameters: KnowOrNot.create_from_azure(
                azure_endpoint="https://example.com",
                azure_api_key="example_key",
                azure_batch_endpoint="https://batch.example.com",
                azure_batch_api_key="batch_key",
                separate_batch_client=True
            )

        3. Using default batch client: KnowOrNot.create_from_azure(
                azure_endpoint="https://example.com",
                azure_api_key="example_key"
            )
        """

        if not azure_endpoint:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise EnvironmentError(
                    "AZURE_OPENAI_ENDPOINT is not set and azure_endpoint is not provided"
                )
        if not azure_api_key:
            azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not azure_api_key:
                raise EnvironmentError(
                    "AZURE_OPENAI_API_KEY is not set and azure_api_key is not provided"
                )

        if not separate_batch_client:
            if azure_batch_endpoint or azure_batch_api_key:
                raise ValueError(
                    "If separate_batch_client is false, azure_batch_endpoint and azure_batch_api_key should not be provided"
                )

            azure_batch_endpoint = azure_endpoint
            azure_batch_api_key = azure_api_key

        else:
            if not azure_batch_endpoint:
                azure_batch_endpoint = os.environ.get("AZURE_OPENAI_BATCH_ENDPOINT")
                if not azure_batch_endpoint:
                    raise EnvironmentError(
                        "AZURE_OPENAI_BATCH_ENDPOINT is not set and azure_batch_endpoint is not provided"
                    )
            if not azure_batch_api_key:
                azure_batch_api_key = os.environ.get("AZURE_OPENAI_BATCH_API_KEY")
                if not azure_batch_api_key:
                    raise EnvironmentError(
                        "AZURE_OPENAI_BATCH_API_KEY is not set and azure_batch_api_key is not provided"
                    )

        return KnowOrNot(
            Config(
                azure_config=AzureOpenAIConfig(
                    endpoint=azure_endpoint, api_key=azure_api_key
                ),
                azure_batch_config=AzureOpenAIConfig(
                    endpoint=azure_batch_endpoint, api_key=azure_batch_api_key
                ),
            )
        )
