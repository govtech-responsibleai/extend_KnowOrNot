import os
from typing import Dict, List, Optional

from .config import AzureOpenAIConfig, Config
from .SyncLLMClient import SyncLLMClient, SyncLLMClientEnum

__all__ = ["KnowOrNot", "SyncLLMClient"]


class KnowOrNot:
    def __init__(self, config: Config):
        """
        Initialize the KnowOrNot instance with the given configuration.

        Note: Directly setting the configuration like this is not recommended
        as it may lead to tightly coupled code and reduced flexibility.
        Consider using dependency injection or a configuration manager instead.
        """
        self.config = config
        self.client_registry: Dict[SyncLLMClientEnum, SyncLLMClient] = {}
        self.default_sync_client: Optional[SyncLLMClient] = None

    def register_client(
        self, client: SyncLLMClient, make_default: bool = False
    ) -> None:
        """
        Register a new LLM client in the registry.

        Parameters:
        - client (SyncLLMClient): The client to register
        - make_default (bool): Whether to set this as the default client
        """
        enum_name = client.enum_name
        self.client_registry[enum_name] = client

        # Set as default if requested or if it's the first client
        if make_default or self.default_sync_client is None:
            self.default_sync_client = client

    def get_client(
        self, client_enum: Optional[SyncLLMClientEnum] = None
    ) -> SyncLLMClient:
        """
        Get a client by its enum type, or return the default if no type specified.

        Parameters:
        - client_enum (Optional[SyncLLMClientEnum]): Enum type of client to retrieve

        Returns:
        - SyncLLMClient: The requested client instance

        Raises:
        - KeyError: If no client of the specified enum type is registered
        - ValueError: If no default client is available (when client_enum is None)
        """
        if client_enum is None:
            if self.default_sync_client is None:
                raise ValueError("No default client available")
            return self.default_sync_client

        if client_enum not in self.client_registry:
            raise KeyError(f"No client with enum {client_enum.name} is registered")

        return self.client_registry[client_enum]

    def get_all_clients(self) -> List[SyncLLMClient]:
        """
        Get all registered clients.

        Returns:
        - List[SyncLLMClient]: List of all registered client instances
        """
        return list(self.client_registry.values())

    def set_default_client(self, client_enum: SyncLLMClientEnum) -> None:
        """
        Set a specific client as the default.

        Parameters:
        - client_enum (SyncLLMClientEnum): Enum of client to set as default

        Raises:
        - KeyError: If no client of the specified type is registered
        """
        if client_enum not in self.client_registry:
            raise KeyError(f"No client with enum {client_enum.name} is registered")

        self.default_sync_client = self.client_registry[client_enum]

    def remove_client(self, client_enum: SyncLLMClientEnum) -> None:
        """
        Remove a client from the registry.

        Parameters:
        - client_enum (SyncLLMClientEnum): Enum of client to remove

        Raises:
        - KeyError: If no client of the specified type is registered
        - ValueError: If attempting to remove the default client
        """
        if client_enum not in self.client_registry:
            raise KeyError(f"No client with enum {client_enum.name} is registered")

        if self.default_sync_client is self.client_registry[client_enum]:
            raise ValueError(
                "Cannot remove the default client. Set a new default first."
            )

        del self.client_registry[client_enum]

    @staticmethod
    def create_from_azure(
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_batch_endpoint: Optional[str] = None,
        azure_batch_api_key: Optional[str] = None,
        azure_batch_api_version: Optional[str] = None,
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
        - azure_api_version (Optional[str]): The Azure API version. If not provided,
        the environment variable `AZURE_OPENAI_API_VERSION` will be used.
        - azure_batch_endpoint (Optional[str]): The Azure batch endpoint. If not
        provided, the environment variable `AZURE_OPENAI_BATCH_ENDPOINT` will be used.
        - azure_batch_api_key (Optional[str]): The Azure batch API key. If not
        provided, the environment variable `AZURE_OPENAI_BATCH_API_KEY` will be used.
        - azure_batch_api_version (Optional[str]): The Azure batch API version. If not
        provided, the environment variable `AZURE_OPENAI_BATCH_API_VERSION` will be used.
        - separate_batch_client (bool): Determines whether to use a separate batch
        client. If `False`, `azure_batch_endpoint`, `azure_batch_api_key`, and
        `azure_batch_api_version` should not be provided, and will default to the
        values of `azure_endpoint`, `azure_api_key`, and `azure_api_version` respectively.
        Is False by default.

        Returns:
        - KnowOrNot: An instance of the `KnowOrNot` class configured with the specified
        Azure settings.

        Example:
        1. Using environment variables: KnowOrNot.create_from_azure()

        2. Providing all parameters: KnowOrNot.create_from_azure(
                azure_endpoint="https://example.com",
                azure_api_key="example_key",
                azure_api_version="2023-05-15",
                azure_batch_endpoint="https://batch.example.com",
                azure_batch_api_key="batch_key",
                azure_batch_api_version="2023-05-15",
                separate_batch_client=True
            )

        3. Using default batch client: KnowOrNot.create_from_azure(
                azure_endpoint="https://example.com",
                azure_api_key="example_key",
                azure_api_version="2023-05-15"
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
        if not azure_api_version:
            azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
            if not azure_api_version:
                raise EnvironmentError(
                    "AZURE_OPENAI_API_VERSION is not set and azure_api_version is not provided"
                )

        if not separate_batch_client:
            if azure_batch_endpoint or azure_batch_api_key or azure_batch_api_version:
                raise ValueError(
                    "If separate_batch_client is false, azure_batch_endpoint, azure_batch_api_key, and azure_batch_api_version should not be provided"
                )

            azure_batch_endpoint = azure_endpoint
            azure_batch_api_key = azure_api_key
            azure_batch_api_version = azure_api_version

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
            if not azure_batch_api_version:
                azure_batch_api_version = os.environ.get(
                    "AZURE_OPENAI_BATCH_API_VERSION"
                )
                if not azure_batch_api_version:
                    raise EnvironmentError(
                        "AZURE_OPENAI_BATCH_API_VERSION is not set and azure_batch_api_version is not provided"
                    )

        return KnowOrNot(
            Config(
                azure_config=AzureOpenAIConfig(
                    endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                ),
                azure_batch_config=AzureOpenAIConfig(
                    endpoint=azure_batch_endpoint,
                    api_key=azure_batch_api_key,
                    api_version=azure_batch_api_version,
                ),
            )
        )
