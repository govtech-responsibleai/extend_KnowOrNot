import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .QuestionExtractor import QuestionExtractor
from .QuestionExtractor.models import FilterMethod
from .common.models import AtomicFactDocument, QuestionDocument
from .config import AzureOpenAIConfig, Config
from .SyncLLMClient import SyncLLMClient, SyncLLMClientEnum
from .SyncLLMClient.azure_client import SyncAzureOpenAIClient
from .FactManager import FactManager
from .PromptManager import PromptManager

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
        self.fact_manager: Optional[FactManager] = None

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

    def _get_fact_manager(self) -> FactManager:
        if self.fact_manager is not None:
            return self.fact_manager

        if not self.default_sync_client:
            raise ValueError(
                "You must set a LLM Client before performing any fact related operations"
            )

        self.fact_manager = FactManager(
            sync_llm_client=self.default_sync_client,
            default_fact_creation_prompt=PromptManager.default_fact_extraction_prompt,
            logger=self.config.logger,
        )

        return self.fact_manager

    def _get_question_manager(self) -> QuestionExtractor:
        if self.question_manager is not None:
            return self.question_manager

        if not self.default_sync_client:
            raise ValueError(
                "You must set a LLM Client before performing any question related operations"
            )

        self.question_manager = QuestionExtractor(
            question_prompt_default=PromptManager.default_question_extraction_prompt,
            default_client=self.default_sync_client,
            logger=self.config.logger,
        )

        return self.question_manager

    @staticmethod
    def create_from_azure(
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_batch_endpoint: Optional[str] = None,
        azure_batch_api_key: Optional[str] = None,
        azure_batch_api_version: Optional[str] = None,
        default_synchronous_model: Optional[str] = None,
        default_batch_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        default_batch_embedding_model: Optional[str] = None,
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
        - default_synchronous_model (Optional[str]): The default model for synchronous operations.
        If not provided, the environment variable `AZURE_OPENAI_DEFAULT_MODEL` will be used,
        with a fallback to "gpt-4o".
        - default_batch_model (Optional[str]): The default model for batch operations.
        If not provided, the environment variable `AZURE_OPENAI_DEFAULT_BATCH_MODEL` will be used,
        with a fallback to "gpt-4o".
        - default_embedding_model (Optional[str]): The default embedding model.
        If not provided, the environment variable `AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL` will be used,
        with a fallback to "text-embedding-3-large".
        - default_batch_embedding_model (Optional[str]): The default embedding model for batch operations.
        If not provided, the environment variable `AZURE_OPENAI_DEFAULT_BATCH_EMBEDDING_MODEL` will be used,
        with a fallback to "text-embedding-3-large".
        - separate_batch_client (bool): Determines whether to use a separate batch
        client. If `False`, `azure_batch_endpoint`, `azure_batch_api_key`,
        `azure_batch_api_version`, and `default_batch_model` should not be provided, and will
        default to the values of `azure_endpoint`, `azure_api_key`, `azure_api_version`,
        and `default_synchronous_model` respectively. Is False by default.

        Returns:
        - KnowOrNot: An instance of the `KnowOrNot` class configured with the specified
        Azure settings.

        Example:
        1. Using environment variables: KnowOrNot.create_from_azure()

        2. Providing all parameters with separate batch client: KnowOrNot.create_from_azure(
                azure_endpoint="https://example.com",
                azure_api_key="example_key",
                azure_api_version="2023-05-15",
                azure_batch_endpoint="https://batch.example.com",
                azure_batch_api_key="batch_key",
                azure_batch_api_version="2023-05-15",
                default_synchronous_model="gpt-4o",
                default_batch_model="gpt-4o-batch",
                default_embedding_model="text-embedding-3-large",
                default_batch_embedding_model="text-embedding-3-large",
                separate_batch_client=True
            )

        3. Using default batch client: KnowOrNot.create_from_azure(
                azure_endpoint="https://example.com",
                azure_api_key="example_key",
                azure_api_version="2023-05-15",
                default_synchronous_model="gpt-4o"
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
        if not default_synchronous_model:
            default_synchronous_model = os.environ.get(
                "AZURE_OPENAI_DEFAULT_MODEL", "gpt-4o"
            )
        if not default_embedding_model:
            default_embedding_model = os.environ.get(
                "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large"
            )

        if not separate_batch_client:
            if (
                azure_batch_endpoint
                or azure_batch_api_key
                or azure_batch_api_version
                or default_batch_model
                or default_batch_embedding_model
            ):
                raise ValueError(
                    "If separate_batch_client is false, azure_batch_endpoint, azure_batch_api_key, azure_batch_api_version, default_batch_model, and default_batch_embedding_model should not be provided"
                )

            azure_batch_endpoint = azure_endpoint
            azure_batch_api_key = azure_api_key
            azure_batch_api_version = azure_api_version
            default_batch_model = default_synchronous_model
            default_batch_embedding_model = default_embedding_model

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
            if not default_batch_model:
                default_batch_model = os.environ.get(
                    "AZURE_OPENAI_DEFAULT_BATCH_MODEL", "gpt-4o"
                )
            if not default_batch_embedding_model:
                default_batch_embedding_model = os.environ.get(
                    "AZURE_OPENAI_DEFAULT_BATCH_EMBEDDING_MODEL",
                    "text-embedding-3-large",
                )

        logger = logging.getLogger(__name__)

        azure_config = AzureOpenAIConfig(
            endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
            default_model=default_synchronous_model,
            default_embedding_model=default_embedding_model,
            logger=logger,
        )

        azure_batch_config = AzureOpenAIConfig(
            endpoint=azure_batch_endpoint,
            api_key=azure_batch_api_key,
            api_version=azure_batch_api_version,
            default_model=default_batch_model,
            default_embedding_model=default_batch_embedding_model,
            logger=logger,
        )

        output = KnowOrNot(
            Config(
                azure_config=azure_config,
                azure_batch_config=azure_batch_config,
                logger=logger,
            )
        )

        azure_sync_client = SyncAzureOpenAIClient(config=azure_config)

        output.register_client(client=azure_sync_client, make_default=True)

        return output

    def create_facts(
        self,
        source_list: List[Path],
        destination_dir: Optional[Path] = None,
        alternative_prompt: Optional[str] = None,
        alt_llm_client: Optional[SyncLLMClient] = None,
    ) -> List[AtomicFactDocument]:
        """
        Parses a list of source files and converts them to atomic facts using a given LLM client.

        This function is part of the main client's Facade interface, providing a simplified
        entry point for fact extraction.  It abstracts away the need to directly interact
        with the underlying `FactManager`, allowing users to call this method directly
        from the client object.

        For detailed information on the parsing process, parameters, return values, and
        potential exceptions, see `FactManager._parse_source_to_atomic_facts`.
        """

        fact_manager = self._get_fact_manager()
        return fact_manager._parse_source_to_atomic_facts(
            source_list=source_list,
            destination_dir=destination_dir,
            alternative_prompt=alternative_prompt,
            alt_llm_client=alt_llm_client,
        )

    def create_questions(
        self,
        context_prompt: str,
        document: AtomicFactDocument,
        method: FilterMethod,
        path_to_store: Path,
        identifier: str,
        alternative_question_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
        llm_client: Optional[SyncLLMClient] = None,
        diversity_threshold_keyword: float = 0.3,
        diversity_threshold_semantic: float = 0.3,
    ) -> QuestionDocument:
        """
        Generates a diverse list of question-answer pairs from an atomic fact document using an LLM client.

        This method iterates over each atomic fact in the document, calls
        `_generate_question_from_single_fact` to generate a question-answer pair for each fact,
        accumulates the generated question-answer pairs in a list and then filters out the
        non-diverse questions based on the filter method.

        Args:
            llm_client (SyncLLMClient): The LLM client used to generate the question-answer pairs.
            document (AtomicFactDocument): The atomic fact document from which to generate the question-answer pairs.
            context_prompt (str): The context to include in the prompt sent to the LLM.
            alternative_question_prompt (Optional[str]): An optional alternative question prompt to use instead of the default.
            ai_model (Optional[str]): The AI model to use for generating the structured response.
            method (FilterMethod): The method to use for filtering out non-diverse questions.
            path_to_store (Path): The path to store the generated questions.
            diversity_threshold_keyword (float): The threshold for filtering out non-diverse questions based on keyword similarity.
            diversity_threshold_semantic (float): The threshold for filtering out non-diverse questions based on semantic similarity.
            identifier (str): The identifier to assign to the generated questions.

        Returns:
            QuestionDocument: A `QuestionDocument` containing the identifier and a list of diverse question-answer pairs generated from the atomic fact document.

        Raises:
            ValueError: If you must set a LLM Client before performing any question related operations or provide one as an argument.
        """
        client = llm_client or self.default_sync_client

        if not client:
            raise ValueError(
                "You must set a LLM Client before performing any question related operations or provide one as an argument"
            )

        question_extractor = self._get_question_manager()
        return question_extractor.generate_questions_from_document(
            llm_client=client,
            document=document,
            context_prompt=context_prompt,
            path_to_save=path_to_store,
            alternative_question_prompt=alternative_question_prompt,
            ai_model=ai_model,
            diversity_threshold_keyword=diversity_threshold_keyword,
            diversity_threshold_semantic=diversity_threshold_semantic,
            method=method,
            identifier=identifier,
        )
