from ..SyncLLMClient import SyncLLMClient
from pathlib import Path
from typing import Optional, List
from .models import AtomicFactDocument, Sentence, SplitSourceDocument

import nltk
from nltk.tokenize import sent_tokenize


class DataManager:
    def __init__(
        self, sync_llm_client: SyncLLMClient, default_fact_creation_prompt: str
    ):
        self.sync_llm_client = sync_llm_client
        self.fact_creation_prompt: str = default_fact_creation_prompt
        nltk.download("punkt", quiet=True)

    def _split_sentences(self, text: str) -> List[Sentence]:
        sentence_list = sent_tokenize(text)
        output = [Sentence(text=t) for t in sentence_list]
        return output

    def _convert_source_document_to_facts(
        self,
        document: SplitSourceDocument,
        llm_client: SyncLLMClient,
        alternative_prompt: Optional[str] = None,
    ) -> AtomicFactDocument:
        prompt_to_use = ""
        if not alternative_prompt:
            prompt_to_use = self.fact_creation_prompt
        else:
            prompt_to_use = alternative_prompt

        return llm_client.get_structured_response(
            prompt=prompt_to_use + "The document is" + str(document),
            response_model=AtomicFactDocument,
        )

    def _load_text_file(self, file_path: Path) -> str:
        """
        Load the contents of a text file.

        Parameters:
        - file_path (Path): The path to the text file to be loaded. Must have a .txt extension.

        Returns:
        - str: The contents of the text file.

        Raises:
        - ValueError: If the provided file does not have a .txt extension.
        """

        if not file_path.suffix.lower() == ".txt":
            raise ValueError(f"File must be a .txt file. Got: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def _parse_source_to_atomic_facts(
        self,
        source_list: List[Path],
        destination_dir: Optional[Path] = None,
        alternative_prompt: Optional[str] = None,
        alt_llm_client: Optional[SyncLLMClient] = None,
    ) -> List[AtomicFactDocument]:
        """
        Parse a list of source files and convert them to atomic facts using a given LLM client.

        Parameters:
        - source_list (List[Path]): A list of source files to be parsed. Must be .txt files.
        - destination_dir (Optional[Path]): The directory where the parsed files should be saved. If not provided, the parsed files will not be saved.
        - alternative_prompt (Optional[str]): An alternative prompt to use for fact creation. If not provided, the default prompt will be used.
        - alt_llm_client (Optional[SyncLLMClient]): An alternative LLM client to use for fact creation. If not provided, the default LLM client will be used.

        Returns:
        - List[AtomicFactDocument]: A list of atomic facts generated from the source files.

        Raises:
        - ValueError: If the destination is not a directory or if the LLM client cannot use instructor.
        """
        if destination_dir is not None and not destination_dir.is_dir():
            raise ValueError(f"Destination must be a directory. Got: {destination_dir}")

        output: List[AtomicFactDocument] = []
        llm_client: SyncLLMClient = alt_llm_client or self.sync_llm_client

        if not llm_client.can_use_instructor:
            raise ValueError(
                f"{llm_client.enum_name} cannot use instructor. Use a LLM client that can use instructor"
            )

        for source in source_list:
            source_text = self._load_text_file(source)
            split_document = SplitSourceDocument(
                sentences=self._split_sentences(text=source_text)
            )

            document = self._convert_source_document_to_facts(
                document=split_document,
                llm_client=llm_client,
                alternative_prompt=alternative_prompt,
            )
            output.append(document)
            if destination_dir:
                file_path = destination_dir / f"{source.stem}.json"
                document.save_to_json(file_path)

        return output
