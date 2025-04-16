import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


from src.knowornot.FactManager import FactManager
from src.knowornot.common.models import (
    SplitSourceDocument,
    Sentence,
    AtomicFactDocument,
    AtomicFact,
)
from src.knowornot.SyncLLMClient import SyncLLMClient
from src.knowornot import KnowOrNot


class TestFactManager(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MagicMock(spec=SyncLLMClient)
        self.mock_llm_client.can_use_instructor = True
        self.default_prompt = "Test prompt"

        self.fact_manager = FactManager(
            sync_llm_client=self.mock_llm_client,
            default_fact_creation_prompt=self.default_prompt,
            logger=MagicMock(),
        )

    @patch("src.knowornot.FactManager.nltk.download")
    def test_init(self, mock_download):
        # Test initialization downloads NLTK data
        fact_manager = FactManager(
            sync_llm_client=self.mock_llm_client,
            default_fact_creation_prompt=self.default_prompt,
            logger=MagicMock(),
        )

        mock_download.assert_called_once_with("punkt_tab")
        self.assertEqual(fact_manager.sync_llm_client, self.mock_llm_client)
        self.assertEqual(fact_manager.fact_creation_prompt, self.default_prompt)

    @patch("src.knowornot.FactManager.sent_tokenize")
    def test_split_sentences_normal(self, mock_sent_tokenize):
        # Test normal text splitting
        mock_sent_tokenize.return_value = ["Sentence one.", "Sentence two."]

        result = self.fact_manager._split_sentences("Sentence one. Sentence two.")

        mock_sent_tokenize.assert_called_once_with("Sentence one. Sentence two.")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Sentence one.")
        self.assertEqual(result[1].text, "Sentence two.")

    @patch("src.knowornot.FactManager.sent_tokenize")
    def test_split_sentences_empty(self, mock_sent_tokenize):
        # Test empty text
        mock_sent_tokenize.return_value = []

        result = self.fact_manager._split_sentences("")

        mock_sent_tokenize.assert_called_once_with("")
        self.assertEqual(len(result), 0)

    def test_update_llm_client(self):
        new_client = MagicMock(spec=SyncLLMClient)

        self.fact_manager.update_llm_client(new_client)

        self.assertEqual(self.fact_manager.sync_llm_client, new_client)

    def test_convert_source_document_to_facts_default_prompt(self):
        document = SplitSourceDocument(
            sentences=[Sentence(text="This is a test sentence.")]
        )

        expected_facts = AtomicFactDocument(
            fact_list=[AtomicFact(fact_text="This is a test fact.", source_citation=0)]
        )

        self.mock_llm_client.get_structured_response.return_value = expected_facts

        result = self.fact_manager._convert_source_document_to_facts(
            document=document, llm_client=self.mock_llm_client
        )

        self.mock_llm_client.get_structured_response.assert_called_once()
        self.assertEqual(result, expected_facts)

    def test_convert_source_document_to_facts_alternative_prompt(self):
        document = SplitSourceDocument(
            sentences=[Sentence(text="This is a test sentence.")]
        )

        expected_facts = AtomicFactDocument(
            fact_list=[AtomicFact(fact_text="This is a test fact.", source_citation=0)]
        )

        alt_prompt = "Alternative prompt"

        self.mock_llm_client.get_structured_response.return_value = expected_facts

        result = self.fact_manager._convert_source_document_to_facts(
            document=document,
            llm_client=self.mock_llm_client,
            alternative_prompt=alt_prompt,
        )

        # Check that alternative prompt was used
        args, kwargs = self.mock_llm_client.get_structured_response.call_args
        self.assertIn(alt_prompt, kwargs.get("prompt", ""))

        self.assertEqual(result, expected_facts)

    @patch("builtins.open", new_callable=mock_open, read_data="Test content")
    def test_load_text_file_valid(self, mock_file):
        file_path = Path("test.txt")

        result = self.fact_manager._load_text_file(file_path)

        mock_file.assert_called_once_with(file_path, "r", encoding="utf-8")
        self.assertEqual(result, "Test content")

    def test_load_text_file_invalid_extension(self):
        file_path = Path("test.pdf")

        with self.assertRaises(ValueError) as context:
            self.fact_manager._load_text_file(file_path)

        self.assertIn("File must be a .txt file", str(context.exception))

    @patch.object(FactManager, "_load_text_file")
    @patch.object(FactManager, "_split_sentences")
    @patch.object(FactManager, "_convert_source_document_to_facts")
    def test_parse_source_to_atomic_facts_normal(
        self, mock_convert, mock_split, mock_load
    ):
        # Setup mocks
        mock_load.return_value = "Test content"
        mock_split.return_value = [Sentence(text="Test sentence.")]
        expected_facts = AtomicFactDocument(
            fact_list=[AtomicFact(fact_text="Test fact.", source_citation=0)]
        )
        mock_convert.return_value = expected_facts

        # Test with single source
        source_list = [Path("test.txt")]

        result = self.fact_manager._parse_source_to_atomic_facts(source_list)

        mock_load.assert_called_once_with(Path("test.txt"))
        mock_split.assert_called_once()
        mock_convert.assert_called_once()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], expected_facts)

    @patch.object(FactManager, "_load_text_file")
    @patch.object(FactManager, "_split_sentences")
    @patch.object(FactManager, "_convert_source_document_to_facts")
    @patch.object(AtomicFactDocument, "save_to_json")
    def test_parse_source_to_atomic_facts_with_destination(
        self, mock_save, mock_convert, mock_split, mock_load
    ):
        # Setup mocks
        mock_load.return_value = "Test content"
        mock_split.return_value = [Sentence(text="Test sentence.")]
        expected_facts = AtomicFactDocument(
            fact_list=[AtomicFact(fact_text="Test fact.", source_citation=0)]
        )
        mock_convert.return_value = expected_facts

        # Mock Path.is_dir to return True
        with patch.object(Path, "is_dir", return_value=True):
            # Test with destination directory
            source_list = [Path("test.txt")]
            destination_dir = Path("/output")

            result = self.fact_manager._parse_source_to_atomic_facts(
                source_list=source_list, destination_dir=destination_dir
            )

            mock_save.assert_called_once_with(Path("/output/test.json"))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], expected_facts)

    def test_parse_source_to_atomic_facts_invalid_destination(self):
        # Mock Path.is_dir to return False
        with patch.object(Path, "is_dir", return_value=False):
            source_list = [Path("test.txt")]
            destination_dir = Path("/output")

            with self.assertRaises(ValueError) as context:
                self.fact_manager._parse_source_to_atomic_facts(
                    source_list=source_list, destination_dir=destination_dir
                )

            self.assertIn("Destination must be a directory", str(context.exception))

    def test_parse_source_to_atomic_facts_llm_cant_use_instructor(self):
        # Set LLM client to not support instructor
        self.mock_llm_client.can_use_instructor = False

        source_list = [Path("test.txt")]

        with self.assertRaises(ValueError) as context:
            self.fact_manager._parse_source_to_atomic_facts(source_list=source_list)

        self.assertIn("cannot use instructor", str(context.exception))

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"fact_list": [{"fact_text": "Test fact.", "source_citation": 0}]}',
    )
    @patch.object(Path, "is_dir", return_value=True)
    @patch.object(AtomicFactDocument, "save_to_json")
    def test_parse_and_read_saved_facts(self, mock_save, mock_is_dir, mock_file):
        # Setup mocks
        expected_facts = AtomicFactDocument(
            fact_list=[AtomicFact(fact_text="Test fact.", source_citation=0)]
        )

        # Mock the parsing process
        with (
            patch.object(
                self.fact_manager, "_load_text_file", return_value="Test content"
            ),
            patch.object(
                self.fact_manager,
                "_split_sentences",
                return_value=[Sentence(text="Test sentence.")],
            ),
            patch.object(
                self.fact_manager,
                "_convert_source_document_to_facts",
                return_value=expected_facts,
            ),
        ):
            # Test with destination directory
            source_list = [Path("test.txt")]
            destination_dir = Path("/output")

            # Run parse_source_to_atomic_facts
            self.fact_manager._parse_source_to_atomic_facts(
                source_list=source_list, destination_dir=destination_dir
            )

            # Verify save was called
            mock_save.assert_called_once_with(Path("/output/test.json"))

            # Parse the JSON content back to an AtomicFactDocument
            loaded_facts = AtomicFactDocument.load_from_json(Path("/output/test.json"))

            # Verify the loaded facts match the original
            self.assertEqual(len(loaded_facts.fact_list), 1)
            self.assertEqual(loaded_facts.fact_list[0].fact_text, "Test fact.")
            self.assertEqual(loaded_facts.fact_list[0].source_citation, 0)
            self.assertEqual(loaded_facts, expected_facts)


class TestKnowOrNotFactManager(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.mock_llm_client = MagicMock(spec=SyncLLMClient)
        self.know_or_not = KnowOrNot(self.config)
        self.know_or_not.default_sync_client = self.mock_llm_client

    @patch("src.knowornot.FactManager")
    def test_get_fact_manager_creates_new(self, mock_fact_manager_class):
        mock_instance = MagicMock()
        mock_fact_manager_class.return_value = mock_instance

        # Test when fact_manager is None
        result = self.know_or_not._get_fact_manager()

        mock_fact_manager_class.assert_called_once()
        self.assertEqual(result, mock_instance)
        self.assertEqual(self.know_or_not.fact_manager, mock_instance)

    def test_get_fact_manager_returns_existing(self):
        # Set an existing fact manager
        mock_fact_manager = MagicMock()
        self.know_or_not.fact_manager = mock_fact_manager

        result = self.know_or_not._get_fact_manager()

        self.assertEqual(result, mock_fact_manager)

    def test_get_fact_manager_no_default_client(self):
        # Remove default client
        self.know_or_not.default_sync_client = None

        with self.assertRaises(ValueError) as context:
            self.know_or_not._get_fact_manager()

        self.assertIn("You must set a LLM Client", str(context.exception))
