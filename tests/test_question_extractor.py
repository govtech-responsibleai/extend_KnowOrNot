import pytest
from unittest.mock import patch, MagicMock

from src.knowornot.QuestionExtractor import QuestionExtractor
from src.knowornot.QuestionExtractor.models import QAPair, QAPairLLM
from src.knowornot.FactManager.models import AtomicFactDocument, AtomicFact
from src.knowornot.SyncLLMClient import SyncLLMClient


class TestQuestionExtractor:  # No longer inheriting from unittest.TestCase
    @pytest.fixture(autouse=True)  # Run before each test
    def setup(self):
        self.mock_llm_client = MagicMock(spec=SyncLLMClient)
        self.question_generation_prompt = "Generate a question about this fact:"
        self.question_extractor = QuestionExtractor(
            question_generation_prompt=self.question_generation_prompt,
            default_client=self.mock_llm_client,
        )

    def test_construct_text_to_llm(self):
        fact = AtomicFact(fact_text="Test fact", source_citation=0)
        expected_text = self.question_generation_prompt + "The fact is " + str(fact)

        actual_text = self.question_extractor._construct_text_to_llm(
            prompt=self.question_generation_prompt, fact=fact
        )

        assert actual_text == expected_text  # Use assert instead of self.assertEqual

    def test_generate_question_from_single_fact(self):
        fact = AtomicFact(fact_text="Test fact", source_citation=0)
        qa_pair_llm = QAPairLLM(question="Test question?", answer="Test answer.")
        self.mock_llm_client.get_structured_response.return_value = qa_pair_llm

        expected_qa_pair = QAPair(
            question="Test question?", answer="Test answer.", source=fact
        )

        actual_qa_pair = self.question_extractor._generate_question_from_single_fact(
            llm_client=self.mock_llm_client, fact=fact
        )

        self.mock_llm_client.get_structured_response.assert_called_once()
        assert actual_qa_pair == expected_qa_pair

    def test_generate_question_from_document(self):
        fact1 = AtomicFact(fact_text="Fact 1", source_citation=0)
        fact2 = AtomicFact(fact_text="Fact 2", source_citation=1)
        document = AtomicFactDocument(fact_list=[fact1, fact2])

        qa_pair_llm1 = QAPairLLM(question="Question 1?", answer="Answer 1.")
        qa_pair_llm2 = QAPairLLM(question="Question 2?", answer="Answer 2.")

        self.mock_llm_client.get_structured_response.side_effect = [
            qa_pair_llm1,
            qa_pair_llm2,
        ]

        expected_qa_pairs = [
            QAPair(question="Question 1?", answer="Answer 1.", source=fact1),
            QAPair(question="Question 2?", answer="Answer 2.", source=fact2),
        ]

        actual_qa_pairs = self.question_extractor.generate_question_from_document(
            llm_client=self.mock_llm_client, document=document
        )

        assert len(actual_qa_pairs) == 2
        assert actual_qa_pairs == expected_qa_pairs
        assert (
            self.mock_llm_client.get_structured_response.call_count == 2
        )  # Verify called twice

    @pytest.mark.asyncio
    async def test_generate_question_from_document_async(self):
        # Arrange
        fact1 = AtomicFact(fact_text="Fact 1", source_citation=0)
        fact2 = AtomicFact(fact_text="Fact 2", source_citation=1)
        document = AtomicFactDocument(fact_list=[fact1, fact2])

        qa_pair_llm1 = QAPairLLM(question="Question 1?", answer="Answer 1.")
        qa_pair_llm2 = QAPairLLM(question="Question 2?", answer="Answer 2.")

        # Mock the calls to get_structured_response such that it returns a QAPairLLM for each fact
        self.mock_llm_client.get_structured_response.side_effect = [
            qa_pair_llm1,
            qa_pair_llm2,
        ]

        # Mock the return value of the single fact generation method
        with patch.object(
            self.question_extractor,
            "_generate_question_from_single_fact",
            side_effect=[
                QAPair(question="Question 1?", answer="Answer 1.", source=fact1),
                QAPair(question="Question 2?", answer="Answer 2.", source=fact2),
            ],
        ) as mock_single_fact:
            # Act
            actual_qa_pairs = (
                await self.question_extractor.generate_question_from_document_async(
                    llm_client=self.mock_llm_client, document=document
                )
            )

        # Assert
        expected_qa_pairs = [
            QAPair(question="Question 1?", answer="Answer 1.", source=fact1),
            QAPair(question="Question 2?", answer="Answer 2.", source=fact2),
        ]

        assert len(actual_qa_pairs) == 2
        assert actual_qa_pairs == expected_qa_pairs
        assert mock_single_fact.call_count == 2

    def test_generate_question_from_single_fact_alternative_prompt(self):
        fact = AtomicFact(fact_text="Test fact", source_citation=0)
        qa_pair_llm = QAPairLLM(question="Test question?", answer="Test answer.")
        self.mock_llm_client.get_structured_response.return_value = qa_pair_llm
        alternative_prompt = "Alternative prompt"

        expected_qa_pair = QAPair(
            question="Test question?", answer="Test answer.", source=fact
        )

        actual_qa_pair = self.question_extractor._generate_question_from_single_fact(
            llm_client=self.mock_llm_client,
            fact=fact,
            alternative_prompt=alternative_prompt,
        )

        self.mock_llm_client.get_structured_response.assert_called_once()
        # Check that alternative prompt was used
        args, kwargs = self.mock_llm_client.get_structured_response.call_args
        assert alternative_prompt in kwargs.get("prompt", "")
        assert actual_qa_pair == expected_qa_pair
