import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.knowornot.QuestionExtractor import QuestionExtractor
from src.knowornot.QuestionExtractor.models import QuestionList, FilterMethod
from src.knowornot.common.models import QAPairIntermediate, QAPairLLM
from src.knowornot.common.models import AtomicFactDocument, AtomicFact
from src.knowornot.SyncLLMClient import SyncLLMClient


class TestQuestionExtractor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_llm_client = MagicMock(spec=SyncLLMClient)
        self.question_prompt_default = "Generate a question about this document:"
        self.context_prompt = "Some context."
        self.question_extractor = QuestionExtractor(
            question_prompt_default=self.question_prompt_default,
            default_client=self.mock_llm_client,
            logger=MagicMock(),
        )

    def test_construct_text_to_llm(self):
        document = AtomicFactDocument(
            fact_list=[AtomicFact(fact_text="Test fact", source_citation=0)]
        )
        expected_text = (
            self.context_prompt
            + self.question_prompt_default
            + "The document is "
            + str(document)
        )

        actual_text = self.question_extractor._construct_text_to_llm(
            context_prompt=self.context_prompt,
            question_prompt=self.question_prompt_default,
            document=document,
        )

        assert actual_text == expected_text

    def test_generate_questions_from_document(self):
        document = AtomicFactDocument(
            fact_list=[
                AtomicFact(fact_text="Fact 1", source_citation=0),
                AtomicFact(fact_text="Fact 2", source_citation=1),
            ]
        )

        # Create question list to return
        qa_pair_llm1 = QAPairLLM(question="Question 1?", answer="Answer 1.")
        qa_pair_llm2 = QAPairLLM(question="Question 2?", answer="Answer 2.")
        question_list = QuestionList(questions=[qa_pair_llm1, qa_pair_llm2])

        self.mock_llm_client.get_structured_response.return_value = question_list

        expected_qa_pairs = [
            QAPairIntermediate(question="Question 1?", answer="Answer 1."),
            QAPairIntermediate(question="Question 2?", answer="Answer 2."),
        ]

        actual_qa_pairs = self.question_extractor._generate_questions_from_document(
            llm_client=self.mock_llm_client,
            document=document,
            context_prompt=self.context_prompt,
        )

        assert len(actual_qa_pairs) == 2
        assert actual_qa_pairs == expected_qa_pairs
        self.mock_llm_client.get_structured_response.assert_called_once()

    def test_get_diverse_questions_keyword(self):
        qa_pairs = [
            QAPairIntermediate(
                question="What is machine learning?", answer="A field of AI"
            ),
            QAPairIntermediate(
                question="Define machine learning", answer="A technique"
            ),
            QAPairIntermediate(
                question="What is quantum computing?",
                answer="Computing using quantum mechanics",
            ),
        ]

        with patch.object(
            self.question_extractor,
            "_filter_keyword_duplicates",
            return_value=[qa_pairs[0], qa_pairs[2]],
        ) as mock_filter:
            result = self.question_extractor._get_diverse_questions(
                qa_pairs, method=FilterMethod.KEYWORD, diversity_threshold_keyword=0.3
            )

            assert result == [qa_pairs[0], qa_pairs[2]]
            mock_filter.assert_called_once_with(qa_pairs, 0.3)

    def test_get_diverse_questions_semantic(self):
        qa_pairs = [
            QAPairIntermediate(
                question="What is machine learning?", answer="A field of AI"
            ),
            QAPairIntermediate(
                question="Define artificial intelligence",
                answer="Intelligence demonstrated by machines",
            ),
        ]

        with patch.object(
            self.question_extractor,
            "_filter_semantic_duplicates",
            return_value=[qa_pairs[0]],
        ) as mock_filter:
            result = self.question_extractor._get_diverse_questions(
                qa_pairs, method=FilterMethod.SEMANTIC, diversity_threshold_semantic=0.3
            )

            assert result == [qa_pairs[0]]
            mock_filter.assert_called_once_with(qa_pairs, 0.3)

    def test_get_diverse_questions_both(self):
        qa_pairs = [
            QAPairIntermediate(
                question="What is machine learning?", answer="A field of AI"
            ),
            QAPairIntermediate(question="Define ML", answer="A technique"),
            QAPairIntermediate(
                question="What is quantum computing?",
                answer="Computing using quantum mechanics",
            ),
        ]

        with patch.object(
            self.question_extractor,
            "_filter_keyword_duplicates",
            return_value=[qa_pairs[0], qa_pairs[2]],
        ) as mock_keyword_filter:
            with patch.object(
                self.question_extractor,
                "_filter_semantic_duplicates",
                return_value=[qa_pairs[2]],
            ) as mock_semantic_filter:
                result = self.question_extractor._get_diverse_questions(
                    qa_pairs,
                    method=FilterMethod.BOTH,
                    diversity_threshold_keyword=0.3,
                    diversity_threshold_semantic=0.3,
                )

                assert result == [qa_pairs[2]]
                mock_keyword_filter.assert_called_once_with(qa_pairs, 0.3)
                mock_semantic_filter.assert_called_once_with(
                    [qa_pairs[0], qa_pairs[2]], 0.3
                )

    def test_no_diverse_questions_raises_error(self):
        document = AtomicFactDocument(
            fact_list=[AtomicFact(fact_text="Test fact", source_citation=0)]
        )

        # Mock _generate_questions_from_document to return some questions
        with patch.object(
            self.question_extractor,
            "_generate_questions_from_document",
            return_value=[QAPairIntermediate(question="Q?", answer="A")],
        ):
            # Mock _get_diverse_questions to return an empty list
            with patch.object(
                self.question_extractor, "_get_diverse_questions", return_value=[]
            ):
                # Test that it raises a ValueError
                with pytest.raises(ValueError, match="No diverse questions generated"):
                    self.question_extractor.generate_questions_from_document(
                        llm_client=self.mock_llm_client,
                        identifier="test-123",
                        document=document,
                        context_prompt=self.context_prompt,
                        method=FilterMethod.KEYWORD,
                        path_to_save=Path("/tmp/error_test.json"),
                    )
