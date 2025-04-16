import pytest
import numpy as np
from unittest.mock import MagicMock

from src.knowornot.RetrievalStrategy import RetrievalType


from src.knowornot.RetrievalStrategy.direct_experiment import (
    DirectRetrievalStrategy,
)
from src.knowornot.RetrievalStrategy.basic_rag import BasicRAGStrategy
from src.knowornot.RetrievalStrategy.long_in_context import (
    LongInContextStrategy,
)
from src.knowornot.RetrievalStrategy.hyde_rag import HydeRAGStrategy
from src.knowornot.common.models import (
    QAPair,
    QAWithContext,
)
from src.knowornot.SyncLLMClient import SyncLLMClient


class TestExperimentCategories:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create a mock LLM client
        self.mock_llm_client = MagicMock(spec=SyncLLMClient)
        self.mock_llm_client.can_use_instructor = True

        # Initialize experiment classes
        self.direct_exp = DirectRetrievalStrategy(
            default_client=self.mock_llm_client, closest_k=3, logger=MagicMock()
        )
        self.basic_rag = BasicRAGStrategy(
            default_client=self.mock_llm_client, closest_k=3, logger=MagicMock()
        )
        self.long_ctx = LongInContextStrategy(
            default_client=self.mock_llm_client, closest_k=3, logger=MagicMock()
        )

        # For HydeRAGStrategy, we need a hypothetical question prompt
        self.hyde_prompt = "Generate a hypothetical answer for this question:"
        self.hyde_rag = HydeRAGStrategy(
            default_client=self.mock_llm_client,
            hypothetical_question_prompt=self.hyde_prompt,
            closest_k=3,
            logger=MagicMock(),
        )

        # Create sample QA pairs for testing
        self.sample_qa_pairs = [
            QAPair(
                question=f"Question {i}?",
                answer=f"Answer {i}",
            )
            for i in range(1, 6)
        ]

        # Create sample embeddings that will have known cosine similarity
        self.sample_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Q1
                [0.9, 0.1, 0.0],  # Q2 (close to Q1)
                [0.0, 1.0, 0.0],  # Q3
                [0.0, 0.9, 0.1],  # Q4 (close to Q3)
                [0.5, 0.5, 0.0],  # Q5 (between Q1 and Q3)
            ]
        )

        # Make sure the number of QA pairs matches the number of embeddings
        assert len(self.sample_qa_pairs) == self.sample_embeddings.shape[0]

    # Test initialization with invalid client
    def test_init_with_invalid_client(self):
        invalid_client = MagicMock(spec=SyncLLMClient)
        invalid_client.can_use_instructor = False

        with pytest.raises(
            ValueError, match="Default client must be able to use instructor"
        ):
            DirectRetrievalStrategy(default_client=invalid_client, logger=MagicMock())

        with pytest.raises(
            ValueError, match="Default client must be able to use instructor"
        ):
            BasicRAGStrategy(default_client=invalid_client, logger=MagicMock())

        with pytest.raises(
            ValueError, match="Default client must be able to use instructor"
        ):
            LongInContextStrategy(default_client=invalid_client, logger=MagicMock())

        with pytest.raises(
            ValueError, match="Default client must be able to use instructor"
        ):
            HydeRAGStrategy(
                default_client=invalid_client,
                hypothetical_question_prompt="prompt",
                logger=MagicMock(),
            )

    # Test experiment type properties
    def test_experiment_types(self):
        assert self.direct_exp.experiment_type == RetrievalType.DIRECT
        assert self.basic_rag.experiment_type == RetrievalType.BASIC_RAG
        assert self.long_ctx.experiment_type == RetrievalType.LONG_IN_CONTEXT
        assert self.hyde_rag.experiment_type == RetrievalType.HYDE_RAG

    # Test embedding functions
    def test_embed_qa_pair_list(self):
        qa_pairs = self.sample_qa_pairs[:2]
        expected_texts = [str(qa) for qa in qa_pairs]
        self.mock_llm_client.get_embedding.return_value = self.sample_embeddings[
            :2
        ].tolist()

        result = self.direct_exp._embed_qa_pair_list(qa_pairs)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, self.sample_embeddings[:2])
        self.mock_llm_client.get_embedding.assert_called_once_with(expected_texts)

    def test_embed_single_qa_pair(self):
        qa_pair = self.sample_qa_pairs[0]
        self.mock_llm_client.get_embedding.return_value = [
            self.sample_embeddings[0].tolist()
        ]

        result = self.direct_exp._embed_single_qa_pair(qa_pair)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, self.sample_embeddings[0])
        self.mock_llm_client.get_embedding.assert_called_once_with([str(qa_pair)])

    def test_get_n_closest_by_cosine_similarity(self):
        # Setup: Using sample_embeddings where we know the similarity relationships
        single_embedding = self.sample_embeddings[0]  # This is [1.0, 0.0, 0.0]
        embeddings = self.sample_embeddings

        # Q1 is most similar to Q2, then Q5, then Q3/Q4
        closest_indices = self.direct_exp._get_n_closest_by_cosine_similarity(
            single_embedding, embeddings
        )

        # Should return indices [0, 1, 4] (closest_k=3) in order of similarity
        assert len(closest_indices) == 3
        assert closest_indices[0] == 0  # Self (Q1)
        assert closest_indices[1] == 1  # Q2
        assert closest_indices[2] == 4  # Q5

    # DirectRetrievalStrategy tests
    def test_direct_experiment_removal(self):
        question = self.sample_qa_pairs[0]
        removed_index = 0
        remaining_qa = self.sample_qa_pairs[1:]

        result = self.direct_exp._create_single_removal_experiment(
            question_to_ask=question,
            removed_index=removed_index,
            remaining_qa=remaining_qa,
            embeddings=self.sample_embeddings,
        )

        assert isinstance(result, QAWithContext)
        assert result.question == question.question
        assert result.expected_answer == question.answer
        assert result.context_questions is None  # Direct experiment uses no context

    def test_direct_experiment_synthetic(self):
        question = QAPair(
            question="Synthetic question?",
            answer="Synthetic answer",
        )

        result = self.direct_exp._create_single_synthetic_experiment(
            question_to_ask=question,
            question_list=self.sample_qa_pairs,
            embeddings=self.sample_embeddings,
        )

        assert isinstance(result, QAWithContext)
        assert result.question == question.question
        assert (
            result.expected_answer is None
        )  # For synthetic, there's no expected answer
        assert result.context_questions is None  # Direct experiment uses no context

    # BasicRAGStrategy tests
    def test_basic_rag_removal(self):
        # Setup mock for embedding functions
        self.mock_llm_client.get_embedding.return_value = (
            self.sample_embeddings.tolist()
        )

        # Test removing the first question
        question = self.sample_qa_pairs[0]
        removed_index = 0
        remaining_qa = self.sample_qa_pairs[1:]

        # For removal experiment, we're asking Q1 after removing it
        # Q1 embedding is closest to Q2, then Q5, then Q3/Q4
        result = self.basic_rag._create_single_removal_experiment(
            question_to_ask=question,
            removed_index=removed_index,
            remaining_qa=remaining_qa,
            embeddings=self.sample_embeddings,
        )

        assert isinstance(result, QAWithContext)
        assert result.question == question.question
        assert result.expected_answer == question.answer
        assert result.context_questions is not None, (
            "Context questions should not be None for RAG experiment"
        )
        assert (
            len(result.context_questions) == 3
        )  # Should get 3 closest questions (closest_k=3)
        # The closest to Q1 should be Q2, Q5, and then either Q3 or Q4
        assert self.sample_qa_pairs[1] in result.context_questions  # Q2
        assert self.sample_qa_pairs[4] in result.context_questions  # Q5

    def test_basic_rag_synthetic(self):
        synthetic_question = QAPair(
            question="Synthetic question?",
            answer="Synthetic answer",
        )

        # Mock embedding for synthetic question - make it closest to Q3
        synthetic_embedding = np.array([0.1, 0.9, 0.0])
        self.basic_rag._embed_single_qa_pair = MagicMock(
            return_value=synthetic_embedding
        )

        result = self.basic_rag._create_single_synthetic_experiment(
            question_to_ask=synthetic_question,
            question_list=self.sample_qa_pairs,
            embeddings=self.sample_embeddings,
        )

        assert isinstance(result, QAWithContext)
        assert result.question == synthetic_question.question
        assert result.expected_answer is None
        assert result.context_questions is not None, (
            "Context questions should not be None for RAG experiment"
        )
        assert len(result.context_questions) == 3
        # The closest to our synthetic question should be Q3, Q4, and then Q5
        assert self.sample_qa_pairs[2] in result.context_questions  # Q3
        assert self.sample_qa_pairs[3] in result.context_questions  # Q4
        assert self.sample_qa_pairs[4] in result.context_questions  # Q5

    # LongInContextStrategy tests
    def test_long_in_context_removal(self):
        question = self.sample_qa_pairs[0]
        removed_index = 0
        remaining_qa = self.sample_qa_pairs[1:]

        result = self.long_ctx._create_single_removal_experiment(
            question_to_ask=question,
            removed_index=removed_index,
            remaining_qa=remaining_qa,
            embeddings=self.sample_embeddings,
        )

        assert isinstance(result, QAWithContext)
        assert result.question == question.question
        assert result.expected_answer == question.answer
        assert result.context_questions is not None, (
            "Context questions should not be None for LongInContextStrategy experiment"
        )
        assert len(result.context_questions) == len(remaining_qa)
        assert (
            result.context_questions == remaining_qa
        )  # LongInContextStrategy includes all remaining questions

    def test_long_in_context_synthetic(self):
        synthetic_question = QAPair(
            question="Synthetic question?",
            answer="Synthetic answer",
        )

        result = self.long_ctx._create_single_synthetic_experiment(
            question_to_ask=synthetic_question,
            question_list=self.sample_qa_pairs,
            embeddings=self.sample_embeddings,
        )

        assert isinstance(result, QAWithContext)
        assert result.question == synthetic_question.question
        assert result.context_questions is not None, (
            "Context questions should not be None for LongInContextStrategy experiment"
        )
        assert result.expected_answer is None
        assert (
            result.context_questions == self.sample_qa_pairs
        )  # LongInContextStrategy includes all context questions

    # HydeRAGStrategy tests
    def test_hyde_rag_get_hypothetical_answer(self):
        question = self.sample_qa_pairs[0]
        hypothetical_answer = QAPair(
            question="Hypothetical question?",
            answer="Hypothetical answer",
        )

        self.mock_llm_client.get_structured_response.return_value = hypothetical_answer

        result = self.hyde_rag._get_hypothetical_question_answer(
            question_to_ask=question
        )

        assert result == hypothetical_answer
        self.mock_llm_client.get_structured_response.assert_called_once_with(
            prompt=self.hyde_prompt + str(question),
            response_model=QAPair,
            ai_model=None,
        )

    def test_hyde_rag_removal(self):
        question = self.sample_qa_pairs[0]
        removed_index = 0
        remaining_qa = self.sample_qa_pairs[1:]

        # Mock the hypothetical answer generation
        hypothetical_answer = QAPair(
            question="Hypothetical Q1?",
            answer="Hypothetical A1",
        )
        self.hyde_rag._get_hypothetical_question_answer = MagicMock(
            return_value=hypothetical_answer
        )

        # Mock the embedding of the hypothetical answer to be closest to Q3
        hypo_embedding = np.array([0.1, 0.9, 0.0])
        self.hyde_rag._embed_single_qa_pair = MagicMock(return_value=hypo_embedding)

        result = self.hyde_rag._create_single_removal_experiment(
            question_to_ask=question,
            removed_index=removed_index,
            remaining_qa=remaining_qa,
            embeddings=self.sample_embeddings,
        )

        assert isinstance(result, QAWithContext)
        assert result.context_questions is not None, (
            "Context questions should not be None for HydeRAGStrategy experiment"
        )
        assert result.question == question.question
        assert result.expected_answer == question.answer
        assert len(result.context_questions) == 3
        # Based on our mocked hypo_embedding, closest should be Q3, Q4, Q5 from remaining_qa
        assert self.sample_qa_pairs[2] in result.context_questions  # Q3
        assert self.sample_qa_pairs[3] in result.context_questions  # Q4
        assert self.sample_qa_pairs[4] in result.context_questions  # Q5

    def test_hyde_rag_synthetic(self):
        synthetic_question = QAPair(
            question="Synthetic question?",
            answer="Synthetic answer",
        )

        # Mock the hypothetical answer generation
        hypothetical_answer = QAPair(
            question="Hypothetical synthetic Q?",
            answer="Hypothetical synthetic A",
        )
        self.hyde_rag._get_hypothetical_question_answer = MagicMock(
            return_value=hypothetical_answer
        )

        # Mock the embedding of the hypothetical answer to be closest to Q1
        hypo_embedding = np.array([0.9, 0.1, 0.0])
        self.hyde_rag._embed_single_qa_pair = MagicMock(return_value=hypo_embedding)

        result = self.hyde_rag._create_single_synthetic_experiment(
            question_to_ask=synthetic_question,
            question_list=self.sample_qa_pairs,
            embeddings=self.sample_embeddings,
        )

        assert isinstance(result, QAWithContext)
        assert result.question == synthetic_question.question
        assert (
            result.expected_answer == synthetic_question.answer
        )  # For HydeRAGStrategy synthetic, it keeps the answer
        assert result.context_questions is not None, (
            "Context questions should not be None for HydeRAGStrategy experiment"
        )
        assert len(result.context_questions) == 3
        # Based on our mocked hypo_embedding, closest should be Q1, Q2, Q5
        assert self.sample_qa_pairs[0] in result.context_questions  # Q1
        assert self.sample_qa_pairs[1] in result.context_questions  # Q2
        assert self.sample_qa_pairs[4] in result.context_questions  # Q5

    def test_hyde_rag_with_alternative_client(self):
        question = self.sample_qa_pairs[0]
        alternative_client = MagicMock(spec=SyncLLMClient)
        alternative_client.can_use_instructor = True
        alternative_client.get_structured_response.return_value = QAPair(
            question="Alt hypothetical Q?",
            answer="Alt hypothetical A",
        )

        result = self.hyde_rag._get_hypothetical_question_answer(
            question_to_ask=question, alternative_llm_client=alternative_client
        )

        assert isinstance(result, QAPair)
        alternative_client.get_structured_response.assert_called_once()
        self.mock_llm_client.get_structured_response.assert_not_called()

    def test_hyde_rag_with_alternative_client_invalid(self):
        question = self.sample_qa_pairs[0]
        invalid_client = MagicMock(spec=SyncLLMClient)
        invalid_client.can_use_instructor = False

        with pytest.raises(
            ValueError, match="Alternative client must be able to use instructor"
        ):
            self.hyde_rag._get_hypothetical_question_answer(
                question_to_ask=question, alternative_llm_client=invalid_client
            )

    # Test full pipeline for creating experiments
    def test_create_removal_experiments(self):
        # Mock embedding function to return our sample embeddings
        self.direct_exp._embed_qa_pair_list = MagicMock(
            return_value=self.sample_embeddings
        )

        # Spy on _create_single_removal_experiment to verify it's called correctly
        create_spy = MagicMock(wraps=self.direct_exp._create_single_removal_experiment)
        self.direct_exp._create_single_removal_experiment = create_spy

        results = self.direct_exp.create_removal_experiments(self.sample_qa_pairs)

        # Should create one experiment per QA pair
        assert len(results) == len(self.sample_qa_pairs)
        assert create_spy.call_count == len(self.sample_qa_pairs)

        # Each result should be a QAWithContext
        for result in results:
            assert isinstance(result, QAWithContext)

    def test_create_synthetic_experiments(self):
        # Create some synthetic questions
        synthetic_questions = [
            QAPair(
                question=f"Synthetic Q{i}?",
                answer=f"Synthetic A{i}",
            )
            for i in range(1, 4)
        ]

        # Mock embedding function
        self.basic_rag._embed_qa_pair_list = MagicMock(
            return_value=self.sample_embeddings
        )
        self.basic_rag._embed_single_qa_pair = MagicMock(
            return_value=np.array([0.5, 0.5, 0.0])
        )

        # Spy on _create_single_synthetic_experiment
        create_spy = MagicMock(wraps=self.basic_rag._create_single_synthetic_experiment)
        self.basic_rag._create_single_synthetic_experiment = create_spy

        results = self.basic_rag.create_synthetic_experiments(
            synthetic_questions=synthetic_questions,
            context_questions=self.sample_qa_pairs,
        )

        # Should create one experiment per synthetic question
        assert len(results) == len(synthetic_questions)
        assert create_spy.call_count == len(synthetic_questions)

        # Each result should be a QAWithContext
        for result in results:
            assert isinstance(result, QAWithContext)

    # Test edge cases
    def test_empty_question_list(self):
        # For removal experiments
        empty_results = self.direct_exp.create_removal_experiments([])
        assert empty_results == []

        # For synthetic experiments
        empty_synthetic = self.basic_rag.create_synthetic_experiments(
            [], self.sample_qa_pairs
        )
        assert empty_synthetic == []

        empty_context = self.long_ctx.create_synthetic_experiments(
            self.sample_qa_pairs, []
        )
        assert len(empty_context) == len(self.sample_qa_pairs)
