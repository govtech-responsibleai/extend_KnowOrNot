import pytest
import numpy as np
from unittest.mock import MagicMock

from src.knowornot.RetrievalStrategy import RetrievalType, RetrievalStrategyConfig
from src.knowornot.RetrievalStrategy.direct_experiment import DirectRetrievalStrategy
from src.knowornot.RetrievalStrategy.basic_rag import BasicRAGStrategy
from src.knowornot.RetrievalStrategy.long_in_context import LongInContextStrategy
from src.knowornot.RetrievalStrategy.hyde_rag import (
    HydeRAGStrategy,
    HypotheticalAnswers,
)
from src.knowornot.common.models import QAPairFinal, QAWithContext
from src.knowornot.SyncLLMClient import SyncLLMClient


class TestExperimentCategories:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create mock config objects first
        self.mock_embedding_config = MagicMock()
        self.mock_embedding_config.can_use_embeddings = True
        self.mock_embedding_config.default_embedding_model = "text-embedding-3-large"

        self.mock_hyde_config = MagicMock()
        self.mock_hyde_config.default_model = "gpt-4"

        # Create mock LLM clients with proper config
        self.mock_embedding_client = MagicMock(spec=SyncLLMClient)
        self.mock_embedding_client.config = self.mock_embedding_config
        self.mock_embedding_client.enum_name = "MOCK_EMBEDDING"

        self.mock_hyde_client = MagicMock(spec=SyncLLMClient)
        self.mock_hyde_client.config = self.mock_hyde_config
        self.mock_hyde_client.can_use_instructor = True
        self.mock_hyde_client.enum_name = "MOCK_HYDE"

        self.mock_logger = MagicMock()

        # Create config for strategies
        self.config = RetrievalStrategyConfig(
            embedding_client=self.mock_embedding_client,
            embedding_model="text-embedding-3-large",
            logger=self.mock_logger,
            closest_k=3,
        )

        # Initialize experiment classes
        self.direct_exp = DirectRetrievalStrategy(self.config)
        self.basic_rag = BasicRAGStrategy(self.config)
        self.long_ctx = LongInContextStrategy(self.config)

        # For HydeRAGStrategy, we need additional parameters
        self.hyde_prompt = "Generate a hypothetical answer for this question:"
        self.hyde_rag = HydeRAGStrategy(
            config=self.config,
            hyde_client=self.mock_hyde_client,
            hypothetical_answer_prompt=self.hyde_prompt,
            ai_model_for_hyde="gpt-4",
        )

        # Create sample QA pairs for testing
        self.sample_qa_pairs = [
            QAPairFinal(
                identifier=f"Question {i}",
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
    def test_init_with_invalid_embedding_client(self):
        # Create invalid config properly
        invalid_config_obj = MagicMock()
        invalid_config_obj.can_use_embeddings = False

        invalid_embedding_client = MagicMock(spec=SyncLLMClient)
        invalid_embedding_client.config = invalid_config_obj
        invalid_embedding_client.enum_name = "INVALID"

        invalid_config = RetrievalStrategyConfig(
            embedding_client=invalid_embedding_client,
            embedding_model="text-embedding-3-large",
            logger=self.mock_logger,
        )

        with pytest.raises(ValueError, match="must be able to use embeddings"):
            DirectRetrievalStrategy(invalid_config)

        with pytest.raises(ValueError, match="must be able to use embeddings"):
            BasicRAGStrategy(invalid_config)

        with pytest.raises(ValueError, match="must be able to use embeddings"):
            LongInContextStrategy(invalid_config)

    def test_init_hyde_with_invalid_hyde_client(self):
        invalid_hyde_client = MagicMock(spec=SyncLLMClient)
        invalid_hyde_client.can_use_instructor = False
        invalid_hyde_client.enum_name = "INVALID_HYDE"

        with pytest.raises(
            ValueError, match="Hyde client must be able to use instructor"
        ):
            HydeRAGStrategy(
                config=self.config,
                hyde_client=invalid_hyde_client,
                hypothetical_answer_prompt="prompt",
                ai_model_for_hyde="gpt-4",
            )

    # Test embedding functions
    def test_embed_qa_pair_list(self):
        qa_pairs = self.sample_qa_pairs[:2]
        expected_texts = [str(qa) for qa in qa_pairs]
        self.mock_embedding_client.get_embedding.return_value = self.sample_embeddings[
            :2
        ].tolist()

        result = self.direct_exp._embed_qa_pair_list(qa_pairs)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, self.sample_embeddings[:2])
        self.mock_embedding_client.get_embedding.assert_called_once_with(
            expected_texts, model="text-embedding-3-large"
        )

    def test_embed_single_qa_pair(self):
        qa_pair = self.sample_qa_pairs[0]
        self.mock_embedding_client.get_embedding.return_value = [
            self.sample_embeddings[0].tolist()
        ]

        result = self.direct_exp._embed_single_qa_pair(qa_pair)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, self.sample_embeddings[0])
        self.mock_embedding_client.get_embedding.assert_called_once_with(
            [str(qa_pair)], model="text-embedding-3-large"
        )

    # ... rest of the tests remain the same ...
