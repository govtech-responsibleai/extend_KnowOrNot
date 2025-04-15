import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
import logging

from src.knowornot.SyntheticExperimentCreator.models import CanBeAnswered
from src.knowornot.SyntheticExperimentCreator import SyntheticExperimentCreator
from src.knowornot.common.models import QAPair, AtomicFact
from src.knowornot.SyncLLMClient import SyncLLMClient


class TestSyntheticExperimentCreator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_llm_client = MagicMock(spec=SyncLLMClient)
        self.default_synthetic_prompt = "Generate a synthetic question"
        self.default_synthetic_check_prompt = "Check if this question can be answered"
        self.creator = SyntheticExperimentCreator(
            default_client=self.mock_llm_client,
            default_synthetic_prompt=self.default_synthetic_prompt,
            default_synthetic_check_prompt=self.default_synthetic_check_prompt,
            random_state=42,
            logger=MagicMock(),
        )

        self.sample_qa_pairs = [
            QAPair(
                question="Q1?",
                answer="A1",
                source=AtomicFact(fact_text="Fact 1", source_citation=0),
            ),
            QAPair(
                question="Q2?",
                answer="A2",
                source=AtomicFact(fact_text="Fact 2", source_citation=1),
            ),
            QAPair(
                question="Q3?",
                answer="A3",
                source=AtomicFact(fact_text="Fact 3", source_citation=2),
            ),
            QAPair(
                question="Q4?",
                answer="A4",
                source=AtomicFact(fact_text="Fact 4", source_citation=3),
            ),
        ]
        self.sample_embeddings = np.array(
            [
                [0.1, 0.2],
                [0.15, 0.25],  # Cluster 0
                [0.8, 0.9],
                [0.85, 0.95],  # Cluster 1
            ]
        )
        assert len(self.sample_qa_pairs) == self.sample_embeddings.shape[0]

    def test_embed_qa_pair_list(self):
        expected_texts = [
            f"Q: {qa.question} A: {qa.answer}" for qa in self.sample_qa_pairs
        ]
        self.mock_llm_client.get_embedding.return_value = (
            self.sample_embeddings.tolist()
        )
        result = self.creator._embed_qa_pair_list(self.sample_qa_pairs)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, self.sample_embeddings)
        self.mock_llm_client.get_embedding.assert_called_once_with(expected_texts)

    def test_embed_qa_pair_list_empty(self):
        result = self.creator._embed_qa_pair_list([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        self.mock_llm_client.get_embedding.assert_not_called()

    # --- Tests for _cluster_qa_pair_list ---

    # Corrected Patch Target
    @patch("src.knowornot.SyntheticExperimentCreator.KMeans")
    def test_cluster_qa_pair_list_basic(self, mock_kmeans_cls):
        """Test basic clustering with valid num_clusters"""
        n_samples = len(self.sample_qa_pairs)
        num_clusters_target = 2
        assert 1 < num_clusters_target <= n_samples

        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.labels_ = np.array([0, 0, 1, 1])
        mock_kmeans_instance.fit.return_value = mock_kmeans_instance
        # When mock_kmeans_cls() is called, return mock_kmeans_instance
        mock_kmeans_cls.return_value = mock_kmeans_instance

        clusters = self.creator._cluster_qa_pair_list(
            self.sample_qa_pairs,
            self.sample_embeddings,
            num_clusters=num_clusters_target,
        )

        # Assertions
        # Check if the class was instantiated (called)
        mock_kmeans_cls.assert_called_once_with(
            n_clusters=num_clusters_target,
            random_state=self.creator.random_state,
            n_init="auto",
        )
        # Check if the fit method was called on the instance
        mock_kmeans_instance.fit.assert_called_once_with(self.sample_embeddings)

        assert len(clusters) == num_clusters_target
        assert len(clusters[0]) == 2
        assert len(clusters[1]) == 2
        assert self.sample_qa_pairs[0] in clusters[0]
        assert self.sample_qa_pairs[1] in clusters[0]
        assert self.sample_qa_pairs[2] in clusters[1]
        assert self.sample_qa_pairs[3] in clusters[1]

    # Corrected Patch Target
    @patch("src.knowornot.SyntheticExperimentCreator.KMeans")
    def test_cluster_qa_pair_list_num_clusters_one(self, mock_kmeans_cls):
        """Test clustering with num_clusters=1"""
        clusters = self.creator._cluster_qa_pair_list(
            self.sample_qa_pairs, self.sample_embeddings, num_clusters=1
        )
        assert len(clusters) == 1
        assert clusters[0] == self.sample_qa_pairs
        mock_kmeans_cls.assert_not_called()

    # Corrected Patch Target and Corrected Regex
    @patch("src.knowornot.SyntheticExperimentCreator.KMeans")
    def test_cluster_qa_pair_list_num_clusters_zero(self, mock_kmeans_cls):
        """Test clustering with num_clusters=0 (invalid)"""
        # Use more specific regex matching the warning
        with pytest.warns(
            UserWarning, match=r"Requested num_clusters \(0\) is invalid"
        ):
            clusters = self.creator._cluster_qa_pair_list(
                self.sample_qa_pairs, self.sample_embeddings, num_clusters=0
            )
        assert len(clusters) == 1
        assert clusters[0] == self.sample_qa_pairs
        mock_kmeans_cls.assert_not_called()

    # Corrected Patch Target
    @patch("src.knowornot.SyntheticExperimentCreator.KMeans")
    def test_cluster_qa_pair_list_num_clusters_too_high(self, mock_kmeans_cls):
        """Test clustering with num_clusters > n_samples"""
        n_samples = len(self.sample_qa_pairs)
        num_clusters_target = n_samples + 2

        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.labels_ = np.arange(n_samples)
        mock_kmeans_instance.fit.return_value = mock_kmeans_instance
        mock_kmeans_cls.return_value = mock_kmeans_instance

        # Check warning using specific string match
        with pytest.warns(UserWarning) as record:
            clusters = self.creator._cluster_qa_pair_list(
                self.sample_qa_pairs,
                self.sample_embeddings,
                num_clusters=num_clusters_target,
            )
        # Assert that the specific warning was raised
        assert len(record) == 1
        assert f"Setting num_clusters to n_samples = {n_samples}" in str(
            record[0].message
        )

        mock_kmeans_cls.assert_called_once_with(
            n_clusters=n_samples,  # Called with capped value
            random_state=self.creator.random_state,
            n_init="auto",
        )
        mock_kmeans_instance.fit.assert_called_once_with(self.sample_embeddings)

        assert len(clusters) == n_samples
        for i in range(n_samples):
            assert len(clusters[i]) == 1
            assert clusters[i][0] == self.sample_qa_pairs[i]

    def test_cluster_qa_pair_list_empty_input(self):
        clusters = self.creator._cluster_qa_pair_list([], np.array([]), num_clusters=3)
        assert clusters == []

    # --- Test _check_if_question_can_be_answered ---
    def test_check_if_question_can_be_answered(self):
        question_to_check = self.sample_qa_pairs[0]
        validation_pool = self.sample_qa_pairs[1:]
        self.mock_llm_client.get_structured_response.return_value = CanBeAnswered(
            can_be_answered=True
        )
        result = self.creator._check_if_question_can_be_answered(
            question_to_check, validation_pool
        )
        assert result is True
        expected_question_str = f"Question to check:\nQ: {question_to_check.question}\nA: {question_to_check.answer}\n\n"
        expected_pool_str = "Validation Pool:\n" + "\n".join(
            [f"Q: {qa.question} A: {qa.answer}" for qa in validation_pool]
        )
        expected_full_prompt = (
            self.default_synthetic_check_prompt
            + "\n"
            + expected_question_str
            + expected_pool_str
        )
        self.mock_llm_client.get_structured_response.assert_called_once_with(
            prompt=expected_full_prompt, response_model=CanBeAnswered, ai_model=None
        )

    # --- Test _generate_synthetic_questions_for_cluster ---
    def test_generate_synthetic_questions_for_cluster(self):
        cluster = self.sample_qa_pairs[:2]
        synthetic_qa = QAPair(
            question="New Q?",
            answer="New A",
            source=AtomicFact(fact_text="New Fact", source_citation=5),
        )
        self.mock_llm_client.get_structured_response.side_effect = [
            synthetic_qa,
            CanBeAnswered(can_be_answered=False),
        ]
        result = self.creator._generate_synthetic_questions_for_cluster(cluster)
        assert len(result) == 1
        assert result[0] == synthetic_qa
        assert self.mock_llm_client.get_structured_response.call_count == 2
        cluster_str = "Existing Questions in Cluster:\n" + "\n".join(
            [f"Q: {qa.question} A: {qa.answer}" for qa in cluster]
        )
        gen_prompt_expected = (
            self.default_synthetic_prompt
            + "\n"
            + cluster_str
            + "\n\nGenerate a new, distinct question based on the themes above:"
        )
        check_question_str = f"Question to check:\nQ: {synthetic_qa.question}\nA: {synthetic_qa.answer}\n\n"
        check_pool_str = "Validation Pool:\n" + "\n".join(
            [f"Q: {qa.question} A: {qa.answer}" for qa in cluster]
        )
        check_prompt_expected = (
            self.default_synthetic_check_prompt
            + "\n"
            + check_question_str
            + check_pool_str
        )
        self.mock_llm_client.get_structured_response.assert_has_calls(
            [
                call(prompt=gen_prompt_expected, response_model=QAPair, ai_model=None),
                call(
                    prompt=check_prompt_expected,
                    response_model=CanBeAnswered,
                    ai_model=None,
                ),
            ]
        )

    def test_generate_synthetic_questions_for_cluster_needs_retry_and_reject(
        self, caplog
    ):
        cluster = self.sample_qa_pairs[:2]
        synthetic_qa_good = QAPair(
            question="Good Q?",
            answer="Good A",
            source=AtomicFact(fact_text="G Fact", source_citation=6),
        )
        synthetic_qa_bad = QAPair(
            question="Bad Q?",
            answer="Bad A",
            source=AtomicFact(fact_text="B Fact", source_citation=7),
        )

        self.mock_llm_client.get_structured_response.side_effect = [
            synthetic_qa_bad,
            CanBeAnswered(can_be_answered=True),  # Reject
            synthetic_qa_good,
            CanBeAnswered(can_be_answered=False),  # Accept
        ]

        with caplog.at_level(logging.WARNING):
            result = self.creator._generate_synthetic_questions_for_cluster(cluster)

        assert len(result) == 1
        assert result[0] == synthetic_qa_good
        assert self.mock_llm_client.get_structured_response.call_count == 4
        # Check that the correct log message was emitted
        mock_calls = [call[0][0] for call in self.creator.logger.warning.call_args_list]  # type: ignore

        assert any(
            "deemed answerable by validation pool. Discarding." in msg
            for msg in mock_calls
        )

    def test_retries_on_generation_failure(self, caplog):
        cluster = self.sample_qa_pairs[:2]
        synthetic_qa = QAPair(
            question="Retry Q?",
            answer="Retry A",
            source=AtomicFact(fact_text="R Fact", source_citation=8),
        )
        self.mock_llm_client.get_structured_response.side_effect = [
            Exception("LLM generation error"),
            synthetic_qa,
            CanBeAnswered(can_be_answered=False),
        ]
        # Check warning using specific string match
        with caplog.at_level(logging.WARNING):
            result = self.creator._generate_synthetic_questions_for_cluster(cluster)

        assert len(result) == 1
        assert result[0] == synthetic_qa
        assert self.mock_llm_client.get_structured_response.call_count == 3
        # Check that the correct log message was emitted
        mock_calls = [call[0][0] for call in self.creator.logger.error.call_args_list]  # type: ignore
        # Check that the correct warning was issued
        assert any(
            "Error generating question" in msg and "LLM generation error" in msg
            for msg in mock_calls
        )

    # --- Test generate_synthetic_dataset ---

    # Corrected Patch Targets
    @patch(
        "src.knowornot.SyntheticExperimentCreator.SyntheticExperimentCreator._generate_synthetic_questions_for_cluster"
    )
    @patch(
        "src.knowornot.SyntheticExperimentCreator.SyntheticExperimentCreator._cluster_qa_pair_list"
    )
    @patch(
        "src.knowornot.SyntheticExperimentCreator.SyntheticExperimentCreator._embed_qa_pair_list"
    )
    def test_generate_synthetic_dataset(self, mock_embed, mock_cluster, mock_generate):
        num_clusters_to_request = 2

        mock_embed.return_value = self.sample_embeddings
        mock_clusters_data = [self.sample_qa_pairs[:2], self.sample_qa_pairs[2:]]
        mock_cluster.return_value = mock_clusters_data

        mock_synth_q1 = [
            QAPair(
                question="Synth Q1",
                answer="Synth A1",
                source=AtomicFact(fact_text="S1", source_citation=10),
            )
        ]
        mock_synth_q2 = [
            QAPair(
                question="Synth Q2",
                answer="Synth A2",
                source=AtomicFact(fact_text="S2", source_citation=11),
            )
        ]
        mock_generate.side_effect = [mock_synth_q1, mock_synth_q2]

        synthetic_questions, clusters = self.creator.generate_synthetic_dataset(
            qa_pair_list=self.sample_qa_pairs, num_clusters=num_clusters_to_request
        )

        mock_embed.assert_called_once_with(self.sample_qa_pairs)
        mock_cluster.assert_called_once_with(
            self.sample_qa_pairs, self.sample_embeddings, num_clusters_to_request
        )
        assert mock_generate.call_count == len(mock_clusters_data)
        mock_generate.assert_has_calls(
            [
                call(mock_clusters_data[0], None, None, None, None, None, None, None),
                call(mock_clusters_data[1], None, None, None, None, None, None, None),
            ]
        )

        assert clusters == mock_clusters_data
        assert len(synthetic_questions) == 2
        assert synthetic_questions == mock_synth_q1 + mock_synth_q2

    # Corrected Patch Targets
    @patch(
        "src.knowornot.SyntheticExperimentCreator.SyntheticExperimentCreator._generate_synthetic_questions_for_cluster"
    )
    @patch(
        "src.knowornot.SyntheticExperimentCreator.SyntheticExperimentCreator._cluster_qa_pair_list"
    )
    @patch(
        "src.knowornot.SyntheticExperimentCreator.SyntheticExperimentCreator._embed_qa_pair_list"
    )
    def test_generate_synthetic_dataset_empty_input(
        self, mock_embed, mock_cluster, mock_generate
    ):
        synthetic_questions, clusters = self.creator.generate_synthetic_dataset(
            [], num_clusters=3
        )
        assert synthetic_questions == []
        assert clusters == []
        mock_embed.assert_not_called()
        mock_cluster.assert_not_called()
        mock_generate.assert_not_called()

    def test_generate_synthetic_dataset_invalid_num_clusters(self):
        # Test that ValueError is raised with appropriate message
        with pytest.raises(ValueError, match="Invalid num_clusters"):
            self.creator.generate_synthetic_dataset(
                self.sample_qa_pairs, num_clusters=0
            )

    # --- Test alternative client ---
    def test_alternative_client(self):
        alternative_client = MagicMock(spec=SyncLLMClient)
        expected_texts = [
            f"Q: {qa.question} A: {qa.answer}" for qa in self.sample_qa_pairs
        ]
        alternative_client.get_embedding.return_value = self.sample_embeddings.tolist()
        result = self.creator._embed_qa_pair_list(
            self.sample_qa_pairs, alternative_client=alternative_client
        )
        assert np.array_equal(result, self.sample_embeddings)
        alternative_client.get_embedding.assert_called_once_with(expected_texts)
        self.mock_llm_client.get_embedding.assert_not_called()
