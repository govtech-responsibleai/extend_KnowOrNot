from . import BaseExperiment, ExperimentTypeEnum
from ..SyncLLMClient import SyncLLMClient
from ..common.models import QAPair, SingleExperimentInput
from typing import List
import numpy as np


class BasicRAG(BaseExperiment):
    def __init__(self, default_client: SyncLLMClient, closest_k: int = 5):
        super().__init__(default_client=default_client, closest_k=closest_k)

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPair,
        removed_index: int,
        remaining_qa: List[QAPair],
        embeddings: np.ndarray,
    ) -> SingleExperimentInput:
        remaining_embeddings = np.delete(embeddings, removed_index, axis=0)
        question_embedding = embeddings[removed_index]

        closest_indices = self._get_n_closest_by_cosine_similarity(
            question_embedding, remaining_embeddings
        )

        closest_questions = [remaining_qa[i] for i in closest_indices]

        return SingleExperimentInput(
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=closest_questions,
        )

    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPair,
        question_list: List[QAPair],
        embeddings: np.ndarray,
    ) -> SingleExperimentInput:
        question_embedding = self._embed_single_qa_pair(question_to_ask)

        closest_indices = self._get_n_closest_by_cosine_similarity(
            question_embedding, embeddings
        )

        closest_questions = [question_list[i] for i in closest_indices]

        return SingleExperimentInput(
            question=question_to_ask.question,
            expected_answer=None,
            context_questions=closest_questions,
        )

    @property
    def experiment_type(self):
        return ExperimentTypeEnum.BASIC_RAG
