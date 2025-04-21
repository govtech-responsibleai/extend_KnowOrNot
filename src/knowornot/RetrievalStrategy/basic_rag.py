from . import BaseRetrievalStrategy, RetrievalType
from ..SyncLLMClient import SyncLLMClient
from ..common.models import QAPairFinal, QAWithContext
from typing import List, Optional
import numpy as np
import logging


class BasicRAGStrategy(BaseRetrievalStrategy):
    def __init__(
        self, default_client: SyncLLMClient, logger: logging.Logger, closest_k: int = 5
    ):
        super().__init__(
            default_client=default_client, logger=logger, closest_k=closest_k
        )

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPairFinal,
        removed_index: int,
        remaining_qa: List[QAPairFinal],
        embeddings: np.ndarray,
        alterative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> QAWithContext:
        remaining_embeddings = np.delete(embeddings, removed_index, axis=0)
        question_embedding = embeddings[removed_index]

        closest_indices = self._get_n_closest_by_cosine_similarity(
            question_embedding, remaining_embeddings
        )

        closest_questions = [remaining_qa[i] for i in closest_indices]

        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=closest_questions,
        )

    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPairFinal,
        question_list: List[QAPairFinal],
        embeddings: np.ndarray,
        alternative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> QAWithContext:
        question_embedding = self._embed_single_qa_pair(question_to_ask)

        closest_indices = self._get_n_closest_by_cosine_similarity(
            question_embedding, embeddings
        )

        closest_questions = [question_list[i] for i in closest_indices]

        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer="",
            context_questions=closest_questions,
        )

    @property
    def experiment_type(self):
        return RetrievalType.BASIC_RAG
