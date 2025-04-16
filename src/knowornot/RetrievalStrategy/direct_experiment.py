import numpy as np

from . import BaseRetrievalStrategy, RetrievalType
from ..SyncLLMClient import SyncLLMClient
from typing import List, Optional
from ..common.models import QAPair, QAWithContext
import logging


class DirectRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(
        self, default_client: SyncLLMClient, logger: logging.Logger, closest_k: int = 5
    ):
        super().__init__(
            default_client=default_client, logger=logger, closest_k=closest_k
        )

    @property
    def experiment_type(self):
        return RetrievalType.DIRECT

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPair,
        removed_index: int,
        remaining_qa: List[QAPair],
        embeddings: np.ndarray,
        alterative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> QAWithContext:
        return QAWithContext(
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=None,
        )

    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPair,
        question_list: List[QAPair],
        embeddings: np.ndarray,
        alternative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> QAWithContext:
        return QAWithContext(
            question=question_to_ask.question,
            expected_answer="",
            context_questions=None,
        )
