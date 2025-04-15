from . import BaseRetrievalExperiment, RetrievalType
from ..SyncLLMClient import SyncLLMClient
from ..common.models import QAPair, SingleExperimentInput
from typing import List, Optional
import numpy as np
import logging


class LongInContextRetrievalExperiment(BaseRetrievalExperiment):
    def __init__(
        self, default_client: SyncLLMClient, logger: logging.Logger, closest_k: int = 5
    ):
        super().__init__(
            default_client=default_client, logger=logger, closest_k=closest_k
        )

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPair,
        removed_index: int,
        remaining_qa: List[QAPair],
        embeddings: np.ndarray,
        alterative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> SingleExperimentInput:
        return SingleExperimentInput(
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=remaining_qa,
        )

    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPair,
        question_list: List[QAPair],
        embeddings: np.ndarray,
        alternative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> SingleExperimentInput:
        return SingleExperimentInput(
            question=question_to_ask.question,
            expected_answer=None,
            context_questions=question_list,
        )

    @property
    def experiment_type(self):
        return RetrievalType.LONG_IN_CONTEXT
