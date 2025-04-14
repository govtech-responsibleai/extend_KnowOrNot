from . import BaseExperiment, ExperimentTypeEnum
from ..SyncLLMClient import SyncLLMClient
from ..common.models import QAPair, SingleExperimentInput
from typing import List
import numpy as np


class LongInContext(BaseExperiment):
    def __init__(self, default_client: SyncLLMClient):
        super().__init__(default_client)

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPair,
        removed_index: int,
        remaining_qa: List[QAPair],
        embeddings: np.ndarray,
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
    ) -> SingleExperimentInput:
        return SingleExperimentInput(
            question=question_to_ask.question,
            expected_answer=None,
            context_questions=question_list,
        )

    @property
    def experiment_type(self):
        return ExperimentTypeEnum.LONG_IN_CONTEXT
