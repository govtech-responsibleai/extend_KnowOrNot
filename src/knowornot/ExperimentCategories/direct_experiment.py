import numpy as np
from . import BaseExperiment, ExperimentTypeEnum
from ..SyncLLMClient import SyncLLMClient
from typing import List
from ..common.models import QAPair, SingleExperimentInput


class DirectExperiment(BaseExperiment):
    def __init__(self, default_client: SyncLLMClient):
        super().__init__(default_client)

    @property
    def experiment_type(self):
        return ExperimentTypeEnum.DIRECT

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
            context_questions=None,
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
            context_questions=None,
        )

    def _generate_synthetic_questions(
        self, question_list: List[QAPair]
    ) -> List[QAPair]:
        raise NotImplementedError("")
