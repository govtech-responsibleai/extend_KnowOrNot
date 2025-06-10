from . import BaseRetrievalStrategy, RetrievalType, RetrievalStrategyConfig
from ..common.models import QAPairFinal, QAWithContext
from typing import List
import numpy as np


class DirectRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, config: RetrievalStrategyConfig):
        super().__init__(config)

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPairFinal,
        removed_index: int,
        remaining_qa: List[QAPairFinal],
        embeddings: np.ndarray,
    ) -> QAWithContext:
        """Creates a direct removal experiment with no context."""
        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=None,
        )

    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPairFinal,
        question_list: List[QAPairFinal],
        embeddings: np.ndarray,
    ) -> QAWithContext:
        """Creates a direct synthetic experiment with no context."""
        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer="",
            context_questions=None,
        )

    @property
    def experiment_type(self):
        return RetrievalType.DIRECT
