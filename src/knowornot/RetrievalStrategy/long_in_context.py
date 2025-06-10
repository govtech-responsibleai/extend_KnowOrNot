from . import BaseRetrievalStrategy, RetrievalType, RetrievalStrategyConfig
from ..common.models import QAPairFinal, QAWithContext
from typing import List
import numpy as np


class LongInContextStrategy(BaseRetrievalStrategy):
    def __init__(self, config: RetrievalStrategyConfig):
        super().__init__(config)

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPairFinal,
        removed_index: int,
        remaining_qa: List[QAPairFinal],
        embeddings: np.ndarray,
    ) -> QAWithContext:
        """Creates a long-in-context removal experiment with all remaining context."""
        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=remaining_qa,
        )

    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPairFinal,
        question_list: List[QAPairFinal],
        embeddings: np.ndarray,
    ) -> QAWithContext:
        """Creates a long-in-context synthetic experiment with all available context."""
        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer="",
            context_questions=question_list,
        )

    @property
    def experiment_type(self):
        return RetrievalType.LONG_IN_CONTEXT
