from . import BaseRetrievalStrategy, RetrievalType, RetrievalStrategyConfig
from ..SyncLLMClient import SyncLLMClient
from typing import List
from ..common.models import QAPairFinal, QAWithContext
import numpy as np
from pydantic import BaseModel


class HypotheticalAnswers(BaseModel):
    answers: List[str]


class HydeRAGStrategy(BaseRetrievalStrategy):
    def __init__(
        self,
        config: RetrievalStrategyConfig,
        hyde_client: SyncLLMClient,
        hypothetical_answer_prompt: str,
        ai_model_for_hyde: str,
    ):
        super().__init__(config)
        self.hypothetical_answer_prompt = hypothetical_answer_prompt
        self.hyde_client = hyde_client
        self.ai_model_for_hyde = ai_model_for_hyde

        if not self.hyde_client.can_use_instructor:
            raise ValueError("Hyde client must be able to use instructor")

    @property
    def experiment_type(self):
        return RetrievalType.HYDE_RAG

    def _get_hypothetical_question_answer(
        self,
        question_to_ask: QAPairFinal,
    ) -> HypotheticalAnswers:
        """Generate hypothetical answers using the HYDE client."""
        output = self.hyde_client.get_structured_response(
            prompt=self.hypothetical_answer_prompt + str(question_to_ask),
            response_model=HypotheticalAnswers,
            ai_model=self.ai_model_for_hyde,
        )
        return output

    def _convert_to_qa_pair_list(
        self, question: str, hypothetical_answers: HypotheticalAnswers
    ) -> List[QAPairFinal]:
        """Convert hypothetical answers to QAPairFinal objects."""
        return [
            QAPairFinal(identifier="", question=question, answer=answer)
            for answer in hypothetical_answers.answers
        ]

    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPairFinal,
        removed_index: int,
        remaining_qa: List[QAPairFinal],
        embeddings: np.ndarray,
    ) -> QAWithContext:
        """Creates a HYDE removal experiment using hypothetical document embeddings."""
        # Generate hypothetical answers using HYDE client
        hypothetical_answers = self._get_hypothetical_question_answer(question_to_ask)

        # Convert to QA pairs and embed them
        hypothetical_questions = self._convert_to_qa_pair_list(
            question_to_ask.question, hypothetical_answers
        )

        # Average the hypothetical embeddings
        hypothetical_question_embeddings: np.ndarray = self._embed_qa_pair_list(
            hypothetical_questions
        ).mean(axis=0)

        # Find closest real context using hypothetical embeddings
        remaining_embeddings = np.delete(embeddings, removed_index, axis=0)
        closest_real_indices = self._get_n_closest_by_cosine_similarity(
            hypothetical_question_embeddings, remaining_embeddings
        )

        closest_questions = [remaining_qa[i] for i in closest_real_indices]

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
    ) -> QAWithContext:
        """Creates a HYDE synthetic experiment using hypothetical document embeddings."""
        # Generate hypothetical answers using HYDE client
        hypothetical_answers = self._get_hypothetical_question_answer(question_to_ask)

        # Convert to QA pairs and embed them
        hypothetical_questions = self._convert_to_qa_pair_list(
            question_to_ask.question, hypothetical_answers
        )

        # Average the hypothetical embeddings
        hypothetical_question_embeddings: np.ndarray = self._embed_qa_pair_list(
            hypothetical_questions
        ).mean(axis=0)

        # Find closest real context using hypothetical embeddings
        closest_real_indices = self._get_n_closest_by_cosine_similarity(
            hypothetical_question_embeddings, embeddings
        )

        closest_questions = [question_list[i] for i in closest_real_indices]

        return QAWithContext(
            identifier=question_to_ask.identifier,
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=closest_questions,
        )
