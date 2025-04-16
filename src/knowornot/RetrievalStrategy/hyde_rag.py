from . import BaseRetrievalStrategy, RetrievalType
from ..SyncLLMClient import SyncLLMClient
from typing import Optional, List
from ..common.models import QAPair, QAWithContext
import numpy as np
import logging
from pydantic import BaseModel


class HypotheticalAnswers(BaseModel):
    answers: List[str]


class HydeRAGStrategy(BaseRetrievalStrategy):
    def __init__(
        self,
        default_client: SyncLLMClient,
        logger: logging.Logger,
        hypothetical_answer_prompt: str,
        closest_k: int = 5,
    ):
        super().__init__(
            default_client=default_client, logger=logger, closest_k=closest_k
        )
        self.hypothetical_answer_prompt = hypothetical_answer_prompt

    @property
    def experiment_type(self):
        return RetrievalType.HYDE_RAG

    def _get_hypothetical_question_answer(
        self,
        question_to_ask: QAPair,
        alterative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> HypotheticalAnswers:
        prompt_to_use = alterative_prompt or self.hypothetical_answer_prompt

        if alternative_llm_client and not alternative_llm_client.can_use_instructor:
            raise ValueError("Alternative client must be able to use instructor")

        client = alternative_llm_client or self.default_client

        output = client.get_structured_response(
            prompt=prompt_to_use + str(question_to_ask),
            response_model=HypotheticalAnswers,
            ai_model=ai_model,
        )

        return output

    def _convert_to_qa_pair_list(
        self, question: str, hypothetical_answers: HypotheticalAnswers
    ) -> List[QAPair]:
        return [
            QAPair(question=question, answer=answer)
            for answer in hypothetical_answers.answers
        ]

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
        hypothetical_answers = self._get_hypothetical_question_answer(
            question_to_ask=question_to_ask,
            alterative_prompt=alterative_prompt,
            alternative_llm_client=alternative_llm_client,
            ai_model=ai_model,
        )

        hypothetical_questions = self._convert_to_qa_pair_list(
            question_to_ask.question, hypothetical_answers
        )

        hypothetical_question_embeddings: np.ndarray = self._embed_qa_pair_list(
            hypothetical_questions
        ).mean(axis=0)

        remaining_embeddings = np.delete(embeddings, removed_index, axis=0)

        closest_real_indices = self._get_n_closest_by_cosine_similarity(
            hypothetical_question_embeddings, remaining_embeddings
        )

        closest_questions = [remaining_qa[i] for i in closest_real_indices]

        return QAWithContext(
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=closest_questions,
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
        hypothetical_answers = self._get_hypothetical_question_answer(
            question_to_ask=question_to_ask,
            alterative_prompt=alternative_prompt,
            alternative_llm_client=alternative_llm_client,
            ai_model=ai_model,
        )

        hypothetical_questions = self._convert_to_qa_pair_list(
            question_to_ask.question, hypothetical_answers
        )

        hypothetical_question_embeddings: np.ndarray = self._embed_qa_pair_list(
            hypothetical_questions
        ).mean(axis=0)

        closest_real_indices = self._get_n_closest_by_cosine_similarity(
            hypothetical_question_embeddings, embeddings
        )

        closest_questions = [question_list[i] for i in closest_real_indices]

        return QAWithContext(
            question=question_to_ask.question,
            expected_answer=question_to_ask.answer,
            context_questions=closest_questions,
        )
