from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, cast
from ..SyncLLMClient import SyncLLMClient
from ..common.models import SingleExperimentInput, QAPair
import numpy as np
import logging


class RetrievalType(Enum):
    DIRECT = "DIRECT"
    LONG_IN_CONTEXT = "LONG_IN_CONTEXT"
    BASIC_RAG = "BASIC_RAG"
    HYDE_RAG = "HYDE_RAG"


class BaseRetrievalStrategy(ABC):
    def __init__(
        self, default_client: SyncLLMClient, logger: logging.Logger, closest_k: int = 5
    ):
        if not default_client.can_use_instructor:
            raise ValueError("Default client must be able to use instructor")
        self.closest_k = closest_k
        self.default_client = default_client
        self.logger = logger

        self.logger.info(f"Initializing {self.experiment_type} experiment")

    @abstractmethod
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
        """
        An abstract method that creates a single experiment input where the question_to_ask
        is removed from the context.

        This method should be implemented by the concrete subclass of BaseRetrievalStrategy.

        Args:
            question_to_ask (QAPair): The question-answer pair to be removed and
            used for the experiment.
            removed_index (int): The index of the question_to_ask in the original list
            of question-answer pairs.
            remaining_qa (List[QAPair]): The list of question-answer pairs that are
            not removed.
            embeddings (np.ndarray): A 2D numpy array of embeddings that is
            len(question_to_ask) + len(remaining_qa) long.
            alterative_prompt (Optional[str]): An optional prompt to use for the
            experiment only in HydeRAG. Otherwise the default prompt is used in HydeRAG.
            alternative_llm_client (Optional[SyncLLMClient]): An optional LLM client to
            use for the experiment only in HydeRAG. Otherwise the default client is used in HydeRAG.
            ai_model (Optional[str]): An optional model to use for the experiment only in HydeRAG.
            Otherwise the default model from the specified client is used in HydeRAG.


        Returns:
            SingleExperimentInput: An experiment input object that represents the
            removal of question_to_ask from the original list of question-answer pairs.
        """
        pass

    def _embed_qa_pair_list(self, qa_pair_list: List[QAPair]) -> np.ndarray:
        """
        An abstract method that embeds the question list and returns a numpy 2D array.
        It should call sync client llm's get embedding method.

        Args:
            qa_pair_list (List[QAPair]): The question-answer pairs to be embedded.

        Returns:
            np.ndarray: A 2D numpy array of float embeddings.
        """
        embeddings = self.default_client.get_embedding(
            [str(qa_pair) for qa_pair in qa_pair_list]
        )
        return np.array(embeddings)

    def _get_n_closest_by_cosine_similarity(
        self, single_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[int]:
        """
        Returns the indices of the n closest embeddings to a single embedding vector.

        Computes the cosine similarity between the single embedding vector and the 2D embeddings array.
        Then it returns the indices of the n closest embeddings by sorting the cosine similarities.

        Args:
            single_embedding (np.ndarray): A 1D numpy array of float embeddings.
            embeddings (np.ndarray): A 2D numpy array of float embeddings.

        Returns:
            List[int]: A list of indices of the closest embeddings to the single embedding vector.
        """
        similarities = np.dot(embeddings, single_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(single_embedding)
        )
        sorted_indices = np.argsort(-similarities)
        return cast(List[int], sorted_indices[: self.closest_k].tolist())

    def _embed_single_qa_pair(self, qa_pair: QAPair) -> np.ndarray:
        """
        Embeds a single question-answer pair and returns a 1D numpy array.

        Args:
            qa_pair (QAPair): The question-answer pair to be embedded.

        Returns:
            np.ndarray: A 1D numpy array of float embeddings.
        """
        embedding = self.default_client.get_embedding([str(qa_pair)])
        return np.array(embedding[0])

    def _split_question_list(
        self, question_list: List[QAPair]
    ) -> List[Tuple[QAPair, List[QAPair], int]]:
        output: List[Tuple[QAPair, List[QAPair], int]] = []
        for i in range(len(question_list)):
            question_to_add = question_list[i]
            rest = question_list[0:i] + question_list[i + 1 :]
            output.append((question_to_add, rest, i))

        return output

    def create_removal_experiments(
        self,
        question_list: List[QAPair],
        alternative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> List[SingleExperimentInput]:
        """
        Creates a list of removal experiments from a list of questions.

        This method splits a list of questions into individual questions and their
        respective remaining questions. For each split, it generates a `SingleExperimentInput`
        using the `_create_single_removal_experiment` method, which represents an
        experiment based on removing the selected question from the context.

        Args:
            question_list (List[QAPair]): A list of question-answer pairs to process.

        Returns:
            List[SingleExperimentInput]: A list of experiments where each experiment
            is based on one question removed from the context.
        """
        embeddings = self._embed_qa_pair_list(question_list)
        experiment_list: List[SingleExperimentInput] = []
        split_questions = self._split_question_list(question_list)
        for question, remaining, index in split_questions:
            self.logger.info(f"Creating removal experiment {index} for {question}")
            experiment_list.append(
                self._create_single_removal_experiment(
                    question_to_ask=question,
                    removed_index=index,
                    remaining_qa=remaining,
                    embeddings=embeddings,
                    alterative_prompt=alternative_prompt,
                    alternative_llm_client=alternative_llm_client,
                    ai_model=ai_model,
                )
            )

        return experiment_list

    @abstractmethod
    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPair,
        question_list: List[QAPair],
        embeddings: np.ndarray,
        alternative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> SingleExperimentInput:
        pass

    def create_synthetic_experiments(
        self,
        synthetic_questions: List[QAPair],
        context_questions: List[QAPair],
        alternative_prompt: Optional[str] = None,
        alternative_llm_client: Optional[SyncLLMClient] = None,
        ai_model: Optional[str] = None,
    ) -> List[SingleExperimentInput]:
        embeddings = self._embed_qa_pair_list(context_questions)
        experiment_list: List[SingleExperimentInput] = []
        for idx, question in enumerate(synthetic_questions):
            self.logger.info(f"Creating synthetic experiment {idx} for {question}")
            experiment_list.append(
                self._create_single_synthetic_experiment(
                    question_to_ask=question,
                    question_list=context_questions,
                    embeddings=embeddings,
                    alternative_llm_client=alternative_llm_client,
                    alternative_prompt=alternative_prompt,
                    ai_model=ai_model,
                )
            )

        return experiment_list

    @property
    @abstractmethod
    def experiment_type(self) -> RetrievalType:
        pass
