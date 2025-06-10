from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, cast
from ..SyncLLMClient import SyncLLMClient
from ..common.models import QAPairFinal, QAWithContext, RetrievalType
import numpy as np
import logging


@dataclass
class RetrievalStrategyConfig:
    embedding_client: SyncLLMClient
    embedding_model: str
    logger: logging.Logger
    closest_k: int = 5


class BaseRetrievalStrategy(ABC):
    def __init__(self, config: RetrievalStrategyConfig):
        if not config.embedding_client.config.can_use_embeddings:
            raise ValueError(
                f"Embedding client {config.embedding_client.enum_name} must be able to use embeddings"
            )

        self.closest_k = config.closest_k
        self.embedding_client = config.embedding_client
        self.embedding_model = config.embedding_model
        self.logger = config.logger

        self.logger.info(
            f"Initializing {self.experiment_type} experiment with embedding client {config.embedding_client.enum_name}"
        )

    @abstractmethod
    def _create_single_removal_experiment(
        self,
        question_to_ask: QAPairFinal,
        removed_index: int,
        remaining_qa: List[QAPairFinal],
        embeddings: np.ndarray,
    ) -> QAWithContext:
        """
        Creates a single experiment input where the question_to_ask is removed from the context.

        Args:
            question_to_ask (QAPairFinal): The question-answer pair to be removed and used for the experiment.
            removed_index (int): The index of the question_to_ask in the original list of question-answer pairs.
            remaining_qa (List[QAPairFinal]): The list of question-answer pairs that are not removed.
            embeddings (np.ndarray): A 2D numpy array of embeddings for all original questions.

        Returns:
            QAWithContext: An experiment input object representing the removal scenario.
        """
        pass

    @abstractmethod
    def _create_single_synthetic_experiment(
        self,
        question_to_ask: QAPairFinal,
        question_list: List[QAPairFinal],
        embeddings: np.ndarray,
    ) -> QAWithContext:
        """
        Creates a single experiment input for a synthetic question.

        Args:
            question_to_ask (QAPairFinal): The synthetic question to ask.
            question_list (List[QAPairFinal]): The list of available context questions.
            embeddings (np.ndarray): A 2D numpy array of embeddings for the context questions.

        Returns:
            QAWithContext: An experiment input object for the synthetic scenario.
        """
        pass

    def _embed_qa_pair_list(self, qa_pair_list: List[QAPairFinal]) -> np.ndarray:
        """
        Embeds a list of question-answer pairs.

        Args:
            qa_pair_list (List[QAPairFinal]): The question-answer pairs to be embedded.

        Returns:
            np.ndarray: A 2D numpy array of float embeddings.
        """
        embeddings = self.embedding_client.get_embedding(
            [str(qa_pair) for qa_pair in qa_pair_list], model=self.embedding_model
        )
        return np.array(embeddings)

    def _get_n_closest_by_cosine_similarity(
        self, single_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[int]:
        """
        Returns the indices of the n closest embeddings to a single embedding vector.

        Args:
            single_embedding (np.ndarray): A 1D numpy array of float embeddings.
            embeddings (np.ndarray): A 2D numpy array of float embeddings.

        Returns:
            List[int]: A list of indices of the closest embeddings.
        """
        similarities = np.dot(embeddings, single_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(single_embedding)
        )
        sorted_indices = np.argsort(-similarities)
        return cast(List[int], sorted_indices[: self.closest_k].tolist())

    def _embed_single_qa_pair(self, qa_pair: QAPairFinal) -> np.ndarray:
        """
        Embeds a single question-answer pair.

        Args:
            qa_pair (QAPairFinal): The question-answer pair to be embedded.

        Returns:
            np.ndarray: A 1D numpy array of float embeddings.
        """
        embedding = self.embedding_client.get_embedding(
            [str(qa_pair)], model=self.embedding_model
        )
        return np.array(embedding[0])

    def _split_question_list(
        self, question_list: List[QAPairFinal]
    ) -> List[Tuple[QAPairFinal, List[QAPairFinal], int]]:
        """Split question list for removal experiments."""
        output: List[Tuple[QAPairFinal, List[QAPairFinal], int]] = []
        for i in range(len(question_list)):
            question_to_add = question_list[i]
            rest = question_list[0:i] + question_list[i + 1 :]
            output.append((question_to_add, rest, i))
        return output

    def create_removal_experiments(
        self, question_list: List[QAPairFinal]
    ) -> List[QAWithContext]:
        """
        Creates a list of removal experiments from a list of questions.

        Args:
            question_list (List[QAPairFinal]): A list of question-answer pairs to process.

        Returns:
            List[QAWithContext]: A list of experiments where each experiment
            is based on one question removed from the context.
        """
        embeddings = self._embed_qa_pair_list(question_list)
        experiment_list: List[QAWithContext] = []
        split_questions = self._split_question_list(question_list)

        for question, remaining, index in split_questions:
            self.logger.info(f"Creating removal experiment {index} for {question}")
            experiment_list.append(
                self._create_single_removal_experiment(
                    question_to_ask=question,
                    removed_index=index,
                    remaining_qa=remaining,
                    embeddings=embeddings,
                )
            )
        return experiment_list

    def create_synthetic_experiments(
        self,
        synthetic_questions: List[QAPairFinal],
        context_questions: List[QAPairFinal],
    ) -> List[QAWithContext]:
        """
        Creates a list of synthetic experiments.

        Args:
            synthetic_questions (List[QAPairFinal]): Questions to ask.
            context_questions (List[QAPairFinal]): Available context questions.

        Returns:
            List[QAWithContext]: A list of synthetic experiments.
        """
        embeddings = self._embed_qa_pair_list(context_questions)
        experiment_list: List[QAWithContext] = []

        for idx, question in enumerate(synthetic_questions):
            self.logger.info(f"Creating synthetic experiment {idx} for {question}")
            experiment_list.append(
                self._create_single_synthetic_experiment(
                    question_to_ask=question,
                    question_list=context_questions,
                    embeddings=embeddings,
                )
            )
        return experiment_list

    @property
    @abstractmethod
    def experiment_type(self) -> RetrievalType:
        """Return the retrieval type for this strategy."""
        pass
