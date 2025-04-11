from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Tuple
from ..SyncLLMClient import SyncLLMClient
from ..common.models import SingleExperimentInput, QAPair


class ExperimentTypeEnum(Enum):
    DIRECT = "DIRECT"
    LONG_IN_CONTEXT = "LONG_IN_CONTEXT"
    BASIC_RAG = "BASIC_RAG"
    HYDE_RAG = "HYDE_RAG"


class BaseExperiment(ABC):
    def __init__(self, default_client: SyncLLMClient):
        self.default_client = default_client

    @abstractmethod
    def _create_single_removal_experiment(
        self, question_to_ask: QAPair, remaining_qa: List[QAPair]
    ) -> SingleExperimentInput:
        """
        An abstract method that creates a single experiment input where the question_to_ask
        is removed from the remaining_qa context.

        This method should be implemented by the concrete subclass of BaseExperiment.

        Args:
            question_to_ask (QAPair): The question-answer pair to be removed and
            used for the experiment.
            remaining_qa (List[QAPair]): The list of remaining question-answer pairs
            after removing the question_to_ask.

        Returns:
            SingleExperimentInput: An experiment input object that represents the
            removal of question_to_ask from the remaining_qa context.
        """

        pass

    @abstractmethod
    def _create_single_synthetic_experiment(
        self, question_to_ask: QAPair, remaining_qa: List[QAPair]
    ) -> SingleExperimentInput:
        """
        An abstract method that creates a single experiment input where the question_to_ask is synthetically generated from the remaining_qa context.

        This method should be implemented by the concrete subclass of BaseExperiment.
        The user should refer to the specific concrete implementation for the details of how the synthetic experiment is created.

        Args:
            question_to_ask (QAPair): The question to ask in the experiment.
            remaining_qa (List[QAPair]): The remaining question-answer pairs that are used to generate the synthetic input.

        Returns:
            SingleExperimentInput: A single experiment input.
        """
        pass

    def _split_question_list(
        self, question_list: List[QAPair]
    ) -> List[Tuple[QAPair, List[QAPair]]]:
        output: List[Tuple[QAPair, List[QAPair]]] = []
        for i in range(len(question_list)):
            question_to_add = question_list[i]
            rest = question_list[0:i] + question_list[i + 1 :]
            output.append((question_to_add, rest))

        return output

    def create_removal_experiment(
        self, question_list: List[QAPair]
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

        experiment_list: List[SingleExperimentInput] = []
        split_questions = self._split_question_list(question_list)
        for question, remaining in split_questions:
            experiment_list.append(
                self._create_single_removal_experiment(
                    question_to_ask=question, remaining_qa=remaining
                )
            )

        return experiment_list

    def create_synthetic_experiment(
        self, question_list: List[QAPair]
    ) -> List[SingleExperimentInput]:
        """
        Creates a list of synthetic experiments from a list of questions.

        This method splits a list of questions into individual questions and their
        respective remaining questions. For each split, it generates a `SingleExperimentInput`
        using the `_create_single_synthetic_experiment` method, which represents an
        experiment based on synthetically generating the selected question based on the
        context.

        Args:
            question_list (List[QAPair]): A list of question-answer pairs to process.

        Returns:
            List[SingleExperimentInput]: A list of experiments where each experiment
            is based on one question synthetically generated from the context.
        """

        experiment_list: List[SingleExperimentInput] = []
        split_questions = self._split_question_list(question_list)
        for question, remaining in split_questions:
            experiment_list.append(
                self._create_single_synthetic_experiment(
                    question_to_ask=question, remaining_qa=remaining
                )
            )

        return experiment_list

    @property
    @abstractmethod
    def experiment_type(self) -> ExperimentTypeEnum:
        pass
