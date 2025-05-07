import logging
from pathlib import Path
import random
from collections import defaultdict
from typing import List, Dict, Tuple

from ..common.models import (
    HumanLabel,
    LabelTask,
    LabeledDataSample,
    ExperimentOutputDocument,
    SavedLLMResponse,
    ExperimentMetadata,
    ContextOptionsEnum,
)


class DataLabeller:
    """
    A class responsible for preparing experiment data for manual labelling.

    This includes methods for sampling data from experiment outputs based on
    specific criteria, such as stratified sampling based on experiment metadata.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initializes the DataLabeller with a logger.

        Args:
            logger: A logging.Logger instance for logging messages.
        """
        self.logger = logger

    def sample_data_stratified(
        self,
        experiments: List[ExperimentOutputDocument],
        percentage_to_sample: float,
        json_path: Path,
    ) -> List[LabeledDataSample]:
        """
        Samples data points (LLM responses with context and metadata) for labelling
        using a stratified sampling approach.

        Stratification is performed based on a combination of key experiment
        metadata fields: system prompt, retrieval type, AI model used, and
        knowledge base identifier.

        The method calculates the number of samples to take from each stratum
        proportionally based on the requested `percentage_to_sample` and the
        size of each stratum. It then samples randomly within each stratum.

        Args:
            experiments: A list of ExperimentOutputDocument objects, each containing
                         experiment metadata and a list of LLM responses.
            percentage_to_sample: The percentage of responses to sample from *each stratum*.
                                Must be a float between 0.0 and 1.0.
            json_path: The path to save the sampled data to as a JSON file.

        Returns:
            A list of LabeledDataSample objects, each representing a response
            selected for labelling along with its relevant metadata, input,
            and the stratum key it belonged to. Returns an empty list if no
            responses are found or if the percentage is invalid.
        """

        if not 0.0 <= percentage_to_sample <= 1.0:
            self.logger.error(
                f"percentage_to_sample must be between 0 and 1. Got {percentage_to_sample}"
            )
            raise ValueError(
                f"percentage_to_sample must be between 0 and 1. Got {percentage_to_sample}"
            )

        if not json_path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {json_path}")

        # Dictionary to hold responses paired with metadata, grouped by stratum key
        # Stratum key: (system_prompt_id, retrieval_type, ai_model, knowledge_base_id)
        response_metadata_pairs_by_stratum: Dict[
            Tuple[str, str, str, str], List[Tuple[ExperimentMetadata, SavedLLMResponse]]
        ] = defaultdict(list)
        all_response_metadata_pairs: List[
            Tuple[ExperimentMetadata, SavedLLMResponse]
        ] = []

        for exp_doc in experiments:
            metadata = exp_doc.metadata
            # Define the stratum key based on relevant metadata fields
            stratum_key_tuple = (
                metadata.system_prompt.identifier,
                metadata.retrieval_type.value,
                metadata.ai_model_used,
                metadata.knowledge_base_identifier,
            )
            for response in exp_doc.responses:
                response_metadata_pairs_by_stratum[stratum_key_tuple].append(
                    (metadata, response)
                )
                all_response_metadata_pairs.append((metadata, response))

        total_responses = len(all_response_metadata_pairs)
        num_strata = len(response_metadata_pairs_by_stratum)
        self.logger.info(
            f"Found {total_responses} total responses across {num_strata} strata."
        )

        if total_responses == 0:
            self.logger.warning(
                "No responses found in the provided experiments. Returning empty list."
            )
            return []

        sampled_data_samples: List[LabeledDataSample] = []
        total_sampled_count = 0

        # Sample proportionally from each stratum
        for (
            stratum_key_tuple,
            stratum_pairs,
        ) in response_metadata_pairs_by_stratum.items():
            stratum_size = len(stratum_pairs)

            # Calculate number to sample from this stratum, ensuring it's an integer
            # and does not exceed the stratum size.
            num_to_sample = round(stratum_size * percentage_to_sample)
            num_to_sample = min(num_to_sample, stratum_size)

            self.logger.debug(
                f"Stratum {stratum_key_tuple}: Population size {stratum_size}, "
                f"Target sample size {stratum_size * percentage_to_sample:.2f}, "
                f"Actual samples to take {num_to_sample}"
            )

            if num_to_sample <= 0:
                # No samples needed for this stratum based on the percentage
                continue

            # Randomly sample the required number of pairs from this stratum
            sampled_pairs_in_stratum = random.sample(stratum_pairs, num_to_sample)
            total_sampled_count += len(sampled_pairs_in_stratum)

            # Create LabeledDataSample objects for the sampled pairs
            stratum_key_str = str(stratum_key_tuple)  # Store the tuple as a string key
            for metadata, response in sampled_pairs_in_stratum:
                try:
                    labeled_sample = LabeledDataSample(
                        experiment_metadata=metadata,
                        question=response.experiment_input.source_context_qa.question,
                        expected_answer=response.experiment_input.source_context_qa.expected_answer,
                        context_questions=response.experiment_input.source_context_qa.context_questions,
                        llm_system_prompt=metadata.system_prompt,
                        llm_response=response,
                        stratum_key=stratum_key_str,
                        label_tasks=None,  # Initially no label tasks assigned
                    )
                    sampled_data_samples.append(labeled_sample)
                except Exception as e:
                    # Log error if sample creation fails for a specific response
                    self.logger.error(
                        f"Failed to create LabeledDataSample for response {response.identifier} in stratum {stratum_key_str}: {e}"
                    )

        self.logger.info(
            f"Successfully sampled {total_sampled_count} LabeledDataSamples."
        )

        LabeledDataSample.save_list_to_json(sampled_data_samples, json_path)

        return sampled_data_samples

    def _get_input_from_user(
        self,
        human_labeller_id: str,
        label_task: LabelTask,
        llm_response: SavedLLMResponse,
    ) -> HumanLabel:
        string_to_print = ""
        for key in label_task.content_in_context:
            if key == ContextOptionsEnum.QUESTION:
                string_to_print += f"Question: {llm_response.experiment_input.source_context_qa.question}\n"
            elif key == ContextOptionsEnum.EXPECTED_ANSWER:
                string_to_print += f"Expected answer: {llm_response.experiment_input.source_context_qa.expected_answer}\n"
            elif key == ContextOptionsEnum.CONTEXT:
                string_to_print += f"Context: {llm_response.experiment_input.source_context_qa.context_questions}\n"
            elif key == ContextOptionsEnum.CITED_QA:
                string_to_print += f"Cited QA: {llm_response.cited_QA}\n"

        string_to_print += f"The LLM's answer was {llm_response.llm_response.response} "

        string_to_print += (
            f"The task is to decide what the value is for the label {label_task.name} "
        )
        string_to_print += f"Your options are {label_task.values} "

        label_value = input(string_to_print)

        if label_value not in label_task.values:
            raise ValueError("Label value must be one of the values in label_task")

        return HumanLabel(
            labeller_id=human_labeller_id,
            label_task=label_task,
            label_value=label_value,
        )

    def label_samples(
        self,
        labeled_samples: List[LabeledDataSample],
        label_task: LabelTask,
    ) -> List[LabeledDataSample]:
        """
        Label a list of LabeledDataSample objects with a specific label task.

        Args:
            labeled_samples (List[LabeledDataSample]): The list of LabeledDataSample objects to label.
            label_task (LabelTask): The label task to use for labelling.

        Returns:
            List[LabeledDataSample]: The list of labeled LabeledDataSample objects.

        """

        for sample in labeled_samples:
            if not sample.label_tasks:
                sample.label_tasks = []
            sample.label_tasks.append(label_task)

        print("What do you want the human labeller id to be for this?")

        human_labeller_id = input()

        for sample in labeled_samples:
            human_label = self._get_input_from_user(
                human_labeller_id=human_labeller_id,
                label_task=label_task,
                llm_response=sample.llm_response,
            )
            sample.human_labels.append(human_label)

        return labeled_samples
