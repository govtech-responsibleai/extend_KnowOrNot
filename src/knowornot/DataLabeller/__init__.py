import logging
from pathlib import Path
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np

from ..common.models import (
    HumanLabel,
    InterAnnotatorAgreement,
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

        string_to_print += (
            f"The LLM's answer was: \n {llm_response.llm_response.response} \n"
        )

        string_to_print += f"The task is to decide what the value is for the label {label_task.name} \n"
        string_to_print += "Your options are:\n"
        for i, value in enumerate(label_task.values, start=1):
            string_to_print += f"{i}. {value}\n"

        while True:
            user_input = input(string_to_print)
            try:
                value_number = int(user_input)
                if value_number >= 1 and value_number <= len(label_task.values):
                    label_value = label_task.values[value_number - 1]
                    break

            except Exception:
                pass

            print(
                f"ERROR: Please enter a number between 1 and {len(label_task.values)}"
            )

        return HumanLabel(
            labeller_id=human_labeller_id,
            label_task=label_task,
            label_value=label_value,
        )

    def label_samples(
        self,
        labeled_samples: List[LabeledDataSample],
        label_task: LabelTask,
        path_to_save: Path,
    ) -> List[LabeledDataSample]:
        """
        Label a list of LabeledDataSample objects with a specific label task.

        Args:
            labeled_samples (List[LabeledDataSample]): The list of LabeledDataSample objects to label.
            label_task (LabelTask): The label task to use for labelling.

        Returns:
            List[LabeledDataSample]: The list of labeled LabeledDataSample objects.

        """

        if path_to_save.suffix != ".json":
            raise ValueError(
                f"Expected path_to_save to be a .json file, but got {path_to_save}"
            )

        for sample in labeled_samples:
            if not sample.label_tasks:
                sample.label_tasks = []
            sample.label_tasks.append(label_task)

        print("What do you want the human labeller id to be for this?")

        human_labeller_id = input()

        for index, sample in enumerate(labeled_samples):
            human_label = self._get_input_from_user(
                human_labeller_id=human_labeller_id,
                label_task=label_task,
                llm_response=sample.llm_response,
            )
            sample.human_labels.append(human_label)
            print(f"Done {index + 1}/{len(labeled_samples)}")
            print(f"Remaining: {len(labeled_samples) - index - 1}")
            LabeledDataSample.save_list_to_json(labeled_samples, path_to_save)

        return labeled_samples

    def calculate_inter_annotator_agreement(
        self, labeled_samples: List[LabeledDataSample], task_name: str
    ) -> InterAnnotatorAgreement:
        """
        Calculate inter-annotator agreement for multiple annotators on a specific labeling task.

        Args:
            labeled_samples: List of LabeledDataSample objects
            task_name: Name of the label task to analyze

        Returns:
            InterAnnotatorAgreement object containing kappa statistics and information about the common dataset

        Raises:
            ValueError: If no annotators are found for the specified task, no common samples exist,
                        or the task is not found in any sample
        """
        # Step 1: Create a mapping from sample_id to sample object for easy lookup

        # Step 2: Organize data by annotator and identify which samples each has labeled
        annotator_samples: Dict[
            str, Dict[str, str]
        ] = {}  # {annotator_id: {sample_id: label_value}}

        for sample in labeled_samples:
            for human_label in sample.human_labels:
                if human_label.label_task.name == task_name:
                    annotator_id = human_label.labeller_id

                    if annotator_id not in annotator_samples:
                        annotator_samples[annotator_id] = {}

                    annotator_samples[annotator_id][sample.sample_id] = (
                        human_label.label_value
                    )

        # Step 3: Find the intersection of samples labeled by all annotators
        annotators = list(annotator_samples.keys())
        if not annotators:
            raise ValueError(f"No annotators found for task '{task_name}'")

        # Get sets of sample_ids for each annotator
        annotator_sample_sets: Dict[str, Set[str]] = {
            annotator: set(samples.keys())
            for annotator, samples in annotator_samples.items()
        }

        # Find the intersection (samples that all annotators have labeled)
        common_samples = set.intersection(*annotator_sample_sets.values())

        if not common_samples:
            raise ValueError(
                f"No common samples found across all annotators for task '{task_name}'"
            )

        # Step 4: Get all possible label values for this task and create a mapping
        # Find the label task definition
        label_task = None
        for sample in labeled_samples:
            if sample.label_tasks:
                for task in sample.label_tasks:
                    if task.name == task_name:
                        label_task = task
                        break
            if label_task:
                break

        if not label_task:
            raise ValueError(f"Task '{task_name}' not found in any sample")

        label_values = label_task.values
        label_to_idx = {value: idx for idx, value in enumerate(label_values)}

        # Step 5: Create the rating matrix for Fleiss' Kappa
        n_samples = len(common_samples)
        n_categories = len(label_values)

        # For Fleiss' kappa, we need a matrix where:
        # - Each row represents one item (sample)
        # - Each column represents one category (label value)
        # - Each cell contains the number of raters who assigned that category to that item
        rating_matrix = np.zeros((n_samples, n_categories))

        # Also track raw ratings for potential pairwise Cohen's Kappa
        raw_ratings = {annotator: [] for annotator in annotators}
        common_sample_list = sorted(common_samples)  # Ensure consistent order

        for i, sample_id in enumerate(common_sample_list):
            for annotator in annotators:
                label = annotator_samples[annotator][sample_id]
                label_idx = label_to_idx[label]
                rating_matrix[i, label_idx] += 1
                raw_ratings[annotator].append(label)

        # Step 6: Calculate Fleiss' Kappa for multiple raters
        kappa_value = fleiss_kappa(rating_matrix)

        # Step 7: Calculate pairwise Cohen's Kappa for all pairs
        pairwise_kappa = {}
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                annotator1 = annotators[i]
                annotator2 = annotators[j]

                ratings1 = raw_ratings[annotator1]
                ratings2 = raw_ratings[annotator2]

                # Calculate Cohen's Kappa
                cohens_kappa_value = self._compute_cohens_kappa(ratings1, ratings2)

                pairwise_kappa[(annotator1, annotator2)] = cohens_kappa_value

        # Step 8: Create and return the InterAnnotatorAgreement object
        return InterAnnotatorAgreement(
            task_name=task_name,
            fleiss_kappa=kappa_value,
            pairwise_cohens_kappa=pairwise_kappa,
            common_samples=len(common_samples),
            sample_ids=common_sample_list,
            annotators=annotators,
            possible_labels=label_values,
        )

    def _compute_cohens_kappa(self, ratings1, ratings2):
        """
        Compute Cohen's Kappa for two lists of ratings.

        Args:
            ratings1: List of ratings from annotator 1
            ratings2: List of ratings from annotator 2

        Returns:
            Cohen's Kappa coefficient
        """
        # Get unique categories
        all_categories = sorted(set(ratings1 + ratings2))

        # Create confusion matrix
        n_categories = len(all_categories)
        confusion_matrix = np.zeros((n_categories, n_categories))

        # Map categories to indices
        category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}

        # Fill confusion matrix
        n_items = len(ratings1)
        for i in range(n_items):
            confusion_matrix[
                category_to_idx[ratings1[i]], category_to_idx[ratings2[i]]
            ] += 1

        # Calculate observed agreement
        observed_agreement = (
            sum(confusion_matrix[i, i] for i in range(n_categories)) / n_items
        )

        # Calculate expected agreement by chance
        row_sums = confusion_matrix.sum(axis=1) / n_items
        col_sums = confusion_matrix.sum(axis=0) / n_items
        expected_agreement = sum(row_sums[i] * col_sums[i] for i in range(n_categories))

        # Calculate kappa
        if expected_agreement == 1:  # Avoid division by zero
            return 1.0
        else:
            kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
            return kappa
