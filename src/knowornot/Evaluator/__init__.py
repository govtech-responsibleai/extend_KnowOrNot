from datetime import datetime
import json
from pathlib import Path
from ..SyncLLMClient import SyncLLMClient, SyncLLMClientEnum
import logging
from ..common.models import (
    ContextOptionsEnum,
    DocumentEvaluationContext,
    ExperimentOutputDocument,
    EvaluatedExperimentDocument,
    EvaluationMetadata,
    LLMResponseWithEvaluation,
    LabeledDataSample,
    Prompt,
    SavedLLMResponse,
    EvaluationSpec,
    EvaluationOutput,
    LabelTask,
)

from typing import Dict, List, Literal, Optional, Tuple, Union
import asyncio
import concurrent.futures
from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        default_client: SyncLLMClient,
        logger: logging.Logger,
        evaluation_dict: Optional[Dict[str, EvaluationSpec]],
        evaluation_model: Optional[str] = None,
    ):
        self.default_client = default_client
        self.logger = logger
        self.evaluation_dict = evaluation_dict
        self.evaluation_model = (
            evaluation_model or self.default_client.config.default_model
        )

        self.logger.info("Initializing evaluator")

    def add_evaluation_spec(self, evaluation_spec: EvaluationSpec) -> None:
        """
        Adds a new evaluation specification to the evaluator's dictionary.

        This method stores the given `EvaluationSpec` in the evaluator's internal dictionary
        for future evaluation tasks. It ensures that each evaluation specification is unique
        by checking the name of the `EvaluationSpec`. If a specification with the same name
        already exists in the dictionary, a `ValueError` is raised.

        Args:
            evaluation_spec (EvaluationSpec): The evaluation specification to add.

        Raises:
            ValueError: If an evaluation specification with the same name already exists
            in the evaluator's dictionary.
        """

        if not self.evaluation_dict:
            self.evaluation_dict = {}

        if evaluation_spec.name in self.evaluation_dict:
            raise ValueError(
                f"An evaluation spec with name {evaluation_spec.name} already exists"
            )
        self.evaluation_dict[evaluation_spec.name] = evaluation_spec

    def _create_context(
        self, evaluation_metadata: EvaluationMetadata, response: SavedLLMResponse
    ) -> str:
        """
        Construct the prompt to be given to the evaluator LLM.

        This adds the model's answer, and if requested, the question, expected answer, context, and cited QA from the original experiment input.

        If the EvaluationSpec specifies default XML prompting, this is also added to the prompt.

        :param evaluation_metadata: The metadata for the evaluation being performed
        :param response: The response to be evaluated
        :return: The constructed prompt
        """
        output = evaluation_metadata.evaluation_prompt.content

        output += f"\nThe model's answer was {response.llm_response.response}"

        if ContextOptionsEnum.QUESTION in evaluation_metadata.in_context:
            output += (
                f"\nQuestion: {response.experiment_input.source_context_qa.question}"
            )

        if ContextOptionsEnum.EXPECTED_ANSWER in evaluation_metadata.in_context:
            output += f"\n Actual Answer: {response.experiment_input.source_context_qa.expected_answer}"

        if ContextOptionsEnum.CONTEXT in evaluation_metadata.in_context:
            output += f"\nContext: {response.experiment_input.source_context_qa.context_questions}"

        if ContextOptionsEnum.CITED_QA in evaluation_metadata.in_context:
            output += f"\nCited QA: {response.cited_QA}"

        if evaluation_metadata.use_default_xml_prompting:
            output += f"""Your task is to decide what the value is for the label {evaluation_metadata.tag_name}.

            Think step by step and think out loud. Then in the end classify it according to what you think is the best output for this task.

            Give an XML tag <{evaluation_metadata.tag_name}> </{evaluation_metadata.tag_name}> with ONLY one of the following values: {", ".join(evaluation_metadata.evaluation_outcomes_list)}. This is compulsory and the answer will not parsed otherwise.
            """

        return output

    def _create_metadata_list(
        self,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        path_to_store: Path,
    ) -> List[EvaluationMetadata]:
        """
        Creates a list of EvaluationMetadata objects based on the evaluation specifications
        stored in `self.evaluation_dict` and the provided client registry.

        The method iterates through each evaluation spec in `self.evaluation_dict` and
        creates an EvaluationMetadata object using the following rules:
        - If the evaluation spec does not specify a model, the default model from the
          client registry is used.
        - If the evaluation spec does not specify a client, the default client from the
          client registry is used.
        - The evaluation name is the same as the key in `self.evaluation_dict`.
        - The evaluation prompt is the same as the evaluation prompt in the evaluation spec.
        - The tag_name is the same as the tag_name in the evaluation spec.
        - The evaluation outcomes list is the same as the evaluation outcomes list in the
          evaluation spec.
        - The in_context list is the same as the in_context list in the evaluation spec.
        - The use_default_xml_prompting flag is the same as the use_default_xml_prompting
          flag in the evaluation spec.
        - The additional_tags list is the same as the additional_tags list in the
          evaluation spec.

        The method raises a ValueError if the path does not end with .json or if the
        evaluation dictionary is empty. It also raises a ValueError if the client registry
        does not contain a client for the specified evaluation client enum.

        Args:
            client_registry: A dictionary mapping client enums to their corresponding SyncLLMClient instances.
            path_to_store: The path to the file where the evaluated experiment document will be saved.

        Returns:
            A list of EvaluationMetadata objects.
        """
        output: List[EvaluationMetadata] = []
        if not path_to_store.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path_to_store}")

        if not self.evaluation_dict:
            raise ValueError(
                "You must add at least one evaluation specfication before calling _create_metadata_list"
            )

        for evaluation_name, spec in self.evaluation_dict.items():
            used_evaluator_client_enum = (
                spec.recommended_llm_client_enum or self.default_client.enum_name
            )

            if used_evaluator_client_enum not in client_registry:
                raise ValueError(
                    f"{used_evaluator_client_enum} not in client registry. Please add a client for {used_evaluator_client_enum}"
                )

            used_evaluator_client = client_registry[used_evaluator_client_enum]

            used_model = (
                spec.recommended_llm_model or used_evaluator_client.config.default_model
            )

            metadata = EvaluationMetadata(
                evaluator_client_enum=used_evaluator_client.enum_name,
                evaluator_model=used_model,
                evaluation_name=evaluation_name,
                evaluation_prompt=spec.prompt,
                tag_name=spec.tag_name,
                evaluation_outcomes_list=spec.evaluation_outcomes,
                in_context=spec.in_context,
                use_default_xml_prompting=spec.use_default_xml_prompting,
                additional_tags=spec.additional_tags,
            )

            output.append(metadata)

        return output

    def _perform_single_evaluation(
        self,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        evaluation_kind: EvaluationMetadata,
        response: SavedLLMResponse,
        on_multiple: Literal["error", "first", "last"],
    ) -> EvaluationOutput:
        """
        Perform a single evaluation using the specified evaluation metadata and LLM response.

        This method constructs the evaluation context using the given metadata and response
        and then uses the appropriate LLM client to extract evaluation outcomes based on the
        provided tags. It handles multiple outcomes according to the specified `on_multiple`
        behavior and constructs an `EvaluationOutput` with the result.

        Args:
            client_registry (Dict[SyncLLMClientEnum, SyncLLMClient]): A dictionary mapping
                client enums to their corresponding SyncLLMClient instances.
            evaluation_kind (EvaluationMetadata): Metadata defining the evaluation criteria,
                including client, model, prompt, and expected tags.
            response (SavedLLMResponse): The response from the LLM to be evaluated.
            on_multiple (Literal["error", "first", "last"]): Defines how to handle multiple
                results for the tag name in the output. "error" raises an exception, "first" selects the first outcome, and
                "last" selects the last outcome.

        Returns:
            EvaluationOutput: An object containing the evaluation details, including the
            evaluation name, outcome, timestamp, unique ID, and any additional extracted tags.

        Raises:
            ValueError: If no evaluation outcome is found or if multiple outcomes are found
                and `on_multiple` is set to "error".
        """

        evaluator_client = client_registry[evaluation_kind.evaluator_client_enum]

        context = self._create_context(evaluation_kind, response)

        evaluation_dict = evaluator_client.prompt_and_extract_tags(
            prompt=context,
            ai_model=evaluation_kind.evaluator_model,
            validated_tags={
                evaluation_kind.tag_name: evaluation_kind.evaluation_outcomes_list
            },
            free_tags=evaluation_kind.additional_tags,
        )

        if evaluation_kind.tag_name not in evaluation_dict:
            raise ValueError(
                f"No evaluation outcome found for {evaluation_kind.tag_name} in {evaluation_kind.evaluator_model}"
            )

        if len(evaluation_dict[evaluation_kind.tag_name]) == 0:
            raise ValueError(
                f"No evaluation outcome found for {evaluation_kind.tag_name} in {evaluation_kind.evaluator_model}"
            )

        if len(evaluation_dict[evaluation_kind.tag_name]) > 1:
            if on_multiple == "error":
                raise ValueError(
                    f"Multiple evaluation outcomes found for {evaluation_kind.tag_name} in {evaluation_kind.evaluator_model}"
                )
            elif on_multiple == "first":
                evaluation_dict[evaluation_kind.tag_name] = [
                    evaluation_dict[evaluation_kind.tag_name][0]
                ]
            elif on_multiple == "last":
                evaluation_dict[evaluation_kind.tag_name] = [
                    evaluation_dict[evaluation_kind.tag_name][-1]
                ]

        evaluation_raw = evaluation_dict[evaluation_kind.tag_name][0]

        evaluation_timestamp = datetime.now()
        evaluation_timestamp_str = evaluation_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        evaluation_id = f"{evaluation_kind.evaluation_name}_{evaluation_timestamp_str}_{response.identifier}_{evaluation_kind.evaluator_model}"

        free_tags = {
            k: v for k, v in evaluation_dict.items() if k != evaluation_kind.tag_name
        }

        evaluation_output = EvaluationOutput(
            evaluation_name=evaluation_kind.evaluation_name,
            evaluation_outcome=evaluation_raw,
            evaluation_timestamp=evaluation_timestamp,
            evaluation_id=evaluation_id,
            additional_tags_info=free_tags,
        )

        return evaluation_output

    def _prepare_evaluation_context(
        self,
        document: Union[ExperimentOutputDocument, EvaluatedExperimentDocument],
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        path_to_store: Path,
    ) -> DocumentEvaluationContext:
        """
        Prepares the evaluation context from a document.

        This extracts all necessary information from the document and determines
        which evaluations need to be run.

        Args:
            document: The document to evaluate
            client_registry: Registry of available LLM clients
            path_to_store: Path where the output document will be saved

        Returns:
            DocumentEvaluationContext containing all necessary information for evaluation
        """
        new_evaluations_metadata = self._create_metadata_list(
            client_registry=client_registry, path_to_store=path_to_store
        )

        if isinstance(document, EvaluatedExperimentDocument):
            existing_eval_names = {
                meta.evaluation_name for meta in document.evaluation_metadata
            }

            new_evaluations_metadata = [
                meta
                for meta in new_evaluations_metadata
                if meta.evaluation_name not in existing_eval_names
            ]

            return DocumentEvaluationContext(
                path_to_store=path_to_store,
                experiment_metadata=document.experiment_metadata,
                existing_metadata=list(document.evaluation_metadata),
                new_evaluations_metadata=new_evaluations_metadata,
                responses=[resp.llm_response for resp in document.responses],
                existing_evaluations={
                    resp.llm_response.identifier: resp.evaluations
                    for resp in document.responses
                },
            )
        else:
            return DocumentEvaluationContext(
                path_to_store=path_to_store,
                experiment_metadata=document.metadata,
                existing_metadata=[],
                new_evaluations_metadata=new_evaluations_metadata,
                responses=document.responses,
                existing_evaluations={
                    resp.identifier: [] for resp in document.responses
                },
            )

    def evaluate_document(
        self,
        document: Union[ExperimentOutputDocument, EvaluatedExperimentDocument],
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        path_to_store: Path,
    ) -> EvaluatedExperimentDocument:
        """
        Evaluates a given experiment document and saves the results.

        This method processes an existing experiment document, performs evaluations
        based on provided specifications, and saves the annotated results as a new
        EvaluatedExperimentDocument in JSON format at the specified path.

        Args:
            document (Union[ExperimentOutputDocument, EvaluatedExperimentDocument]):
                The experiment document to evaluate. Can be either a new experiment
                output document or an already evaluated one.
            client_registry (Dict[SyncLLMClientEnum, SyncLLMClient]):
                A mapping of SyncLLMClientEnum to SyncLLMClient instances, used to
                perform evaluations with the appropriate client and model.
            path_to_store (Path):
                The file path where the evaluated experiment document will be saved.
                The path must end in a `.json` suffix.

        Returns:
            EvaluatedExperimentDocument: The document containing the results of the
            evaluations, including metadata and the evaluated responses.
        """

        context = self._prepare_evaluation_context(
            document, client_registry, path_to_store
        )

        experiment_metadata = context.experiment_metadata
        new_evaluations_metadata = context.new_evaluations_metadata
        existing_metadata = context.existing_metadata
        responses = context.responses
        existing_evaluations = context.existing_evaluations

        evaluated_llm_responses = []
        for response in responses:
            response_id = response.identifier
            evaluation_outputs = list(existing_evaluations.get(response_id, []))

            # Add new evaluations
            for evaluation_kind in new_evaluations_metadata:
                evaluation_outputs.append(
                    self._perform_single_evaluation(
                        client_registry=client_registry,
                        evaluation_kind=evaluation_kind,
                        response=response,
                        on_multiple="last",
                    )
                )

            evaluated_llm_responses.append(
                LLMResponseWithEvaluation(
                    llm_response=response, evaluations=evaluation_outputs
                )
            )

        # Create output document with combined metadata
        output = EvaluatedExperimentDocument(
            path_to_store=path_to_store,
            experiment_metadata=experiment_metadata,
            evaluation_metadata=existing_metadata + new_evaluations_metadata,
            responses=evaluated_llm_responses,
        )

        output.save_to_json()
        return output

    async def evaluate_document_async(
        self,
        document: Union[ExperimentOutputDocument, EvaluatedExperimentDocument],
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        path_to_store: Path,
        max_workers: int = 8,
    ) -> EvaluatedExperimentDocument:
        """
        Asynchronously evaluates a completed experiment using the configured evaluation kinds.

        This function creates a new EvaluatedExperimentDocument JSON file at the specified
        path_to_store, containing the experiment results annotated with all evaluation outcomes.
        The parent directory for the output file must exist or be creatable.

        The method first prepares the context for evaluation by extracting all necessary
        information from the document and determining which evaluations need to be run.
        Then, it creates tasks for all response+evaluation combinations and processes them
        concurrently with a progress bar.

        Args:
            document (Union[ExperimentOutputDocument, EvaluatedExperimentDocument]):
                The completed experiment document to evaluate.
            client_registry (Dict[SyncLLMClientEnum, SyncLLMClient]):
                A dictionary mapping SyncLLMClientEnum to SyncLLMClient instances.
            path_to_store (Path):
                The path to save the evaluated experiment document.
            max_workers (int):
                The maximum number of threads to use for concurrent evaluation. Defaults to 8.

        Returns:
            EvaluatedExperimentDocument: The evaluated experiment document.
        """
        context = self._prepare_evaluation_context(
            document, client_registry, path_to_store
        )

        experiment_metadata = context.experiment_metadata
        new_evaluations_metadata = context.new_evaluations_metadata
        existing_metadata = context.existing_metadata
        responses = context.responses
        existing_evaluations = context.existing_evaluations
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        loop = asyncio.get_event_loop()

        evaluated_llm_responses: List[Optional[LLMResponseWithEvaluation]] = [
            None
        ] * len(responses)

        async def process_evaluation(
            response_idx: int,
            response: SavedLLMResponse,
            evaluation_idx: int,
            evaluation_kind: EvaluationMetadata,
        ) -> Tuple[int, int, EvaluationOutput]:
            # Run the synchronous API call in a thread
            """
            Evaluates a single sample using the specified evaluation kind.

            This function performs up to max_tries attempts to evaluate the sample. If
            all attempts fail, an error is logged and the function raises the last
            exception encountered.

            Args:
                response_idx: The index of the response in the list of responses
                    provided in the document.
                response: The response to evaluate.
                evaluation_idx: The index of the evaluation kind in the list of
                    evaluation kinds provided in the document.
                evaluation_kind: The evaluation kind to use for evaluation.

            Returns:
                A tuple containing the response index, evaluation index, and the
                evaluation output.
            """
            max_tries = 3
            for attempt in range(max_tries):
                try:
                    evaluation_output = await loop.run_in_executor(
                        executor,
                        lambda: self._perform_single_evaluation(
                            client_registry=client_registry,
                            evaluation_kind=evaluation_kind,
                            response=response,
                            on_multiple="last",
                        ),
                    )
                    return (response_idx, evaluation_idx, evaluation_output)
                except Exception as e:
                    if attempt == max_tries - 1:
                        self.logger.error(
                            f"Failed to evaluate sample {response.identifier} after {max_tries} attempts: {e}"
                        )
                        # Return a placeholder or error indicator
                        raise e
                    else:
                        self.logger.warning(
                            f"Attempt {attempt + 1} for sample {response.identifier} failed with error {e}, retrying..."
                        )
                        await asyncio.sleep(1)  # Add a short delay before retrying

            raise ValueError("Should never get here")

        # Create tasks for all response+evaluation combinations
        tasks = []
        total_tasks = 0

        # Initialize evaluation results storage with the right structure
        evaluations_by_response = {}
        for idx, response in enumerate(responses):
            response_id = response.identifier
            # Start with existing evaluations if any
            evaluations_by_response[idx] = list(
                existing_evaluations.get(response_id, [])
            )

            # Create a placeholder for each new evaluation to be added
            initial_eval_count = len(evaluations_by_response[idx])
            evaluations_by_response[idx].extend([None] * len(new_evaluations_metadata))

            # Create tasks for each new evaluation
            for eval_idx, evaluation_kind in enumerate(new_evaluations_metadata):
                tasks.append(
                    process_evaluation(
                        idx, response, initial_eval_count + eval_idx, evaluation_kind
                    )
                )
                total_tasks += 1

        # Process all evaluations concurrently with a progress bar
        self.logger.info(f"Processing {total_tasks} evaluations")
        pbar = tqdm(total=total_tasks, desc="Processing evaluations")

        for coro in asyncio.as_completed(tasks):
            response_idx, eval_idx, evaluation_output = await coro

            evaluations_by_response[response_idx][eval_idx] = evaluation_output
            pbar.update(1)

        pbar.close()

        # Clean up
        executor.shutdown()

        # Create the final structure, maintaining both response and evaluation order
        for idx, response in enumerate(responses):
            evaluated_llm_responses[idx] = LLMResponseWithEvaluation(
                llm_response=response, evaluations=evaluations_by_response[idx]
            )

        # Verify that all evaluations are properly placed
        for resp_idx, resp in enumerate(evaluated_llm_responses):
            assert resp is not None, f"Missing response at index {resp_idx}"
            assert all(eval_result is not None for eval_result in resp.evaluations), (
                f"Missing evaluations for response at index {resp_idx}"
            )

        confirmed_llm_responses = [
            resp for resp in evaluated_llm_responses if resp is not None
        ]

        output = EvaluatedExperimentDocument(
            path_to_store=path_to_store,
            experiment_metadata=experiment_metadata,
            evaluation_metadata=existing_metadata + new_evaluations_metadata,
            responses=confirmed_llm_responses,
        )

        output.save_to_json()

        return output

    def create_evaluation_spec(
        self,
        prompt: str,
        prompt_id: str,
        label: LabelTask,
        tag_name: Optional[str] = None,
        recommended_llm_client_enum: Optional[SyncLLMClientEnum] = None,
        recommended_llm_model: Optional[str] = None,
        use_default_xml_prompting: bool = True,
        additional_tags: List[str] = [],
    ) -> EvaluationSpec:
        """
        Creates an EvaluationSpec object based on the provided parameters.

        Args:
            prompt (str): The content of the prompt to be used for the evaluation.
            prompt_id (str): The identifier for the prompt.
            label (LabelTask): The label task containing the evaluation name and possible outcomes.
            tag_name (Optional[str], optional): The specific XML tag name to use for the evaluation output. Defaults to the evaluation name.
            recommended_llm_client_enum (Optional[SyncLLMClientEnum], optional): The recommended LLM client enum for this evaluation. Defaults to None.
            recommended_llm_model (Optional[str], optional): The recommended LLM model for this evaluation. Defaults to None.
            use_default_xml_prompting (bool, optional): Whether to include XML prompting in the evaluation (or use your own custom XML prompting). Defaults to True.
            additional_tags (List[str], optional): Additional tags to add to the evaluation and extract using XML. Defaults to [].
        Returns:
            EvaluationSpec: The created EvaluationSpec object.
        """

        evaluation_name = label.name
        evaluation_prompt = Prompt(identifier=prompt_id, content=prompt)
        evaluation_outcomes = label.values

        return EvaluationSpec(
            name=evaluation_name,
            prompt=evaluation_prompt,
            tag_name=tag_name or evaluation_name,
            recommended_llm_client_enum=recommended_llm_client_enum,
            recommended_llm_model=recommended_llm_model,
            evaluation_outcomes=evaluation_outcomes,
            in_context=label.content_in_context,
            use_default_xml_prompting=use_default_xml_prompting,
            additional_tags=additional_tags,
        )

    async def evaluate_and_compare_to_human_labels_async(
        self,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        labelled_samples: List[LabeledDataSample],
        task_name: str,
        annotators_to_compare: List[str],
        prompt: str,
        prompt_id: str,
        output_path: Optional[Path] = None,
        recommended_llm_client_enum: Optional[SyncLLMClientEnum] = None,
        recommended_llm_model: Optional[str] = None,
        max_workers: int = 8,
    ) -> Dict:
        """
        Evaluates a model against human labels for a given task and comparison set of annotators.

        Args:
            client_registry (Dict[SyncLLMClientEnum, SyncLLMClient]): A dictionary mapping SyncLLMClientEnum to SyncLLMClient objects.
            labelled_samples (List[LabeledDataSample]): A list of labelled data samples.
            task_name (str): The name of the task to evaluate.
            annotators_to_compare (List[str]): A list of annotator IDs to compare the model against.
            prompt (str): The prompt to use for evaluation.
            prompt_id (str): The ID of the prompt to use for evaluation.
            output_path (Optional[Path], optional): The file path to save the evaluation results JSON file to. Defaults to None.
            recommended_llm_client_enum (Optional[SyncLLMClientEnum], optional): The recommended LLM client enum to use for evaluation. Defaults to None.
            recommended_llm_model (Optional[str], optional): The recommended LLM model to use for evaluation. Defaults to None.
            max_workers (int, optional): The maximum number of threads to use for concurrent evaluation. Defaults to 8.

        Returns:
            Dict: A dictionary containing the evaluation results and comparison statistics.
        """
        label_tasks: List[LabelTask] = []

        # Find all label tasks with that name
        # assume that every sample has the label task and every label task with that name is the same
        # works for now
        for sample in labelled_samples:
            if not sample.label_tasks:
                continue
            for task in sample.label_tasks:
                if task.name == task_name:
                    label_tasks = [task]
                    break

        if len(label_tasks) == 0:
            raise ValueError(
                f"No label tasks with name {task_name} found in the provided samples"
            )

        # Create evaluation specs
        label_task = list(label_tasks)[0]
        evaluation_spec = self.create_evaluation_spec(
            prompt=prompt,
            prompt_id=prompt_id,
            label=label_task,
            recommended_llm_client_enum=recommended_llm_client_enum,
            recommended_llm_model=recommended_llm_model,
        )

        if evaluation_spec.recommended_llm_client_enum is not None:
            evaluator_client = client_registry[
                evaluation_spec.recommended_llm_client_enum
            ]
        else:
            evaluator_client = self.default_client

        model = (
            evaluation_spec.recommended_llm_model
            or evaluator_client.config.default_model
        )

        evaluation_metadata = EvaluationMetadata(
            evaluation_name=evaluation_spec.name,
            evaluator_client_enum=evaluation_spec.recommended_llm_client_enum
            or self.default_client.enum_name,
            evaluator_model=evaluation_spec.recommended_llm_model
            or evaluator_client.config.default_model,
            evaluation_prompt=evaluation_spec.prompt,
            tag_name=evaluation_spec.tag_name,
            evaluation_outcomes_list=evaluation_spec.evaluation_outcomes,
            in_context=evaluation_spec.in_context,
            use_default_xml_prompting=evaluation_spec.use_default_xml_prompting,
            additional_tags=evaluation_spec.additional_tags,
        )

        if not self.evaluation_dict:
            self.evaluation_dict = {}

        self.evaluation_dict[label_task.name] = evaluation_spec

        # Track results for comparison
        model_evaluations = {}
        human_evaluations = {annotator: {} for annotator in annotators_to_compare}

        # Create results dictionary to store all evaluation data
        results = {
            "metadata": {
                "task_name": task_name,
                "model": model,
                "evaluation_timestamp": datetime.now().isoformat(),
                "annotators": annotators_to_compare,
                "possible_outcomes": evaluation_spec.evaluation_outcomes,
            },
            "evaluations": {
                "model": {},
                "human": {},
            },
            "agreement": {
                "model_vs_human": {},
                "inter_annotator": {},
            },
            "distribution": {
                "model": {},
                "human": {},
            },
        }

        # Set up async execution environment
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        loop = asyncio.get_event_loop()

        # Collect human evaluations first (this doesn't need async as it's just data retrieval)
        for sample in labelled_samples:
            for human_label in sample.human_labels:
                if (
                    human_label.labeller_id in annotators_to_compare
                    and human_label.label_task.name == task_name
                ):
                    human_evaluations[human_label.labeller_id][sample.sample_id] = (
                        human_label.label_value
                    )

                    if human_label.labeller_id not in results["evaluations"]["human"]:
                        results["evaluations"]["human"][human_label.labeller_id] = {}

                    results["evaluations"]["human"][human_label.labeller_id][
                        sample.sample_id
                    ] = human_label.label_value

        # Define async evaluation function
        async def evaluate_sample(sample: LabeledDataSample) -> Tuple[str, str]:
            context = self._create_context(evaluation_metadata, sample.llm_response)

            # Run the synchronous API call in a thread with retry logic
            max_tries = 3
            for attempt in range(max_tries):
                try:
                    model_eval = await loop.run_in_executor(
                        executor,
                        lambda: evaluator_client.prompt_and_extract_tag(
                            prompt=context,
                            ai_model=model,
                            tag_name=evaluation_spec.tag_name,
                            allowed_list=evaluation_spec.evaluation_outcomes,
                            on_multiple="last",
                        ),
                    )
                    return sample.sample_id, model_eval
                except Exception as e:
                    if attempt == max_tries - 1:
                        self.logger.error(
                            f"Failed to evaluate sample {sample.sample_id} after {max_tries} attempts: {e}"
                        )
                        # Return a placeholder or error indicator
                        return sample.sample_id, "evaluation_failed"
                    else:
                        self.logger.warning(
                            f"Attempt {attempt + 1} for sample {sample.sample_id} failed with error {e}, retrying..."
                        )
                        await asyncio.sleep(1)  # Add a short delay before retrying
            return sample.sample_id, "evaluation_failed"

        # Create evaluation tasks for all samples
        self.logger.info(f"Evaluating {len(labelled_samples)} samples with model")
        print(f"\nEvaluating {len(labelled_samples)} samples using {model} model...")

        # Process evaluations concurrently with a progress bar
        tasks = [evaluate_sample(sample) for sample in labelled_samples]

        # Use tqdm to display progress
        pbar = tqdm(total=len(tasks), desc="Model evaluation")

        for task in asyncio.as_completed(tasks):
            sample_id, model_eval = await task

            # Only store successful evaluations
            if model_eval != "evaluation_failed":
                model_evaluations[sample_id] = model_eval
                results["evaluations"]["model"][sample_id] = model_eval

            pbar.update(1)

        pbar.close()

        # Clean up executor
        executor.shutdown()

        # Calculate agreement statistics
        self.logger.info("Calculating agreement statistics")
        print("\n===== MODEL vs HUMAN ANNOTATOR AGREEMENT =====")

        # Calculate agreement between model and each human annotator
        for annotator in annotators_to_compare:
            matching_samples = [
                sample_id
                for sample_id in model_evaluations
                if sample_id in human_evaluations[annotator]
            ]

            if not matching_samples:
                print(f"\nNo matching samples found for annotator {annotator}")
                results["agreement"]["model_vs_human"][annotator] = {
                    "matching_samples": 0,
                    "agreement_count": 0,
                    "agreement_percentage": 0,
                    "confusion_matrix": None,
                }
                continue

            agreement_count = sum(
                1
                for sample_id in matching_samples
                if model_evaluations[sample_id]
                == human_evaluations[annotator][sample_id]
            )

            agreement_percentage = (agreement_count / len(matching_samples)) * 100

            print(
                f"\nAgreement between model and {annotator}: {agreement_percentage:.2f}% ({agreement_count}/{len(matching_samples)})"
            )

            # Store in results dict
            results["agreement"]["model_vs_human"][annotator] = {
                "matching_samples": len(matching_samples),
                "agreement_count": agreement_count,
                "agreement_percentage": agreement_percentage,
            }

            # Create confusion matrix for multi-class comparison
            if len(evaluation_spec.evaluation_outcomes) > 2:
                confusion = {}
                for outcome in evaluation_spec.evaluation_outcomes:
                    confusion[outcome] = {
                        other: 0 for other in evaluation_spec.evaluation_outcomes
                    }

                for sample_id in matching_samples:
                    model_outcome = model_evaluations[sample_id]
                    human_outcome = human_evaluations[annotator][sample_id]
                    confusion[model_outcome][human_outcome] += 1

                print(f"\nConfusion matrix for model vs {annotator}:")
                # Print header
                header = "Model \\ Human"
                for outcome in evaluation_spec.evaluation_outcomes:
                    header += f"\t{outcome}"
                print(header)

                # Print rows
                for model_outcome in evaluation_spec.evaluation_outcomes:
                    row = model_outcome
                    for human_outcome in evaluation_spec.evaluation_outcomes:
                        row += f"\t{confusion[model_outcome][human_outcome]}"
                    print(row)

                # Store confusion matrix in results
                results["agreement"]["model_vs_human"][annotator][
                    "confusion_matrix"
                ] = confusion

        # Calculate agreement among human annotators for comparison
        if len(annotators_to_compare) > 1:
            self.logger.info("Calculating inter-annotator agreement among humans")
            print("\n===== HUMAN INTER-ANNOTATOR AGREEMENT =====")

            # Find samples that all specified annotators have labeled
            common_samples = set()
            for sample_id in set().union(
                *[set(human_evaluations[a].keys()) for a in annotators_to_compare]
            ):
                if all(
                    sample_id in human_evaluations[a] for a in annotators_to_compare
                ):
                    common_samples.add(sample_id)

            results["agreement"]["inter_annotator"]["common_samples"] = list(
                common_samples
            )

            if not common_samples:
                print(
                    "\nNo samples found that were labeled by all specified annotators"
                )
                results["agreement"]["inter_annotator"]["pairwise"] = {}
                results["agreement"]["inter_annotator"]["full_agreement"] = {
                    "count": 0,
                    "percentage": 0,
                }
            else:
                print(
                    f"\nFound {len(common_samples)} samples labeled by all specified annotators"
                )

                # Calculate pairwise agreement between human annotators
                results["agreement"]["inter_annotator"]["pairwise"] = {}

                for i, annotator1 in enumerate(annotators_to_compare):
                    for annotator2 in annotators_to_compare[i + 1 :]:
                        pair_key = f"{annotator1}_vs_{annotator2}"
                        agreement_count = sum(
                            1
                            for sample_id in common_samples
                            if human_evaluations[annotator1][sample_id]
                            == human_evaluations[annotator2][sample_id]
                        )

                        agreement_percentage = (
                            agreement_count / len(common_samples)
                        ) * 100

                        print(
                            f"\nAgreement between {annotator1} and {annotator2}: {agreement_percentage:.2f}% ({agreement_count}/{len(common_samples)})"
                        )

                        # Store in results
                        results["agreement"]["inter_annotator"]["pairwise"][
                            pair_key
                        ] = {
                            "agreement_count": agreement_count,
                            "agreement_percentage": agreement_percentage,
                        }

                # Calculate overall agreement percentage across all human annotators
                full_agreement_count = sum(
                    1
                    for sample_id in common_samples
                    if len(
                        set(
                            human_evaluations[a][sample_id]
                            for a in annotators_to_compare
                        )
                    )
                    == 1
                )

                full_agreement_percentage = (
                    full_agreement_count / len(common_samples)
                ) * 100

                print(
                    f"\nFull agreement among all human annotators: {full_agreement_percentage:.2f}% ({full_agreement_count}/{len(common_samples)})"
                )

                # Store in results
                results["agreement"]["inter_annotator"]["full_agreement"] = {
                    "count": full_agreement_count,
                    "percentage": full_agreement_percentage,
                }

        # Print distribution of labels
        print("\n===== LABEL DISTRIBUTION =====")

        # Model label distribution
        model_distribution = {
            outcome: 0 for outcome in evaluation_spec.evaluation_outcomes
        }
        for label in model_evaluations.values():
            model_distribution[label] += 1

        print("\nModel label distribution:")
        for outcome, count in model_distribution.items():
            percentage = (
                (count / len(model_evaluations)) * 100 if model_evaluations else 0
            )
            print(f"{outcome}: {count} ({percentage:.2f}%)")
            results["distribution"]["model"][outcome] = {
                "count": count,
                "percentage": percentage,
            }

        # Human label distribution by annotator
        results["distribution"]["human"] = {}

        for annotator in annotators_to_compare:
            if not human_evaluations[annotator]:
                continue

            print(f"\n{annotator} label distribution:")
            human_distribution = {
                outcome: 0 for outcome in evaluation_spec.evaluation_outcomes
            }
            for label in human_evaluations[annotator].values():
                human_distribution[label] += 1

            results["distribution"]["human"][annotator] = {}

            for outcome, count in human_distribution.items():
                percentage = (
                    (count / len(human_evaluations[annotator])) * 100
                    if human_evaluations[annotator]
                    else 0
                )
                print(f"{outcome}: {count} ({percentage:.2f}%)")
                results["distribution"]["human"][annotator][outcome] = {
                    "count": count,
                    "percentage": percentage,
                }

        print("\nEvaluation and comparison completed")
        self.logger.info("Evaluation and comparison completed")

        # Save results to JSON file if path provided
        if output_path:
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)

            if not output_path.suffix == ".json":
                output_path = output_path.with_suffix(".json")

            with open(output_path, "w") as f:
                # Convert datetime objects to strings for JSON serialization
                json_results = json.dumps(
                    results,
                    default=lambda o: o.isoformat() if isinstance(o, datetime) else o,
                    indent=2,
                )
                f.write(json_results)

            print(f"Results saved to {output_path}")

        return results
