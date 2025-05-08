from datetime import datetime
from pathlib import Path
from ..SyncLLMClient import SyncLLMClient, SyncLLMClientEnum
import logging
from ..common.models import (
    ContextOptionsEnum,
    ExperimentOutputDocument,
    EvaluatedExperimentDocument,
    EvaluationMetadata,
    LLMResponseWithEvaluation,
    Prompt,
    SavedLLMResponse,
    EvaluationSpec,
    EvaluationOutput,
    LabelTask,
)

from typing import Dict, List, Optional, Tuple
import asyncio
import concurrent.futures
from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        default_client: SyncLLMClient,
        logger: logging.Logger,
        evaluation_dict: Dict[str, EvaluationSpec],
        evaluation_model: Optional[str] = None,
    ):
        self.default_client = default_client
        self.logger = logger
        self.evaluation_dict = evaluation_dict
        self.evaluation_model = (
            evaluation_model or self.default_client.config.default_model
        )

        self.logger.info("Initializing evaluator")

    def _create_context(
        self, evaluation_metadata: EvaluationMetadata, response: SavedLLMResponse
    ) -> str:
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
        output: List[EvaluationMetadata] = []
        if not path_to_store.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path_to_store}")

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
            )

            output.append(metadata)

        return output

    def _create_single_evaluation_output(
        self,
        evaluation_raw: str,
        evaluation_kind: EvaluationMetadata,
        response: SavedLLMResponse,
    ) -> EvaluationOutput:
        evaluation_timestamp = datetime.now()
        evaluation_timestamp_str = evaluation_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        evaluation_id = f"{evaluation_kind.evaluation_name}_{evaluation_timestamp_str}_{response.identifier}_{evaluation_kind.evaluator_model}"

        evaluation_output = EvaluationOutput(
            evaluation_name=evaluation_kind.evaluation_name,
            evaluation_outcome=evaluation_raw,
            evaluation_timestamp=evaluation_timestamp,
            evaluation_id=evaluation_id,
        )

        return evaluation_output

    def evaluate_document(
        self,
        document: ExperimentOutputDocument,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        path_to_store: Path,
    ) -> EvaluatedExperimentDocument:
        metadata_items = self._create_metadata_list(
            client_registry=client_registry, path_to_store=path_to_store
        )

        evaluated_llm_responses = []
        for response in document.responses:
            evaluation_outputs: List[EvaluationOutput] = []
            for evaluation_kind in metadata_items:
                evaluator_client = client_registry[
                    evaluation_kind.evaluator_client_enum
                ]
                context = self._create_context(evaluation_kind, response)
                evaluation_raw = evaluator_client.prompt_and_extract_tag(
                    prompt=context,
                    ai_model=evaluation_kind.evaluator_model,
                    tag_name=evaluation_kind.tag_name,
                    allowed_list=evaluation_kind.evaluation_outcomes_list,
                    on_multiple="last",
                )

                evaluation_outputs.append(
                    self._create_single_evaluation_output(
                        evaluation_raw=evaluation_raw,
                        evaluation_kind=evaluation_kind,
                        response=response,
                    )
                )

            evaluated_llm_responses.append(
                LLMResponseWithEvaluation(
                    llm_response=response, evaluations=evaluation_outputs
                )
            )

        output = EvaluatedExperimentDocument(
            path_to_store=path_to_store,
            experiment_metadata=document.metadata,
            evaluation_metadata=metadata_items,
            responses=evaluated_llm_responses,
        )

        output.save_to_json()

        return output

    async def evaluate_document_async(
        self,
        document: ExperimentOutputDocument,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        path_to_store: Path,
        max_workers: int = 8,
    ) -> EvaluatedExperimentDocument:
        metadata_items = self._create_metadata_list(
            client_registry=client_registry, path_to_store=path_to_store
        )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        loop = asyncio.get_event_loop()

        evaluated_llm_responses: List[Optional[LLMResponseWithEvaluation]] = [
            None
        ] * len(document.responses)

        async def process_evaluation(
            response_idx: int,
            response: SavedLLMResponse,
            evaluation_idx: int,
            evaluation_kind: EvaluationMetadata,
        ) -> Tuple[int, int, SavedLLMResponse, EvaluationMetadata, str]:
            evaluator_client = client_registry[evaluation_kind.evaluator_client_enum]
            context = self._create_context(evaluation_kind, response)

            # Run the synchronous API call in a thread
            evaluation_raw = await loop.run_in_executor(
                executor,
                lambda: evaluator_client.prompt_and_extract_tag(
                    prompt=context,
                    ai_model=evaluation_kind.evaluator_model,
                    tag_name=evaluation_kind.tag_name,
                    allowed_list=evaluation_kind.evaluation_outcomes_list,
                    on_multiple="last",
                ),
            )

            return (
                response_idx,
                evaluation_idx,
                response,
                evaluation_kind,
                evaluation_raw,
            )

        # Create tasks for all response+evaluation combinations
        tasks = []
        total_tasks = 0

        # Initialize evaluation results storage with the right structure
        evaluations_by_response = {}
        for idx, response in enumerate(document.responses):
            # For each response, create a list with None placeholders for each evaluation
            evaluations_by_response[idx] = [None] * len(metadata_items)

            # Create tasks for each evaluation
            for eval_idx, evaluation_kind in enumerate(metadata_items):
                tasks.append(
                    process_evaluation(idx, response, eval_idx, evaluation_kind)
                )
                total_tasks += 1

        # Process all evaluations concurrently with a progress bar
        self.logger.info(f"Processing {total_tasks} evaluations")
        pbar = tqdm(total=total_tasks, desc="Processing evaluations")

        for coro in asyncio.as_completed(tasks):
            (
                response_idx,
                eval_idx,
                response,
                evaluation_kind,
                evaluation_raw,
            ) = await coro

            evaluation_output = self._create_single_evaluation_output(
                evaluation_raw=evaluation_raw,
                evaluation_kind=evaluation_kind,
                response=response,
            )

            # Store the evaluation at its correct position in the response's evaluation list
            evaluations_by_response[response_idx][eval_idx] = evaluation_output
            pbar.update(1)

        pbar.close()

        # Clean up
        executor.shutdown()

        # Create the final structure, maintaining both response and evaluation order
        for idx, response in enumerate(document.responses):
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
        for resp_idx, resp in enumerate(confirmed_llm_responses):
            assert all(eval_result is not None for eval_result in resp.evaluations), (
                f"Missing evaluations for response at index {resp_idx}"
            )

        output = EvaluatedExperimentDocument(
            path_to_store=path_to_store,
            experiment_metadata=document.metadata,
            evaluation_metadata=metadata_items,
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
        )
