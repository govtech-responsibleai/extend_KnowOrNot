from datetime import datetime
from pathlib import Path
from ..SyncLLMClient import SyncLLMClient, SyncLLMClientEnum
import logging
from ..common.models import (
    ExperimentOutputDocument,
    EvaluatedExperimentDocument,
    EvaluationMetadata,
    LLMResponseWithEvaluation,
    SavedLLMResponse,
    EvaluationSpec,
    EvaluationOutput,
)

from typing import Dict, List, Optional


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

        if "question" in evaluation_metadata.in_context:
            output += (
                f"\nQuestion: {response.experiment_input.source_context_qa.question}"
            )

        if "expected_answer" in evaluation_metadata.in_context:
            output += f"\n Actual Answer: {response.experiment_input.source_context_qa.expected_answer}"

        if "context" in evaluation_metadata.in_context:
            output += f"\nContext: {response.experiment_input.source_context_qa.context_questions}"

        return output

    def evaluate_document(
        self,
        document: ExperimentOutputDocument,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        path_to_store: Path,
    ) -> EvaluatedExperimentDocument:
        if not path_to_store.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path_to_store}")
        metadata_items: List[EvaluationMetadata] = []

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
                evaluation_outcomes_enum=spec.evaluation_outcome,
            )

            metadata_items.append(metadata)

        evaluated_llm_responses = []
        for response in document.responses:
            evaluation_outputs: List[EvaluationOutput] = []
            for evaluation_kind in metadata_items:
                evaluator_client = client_registry[
                    evaluation_kind.evaluator_client_enum
                ]
                context = self._create_context(evaluation_kind, response)
                evaluation_raw = evaluator_client.prompt_for_enum(
                    prompt=context,
                    ai_model=evaluation_kind.evaluator_model,
                    tag_name=evaluation_kind.tag_name,
                    enum_class=evaluation_kind.evaluation_outcomes_enum,
                    on_multiple="last",
                )

                evaluation_timestamp = datetime.now()
                evaluation_timestamp_str = evaluation_timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                evaluation_id = f"{evaluation_kind.evaluation_name}_{evaluation_timestamp_str}_{response.identifier}_{evaluation_kind.evaluator_model}"

                evaluation_output = EvaluationOutput(
                    evaluation_name=evaluation_kind.evaluation_name,
                    evaluation_outcome=evaluation_raw,
                    evaluation_timestamp=evaluation_timestamp,
                    evaluation_id=evaluation_id,
                )

                evaluation_outputs.append(evaluation_output)

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
