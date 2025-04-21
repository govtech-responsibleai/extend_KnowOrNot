from enum import Enum
from ..SyncLLMClient import SyncLLMClient, SyncLLMClientEnum
import logging
from ..common.models import (
    ExperimentOutputDocument,
    EvaluatedExperimentDocument,
    EvaluationMetadata,
    SavedLLMResponse,
    EvaluationSpec,
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
        output = evaluation_metadata.evaluation_prompt  # noqa: F841

        raise NotImplementedError()

    def evaluate_document(
        self,
        document: ExperimentOutputDocument,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
    ) -> EvaluatedExperimentDocument:
        metadata_items: List[EvaluationMetadata[Enum]] = []

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
                evaluation_outcomes_enum=spec.evaluation_outcome,
            )

            metadata_items.append(metadata)

        for evaluation_kind in metadata_items:
            evaluator_client = client_registry[evaluation_kind.evaluator_client_enum]  # noqa: F841
            for response in document.responses:
                continue

        raise NotImplementedError()
