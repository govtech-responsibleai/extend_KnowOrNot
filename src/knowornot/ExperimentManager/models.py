from dataclasses import dataclass
from ..RetrievalStrategy import RetrievalType
from ..SyncLLMClient import SyncLLMClientEnum, SyncLLMClient
from ..common.models import QAPair, ExperimentType, Prompt
from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path


@dataclass
class ExperimentParams:
    system_prompt: Prompt
    retrieval_type: RetrievalType
    input_path: Path
    output_path: Path
    questions: List[QAPair]
    llm_client_enum: SyncLLMClientEnum
    ai_model_for_experiment: str
    knowledge_base_identifier: str
    experiment_type: ExperimentType
    alternative_prompt_for_hyde: Optional[Prompt]
    alternative_llm_client_for_hyde: Optional[SyncLLMClient]
    ai_model_for_hyde: Optional[str]


class EvaluationMetadata(BaseModel):
    evaluator_client: SyncLLMClientEnum
