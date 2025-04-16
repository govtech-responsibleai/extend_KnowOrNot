from dataclasses import dataclass
from ..RetrievalStrategy import RetrievalType
from ..SyncLLMClient import SyncLLMClientEnum, SyncLLMClient
from ..common.models import QAPair, ExperimentType, Prompt
from typing import List, Optional
from pydantic import BaseModel


@dataclass
class ExperimentParams:
    system_prompt: Prompt
    retrieval_type: RetrievalType
    questions: List[QAPair]
    llm_client_enum: SyncLLMClientEnum
    ai_model_for_experiment: str
    knowledge_base_identifier: str
    experiment_type: ExperimentType
    alternative_prompt_for_hyde: Optional[Prompt] = None
    alternative_llm_client_for_hyde: Optional[SyncLLMClient] = None
    ai_model_for_hyde: Optional[str] = None


class EvaluationMetadata(BaseModel):
    evaluator_client: SyncLLMClientEnum
