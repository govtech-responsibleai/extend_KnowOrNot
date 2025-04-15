from dataclasses import dataclass
from ..RetrievalStrategy import RetrievalType
from ..SyncLLMClient import SyncLLMClientEnum
from ..common.models import QAPairIntermediate, ExperimentType
from typing import List
from pydantic import BaseModel
from datetime import datetime


@dataclass
class ExperimentInput:
    system_prompt: str
    retrieval_type: RetrievalType
    questions: List[QAPairIntermediate]
    use_batch_client: bool
    experiment_type: ExperimentType


class ExperimentMetadata(BaseModel):
    experiment_type: ExperimentType
    retrieval_type: RetrievalType
    creation_timestamp: datetime
    client: SyncLLMClientEnum
    ai_model_used: str
    user_identifier: str
    used_batch: bool


class EvaluationMetadata(BaseModel):
    evaluator_client: SyncLLMClientEnum


class Experiment(BaseModel):
    user_identifier: str
    creation_timestamp: datetime
