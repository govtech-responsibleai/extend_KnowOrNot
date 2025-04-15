from dataclasses import dataclass
from ..RetrievalStrategy import RetrievalType
from ..SyncLLMClient import SyncLLMClientEnum
from ..common.models import QAPairIntermediate, ExperimentType
from typing import List
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path


@dataclass
class ExperimentInput:
    system_prompt: str
    retrieval_type: RetrievalType
    questions: List[QAPairIntermediate]
    use_batch_client: bool
    experiment_type: ExperimentType


class QAPairToLLM(QAPairIntermediate):
    identifier: str
    index: int


class Prompt(BaseModel):
    system_prompt: str
    identifier: str
    creation_timestamp: datetime
    save_location: Path

    def save_to_json(self):
        data = self.model_dump_json()
        with open(self.save_location, "w") as f:
            f.write(data)
        return


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
