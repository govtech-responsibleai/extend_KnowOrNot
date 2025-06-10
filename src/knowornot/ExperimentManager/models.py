from dataclasses import dataclass
from ..RetrievalStrategy import RetrievalType
from ..SyncLLMClient import SyncLLMClientEnum, SyncLLMClient
from ..common.models import ExperimentType, Prompt, QAPairFinal
from typing import List
from pathlib import Path


@dataclass
class ExperimentParams:
    system_prompt: Prompt
    retrieval_type: RetrievalType
    input_path: Path
    output_path: Path
    questions: List[QAPairFinal]
    llm_client_enum_experiment: SyncLLMClientEnum  # For generation
    ai_model_for_experiment: str
    knowledge_base_identifier: str
    experiment_type: ExperimentType
    embedding_client: SyncLLMClient  # For embeddings
    embedding_model: str  # For embeddings
    hyde_prompt: Prompt
    hyde_client: SyncLLMClient  # For HYDE
    ai_model_for_hyde: str
