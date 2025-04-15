from dataclasses import dataclass
from ..ExperimentCategories import RetrievalType
from ..common.models import QAPair
from typing import List


@dataclass
class ExperimentParams:
    experiment_type: RetrievalType
    questions: List[QAPair]
