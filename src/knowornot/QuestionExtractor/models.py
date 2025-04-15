from enum import Enum
from pydantic import BaseModel
from typing import List
from ..common.models import QAPairLLM


class FilterMethod(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    BOTH = "both"


class QuestionList(BaseModel):
    questions: List[QAPairLLM]
