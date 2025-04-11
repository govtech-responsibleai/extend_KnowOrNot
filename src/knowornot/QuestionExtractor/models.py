from pydantic import BaseModel
from ..FactManager.models import AtomicFact


class QAPairLLM(BaseModel):
    question: str
    answer: str

    def __str__(self):
        return f"Question: {self.question} \n Answer: {self.answer}"


class QAPair(QAPairLLM):
    source: AtomicFact
