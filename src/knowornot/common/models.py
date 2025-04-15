from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional


class AtomicFact(BaseModel):
    fact_text: str
    source_citation: int


class Sentence(BaseModel):
    text: str


class SplitSourceDocument(BaseModel):
    sentences: List[Sentence]

    def __str__(self) -> str:
        result = "Source document: \n"
        for i, sentence in enumerate(self.sentences):
            result += f"{i}: {sentence.text}\n"
        return result.strip()


class AtomicFactDocument(BaseModel):
    fact_list: List[AtomicFact]

    def __str__(self) -> str:
        result = "Atomic facts document: \n"
        for i, fact in enumerate(self.fact_list):
            result += f"{i}: {fact.fact_text}\n"
        return result.strip()

    def save_to_json(self, path: Path) -> None:
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @staticmethod
    def load_from_json(path: Path) -> "AtomicFactDocument":
        """
        Read an AtomicFactDocument from a JSON file at the specified path.

        Args:
            path (Path): Path to the JSON file

        Returns:
            AtomicFactDocument: The loaded document

        Raises:
            ValueError: If the path doesn't end with .json
            FileNotFoundError: If the file doesn't exist
        """
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "r") as f:
            text = f.read()

        return AtomicFactDocument.model_validate_json(text)


class QAPairLLM(BaseModel):
    question: str
    answer: str

    def __str__(self):
        return f"Question: {self.question} \n Answer: {self.answer}"


class QAPairIntermediate(BaseModel):
    question: str
    answer: Optional[str]

    def __str__(self):
        return f"Question: {self.question} \n Answer: {self.answer}"


class QAPairFinal(BaseModel):
    identifier: str
    index: int
    question: str
    answer: Optional[str]
    source: AtomicFactDocument

    def __str__(self):
        return f"Question: {self.question} \n Answer: {self.answer}"


class SingleExperimentInput(BaseModel):
    question: str
    expected_answer: Optional[str]
    context_questions: Optional[List[QAPairIntermediate]]


class ExperimentType(Enum):
    REMOVAL = "removal"
    SYNTHETIC = "synthetic"


class QuestionDocument(BaseModel):
    path_to_store: Path
    identifier: str
    creation_timestamp: datetime = Field(default_factory=datetime.now)
    questions: List[QAPairFinal]

    def save_to_json(self) -> None:
        self.path_to_store.write_text(self.model_dump_json(indent=2))
        return
