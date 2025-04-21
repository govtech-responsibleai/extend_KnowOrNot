from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Generic, List, Optional, Type, TypeVar, Union, Literal
from ..SyncLLMClient import SyncLLMClientEnum


class RetrievalType(Enum):
    DIRECT = "DIRECT"
    LONG_IN_CONTEXT = "LONG_IN_CONTEXT"
    BASIC_RAG = "BASIC_RAG"
    HYDE_RAG = "HYDE_RAG"


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


class QAPair(BaseModel):
    question: str
    answer: str

    def __str__(self):
        return f"Question: {self.question} \n Answer: {self.answer}"


class QAPairFinal(QAPair):
    identifier: str

    def __str__(self):
        return f"Index: {self.identifier} Question: {self.question} \n Answer: {self.answer}"


class QAResponse(BaseModel):
    response: str
    citation: Union[int, Literal["no citation"]]


class Prompt(BaseModel):
    identifier: str
    content: str


class QAWithContext(BaseModel):
    question: str
    expected_answer: str
    context_questions: Optional[List[QAPair]]


class SavedLLMResponse(BaseModel):
    identifier: str
    llm_response: QAResponse
    cited_QA: Optional[QAPair]


class IndividualExperimentInput(BaseModel):
    question_to_ask: str
    expected_answer: str
    context: Optional[List[QAPair]]


class ExperimentType(Enum):
    REMOVAL = "removal"
    SYNTHETIC = "synthetic"


class QuestionDocument(BaseModel):
    path_to_store: Path
    knowledge_base_identifier: str
    creation_timestamp: datetime = Field(default_factory=datetime.now)
    questions: List[QAPairFinal]

    def save_to_json(self) -> None:
        self.path_to_store.write_text(self.model_dump_json(indent=2))
        return

    @staticmethod
    def load_from_json(path: Path) -> "QuestionDocument":
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "r") as f:
            text = f.read()

        return QuestionDocument.model_validate_json(text)


class ExperimentMetadata(BaseModel):
    experiment_type: ExperimentType
    system_prompt: Prompt
    input_path: Path
    output_path: Path
    retrieval_type: RetrievalType
    creation_timestamp: datetime
    client: SyncLLMClientEnum
    ai_model_used: str
    knowledge_base_identifier: str


class ExperimentInputDocument(BaseModel):
    metadata: ExperimentMetadata
    questions: List[IndividualExperimentInput]

    def save_to_json(self) -> None:
        self.metadata.input_path.write_text(self.model_dump_json(indent=2))


class ExperimentOutputDocument(BaseModel):
    metadata: ExperimentMetadata
    responses: List[SavedLLMResponse]

    def save_to_json(self) -> None:
        self.metadata.output_path.write_text(self.model_dump_json(indent=2))

    @staticmethod
    def load_from_json(path: Path) -> "ExperimentOutputDocument":
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "r") as f:
            text = f.read()

        return ExperimentOutputDocument.model_validate_json(text)


T = TypeVar("T", bound=Enum, covariant=True)


class EvaluationSpec(BaseModel):
    name: str
    prompt: Prompt
    recommended_llm_client_enum: Optional[SyncLLMClientEnum]
    recommended_llm_model: Optional[str]
    evaluation_outcome: Type[Enum]
    in_context: List[Literal["question", "llm_answer", "context"]] = [
        "question",
        "llm_answer",
        "context",
    ]


class EvaluationOutput(BaseModel, Generic[T]):
    evaluation_id: str
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_name: str
    evaluation_outcome: T


class LLMResponseWithEvaluation(BaseModel, Generic[T]):
    llm_response: SavedLLMResponse
    evaluation: EvaluationOutput[T]


class EvaluationMetadata(BaseModel, Generic[T]):
    evaluation_name: str
    evaluator_client_enum: SyncLLMClientEnum
    evaluator_model: str
    evaluation_prompt: Prompt
    evaluation_outcomes_enum: Type[T]
    in_context: List[Literal["question", "llm_answer", "context"]] = [
        "question",
        "llm_answer",
        "context",
    ]


class EvaluatedExperimentDocument(BaseModel):
    experiment_metadata: ExperimentMetadata
    evaluation_metadata: List[EvaluationMetadata[Enum]]
    responses: List[LLMResponseWithEvaluation[Enum]]
