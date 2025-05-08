from datetime import datetime
from enum import Enum
import json
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
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
    def load_from_json(path: Path | str) -> "AtomicFactDocument":
        """
        Read an AtomicFactDocument from a JSON file at the specified path.

        Args:
            path (Path | str): Path or string to the JSON file

        Returns:
            AtomicFactDocument: The loaded document

        Raises:
            ValueError: If the path doesn't end with .json
            FileNotFoundError: If the file doesn't exist
        """
        if isinstance(path, str):
            path = Path(path)

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

    def __str__(self, show_identifier: bool = False):
        if show_identifier:
            return f"Identifier: {self.identifier} \n Question: {self.question} \n Answer: {self.answer}"
        return f"Question: {self.question} \n Answer: {self.answer}"


class QAResponse(BaseModel):
    response: str
    citation: Union[int, Literal["no citation"]]


class Prompt(BaseModel):
    identifier: str
    content: str


class QAWithContext(BaseModel):
    identifier: str
    question: str
    expected_answer: str
    context_questions: Optional[List[QAPairFinal]]


class IndividualExperimentInput(BaseModel):
    prompt_to_llm: str
    source_context_qa: QAWithContext


class SavedLLMResponse(BaseModel):
    identifier: str
    experiment_input: IndividualExperimentInput
    llm_response: QAResponse
    cited_QA: Optional[QAPairFinal]


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
    def load_from_json(path: Path | str) -> "QuestionDocument":
        if isinstance(path, str):
            path = Path(path)
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
    client_enum: SyncLLMClientEnum
    ai_model_used: str
    knowledge_base_identifier: str


class ExperimentInputDocument(BaseModel):
    metadata: ExperimentMetadata
    questions: List[IndividualExperimentInput]

    def save_to_json(self) -> None:
        self.metadata.input_path.write_text(self.model_dump_json(indent=2))

    @staticmethod
    def load_from_json(path: Path | str) -> "ExperimentInputDocument":
        if isinstance(path, str):
            path = Path(path)
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "r") as f:
            text = f.read()

        return ExperimentInputDocument.model_validate_json(text)


class ExperimentOutputDocument(BaseModel):
    metadata: ExperimentMetadata
    responses: List[SavedLLMResponse]

    def save_to_json(self) -> None:
        self.metadata.output_path.write_text(self.model_dump_json(indent=2))

    @staticmethod
    def load_from_json(path: Path | str) -> "ExperimentOutputDocument":
        if isinstance(path, str):
            path = Path(path)
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "r") as f:
            text = f.read()

        return ExperimentOutputDocument.model_validate_json(text)


class ContextOptionsEnum(Enum):
    QUESTION = "question"
    EXPECTED_ANSWER = "expected_answer"
    CONTEXT = "context"
    CITED_QA = "cited_qa"


class EvaluationSpec(BaseModel):
    name: str
    prompt: Prompt
    tag_name: str
    recommended_llm_client_enum: Optional[SyncLLMClientEnum]
    recommended_llm_model: Optional[str]
    evaluation_outcomes: List[str]
    in_context: List[ContextOptionsEnum] = list(ContextOptionsEnum)


class EvaluationOutput(BaseModel):
    evaluation_id: str
    evaluation_timestamp: datetime
    evaluation_name: str
    evaluation_outcome: str


class LLMResponseWithEvaluation(BaseModel):
    llm_response: SavedLLMResponse
    evaluations: List[EvaluationOutput]


class EvaluationMetadata(BaseModel):
    evaluation_name: str
    evaluator_client_enum: SyncLLMClientEnum
    evaluator_model: str
    evaluation_prompt: Prompt
    tag_name: str
    evaluation_outcomes_list: List[str]
    in_context: List[ContextOptionsEnum] = list(ContextOptionsEnum)


class EvaluatedExperimentDocument(BaseModel):
    path_to_store: Path
    experiment_metadata: ExperimentMetadata
    evaluation_metadata: List[EvaluationMetadata]
    responses: List[LLMResponseWithEvaluation]

    def save_to_json(self) -> None:
        self.path_to_store.write_text(self.model_dump_json(indent=2))

    @staticmethod
    def load_from_json(path: Path | str) -> "EvaluatedExperimentDocument":
        if isinstance(path, str):
            path = Path(path)
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "r") as f:
            text = f.read()

        return EvaluatedExperimentDocument.model_validate_json(text)


class LabelTask(BaseModel):
    name: str
    values: List[str]
    content_in_context: List[ContextOptionsEnum] = [
        ContextOptionsEnum.QUESTION,
        ContextOptionsEnum.EXPECTED_ANSWER,
        ContextOptionsEnum.CONTEXT,
        ContextOptionsEnum.CITED_QA,
    ]


class HumanLabel(BaseModel):
    """Represents a single human label for a specific task on a sampled response."""

    labeller_id: str
    label_task: LabelTask
    label_value: str
    timestamp: datetime = Field(default_factory=datetime.now)


class LabeledDataSample(BaseModel):
    """
    Represents a single data point sampled from experiment results, intended for
    manual labeling across potentially multiple labeling tasks.

    Stores the original experiment context, LLM interaction details, stratification
    key, and a list of human labels collected for various tasks on this sample.
    """

    sample_id: str = Field(
        default_factory=lambda: f"sample_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}"
    )
    experiment_metadata: ExperimentMetadata
    question: str
    expected_answer: str
    context_questions: Optional[List[QAPairFinal]]
    llm_system_prompt: Prompt
    llm_response: SavedLLMResponse
    stratum_key: str

    human_labels: List[HumanLabel] = Field(default_factory=list)
    label_tasks: Optional[List[LabelTask]] = Field(default_factory=list)

    @staticmethod
    def save_list_to_json(samples: List["LabeledDataSample"], path: Path) -> None:
        """
        Save a list of LabeledDataSample models to a JSON file.

        Args:
            samples (List[LabeledDataSample]): The list of samples to save.
            path (Path): The path to the JSON file.
        """
        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    [sample.model_dump(mode="json") for sample in samples], indent=2
                )
            )

    @staticmethod
    def load_list_from_json(path: Path | str) -> List["LabeledDataSample"]:
        """
        Load a list of LabeledDataSample models from a JSON file.

        Args:
            path (Path | str): Path or string to the JSON file.

        Returns:
            List[LabeledDataSample]: The loaded list of samples.

        Raises:
            ValueError: If the path doesn't end with .json.
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file content is invalid JSON.
            ValidationError: If the JSON structure doesn't match the model.
        """
        if isinstance(path, str):
            path = Path(path)

        if not path.suffix == ".json":
            raise ValueError(f"The path must end with .json. Got: {path}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [LabeledDataSample.model_validate(item) for item in data]


class InterAnnotatorAgreement(BaseModel):
    """
    Represents the results of inter-annotator agreement calculations for a labeling task.
    """

    task_name: str
    fleiss_kappa: float
    pairwise_cohens_kappa: Dict[Tuple[str, str], float]
    common_samples: int
    sample_ids: List[str]
    annotators: List[str]
    possible_labels: List[str]
    calculation_timestamp: datetime = Field(default_factory=datetime.now)
