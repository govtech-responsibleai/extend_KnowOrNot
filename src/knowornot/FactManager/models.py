from pathlib import Path
from pydantic import BaseModel
from typing import List


class Sentence(BaseModel):
    text: str


class SplitSourceDocument(BaseModel):
    sentences: List[Sentence]

    def __str__(self) -> str:
        result = "Source document: \n"
        for i, sentence in enumerate(self.sentences):
            result += f"{i}: {sentence.text}\n"
        return result.strip()


class AtomicFact(BaseModel):
    fact_text: str
    source_citation: int


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
