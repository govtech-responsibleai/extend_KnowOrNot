from pydantic import BaseModel
from typing import List


class Sentence(BaseModel):
    text: str


class SplitSourceDocument(BaseModel):
    sentences: List[Sentence]

    def __str__(self) -> str:
        result = ""
        for i, sentence in enumerate(self.sentences):
            result += f"{i}: {sentence.text}\n"
        return result.strip()
