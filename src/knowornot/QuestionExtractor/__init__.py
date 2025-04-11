from typing import Optional, List
from ..SyncLLMClient import SyncLLMClient
from ..FactManager.models import AtomicFact, AtomicFactDocument
from .models import QAPair, QAPairLLM
import asyncio
import concurrent.futures


class QuestionExtractor:
    def __init__(self, question_generation_prompt: str, default_client: SyncLLMClient):
        self.question_generation_prompt = question_generation_prompt
        self.default_client = default_client

    def _construct_text_to_llm(self, prompt: str, fact: AtomicFact) -> str:
        text_to_llm = prompt + "The fact is " + str(fact)
        return text_to_llm

    def _generate_question_from_single_fact(
        self,
        llm_client: SyncLLMClient,
        fact: AtomicFact,
        alternative_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
    ) -> QAPair:
        prompt_to_use = alternative_prompt or self.question_generation_prompt
        text_to_llm = self._construct_text_to_llm(prompt=prompt_to_use, fact=fact)
        qa_pair = llm_client.get_structured_response(
            prompt=text_to_llm, response_model=QAPairLLM, ai_model=ai_model
        )

        output = QAPair(question=qa_pair.question, answer=qa_pair.answer, source=fact)

        return output

    def generate_question_from_document(
        self,
        llm_client: SyncLLMClient,
        document: AtomicFactDocument,
        alternative_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
    ) -> List[QAPair]:
        output: List[QAPair] = []

        for fact in document.fact_list:
            qa_pair = self._generate_question_from_single_fact(
                llm_client=llm_client,
                fact=fact,
                alternative_prompt=alternative_prompt,
                ai_model=ai_model,
            )
            output.append(qa_pair)

        return output

    async def generate_question_from_document_async(
        self,
        llm_client: SyncLLMClient,
        document: AtomicFactDocument,
        alternative_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
    ) -> List[QAPair]:
        output: List[QAPair] = []
        loop = asyncio.get_running_loop()
        futures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for fact in document.fact_list:
                future = loop.run_in_executor(
                    executor,
                    self._generate_question_from_single_fact,
                    llm_client,
                    fact,
                    alternative_prompt,
                    ai_model,
                )
                futures.append(future)

            results = await asyncio.gather(*futures)
            output.extend(results)

        return output
