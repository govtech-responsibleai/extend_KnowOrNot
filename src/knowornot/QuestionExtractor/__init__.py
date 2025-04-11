from typing import Optional, List
from ..SyncLLMClient import SyncLLMClient
from ..common.models import QAPair, QAPairLLM, AtomicFact, AtomicFactDocument
import asyncio
import concurrent.futures


class QuestionExtractor:
    def __init__(self, question_prompt_default: str, default_client: SyncLLMClient):
        self.question_prompt_default = question_prompt_default
        self.default_client = default_client

    def _construct_text_to_llm(
        self, context_prompt: str, question_prompt: str, fact: AtomicFact
    ) -> str:
        text_to_llm = context_prompt + question_prompt + "The fact is " + str(fact)
        return text_to_llm

    def _generate_question_from_single_fact(
        self,
        llm_client: SyncLLMClient,
        fact: AtomicFact,
        context_prompt: str,
        alternative_question_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
    ) -> QAPair:
        """
        Generates a question-answer pair from a single atomic fact using an LLM client.

        This method constructs a prompt by combining the context prompt with either an
        alternative question prompt or the default question generation question prompt. It then sends the
        constructed prompt to the LLM client to generate a structured response, which is
        converted into a QAPair.

        Args:
            llm_client (SyncLLMClient): The LLM client used to generate the question-answer pair.
            fact (AtomicFact): The atomic fact from which to generate the question-answer pair.
            context_prompt (str): The context to include in the prompt sent to the LLM.
            alternative_question_prompt (Optional[str]): An optional alternative question prompt to use instead of the default.
            ai_model (Optional[str]): The AI model to use for generating the structured response.

        Returns:
            QAPair: A question-answer pair generated from the atomic fact.
        """
        question_prompt_to_use = (
            alternative_question_prompt or self.question_prompt_default
        )
        text_to_llm = self._construct_text_to_llm(
            context_prompt=context_prompt,
            question_prompt=question_prompt_to_use,
            fact=fact,
        )
        qa_pair = llm_client.get_structured_response(
            prompt=text_to_llm, response_model=QAPairLLM, ai_model=ai_model
        )

        output = QAPair(question=qa_pair.question, answer=qa_pair.answer, source=fact)

        return output

    def generate_question_from_document(
        self,
        llm_client: SyncLLMClient,
        document: AtomicFactDocument,
        context_prompt: str,
        alternative_question_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
    ) -> List[QAPair]:
        """
        Generates a list of question-answer pairs from an atomic fact document using an LLM client.

        This method iterates over each atomic fact in the document and calls
        `_generate_question_from_single_fact` to generate a question-answer pair for each fact.
        It then accumulates the generated question-answer pairs in a list and returns them.

        Args:
            llm_client (SyncLLMClient): The LLM client used to generate the question-answer pairs.
            document (AtomicFactDocument): The atomic fact document from which to generate the question-answer pairs.
            context_prompt (str): The context to include in the prompt sent to the LLM.
            alternative_question_prompt (Optional[str]): An optional alternative question prompt to use instead of the default.
            ai_model (Optional[str]): The AI model to use for generating the structured response.

        Returns:
            List[QAPair]: A list of question-answer pairs generated from the atomic fact document.
        """
        output: List[QAPair] = []

        for fact in document.fact_list:
            qa_pair = self._generate_question_from_single_fact(
                llm_client=llm_client,
                fact=fact,
                context_prompt=context_prompt,
                alternative_question_prompt=alternative_question_prompt,
                ai_model=ai_model,
            )
            output.append(qa_pair)

        return output

    async def generate_question_from_document_async(
        self,
        llm_client: SyncLLMClient,
        document: AtomicFactDocument,
        context_prompt: str,
        alternative_question_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
    ) -> List[QAPair]:
        """
        Generates a list of question-answer pairs from an atomic fact document using an LLM client.

        This method constructs a prompt by combining the context prompt with either an
        alternative question prompt or the default question generation question prompt. It then sends the
        constructed prompt to the LLM client to generate a structured response, which is
        converted into a QAPair. The calls to the LLM client are run in parallel using
        asyncio.

        Args:
            llm_client (SyncLLMClient): The LLM client used to generate the question-answer pairs.
            document (AtomicFactDocument): The atomic fact document from which to generate the question-answer pairs.
            context_prompt (str): The context to include in the prompt sent to the LLM.
            alternative_question_prompt (Optional[str]): An optional alternative question prompt to use instead of the default.
            ai_model (Optional[str]): The AI model to use for generating the structured response.

        Returns:
            List[QAPair]: A list of question-answer pairs generated from the atomic fact document.
        """
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
                    context_prompt,
                    alternative_question_prompt,
                    ai_model,
                )
                futures.append(future)

            results = await asyncio.gather(*futures)
            output.extend(results)

        return output
