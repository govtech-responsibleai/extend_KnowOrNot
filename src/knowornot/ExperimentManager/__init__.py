import asyncio
from datetime import datetime
from ..common.models import (
    ExperimentOutputDocument,
    IndividualExperimentInput,
    QAPairFinal,
    QAWithContext,
    QAResponse,
    SavedLLMResponse,
    ExperimentInputDocument,
    ExperimentType,
    ExperimentMetadata,
)
from ..SyncLLMClient import SyncLLMClient, SyncLLMClientEnum
from ..RetrievalStrategy import RetrievalType, BaseRetrievalStrategy
from ..RetrievalStrategy.direct_experiment import DirectRetrievalStrategy
from ..RetrievalStrategy.basic_rag import BasicRAGStrategy
from ..RetrievalStrategy.long_in_context import LongInContextStrategy
from ..RetrievalStrategy.hyde_rag import HydeRAGStrategy
from .models import (
    ExperimentParams,
)
from typing import Dict, List, Optional
import logging
import concurrent.futures
from tqdm import tqdm


class ExperimentManager:
    def __init__(
        self,
        default_client: SyncLLMClient,
        logger: logging.Logger,
        hypothetical_answer_prompt: str,
    ):
        self.default_client = default_client
        self.logger = logger

        self.retrieval_strategies: Dict[RetrievalType, BaseRetrievalStrategy] = {
            RetrievalType.DIRECT: DirectRetrievalStrategy(
                default_client=default_client, logger=logger
            ),
            RetrievalType.BASIC_RAG: BasicRAGStrategy(
                default_client=default_client, logger=logger
            ),
            RetrievalType.LONG_IN_CONTEXT: LongInContextStrategy(
                default_client=default_client, logger=logger
            ),
            RetrievalType.HYDE_RAG: HydeRAGStrategy(
                default_client=default_client,
                logger=logger,
                hypothetical_answer_prompt=hypothetical_answer_prompt,
            ),
        }

    def get_retrieval_strategy(self, retrieval_type: RetrievalType):
        try:
            return self.retrieval_strategies[retrieval_type]
        except KeyError as e:
            raise KeyError(f"Retrieval type {retrieval_type} is not registered") from e

    def _create_context_string(self, qa_pairs: Optional[List[QAPairFinal]]) -> str:
        if not qa_pairs:
            return ""

        context_string = ""
        for idx, qa_pair in enumerate(qa_pairs):
            context_string += (
                f"Question {idx}: {qa_pair.question}\nAnswer {idx}: {qa_pair.answer}\n"
            )

        return context_string

    def _process_QA_response(
        self,
        QA_response: QAResponse,
        experiment_input: IndividualExperimentInput,
        knowledge_base_identifier: str,
        idx: int,
    ) -> SavedLLMResponse:
        """
        Process a QA response and create a SavedLLMResponse object

        Returns a SavedLLMResponse object with citation=None if the response processing fails due to citation errors

        Args:
            QA_response: The response from the LLM
            experiment_input: The original input to the experiment
            knowledge_base_identifier: Identifier for the knowledge base
            idx: Index of the question being processed

        Returns:
            SavedLLMResponse
        """
        if QA_response.citation == "no citation":
            citation = None
        else:
            context_questions = experiment_input.source_context_qa.context_questions

            if context_questions is None:
                self.logger.error(
                    f"Context is None but citation is not 'no citation'. This should not happen. {experiment_input}"
                )
                citation = None
            elif (
                QA_response.citation >= len(context_questions)
                or QA_response.citation < 0
            ):
                self.logger.error(
                    f"Citation is out of bounds. This should not happen. {QA_response.citation} is the citation but there are only {len(context_questions)} context questions"
                )
                citation = None
            else:
                citation = context_questions[QA_response.citation]

        identifier = (
            knowledge_base_identifier
            + "_"
            + str(idx)
            + "_"
            + datetime.now().strftime("%Y%m%d%H%M%S")
        )

        return SavedLLMResponse(
            identifier=identifier,
            experiment_input=experiment_input,
            llm_response=QA_response,
            cited_QA=citation,
        )

    def convert_qa_with_context_to_individual_experiment_inputs(
        self, system_prompt: str, qa_with_context_list: List[QAWithContext]
    ) -> List[IndividualExperimentInput]:
        individual_experiment_inputs = []
        for qa_with_context in qa_with_context_list:
            individual_experiment_inputs.append(
                IndividualExperimentInput(
                    prompt_to_llm=system_prompt
                    + qa_with_context.question
                    + "\n"
                    + f"The context is {self._create_context_string(qa_with_context.context_questions)}",
                    source_context_qa=qa_with_context,
                )
            )
        return individual_experiment_inputs

    def create_experiment(
        self, experiment_params: ExperimentParams
    ) -> ExperimentInputDocument:
        retrival_strategy = self.get_retrieval_strategy(
            experiment_params.retrieval_type
        )

        if experiment_params.experiment_type == ExperimentType.REMOVAL:
            qas_with_context = retrival_strategy.create_removal_experiments(
                question_list=experiment_params.questions,
                alternative_prompt=experiment_params.alternative_prompt_for_hyde.content
                if experiment_params.alternative_prompt_for_hyde
                else None,
                alternative_llm_client=experiment_params.alternative_llm_client_for_hyde,
                ai_model=experiment_params.ai_model_for_hyde,
            )

        elif experiment_params.experiment_type == ExperimentType.SYNTHETIC:
            qas_with_context = retrival_strategy.create_synthetic_experiments(
                synthetic_questions=experiment_params.questions,
                context_questions=experiment_params.questions,
                alternative_prompt=experiment_params.alternative_prompt_for_hyde.content
                if experiment_params.alternative_prompt_for_hyde
                else None,
                alternative_llm_client=experiment_params.alternative_llm_client_for_hyde,
                ai_model=experiment_params.ai_model_for_hyde,
            )

        else:
            raise ValueError(
                f"Unknown experiment type {experiment_params.experiment_type}"
            )

        individual_experiment_inputs = (
            self.convert_qa_with_context_to_individual_experiment_inputs(
                system_prompt=experiment_params.system_prompt.content,
                qa_with_context_list=qas_with_context,
            )
        )

        metadata = ExperimentMetadata(
            experiment_type=experiment_params.experiment_type,
            retrieval_type=experiment_params.retrieval_type,
            creation_timestamp=datetime.now(),
            system_prompt=experiment_params.system_prompt,
            input_path=experiment_params.input_path,
            output_path=experiment_params.output_path,
            client_enum=experiment_params.llm_client_enum,
            ai_model_used=experiment_params.ai_model_for_experiment,
            knowledge_base_identifier=experiment_params.knowledge_base_identifier,
        )

        return ExperimentInputDocument(
            questions=individual_experiment_inputs,
            metadata=metadata,
        )

    def run_experiment_sync(
        self,
        experiment: ExperimentInputDocument,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
    ) -> ExperimentOutputDocument:
        client_enum = experiment.metadata.client_enum
        sync_client = client_registry[client_enum]

        self.logger.info(
            f"Running experiment with client {client_enum} and model {experiment.metadata.ai_model_used}"
        )
        self.logger.info(f"There are {len(experiment.questions)} questions")
        self.logger.info(
            f"The experiment type is {experiment.metadata.experiment_type}"
        )

        llm_response_list: List[SavedLLMResponse] = []
        for idx, experiment_input in enumerate(experiment.questions):
            self.logger.info(f"Running question {idx}")
            answer = sync_client.get_structured_response(
                prompt=experiment_input.prompt_to_llm,
                ai_model=experiment.metadata.ai_model_used,
                response_model=QAResponse,
            )

            final_response = self._process_QA_response(
                experiment_input=experiment_input,
                knowledge_base_identifier=experiment.metadata.knowledge_base_identifier,
                QA_response=answer,
                idx=idx,
            )

            llm_response_list.append(final_response)
            self.logger.info(f"Finished question {idx}")

        return ExperimentOutputDocument(
            metadata=experiment.metadata, responses=llm_response_list
        )

    async def run_experiment_async(
        self,
        experiment: ExperimentInputDocument,
        client_registry: Dict[SyncLLMClientEnum, SyncLLMClient],
        max_workers: int = 8,
    ) -> ExperimentOutputDocument:
        client_enum = experiment.metadata.client_enum
        sync_client = client_registry[client_enum]

        self.logger.info(
            f"Running async experiment with client {client_enum} and model {experiment.metadata.ai_model_used}"
        )
        self.logger.info(f"There are {len(experiment.questions)} questions")
        self.logger.info(
            f"The experiment type is {experiment.metadata.experiment_type}"
        )

        # Create a thread pool executor with the specified number of workers
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        loop = asyncio.get_event_loop()

        llm_response_list: List[Optional[SavedLLMResponse]] = [None] * len(
            experiment.questions
        )

        async def process_question(idx: int, experiment_input):
            self.logger.info(f"Starting question {idx}")

            # Run the synchronous API call in a thread
            answer = await loop.run_in_executor(
                executor,
                lambda: sync_client.get_structured_response(
                    prompt=experiment_input.prompt_to_llm,
                    ai_model=experiment.metadata.ai_model_used,
                    response_model=QAResponse,
                ),
            )

            final_response = self._process_QA_response(
                experiment_input=experiment_input,
                knowledge_base_identifier=experiment.metadata.knowledge_base_identifier,
                QA_response=answer,
                idx=idx,
            )

            self.logger.info(f"Finished question {idx}")
            return idx, final_response

        # Create tasks for all questions
        tasks = []
        for idx, experiment_input in enumerate(experiment.questions):
            tasks.append(process_question(idx, experiment_input))

        # Wait for all tasks to complete
        pbar = tqdm(total=len(tasks), desc="Processing questions")
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            if result is not None:
                llm_response_list[idx] = result
            pbar.update(1)
        pbar.close()

        # Remove any None values (from failed processing)
        llm_response_list = [r for r in llm_response_list if r is not None]

        # Clean up
        executor.shutdown()

        assert all(r is not None for r in llm_response_list), (
            "All responses must be processed correctly"
        )
        new_response_list = [r for r in llm_response_list if r is not None]

        return ExperimentOutputDocument(
            metadata=experiment.metadata, responses=new_response_list
        )
