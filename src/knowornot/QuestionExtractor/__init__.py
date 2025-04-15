from typing import Optional, List
from ..SyncLLMClient import SyncLLMClient
from ..common.models import (
    QAPairIntermediate,
    QAPairLLM,
    AtomicFact,
    AtomicFactDocument,
)
from .models import FilterMethod
import asyncio
import concurrent.futures
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


class QuestionExtractor:
    def __init__(
        self,
        question_prompt_default: str,
        default_client: SyncLLMClient,
        logger: logging.Logger,
    ):
        self.question_prompt_default = question_prompt_default
        self.default_client = default_client
        self.logger = logger
        # Download NLTK resources if needed
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

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
    ) -> QAPairIntermediate:
        """
        Generates a question-answer pair from a single atomic fact using an LLM client.

        This method constructs a prompt by combining the context prompt with either an
        alternative question prompt or the default question generation question prompt. It then sends the
        constructed prompt to the LLM client to generate a structured response, which is
        converted into a QAPairIntermediate.

        Args:
            llm_client (SyncLLMClient): The LLM client used to generate the question-answer pair.
            fact (AtomicFact): The atomic fact from which to generate the question-answer pair.
            context_prompt (str): The context to include in the prompt sent to the LLM.
            alternative_question_prompt (Optional[str]): An optional alternative question prompt to use instead of the default.
            ai_model (Optional[str]): The AI model to use for generating the structured response.

        Returns:
            QAPairIntermediate: A question-answer pair generated from the atomic fact.
        """
        question_prompt_to_use = (
            alternative_question_prompt or self.question_prompt_default
        )
        text_to_llm = self._construct_text_to_llm(
            context_prompt=context_prompt,
            question_prompt=question_prompt_to_use,
            fact=fact,
        )
        self.logger.debug(f"prompt to llm: {text_to_llm}")
        self.logger.info(f"generating question from fact: {fact}")
        qa_pair = llm_client.get_structured_response(
            prompt=text_to_llm, response_model=QAPairLLM, ai_model=ai_model
        )

        output = QAPairIntermediate(
            question=qa_pair.question, answer=qa_pair.answer, source=fact
        )

        return output

    def generate_question_from_document(
        self,
        llm_client: SyncLLMClient,
        document: AtomicFactDocument,
        context_prompt: str,
        alternative_question_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
    ) -> List[QAPairIntermediate]:
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
            List[QAPairIntermediate]: A list of question-answer pairs generated from the atomic fact document.
        """
        output: List[QAPairIntermediate] = []

        for idx, fact in enumerate(document.fact_list):
            self.logger.info(f"generating question from {idx} fact: {fact}")
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
    ) -> List[QAPairIntermediate]:
        """
        Generates a list of question-answer pairs from an atomic fact document using an LLM client.

        This method constructs a prompt by combining the context prompt with either an
        alternative question prompt or the default question generation question prompt. It then sends the
        constructed prompt to the LLM client to generate a structured response, which is
        converted into a QAPairIntermediate. The calls to the LLM client are run in parallel using
        asyncio.

        Args:
            llm_client (SyncLLMClient): The LLM client used to generate the question-answer pairs.
            document (AtomicFactDocument): The atomic fact document from which to generate the question-answer pairs.
            context_prompt (str): The context to include in the prompt sent to the LLM.
            alternative_question_prompt (Optional[str]): An optional alternative question prompt to use instead of the default.
            ai_model (Optional[str]): The AI model to use for generating the structured response.

        Returns:
            List[QAPairIntermediate]: A list of question-answer pairs generated from the atomic fact document.
        """
        output: List[QAPairIntermediate] = []
        loop = asyncio.get_running_loop()
        futures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for idx, fact in enumerate(document.fact_list):
                self.logger.info(f"generating question from {idx} fact: {fact}")
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

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TF-IDF analysis:
        - Convert to lowercase
        - Remove punctuation
        - Remove stop words
        - Apply stemming

        Args:
            text: The text to preprocess

        Returns:
            Preprocessed text
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        words = text.split()
        words = [
            self.stemmer.stem(word) for word in words if word not in self.stop_words
        ]
        return " ".join(words)

    def _filter_keyword_duplicates(
        self, questions: List[QAPairIntermediate], diversity_threshold: float = 0.3
    ) -> List[QAPairIntermediate]:
        """
        Filter questions based on keyword diversity using TF-IDF.

        Args:
            questions: List of QAPairIntermediate objects to filter
            diversity_threshold: Minimum TF-IDF uniqueness score required (higher = stricter filtering)

        Returns:
            List of filtered QAPairIntermediate objects with unique keyword content
        """
        if not questions:
            return []

        # Extract and preprocess question texts
        question_texts = [self._preprocess_text(qa.question) for qa in questions]

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
        )

        tfidf_matrix = vectorizer.fit_transform(question_texts)

        # Calculate uniqueness scores (sum of TF-IDF values for each question)
        uniqueness_scores = np.sum(tfidf_matrix.toarray(), axis=1)  # type: ignore

        # Sort questions by uniqueness score (most unique first)
        sorted_indices = np.argsort(-uniqueness_scores)

        # Calculate threshold based on the distribution of scores
        min_score = uniqueness_scores.min()
        max_score = uniqueness_scores.max()
        score_range = max_score - min_score
        threshold_value = min_score + (score_range * diversity_threshold)

        # Filter questions above the threshold
        selected_indices = [
            idx for idx in sorted_indices if uniqueness_scores[idx] >= threshold_value
        ]

        self.logger.info(
            f"Keyword filtering: Selected {len(selected_indices)} out of {len(questions)} questions"
        )

        return [questions[idx] for idx in selected_indices]

    def _filter_semantic_duplicates(
        self, questions: List[QAPairIntermediate], min_distance: float = 0.3
    ) -> List[QAPairIntermediate]:
        """
        Filter questions based on semantic diversity using embeddings.

        Args:
            questions: List of QAPairIntermediate objects to filter
            min_distance: Minimum cosine distance required between questions (higher = more diverse)

        Returns:
            List of filtered QAPairIntermediate objects with diverse semantic meaning
        """
        if not questions:
            return []

        # Extract question texts, possibly combining with answers for better semantic understanding
        question_texts = [qa.question for qa in questions]

        # Get embeddings using the LLM client
        embeddings = self.default_client.get_embedding(question_texts)

        # Convert to numpy array for easier manipulation
        embeddings_array = np.array(embeddings)

        # Normalize embeddings to make cosine distance calculations easier
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized_embeddings = embeddings_array / norms

        # Greedy selection algorithm
        selected_indices = [0]  # Start with the first question
        remaining_indices = list(range(1, len(questions)))

        while remaining_indices:
            max_min_distance = -1
            best_idx = -1

            for idx in remaining_indices:
                # Calculate distances to all selected questions
                distances = []
                for selected_idx in selected_indices:
                    # Cosine distance = 1 - cosine similarity
                    similarity = np.dot(
                        normalized_embeddings[idx], normalized_embeddings[selected_idx]
                    )
                    distance = 1.0 - similarity
                    distances.append(distance)

                # Find the minimum distance to any selected question
                min_distance_to_selected = min(distances)

                # Keep track of the question with the maximum minimum distance
                if min_distance_to_selected > max_min_distance:
                    max_min_distance = min_distance_to_selected
                    best_idx = idx

            # If the best candidate is diverse enough, add it
            if max_min_distance >= min_distance:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                # No remaining questions are diverse enough
                break

        self.logger.info(
            f"Semantic filtering: Selected {len(selected_indices)} out of {len(questions)} questions"
        )

        return [questions[idx] for idx in selected_indices]

    def get_diverse_questions(
        self,
        question_list: List[QAPairIntermediate],
        method: FilterMethod = FilterMethod.SEMANTIC,
        diversity_threshold: float = 0.3,
    ) -> List[QAPairIntermediate]:
        """
        Generate diverse questions from a list of questions.

        Args:
            question_list: A list of questions to filter for diversity
            method: Filtering method - KEYWORD, SEMANTIC, or BOTH
            diversity_threshold: Threshold for filtering (higher = stricter)

        Returns:
            List of diverse questions
        """
        if not question_list:
            return []

        self.logger.info(
            f"Generating diverse questions using {method.value} method with threshold {diversity_threshold}"
        )

        # Apply the selected filtering method(s)
        if method == FilterMethod.KEYWORD:
            return self._filter_keyword_duplicates(question_list, diversity_threshold)

        elif method == FilterMethod.SEMANTIC:
            return self._filter_semantic_duplicates(question_list, diversity_threshold)

        elif method == FilterMethod.BOTH:
            # Apply keyword filtering first, then semantic filtering
            keyword_filtered = self._filter_keyword_duplicates(
                question_list, diversity_threshold
            )
            return self._filter_semantic_duplicates(
                keyword_filtered, diversity_threshold
            )

        else:
            raise ValueError(f"Unknown filtering method: {method}")
