from pathlib import Path
from typing import Optional, List
from ..SyncLLMClient import SyncLLMClient
from ..common.models import (
    QAPairFinal,
    QAPair,
    AtomicFactDocument,
    QuestionDocument,
)
from .models import FilterMethod
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from .models import QuestionList
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
        self, context_prompt: str, question_prompt: str, document: AtomicFactDocument
    ) -> str:
        text_to_llm = (
            context_prompt + question_prompt + "The document is " + str(document)
        )
        return text_to_llm

    def _generate_questions_from_document(
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
        converted into a list of QAPair.

        Args:
            llm_client (SyncLLMClient): The LLM client used to generate the question-answer pairs.
            document (AtomicFactDocument): The atomic fact document from which to generate the question-answer pairs.
            context_prompt (str): The context to include in the prompt sent to the LLM.
            alternative_question_prompt (Optional[str]): An optional alternative question prompt to use instead of the default.
            ai_model (Optional[str]): The AI model to use for generating the structured response.

        Returns:
            List[QAPair]: A list of question-answer pairs generated from the atomic fact document.
        """
        question_prompt_to_use = (
            alternative_question_prompt or self.question_prompt_default
        )
        text_to_llm = self._construct_text_to_llm(
            context_prompt=context_prompt,
            question_prompt=question_prompt_to_use,
            document=document,
        )
        self.logger.debug(f"prompt to llm: {text_to_llm}")
        question_document = llm_client.get_structured_response(
            prompt=text_to_llm, response_model=QuestionList, ai_model=ai_model
        )

        output: List[QAPair] = []
        for qapair in question_document.questions:
            output.append(QAPair(question=qapair.question, answer=qapair.answer))

        return output

    def generate_questions_from_documents(
        self,
        knowledge_base_identifier: str,
        documents: List[AtomicFactDocument],
        context_prompt: str,
        method: FilterMethod,
        path_to_save: Path,
        alternative_question_prompt: Optional[str] = None,
        ai_model: Optional[str] = None,
        diversity_threshold_keyword: Optional[float] = 0.3,
        diversity_threshold_semantic: Optional[float] = 0.3,
        alternative_llm_client: Optional[SyncLLMClient] = None,
    ) -> QuestionDocument:
        """
        Generates a diverse list of question-answer pairs from an atomic fact document using an LLM client.

        This method iterates over each atomic fact in the document, calls
        `_generate_question_from_single_fact` to generate a question-answer pair for each fact,
        accumulates the generated question-answer pairs in a list and then filters out the
        non-diverse questions based on the filter method.

        Args:
            llm_client (SyncLLMClient): The LLM client used to generate the question-answer pairs.
            document (AtomicFactDocument): The atomic fact document from which to generate the question-answer pairs.
            context_prompt (str): The context to include in the prompt sent to the LLM.
            alternative_question_prompt (Optional[str]): An optional alternative question prompt to use instead of the default.
            ai_model (Optional[str]): The AI model to use for generating the structured response.
            method (FilterMethod): The method to use for filtering out non-diverse questions.
            diversity_threshold_keyword (float): The threshold for filtering out non-diverse questions based on keyword similarity.
            diversity_threshold_semantic (float): The threshold for filtering out non-diverse questions based on semantic similarity.

        Returns:
            QuestionDocument: A `QuestionDocument` containing the identifier and a list of diverse question-answer pairs generated from the atomic fact document.
        """

        total_questions: List[QAPair] = []

        llm_client = alternative_llm_client or self.default_client

        for document in documents:
            qa_pairs = self._generate_questions_from_document(
                llm_client=llm_client,
                document=document,
                context_prompt=context_prompt,
                alternative_question_prompt=alternative_question_prompt,
                ai_model=ai_model,
            )

            total_questions.extend(qa_pairs)

        diversity_threshold_keyword = diversity_threshold_keyword or 0.3
        diversity_threshold_semantic = diversity_threshold_semantic or 0.3
        intermediate_pairs = self._get_diverse_questions(
            question_list=total_questions,
            method=method,
            diversity_threshold_keyword=diversity_threshold_keyword,
            diversity_threshold_semantic=diversity_threshold_semantic,
        )

        if not intermediate_pairs:
            raise ValueError(
                "No diverse questions generated. Please try again with different parameters."
            )

        final_questions: List[QAPairFinal] = []

        for idx, qapair in enumerate(intermediate_pairs):
            identifier = f"{knowledge_base_identifier}_{idx}"
            final_questions.append(
                QAPairFinal(
                    identifier=identifier,
                    question=qapair.question,
                    answer=qapair.answer,
                )
            )

        output = QuestionDocument(
            knowledge_base_identifier=knowledge_base_identifier,
            questions=final_questions,
            path_to_store=path_to_save,
        )

        self.logger.info(
            f"Generated {len(output.questions)} questions from {len(intermediate_pairs)} pairs."
        )
        self.logger.info(
            f"{len(intermediate_pairs) - len(output.questions)} pairs were filtered out because they were too similar"
        )

        return output

    def filter_questions(
        self,
        method: FilterMethod,
        path_to_save: Path,
        identifier: str,
        questions: List[QAPair],
        diversity_threshold_keyword: float = 0.3,
        diversity_threshold_semantic: float = 0.3,
    ) -> QuestionDocument:
        filtered_questions = self._get_diverse_questions(
            question_list=questions,
            method=method,
            diversity_threshold_keyword=diversity_threshold_keyword,
            diversity_threshold_semantic=diversity_threshold_semantic,
        )

        final_qa_pairs: List[QAPairFinal] = []

        for idx, qapair in enumerate(filtered_questions):
            identifier = f"{identifier}_{idx}"
            final_qa_pairs.append(
                QAPairFinal(
                    identifier=identifier,
                    question=qapair.question,
                    answer=qapair.answer,
                )
            )

        output = QuestionDocument(
            knowledge_base_identifier=identifier,
            questions=final_qa_pairs,
            path_to_store=path_to_save,
        )

        self.logger.info(
            f"Generated {len(output.questions)} questions from {len(filtered_questions)} pairs."
        )
        self.logger.info(
            f"{len(filtered_questions) - len(output.questions)} pairs were filtered out because they were too similar"
        )

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
        self, questions: List[QAPair], diversity_threshold: float = 0.3
    ) -> List[QAPair]:
        """
        Filter questions based on keyword diversity using TF-IDF.

        Args:
            questions: List of QAPair objects to filter
            diversity_threshold: Minimum TF-IDF uniqueness score required (higher = stricter filtering)

        Returns:
            List of filtered QAPair objects with unique keyword content
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
        self, questions: List[QAPair], min_distance: float = 0.3
    ) -> List[QAPair]:
        """
        Filter questions based on semantic diversity using embeddings.

        Args:
            questions: List of QAPair objects to filter
            min_distance: Minimum cosine distance required between questions (higher = more diverse)

        Returns:
            List of filtered QAPair objects with diverse semantic meaning
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

    def _get_diverse_questions(
        self,
        question_list: List[QAPair],
        method: FilterMethod,
        diversity_threshold_keyword: float = 0.3,
        diversity_threshold_semantic: float = 0.3,
    ) -> List[QAPair]:
        """
        Generate diverse questions from a list of questions.

        Args:
            question_list: A list of questions to filter for diversity
            method: Filtering method - KEYWORD, SEMANTIC, or BOTH
            diversity_threshold_keyword: Threshold for keyword filtering (higher = stricter)
            diversity_threshold_semantic: Threshold for semantic filtering (higher = stricter)

        Returns:
            List of diverse questions

        The method works as follows:
        - KEYWORD: Filter questions using TF-IDF uniqueness score. Questions with scores below the threshold are removed.
        - SEMANTIC: Filter questions using cosine similarity of embeddings. Questions with similarity above the threshold are removed.
        - BOTH: Apply both keyword and semantic filtering in sequence.
        """
        if not question_list:
            return []

        self.logger.info(
            f"Generating diverse questions using {method.value} method with thresholds: {diversity_threshold_keyword}, {diversity_threshold_semantic}"
        )

        # Apply the selected filtering method(s)
        if method == FilterMethod.KEYWORD:
            return self._filter_keyword_duplicates(
                question_list, diversity_threshold_keyword
            )

        elif method == FilterMethod.SEMANTIC:
            return self._filter_semantic_duplicates(
                question_list, diversity_threshold_semantic
            )

        elif method == FilterMethod.BOTH:
            keyword_filtered = self._filter_keyword_duplicates(
                question_list, diversity_threshold_keyword
            )
            return self._filter_semantic_duplicates(
                keyword_filtered, diversity_threshold_semantic
            )

        else:
            raise ValueError(f"Unknown filtering method: {method}")
