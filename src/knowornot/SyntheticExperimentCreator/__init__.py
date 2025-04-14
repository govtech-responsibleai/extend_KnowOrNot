# -*- coding: utf-8 -*-
from ..SyncLLMClient import SyncLLMClient
from ..common.models import QAPair
from typing import List, Optional, Tuple, cast
from sklearn.cluster import KMeans

# No longer need silhouette_score or kneed
from .models import CanBeAnswered
import numpy as np
from math import floor
import warnings


class SyntheticExperimentCreator:
    def __init__(
        self,
        default_client: SyncLLMClient,
        default_synthetic_prompt: str,
        default_synthetic_check_prompt: str,
        # minimum_num_clusters is no longer relevant here
        random_state: Optional[int] = None,
        max_retries: int = 3,
        default_percentage: float = 0.5,
    ):
        """
        Initialize the SyntheticExperimentCreator with the specified parameters.
        The number of clusters for generation is now provided directly to the
        generate_synthetic_dataset method.

        Args:
            default_client (SyncLLMClient): The default synchronous LLM client to use for embedding and other operations.
            default_synthetic_prompt (str): The default prompt used for synthetic experiment creation.
            default_synthetic_check_prompt (str): The default prompt used for checking synthetic experiments.
            random_state (Optional[int]): The random state to use for reproducibility of random operations. Defaults to 42 if not specified.
            max_retries (int): The maximum number of retries for certain operations. Defaults to 3.
            default_percentage (float): The default percentage threshold used in the experiments. Defaults to 0.5.
        """
        self.default_synthetic_prompt = default_synthetic_prompt
        self.default_synthetic_check_prompt = default_synthetic_check_prompt
        self.default_client = default_client
        # self.minimum_num_clusters = max(2, minimum_num_clusters or 5) # REMOVED
        self.random_state = random_state or 42
        self.max_retries = max_retries
        self.default_percentage = default_percentage

    def _embed_qa_pair_list(
        self,
        qa_pair_list: List[QAPair],
        alternative_client: Optional[SyncLLMClient] = None,
    ) -> np.ndarray:
        """
        Embeds the question list and returns a numpy 2D array.
        It calls sync client llm's get embedding method.

        Args:
            qa_pair_list (List[QAPair]): The question-answer pairs to be embedded.
            alternative_client (Optional[SyncLLMClient]): An optional alternative client to use for embedding.

        Returns:
            np.ndarray: A 2D numpy array of float embeddings. Returns empty array if input list is empty.
        """
        if not qa_pair_list:
            return np.array([])  # Handle empty list
        client = alternative_client or self.default_client
        # Convert QAPair to string for embedding
        texts_to_embed = [f"Q: {qa.question} A: {qa.answer}" for qa in qa_pair_list]
        embeddings = client.get_embedding(texts_to_embed)
        return np.array(embeddings)

    # REMOVED: _find_optimal_num_clusters method

    def _cluster_qa_pair_list(
        self,
        qa_pair_list: List[QAPair],
        embeddings: np.ndarray,
        num_clusters: int,  # Now takes num_clusters directly
    ) -> List[List[QAPair]]:
        """
        Clusters the question list based on the embeddings into a specified number of clusters.

        Args:
            qa_pair_list (List[QAPair]): The question-answer pairs to be clustered.
            embeddings (np.ndarray): A 2D numpy array of float embeddings.
            num_clusters (int): The target number of clusters.

        Returns:
            List[List[QAPair]]: A list of clusters, where each cluster is a list of QA pairs.
                                Returns empty list if input is empty.
                                Handles cases where num_clusters is invalid relative to data size.
        """
        n_samples = len(qa_pair_list)

        if n_samples == 0:
            return []

        # --- Validate num_clusters ---
        if num_clusters <= 0:
            warnings.warn(
                f"Requested num_clusters ({num_clusters}) is invalid. Must be >= 1. Returning all items in one cluster."
            )
            return [qa_pair_list]
        if num_clusters == 1:
            # No need to run KMeans for 1 cluster
            return [qa_pair_list]
        if num_clusters > n_samples:
            warnings.warn(
                f"Requested num_clusters ({num_clusters}) is greater than the number of samples ({n_samples}). "
                f"Setting num_clusters to n_samples = {n_samples}."
            )
            num_clusters = n_samples  # Cap at n_samples

        # --- Perform Clustering ---
        try:
            # n_init='auto' suppresses a future warning in scikit-learn
            kmeans = KMeans(
                n_clusters=num_clusters, random_state=self.random_state, n_init="auto"
            ).fit(embeddings)
        except ValueError as e:
            # This might happen if, e.g., embeddings are degenerate
            warnings.warn(
                f"KMeans clustering failed for {num_clusters} clusters: {e}. Returning all items in one cluster as fallback."
            )
            return [qa_pair_list]

        clusters: List[List[QAPair]] = [[] for _ in range(num_clusters)]
        labels: np.ndarray = cast(np.ndarray, kmeans.labels_)

        for index, cluster_index in enumerate(labels):
            # Ensure cluster_index is within bounds
            if 0 <= cluster_index < num_clusters:
                clusters[cluster_index].append(qa_pair_list[index])
            else:
                warnings.warn(
                    f"Unexpected cluster index {cluster_index} encountered for item {index}. Max index should be {num_clusters - 1}. Skipping item."
                )

        # Remove empty clusters if any occurred (can happen if k > n_samples initially, or due to rare KMeans behavior)
        clusters = [cluster for cluster in clusters if cluster]

        return clusters

    def _check_if_question_can_be_answered(
        self,
        question: QAPair,
        validation_pool: List[QAPair],
        check_client: Optional[SyncLLMClient] = None,
        check_prompt: Optional[str] = None,
        check_model: Optional[str] = None,
    ) -> bool:
        """
        Checks if a question can be answered using the validation pool.

        Args:
            question (QAPair): The question to check.
            validation_pool (List[QAPair]): All questions to check against (original + accepted synthetic).
            check_client (Optional[SyncLLMClient]): Client for checking questions.
            check_prompt (Optional[str]): Prompt for checking questions.
            check_model (Optional[str]): Model to use for checking.

        Returns:
            bool: Whether the question can be answered using the validation pool.
        """
        client = check_client or self.default_client
        prompt = check_prompt or self.default_synthetic_check_prompt

        # Format the prompt input
        question_str = (
            f"Question to check:\nQ: {question.question}\nA: {question.answer}\n\n"
        )
        pool_str = "Validation Pool:\n" + "\n".join(
            [f"Q: {qa.question} A: {qa.answer}" for qa in validation_pool]
        )
        full_prompt = prompt + "\n" + question_str + pool_str

        try:
            response = client.get_structured_response(
                prompt=full_prompt,
                response_model=CanBeAnswered,
                ai_model=check_model,
            )
            return response.can_be_answered
        except Exception as e:
            warnings.warn(
                f"Error during question answerability check: {e}. Assuming question *cannot* be answered (treating as novel)."
            )
            return False  # Fail safely by assuming the question is novel

    def _generate_synthetic_questions_for_cluster(
        self,
        cluster: List[QAPair],
        gen_client: Optional[SyncLLMClient] = None,
        gen_prompt: Optional[str] = None,
        gen_model: Optional[str] = None,
        check_client: Optional[SyncLLMClient] = None,
        check_prompt: Optional[str] = None,
        check_model: Optional[str] = None,
        percentage: Optional[float] = None,
    ) -> List[QAPair]:
        """
        Generates synthetic questions for a cluster with validation against a growing pool.

        Args:
            cluster (List[QAPair]): The original cluster of questions.
            gen_client (Optional[SyncLLMClient]): Client for generating questions. Defaults to `self.default_client`.
            gen_prompt (Optional[str]): Prompt for generating questions. Defaults to `self.default_synthetic_prompt`.
            gen_model (Optional[str]): Model to use for generation. Defaults to `None`.
            check_client (Optional[SyncLLMClient]): Client for checking questions. Defaults to `self.default_client`.
            check_prompt (Optional[str]): Prompt for checking questions. Defaults to `self.default_synthetic_check_prompt`.
            check_model (Optional[str]): Model to use for checking. Uses the client's default model if not provided.
            percentage (Optional[float]): Percentage of the cluster size for new questions. Defaults to `self.default_percentage`.

        Returns:
            List[QAPair]: A list of accepted synthetic questions.
        """
        if not cluster:
            return []

        client = gen_client or self.default_client
        prompt = gen_prompt or self.default_synthetic_prompt
        effective_percentage = (
            percentage if percentage is not None else self.default_percentage
        )
        target_questions = floor(len(cluster) * effective_percentage)

        if target_questions <= 0:
            return []

        accepted_questions: List[QAPair] = []
        validation_pool = cluster.copy()

        cluster_str = "Existing Questions in Cluster:\n" + "\n".join(
            [f"Q: {qa.question} A: {qa.answer}" for qa in cluster]
        )
        generation_base_prompt = (
            prompt
            + "\n"
            + cluster_str
            + "\n\nGenerate a new, distinct question based on the themes above:"
        )

        attempts = 0
        max_total_attempts = target_questions * (self.max_retries + 1)

        while (
            len(accepted_questions) < target_questions and attempts < max_total_attempts
        ):
            attempts += 1
            retries = 0
            generated_successfully = False
            question: Optional[QAPair] = None

            while retries < self.max_retries:
                try:
                    question = client.get_structured_response(
                        prompt=generation_base_prompt,
                        response_model=QAPair,
                        ai_model=gen_model,
                    )
                    if question and question.question and question.answer:
                        generated_successfully = True
                        break
                    else:
                        warnings.warn(
                            f"Generated empty question/answer pair. Retrying ({retries + 1}/{self.max_retries})..."
                        )
                        retries += 1
                except Exception as e:
                    warnings.warn(
                        f"Error generating question (attempt {retries + 1}/{self.max_retries}): {e}"
                    )
                    retries += 1

            if not generated_successfully or question is None:
                warnings.warn(
                    f"Failed to generate valid question after {self.max_retries} retries. Moving on."
                )
                continue

            try:
                can_be_answered = self._check_if_question_can_be_answered(
                    question,
                    validation_pool,
                    check_client,
                    check_prompt,
                    check_model,
                )
                if not can_be_answered:
                    accepted_questions.append(question)
                    validation_pool.append(question)
                else:
                    # Add warning for clarity when a question is rejected due to novelty check
                    warnings.warn(
                        f"Generated question (Q: {question.question}) deemed answerable by validation pool. Discarding."
                    )

            except Exception as e:
                warnings.warn(
                    f"Error checking question novelty: {e}. Discarding generated question."
                )

        if len(accepted_questions) < target_questions:
            warnings.warn(
                f"Generated {len(accepted_questions)} out of {target_questions} target synthetic questions for the cluster."
            )

        return accepted_questions

    def generate_synthetic_dataset(
        self,
        qa_pair_list: List[QAPair],
        num_clusters: int,  # User must provide this now
        gen_client: Optional[SyncLLMClient] = None,
        gen_prompt: Optional[str] = None,
        gen_model: Optional[str] = None,
        check_client: Optional[SyncLLMClient] = None,
        check_prompt: Optional[str] = None,
        check_model: Optional[str] = None,
        # min_clusters_override is removed
        percentage: Optional[float] = None,
    ) -> Tuple[List[QAPair], List[List[QAPair]]]:
        """
        Generates a complete synthetic dataset by clustering into a specified
        number of clusters and generating novel questions for each.

        Args:
            qa_pair_list (List[QAPair]): The list of question-answer pairs.
            num_clusters (int): The exact number of clusters to create. Must be >= 1.
                                If num_clusters > len(qa_pair_list), it will be capped.
            gen_client (Optional[SyncLLMClient]): Client for generating synthetic questions.
            gen_prompt (Optional[str]): Prompt for generating synthetic questions.
            gen_model (Optional[str]): AI model for generating synthetic questions.
            check_client (Optional[SyncLLMClient]): Client for validation.
            check_prompt (Optional[str]): Prompt for validation.
            check_model (Optional[str]): AI model for validation.
            percentage (Optional[float]): Percentage of questions to retain.

        Returns:
            Tuple[List[QAPair], List[List[QAPair]]]: A tuple containing:
            - A list of synthetic question-answer pairs.
            - A list of the actual clusters used for generation.
        """
        if not qa_pair_list:
            return ([], [])

        if num_clusters <= 0:
            warnings.warn(
                f"Invalid num_clusters ({num_clusters}) provided. Must be >= 1. Cannot generate dataset."
            )
            return (
                [],
                [],
            )  # Or maybe return original list as one cluster? Empty seems safer.

        embeddings = self._embed_qa_pair_list(qa_pair_list)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(qa_pair_list):
            warnings.warn(
                f"Embedding failed or produced unexpected shape ({embeddings.shape}). Cannot proceed with clustering."
            )
            return (
                [],
                [qa_pair_list] if qa_pair_list and num_clusters >= 1 else [],
            )  # Return original as 1 cluster if possible

        # Directly call cluster with the specified num_clusters
        clusters = self._cluster_qa_pair_list(qa_pair_list, embeddings, num_clusters)

        if not clusters:
            warnings.warn(
                "Clustering resulted in no clusters (potentially due to invalid num_clusters or KMeans errors). No synthetic data generated."
            )
            return ([], [])

        synthetic_questions = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue  # Skip empty clusters

            cluster_questions = self._generate_synthetic_questions_for_cluster(
                cluster,
                gen_client,
                gen_prompt,
                gen_model,
                check_client,
                check_prompt,
                check_model,
                percentage,
            )
            synthetic_questions.extend(cluster_questions)

        return synthetic_questions, clusters
