# `create_questions`

```python
create_questions(
    source_paths: List[Path],
    knowledge_base_identifier: str,
    context_prompt: str,
    path_to_save_questions: Path,
    filter_method: Literal["keyword", "semantic", "both"],
    alternative_fact_creation_llm_prompt: Optional[Prompt] = None,
    alternative_fact_creator_llm_client: Optional[SyncLLMClient] = None,
    alternative_fact_creation_llm_model: Optional[str] = None,
    alternative_question_creation_llm_prompt: Optional[Prompt] = None,
    alternative_question_creator_llm_client: Optional[SyncLLMClient] = None,
    fact_storage_dir: Optional[Path] = None,
    semantic_filter_threshold: Optional[float] = None,
    keyword_filter_threshold: Optional[float] = None,
) -> QuestionDocument
```

This method is the primary way to create the question bank (`QuestionDocument`) that serves as the knowledge base for KnowOrNot experiments. It processes raw text documents to generate a set of diverse question-answer pairs relevant to their content.

The internal workflow involves:
1.  **Fact Extraction:** Parsing the source text into smaller, "atomic" facts using an LLM (via `FactManager`).
2.  **Initial Question Generation:** Creating a potentially larger set of question-answer pairs based on these atomic facts using an LLM (via `QuestionExtractor`).
3.  **Diversity Filtering:** Applying a chosen filtering method (keyword, semantic, or both) to select a diverse subset of the generated questions.
4.  **Packaging and Saving:** Formatting the final diverse questions into a `QuestionDocument` object and saving it to disk.

## Parameters

*   **`source_paths`** (`List[Path]`):
    *   **Required.**
    *   A list of `pathlib.Path` objects pointing to the input source files.
    *   **Constraint:** All paths must exist and point to files with a `.txt` extension.
    *   These documents are the source material from which facts and questions are derived.

*   **`knowledge_base_identifier`** (`str`):
    *   **Required.**
    *   A unique string that identifies the knowledge base represented by the generated questions. This identifier is included in the output `QuestionDocument` and is crucial for linking experiments and evaluations back to the source data.

*   **`context_prompt`** (`str`):
    *   **Required.**
    *   Additional natural language text prepended to the prompt sent to the LLM during the *question generation* step. This is distinct from the internal `question_prompt_default` and is used to guide the LLM on the desired style or focus of the questions (e.g., "Generate questions about key findings...").

*   **`path_to_save_questions`** (`Path`):
    *   **Required.**
    *   The `pathlib.Path` where the final `QuestionDocument` JSON file will be saved.
    *   **Constraint:** The path must end with a `.json` suffix.
    *   The parent directory must exist or be creatable.

*   **`filter_method`** (`Literal["keyword", "semantic", "both"]`):
    *   **Required.**
    *   Determines how the generated questions are filtered for diversity.
        *   `"keyword"`: Uses TF-IDF vectorization on question text to identify and filter out questions with high keyword overlap.
        *   `"semantic"`: Uses embedding similarity (cosine distance) to identify and filter out questions that are semantically very similar. Requires a client capable of generating embeddings.
        *   `"both"`: Applies keyword filtering, then applies semantic filtering to the result of the keyword filtering.

*   **`alternative_fact_creation_llm_prompt`** (`Optional[Prompt]`):
    *   Optional.
    *   An alternative `Prompt` object to use instead of the default internal prompt for the *atomic fact creation* step. Provides fine-grained control over how the LLM breaks down the source text into facts.

*   **`alternative_fact_creator_llm_client`** (`Optional[SyncLLMClient]`):
    *   Optional.
    *   A specific `SyncLLMClient` instance to use *only* for the *atomic fact creation* step. If not provided, the default client registered with the `KnowOrNot` instance is used.

*   **`alternative_fact_creation_llm_model`** (`Optional[str]`):
    *   Optional.
    *   A specific AI model (deployment name) to use *only* for the *atomic fact creation* step. If not provided, the default model of the chosen fact creation client (either `alternative_fact_creator_llm_client` if provided, or the `KnowOrNot` instance's default client) will be used.

*   **`alternative_question_creation_llm_prompt`** (`Optional[Prompt]`):
    *   Optional.
    *   An alternative `Prompt` object to use instead of the default internal prompt for the *question generation* step. Provides fine-grained control over how the LLM turns atomic facts into initial Q&A pairs.

*   **`alternative_question_creator_llm_client`** (`Optional[SyncLLMClient]`):
    *   Optional.
    *   A specific `SyncLLMClient` instance to use *only* for the *question generation* step. If not provided, the default client registered with the `KnowOrNot` instance is used.

*   **`fact_storage_dir`** (`Optional[Path]`):
    *   Optional.
    *   If provided, the intermediate `AtomicFactDocument`s generated from each source file will be saved as JSON files in this directory.
    *   **Constraint:** Must be an existing directory.
    *   Useful for auditing, debugging, or reusing the extracted facts later.

*   **`semantic_filter_threshold`** (`Optional[float]`):
    *   Optional.
    *   The minimum cosine distance required between questions for them to be considered semantically diverse when `filter_method` is `"semantic"` or `"both"`. Questions with a distance below this threshold to *any* selected question will be filtered out.
    *   Higher values result in fewer, more semantically distinct questions.
    *   Defaults to `0.3`.

*   **`keyword_filter_threshold`** (`Optional[float]`):
    *   Optional.
    *   The minimum TF-IDF uniqueness score required for a question to be kept when `filter_method` is `"keyword"` or `"both"`.
    *   Higher values result in fewer, more keyword-unique questions.
    *   Defaults to `0.3`.

## Returns

*   **`QuestionDocument`**: An object containing the `knowledge_base_identifier`, a creation timestamp, and the final list of diverse `QAPairFinal` objects generated from the source documents. This object is also automatically saved as a JSON file to the path specified by `path_to_save_questions`.

## Important Considerations

*   **LLM Dependency:** This method relies heavily on LLM calls (for fact extraction and question generation). The cost, speed, and quality of the output are directly influenced by the performance and configuration of the chosen LLM client and model(s).
*   **Prompt Tuning:** The `context_prompt` and `alternative_*_prompt` parameters are critical for guiding the LLMs. Experimenting with these prompts is often necessary to achieve the desired type and coverage of questions.
*   **Filtering Thresholds:** The `semantic_filter_threshold` and `keyword_filter_threshold` parameters directly control the trade-off between the total number of questions generated and their perceived diversity. Setting these too high might result in very few questions, while setting them too low might leave many near-duplicates.
*   **Embedding Model:** Semantic filtering relies on embeddings. If the default client (or the `alternative_question_creator_llm_client`) cannot produce embeddings, or if no embeddings model is configured, semantic filtering will not work correctly.
*   **NLTK:** Keyword filtering utilizes NLTK for tokenization, stop word removal, and stemming. Necessary NLTK data (`punkt_tab`, `stopwords`) is automatically downloaded by the `QuestionExtractor` if not found, but this requires internet access during the first run.
*   **Empty Output:** If the filtering process is too aggressive (e.g., high thresholds) or if the initial LLM generation fails to produce valid Q&A pairs, the method might result in an empty list of diverse questions, raising a `ValueError`.

## Examples

Assuming you have initialized a `KnowOrNot` instance and added at least one LLM client

```python
from knowornot import KnowOrNot
from knowornot.common.models import Prompt # Import Prompt if using alternative prompts
from pathlib import Path

# Assume 'kon' is your initialized KnowOrNot instance with a default client
# kon = KnowOrNot()
# kon.add_azure(...) # or other client setup

# Define input source files
source_files = [
    Path("data/source_documents/annual_report_summary.txt"),
    Path("data/source_documents/press_release.txt")
]

# Define output paths
questions_output_path = Path("data/generated_questions/company_overview_kb.json")
fact_storage_directory = Path("data/generated_facts/") # Optional: Directory to save intermediate facts

# Ensure output directories exist
questions_output_path.parent.mkdir(parents=True, exist_ok=True)
fact_storage_directory.mkdir(parents=True, exist_ok=True) # Create if using fact_storage_dir

# --- Example 1: Basic Usage with default filters ---
print("Generating questions with default settings...")
try:
    question_doc_default = kon.create_questions(
        source_paths=source_files,
        knowledge_base_identifier="company-overview-default",
        context_prompt="Generate factual questions based *strictly* on the provided text.",
        path_to_save_questions=questions_output_path.with_stem("company_overview_kb_default"), # Save with different name
        filter_method="both" # Use keyword and semantic filtering with default thresholds (0.3)
    )
    print(f"Successfully created QuestionDocument with {len(question_doc_default.questions)} questions.")
    print(f"Saved to: {question_doc_default.path_to_store}")

except Exception as e:
    print(f"An error occurred during default question creation: {e}")


# --- Example 2: Using Semantic Filtering with a custom threshold and saving facts ---
print("\nGenerating questions with semantic filtering and custom threshold...")
try:
    question_doc_semantic = kon.create_questions(
        source_paths=source_files,
        knowledge_base_identifier="company-overview-semantic",
        context_prompt="Extract distinct facts and ask questions directly about them.",
        path_to_save_questions=questions_output_path.with_stem("company_overview_kb_semantic"),
        filter_method="semantic", # Use only semantic filtering
        semantic_filter_threshold=0.5, # Require higher semantic diversity (0.5 instead of 0.3)
        fact_storage_dir=fact_storage_directory # Save intermediate facts here
    )
    print(f"Successfully created QuestionDocument with {len(question_doc_semantic.questions)} questions.")
    print(f"Saved to: {question_doc_semantic.path_to_store}")
    print(f"Intermediate facts saved to: {fact_storage_directory}")

except Exception as e:
    print(f"An error occurred during semantic question creation: {e}")


# --- Example 3: Using alternative prompts (requires Prompt model import) ---
# (Make sure you have defined custom prompts if using this example)
# custom_fact_prompt_content = "Break down the text into distinct, verifiable facts. Format as a list."
# from knowornot.common.models import Prompt

# custom_fact_prompt = Prompt(identifier="custom-facts", content=custom_fact_prompt_content)

# custom_question_prompt_content = "From the facts provided, generate simple, clear questions and their answers."
# custom_question_prompt = Prompt(identifier="custom-questions", content=custom_question_prompt_content)

# print("\nGenerating questions with alternative prompts...")
# try:
#     question_doc_custom_prompts = kon.create_questions(
#         source_paths=source_files,
#         knowledge_base_identifier="company-overview-custom-prompts",
#         context_prompt="Based on the following, provide questions.",
#         path_to_save_questions=questions_output_path.with_stem("company_overview_kb_custom_prompts"),
#         filter_method="keyword", # Example using keyword filtering
#         alternative_fact_creation_llm_prompt=custom_fact_prompt, # Use custom fact prompt
#         alternative_question_creation_llm_prompt=custom_question_prompt # Use custom question prompt
#         # Optional: Could also specify alternative clients/models here
#     )
#     print(f"Successfully created QuestionDocument with {len(question_doc_custom_prompts.questions)} questions.")
#     print(f"Saved to: {question_doc_custom_prompts.path_to_store}")

# except Exception as e:
#     print(f"An error occurred during custom prompt question creation: {e}")
```
