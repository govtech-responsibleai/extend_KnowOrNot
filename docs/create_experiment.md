# `create_experiment_input`

```python
create_experiment_input(
    question_document: Union[Path, QuestionDocument],
    system_prompt: Prompt,
    experiment_type: Literal["removal", "synthetic"],
    retrieval_type: Literal["DIRECT", "BASIC_RAG", "LONG_IN_CONTEXT", "HYDE_RAG"],
    input_store_path: Path,
    output_store_path: Path,
    alternative_llm_client: Optional[SyncLLMClient] = None,
    ai_model_to_use: Optional[str] = None,
    alternative_prompt_for_hyde: Optional[Prompt] = None,
    alternative_llm_client_for_hyde: Optional[SyncLLMClient] = None,
    ai_model_for_hyde: Optional[str] = None,
) -> ExperimentInputDocument
```

This method prepares and defines an experiment by creating an `ExperimentInputDocument`. This document specifies *exactly* what prompt and context will be sent to the LLM for each question when the experiment is executed later using `run_experiment_sync`.

It takes the generated questions (`QuestionDocument`) and combines them with the experiment's rules (system prompt, experiment type, retrieval strategy) to pre-calculate the input needed for each individual LLM call. The resulting `ExperimentInputDocument` is saved to disk, ready to be used by the execution method.

### Parameters

*   **`question_document`** (`Union[Path, QuestionDocument]`):
    *   **Required.**
    *   The source of the question/answer pairs for the experiment. Can be a `pathlib.Path` to a previously saved `QuestionDocument` JSON file, or an already loaded `QuestionDocument` object.

*   **`system_prompt`** (`Prompt`):
    *   **Required.**
    *   A `Prompt` object containing the main system-level instructions for the LLM during the experiment run. This prompt will be prepended to the context and question for each individual LLM call in the experiment. *Example: "Answer based only on the context provided. If the context does not contain the answer, state clearly 'I do not know based on the provided context'."*

*   **`experiment_type`** (`Literal["removal", "synthetic"]`):
    *   **Required.**
    *   Defines the nature of the experiment:
        *   `"removal"`: Experiments based on questions originally derived from the knowledge base. For each question, the *specific fact* corresponding to its answer is conceptually "removed" from the available context during the retrieval step. This tests the model's ability to answer when the direct source is gone, or to abstain if it can't.
        *   `"synthetic"`: Experiments based on questions that are *not* directly answered by any single fact in the knowledge base. This tests the model's ability to abstain from answering novel questions, even when potentially misleading or irrelevant context from the knowledge base is provided.

*   **`retrieval_type`** (`Literal["DIRECT", "BASIC_RAG", "LONG_IN_CONTEXT", "HYDE_RAG"]`):
    *   **Required.**
    *   Determines *how* the context is selected for each question in the `ExperimentInputDocument`. This context is then included in the prompt sent to the LLM during the execution step (`run_experiment_sync`). The available strategies are:
        *   `"DIRECT"`: Provides *no* context (`context_questions` is `None`). The LLM is expected to answer based solely on its internal training data and the `system_prompt`.
        *   `"BASIC_RAG"`: Uses basic semantic similarity (embeddings) to find the `closest_k` (`5` by default) question-answer pairs from the knowledge base to use as context, excluding the removed question in the case of removal experiments
        *   `"LONG_IN_CONTEXT"`: Provides *all* question-answer pairs from the knowledge base as context, excluding the removed question in the case of removal experiments, up to the LLM's context window limits (though this implementation currently passes all).
        *   `"HYDE_RAG"`: Generates a hypothetical answer for the question using an LLM, embeds the hypothetical answer, and then uses the embedding to find the `closest_k` (`5` by default) question-answer pairs from the knowledge base excluding the removed question in the case of removal experiments. This aims to improve retrieval relevance.

*   **`input_store_path`** (`Path`):
    *   **Required.**
    *   The `pathlib.Path` where the generated `ExperimentInputDocument` JSON file will be saved. This file contains all the parameters and the pre-calculated prompts/contexts for each individual experiment question.
    *   The parent directory must exist or be creatable. The path must end with a `.json` suffix.

*   **`output_store_path`** (`Path`):
    *   **Required.**
    *   The `pathlib.Path` where the resulting `ExperimentOutputDocument` will be saved *after* the experiment is run using `run_experiment_sync`. This path is stored in the `ExperimentInputDocument`'s metadata for tracking purposes. The path must end with a `.json` suffix.

*   **`alternative_llm_client`** (`Optional[SyncLLMClient]`):
    *   Optional.
    *   A specific `SyncLLMClient` instance to record in the `ExperimentMetadata` as the client *intended to run this experiment*. If not provided, the default client registered with the `KnowOrNot` instance will be used and recorded. This client will be used by `run_experiment_sync` to execute the LLM calls.

*   **`ai_model_to_use`** (`Optional[str]`):
    *   Optional.
    *   A specific AI model (deployment name) to record in the `ExperimentMetadata` as the model *intended to run this experiment*. If not provided, the default model of the client specified by `alternative_llm_client` (or the `KnowOrNot` default client) will be used and recorded. This model will be used by `run_experiment_sync`.

*   **`alternative_prompt_for_hyde`** (`Optional[Prompt]`):
    *   Optional.
    *   An alternative `Prompt` object used *only* if `retrieval_type` is `"HYDE_RAG"`. This prompt guides the LLM in generating the hypothetical answer used for retrieval. If not provided, the default HYDE prompt configured in the `ExperimentManager` is used.

*   **`alternative_llm_client_for_hyde`** (`Optional[SyncLLMClient]`):
    *   Optional.
    *   A specific `SyncLLMClient` instance used *only* if `retrieval_type` is `"HYDE_RAG"`. This client is used *specifically* for generating the hypothetical answer used for retrieval. If not provided, the client specified by `alternative_llm_client` (or the `KnowOrNot` default client) is used for the HYDE step.

*   **`ai_model_for_hyde`** (`Optional[str]`):
    *   Optional.
    *   A specific AI model (deployment name) used *only* if `retrieval_type` is `"HYDE_RAG"`. This model is used *specifically* for generating the hypothetical answer. If not provided, the default model of the client specified by `alternative_llm_client_for_hyde` (or the client for the main experiment) is used for the HYDE step.

### How Experiment Type and Retrieval Type Determine Context

The combination of `experiment_type` and `retrieval_type` dictates the list of `QAPairFinal` objects included in the `context_questions` field for *each* `QAWithContext` within the output `ExperimentInputDocument`. This `context_questions` list is dynamically generated for each question based on the chosen strategy and included in the final prompt sent to the LLM during execution.

| `experiment_type` | `retrieval_type` | Context Provided (`context_questions`) for each question                                       | Purpose                                                                                                                            |
| :---------------- | :--------------- | :--------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| `"removal"`       | `"DIRECT"`       | `None`                                                                                         | Test raw LLM knowledge/abstention when the source fact is 'removed' from consideration.                                            |
| `"removal"`       | `"BASIC_RAG"`    | `closest_k` QA pairs from the KB, *excluding* the pair corresponding to the question being asked, retrieved by semantic similarity to the question. | Test if RAG helps when the exact answer isn't available, or if LLM uses similar context to hallucinate.                            |
| `"removal"`       | `"LONG_IN_CONTEXT"`| *All* QA pairs from the KB, *excluding* the pair corresponding to the question being asked.      | Test LLM's ability with potentially large amounts of context, where the direct source is missing.                                  |
| `"removal"`       | `"HYDE_RAG"`     | `closest_k` QA pairs from the KB, *excluding* the pair corresponding to the question being asked, retrieved using a hypothetical answer generated by an LLM. | Test if HYDE retrieval improves finding relevant context when the direct source is gone.                                           |
| `"synthetic"`     | `"DIRECT"`       | `None`                                                                                         | Test raw LLM abstention on novel questions with no external context.                                                               |
| `"synthetic"`     | `"BASIC_RAG"`    | `closest_k` QA pairs from the KB, retrieved by semantic similarity to the synthetic question.      | Test LLM abstention/hallucination on novel questions when basic RAG might pull in irrelevant KB context.                           |
| `"synthetic"`     | `"LONG_IN_CONTEXT"`| *All* QA pairs from the KB.                                                                    | Test LLM abstention/hallucination on novel questions with a large amount of irrelevant KB context.                               |
| `"synthetic"`     | `"HYDE_RAG"`     | `closest_k` QA pairs from the KB, retrieved using a hypothetical answer generated for the *synthetic* question. | Test LLM abstention/hallucination on novel questions when HYDE retrieval might pull in potentially misleading KB context.             |

## LLM Usage in this Step

While the primary LLM execution to get the *experiment responses* happens in `run_experiment_sync`, an LLM might be used *during* `create_experiment_input` for tasks necessary to *prepare* the context:

*   **Embeddings:** If `retrieval_type` is `"BASIC_RAG"`, or `"HYDE_RAG"`, embeddings of the Q&A pairs from the `question_document` are generated to calculate similarity. This uses the LLM client designated for the main experiment (`alternative_llm_client` or the default client registered with `KnowOrNot`). This client *must* be capable of generating embeddings.
*   **Hypothetical Answer Generation:** If `retrieval_type` is `"HYDE_RAG"`, an LLM is called to generate hypothetical answers for each question. This uses the client/model specified by `alternative_llm_client_for_hyde`/`ai_model_for_hyde`, or falls back to the client/model for the main experiment. This client *must* be capable of structured responses (often requires clients compatible with libraries like Instructor).

The LLM client and model specified by `alternative_llm_client`/`ai_model_to_use` are primarily recorded in the `ExperimentMetadata` within the `ExperimentInputDocument` and are used *later* by `run_experiment_sync` to get the *actual* responses to the generated prompts.

## Returns

*   **`ExperimentInputDocument`**: An object containing the complete definition of the experiment, including metadata (`ExperimentMetadata`) and the list of inputs for each individual question prompt (`List[IndividualExperimentInput]`). This object is also automatically saved as a JSON file to the path specified by `input_store_path`.

## Important Considerations

*   **Requires `QuestionDocument`:** This method cannot be run until you have successfully created a `QuestionDocument` using `create_questions`.
*   **Retrieval Strategy Configuration:** The `ExperimentManager` is initialized with default `closest_k=5` for RAG strategies and a default `hypothetical_answer_prompt` for HYDE. These cannot be directly configured via `create_experiment_input` parameters; they are set when the `ExperimentManager` (and thus the `KnowOrNot` instance) is initialized.
*   **Context Window Limits:** For the `"LONG_IN_CONTEXT"` strategy, the implementation provides *all* questions from the knowledge base as context. You must ensure that the total token count of the system prompt + context + question for *any* individual experiment input does not exceed the context window of the LLM you plan to use in `run_experiment_sync`. Exceeding the context window will likely cause errors during experiment execution.
*   **LLM Capability:** The client used for embeddings (the main experiment client) must support embedding calls. The client used for HYDE hypothetical answers (potentially a different client) must support structured responses (`can_use_instructor=True`). Ensure your registered clients meet these requirements for the strategies you intend to use.
*   **Output Tracking:** The `output_store_path` parameter is primarily for metadata tracking within the input document itself, indicating where the results *should* be saved. The actual output file is created and saved by the `run_experiment_sync` method.

## Examples

Assuming you have initialized a `KnowOrNot` instance and created or loaded a `QuestionDocument` (see [Quick Start - Initialization and Azure Client Setup](#quick-start-initialization-and-azure-client-setup) and [`create_questions` documentation](#create_questions)).

```python
from knowornot import KnowOrNot
from knowornot.common.models import Prompt, QuestionDocument
from knowornot.SyncLLMClient import SyncLLMClientEnum # To reference client enums if needed
from pathlib import Path

# Assume 'kon' is your initialized KnowOrNot instance with a default client
# kon = KnowOrNot()
# kon.add_azure(...) # or other client setup

from knowornot.common.models import Prompt

# Example loading a previously saved QuestionDocument
try:
    question_document_path = Path("data/generated_questions/company_overview_kb_default.json")
    question_document = QuestionDocument.load_from_json(question_document_path)
    print(f"Loaded QuestionDocument from {question_document_path}")
except Exception as e:
    print(f"Could not load QuestionDocument: {e}")
    # In a real script, you might raise the exception or exit here

# Define the system prompt that sets the behavior for the LLM
experiment_system_prompt = Prompt(
    identifier="abstention-prompt",
    content="Carefully consider the following context to answer the question. If the context does not contain the answer, state clearly 'I do not know based on the provided context'."
)

# Define paths where the experiment definition (input) and results (output) will be stored
experiment_input_path = Path("data/experiments/company_overview_exp_basic_rag_removal_input.json")
experiment_output_path = Path("data/experiments/company_overview_exp_basic_rag_removal_output.json")

# Ensure the input directory exists before saving the input document
experiment_input_path.parent.mkdir(parents=True, exist_ok=True)


# --- Example 1: Create a Basic RAG Removal Experiment ---
print("\nCreating a Basic RAG Removal experiment...")
try:
    experiment_input_doc = kon.create_experiment_input(
        question_document=question_document,
        system_prompt=experiment_system_prompt,
        experiment_type="removal",      # Test questions where the direct source is removed
        retrieval_type="BASIC_RAG",     # Use semantic similarity for context retrieval
        input_store_path=experiment_input_path,
        output_store_path=experiment_output_path
        # Uses the default LLM client and model configured in the KnowOrNot instance
        # for embeddings (in this step) and execution (in the next step run_experiment_sync)
    )

    print(f"Successfully created ExperimentInputDocument.")
    print(f"Metadata: {experiment_input_doc.metadata}")
    print(f"Contains {len(experiment_input_doc.questions)} individual experiment inputs.")
    print(f"Saved experiment input definition to: {experiment_input_doc.metadata.input_path}")
    print(f"Experiment output will be saved to: {experiment_input_doc.metadata.output_path}")


except Exception as e:
    print(f"An error occurred while creating the experiment input: {e}")


# --- Example 2: Create a HYDE RAG Synthetic Experiment ---
# This tests how the LLM responds to questions NOT in the KB when HYDE-retrieved context is provided.
# (Requires that the client used for HYDE can generate structured responses/use Instructor)

# Define paths for this specific experiment
hyde_synthetic_input_path = Path("data/experiments/company_overview_exp_hyde_synthetic_input.json")
hyde_synthetic_output_path = Path("data/experiments/company_overview_exp_hyde_synthetic_output.json")
hyde_synthetic_input_path.parent.mkdir(parents=True, exist_ok=True)

# Define a custom prompt for HYDE
# from knowornot.common.models import Prompt
# hypothetical answer generation (optional, overrides default)
# custom_hyde_prompt = Prompt(
#     identifier="hyde-generator",
#     content="Generate a plausible hypothetical answer to the following question, formatted as a list of likely answers:"
# )

print("\nCreating a HYDE RAG Synthetic experiment...")
try:
    experiment_input_hyde_synthetic = kon.create_experiment_input(
        question_document=question_document,
        system_prompt=experiment_system_prompt, # Use the same system prompt for the experiment execution
        experiment_type="synthetic",    # Test novel questions
        retrieval_type="HYDE_RAG",      # Use hypothetical answers for context retrieval preparation
        input_store_path=hyde_synthetic_input_path,
        output_store_path=hyde_synthetic_output_path,
        # Optional: Specify alternative LLM/model just for the HYDE step within this method
        # alternative_prompt_for_hyde=custom_hyde_prompt,
        # alternative_llm_client_for_hyde=kon.get_client(SyncLLMClientEnum.AZURE_OPENAI), # Example: specify a client by enum
        # ai_model_for_hyde="gpt-4o-hypothetical-deployment" # Example: specify a model just for HYDE
    )

    print(f"Successfully created ExperimentInputDocument for HYDE Synthetic.")
    print(f"Saved experiment input definition to: {experiment_input_hyde_synthetic.metadata.input_path}")
    print(f"Experiment output will be saved to: {experiment_input_hyde_synthetic.metadata.output_path}")

except Exception as e:
    print(f"An error occurred while creating the HYDE experiment input: {e}")
```
