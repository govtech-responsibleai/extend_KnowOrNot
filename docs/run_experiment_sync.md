## run_experiment_sync

```python
run_experiment_sync(
    experiment_input: ExperimentInputDocument,
) -> ExperimentOutputDocument
```

This method executes an experiment synchronously based on the configuration defined in a previously created `ExperimentInputDocument`. It iterates through each question defined in the input document, sends the pre-calculated prompt and context to the specified LLM, and records the LLM's response, including any detected citations.

This is the step where the actual LLM calls for the experiment are made.

### Parameters

*   **`experiment_input`** (`ExperimentInputDocument`):
    *   **Required.**
    *   An `ExperimentInputDocument` object, typically loaded from a JSON file created by the `create_experiment_input` method. This document contains all the necessary information to run the experiment, including the list of prompts and context for each question, and metadata specifying which LLM client and model to use.

### Process

1.  The method retrieves the `ExperimentInputDocument`'s metadata to identify the intended LLM client (`client_enum`) and model (`ai_model_used`) for this experiment.
2.  It fetches the corresponding `SyncLLMClient` instance from the `KnowOrNot` instance's internal registry using the `client_enum` from the metadata. **The client specified in the `ExperimentInputDocument` metadata must be registered with the `KnowOrNot` instance *before* calling this method.**
3.  It iterates through the list of `IndividualExperimentInput` objects contained within `experiment_input.questions`.
4.  For each `IndividualExperimentInput`, it sends the `prompt_to_llm` (which includes the system prompt, question, and generated context) to the retrieved `SyncLLMClient` using the specified `ai_model_used`.
5.  The LLM's response is expected to conform to the `QAResponse` model, which includes the text response and a citation index (or "no citation").
6.  If a citation index is returned and is valid within the provided context questions, the corresponding `QAPairFinal` from the context is recorded.
7.  Each response, along with its corresponding input and any cited context, is stored as a `SavedLLMResponse`.
8.  All `SavedLLMResponse` objects are collected.
9.  An `ExperimentOutputDocument` is created, containing the original `ExperimentInputDocument`'s metadata and the list of `SavedLLMResponse` results.
10. The `ExperimentOutputDocument` is automatically saved as a JSON file to the path specified in the original experiment input's metadata (`experiment_input.metadata.output_path`).

### Returns

*   **`ExperimentOutputDocument`**: An object containing the results of the experiment. This includes the original experiment metadata and a list of `SavedLLMResponse` objects, where each object pairs an individual experiment input (prompt + context) with the LLM's response and any cited context. This document is also automatically saved as a JSON file to the path specified in the experiment input metadata.

### Important Considerations

*   **LLM Calls and Cost:** This method performs one LLM call per question defined in the `ExperimentInputDocument`. Be aware that running experiments can incur significant costs depending on the number of questions, the LLM model used, and the size of the context included in the prompts.
*   **Duration:** Synchronous execution means the method will block until all LLM calls are completed. This can take a considerable amount of time for large numbers of questions.
*   **Client Availability:** The `SyncLLMClient` specified by the `client_enum` in the `ExperimentInputDocument`'s metadata *must* be registered with the `KnowOrNot` instance using `register_client` *before* calling `run_experiment_sync`. If the client is not registered, a `ValueError` will be raised.
*   **Model Availability:** The `ai_model_used` specified in the metadata must be a valid deployment name accessible by the chosen client.
*   **Output Path:** The output is automatically saved to the path specified during the `create_experiment_input` step (`ExperimentInputDocument.metadata.output_path`). Ensure this path is valid and the directory exists.
*   **Citation Format:** Correct citation extraction relies on the LLM's ability to follow the prompt instructions and return a citation index in the format expected by the `QAResponse` model. Evaluation (using `evaluate_experiment`) will analyze whether citations were made and if they were correct.

### Examples

Assuming you have initialized a `KnowOrNot` instance, added necessary LLM clients, and created or loaded an `ExperimentInputDocument` (see [Quick Start](#quick-start-initialization-and-azure-client-setup) and [`create_experiment_input` documentation](#create_experiment_input)).

```python
from knowornot import KnowOrNot
from knowornot.common.models import ExperimentInputDocument
from pathlib import Path

# Assume 'kon' is your initialized KnowOrNot instance with the necessary client(s) registered
# kon = KnowOrNot()
# kon.add_azure(...) # Example: Ensure Azure client is registered if used in the input document

# Example loading a previously created ExperimentInputDocument
try:
    experiment_input_path = Path("data/experiments/company_overview_exp_basic_rag_removal_input.json")
    experiment_input_doc = ExperimentInputDocument.load_from_json(experiment_input_path)
    print(f"Loaded ExperimentInputDocument from {experiment_input_path}")
    print(f"Ready to run experiment '{experiment_input_doc.metadata.experiment_type.value}' with retrieval '{experiment_input_doc.metadata.retrieval_type.value}'")
    print(f"Using client: {experiment_input_doc.metadata.client_enum.name}, Model: {experiment_input_doc.metadata.ai_model_used}")
    print(f"Expecting output at: {experiment_input_doc.metadata.output_path}")

except FileNotFoundError:
    print(f"Error: Experiment input file not found at {experiment_input_path}.")
    print("Please run create_experiment_input first.")
    # Exit or handle error appropriately
    exit()
except Exception as e:
    print(f"An error occurred loading the experiment input: {e}")
    exit()


# --- Run the Experiment ---
print("\nRunning the experiment synchronously...")
try:
    experiment_output_doc = kon.run_experiment_sync(
        experiment_input=experiment_input_doc
    )

    print("Experiment run complete.")
    print(f"Successfully saved ExperimentOutputDocument to: {experiment_output_doc.metadata.output_path}")
    print(f"Experiment results contain responses for {len(experiment_output_doc.responses)} questions.")

    # You can now load experiment_output_doc.metadata.output_path
    # and use it with evaluate_experiment

except ValueError as e:
    print(f"Error running experiment: {e}")
    print("This might be because the client specified in the experiment input metadata is not registered.")
except Exception as e:
    print(f"An unexpected error occurred during experiment execution: {e}")

```
