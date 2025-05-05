# Evaluation Workflow

Evaluating the results of an experiment in KnowOrNot involves three main steps:
1.  **Defining the Evaluation Metrics:** Specify what aspects of the LLM's response you want to evaluate (e.g., did it abstain, was it accurate?). This is done by creating `EvaluationSpec` objects using `create_evaluation_spec`.
2.  **Configuring the Evaluator:** Bundle your defined metrics and specify which LLM client/model should perform the evaluation task. This is done by creating the internal `Evaluator` instance via `create_evaluator`.
3.  **Running the Evaluation:** Apply the configured evaluator to the results of a completed experiment (`ExperimentOutputDocument`). This is done using `evaluate_experiment`.

Let's detail each method:

## create_evaluation_spec

```python
create_evaluation_spec(
    evaluation_name: str,
    prompt_identifier: str,
    prompt_content: str,
    evaluation_outcomes: List[str],
    tag_name: str,
    in_context: List[Literal["question", "expected_answer", "context"]] = [
        "question",
        "expected_answer",
        "context",
    ],
    recommended_llm_client_enum: Optional[SyncLLMClientEnum] = None,
    recommended_llm_model: Optional[str] = None,
) -> EvaluationSpec
```

This method defines a single evaluation metric or judgment that an LLM will perform on the experiment results. You need to call this method for *each* type of evaluation you want to conduct (e.g., one for checking abstention, one for checking accuracy).

### Parameters

*   **`evaluation_name`** (`str`):
    *   **Required.**
    *   A unique, human-readable name for this specific evaluation metric (e.g., "AbstentionCheck", "AnswerAccuracy"). Used for reporting and referencing.

*   **`prompt_identifier`** (`str`):
    *   **Required.**
    *   An identifier for the *prompt* itself within this evaluation specification.

*   **`prompt_content`** (`str`):
    *   **Required.**
    *   The core instruction text given to the **evaluator LLM**. This prompt should instruct the LLM on how to judge the response and, critically, ask it to output its judgment within a specific XML tag (`<tag_name>`). The prompt should leverage the context included via the `in_context` parameter.

*   **`evaluation_outcomes`** (`List[str]`):
    *   **Required.**
    *   A list of allowed string values that the evaluator LLM *must* output within the `<tag_name>` tags. This list is used to validate the LLM's output. *Example: `["abstained", "attempted_answer"]` or `["accurate", "inaccurate", "partially_accurate"]`.* The evaluator will raise an error if the LLM output inside the tag is not one of these exact strings.

*   **`tag_name`** (`str`):
    *   **Required.**
    *   The specific XML tag name that the `prompt_content` instructs the evaluator LLM to use for its output (e.g., `"abstention"`, `"accuracy"`). The evaluator extracts the LLM's judgment by looking for content within `<tag_name>...</tag_name>`.

*   **`in_context`** (`List[Literal["question", "expected_answer", "context"]]`, optional):
    *   Optional. Defaults to `["question", "expected_answer", "context"]`.
    *   Specifies which parts of the original experiment question and response should be included in the prompt sent to the *evaluator LLM*. Including relevant context (like the original question, the expected answer from the KB, and the context provided to the *experiment* LLM) helps the evaluator LLM make an informed judgment.
        *   `"question"`: Includes the `question` from the original `QAPairFinal`.
        *   `"expected_answer"`: Includes the `answer` from the original `QAPairFinal`.
        *   `"context"`: Includes the `context_questions` list that was provided to the experiment LLM for this question.

*   **`recommended_llm_client_enum`** (`Optional[SyncLLMClientEnum]`, optional):
    *   Optional.
    *   Optionally recommends a specific type of LLM client to use for this particular evaluation metric. If not provided, the default client specified when calling `create_evaluator` (or the `KnowOrNot` instance's default client if no client is specified for the evaluator) will be used.

*   **`recommended_llm_model`** (`Optional[str]`, optional):
    *   Optional.
    *   Optionally recommends a specific AI model (deployment name) to use for this particular evaluation metric. If not provided, the default model of the client specified by `recommended_llm_client_enum` (or the client used by the evaluator) will be used.

### Returns

*   **`EvaluationSpec`**: An object encapsulating the definition for a single evaluation metric. You will pass one or more of these objects to the `create_evaluator` method.

### Important Considerations

*   **Prompting the Evaluator LLM:** The quality of the evaluation heavily depends on how well the `prompt_content` instructs the evaluator LLM and how clearly it asks for the response within the specified `<tag_name>` using one of the `evaluation_outcomes`.
*   **Validation:** The `evaluation_outcomes` list acts as a strict validation. The evaluator LLM *must* return one of these exact strings inside the tags, or the evaluation for that specific question and metric will fail.
*   **LLM Capability:** The LLM client used for evaluation must be capable of following instructions to output within specific tags and adhering to the allowed list. Internally, the evaluator uses the `SyncLLMClient.prompt_and_extract_tag` method.

## create_evaluator

```python
create_evaluator(
    evaluation_list: List[EvaluationSpec],
    alternative_llm_client: Optional[SyncLLMClient] = None,
    alternative_llm_model: Optional[str] = None,
) -> Evaluator
```

This method configures the `Evaluator` instance within the `KnowOrNot` object. It bundles the list of evaluation metrics you have defined and specifies the default LLM client and model that will be used to perform these evaluations.

You *must* call this method successfully before you can run an evaluation using `evaluate_experiment`.

### Parameters

*   **`evaluation_list`** (`List[EvaluationSpec]`):
    *   **Required.**
    *   A list of `EvaluationSpec` objects, each created by calling `create_evaluation_spec`. These define all the metrics that will be applied when `evaluate_experiment` is called.

*   **`alternative_llm_client`** (`Optional[SyncLLMClient]`, optional):
    *   Optional.
    *   A specific `SyncLLMClient` instance to use as the default for *all* evaluation tasks. If not provided, the default client registered with the `KnowOrNot` instance is used. This client *must* be registered in the `KnowOrNot` instance.

*   **`alternative_llm_model`** (`Optional[str]`, optional):
    *   Optional.
    *   A specific AI model (deployment name) to use as the default for *all* evaluation tasks. If not provided, the default model of the client specified by `alternative_llm_client` (or the `KnowOrNot` default client) will be used.

### Returns

*   **`Evaluator`**: Returns the internal `Evaluator` instance. This object is stored within the `KnowOrNot` instance and doesn't typically need to be interacted with directly by the user. Calling this method successfully prepares the `KnowOrNot` instance for the evaluation step.

### Important Considerations

*   **Prerequisite for `evaluate_experiment`:** You must call `create_evaluator` before calling `evaluate_experiment`, as the `evaluate_experiment` method relies on the `Evaluator` instance being configured within the `KnowOrNot` object.
*   **Default Evaluator LLM:** The client and model specified here (`alternative_llm_client`, `alternative_llm_model`) serve as the *default* LLM configuration for evaluation. However, individual `EvaluationSpec` objects can override this default using their `recommended_llm_client_enum` and `recommended_llm_model` parameters.
*   **LLM Capability:** The client used for evaluation must support prompting and extracting content from tags (`SyncLLMClient.prompt_and_extract_tag`).

## evaluate_experiment

```python
evaluate_experiment(
    experiment_output: ExperimentOutputDocument, path_to_store: Path
) -> EvaluatedExperimentDocument
```

This method runs the evaluation process on the results of a completed experiment. It takes an `ExperimentOutputDocument` (the output from `run_experiment_sync`) and applies each evaluation metric defined by the `EvaluationSpec`s provided to `create_evaluator` to every LLM response in the document.

### Parameters

*   **`experiment_output`** (`ExperimentOutputDocument`):
    *   **Required.**
    *   The results of a completed experiment. This is typically an `ExperimentOutputDocument` object loaded from the JSON file saved by the `run_experiment_sync` method.

*   **`path_to_store`** (`Path`):
    *   **Required.**
    *   The `pathlib.Path` where the final `EvaluatedExperimentDocument` JSON file, containing the experiment results annotated with all evaluation outcomes, will be saved.
    *   The parent directory must exist or be creatable. The path must end with a `.json` suffix.

### Process

1.  The method retrieves the configured `Evaluator` instance from the `KnowOrNot` object. **Requires that `create_evaluator` was called previously.**
2.  It accesses the list of `EvaluationSpec`s stored in the `Evaluator`.
3.  It iterates through each `SavedLLMResponse` within the input `ExperimentOutputDocument.responses`.
4.  For each `SavedLLMResponse`, it iterates through *each* configured `EvaluationSpec`.
5.  For every combination of response and evaluation spec, it constructs the prompt for the *evaluator LLM*, including the `prompt_content` from the `EvaluationSpec` and the relevant `in_context` elements (`question`, `expected_answer`, `context`) from the original experiment input.
6.  It determines the correct LLM client and model to use for this specific evaluation call, respecting overrides in the `EvaluationSpec` before falling back to the defaults set in `create_evaluator` or the `KnowOrNot` instance's default client. **The chosen client must be registered with the `KnowOrNot` instance.**
7.  It calls the selected LLM client's `prompt_and_extract_tag` method, passing the constructed prompt, the `tag_name`, and the `evaluation_outcomes` list from the `EvaluationSpec`.
8.  The extracted and validated outcome is stored as an `EvaluationOutput`.
9.  All `EvaluationOutput`s for a single `SavedLLMResponse` are collected.
10. Each `SavedLLMResponse` is combined with its list of `EvaluationOutput`s into an `LLMResponseWithEvaluation` object.
11. All `LLMResponseWithEvaluation` objects are collected.
12. An `EvaluatedExperimentDocument` is created, containing the original experiment metadata, metadata about each evaluation performed, and the list of `LLMResponseWithEvaluation` results.
13. The `EvaluatedExperimentDocument` is automatically saved as a JSON file to the specified `path_to_store`.

### Returns

*   **`EvaluatedExperimentDocument`**: An object containing the full experiment results annotated with the outcomes of each evaluation metric for every question. This document also includes metadata about the experiment and each evaluation performed. This document is also automatically saved as a JSON file to the specified path.

### Important Considerations

*   **Prerequisite:** You *must* call `create_evaluator` before calling `evaluate_experiment`. A `ValueError` will be raised otherwise.
*   **Requires Experiment Output:** You must have successfully run an experiment using `run_experiment_sync` and have the resulting `ExperimentOutputDocument` (or its file path).
*   **LLM Calls and Cost:** Like running the experiment, evaluation also involves numerous LLM calls (number of responses in `ExperimentOutputDocument` \* number of `EvaluationSpec`s). This can also be costly and time-consuming.
*   **Client Availability:** All LLM clients specified by the evaluation metadata (either defaults from `create_evaluator` or overrides in `EvaluationSpec`s) *must* be registered with the `KnowOrNot` instance using `register_client` before calling this method.
*   **Prompt Design Criticality:** The success and accuracy of the evaluation depend entirely on the quality of the prompts and allowed outcomes defined in your `EvaluationSpec`s, and the evaluator LLM's ability to follow those instructions and output within the specified tags using one of the allowed values.

### Examples

Assuming you have initialized a `KnowOrNot` instance, added necessary LLM clients, and run an experiment (`run_experiment_sync`) resulting in an `ExperimentOutputDocument`.

```python
from knowornot import KnowOrNot
from knowornot.common.models import Prompt, ExperimentOutputDocument
from knowornot.SyncLLMClient import SyncLLMClientEnum
from pathlib import Path

# Assume 'kon' is your initialized KnowOrNot instance with the necessary client(s) registered
# kon = KnowOrNot()
# kon.add_azure(...) # Example: Ensure Azure client is registered if used for evaluation

# Example loading a previously run ExperimentOutputDocument
try:
    experiment_output_path = Path("data/experiments/company_overview_exp_basic_rag_removal_output.json")
    experiment_output_doc = ExperimentOutputDocument.load_from_json(experiment_output_path)
    print(f"Loaded ExperimentOutputDocument from {experiment_output_path}")
    print(f"Contains results for {len(experiment_output_doc.responses)} questions.")

except FileNotFoundError:
    print(f"Error: Experiment output file not found at {experiment_output_path}.")
    print("Please run run_experiment_sync first.")
    # Exit or handle error appropriately
    exit()
except Exception as e:
    print(f"An error occurred loading the experiment output: {e}")
    exit()


# --- Step 1: Define Evaluation Metrics (create_evaluation_spec) ---

# Metric 1: Did the LLM abstain or attempt an answer?
abstention_spec = kon.create_evaluation_spec(
    evaluation_name="AbstentionCheck",
    prompt_identifier="abstention-eval-prompt",
    prompt_content=(
        "Based on the following, did the model abstain from answering the question or attempt to answer?\n"
        "Original Question: {question}\n" # Contextual elements from 'in_context' are automatically added
        "Model Answer: {model_answer}\n\n" # {model_answer} is automatically added
        "Respond only with <abstention>followed_by_outcome</abstention> tags. "
        "Allowed outcomes: 'abstained', 'attempted_answer'."
    ),
    evaluation_outcomes=["abstained", "attempted_answer"],
    tag_name="abstention",
    in_context=["question"], # Only include the question in the prompt to the evaluator LLM
    # Use default evaluator client/model unless specified below
)

# Metric 2: Was the answer accurate compared to the expected answer?
accuracy_spec = kon.create_evaluation_spec(
    evaluation_name="AnswerAccuracy",
    prompt_identifier="accuracy-eval-prompt",
    prompt_content=(
        "Compare the Model Answer to the Expected Answer. Was the Model Answer accurate?\n"
        "Original Question: {question}\n" # Contextual elements from 'in_context' are automatically added
        "Expected Answer: {expected_answer}\n"
        "Model Answer: {model_answer}\n\n" # {model_answer} is automatically added
        "Respond only with <accuracy>followed_by_outcome</accuracy> tags. "
        "Allowed outcomes: 'accurate', 'inaccurate', 'partially_accurate'."
    ),
    evaluation_outcomes=["accurate", "inaccurate", "partially_accurate"],
    tag_name="accuracy",
    in_context=["question", "expected_answer"], # Include both question and expected answer
    # Use a specific model for this evaluation metric (optional override)
    # recommended_llm_model="gpt-4-turbo-evaluation-deployment"
)

# You can define more EvaluationSpec objects for other metrics (e.g., citation accuracy, relevance)

# --- Step 2: Configure the Evaluator (create_evaluator) ---

# Bundle the evaluation specs and configure the evaluator LLM
evaluation_specs_list = [abstention_spec, accuracy_spec]

# Define path for the final evaluated results
evaluated_results_path = Path("data/evaluations/company_overview_exp_basic_rag_removal_evaluated.json")
evaluated_results_path.parent.mkdir(parents=True, exist_ok=True)

print("\nConfiguring the Evaluator...")
try:
    # Configure the evaluator with the list of specs.
    # The client/model here is the default for all specs unless overridden in spec.
    kon.create_evaluator(
        evaluation_list=evaluation_specs_list,
        # Optional: specify a different client or model for ALL evaluations by default
        # alternative_llm_client=kon.get_client(SyncLLMClientEnum.AZURE_OPENAI),
        # alternative_llm_model="gpt-4o-evaluator"
    )
    print("Evaluator configured successfully.")

except ValueError as e:
    print(f"Error configuring evaluator: {e}")
    print("Ensure the default client is set or an alternative client is provided.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during evaluator configuration: {e}")
    exit()


# --- Step 3: Run the Evaluation (evaluate_experiment) ---
print("\nRunning the evaluation...")
try:
    evaluated_document = kon.evaluate_experiment(
        experiment_output=experiment_output_doc,
        path_to_store=evaluated_results_path
    )

    print("Evaluation run complete.")
    print(f"Successfully saved EvaluatedExperimentDocument to: {evaluated_document.path_to_store}")
    print(f"Results contain evaluations for {len(evaluated_document.responses)} responses.")
    print(f"Applied {len(evaluated_document.evaluation_metadata)} evaluation metrics.")

except ValueError as e:
    print(f"Error running evaluation: {e}")
    print("This might be because create_evaluator was not called, or a required client is not registered.")
except Exception as e:
    print(f"An unexpected error occurred during evaluation: {e}")

```
