## An End-to-End Walkthrough of `KnowOrNot`: Implementing the OOKB Evaluation Pipeline

This document provides a practical, step-by-step guide to using the `KnowOrNot` library, as introduced in our accompanying paper, to implement the proposed methodology for evaluating Out-Of-Knowledge-Base (OOKB) robustness in RAG systems. It walks through a complete pipeline, from preparing source data to running experiments and evaluating the results, demonstrating the library's unified API, modular design, and focus on reproducibility.

The goal is to show you how you can take a set of text documents, automatically generate diverse questions from them, design and run experiments that simulate OOKB scenarios under various RAG configurations, and automatically evaluate the target LLM's responses – especially its ability to correctly abstain when the necessary information is missing.

Let's get started.

First, ensure you have the library installed and necessary dependencies like `python-dotenv`:

```bash
pip install knowornot
```

You'll also need to set up environment variables for at least one LLM provider (e.g., `OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `GEMINI_API_KEY`, etc.) or create a `.env` file in your project root.

The entire process is orchestrated through the main `KnowOrNot` class:

```python
import asyncio
import os
from pathlib import Path
from typing import List
from knowornot import KnowOrNot
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(asctime)s.%(msecs)03d %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

async def run_knowornot_pipeline(text_files: List[Path]):
    """
    Run a complete KnowOrNot pipeline from a list of text files using async methods.

    Args:
        text_files: List of Path objects pointing to .txt files to analyze
    """
    logger.info("Initializing KnowOrNot pipeline...")
    # 1. Initialize KnowOrNot
    kon = KnowOrNot()
    load_dotenv() # Load environment variables (e.g., API keys)

    # ... rest of the pipeline will go here ...
```

### Step 1: Setting Up LLM Clients

LLMs are used throughout the pipeline (for generating questions, running the experiments, and evaluating responses). You need to register at least one LLM client with the `KnowOrNot` instance. The library supports clients for various providers, configured using environment variables.

Choose one of the following options to add your preferred client:

**Option 1: Add OpenAI Client**

```python
    # Option 1: Add OpenAI Client
    # Requires OPENAI_API_KEY, OPENAI_DEFAULT_MODEL, OPENAI_DEFAULT_EMBEDDING_MODEL in environment/dotenv
    try:
        kon.add_openai()
        logger.info("OpenAI client added and set as default.")
    except EnvironmentError as e:
         logger.error(f"Failed to add OpenAI client: {e}. Please ensure required environment variables are set.")
         return # Exit if client setup fails
```

**Option 2: Add Azure OpenAI Client**

```python
    # Option 2: Add Azure OpenAI Client
    # Requires AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
    # AZURE_OPENAI_DEFAULT_MODEL, AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL in environment/dotenv
    try:
        kon.add_azure()
        logger.info("Azure OpenAI client added and set as default.")
    except EnvironmentError as e:
         logger.error(f"Failed to add Azure OpenAI client: {e}. Please ensure required environment variables are set.")
         return # Exit if client setup fails
```

**Option 3: Add Gemini Client**

```python
    # Option 3: Add Gemini Client
    # Requires GEMINI_API_KEY, GEMINI_DEFAULT_MODEL, GEMINI_DEFAULT_EMBEDDING_MODEL in environment/dotenv
    try:
        kon.add_gemini()
        logger.info("Gemini client added and set as default.")
    except EnvironmentError as e:
         logger.error(f"Failed to add Gemini client: {e}. Please ensure required environment variables are set.")
         return # Exit if client setup fails
```

**Option 4: Add Groq Client**

```python
    # Option 4: Add Groq Client
    # Requires GROQ_API_KEY and GROQ_DEFAULT_MODEL in environment/dotenv
    try:
        kon.add_groq()
        logger.info("Groq client added and set as default.")
    except EnvironmentError as e:
         logger.error(f"Failed to add Groq client: {e}. Please ensure required environment variables are set.")
         return # Exit if client setup fails
```

**Option 5: Add Anthropic Client**

```python
    # Option 5: Add Anthropic Client
    # Requires ANTHROPIC_API_KEY and ANTHROPIC_DEFAULT_MODEL in environment/dotenv
    try:
        kon.add_anthropic()
        logger.info("Anthropic client added and set as default.")
    except EnvironmentError as e:
         logger.error(f"Failed to add Anthropic client: {e}. Please ensure required environment variables are set.")
         return # Exit if client setup fails
```

**Option 6: Add Bedrock Client**

```python
    # Option 6: Add Bedrock Client
    # Requires AWS_BEARER_TOKEN_BEDROCK and BEDROCK_DEFAULT_MODEL in environment/dotenv
    try:
        kon.add_bedrock()
        logger.info("Bedrock client added and set as default.")
    except EnvironmentError as e:
         logger.error(f"Failed to add Bedrock client: {e}. Please ensure required environment variables are set.")
         return # Exit if client setup fails
```

For this guide, we'll assume you've added an OpenAI client, but the process is identical for others. The first client added becomes the default, used for operations unless a specific client is explicitly requested.

### Step 2: Preparing Source Data

The starting point for creating our OOKB benchmark is a set of text documents that represent your knowledge base.

```python
    # Example usage:
    # Replace with your actual text files.
    # Example files space_1.txt and space_2.txt can be found in the 'example/' directory
    # of the KnowOrNot GitHub repository: https://github.com/govtech-responsibleai/KnowOrNot
    my_text_files = [
        Path("example/space_1.txt"),
        Path("example/space_2.txt"),
        # Add more paths as needed
    ]

    # Ensure example files exist for the quickstart
    for f_path in my_text_files:
        if not f_path.exists():
            logger.error(f"Error: Example file not found at {f_path}. Please clone the repository.")
            return # Exit if files are missing

    # 3. Create a unique identifier for this knowledge base
    kb_identifier = "quickstart_knowledge_base"
    logger.info(f"Using Knowledge Base Identifier: {kb_identifier}")

    # 4. Set up paths for outputs
    base_dir = Path("kon_outputs")
    base_dir.mkdir(exist_ok=True, parents=True)
    questions_path = base_dir / f"{kb_identifier}_questions.json" # Include KB ID in filename
```

We define a list of `Path` objects pointing to your source `.txt` files. These files constitute the knowledge base from which test questions will be derived. We also set up a unique identifier for this KB and define output directories to keep results organized.

### Step 3: Generating Diverse Questions

Next, we turn the raw text into structured, verifiable Question-Answer pairs.

```python
    # 5. Create questions from the text files
    # This step involves LLM-assisted fact extraction and Q&A generation,
    # followed by diversity filtering as described in the paper's methodology.
    logger.info(f"Creating questions from source documents: {text_files}")
    context_prompt = "Answer questions based solely on the provided information." # A prompt used during Q&A generation
    question_doc = kon.create_questions(
        source_paths=text_files,
        knowledge_base_identifier=kb_identifier,
        context_prompt=context_prompt,
        path_to_save_questions=questions_path,
        filter_method="both",  # Apply both keyword and semantic diversity filtering
    )

    logger.info(f"Created {len(question_doc.questions)} diverse questions. Saved to {questions_path}")
```

The `kon.create_questions()` method processes the source files. It first uses an LLM (via the registered client) to break down the text into atomic facts. Then, for each fact, it generates a corresponding question and answer, ensuring the answer is *directly grounded* in that single fact. Finally, it applies diversity filters (`filter_method="both"`) to ensure the generated QA pairs are informationally distinct from one another, as detailed in the paper. The output is a `QuestionDocument` object containing the curated QA pairs, saved to the specified path.

### Step 4: Designing the Out-of-Knowledge Base Experiments

With the diverse questions ready, we define the specific experiment configurations we want to run.

```python
    # 6. Set up experiment directories
    experiment_dir = base_dir / "experiments"
    experiment_dir.mkdir(exist_ok=True, parents=True)

    # 7. Create all experiment variations
    # This step defines the specific experiment configurations (Prompt, Retrieval Strategy, Target LLM Model)
    # for each question generated in Step 5. The "removal" type implements the core OOKB
    # leave-one-out setup described in the paper.
    logger.info("Creating all experiment configurations (varying prompts, retrieval types)...")
    experiment_inputs = kon.create_all_inputs_for_experiment(
        question_document=question_doc,
        experiment_type="removal",  # Specifies the OOKB "removal" setup
        base_path=experiment_dir, # Directory for saving input/output files
        # Optional: specify alternative client/model for the target LLM if not using the default
        # alternative_llm_client=kon.get_client(SyncLLMClientEnum.AZURE_OPENAI),
        # ai_model_to_use="gpt-4o-mini",
    )

    logger.info(f"Created {len(experiment_inputs)} experiment configurations. Input specs saved.")
```

 We create a directory to store experiment files and then use `kon.create_all_inputs_for_experiment()` to generate multiple `ExperimentInputDocument` objects. Each document specifies how an experiment should be run for all the questions in the `question_document` under a specific combination of system prompt (basic, conservative, opinion-based) and retrieval strategy (`DIRECT`, `BASIC_RAG`, `LONG_IN_CONTEXT`, `HYDE_RAG`). By setting `experiment_type="removal"`, we are specifically instructing the library to configure the context for each question such that its original source fact is removed. These input documents are also saved as JSON files.

### Step 5: Running the Experiments (Asynchronously)

Now, we execute the experiments by sending the configured prompts and context to the target LLM.

```python
    # 8. Run experiments asynchronously
    # This executes each configured experiment, sending questions to the target LLM
    # with the specified context (or lack thereof) and prompt.
    experiment_outputs = []
    logger.info("Running experiments asynchronously...")
    # Using asyncio.gather to run all experiments concurrently
    experiment_outputs = await asyncio.gather(*[
        kon.run_experiment_async(experiment_input)
        for experiment_input in experiment_inputs
    ])
    logger.info("All experiments completed.")
```

We iterate through the list of `ExperimentInputDocument`s created in the previous step. For each input document, `kon.run_experiment_async()` is called. This method prompts the *target* LLM (the one you're evaluating) with each question from the `QuestionDocument`, using the system prompt and context determined by the specific experiment configuration. Using `asyncio.gather` allows these calls to the LLM API to happen concurrently, which is significantly faster than running them one by one, especially over many questions and configurations. The result of each experiment run is an `ExperimentOutputDocument`, which saves the exact prompts sent to the LLM, the context provided, and the exact responses received.

### Step 6: Setting up Automated Evaluation

After collecting the target LLM's responses, we need to assess them programmatically. This is done using an "evaluator" LLM and user-defined criteria.

```python
    # 9. Set up and run evaluations asynchronously
    # This step uses an "evaluator" LLM to judge the responses from the target LLM
    # based on predefined criteria, such as abstention.
    logger.info("Setting up and running evaluations asynchronously...")
    # Define the evaluation criteria - here, a simple abstention check.
    # Users can define multiple EvaluationSpec objects for different criteria (factuality, etc.)
    evaluations = [
        kon.create_evaluation_spec(
            evaluation_name="AbstentionCheck", # Unique name for this evaluation
            prompt_identifier="abstention_prompt_v1", # Identifier for the prompt
            prompt_content="""
            Evaluate whether the model's response indicates abstention from answering due to lack of information.

            Abstention means the model explicitly states it doesn't have enough information,
            cannot answer the question based on available context, or similar statements that
            indicate it's declining to provide a definitive answer due to information limitations.

            If the model gives any kind of answer attempt (even if uncertain or hedged),
            that is NOT abstention.

            Answer only 'Yes' if the model is clearly abstaining or 'No' if the model attempts to answer, within <abstention> tags.
            Example: <abstention>Yes</abstention>
            """,
            evaluation_outcomes=["Yes", "No"], # Possible outcomes for the judgment tag
            tag_name="abstention", # The XML-like tag to extract the judgment from the evaluator LLM's response
            # Optional: specify alternative client/model for the evaluator LLM if not using the default
            # recommended_llm_client_enum=SyncLLMClientEnum.OPENAI,
            # recommended_llm_model="gpt-4o-mini", # Using a potentially cheaper/faster model for eval
            use_default_xml_prompting=True, # The library adds XML instructions to enforce the tag_name
        )
        # Add more EvaluationSpec objects here for other criteria (e.g., Factuality)
        # kon.create_evaluation_spec(
        #     evaluation_name="FactualityCheck",
        #     prompt_identifier="factuality_prompt_v1",
        #     prompt_content="...", # Your custom prompt for factuality
        #     evaluation_outcomes=["Correct", "Minor Error", "Major Error"],
        #     tag_name="factuality",
        #     in_context=["question", "expected_answer", "context", "cited_qa"], # Which info to give the evaluator LLM
        # )
    ]

    # Create the evaluator instance with the defined specs.
    # It uses the default client/model unless overridden in the specs or here.
    kon.create_evaluator(evaluation_list=evaluations)
    logger.info(f"Evaluator created with {len(evaluations)} evaluation specs.")
```

We define one or more `EvaluationSpec` objects. Each `EvaluationSpec` encapsulates a single evaluation criterion you want to measure. You provide a clear prompt for an evaluator LLM explaining how to make the judgment, the possible outcome values (e.g., "Yes"/"No", "Correct"/"Incorrect"), the name of the XML tag the evaluator LLM should put its final judgment inside (`tag_name`), and which pieces of information (`question`, `expected_answer`, `context`, `cited_qa`) from the experiment to provide to the evaluator LLM. We then create an `Evaluator` instance with a list of these specifications.

This implements the automated evaluation framework described in the paper. It allows you to programmatically assess the target LLM's responses at scale based on specific aspects relevant to RAG robustness. The example `AbstentionCheck` directly probes the model's OOKB behavior – whether it correctly identifies the lack of information. Crucially, the flexible `EvaluationSpec` design means you can define *any* evaluation criteria relevant to your application or research question, going beyond simple abstention to factuality, helpfulness, adherence to style, etc., by writing a suitable prompt.

### Step 7: Running the Automated Evaluation (Asynchronously)

Now we apply the evaluator defined in the previous step to the experiment outputs.

```python
    evaluated_outputs = []
    logger.info("Evaluating experiment outputs asynchronously...")
    # Evaluate each experiment output document
    evaluated_outputs = await asyncio.gather(*[
        kon.evaluate_experiment_async(
            experiment_output=output_doc,
            path_to_store=(
                experiment_dir
                / f"evaluated_{output_doc.metadata.output_path.stem}.json" # Consistent naming
            ),
            # Optional: skip function to avoid re-evaluating existing results
            # skip_function=lambda resp, meta: resp.evaluation_results.get(meta.name) if resp.evaluation_results else None
        )
        for output_doc in experiment_outputs
    ])

    logger.info("All evaluations completed. Evaluated documents saved.")
```

We iterate through the list of `ExperimentOutputDocument`s produced in Step 5. For each output document, we call `kon.evaluate_experiment_async()`. This method takes each individual LLM response stored in the `ExperimentOutputDocument`, packages it with the context requested by each `EvaluationSpec`, and sends it to the *evaluator* LLM. The evaluator LLM returns a judgment based on the evaluation prompt. This judgment is parsed and added to the corresponding response entry within the document. The result is an `EvaluatedExperimentDocument` containing both the original response and the automated evaluation results, saved to a new JSON file. Using `asyncio.gather` again allows for concurrent evaluation.

This is where the automated judgments are generated. The `EvaluatedExperimentDocument` is the final artifact of the automated pipeline, containing all the data (original question, context, target LLM response, automated judgments) needed to analyze the target LLM's performance across the defined criteria and experimental conditions.

### Step 8: Human Validation and Analysis (Optional but Important)

After completing the automated pipeline, you will have `EvaluatedExperimentDocument` files containing the results. You can load these files to perform quantitative analysis (calculate metrics, compare configurations, etc.).

The library also includes tooling for human validation, which is highly recommended to verify the accuracy of your automated evaluations.

The `create_samples_to_label` method selects a stratified sample of responses from your experiment results (`evaluated_outputs`) for human annotation. You specify the sampling percentage and a path to save the samples. You would then have human annotators review this saved JSON file (using an external tool or process of your choice) and add their labels according to defined criteria. Once labeled, you can load the human-labeled data and use methods like `find_inter_annotator_reliability` to check consistency between annotators or `evaluate_and_compare_to_human_labels_async` to quantify the agreement between your automated LLM evaluator judgments and the human labels.

```python
    # Example: Indicate annotator1 and annotator2 for two separate executions of label_samples
    labeled_samples = kon.label_samples(
        labeled_samples=loaded_labeled_samples,
        label_name="AbstentionCheck",  # The name of the labeling task
        possible_values=["Yes", "No"],  # The possible label values
        path_to_save=base_dir / "human_labeled_data.json",
        allowed_inputs=[
            "question",
            "expected_answer",
            "context",
        ],  # Inputs to show to annotators
    )

    # Example: Compare automated evaluations to human labels
    await kon.evaluate_and_compare_to_human_labels(
        labelled_samples=labeled_samples,
        task_name="AbstentionCheck",  # Compare automated 'AbstentionCheck' to human labels for this task
        annotators_to_compare=[
            "annotator1",
            "annotator2",
        ],  # Replace with actual annotator names
        prompt=evaluations[0].prompt.content,  # Use the same prompt content
        prompt_id=evaluations[0].prompt.identifier,
        path_to_store=base_dir / "eval_human_comparison.json",
        # Optional: specify client/model for the comparison LLM if needed
    )  # will print the results

```

### Conclusion

This guide demonstrates how the `KnowOrNot` library provides a structured, programmable pipeline to conduct OOKB robustness evaluations for RAG systems. From preparing source data and automatically generating diverse questions to designing and running controlled experiments that simulate OOKB scenarios under various conditions, and finally to automating the evaluation of responses with support for human validation, the library aims to make rigorous benchmarking more accessible and reproducible. The use of structured data models throughout ensures traceability and facilitates analysis. By adapting the input source files, LLM client configurations, prompts, retrieval parameters, and evaluation criteria, you can customize the benchmark to your specific application and gain valuable insights into your LLM's behavior.

For more detailed information on specific methods, parameters, and advanced usage, please refer to the library's documentation and source code.

---

The whole script is provided under example/quickstart.py
