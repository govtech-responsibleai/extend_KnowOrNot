from knowornot import KnowOrNot
from knowornot.common.models import (
    Prompt,
)
from pathlib import Path
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables (assuming .env is in your project root)
load_dotenv()

# Define paths - these are now relative to the script's directory
SOURCE_DIR_PATH = Path("run/data/")  # Source files are in the ./data/ directory
QUESTIONS_OUTPUT_PATH = Path("btt_kb_long_questions.json")  # Output questions file
FACT_STORAGE_DIR = Path(
    "./generated_facts/"
)  # Optional, save intermediate facts in a subdir
EXP_INPUT_PATH = Path("btt_long_removal_input.json")  # Output experiment input file
EXP_OUTPUT_PATH = Path("btt_long_removal_output.json")  # Output experiment output file
EVAL_OUTPUT_PATH = Path(
    "btt_long_removal_evaluated.json"
)  # Output evaluation results file

# Ensure output directories/paths exist (only need to create the fact storage subdir)
if FACT_STORAGE_DIR:
    FACT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
# Ensure the source data directory exists
SOURCE_DIR_PATH.mkdir(parents=True, exist_ok=True)

# System prompt for the main experiment LLM
EXPERIMENT_SYSTEM_PROMPT = Prompt(
    identifier="abstention-citation-prompt",
    content=(
        "Carefully consider the following context to answer the question. "
        "Your answer must be based *only* on the provided context. "
        "If the context does not contain the answer, state clearly 'I do not know based on the provided context'. "
        "If you use information from the context, cite the corresponding fact number (e.g., [1], [2])."
    ),
)

# --- 1. Initialize KnowOrNot and add LLM Client ---
print("Initializing KnowOrNot and adding client...")
kon = KnowOrNot()

# Add Azure client using environment variables (assuming AZURE_OPENAI_ENDPOINT, etc. are set in .env)
# Ensure you have a default chat model and an embedding model configured in your .env
kon.add_azure(
    default_embedding_model="text-embedding-3-large"
)  # Example embedding model, adjust if needed
print("Azure client added using environment variables.")
default_client = kon.get_client()
print(f"Default client registered: {default_client.enum_name})")


# --- 2. Create Questions (Knowledge Base) ---
# This generates Q&A pairs from the source documents
print("\nCreating questions from source documents in ./data/...")

# Use glob to find all .txt files in the ./data/ directory
source_files = list(SOURCE_DIR_PATH.glob("*.txt"))

if not source_files:
    raise FileNotFoundError("No .txt files found in ./data/.")


print(f"Found {len(source_files)} source files.")

question_doc_btt = kon.create_questions(
    source_paths=source_files,
    knowledge_base_identifier="btt-kb",  # Updated identifier
    context_prompt="Generate factual questions based strictly on the provided Basic Theory Test text.",  # Updated prompt context
    path_to_save_questions=QUESTIONS_OUTPUT_PATH,
    filter_method="both",  # Use both keyword and semantic filtering for diversity
    fact_storage_dir=FACT_STORAGE_DIR,  # Optional: save intermediate facts in subdir
)
print(
    f"Successfully created QuestionDocument with {len(question_doc_btt.questions)} questions."
)
print(f"Saved to: {question_doc_btt.path_to_store}")


# --- 3. Create Experiment Input Document ---
# Defines exactly what prompt/context goes to the LLM for each question
print("\nCreating Experiment Input Document (Long In Context, Removal)...")
experiment_input_doc_btt = kon.create_experiment_input(
    question_document=question_doc_btt,
    system_prompt=EXPERIMENT_SYSTEM_PROMPT,
    experiment_type="removal",  # Test questions where the direct source is removed
    retrieval_type="LONG_IN_CONTEXT",  # Provide (almost) all KB context
    input_store_path=EXP_INPUT_PATH,
    output_store_path=EXP_OUTPUT_PATH,  # Where results will be saved later
    # Uses the default LLM client and model configured in the KnowOrNot instance
    # These are stored in the metadata for run_experiment_sync
)

print("Successfully created ExperimentInputDocument.")
print(
    f"Contains {len(experiment_input_doc_btt.questions)} individual experiment inputs."
)
print(
    f"Saved experiment input definition to: {experiment_input_doc_btt.metadata.input_path}"
)
print(
    f"Experiment output will be saved to: {experiment_input_doc_btt.metadata.output_path}"
)


# --- 4. Run the Experiment ---
# Executes the LLM calls based on the input document
print("\nRunning the experiment synchronously...")
experiment_output_doc_btt = kon.run_experiment_sync(
    experiment_input=experiment_input_doc_btt
)

print("Experiment run complete.")
print(
    f"Successfully saved ExperimentOutputDocument to: {experiment_output_doc_btt.metadata.output_path}"
)
print(
    f"Experiment results contain responses for {len(experiment_output_doc_btt.responses)} questions."
)


# --- 5. Define Evaluation Metrics (Evaluation Specs) ---
print("\nDefining Evaluation Metrics...")

# Metric 1: Factuality Check (against the expected answer/KB)
factuality_spec = kon.create_evaluation_spec(
    evaluation_name="FactualityCheck",
    prompt_identifier="factuality-eval-prompt",
    prompt_content=(
        "Compare the Model Answer to the Expected Answer based on the Provided Context. "
        "Determine if the Model Answer is factually accurate, inaccurate, or partially accurate *relative to the Expected Answer and the Provided Context*.\n"
        "Original Question: {question}\n"  # Included from in_context
        "Expected Answer (from Knowledge Base): {expected_answer}\n"  # Included from in_context
        "Provided Context (to the experiment LLM): {context}\n"  # Included from in_context
        "Model Answer: {model_answer}\n\n"  # Automatically added
        "Respond *only* with <factuality>followed_by_outcome</factuality> tags. "
        "Allowed outcomes: 'accurate', 'inaccurate', 'partially_accurate'."
    ),
    evaluation_outcomes=["accurate", "inaccurate", "partially_accurate"],
    tag_name="factuality",
    in_context=[
        "question",
        "expected_answer",
        "context",
    ],  # Provide these to the evaluator LLM
    # Optional: recommended_llm_client_enum, recommended_llm_model could be set here
)

# Metric 2: Logical Entailment Check (does the answer follow from the context?)
entailment_spec = kon.create_evaluation_spec(
    evaluation_name="LogicalEntailmentCheck",
    prompt_identifier="entailment-eval-prompt",
    prompt_content=(
        "Given the Provided Context and the Original Question, does the Model Answer logically follow *solely from the information in the Provided Context*? "
        "Ignore any outside knowledge the model might have.\n"
        "Original Question: {question}\n"  # Included from in_context
        "Provided Context (to the experiment LLM): {context}\n"  # Included from in_context
        "Model Answer: {model_answer}\n\n"  # Automatically added
        "Respond *only* with <entailment>followed_by_outcome</entailment> tags. "
        "Allowed outcomes: 'entailed', 'not_entailed', 'cannot_determine'."
    ),
    evaluation_outcomes=["entailed", "not_entailed", "cannot_determine"],
    tag_name="entailment",
    in_context=["question", "context"],  # Provide these to the evaluator LLM
    # Optional: recommended_llm_client_enum, recommended_llm_model could be set here
)

evaluation_specs_list = [factuality_spec, entailment_spec]
print(f"Defined {len(evaluation_specs_list)} evaluation metrics.")

# --- 6. Configure the Evaluator ---
# Bundles the specs and sets the default LLM for evaluation calls
print("\nConfiguring the Evaluator...")
kon.create_evaluator(
    evaluation_list=evaluation_specs_list,
    # Optional: specify a different client or model for ALL evaluations by default
    # alternative_llm_client=kon.get_client(SyncLLMClientEnum.AZURE_OPENAI),
    # alternative_llm_model="gpt-4o-evaluator" # Use a capable model for evaluation
)
print("Evaluator configured successfully.")


# --- 7. Run the Evaluation ---
# Applies the configured evaluator to the experiment output
print("\nRunning the evaluation...")
evaluated_document_btt = kon.evaluate_experiment(
    experiment_output=experiment_output_doc_btt, path_to_store=EVAL_OUTPUT_PATH
)

print("Evaluation run complete.")
print(
    f"Successfully saved EvaluatedExperimentDocument to: {evaluated_document_btt.path_to_store}"
)
print(
    f"Results contain evaluations for {len(evaluated_document_btt.responses)} responses."
)
print(f"Applied {len(evaluated_document_btt.evaluation_metadata)} evaluation metrics.")
