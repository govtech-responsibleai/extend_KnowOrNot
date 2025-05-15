import asyncio
from pathlib import Path
from typing import List
from knowornot import KnowOrNot
from knowornot.common.models import LabeledDataSample
from dotenv import load_dotenv


async def run_knowornot_pipeline(text_files: List[Path]):
    """
    Run a complete KnowOrNot pipeline from a list of text files using async methods.

    Args:
        text_files: List of Path objects pointing to .txt files to analyze
    """
    # 1. Initialize KnowOrNot
    kon = KnowOrNot()
    load_dotenv()  # Load environment variables (e.g., API keys)

    # 2. Add an LLM client (using OpenAI - requires API key in environment variables)
    # The library supports Azure, Gemini, and OpenRouter clients via similar add_* methods.
    # Ensure OPENAI_API_KEY, OPENAI_DEFAULT_MODEL, and OPENAI_DEFAULT_EMBEDDING_MODEL are set in your environment or .env file.
    try:
        kon.add_openai()
        kon.logger.info("OpenAI client added and set as default.")
    except EnvironmentError as e:
        kon.logger.error(
            f"Failed to add OpenAI client: {e}. Please ensure required environment variables are set."
        )
        return  # Exit if client setup fails

    # 3. Create a unique identifier for this knowledge base
    kb_identifier = "quickstart_knowledge_base"
    kon.logger.info(f"Using Knowledge Base Identifier: {kb_identifier}")

    # 4. Set up paths for outputs
    base_dir = Path("kon_outputs")
    base_dir.mkdir(exist_ok=True, parents=True)
    questions_path = (
        base_dir / f"{kb_identifier}_questions.json"
    )  # Include KB ID in filename

    # 5. Create questions from the text files
    # This step involves LLM-assisted fact extraction and Q&A generation,
    # followed by diversity filtering as described in the paper's methodology.
    kon.logger.info(f"Creating questions from source documents: {text_files}")
    context_prompt = "Answer questions based solely on the provided information."
    question_doc = kon.create_questions(
        source_paths=text_files,
        knowledge_base_identifier=kb_identifier,
        context_prompt=context_prompt,  # Base prompt used during Q&A generation
        path_to_save_questions=questions_path,
        filter_method="both",  # Apply both keyword and semantic diversity filtering
    )

    kon.logger.info(
        f"Created {len(question_doc.questions)} diverse questions. Saved to {questions_path}"
    )

    # 6. Set up experiment directories
    experiment_dir = base_dir / "experiments"
    experiment_dir.mkdir(exist_ok=True, parents=True)

    # 7. Create all experiment variations
    # This step defines the specific experiment configurations (Prompt, Retrieval Strategy, Target LLM Model)
    # for each question generated in Step 5. The "removal" type implements the core OOKB
    # leave-one-out setup described in the paper.
    kon.logger.info(
        "Creating all experiment configurations (varying prompts, retrieval types)..."
    )
    experiment_inputs = kon.create_all_inputs_for_experiment(
        question_document=question_doc,
        experiment_type="removal",  # Specifies the OOKB "removal" setup
        base_path=experiment_dir,  # Directory for saving input/output files
        # Optional: specify alternative client/model for the target LLM if not using the default
        # alternative_llm_client=kon.get_client(SyncLLMClientEnum.AZURE_OPENAI),
        # ai_model_to_use="gpt-4o-mini",
    )

    kon.logger.info(
        f"Created {len(experiment_inputs)} experiment configurations. Input specs saved."
    )

    # 8. Run experiments asynchronously
    # This executes each configured experiment, sending questions to the target LLM
    # with the specified context (or lack thereof) and prompt.
    experiment_outputs = []
    kon.logger.info("Running experiments asynchronously...")
    # Using asyncio.gather to run all experiments concurrently
    experiment_outputs = await asyncio.gather(
        *[
            kon.run_experiment_async(experiment_input)
            for experiment_input in experiment_inputs
        ]
    )
    kon.logger.info("All experiments completed.")

    # 9. Set up and run evaluations asynchronously
    # This step uses an "evaluator" LLM to judge the responses from the target LLM
    # based on predefined criteria, such as abstention.
    kon.logger.info("Setting up and running evaluations asynchronously...")
    # Define the evaluation criteria - here, a simple abstention check.
    # Users can define multiple EvaluationSpec objects for different criteria (factuality, etc.)
    evaluations = [
        kon.create_evaluation_spec(
            evaluation_name="AbstentionCheck",  # Unique name for this evaluation
            prompt_identifier="abstention_prompt_v1",  # Identifier for the prompt
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
            evaluation_outcomes=["Yes", "No"],  # Possible outcomes for the judgment tag
            tag_name="abstention",  # The XML-like tag to extract the judgment from the evaluator LLM's response
            # Optional: specify alternative client/model for the evaluator LLM if not using the default
            # recommended_llm_client_enum=SyncLLMClientEnum.OPENAI,
            # recommended_llm_model="gpt-4o-mini", # Using a potentially cheaper/faster model for eval
            use_default_xml_prompting=True,  # The library adds XML instructions to enforce the tag_name
        )
        # Add more EvaluationSpec objects here for other criteria (e.g., Factuality)
        # kon.create_evaluation_spec(...)
    ]

    # Create the evaluator instance with the defined specs.
    # It uses the default client/model unless overridden in the specs or here.
    kon.create_evaluator(evaluation_list=evaluations)
    kon.logger.info(f"Evaluator created with {len(evaluations)} evaluation specs.")

    evaluated_outputs = []
    kon.logger.info("Evaluating experiment outputs asynchronously...")
    # Evaluate each experiment output document
    evaluated_outputs = await asyncio.gather(
        *[  # noqa: F841
            kon.evaluate_experiment_async(
                experiment_output=output_doc,
                path_to_store=(
                    experiment_dir
                    / f"evaluated_{output_doc.metadata.output_path.stem}.json"  # Consistent naming
                ),
                # Optional: skip function to avoid re-evaluating existing results
                # skip_function=lambda resp, meta: resp.evaluation_results.get(meta.name) if resp.evaluation_results else None
            )
            for output_doc in experiment_outputs
        ]
    )

    kon.logger.info(
        f"All {len(evaluated_outputs)} evaluations completed. Evaluated documents saved."
    )

    # 10. Analyze Results and Human Validation (Optional - for demonstration purposes)
    # The EvaluatedExperimentDocument objects can now be loaded and analyzed to
    # calculate metrics (like abstention rate, factuality scores) under different
    # experimental conditions.
    # For human validation, you would use the data labeller:
    kon.logger.info("Creating samples for human labeling...")
    labeled_samples_path = base_dir / "human_labels_samples.json"
    samples_for_labeling = kon.create_samples_to_label(
        experiment_outputs=evaluated_outputs,
        percentage_to_sample=0.1,  # Sample 10%
        path_to_store=labeled_samples_path,
    )
    kon.logger.info(f"Created {len(samples_for_labeling)} samples for human labeling.")

    # Human annotators would then use a tool/process
    # to add labels to the JSON file created above (labeled_samples_path).
    # The library provides methods to load these labeled samples and analyze agreement
    # or compare automated evals to human labels:

    # Load labeled samples that have been annotated by humans
    loaded_labeled_samples = LabeledDataSample.load_list_from_json(labeled_samples_path)

    # Add additional human labels if needed
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

    # Compare automated evaluations to human labels
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

    kon.logger.info("KnowOrNot pipeline completed successfully!")


# Example usage:
if __name__ == "__main__":
    # Replace with your actual text files.
    # Example files space_1.txt and space_2.txt can be found in the 'usage/' directory
    # of the KnowOrNot GitHub repository: https://github.com/govtech-responsibleai/KnowOrNot
    my_text_files = [
        Path("usage/space_1.txt"),
        Path("usage/space_2.txt"),
        # Add more paths as needed
    ]

    # Ensure example files exist for the quickstart
    for f_path in my_text_files:
        if not f_path.exists():
            print(
                f"Error: Example file not found at {f_path}. Please clone the repository."
            )
            exit()

    # Run the async pipeline
    asyncio.run(run_knowornot_pipeline(my_text_files))
