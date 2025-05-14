from pathlib import Path
from typing import List
from knowornot import KnowOrNot
from dotenv import load_dotenv


def run_knowornot_pipeline(text_files: List[Path]):
    """
    Run a complete KnowOrNot pipeline from a list of text files.

    Args:
        text_files: List of Path objects pointing to .txt files to analyze
    """
    # 1. Initialize KnowOrNot
    kon = KnowOrNot()
    load_dotenv()

    # 2. Add an LLM client (using OpenAI - requires API key in environment variables)
    # You can replace this with add_azure() or add_gemini() if preferred
    kon.add_openai(
        # Uses OPENAI_API_KEY, OPENAI_DEFAULT_MODEL, and OPENAI_DEFAULT_EMBEDDING_MODEL from env vars
    )

    # 3. Create a unique identifier for this knowledge base
    kb_identifier = "quickstart_knowledge_base"

    # 4. Set up paths for outputs
    base_dir = Path("kon_outputs")
    base_dir.mkdir(exist_ok=True, parents=True)
    questions_path = base_dir / "questions.json"

    # 5. Create questions from the text files
    context_prompt = "Answer questions based solely on the provided information."
    question_doc = kon.create_questions(
        source_paths=text_files,
        knowledge_base_identifier=kb_identifier,
        context_prompt=context_prompt,
        path_to_save_questions=questions_path,
        filter_method="both",  # Use both keyword and semantic filtering
    )

    print(f"Created {len(question_doc.questions)} questions from source documents")

    # 6. Set up experiment directories
    experiment_dir = base_dir / "experiments"
    experiment_dir.mkdir(exist_ok=True, parents=True)

    # 7. Create all experiment variations
    experiment_inputs = kon.create_all_inputs_for_experiment(
        question_document=question_doc,
        experiment_type="removal",  # Use removal experiment type
        base_path=experiment_dir,
    )

    print(f"Created {len(experiment_inputs)} experiment configurations")

    # 8. Run experiments and evaluate them
    for experiment_input in experiment_inputs:
        # Run the experiment
        print(f"Running experiment: {experiment_input.metadata.input_path}")
        experiment_output = kon.run_experiment_sync(experiment_input)

        # Set up evaluator with abstention evaluation
        evaluations = [
            kon.create_evaluation_spec(
                evaluation_name="abstention_evaluation",
                prompt_identifier="abstention_prompt",
                prompt_content="""
                Evaluate whether the model's response indicates abstention from answering due to lack of information.

                Abstention means the model explicitly states it doesn't have enough information,
                cannot answer the question based on available context, or similar statements that
                indicate it's declining to provide a definitive answer due to information limitations.

                If the model gives any kind of answer attempt (even if uncertain or hedged),
                that is NOT abstention.

                Answer only 'Yes' if the model is clearly abstaining or 'No' if the model attempts to answer.
                """,
                evaluation_outcomes=["Yes", "No"],
                tag_name="abstention",
            )
        ]

        # Create evaluator
        kon.create_evaluator(evaluation_list=evaluations)

        # Evaluate experiment
        print(f"Evaluating experiment: {experiment_input.metadata.input_path}")

        eval_path = (
            experiment_dir
            / f"evaluated_{experiment_input.metadata.output_path.stem}.json"
        )
        kon.evaluate_experiment(
            experiment_output=experiment_output,
            path_to_store=eval_path,
        )

        print(f"Evaluation complete: {eval_path}")

    print("KnowOrNot pipeline completed successfully!")


# Example usage:
if __name__ == "__main__":
    # Replace with your actual text files
    my_text_files = [
        Path("document1.txt"),
        Path("document2.txt"),
        # Add more paths as needed
    ]

    run_knowornot_pipeline(my_text_files)
