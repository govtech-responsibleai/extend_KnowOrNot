from pathlib import Path
from typing import List
from knowornot import KnowOrNot
from knowornot.common.models import Prompt
from dotenv import load_dotenv


def run_simple_direct_test(text_files: List[Path]):
    """
    Run a simplified KnowOrNot test with just one DIRECT experiment and one evaluation.

    Args:
        text_files: List of Path objects pointing to .txt files to analyze
    """
    # Load environment variables
    load_dotenv()

    # Initialize KnowOrNot
    kon = KnowOrNot()

    # Add OpenAI client
    kon.add_gemini()

    # Create a unique identifier for this knowledge base
    kb_identifier = "direct_test_kb"

    # Set up paths for outputs
    base_dir = Path("kon_direct_test")
    base_dir.mkdir(exist_ok=True, parents=True)
    questions_path = base_dir / "questions.json"

    # Create questions from the text files
    print("Generating questions from source documents...")
    context_prompt = "Answer questions based solely on the provided information."
    question_doc = kon.create_questions(
        source_paths=text_files,
        knowledge_base_identifier=kb_identifier,
        context_prompt=context_prompt,
        path_to_save_questions=questions_path,
        filter_method="both",
    )

    print(f"Created {len(question_doc.questions)} questions from source documents")

    # Create experiment directory
    experiment_dir = base_dir / "experiments"
    experiment_dir.mkdir(exist_ok=True, parents=True)

    # Create abstention-focused system prompt
    abstention_system_prompt = Prompt(
        identifier="abstention_system_prompt",
        content="""You are a helpful assistant that answers questions based only on the provided information.
        If you don't have enough information to answer a question confidently, you should explicitly say so
        rather than guessing or making up information. It's better to acknowledge when you don't know something
        than to provide potentially incorrect information.

        Only answer what you can directly determine from the provided context. If the context doesn't contain
        relevant information, state that you cannot answer the question based on the available information.""",
    )

    # Manually create a single DIRECT experiment
    input_path = experiment_dir / "direct_test_input.json"
    output_path = experiment_dir / "direct_test_output.json"

    print("Creating single DIRECT experiment with abstention-focused prompt...")
    experiment_input = kon.create_experiment_input(
        question_document=question_doc,
        system_prompt=abstention_system_prompt,
        experiment_type="removal",  # Use removal experiment
        retrieval_type="DIRECT",  # Use DIRECT retrieval (no RAG)
        input_store_path=input_path,
        output_store_path=output_path,
    )

    # Run the experiment
    print("Running DIRECT experiment...")
    experiment_output = kon.run_experiment_sync(experiment_input)

    # Set up abstention evaluation
    abstention_eval = kon.create_evaluation_spec(
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

    # Create evaluator with just the abstention evaluation
    kon.create_evaluator(evaluation_list=[abstention_eval])

    # Evaluate experiment
    print("Evaluating experiment...")
    eval_path = experiment_dir / "evaluated_direct_test.json"
    evaluated_output = kon.evaluate_experiment(  # noqa: F841
        experiment_output=experiment_output,
        path_to_store=eval_path,
    )

    print(f"Evaluation complete and saved to: {eval_path}")
    print("Test completed successfully!")


if __name__ == "__main__":
    # Use the space exploration text files
    test_files = [
        Path("usage/space_1.txt"),
        Path("usage/space_2.txt"),
    ]

    run_simple_direct_test(test_files)
