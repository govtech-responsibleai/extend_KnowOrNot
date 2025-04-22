# KnowOrNot

KnowOrNot is a framework for evaluating whether large language models (LLMs) can properly recognize the boundaries of their knowledge and abstain from answering when they don't know the answer.

## Overview

Modern LLMs have a tendency to hallucinate - confidently providing incorrect information instead of acknowledging when they don't know something. KnowOrNot provides a systematic approach to:

1. Test if LLMs can identify what they do and don't know
2. Measure an LLM's propensity to hallucinate vs. abstain
3. Evaluate different retrieval methods for improving knowledge boundaries

## Installation

```bash
pip install knowornot # in this case not so, do uv add ../KnoworNot
```

## Quick Start

```python
from knowornot import KnowOrNot
from pathlib import Path

# Initialize KnowOrNot
kon = KnowOrNot()

# Connect to an LLM provider (Azure OpenAI in this example)
kon.add_azure(
    azure_endpoint="https://your-endpoint.openai.azure.com",
    azure_api_key="your-api-key",
    azure_api_version="2023-05-15"
)

# Create a knowledge base from documents
questions = kon.create_questions(
    source_paths=[Path("document1.txt"), Path("document2.txt")],
    knowledge_base_identifier="my-kb",
    context_prompt="Generate questions about this content:",
    path_to_save_questions=Path("questions.json"),
    filter_method="both"
)

# Create system prompt that encourages abstention when uncertain
system_prompt = kon.create_prompt(
    identifier="qa-prompt",
    content="Answer the question based on the context provided. If you cannot find the answer in the context, respond with 'I don't know'."
)

# Create experiment to test knowledge boundaries
experiment_input = kon.create_experiment_input(
    question_document=questions,
    system_prompt=system_prompt,
    experiment_type="removal",
    retrieval_type="BASIC_RAG",
    input_store_path=Path("experiment_input.json"),
    output_store_path=Path("experiment_output.json")
)

# Run the experiment
experiment_output = kon.run_experiment_sync(experiment_input)

# Create evaluation metrics
abstention_spec = kon.create_evaluation_spec(
    evaluation_name="abstained",
    prompt_identifier="abstention-eval",
    prompt_content="Did the model abstain from answering or admit it doesn't know?",
    evaluation_outcomes=["abstained", "attempted_answer"],
    tag_name="abstention"
)

accuracy_spec = kon.create_evaluation_spec(
    evaluation_name="accuracy",
    prompt_identifier="accuracy-eval",
    prompt_content="Is the model's answer accurate compared to the expected answer?",
    evaluation_outcomes=["accurate", "inaccurate", "partially_accurate"],
    tag_name="accuracy"
)

# Create evaluator with both metrics
evaluator = kon.create_evaluator([abstention_spec, accuracy_spec])

# Evaluate the results
evaluation_results = kon.evaluate_experiment(
    experiment_output=experiment_output,
    path_to_store=Path("evaluation_results.json")
)
```

## Key Features

### Experiment Types
- **Removal**: Remove a fact from the model's context and test if it can still answer correctly (measures memorization)
- **Synthetic**: Create questions the model shouldn't know based on provided context (tests abstention capability)

### Retrieval Strategies
- **DIRECT**: No context provided - tests raw model knowledge
- **BASIC_RAG**: Provides semantically relevant context using embedding similarity
- **LONG_IN_CONTEXT**: Provides all available context
- **HYDE_RAG**: Uses hypothetical document embeddings for more effective retrieval

### Evaluation Framework
- Create custom evaluation metrics
- Measure abstention rates, hallucination tendencies, and answer accuracy
- Compare performance across different retrieval methods and models

## Architecture

KnowOrNot consists of several integrated components:

1. **FactManager**: Extracts structured facts from documents
2. **QuestionExtractor**: Generates diverse question-answer pairs
3. **ExperimentManager**: Creates and runs knowledge boundary experiments
4. **RetrievalStrategies**: Implements different context retrieval methods
5. **Evaluator**: Assesses model responses with customizable metrics
