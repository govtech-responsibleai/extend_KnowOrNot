# KnowOrNot

KnowOrNot is an open-source framework that enables users to develop their own customized evaluation data and pipelines for evaluating out-of-knowledge base robustness, i.e. whether large language models (LLMs) can properly recognize the boundaries of their knowledge and abstain from answering when they don't know the answer.

## Highlights

- Unified, high-level API that streamlines the process of setting up and running robustness benchmark (only a source document is required to get the pipeline running)
- Modular architecture emphasises extensibility and flexibility, allowing users to easily integrate their own LLM clients and RAG settings
- Rigorous data modeling design ensures experiment reproducibility, reliability and traceability
- Comprehensive suite of tools for users to customize their pipelines

## Installation

1. Create and activate a virtual environment. Install uv.

```bash
python3 -m venv knowornot
source knowornot/bin/activate
pip install uv
```

2. Download the source code and enter the created source directory.
```bash
git clone git@github.com:govtech-responsibleai/KnowOrNot.git
cd knowornot
```

3. Install the library
```bash
uv pip install .
```

4. Set up environment variables in a `.env` file, depending on the LLM provider of choice. The sample script `example/quickstart_pipeline.py` depends on OpenAI and would require OpenAI environment variables. Refer to `env.example` for an example.

5. Run a sample evaluation pipeline.
```bash
uv run python example/quickstart_pipeline.py
```

## Quick Start

![Flow](assets/images/flow.png)

Refer to [quickstart.md](docs/quickstart.md) for more information and [quickstart_pipeline.py](example/quickstart_pipeline.py) for an end-to-end example flow.

## Key Features

![Features](assets/images/features.png)

### LLM Provider
- **OpenAI**: use `add_openai()` method
- **Gemini API**: use `add_gemini()` method
- **Azure**: use `add_azure()` method
- **OpenRouter**: use `add_openrouter()` method
- **Groq**: use `add_groq()` method
- **Anthropic**: use `add_anthropic()` method
- **Bedrock**: use `add_bedrock()` method

### Processing of LLM responses
- **Asynchronous**: use `run_experiment_async`, `evaluate_experiment_async` method
- **Synchronous**: use `run_experiment`, `evaluate_experiment` method

### Experiment Types
- **Leave-one-out**: Remove a fact from the model's context and test if it can still answer correctly (measures memorization)
- **Random (Synthetic)**: Create questions the model shouldn't know based on provided context (tests abstention capability)

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

KnowOrNot consists of several integrated components that are key to handling and generating the data artifacts:

1. **FactManager**: Extracts structured facts from documents
2. **QuestionExtractor**: Generates diverse question-answer pairs
3. **ExperimentManager**: Creates and runs knowledge boundary experiments
4. **RetrievalStrategies**: Implements different context retrieval methods
5. **Evaluator**: Assesses model responses with customizable metrics
6. **DataLabeller**: Orchestrates human labelling process for validation of LLM evaluations.

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
