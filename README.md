# KnowOrNot

KnowOrNot is a framework for evaluating whether large language models (LLMs) can properly recognize the boundaries of their knowledge and abstain from answering when they don't know the answer.

## Overview

Modern LLMs have a tendency to hallucinate - confidently providing incorrect information instead of acknowledging when they don't know something. KnowOrNot provides a systematic approach to:

1. Test if LLMs can identify what they do and don't know
2. Measure an LLM's propensity to hallucinate vs. abstain
3. Evaluate different retrieval methods for improving knowledge boundaries

## Installation

```bash
pip install knowornot # in this case - do uv add ../KnoworNot
```

## Quick Start

Refer to docs/quickstart.md for more

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
