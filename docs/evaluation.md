# Evaluation summary

The figures below are historical offline/demo evidence. They were not rerun for this public candidate.

## Intent and source router

- Training set: 154 examples across 12 labels.
- Three-fold cross-validation accuracy: 0.617 ± 0.064.
- Clean holdout: 36 questions, no exact normalized text overlap with training; learned-classifier and router accuracy: 1.000.
- Source-policy accuracy: 1.000 on 18 source-scored examples.

The sample is small and domain-specific. These router metrics do not measure generated-answer quality, source freshness, or production reliability.

## Local LLM integration

A historical single-query Ollama smoke check produced a grounded answer for one passport-service question. It was an integration example, not a quality benchmark, and was not rerun from a clean build of this candidate.
