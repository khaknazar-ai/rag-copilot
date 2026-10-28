from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class LearnedIntentPrediction:
    intent: str
    confidence: float


class LearnedIntentClassifier:
    """Supervised multilingual intent classifier for service-aware RAG.

    This replaces brittle keyword-only routing with a learned model trained on
    labeled consular-service queries. It predicts the service intent and exposes
    a confidence score so the retrieval layer can decide whether to trust the
    prediction or fall back to safer behavior.
    """

    def __init__(self, model_path: str | Path = "models/intent_classifier.joblib") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Intent classifier model not found: {self.model_path}. "
                "Supply a compatible model artifact or use rule-based routing."
            )

        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "joblib/scikit-learn is required to load the learned intent classifier."
            ) from exc

        bundle = joblib.load(self.model_path)
        self.pipeline = bundle["pipeline"]
        self.labels = bundle["labels"]

    def predict(self, text: str) -> LearnedIntentPrediction:
        normalized = text.strip()

        if not normalized:
            return LearnedIntentPrediction(intent="out_of_scope", confidence=0.0)

        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba([normalized])[0]
            best_index = int(probabilities.argmax())
            intent = str(self.pipeline.classes_[best_index])
            confidence = float(probabilities[best_index])
            return LearnedIntentPrediction(intent=intent, confidence=confidence)

        intent = str(self.pipeline.predict([normalized])[0])
        return LearnedIntentPrediction(intent=intent, confidence=1.0)
