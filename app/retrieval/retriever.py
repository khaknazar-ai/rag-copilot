from app.core.config import settings
from app.ingestion.embeddings import BaseEmbeddingClient, get_embedding_client
from app.retrieval.domain_intent import DomainIntent, classify_domain_intent
from app.retrieval.lexical import lexical_overlap_terms
from app.retrieval.reranker import BaseReranker, get_reranker
from app.retrieval.vector_store import QdrantVectorStore


class Retriever:
    """
    Document-grounded RAG retriever.

    Pipeline:
    1. Classify the domain/service intent.
    2. Embed the user question.
    3. Retrieve a broad candidate pool from Qdrant.
    4. Apply intent-based source filtering.
    5. Optionally rerank candidates with a cross-encoder.
    6. Apply lexical / intent relevance guards.
    7. Deduplicate sources and return a compact evidence set.

    Why this design?

    - Vector search gives fast candidate retrieval.
    - Intent routing prevents citation pollution across similar services.
    - Lexical guard reduces false positives for weak local embeddings.
    - Optional cross-encoder reranking improves Top-1 precision in quality mode.
    """

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient | None = None,
        vector_store: QdrantVectorStore | None = None,
        reranker: BaseReranker | None = None,
        score_threshold: float | None = None,
        lexical_min_overlap_terms: int | None = None,
    ) -> None:
        self.embedding_client = embedding_client or get_embedding_client()
        self.vector_store = vector_store or QdrantVectorStore()
        self.reranker = reranker or get_reranker()

        self.score_threshold = (
            settings.retrieval_score_threshold
            if score_threshold is None
            else score_threshold
        )
        self.lexical_min_overlap_terms = (
            settings.lexical_min_overlap_terms
            if lexical_min_overlap_terms is None
            else lexical_min_overlap_terms
        )

    def _candidate_pool_size(self, top_k: int) -> int:
        return max(top_k * 6, 30)

    def _filter_by_intent(
        self,
        candidates: list[dict],
        intent: DomainIntent,
    ) -> list[dict]:
        if not intent.allowed_sources:
            return candidates

        allowed_sources = set(intent.allowed_sources)

        return [
            candidate
            for candidate in candidates
            if candidate.get("source") in allowed_sources
        ]

    def _has_known_intent(self, intent: DomainIntent) -> bool:
        return bool(intent.allowed_sources) and not intent.should_refuse

    def _passes_relevance_guard(
        self,
        candidate: dict,
        question: str,
        intent: DomainIntent,
    ) -> bool:
        overlap_terms = lexical_overlap_terms(
            question=question,
            chunk_text=candidate.get("text", ""),
        )
        overlap_count = len(overlap_terms)
        score = candidate.get("score", 0.0)

        candidate["lexical_overlap_count"] = overlap_count
        candidate["lexical_overlap_terms"] = sorted(overlap_terms)
        candidate["domain_intent"] = intent.name

        # If the service intent explicitly points to an official source,
        # we accept that source even when lexical overlap is low.
        #
        # This matters for multilingual / paraphrased questions:
        # Russian user question may route correctly to an English source,
        # while lexical overlap can still be zero.
        if self._has_known_intent(intent):
            candidate["intent_source_match"] = True
            return True

        # For unknown intent, require lexical evidence by default.
        if overlap_count >= self.lexical_min_overlap_terms:
            candidate["intent_source_match"] = False
            return True

        # Optional score-only fallback if lexical filtering is explicitly disabled.
        if self.lexical_min_overlap_terms <= 0 and score >= self.score_threshold:
            candidate["intent_source_match"] = False
            return True

        return False

    def _deduplicate_by_source(self, chunks: list[dict]) -> list[dict]:
        seen_sources = set()
        deduplicated = []

        for chunk in chunks:
            source = chunk.get("source")

            if source in seen_sources:
                continue

            seen_sources.add(source)
            deduplicated.append(chunk)

        return deduplicated

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        if not question.strip():
            raise ValueError("question must not be empty")

        effective_top_k = top_k or settings.top_k

        intent = classify_domain_intent(question)

        if intent.should_refuse:
            return []

        query_vector = self.embedding_client.embed_query(question)

        candidates = self.vector_store.search(
            query_vector=query_vector,
            top_k=self._candidate_pool_size(effective_top_k),
        )

        candidates = self._filter_by_intent(
            candidates=candidates,
            intent=intent,
        )

        if not candidates:
            return []

        # Optional quality mode:
        # rerank after intent filtering, before final relevance filtering.
        candidates = self.reranker.rerank(
            question=question,
            chunks=candidates,
            top_k=min(len(candidates), self._candidate_pool_size(effective_top_k)),
        )

        filtered_results = []

        for candidate in candidates:
            if self._passes_relevance_guard(
                candidate=candidate,
                question=question,
                intent=intent,
            ):
                filtered_results.append(candidate)

        filtered_results = self._deduplicate_by_source(filtered_results)

        if self._has_known_intent(intent):
            return filtered_results[: max(1, len(intent.allowed_sources))]

        return filtered_results[:effective_top_k]
