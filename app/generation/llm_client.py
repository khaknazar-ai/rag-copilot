from urllib import error, request
import json
import socket
from abc import ABC, abstractmethod

from app.core.config import settings
from app.generation.prompts import build_rag_prompt


import re
from pathlib import Path
from fastapi import HTTPException
class BaseLLMClient(ABC):
    @abstractmethod
    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        pass


class LocalGroundedLLMClient(BaseLLMClient):
    """
    Local deterministic consultation generator.

    This is not a real LLM.
    It is a high-quality local fallback for portfolio demo mode.

    Why not pure sentence extraction?
    - sentence extraction produced ugly answers;
    - chunks can cut words at boundaries;
    - consultation websites need structured answers.

    Behavior:
    - uses only retrieved source identity and indexed official-source facts;
    - formats answers like a service consultation;
    - refuses when no retrieved evidence exists;
    - keeps citations clean.
    """

    def _source(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""
        return chunks[0].get("source", "")

    def _citation(self, chunks: list[dict]) -> str:
        source = self._source(chunks)
        return f"[1] {source}" if source else ""

    def _wrap(
        self,
        title: str,
        answer_lines: list[str],
        sources: list[str],
        required_documents: list[str] | None = None,
        processing_time: str | None = None,
        fees: str | None = None,
        important_notes: list[str] | None = None,
        contact_consulate: bool = True,
    ) -> str:
        parts = [
            f"## {title}",
            "",
            "### Answer",
        ]

        for line in answer_lines:
            parts.append(f"- {line} [1]")

        if required_documents:
            parts.extend(["", "### Required documents"])
            for item in required_documents:
                parts.append(f"- {item} [1]")

        if processing_time:
            parts.extend(["", "### Processing time"])
            parts.append(f"- {processing_time} [1]")

        if fees:
            parts.extend(["", "### Fees"])
            parts.append(f"- {fees} [1]")

        if important_notes:
            parts.extend(["", "### Important notes"])
            for note in important_notes:
                parts.append(f"- {note} [1]")

        if contact_consulate:
            parts.extend(
                [
                    "",
                    "### When to contact the consular section",
                    "- Contact the consular section if your case is urgent, non-standard, legally sensitive, or your documents differ from the standard list.",
                ]
            )

        parts.extend(
            [
                "",
                "### Disclaimer",
                "- This is informational guidance based on indexed sources. It does not replace an official consular decision.",
                "",
                "### Sources",
            ]
        )
        parts.extend(sources)

        return "\n".join(parts)

    def _passport_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Passport service for Kazakhstan citizens in China",
            answer_lines=[
                "Kazakhstan citizens in China may apply through the consular section for passport-related service when they fall into eligible categories described by the service rules.",
            ],
            required_documents=[
                "Kazakhstan passport.",
                "Copy of a valid China visa.",
                "Passport photos sized 3.5 by 4.5 centimeters.",
                "China registration document with translation when applicable.",
                "Confirmation of consular fee payment.",
                "If the passport was lost or stolen, loss or theft confirmation when available.",
            ],
            processing_time="Passport processing through the consular section in China may take about 30 to 40 calendar days.",
            important_notes=[
                "If the applicant does not fall into the eligible categories for passport issuance abroad, the applicant may need to apply in Kazakhstan.",
            ],
            sources=[self._citation(chunks)],
        )

    def _lost_passport_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Lost passport and certificate of return",
            answer_lines=[
                "If a Kazakhstan citizen loses a passport abroad, the citizen should contact the consular section, provide identity information, and provide loss or theft confirmation when available.",
                "A certificate of return may be issued when the identity document is lost, expired, unusable, or otherwise unavailable.",
                "The certificate of return confirms the right of the holder to make a one-time entry into the Republic of Kazakhstan.",
            ],
            important_notes=[
                "A certificate of return is not a general travel document for travel to third countries.",
                "Urgent loss, theft, detention, death, or safety-related cases should be escalated to a consular officer.",
            ],
            sources=[self._citation(chunks)],
        )

    def _visa_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Kazakhstan visa service for foreign citizens in China",
            answer_lines=[
                "Foreign citizens applying for a Kazakhstan visa through the consular section in China submit a visa application package according to the visa category.",
            ],
            required_documents=[
                "Visa application form.",
                "Photo sized 3.5 by 4.5 centimeters.",
                "Valid foreign passport.",
                "Depending on the visa category: invitation from Kazakhstan, employer letter, business license, work permit, medical insurance, certificate of no criminal record, or other supporting documents.",
            ],
            processing_time="Visa processing should be completed within 5 working days after the required documents are received.",
            fees="The paid visa fee is not returned after payment according to the visa service information.",
            important_notes=[
                "The consular office cannot guarantee visa approval through an informational assistant.",
            ],
            sources=[self._citation(chunks)],
        )

    def _registration_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Consular registration for Kazakhstan citizens abroad",
            answer_lines=[
                "Consular registration is used to register Kazakhstan citizens who permanently or temporarily live outside Kazakhstan within the relevant consular district.",
                "The purpose of consular registration is to help protect the rights and lawful interests of Kazakhstan citizens abroad.",
            ],
            required_documents=[
                "For temporary registration: identity document and completed registration card.",
                "For permanent registration: identity document, confirmation of permission for permanent residence abroad, and completed registration card.",
                "For Kazakhstan citizens in China, supporting documents may include a passport, photo, work or study confirmation, and a document confirming registration in China.",
            ],
            important_notes=[
                "The consular officer may refuse to accept the application if the document package is incomplete or if documents are expired.",
            ],
            sources=[self._citation(chunks)],
        )

    def _birth_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Birth registration abroad",
            answer_lines=[
                "For birth registration abroad, applicants provide an application and supporting civil-status documents.",
            ],
            required_documents=[
                "Application.",
                "Identity documents of the parents.",
                "Marriage certificate when applicable.",
                "Medical birth certificate.",
                "Translation and legalization when required.",
                "Confirmation of consular fee payment.",
            ],
            important_notes=[
                "A medical birth certificate issued in China may require translation and legalization before it is accepted for Kazakhstan civil status registration.",
            ],
            sources=[self._citation(chunks)],
        )

    def _notarial_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Notarial actions at Kazakhstan consular offices",
            answer_lines=[
                "Consular officers may perform notarial actions within their competence.",
            ],
            required_documents=[
                "Identity document.",
                "Document that requires the notarial action.",
            ],
            important_notes=[
                "Notarial actions may include certification of transactions, certification of copies, certification of signatures, certification of translations, and inheritance-related certificates.",
                "Legally complex or case-specific notarial questions should be escalated to a consular officer.",
            ],
            sources=[self._citation(chunks)],
        )

    def _legalization_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Document legalization",
            answer_lines=[
                "Document legalization confirms the authenticity of signatures and seals on documents for use across jurisdictions when legalization is required.",
            ],
            required_documents=[
                "Original document or notarized copy when applicable.",
                "Translation into Kazakh or Russian when required.",
            ],
            important_notes=[
                "Legalization is not the same as legal advice about the substance of the document.",
                "If the document type is not covered by the indexed official sources, the assistant should not provide a definitive answer.",
            ],
            sources=[self._citation(chunks)],
        )

    def _safety_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Escalation and safety policy",
            answer_lines=[
                "The assistant provides informational guidance based only on indexed official sources.",
                "The assistant does not replace a consular officer, does not make legal decisions, does not guarantee visa approval, and does not provide case-specific legal conclusions.",
                "Urgent or sensitive cases involving detention, death, lost documents, safety threats, fraud, forged documents, or non-standard legal circumstances should be escalated to a consular officer.",
            ],
            important_notes=[
                "The assistant should not help users bypass official requirements, submit false documents, avoid personal submission requirements, or misrepresent information.",
            ],
            sources=[self._citation(chunks)],
        )

    def _fees_answer(self, chunks: list[dict]) -> str:
        return self._wrap(
            title="Consular fees and processing times",
            answer_lines=[
                "Many consular services are provided on a paid basis through consular fees.",
            ],
            processing_time="Passport processing may take about 30 to 40 calendar days. Visa processing should be completed within 5 working days after all required documents are received.",
            fees="Paid visa fees are not returned after payment according to the visa service information.",
            important_notes=[
                "Fees and timelines may depend on the service category and official fee rules, so applicants should check the latest official source before payment.",
            ],
            sources=[self._citation(chunks)],
        )

    def _clean_generic_context(self, text: str) -> str:
        cleaned_lines = []

        metadata_prefixes = (
            "title:",
            "source_id:",
            "service_category:",
            "jurisdiction:",
            "official_source:",
            "---",
        )

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            lowered = line.lower()

            if line.startswith("#"):
                continue

            if lowered.startswith(metadata_prefixes):
                continue

            cleaned_lines.append(line)

        return " ".join(cleaned_lines).strip()

    def _generic_answer(self, chunks: list[dict]) -> str:
        """
        Fallback for non-consular unit tests and future unknown sources.

        Important:
        AnswerGenerator must still use the retrieved context it receives.
        The domain-specific branches make the website nice, but generic fallback
        should not ignore supplied chunks.
        """

        context = self._clean_generic_context(chunks[0].get("text", ""))

        if not context:
            context = "The retrieved source contains relevant information for this question."

        return self._wrap(
            title="Retrieved document answer",
            answer_lines=[context],
            important_notes=[
                "This answer was generated from the retrieved context supplied to the answer generator.",
            ],
            sources=[self._citation(chunks)],
            contact_consulate=False,
        )

    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return (
                "I don't have enough information in the indexed official sources "
                "to answer that safely. Please contact the consular section or "
                "check the latest official source."
            )

        source = self._source(chunks)

        if "citizen_passport_service_china.md" in source:
            return self._passport_answer(chunks)

        if "lost_passport_return_certificate.md" in source:
            return self._lost_passport_answer(chunks)

        if "foreign_citizen_visa_service_china.md" in source:
            return self._visa_answer(chunks)

        if "consular_registration_china.md" in source:
            return self._registration_answer(chunks)

        if "birth_registration_abroad.md" in source:
            return self._birth_answer(chunks)

        if "notarial_actions_china.md" in source:
            return self._notarial_answer(chunks)

        if "document_legalization.md" in source:
            return self._legalization_answer(chunks)

        if "escalation_and_safety_policy.md" in source:
            return self._safety_answer(chunks)

        if "fees_and_processing_times.md" in source:
            return self._fees_answer(chunks)

        return self._generic_answer(chunks)


class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI-backed answer generator.

    Requires:
    - LLM_PROVIDER=openai
    - OPENAI_API_KEY set in .env
    """

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Set OPENAI_API_KEY in .env or use LLM_PROVIDER=local."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return (
                "I don't have enough information in the indexed official sources "
                "to answer that safely. Please contact the consular section or "
                "check the latest official source."
            )

        prompt = build_rag_prompt(question=question, chunks=chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict document-grounded consular services assistant. "
                        "Answer only from the provided context. "
                        "Use a structured consultation format. "
                        "Do not provide legal decisions. "
                        "Do not guarantee service outcomes. "
                        "Refuse unsupported questions."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content or ""



class OllamaLLMClient(BaseLLMClient):
    """
    Real local LLM backend through Ollama.

    This provider is intended for demo-quality RAG generation:
    - free local inference;
    - no external API key;
    - useful for showing the project with a real LLM;
    - deterministic local provider is still used for tests/evaluation.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = max(timeout_seconds, 180)



    def _is_lost_passport_question(self, question: str) -> bool:
        q = question.lower().strip()

        lost_passport_phrases = (
            "lost passport",
            "stolen passport",
            "passport was stolen",
            "passport was lost",
            "expired passport",
            "passport expired",
            "certificate of return",
            "return certificate",
            "lost my kazakhstan passport",
            "lost my passport",
            "украли паспорт",
            "потерял паспорт",
            "потеряла паспорт",
            "потерян паспорт",
            "утерян паспорт",
            "свидетельство на возвращение",
            "паспорт истек",
            "паспорт просрочен",
        )

        return any(phrase in q for phrase in lost_passport_phrases)

    def _expand_parent_documents(self, chunks: list[dict]) -> list[dict]:
        """
        Expand retrieved chunks into parent source documents for local LLM generation.

        Retrieval still decides which source is relevant.
        Generation gets more complete source context, so small local LLMs are less
        likely to miss facts that are located outside the top retrieved chunk.
        """
        expanded: list[dict] = []
        seen_sources: set[str] = set()

        for chunk in chunks:
            source = (
                chunk.get("source")
                or chunk.get("metadata", {}).get("source")
                or chunk.get("source_path")
            )

            enriched = dict(chunk)

            if source and source not in seen_sources:
                source_path = Path(source)

                if source_path.exists() and source_path.is_file():
                    text = source_path.read_text(encoding="utf-8")

                    # Keep small source documents intact.
                    # Blindly taking text[:6000] can drop facts located later
                    # in the same official source document.
                    max_parent_chars = 3500

                    if len(text) <= max_parent_chars:
                        enriched["text"] = text
                    else:
                        # Fallback for larger documents:
                        # keep the beginning plus the most obviously useful
                        # service sections. This avoids hiding critical facts
                        # such as processing time, required documents, and fees.
                        useful_lines = []
                        keywords = (
                            "required",
                            "documents",
                            "processing",
                            "calendar",
                            "working days",
                            "fee",
                            "fees",
                            "time",
                            "срок",
                            "документ",
                            "сбор",
                            "оплат",
                        )

                        for line in text.splitlines():
                            lowered = line.lower()
                            if any(keyword in lowered for keyword in keywords):
                                useful_lines.append(line)

                        useful_text = "\n".join(useful_lines)
                        enriched["text"] = (
                            text[:4000]
                            + "\n\n# Extracted service facts\n"
                            + useful_text[:8000]
                        )

                seen_sources.add(source)

            expanded.append(enriched)

        return expanded


    def _source_lines(self, chunks: list[dict]) -> list[str]:
        lines: list[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            source = (
                chunk.get("source")
                or chunk.get("metadata", {}).get("source")
                or chunk.get("source_path")
                or "unknown_source"
            )
            lines.append(f"[{idx}] {source}")

        return lines


    def _remove_unsupported_optional_sections(
        self,
        answer: str,
        chunks: list[dict],
        question: str,
    ) -> str:
        """Remove optional markdown sections that are not grounded enough.

        This is a safety guard for small local LLMs. The model may add generic
        Fees or Processing Time sections even when the retrieved source does not
        contain concrete fee/timing facts. For consular/legal RAG, omission is
        safer than unsupported detail.
        """

        context_parts: list[str] = []
        for chunk in chunks:
            context_parts.append(str(chunk.get("text", "")))
            context_parts.append(str(chunk.get("text_preview", "")))
            context_parts.append(str(chunk.get("content", "")))

        context = "\n".join(context_parts).lower()
        q = question.lower()

        asks_fee = any(
            token in q
            for token in (
                "fee",
                "fees",
                "cost",
                "price",
                "refund",
                "payment",
                "сбор",
                "стоимость",
                "цена",
                "оплата",
                "возврат",
            )
        )

        asks_time = any(
            token in q
            for token in (
                "time",
                "timeline",
                "how long",
                "processing",
                "deadline",
                "days",
                "срок",
                "сроки",
                "дней",
                "сколько ждать",
            )
        )

        has_concrete_fee = bool(
            re.search(
                r"(\d+\s*(usd|dollar|dollars|kzt|tenge|тенге|юан|yuan|rmb))|"
                r"(fee\s+(is\s+)?not\s+returned)|"
                r"(paid\s+visa\s+fee\s+is\s+not\s+returned)",
                context,
                flags=re.IGNORECASE,
            )
        )

        has_concrete_time = bool(
            re.search(
                r"(\d+\s*(calendar|working)?\s*days?)|"
                r"(\d+\s*(календарн|рабоч)[а-я]*\s*дн)|"
                r"(processing\s+time)|"
                r"(shorter\s+processing)|"
                r"(longer\s+processing)",
                context,
                flags=re.IGNORECASE,
            )
        )

        def remove_section(markdown: str, section_title: str) -> str:
            pattern = (
                rf"\n###\s+{section_title}\s*\n"
                rf".*?"
                rf"(?=\n###\s+|\n##\s+|\Z)"
            )
            return re.sub(pattern, "\n", markdown, flags=re.IGNORECASE | re.DOTALL)

        cleaned = answer

        if not asks_fee and not has_concrete_fee:
            cleaned = remove_section(cleaned, "Fees")

        if not asks_time and not has_concrete_time:
            cleaned = remove_section(cleaned, "Processing Time")

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned + "\n"

    def _enforce_sources_section(self, answer: str, chunks: list[dict]) -> str:
        source_lines = self._source_lines(chunks)

        if not source_lines:
            return answer

        sources_block = "### Sources\n" + "\n".join(source_lines)

        # Remove weak placeholder source section generated by small local models.
        answer = re.sub(
            r"(?is)\n*###\s*Sources\s*\n(?:.|\n)*$",
            "",
            answer,
        ).strip()

        # Replace common inline placeholder if it appeared inside the answer.
        for line in source_lines:
            marker = line.split(" ", 1)[0]
            source = line.split(" ", 1)[1]
            answer = answer.replace(f"{marker} <source>", f"{marker} {source}")

        return answer.rstrip() + "\n\n" + sources_block


    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return (
                "I don't have enough information in the indexed documents "
                "to answer that question."
            )

        prompt_chunks = self._expand_parent_documents(chunks)
        rag_prompt = build_rag_prompt(question=question, chunks=prompt_chunks)
        rag_prompt = (
            "Fast grounded answer requirements:\n"
            "- Keep the answer concise: maximum 180 words unless the user explicitly asks for details.\n"
            "- Use only the provided context.\n"
            "- Do not invent fees, deadlines, processing times, legal outcomes, or guarantees.\n"
            "- Do not include a Fees section unless the user asks about fees or the context gives a specific fee/refund rule.\n"
            "- Do not include a Processing Time section unless the user asks about timing or the context gives a concrete timing rule.\n"
            "- Prefer bullet points over long paragraphs.\n\n"
            + rag_prompt
        )

        if self._is_lost_passport_question(question):
            rag_prompt += """
Lost passport / certificate of return rule:
- If the question is about a lost, stolen, or expired Kazakhstan passport in China, answer through the certificate of return procedure.
- Do not answer lost-passport questions as ordinary passport renewal.
- Do not use passport-renewal processing time for lost-passport return-certificate questions.
- Mention Chinese exit-status issues cautiously: if the person entered China with a visa, residence permit, or another non-visa-free basis, they should confirm with Chinese exit-entry authorities or the Kazakhstan consular office whether additional exit clearance is required.
"""

        rag_prompt += """

Response style requirements:
- Answer the user's actual question directly.
- Do not include sections that are irrelevant to the question.
- Do not include "Processing time" or "Fees" sections unless the question asks for them or the provided context contains a useful fact for that specific topic.
- Avoid repeating the same information in multiple sections.
- Prefer a concise, citizen-facing answer.
- Keep the disclaimer short.
"""

        prompt = (
            "You are a strict document-grounded RAG assistant for Kazakhstan "
            "consular-service questions. Use only the provided context. "
            "Do not invent requirements, fees, processing times, legal conclusions, "
            "or service guarantees. Cite factual claims using the citation markers "
            "from the context, such as [1], [2]. If the context is insufficient, "
            "say that you do not have enough information in the indexed documents. "
            "Return a clear structured answer with sections.\n\n"
            f"{rag_prompt}"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 256,
            },
        }

        request_body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
                parsed = json.loads(raw_body)
        except error.URLError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Could not reach Ollama at {self.base_url}. "
                    "Make sure Ollama is running and the model is pulled. "
                    "No fallback answer was generated."
                ),
            ) from exc
        except OSError as exc:
            message = str(exc).lower()
            is_timeout = isinstance(exc, TimeoutError) or isinstance(exc, socket.timeout) or "timed out" in message

            if is_timeout:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Ollama model '{self.model}' did not finish within "
                        f"{self.timeout_seconds} seconds. No fallback answer was generated."
                    ),
                ) from exc

            raise HTTPException(
                status_code=503,
                detail=(
                    "The local LLM backend returned a transport error. "
                    "No fallback answer was generated."
                ),
            ) from exc

        answer = parsed.get("response", "").strip()

        if not answer:
            return (
                "The local Ollama model returned an empty answer. "
                "Check Ollama server status and model availability."
            )

        answer = self._remove_unsupported_optional_sections(
            answer=answer,
            chunks=chunks,
            question=question,
        )
        return self._enforce_sources_section(answer, chunks)

def get_llm_client() -> BaseLLMClient:
    provider = settings.llm_provider.lower().strip()

    if provider == "local":
        return LocalGroundedLLMClient()

    if provider == "ollama":
        return OllamaLLMClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    if provider == "openai":
        return OpenAILLMClient(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
