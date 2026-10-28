import os
import re
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st


APP_TITLE = "Kazakhstan Consular Services Assistant"
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("CHAT_REQUEST_TIMEOUT_SECONDS", "240"))

EXAMPLES = [
    ("Passport service", "What documents are needed for passport service for a Kazakhstan citizen in China? Include processing time if the source mentions it."),
    ("Lost passport", "I lost my Kazakhstan passport in China. What should I do?"),
    ("Visa fee refund", "Can a foreign applicant get a visa fee refunded after payment?"),
    ("Consular registration", "What is consular registration abroad and why is it needed?"),
    ("Birth registration", "How can Kazakhstan citizens register the birth of a child abroad?"),
    ("Power of attorney", "Can the consular section certify a power of attorney?"),
    ("Approval guarantee", "Can you guarantee that my visa will be approved?"),
]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🇰🇿",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stDeployButton"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stSidebar"] {display: none !important;}

        .stApp {
            background: #f4fafc;
            color: #0d3048;
        }

        .main .block-container {
            max-width: 920px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, p, li, label, span, div {
            color: #0d3048;
        }

        .top-card {
            background: linear-gradient(135deg, #ddf6ff 0%, #ffffff 86%);
            border: 1px solid rgba(26, 167, 216, 0.22);
            border-radius: 24px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 12px 30px rgba(20, 70, 105, 0.07);
            margin-bottom: 1rem;
        }

        .brand-row {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .logo {
            width: 52px;
            height: 52px;
            border-radius: 17px;
            display: grid;
            place-items: center;
            background: #18a7d8;
            color: #ffffff;
            font-weight: 900;
            box-shadow: 0 10px 22px rgba(24, 167, 216, 0.22);
        }

        .title {
            font-size: 2rem;
            font-weight: 850;
            letter-spacing: -0.04em;
            line-height: 1.1;
            color: #0b2e45;
        }

        .subtitle {
            margin-top: 0.45rem;
            max-width: 720px;
            color: #2f5369;
            line-height: 1.55;
            font-size: 1rem;
        }

        .card {
            background: #ffffff;
            border: 1px solid rgba(13, 48, 72, 0.12);
            border-radius: 20px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 10px 26px rgba(20, 70, 105, 0.055);
            margin-bottom: 1rem;
        }

        .card-title {
            font-size: 1.05rem;
            font-weight: 850;
            margin-bottom: 0.5rem;
            color: #0b2e45;
        }

        .muted {
            color: #2f5369;
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .example-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.85rem;
        }

        .answer-card {
            background: #ffffff;
            border: 1px solid rgba(13, 48, 72, 0.12);
            border-radius: 22px;
            padding: 1.55rem 1.7rem;
            box-shadow: 0 12px 32px rgba(20, 70, 105, 0.07);
            margin-top: 1rem;
        }

        .answer-card h1 {
            font-size: 1.45rem;
            margin-top: 0;
            margin-bottom: 0.8rem;
            color: #0b2e45;
        }

        .answer-card h2 {
            font-size: 1.22rem;
            margin-top: 1.15rem;
            margin-bottom: 0.5rem;
            color: #0b2e45;
        }

        .answer-card h3 {
            font-size: 1.05rem;
            margin-top: 1rem;
            margin-bottom: 0.4rem;
            color: #0b2e45;
        }

        .answer-card p,
        .answer-card li {
            font-size: 1rem;
            line-height: 1.7;
            color: #16384f;
        }

        .answer-card ul {
            padding-left: 1.25rem;
        }

        .source-box {
            margin-top: 1.1rem;
            background: #f0faff;
            border: 1px solid rgba(26, 167, 216, 0.22);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: #174c68;
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .error-card {
            background: #fff5f4;
            border: 1px solid rgba(220, 38, 38, 0.16);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            color: #842f2f;
            margin-top: 1rem;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.8rem;
        }

        .status-pill {
            border-radius: 999px;
            padding: 0.35rem 0.65rem;
            background: #f7fbfd;
            border: 1px solid rgba(13, 48, 72, 0.12);
            color: #24485e;
            font-size: 0.8rem;
            font-weight: 750;
        }

        .status-hit {
            background: #eafaf1;
            color: #1e7047;
            border-color: rgba(30, 112, 71, 0.20);
        }

        .status-miss {
            background: #fff7e6;
            color: #8a6000;
            border-color: rgba(138, 96, 0, 0.20);
        }

        .citation-card {
            border-radius: 16px;
            background: #f8fcff;
            border: 1px solid rgba(13, 48, 72, 0.10);
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
        }

        .citation-title {
            font-weight: 850;
            color: #0b2e45;
            margin-bottom: 0.35rem;
        }

        .citation-meta,
        .citation-preview {
            color: #31556c;
            font-size: 0.9rem;
            line-height: 1.5;
            white-space: pre-wrap;
        }

        .stTextArea textarea {
            background: #ffffff !important;
            color: #0d3048 !important;
            border: 1px solid rgba(13, 48, 72, 0.28) !important;
            border-radius: 16px !important;
            font-size: 1rem !important;
            line-height: 1.55 !important;
            min-height: 135px !important;
        }

        textarea::placeholder,
        input::placeholder {
            color: #637f92 !important;
            opacity: 1 !important;
        }

        .stTextArea label,
        .stRadio label {
            color: #0d3048 !important;
            font-weight: 750 !important;
        }

        .stButton > button {
            border-radius: 999px;
            min-height: 2.35rem;
            background: #ffffff;
            color: #0d3048 !important;
            border: 1px solid rgba(13, 48, 72, 0.16);
            font-weight: 750;
            padding-left: 1rem;
            padding-right: 1rem;
            box-shadow: none;
        }

        .stButton > button:hover {
            background: #eaf8ff;
            border-color: rgba(26, 167, 216, 0.45);
            color: #0b5f82 !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            background: #18a7d8 !important;
            color: #ffffff !important;
            border-color: #18a7d8 !important;
            border-radius: 14px !important;
            min-height: 2.6rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: #0d8fbd !important;
            color: #ffffff !important;
        }

        .stInfo {
            background: #e8f5fb !important;
            border: 1px solid rgba(26, 167, 216, 0.20) !important;
            color: #0d3048 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_api_url(api_url: str) -> str:
    return api_url.strip().rstrip("/")


def ask_backend(api_url: str, question: str, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        response = requests.post(
            f"{normalize_api_url(api_url)}/chat",
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return {
            "ok": False,
            "status_code": 504,
            "cache": "NO_RESPONSE",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": "The local LLM request timed out. No fallback answer was generated.",
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status_code": 503,
            "cache": "NO_RESPONSE",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": f"Could not reach the API: {exc}",
        }

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    cache_status = response.headers.get("x-cache", "UNKNOWN")
    cache_key = response.headers.get("x-cache-key", "")

    if not response.ok:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass

        return {
            "ok": False,
            "status_code": response.status_code,
            "cache": cache_status,
            "cache_key": cache_key,
            "latency_ms": latency_ms,
            "error": detail,
        }

    payload = response.json()

    return {
        "ok": True,
        "status_code": response.status_code,
        "cache": cache_status,
        "cache_key": cache_key,
        "latency_ms": latency_ms,
        "answer": payload.get("answer", ""),
        "citations": payload.get("citations", []),
        "raw": payload,
    }


def clean_answer_for_display(answer: str) -> str:
    answer = re.sub(r"(?is)\n*###\s*Sources\s*\n.*$", "", answer).strip()
    return answer


def pretty_source_name(source: str) -> str:
    filename = Path(source).name.replace(".md", "").replace("_", " ")
    return filename.title()


def render_header(role: str) -> None:
    badge = "Applicant view" if role == "Applicant" else "Consular staff view"

    st.markdown(
        f"""
        <div class="top-card">
            <div class="brand-row">
                <div class="logo">KZ</div>
                <div>
                    <div class="title">Kazakhstan Consular Services Assistant</div>
                    <div class="subtitle">
                        Ask questions about Kazakhstan consular services in China.
                        Answers are grounded in indexed official-source documents.
                    </div>
                    <div style="margin-top: 0.85rem;">
                        <span style="background:#fff5cf;border:1px solid rgba(212,175,55,.45);border-radius:999px;padding:.42rem .75rem;color:#745900;font-size:.82rem;font-weight:800;">
                            {badge}
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return

    names = []
    for citation in citations:
        source = citation.get("source", "")
        if source:
            names.append(pretty_source_name(source))

    names = list(dict.fromkeys(names))
    if not names:
        return

    lines = "<br>".join([f"• {name}" for name in names])

    st.markdown(
        f"""
        <div class="source-box">
            <b>Official sources used</b><br>
            {lines}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_staff_details(result: dict[str, Any]) -> None:
    cache = result.get("cache", "UNKNOWN")
    cache_class = "status-hit" if cache == "HIT" else "status-miss" if cache == "MISS" else ""

    with st.expander("Staff details: retrieval, cache and raw response", expanded=False):
        st.markdown(
            f"""
            <div class="status-row">
                <span class="status-pill">HTTP {result.get("status_code", "n/a")}</span>
                <span class="status-pill {cache_class}">Cache {cache}</span>
                <span class="status-pill">Latency {result.get("latency_ms", "n/a")} ms</span>
                <span class="status-pill">Citations {len(result.get("citations", []))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        evidence_tab, raw_tab = st.tabs(["Evidence", "Raw response"])

        with evidence_tab:
            citations = result.get("citations", [])
            if not citations:
                st.info("No retrieved evidence returned.")
            for idx, citation in enumerate(citations, start=1):
                source = citation.get("source", "unknown source")
                chunk_id = citation.get("chunk_id", "unknown chunk")
                preview = citation.get("text_preview", "")

                st.markdown(
                    f"""
                    <div class="citation-card">
                        <div class="citation-title">[{idx}] {pretty_source_name(source)}</div>
                        <div class="citation-meta">Source: {source}<br>Chunk: {chunk_id}</div>
                        <div class="citation-preview">{preview}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with raw_tab:
            st.json(result)


def render_answer(result: dict[str, Any], role: str) -> None:
    if not result.get("ok"):
        st.markdown(
            f"""
            <div class="error-card">
                <b>Could not generate an answer.</b><br><br>
                {result.get("error", "Unknown error")}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if role == "Consular staff":
            render_staff_details(result)
        return

    answer = clean_answer_for_display(result.get("answer", ""))

    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(answer)
    render_sources(result.get("citations", []))
    st.markdown("</div>", unsafe_allow_html=True)

    if role == "Consular staff":
        render_staff_details(result)


def init_state() -> None:
    if "query_text" not in st.session_state:
        st.session_state.query_text = ""
    if "last_question" not in st.session_state:
        st.session_state.last_question = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def main() -> None:
    inject_css()
    init_state()

    role = st.radio(
        "View mode",
        options=["Applicant", "Consular staff"],
        horizontal=True,
    )

    render_header(role)

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Ask a consular-service question</div>
            <div class="muted">
                The assistant answers from indexed sources and does not replace an official consular decision.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Example questions")

    cols = st.columns(3)
    for idx, (label, question) in enumerate(EXAMPLES):
        with cols[idx % 3]:
            if st.button(label, key=f"example_{idx}"):
                st.session_state.query_text = question
                st.rerun()

    if role == "Consular staff":
        with st.expander("Advanced settings", expanded=False):
            api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL)
            top_k = st.slider("Retrieved chunks", 1, 8, 5)
    else:
        api_url = DEFAULT_API_URL
        top_k = 5

    with st.form("question_form"):
        question = st.text_area(
            "Your question",
            value=st.session_state.query_text,
            placeholder="Ask about Kazakhstan consular services in China...",
        )
        submitted = st.form_submit_button("Get answer")

    if submitted:
        clean_question = question.strip()
        st.session_state.query_text = clean_question

        if clean_question:
            with st.spinner("Preparing answer from official sources..."):
                result = ask_backend(api_url=api_url, question=clean_question, top_k=top_k)

            st.session_state.last_question = clean_question
            st.session_state.last_result = result

    if st.session_state.last_question:
        st.markdown("### Your question")
        st.info(st.session_state.last_question)

    if st.session_state.last_result:
        render_answer(st.session_state.last_result, role)


if __name__ == "__main__":
    main()