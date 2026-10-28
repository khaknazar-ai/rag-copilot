import pandas as pd
import streamlit as st

from app.monitoring.metrics import get_monitoring_store


st.set_page_config(
    page_title="RAG Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("RAG Monitoring Dashboard")
st.caption("PostgreSQL-backed observability for /chat requests, cache behavior, latency, and citations.")

with st.sidebar:
    st.header("Controls")
    days = st.slider("Lookback window, days", min_value=1, max_value=30, value=7)
    limit = st.slider("Recent events", min_value=10, max_value=200, value=50, step=10)

store = get_monitoring_store()

try:
    summary = store.fetch_summary(days=days)
    cache_breakdown = store.fetch_cache_breakdown(days=days)
    latency_by_cache = store.fetch_latency_by_cache_status(days=days)
    recent_events = store.fetch_recent_events(limit=limit)

except Exception as exc:
    st.error("Monitoring data is not available. Check PostgreSQL and METADATA_DB_PROVIDER.")
    st.exception(exc)
    st.stop()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total queries", summary["total_queries"])
col2.metric("Cache hit rate", f"{summary['cache_hit_rate'] * 100:.1f}%")
col3.metric("Avg latency", f"{summary['avg_latency_ms']:.1f} ms")
col4.metric("Avg citations", f"{summary['avg_citation_count']:.2f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Cache status breakdown")

    if cache_breakdown:
        cache_df = pd.DataFrame(cache_breakdown)
        st.bar_chart(cache_df.set_index("cache_status")["count"])
        st.dataframe(cache_df, use_container_width=True)
    else:
        st.info("No cache events available yet.")

with right:
    st.subheader("Latency by cache status")

    if latency_by_cache:
        latency_df = pd.DataFrame(latency_by_cache)
        st.bar_chart(latency_df.set_index("cache_status")["avg_latency_ms"])
        st.dataframe(latency_df, use_container_width=True)
    else:
        st.info("No latency events available yet.")

st.divider()

st.subheader("Recent /chat events")

if recent_events:
    events_df = pd.DataFrame(recent_events)
    st.dataframe(
        events_df[
            [
                "id",
                "created_at",
                "cache_status",
                "latency_ms",
                "citation_count",
                "top_k",
                "embedding_provider",
                "reranker_provider",
                "question",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No query events available yet.")