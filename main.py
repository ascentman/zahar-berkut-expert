# /// script
# dependencies = [
#   "langchain-community",
#   "pypdf",
#   "langchain-text-splitters",
#   "langchain-google-genai",
#   "python-dotenv",
#   "langchain-qdrant",
#   "qdrant-client",
#   "langchain",
#   "streamlit",
# ]
# ///

import os
import time

import streamlit as st
from dotenv import load_dotenv

from chain_factory import create_retrieval_chain_headless
from chunk_utilization import check_utilization, UTILIZATION_THRESHOLD


@st.cache_resource
def init_retrieval_chain(api_key):
    """Initialize and cache the retrieval chain to prevent locking errors and redundant loading."""
    return create_retrieval_chain_headless(api_key)


def main():
    # Load variables from .env file into environment variables
    load_dotenv()

    st.set_page_config(page_title="Захар Беркут Expert", page_icon="🦅")
    st.title("🦅 Захар Беркут Expert")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "retrieval_chain" not in st.session_state:
        api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Missing GOOGLE_API_KEY. Please set it in your .env file or Streamlit secrets.")
            st.stop()

        with st.status("Initializing Knowledge Base...", expanded=False) as status:
            chain, retriever, logger = init_retrieval_chain(api_key)
            st.session_state.retrieval_chain = chain
            st.session_state.retriever = retriever
            st.session_state.logger = logger
            status.update(label="Knowledge Base Ready!", state="complete")

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar: cumulative stats
    if "logger" in st.session_state:
        stats = st.session_state.logger.get_stats()
        if stats["total_queries"] > 0:
            st.sidebar.markdown("### RAG Stats")
            st.sidebar.metric("Total Queries", stats["total_queries"])
            st.sidebar.metric("Avg Retrieval", f"{stats['avg_retrieval_ms']:.0f} ms")
            st.sidebar.metric("Avg Total", f"{stats['avg_total_ms']:.0f} ms")
            st.sidebar.metric("Avg Chunks", f"{stats['avg_chunks']:.1f}")
            st.sidebar.metric("Avg Utilization", f"{stats['avg_utilization']:.0%}")

    # Chat input
    if user_query := st.chat_input("Ask about Zahar Berkut..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                t_start = time.perf_counter()
                response = st.session_state.retrieval_chain.invoke({"input": user_query})
                total_time_ms = (time.perf_counter() - t_start) * 1000

                answer = response["answer"]
                chunks = response.get("context", [])
                retriever = st.session_state.retriever
                logger = st.session_state.logger

                # Check chunk utilization
                util_scores = check_utilization(answer, chunks)

                # Persist to SQLite
                query_id = logger.log_query(
                    user_query=user_query,
                    sub_queries=retriever.last_sub_queries,
                    answer=answer,
                    retrieval_time_ms=retriever.last_retrieval_time_ms,
                    total_time_ms=total_time_ms,
                    num_chunks=len(chunks),
                )
                logger.log_chunks(query_id, chunks, util_scores)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Diagnostics expander
            with st.expander("Retrieval Diagnostics"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Retrieval", f"{retriever.last_retrieval_time_ms:.0f} ms")
                col2.metric("Total", f"{total_time_ms:.0f} ms")
                utilized_count = sum(1 for s in util_scores if s > 0.05)
                col3.metric("Chunks Used", f"{utilized_count}/{len(chunks)}")

                st.markdown("**Sub-queries generated:**")
                for i, sq in enumerate(retriever.last_sub_queries):
                    st.markdown(f"{i+1}. {sq}")

                st.markdown("**Retrieved chunks:**")
                chunk_data = []
                for doc, util in zip(chunks, util_scores):
                    chunk_data.append({
                        "Page": doc.metadata.get("page", "?"),
                        "Score": doc.metadata.get("similarity_score", "?"),
                        "Utilized": util > UTILIZATION_THRESHOLD,
                        "Util Score": f"{util:.3f}",
                        "Snippet": doc.page_content[:150] + "...",
                    })
                st.dataframe(chunk_data, use_container_width=True)


if __name__ == "__main__":
    main()
