from typing import List, Dict, Any
import streamlit as st
from logger import Logger

from backend.core import run_llm

logger = Logger("app.log", color_file=True)


def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))  # type: ignore
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None  # type: ignore
    ]


st.set_page_config(page_title="Langchain Documentation Helper", layout="centered")
st.title("Langchain Documentation Helper")


with st.sidebar:
    st.subheader("Session")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I'll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]


for msg in st.session_state.messages:             # type: ignore
    with st.chat_message(msg['role']):            # type: ignore
        st.markdown(msg['content'])               # type: ignore

        if msg.get("sources"):                    # type: ignore
            with st.expander("Sources"):
                for s in msg["sources"]:          # type: ignore
                    st.markdown(f"- {s}")


prompt = st.chat_input("Ask a question about LangChain...")

if prompt:
    logger.info(f"Prompt: {prompt}")

    st.session_state.messages.append(             # type: ignore
        {
            'role': 'user',
            'content': prompt,
            'sources': [],
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer..."):
                result: Dict[str, Any] = run_llm(
                    query=prompt
                )

                answer = str(result.get("answer", "")).strip() or str("No answer returned. Please try again.") # type: ignore

                sources = _format_sources(result.get("context", []))
                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")
                
                st.session_state.messages.append(             # type: ignore
                    {
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources
                    }
                )

        except Exception as e:
            st.error("Failed to generate a response")
            st.exception(e)
