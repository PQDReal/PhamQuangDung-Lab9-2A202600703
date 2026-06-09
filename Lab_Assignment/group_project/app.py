"""Streamlit chatbot for the group RAG demo."""

from __future__ import annotations

import streamlit as st

from rag_service import answer_question, source_label


st.set_page_config(page_title="Drug Law RAG", layout="wide")
st.title("Drug Law & News RAG Chatbot")

with st.sidebar:
    st.header("Retrieval")
    top_k = st.slider("Top K", min_value=3, max_value=8, value=5)
    st.caption("Answers are generated from the Task 9/10 pipeline with citations.")
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{source_label(source, i)}**")
                    st.write(source.get("content", "")[:700])

prompt = st.chat_input("Hỏi về pháp luật ma túy hoặc tin tức liên quan...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence..."):
            result = answer_question(prompt, st.session_state.messages[:-1], top_k=top_k)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for i, source in enumerate(result["sources"], 1):
                st.markdown(f"**{source_label(source, i)}**")
                st.write(source.get("content", "")[:700])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
