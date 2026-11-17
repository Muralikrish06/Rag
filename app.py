"""Streamlit chat UI for the Space RAG API.

Run (with the API already running on :8000):
    streamlit run frontend/app.py

The UI only talks HTTP to the backend - no LangChain/Chroma imports here. That separation
lets the API scale/deploy independently and keeps the UI a thin client.
"""
import os
import time

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
TIMEOUT_S = 300  # CPU generation can be slow

st.set_page_config(page_title="Space RAG", page_icon="🚀", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)
    except requests.RequestException as e:
        return None, f"API not reachable at {API_URL}: {e}"
    if resp.ok:
        return resp.json(), None
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    return None, f"{resp.status_code}: {detail}"


# --- Sidebar: settings, health, ingestion -----------------------------------
with st.sidebar:
    st.header("Settings")
    types, err = api("GET", "/rag-types")
    rag_options = types["rag_types"] if types else ["naive"]
    rag_type = st.selectbox("RAG type", rag_options,
                            index=rag_options.index("naive") if "naive" in rag_options else 0)
    top_k = st.slider("Top-k chunks", 1, 20, 5)

    st.header("System")
    health, err = api("GET", "/health", params={"rag_type": rag_type})
    if err:
        st.error(err)
    else:
        st.write(("✅" if health["ollama"]["ok"] else "❌") + f" LLM: `{health['ollama'].get('model', '?')}`")
        vs = health["vector_store"]
        st.write(("✅" if vs["ok"] else "❌") + f" Index: `{vs['collection']}` ({vs['chunks']} chunks)")
        st.caption(f"Embeddings: {health['embed_model']}")

    st.header("Ingestion")
    refresh = st.checkbox("Re-download from Wikipedia", value=False)
    if st.button("Run ingestion"):
        _, err = api("POST", "/ingest", json={"rag_type": rag_type, "refresh": refresh})
        if err:
            st.error(err)
        else:
            with st.spinner("Ingesting... (1-3 min)"):
                while True:
                    status, err = api("GET", "/ingest/status")
                    if err or status["state"] != "running":
                        break
                    time.sleep(2)
            if err or status["state"] == "failed":
                st.error(err or status.get("error"))
            else:
                st.success(f"{status['articles']} articles → {status['chunks']} chunks in {status['seconds']}s")

    if st.button("Clear chat"):
        st.session_state.messages = []


# --- Chat -------------------------------------------------------------------
st.title("🚀 Space Exploration RAG")
st.caption("Answers come only from the indexed Wikipedia articles, with numbered citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_details(msg: dict) -> None:
    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for s in msg["sources"]:
                st.markdown(f"**[{s['n']}]** [{s['title']}]({s['url']}) › {s['section']}  "
                            f"<span style='color:gray'>distance {s['distance']:.3f}</span>",
                            unsafe_allow_html=True)
    if msg.get("timings"):
        st.caption(f"{msg['rag_type']} · " + " · ".join(f"{k} {v}s" for k, v in msg["timings"].items()))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_details(msg)

if question := st.chat_input("Ask about missions, rovers, astronauts, agencies..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result, err = api("POST", "/ask", json={"question": question, "rag_type": rag_type, "top_k": top_k},
                              timeout=TIMEOUT_S)
        if err:
            msg = {"role": "assistant", "content": f"⚠️ {err}"}
        else:
            msg = {"role": "assistant", "content": result["answer"], "sources": result["sources"],
                   "timings": result["timings"], "rag_type": result["rag_type"]}
        st.markdown(msg["content"])
        render_details(msg)
    st.session_state.messages.append(msg)
