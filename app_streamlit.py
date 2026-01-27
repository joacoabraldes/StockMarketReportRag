# app_streamlit.py
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import os, json, datetime
from typing import List, Tuple, Dict, Any, Optional
import streamlit as st
import pandas as pd

# chroma/langchain imports if you use them
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# openai optional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from utils import clean_text, clean_keep_ascii_marks, fetch_url_text, chunk_text, SimpleDoc, extract_date_from_text, format_variations_for_prompt
from evaluator import load_dataset, build_eval_prompt, call_evaluator
from compute_variations import TICKER_MAP, compute_variations

# --------- Config / defaults ----------
st.set_page_config(page_title="PARROT RAG", layout="wide")
st.title("🦜 PARROT RAG — informe de rueda (CSV -> resumen)")

DEFAULT_PERSIST   = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
DEFAULT_EMBED     = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_OAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_TEMPLATE", "./systemprompt_template.txt")
RAG_PROMPT_PATH = os.environ.get("PROMPT_TEMPLATE", "./prompt_template.txt")
THRESHOLD_PATH = "./threshold_dataset.jsonl"

# --------- Sidebar ----------
st.sidebar.header("⚙️ Configuración")
persist_dir  = st.sidebar.text_input("Ruta de Chroma (persist)", value=DEFAULT_PERSIST)
client = chromadb.PersistentClient(path=persist_dir)
collections = [c.name for c in client.list_collections()]
if not collections:
    st.sidebar.warning("No hay colecciones. Ingestá documentos con ingest.py")
selected_cols = st.sidebar.multiselect("Colecciones a consultar", options=collections, default=collections[:1] if collections else [])

# news inputs
news_text = st.sidebar.text_area("Texto de noticias (opcional)", value="", height=150)
news_urls = st.sidebar.text_area("URLs de noticias (una por línea)", value="", height=120)

k_total = st.sidebar.slider("Top-k total", 1, 30, 10, 1)
temperature = st.sidebar.slider("Temperature (OpenAI)", 0.0, 1.0, 0.0, 0.1)
openai_model = st.sidebar.text_input("Modelo OpenAI", value=DEFAULT_OAI_MODEL)
embed_model = st.sidebar.text_input("Modelo de Embeddings (HF)", value=DEFAULT_EMBED)
history_path = st.sidebar.text_input("Archivo de historial (JSONL)", value="./history/chat_history.jsonl")
show_sources = st.sidebar.checkbox("Mostrar fuentes", value=True)
show_scores = st.sidebar.checkbox("Mostrar score", value=False)
show_preview = st.sidebar.checkbox("Mostrar preview", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Opciones para informe")
generate_report_checkbox = st.sidebar.checkbox("Generar informe de la rueda", value=False)
date_picker = st.sidebar.date_input("Fecha del informe (default:hoy)", value=None)
st.sidebar.markdown("---")
st.sidebar.caption("Requiere OPENAI_API_KEY si querés usar verificador OpenAI.")

# --------- caches ----------
@st.cache_resource(show_spinner=False)
def get_embedder_cached(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})

embedder = get_embedder_cached(embed_model)

# vectordb per selected col
def get_vectordbs(persist_path, selected_cols, embedder):
    out = {}
    for col in selected_cols:
        out[col] = Chroma(client=chromadb.PersistentClient(path=persist_path), collection_name=col, embedding_function=embedder)
    return out

vectordbs = get_vectordbs(persist_dir, selected_cols, embedder)

# session_state for variations
if "computed_variations" not in st.session_state:
    st.session_state["computed_variations"] = None
if "uploaded_variations" not in st.session_state:
    st.session_state["uploaded_variations"] = None

# compute CSV via button
if generate_report_checkbox:
    try:
        df_out, close = compute_variations(TICKER_MAP, lookback="30d", target_date=date_picker)
        st.session_state["computed_variations"] = df_out
        st.success("CSV generado y almacenado en sesión (computed_variations).")
    except Exception as e:
        st.error(f"Error calculando CSV: {e}")

# option to upload csv
uploaded = st.sidebar.file_uploader("Subir CSV de variaciones (opcional)", type=["csv"])
if uploaded:
    try:
        df_uploaded = pd.read_csv(uploaded)
        st.session_state["uploaded_variations"] = df_uploaded
        st.sidebar.success("CSV subido y guardado en sesión.")
    except Exception as e:
        st.sidebar.error(f"Error al leer CSV: {e}")

# ----- Chat render previo (history) -----
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if show_sources and m.get("sources"):
            with st.expander("Fuentes"):
                for s in m.get("sources", []):
                    st.text(clean_text(s))

# ----- Main interaction -----
user_input = st.chat_input("Escribí tu pregunta (ej: \"Generá resumen para 16/01/2026\")")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # build retrieval context (very similar to previous pipeline)
    candidates_all = []
    local_docs = []

    # fetch content from selected chroma collections
    for col, vdb in vectordbs.items():
        try:
            for qexp in [user_input]:
                for doc, dist in vdb.similarity_search_with_score(qexp, k=max(5, k_total)):
                    md = dict(doc.metadata or {})
                    md["_collection"] = col
                    doc.metadata = md
                    candidates_all.append((doc, dist))
        except Exception:
            continue

    # add news_text chunks
    if news_text and news_text.strip():
        for i, ch in enumerate(chunk_text(news_text, max_chars=1200), 1):
            local_docs.append(SimpleDoc(page_content=ch, metadata={"_collection":"news_user","source":"news_text","chunk_id":i}))

    # add news_urls texts
    if news_urls and news_urls.strip():
        for i, url in enumerate([u.strip() for u in news_urls.splitlines() if u.strip()],1):
            txt = fetch_url_text(url)
            if not txt: continue
            for j, ch in enumerate(chunk_text(txt, max_chars=1200),1):
                local_docs.append(SimpleDoc(page_content=ch, metadata={"_collection":"news_url","source":url,"chunk_id":f"{i}.{j}"}))

    # add CSV rows (uploaded preferred, else computed)
    from utils import df_to_single_doc  # importa si no lo tenés ya

    df_use = st.session_state.get("uploaded_variations") or st.session_state.get("computed_variations")
    if df_use is not None:
        # crea un único SimpleDoc que contiene TODO el CSV
        single_doc = df_to_single_doc(
            df_use,
            source_name="variacion_diaria.csv",
            extra_meta={"_collection": "tickers_csv", "source": "csv"}
        )
        local_docs.append(single_doc)

    # try to embed local_docs and add to candidates_all (so they compete with chroma)
    try:
        embedder_local = embedder
        # embed query if function present
        if hasattr(embedder_local, "embed_query"):
            query_emb = embedder_local.embed_query(user_input)
            texts = [d.page_content for d in local_docs]
            if texts:
                emb_list = embedder_local.embed_documents(texts)
                import numpy as _np
                def cos(a,b):
                    a=_np.array(a); b=_np.array(b)
                    if a.size==0 or b.size==0: return 0.0
                    na=_np.linalg.norm(a); nb=_np.linalg.norm(b)
                    if na==0 or nb==0: return 0.0
                    return float(_np.dot(a,b)/(na*nb))
                for d, emb in zip(local_docs, emb_list):
                    sim = cos(query_emb, emb)
                    dist = 1.0 - sim
                    candidates_all.append((d, dist))
    except Exception:
        # ignore embed errors
        pass

    # Build top docs (simple scoring: use dist as score proxy, no recency for brevity)
    scored = []
    for doc, dist in candidates_all:
        scored.append((doc, dist, 1.0/(1.0+dist)))
    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:k_total]
    top_docs = [doc for (doc,_,_) in top]

    # Build context string for RAG
    context_parts = []
    for i, doc in enumerate(top_docs,1):
        context_parts.append(f"[{i}] ({doc.metadata.get('_collection')})\n{doc.page_content}")
    context = "\n\n".join(context_parts)


    tools = [
    {
        "type": "function",
        "name": "get_horoscope",
        "description": "Get today's horoscope for an astrological sign.",
        "parameters": {
            "type": "object",
            "properties": {
                "sign": {
                    "type": "string",
                    "description": "An astrological sign like Taurus or Aquarius",
                },
            },
            "required": ["sign"],
        },
    },
    ]

    # If generate_report_checkbox is True -> use systemprompt_template and include the CSV (if exists)
    use_system_prompt = generate_report_checkbox and os.path.exists(SYSTEM_PROMPT_PATH)
    if use_system_prompt:
        # build CSV textual block
        if df_use is None:
            st.warning("No hay CSV disponible: subí un CSV o presioná 'Calcular CSV' en la barra lateral.")
        csv_block = format_variations_for_prompt(df_use) if df_use is not None else ""
        # final context: CSV + news/docs
        merged_context = csv_block + "\n\n" + context
        try:
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as fh:
                system_prompt_template = fh.read()
        except Exception:
            st.error("No se pudo cargar system prompt.")
            system_prompt_template = "{context}\n\n{question}"
        system_prompt = system_prompt_template.format(context=merged_context, question=user_input)
    else:
        # normal RAG prompt
        try:
            with open(RAG_PROMPT_PATH, "r", encoding="utf-8") as fh:
                rag_prompt_template = fh.read()
        except Exception:
            rag_prompt_template = "{context}\n\n{question}"
        system_prompt = rag_prompt_template.format(context=context, question=user_input)

    # call LLM (OpenAI) to generate answer
    with st.spinner("Generando respuesta con LLM..."):
        if OpenAI is None:
            answer = "⚠️ OpenAI SDK no disponible en el entorno. Configura OPENAI_API_KEY o instala openai."
        else:
            try:
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=temperature
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"⚠️ Error llamando a OpenAI: {e}"

    # show sources
    sources = []
    for i, (doc, dist, cscore) in enumerate(top, 1):
        tag = f"{doc.metadata.get('_collection')} - {doc.metadata.get('source','')}"
        preview = (doc.page_content[:240] + "…") if len(doc.page_content)>240 else doc.page_content
        sources.append(f"[{i}] ({tag})\n> {preview}")

    # render answer
    with st.chat_message("assistant"):
        st.markdown(answer)
        if show_sources and sources:
            with st.expander("Fuentes"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})

    # If we generated an "informe", run evaluator and show score
    if use_system_prompt and df_use is not None:
        ds = load_dataset(THRESHOLD_PATH)
        few_shot = ds[:3]
        eval_prompt = build_eval_prompt(few_shot, user_input, answer)
        eval_res = call_evaluator(eval_prompt, openai_model=openai_model, temperature=0.0)
        # normalize
        score = eval_res.get("score", 0.0)
        ok = bool(eval_res.get("ok", False))
        reason = eval_res.get("reason", "")
        st.sidebar.markdown("### Evaluación del informe")
        st.sidebar.write(f"Score: **{score:.2f}** — OK: **{ok}**")
        st.sidebar.write(f"Motivo: {reason}")

        # option to append generated example to dataset
        if st.sidebar.button("Agregar ejemplo generado al dataset (threshold_dataset.jsonl)"):
            try:
                rec = {"prompt": format_variations_for_prompt(df_use) + " Noticias: " + (news_text or " "), "response": answer}
                with open(THRESHOLD_PATH, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st.sidebar.success("Ejemplo agregado.")
            except Exception as e:
                st.sidebar.error(f"Error guardando dataset: {e}")

    # save history
    try:
        os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "timestamp": datetime.datetime.now().isoformat(),
                "query": user_input,
                "answer": answer,
                "sources": sources,
                "generate_report": bool(generate_report_checkbox)
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
