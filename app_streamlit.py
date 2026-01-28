# app_streamlit.py
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import os, json, datetime
from typing import List, Tuple, Dict, Any, Optional
import streamlit as st
import pandas as pd

# langchain HuggingFace embeddings (no Chroma/ingest usage)
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

# no vectordb usage: retrieval will use only `news_text`, `news_urls` and uploaded/computed CSV

# session_state for variations
if "computed_variations" not in st.session_state:
    st.session_state["computed_variations"] = None
if "uploaded_variations" not in st.session_state:
    st.session_state["uploaded_variations"] = None
# date for which computed_variations was generated (YYYY-MM-DD or None)
if "computed_variations_date" not in st.session_state:
    st.session_state["computed_variations_date"] = None
# data date mode (la moda de las fechas de los datos)
if "data_date_mode" not in st.session_state:
    st.session_state["data_date_mode"] = None
# failed tickers
if "failed_tickers" not in st.session_state:
    st.session_state["failed_tickers"] = []
# market status warning
if "market_warning" not in st.session_state:
    st.session_state["market_warning"] = None

# Nota: el checkbox "Generar informe" solo indica usar el `systemprompt_template`.
# No se descarga/ejecuta `compute_variations` hasta que el usuario envíe la consulta.
if generate_report_checkbox:
    st.sidebar.info("Al enviar la consulta se usará el sistema de 'informe' (system prompt). El CSV se descargará solo si es necesario.")

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

    # add news_text as individual news items (do NOT chunk)
    if news_text and news_text.strip():
        # split on blank lines to separate distinct news items; fallback to whole text
        parts = [p.strip() for p in news_text.split("\n\n") if p.strip()]
        if not parts:
            parts = [news_text.strip()]
        for i, part in enumerate(parts, 1):
            local_docs.append(SimpleDoc(page_content=part, metadata={"_collection":"news_user","source":"news_text","news_id":i}))

    # add each URL as a single SimpleDoc (do NOT chunk)
    if news_urls and news_urls.strip():
        for i, url in enumerate([u.strip() for u in news_urls.splitlines() if u.strip()], 1):
            txt = fetch_url_text(url)
            if not txt:
                continue
            local_docs.append(SimpleDoc(page_content=txt, metadata={"_collection":"news_url","source":url,"news_id":i}))

    # add CSV rows (uploaded preferred, else computed)
    from utils import df_to_single_doc  # importa si no lo tenés ya

    # If the user selected 'Generar informe' (use system prompt) we compute the CSV
    # only when the user submits the prompt and only if there's no uploaded CSV.
    df_uploaded = st.session_state.get("uploaded_variations")
    df_computed = st.session_state.get("computed_variations")
    # determine date to request for compute: prefer date in user_input, otherwise date_picker
    extracted_date = None
    try:
        extracted_date = extract_date_from_text(user_input)
    except Exception:
        extracted_date = None
    compute_target_date = None
    if extracted_date:
        compute_target_date = extracted_date
    elif date_picker is not None:
        try:
            compute_target_date = date_picker.strftime("%Y-%m-%d")
        except Exception:
            compute_target_date = None
    else:
        # default to today's date if none provided
        try:
            compute_target_date = datetime.date.today().strftime("%Y-%m-%d")
        except Exception:
            compute_target_date = None

    # If user asked for system prompt / informe, compute CSV if no uploaded CSV
    # or if the already-computed CSV was generated for a different date.
    existing_computed_date = st.session_state.get("computed_variations_date")
    existing_data_date_mode = st.session_state.get("data_date_mode")
    
    if generate_report_checkbox and df_uploaded is None:
        need_compute = False
        if df_computed is None:
            need_compute = True
        else:
            # Comparar fechas: recalcular si la fecha solicitada cambió
            if existing_computed_date != compute_target_date:
                need_compute = True
            # TAMBIÉN recalcular si los datos existentes son de una fecha anterior a la solicitada
            # (por si el mercado ya abrió desde la última vez que se calculó)
            elif existing_data_date_mode and compute_target_date:
                data_mode_date = datetime.datetime.strptime(existing_data_date_mode, "%Y-%m-%d").date()
                requested_date = datetime.datetime.strptime(compute_target_date, "%Y-%m-%d").date()
                if data_mode_date < requested_date:
                    # Los datos son viejos, intentar recalcular por si el mercado ya abrió
                    need_compute = True
                    
        if need_compute:
            try:
                df_out, close, data_date_mode, failed_tickers = compute_variations(TICKER_MAP, lookback="30d", target_date=compute_target_date)
                st.session_state["computed_variations"] = df_out
                # store the date used for this computation (can be None)
                st.session_state["computed_variations_date"] = compute_target_date
                st.session_state["data_date_mode"] = data_date_mode
                st.session_state["failed_tickers"] = failed_tickers
                df_computed = df_out
                
                # Verificar si hay tickers que fallaron
                if failed_tickers:
                    st.warning(f"⚠️ Los siguientes tickers fallaron en la descarga después de reintentos: {', '.join(failed_tickers)}")
                
                # Verificar si el mercado abrió comparando la fecha solicitada con la moda de fechas
                if data_date_mode and compute_target_date:
                    requested_date = datetime.datetime.strptime(compute_target_date, "%Y-%m-%d").date()
                    data_mode_date = datetime.datetime.strptime(data_date_mode, "%Y-%m-%d").date()
                    
                    if requested_date > data_mode_date:
                        # El mercado todavía no abrió para la fecha solicitada
                        warning_msg = f"⚠️ **ATENCIÓN**: El mercado de EEUU todavía no abrió para el {requested_date.strftime('%d/%m/%Y')}. Los datos disponibles son del {data_mode_date.strftime('%d/%m/%Y')}. El informe se generará con los datos del día anterior."
                        st.session_state["market_warning"] = warning_msg
                        st.warning(warning_msg)
                    else:
                        st.session_state["market_warning"] = None
                
                st.info("CSV calculado y almacenado en sesión (computed_variations).")
            except Exception as e:
                st.warning(f"No se pudo descargar el CSV: {e}")

    df_use = df_uploaded or df_computed
    if df_use is not None:
        # crea un único SimpleDoc que contiene TODO el CSV
        single_doc = df_to_single_doc(
            df_use,
            source_name="variacion_diaria.csv",
            extra_meta={"_collection": "tickers_csv", "source": "csv"}
        )
        local_docs.append(single_doc)

    # try to embed local_docs and add to candidates_all (so they are considered for retrieval)
    try:
        embedder_local = embedder
        texts = [d.page_content for d in local_docs]
        if texts:
            # compute query embedding (prefer embed_query, fallback to embed_documents)
            query_emb = None
            try:
                if hasattr(embedder_local, "embed_query"):
                    query_emb = embedder_local.embed_query(user_input)
                else:
                    q_embs = embedder_local.embed_documents([user_input])
                    query_emb = q_embs[0] if q_embs else None
            except Exception:
                query_emb = None

            # compute embeddings for local docs
            try:
                emb_list = embedder_local.embed_documents(texts)
            except Exception:
                emb_list = []

            import numpy as _np
            def cos(a,b):
                a=_np.array(a); b=_np.array(b)
                if a.size==0 or b.size==0: return 0.0
                na=_np.linalg.norm(a); nb=_np.linalg.norm(b)
                if na==0 or nb==0: return 0.0
                return float(_np.dot(a,b)/(na*nb))

            if emb_list and query_emb is not None:
                for d, emb in zip(local_docs, emb_list):
                    sim = cos(query_emb, emb)
                    dist = 1.0 - sim
                    candidates_all.append((d, dist))
            else:
                # fallback: include local_docs with neutral distance so they appear in context
                for d in local_docs:
                    candidates_all.append((d, 1.0))
    except Exception:
        # ignore embed errors but ensure local_docs are still considered
        for d in local_docs:
            candidates_all.append((d, 1.0))

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
    # Ensure the date for the summary is present in the question passed to the system prompt.
    # Prefer an explicit date in the user_input (extracted), otherwise use the date_picker if provided.
    try:
        extracted_date = extract_date_from_text(user_input)
    except Exception:
        extracted_date = None
    
    # Diccionario para días de la semana en español
    DIAS_SEMANA = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    
    # Verificar si el mercado no abrió y debemos usar la fecha de los datos reales
    market_warning = st.session_state.get("market_warning")
    data_date_mode = st.session_state.get("data_date_mode")
    
    # Determinar la fecha real a usar para el informe
    # Si el mercado no abrió, usar la fecha de los datos (data_date_mode)
    if market_warning and data_date_mode:
        # Usar la fecha de los datos reales (día anterior)
        report_date_obj = datetime.datetime.strptime(data_date_mode, "%Y-%m-%d").date()
        dia_semana = DIAS_SEMANA[report_date_obj.weekday()]
        qdate = f"{report_date_obj.strftime('%d/%m/%Y')} ({dia_semana})"
        market_note = f"\n\n[NOTA: El mercado de EEUU todavía no abrió para la fecha solicitada. Los datos corresponden al {qdate}. Usá esta fecha para el informe.]"
    else:
        # Usar la fecha extraída, date_picker, o la fecha de los datos si está disponible
        if extracted_date:
            report_date_obj = datetime.datetime.strptime(extracted_date, "%Y-%m-%d").date()
        elif date_picker is not None:
            report_date_obj = date_picker
        elif data_date_mode:
            # Usar la fecha real de los datos del CSV
            report_date_obj = datetime.datetime.strptime(data_date_mode, "%Y-%m-%d").date()
        else:
            report_date_obj = datetime.date.today()
        dia_semana = DIAS_SEMANA[report_date_obj.weekday()]
        qdate = f"{report_date_obj.strftime('%d/%m/%Y')} ({dia_semana})"
        market_note = ""
    
    # Construir question_for_prompt con la fecha y día de la semana correctos
    if user_input and user_input.strip():
        question_for_prompt = f"{user_input} para {qdate}"
    else:
        question_for_prompt = f"Generá resumen para {qdate}"
    
    # Agregar nota de mercado si corresponde
    if market_note:
        question_for_prompt = question_for_prompt + market_note
    
    if use_system_prompt:
        # build CSV textual block
        if df_use is None:
            st.warning("No hay CSV disponible: subí un CSV o presioná 'Calcular CSV' en la barra lateral.")
        csv_block = format_variations_for_prompt(df_use) if df_use is not None else ""
        
        # Filtrar el CSV del context para no duplicarlo (ya está en csv_block)
        context_without_csv = "\n\n".join([
            part for part in context_parts 
            if "tickers_csv" not in part and "variacion_diaria.csv" not in part
        ])
        
        # final context: CSV + news/docs (sin duplicar el CSV)
        if context_without_csv.strip():
            merged_context = csv_block + "\n\n" + context_without_csv
        else:
            merged_context = csv_block
            
        try:
            with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as fh:
                system_prompt_template = fh.read()
        except Exception:
            st.error("No se pudo cargar system prompt.")
            system_prompt_template = "{context}\n\n{question}"
        system_prompt = system_prompt_template.format(context=merged_context, question=question_for_prompt)
    else:
        # normal RAG prompt
        try:
            with open(RAG_PROMPT_PATH, "r", encoding="utf-8") as fh:
                rag_prompt_template = fh.read()
        except Exception:
            rag_prompt_template = "{context}\n\n{question}"
        system_prompt = rag_prompt_template.format(context=context, question=question_for_prompt)

    # Preparar los mensajes que se enviarán al modelo
    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_for_prompt}
    ]

    # DEBUG: Mostrar exactamente lo que recibe el modelo
    with st.expander("🔍 DEBUG: Mensajes enviados al modelo", expanded=False):
        st.markdown("**System:**")
        st.code(llm_messages[0]["content"], language="text")
        st.markdown("**User:**")
        st.code(llm_messages[1]["content"], language="text")

    # call LLM (OpenAI) to generate answer
    with st.spinner("Generando respuesta con LLM..."):
        if OpenAI is None:
            answer = "⚠️ OpenAI SDK no disponible en el entorno. Configura OPENAI_API_KEY o instala openai."
        else:
            try:
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=openai_model,
                    messages=llm_messages,
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
