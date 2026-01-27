import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from embedder import LocalEmbedder
from openai import OpenAI

DB_DIR = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
PROMPT_PATH = os.environ.get("PROMPT_TEMPLATE", "./prompt_template.txt")

def load_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as fh:
        return fh.read()

def build_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "desconocido")
        pg  = d.metadata.get("page", d.metadata.get("page_number", ""))
        tag = f"{src}#{pg}" if pg != "" else f"{src}"
        parts.append(f"[{i}] ({tag})\n{d.page_content}")
    return "\n\n".join(parts)

class AskRequest(BaseModel):
    query: str
    k: int = 5
    temperature: float = 0.2
    openai_model: str = os.environ.get("OPENAI_MODEL","gpt-4o-mini")
    embed_model: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

app = FastAPI(title="RAG mínimo")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        # Embeddings locales
        embedder = LocalEmbedder(model_name=req.embed_model) if req.embed_model else LocalEmbedder()
        hf = HuggingFaceEmbeddings(model_name=embedder.model_name, encode_kwargs={"normalize_embeddings": True})

        vectordb = Chroma(embedding_function=hf, persist_directory=DB_DIR)
        retriever = vectordb.as_retriever(search_kwargs={"k": req.k})
        docs = retriever.get_relevant_documents(req.query)

        context = build_context(docs)
        prompt_tmpl = load_prompt()
        system_prompt = prompt_tmpl.format(context=context, question=req.query)

        client = OpenAI()
        resp = client.chat.completions.create(
            model=req.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.query}
            ],
            temperature=req.temperature,
        )
        answer = resp.choices[0].message.content
        sources = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "desconocido")
            pg  = d.metadata.get("page", d.metadata.get("page_number", ""))
            tag = f"{src}#{pg}" if pg != "" else f"{src}"
            sources.append(f"[{i}] {tag}")
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
