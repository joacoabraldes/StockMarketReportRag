# -*- coding: utf-8 -*-
import os, json, datetime, math, argparse
from typing import List, Dict, Tuple, Any, Optional
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# ========= Funciones utilitarias =========

def load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return (
            "Eres un asistente que responde SOLO con el CONTEXTO provisto.\n"
            "Puedes sintetizar y combinar fragmentos del contexto.\n"
            "Si falta información crítica para responder con precisión, di exactamente:\n"
            "\"No se puede responder con el contexto disponible.\"\n\n"
            "=== CONTEXTO ===\n{context}\n\n=== PREGUNTA ===\n{question}\n"
        )

def recency_score(d: Optional[datetime.date], today: datetime.date, half_life_days: int) -> float:
    if not d:
        return 0.0
    age_days = max(0, (today - d).days)
    return math.exp(-age_days / max(1, half_life_days))

def sim_from_distance(dist: float) -> float:
    return 1.0 / (1.0 + dist)

def combined_score(dist: float, date_: Optional[datetime.date], today: datetime.date,
                   weight: float, half_life_days: int) -> float:
    s = sim_from_distance(dist)
    r = recency_score(date_, today, half_life_days)
    return (1.0 - weight) * s + weight * r

def parse_date_iso(md: Dict[str, Any]) -> Optional[datetime.date]:
    s = md.get("date_iso")
    if not isinstance(s, str):
        return None
    try:
        y, m, d = map(int, s[:10].split("-"))
        return datetime.date(y, m, d)
    except Exception:
        return None

def expand_queries(q: str) -> list[str]:
    q = q.strip()
    expansions = {q}
    lower = q.lower()
    if "régimen" in lower and ("monetario" in lower or "cambiario" in lower):
        expansions |= {
            lower.replace("monetario", "cambiario"),
            lower.replace("régimen", "esquema"),
            "crawling peg", "crawling-peg", "deslizamiento cambiario",
            "bandas cambiarias", "ancla nominal", "programa monetario",
            "liberalización cambiaria", "CEPO", "crawling",
        }
    if "abril" in lower:
        expansions |= {"mediados de abril", "segunda quincena de abril"}
    return [q] + [e for e in expansions if e != q]

def short_preview(t: str, n=200):
    t = " ".join(t.split())
    return t[:n] + "…" if len(t) > n else t

# ========= Main =========

def main():
    p = argparse.ArgumentParser(description="Consulta RAG (CLI).")
    p.add_argument("query", help="Pregunta del usuario")
    p.add_argument("--persist", default="./chroma_db", help="Ruta de la base de Chroma")
    p.add_argument("--collections", nargs="+", help="Colecciones a usar")
    p.add_argument("--k", type=int, default=10, help="Cantidad total de documentos")
    p.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--recency-weight", type=float, default=0.35)
    p.add_argument("--half-life", type=int, default=30)
    p.add_argument("--min-date", default="")
    p.add_argument("--fetch-factor", type=float, default=2.0)
    p.add_argument("--use-query-expansion", action="store_true")
    p.add_argument("--prompt", default="./prompt_template.txt")
    p.add_argument("--list", action="store_true", help="Listar colecciones disponibles")
    args = p.parse_args()

    client = chromadb.PersistentClient(path=args.persist)
    all_cols = [c.name for c in client.list_collections()]

    if args.list:
        print("Colecciones disponibles:", ", ".join(all_cols) or "(ninguna)")
        return

    cols = args.collections or all_cols
    if not cols:
        print("No hay colecciones.")
        return

    embedder = HuggingFaceEmbeddings(model_name=args.embed_model, encode_kwargs={"normalize_embeddings": True})
    vectordbs = {col: Chroma(client=client, collection_name=col, embedding_function=embedder) for col in cols}

    # Recuperación
    queries = [args.query]
    if args.use_query_expansion:
        queries = expand_queries(args.query)

    ks = [max(1, args.k // len(cols)) for _ in cols]
    fetch_k_each = [max(5, int(k * args.fetch_factor)) for k in ks]
    results = []
    today = datetime.date.today()
    min_date = None
    if args.min_date:
        try:
            y, m, d = map(int, args.min_date[:10].split("-"))
            min_date = datetime.date(y, m, d)
        except Exception:
            pass

    for (col, vdb), k_each in zip(vectordbs.items(), fetch_k_each):
        candidates = []
        for qexp in queries:
            candidates += vdb.similarity_search_with_score(qexp, k=k_each)
        seen = set()
        uniq = []
        for doc, dist in candidates:
            key = (doc.page_content[:120], doc.metadata.get("rel_path"), doc.metadata.get("chunk_id"))
            if key not in seen:
                seen.add(key)
                uniq.append((doc, dist))
        for (doc, dist) in uniq:
            md = dict(doc.metadata or {})
            md["_collection"] = col
            doc.metadata = md
            d_iso = parse_date_iso(md)
            if min_date and d_iso and d_iso < min_date:
                continue
            cscore = combined_score(dist, d_iso, today, args.recency_weight, args.half_life)
            results.append((doc, dist, cscore))

    results.sort(key=lambda x: x[2], reverse=True)
    top = results[:args.k]

    if not top:
        print("⚠️ No se encontraron resultados.")
        return

    # Contexto
    parts = []
    for i, (doc, dist, cscore) in enumerate(top, 1):
        md = doc.metadata
        name = os.path.basename(md.get("rel_path") or md.get("source") or "desconocido")
        d = md.get("date_iso", "?")
        parts.append(f"[{i}] ({md.get('_collection')}) {name} (fecha: {d})\n{doc.page_content}")
    context = "\n\n".join(parts)

    system_prompt = load_prompt(args.prompt).format(context=context, question=args.query)

    # OpenAI
    client_oa = OpenAI()
    resp = client_oa.chat.completions.create(
        model=args.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.query},
        ],
        temperature=args.temperature
    )
    answer = resp.choices[0].message.content
    print("\n🧠 RESPUESTA:\n", answer, "\n")
    print("=== Fuentes ===")
    for i, (doc, dist, cscore) in enumerate(top, 1):
        md = doc.metadata
        name = os.path.basename(md.get("rel_path") or md.get("source") or "desconocido")
        d = md.get("date_iso", "?")
        print(f"[{i}] ({md.get('_collection')}) {name}  dist={dist:.3f}  score={cscore:.3f}  fecha={d}")
        print(" ", short_preview(doc.page_content))
        print()

if __name__ == "__main__":
    main()
