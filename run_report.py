#!/usr/bin/env python3
# run_report.py
# ─────────────────────────────────────────────────────────────────────
# Script headless para generar informes de mercado vía cron-job.
# Lee los datos de cierre desde un CSV local (bonos, futuros, etc.)
# en lugar de descargar de Yahoo Finance.
#
# Uso:
#   python run_report.py --csv precios.csv
#   python run_report.py --csv precios.csv --urls https://news1.com https://news2.com
#   python run_report.py --csv US                        
#   python run_report.py --csv AR 
#
#
# Salida: archivo .txt con el informe generado.
# ─────────────────────────────────────────────────────────────────────

import argparse
import datetime
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package is required.  pip install openai")
    sys.exit(1)

import pandas as pd

from config.market_config import get_market_config, MarketConfig
from core.evaluator import (
    load_dataset,
    build_eval_prompt,
    call_evaluator,
    extract_date_from_prompt,
    find_reference_for_date,
)
from core.utils import fetch_url_text, format_variations_for_prompt
from core.compute_variations import compute_variations
from core.debug_logger import DebugSession

DIAS_SEMANA = [
    "lunes", "martes", "miércoles", "jueves",
    "viernes", "sábado", "domingo",
]


# ── Formateo genérico de CSV → bloque de texto para el prompt ────────

def format_csv_for_prompt(df: pd.DataFrame, source_name: str = "datos_mercado.csv") -> str:
    """
    Convierte un DataFrame arbitrario en un bloque pipe-separated listo
    para inyectar en el system prompt.  Acepta cualquier conjunto de columnas.

    Si el CSV tiene columna 'Var_diaria_%' los valores se formatean con
    coma decimal y sufijo '%'.
    """
    SEP = " | "

    def fmt_pct(v):
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        s = f"{v}"
        s = s.replace(".", ",")
        if not s.endswith("%"):
            s += "%"
        return s

    cols = list(df.columns)
    header = SEP.join(str(c) for c in cols)
    rows = [header]

    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r.get(c, "")
            if c == "Var_diaria_%":
                vals.append(fmt_pct(v))
            elif v is None or (isinstance(v, float) and pd.isna(v)):
                vals.append("")
            else:
                vals.append(str(v))
        rows.append(SEP.join(vals))

    body = "\n".join(rows)
    return f"Fuente:{source_name} Tickers:\n{body}"


# ── Carga y validación del CSV ───────────────────────────────────────

def load_market_csv(path: str) -> pd.DataFrame:
    """
    Lee el CSV con los datos de mercado.  Intenta detectar el separador
    (coma o punto-y-coma).  Valida que exista al menos la columna 'Ticker'.
    """
    if not os.path.isfile(path):
        print(f"ERROR: CSV no encontrado: {path}")
        sys.exit(1)

    # Intentar detectar separador
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()

    sep = ","
    if ";" in first_line and "," not in first_line:
        sep = ";"
    elif first_line.count(";") > first_line.count(","):
        sep = ";"

    df = pd.read_csv(path, sep=sep, encoding="utf-8")

    # Normalizar nombres de columnas (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    if "Ticker" not in df.columns:
        # Intentar con la primera columna como Ticker
        first_col = df.columns[0]
        print(f"⚠️  Columna 'Ticker' no encontrada. Usando '{first_col}' como Ticker.")
        df = df.rename(columns={first_col: "Ticker"})

    # Si tiene Close_last y Close_prev pero no Var_diaria_%, calcularla
    if (
        "Close_last" in df.columns
        and "Close_prev" in df.columns
        and "Var_diaria_%" not in df.columns
    ):
        try:
            df["Var_diaria_%"] = (
                (df["Close_last"].astype(float) / df["Close_prev"].astype(float) - 1) * 100
            ).round(2)
            print("ℹ️  Var_diaria_% calculada automáticamente desde Close_last / Close_prev.")
        except Exception:
            pass

    return df


# ── System prompt builder ────────────────────────────────────────────

def build_system_prompt(
    template_path: str,
    csv_block: str,
    question: str,
    news_text: str = "",
    news_urls: Optional[List[str]] = None,
) -> str:
    """Carga template y rellena {context} y {question}."""
    extra = []
    if news_text and news_text.strip():
        extra.append(news_text.strip())
    if news_urls:
        for url in news_urls:
            txt = fetch_url_text(url)
            if txt:
                extra.append(txt)

    context = csv_block
    if extra:
        context += "\n\n" + "\n\n".join(extra)

    if not os.path.isfile(template_path):
        print(f"WARNING: template no encontrado en {template_path}, usando fallback")
        return f"{context}\n\n{question}"

    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()

    return template.format(context=context, question=question)


# ── Loop de generación + evaluación ──────────────────────────────────

def run_generation(
    config: MarketConfig,
    csv_block: str,
    report_date: datetime.date,
    news_text: str = "",
    news_urls: Optional[List[str]] = None,
    temperature: float = 0.0,
    prompt_template: Optional[str] = None,
) -> tuple[str, float, DebugSession]:
    """
    Genera un informe y lo evalúa iterativamente.
    Retorna (respuesta_final, score_final, debug_session).
    """
    debug = DebugSession()
    target_date_str = report_date.strftime("%Y-%m-%d")
    debug.start(market_id=config.market_id, target_date=target_date_str)

    dia = DIAS_SEMANA[report_date.weekday()]
    qdate = f"{report_date.strftime('%d/%m/%Y')} ({dia})"
    question = f"Generá resumen para {qdate}"

    template_path = prompt_template or config.system_prompt_path
    system_prompt = build_system_prompt(
        template_path, csv_block, question, news_text, news_urls,
    )
    user_message = question

    # ── Dataset de evaluación ────────────────────────────────────
    ds = load_dataset(config.threshold_dataset_path)
    few_shot = ds[:3]
    csv_for_eval = csv_block
    if news_text:
        csv_for_eval += f"\n\nNoticias:\n{news_text}"
    query_date = extract_date_from_prompt(csv_block)
    reference_response = None
    if query_date:
        ref_entry = find_reference_for_date(ds, query_date)
        if ref_entry:
            reference_response = ref_entry.get("response")
            print(f"📊 Referencia encontrada en dataset para {query_date}")

    # ── Loop writer / evaluator ──────────────────────────────────
    client = OpenAI()
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    PLATEAU_THRESHOLD = 0.02
    GOOD_ENOUGH_AFTER = 3
    GOOD_ENOUGH_SCORE = 0.88

    best_answer = ""
    best_score = 0.0
    accumulated_feedback: list[dict] = []
    eval_history: list[dict] = []
    consecutive_plateau = 0
    prev_score = 0.0

    for attempt in range(config.max_eval_retries):
        retry_temp = min(temperature + (attempt * 0.03), temperature + 0.12)
        print(f"\n🔄 Iteración {attempt + 1}/{config.max_eval_retries} (temp={retry_temp:.2f})…")

        if attempt == 0:
            send_messages = base_messages.copy()
        else:
            last_fb = accumulated_feedback[-1]
            feedback_block = f"Score: {last_fb['score']:.2f}\n"
            if last_fb.get("reason"):
                feedback_block += f"Análisis: {last_fb['reason']}\n"
            if last_fb.get("datos_correctos") is False:
                feedback_block += "⚠️ HAY VALORES NUMÉRICOS INCORRECTOS. Verificá cada dato contra el CSV.\n"
            if last_fb.get("narrativa_quality"):
                feedback_block += f"Calidad narrativa: {last_fb['narrativa_quality']}\n"
            if last_fb.get("mejoras"):
                feedback_block += "MEJORAS PRIORITARIAS:\n"
                for mejora in last_fb["mejoras"]:
                    feedback_block += f"  • {mejora}\n"

            retry_user_msg = f"""{user_message}

=== DATOS CSV DE REFERENCIA (verificá tus valores contra estos) ===
{csv_for_eval}

=== TU RESPUESTA ANTERIOR (intento {attempt}, score {best_score:.2f}) ===
{best_answer}

=== FEEDBACK DEL EVALUADOR ===
{feedback_block}

=== INSTRUCCIONES DE MEJORA ===
Tomá tu respuesta anterior como base y EDITALA aplicando las mejoras pedidas.
NO reescribas desde cero: mejorá lo que ya tenés sin perder lo que estaba bien.

1. Aplicá las mejoras prioritarias listadas arriba.
2. Verificá que TODOS los valores numéricos coincidan exactamente con el CSV.
3. Mejorá la narrativa: explicá POR QUÉ se movió el mercado, conectando causas y efectos.
4. Integrá noticias y datos macro como causas de los movimientos.
5. No pierdas información correcta que ya tenías en la versión anterior.

Generá la respuesta mejorada:"""
            send_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_user_msg},
            ]

        resp = client.chat.completions.create(
            model=config.openai_model,
            messages=send_messages,
            temperature=min(retry_temp, 1.0),
        )
        answer = resp.choices[0].message.content

        # Evaluar
        eval_prompt = build_eval_prompt(
            few_shot, csv_for_eval, answer, reference_response,
            iteration=attempt + 1,
            previous_attempts=eval_history if eval_history else None,
        )
        eval_res, eval_raw = call_evaluator(
            eval_prompt, openai_model=config.openai_model, temperature=0.0,
        )
        score = eval_res.get("score", 0.0)
        reason = eval_res.get("reason", "")

        writer_user = send_messages[-1]["content"]
        debug.add_iteration(
            iteration=attempt + 1,
            writer_system=system_prompt,
            writer_user=writer_user,
            writer_response=answer,
            writer_temperature=retry_temp,
            evaluator_prompt=eval_prompt,
            evaluator_raw=eval_raw,
            eval_score=score,
            eval_ok=score >= config.min_eval_score,
            eval_reason=reason,
        )

        emoji = "✅" if score >= config.min_eval_score else "⚠️"
        print(f"  {emoji} Score: {score:.2f}  —  {reason[:120]}")

        if score > best_score:
            best_score = score
            best_answer = answer

        if score >= config.min_eval_score:
            break

        # Plateau
        improvement = score - prev_score if attempt > 0 else score
        prev_score = score

        if attempt > 0 and improvement < PLATEAU_THRESHOLD:
            consecutive_plateau += 1
        else:
            consecutive_plateau = 0

        if consecutive_plateau >= 2 and best_score >= GOOD_ENOUGH_SCORE:
            print(f"  ⏹️  Plateau detectado (score estable en ~{best_score:.2f}). Aceptando mejor resultado.")
            break
        if attempt + 1 >= GOOD_ENOUGH_AFTER and best_score >= GOOD_ENOUGH_SCORE:
            print(f"  ⏹️  Suficientemente bueno tras {attempt + 1} intentos (score={best_score:.2f}). Aceptando.")
            break

        if attempt < config.max_eval_retries - 1:
            accumulated_feedback.append({
                "score": score,
                "reason": reason,
                "datos_correctos": eval_res.get("datos_correctos", True),
                "narrativa_quality": eval_res.get("narrativa_quality", ""),
                "mejoras": eval_res.get("mejoras", []),
            })
            eval_history.append({
                "iteration": attempt + 1,
                "response": answer,
                "score": score,
                "reason": reason,
                "mejoras": eval_res.get("mejoras", []),
            })

    debug.finish(final_answer=best_answer, final_score=best_score)
    return best_answer, best_score, debug


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera un informe de mercado a partir de un CSV local con datos "
            "de cierre (bonos, futuros, acciones, etc.) y opcionalmente URLs "
            "de noticias.  Pensado para correr vía cron-job."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_report.py --csv precios.csv
  python run_report.py --csv precios.csv --urls https://news1.com https://news2.com
  python run_report.py --csv precios.csv --market AR --date 2026-02-20
  python run_report.py --csv precios.csv --prompt prompts/mi_template.txt --out informe.txt
  python run_report.py --csv US                                     # yfinance US
  python run_report.py --csv AR --date 2026-02-20                   # yfinance AR

El CSV debe tener al menos una columna 'Ticker' (o la primera se usa como tal).
Columnas recomendadas: Ticker, Close_last, Close_prev, Var_diaria_%, Fecha_last, Plazo, Tipo.
Si hay Close_last y Close_prev pero falta Var_diaria_%%, se calcula automáticamente.
        """,
    )
    parser.add_argument(
        "--csv", required=True,
        help="Ruta al CSV con datos de mercado, o 'US'/'AR' para descargar de yfinance.",
    )
    parser.add_argument(
        "--urls", nargs="*", default=[],
        help="URLs de noticias separadas por espacio.",
    )
    parser.add_argument(
        "--news", default="",
        help="Texto inline de noticias (entre comillas).",
    )
    parser.add_argument(
        "--market", default="US",
        help="ID de mercado para config (US, AR, …). Default: US.",
    )
    parser.add_argument(
        "--date", default=None,
        help="Fecha del informe YYYY-MM-DD. Default: hoy.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Ruta de salida .txt. Default: informe_<market>_<fecha>.txt.",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Ruta a un template de system prompt personalizado (overrides market config).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperatura del LLM. Default: 0.0.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Imprimir resumen de iteraciones del loop writer/evaluator.",
    )

    args = parser.parse_args()

    # Determinar fecha
    if args.date:
        try:
            report_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"ERROR: formato de fecha inválido '{args.date}'. Usar YYYY-MM-DD.")
            sys.exit(1)
    else:
        report_date = datetime.date.today()

    date_str = report_date.strftime("%Y-%m-%d")

    # Detectar si --csv es un market ID (US, AR) o un archivo
    csv_arg = args.csv.strip()
    is_market_id = csv_arg.upper() in ("US", "AR")

    if is_market_id:
        # Usar mercado del --csv (ignora --market si se pasó un market ID directo)
        market_id = csv_arg.upper()
        config = get_market_config(market_id)
        print(f"🦜 PARROT — Informe {config.market_name} para {date_str} (yfinance)")
        print(f"📥 Descargando datos de {config.market_name} vía yfinance…")
        df_out, _close, data_date_mode, failed_tickers = compute_variations(
            config.ticker_map, lookback=config.default_lookback, target_date=date_str,
        )
        if failed_tickers:
            print(f"⚠️  Tickers fallidos: {', '.join(failed_tickers)}")
        # Si el mercado no abrió, data_date_mode tiene la fecha real
        if data_date_mode:
            real_date = datetime.datetime.strptime(data_date_mode, "%Y-%m-%d").date()
            if real_date < report_date:
                print(f"⚠️  Mercado sin datos para {date_str}. Usando datos de {data_date_mode}.")
            report_date = real_date
        print(f"   {len(df_out)} instrumentos cargados.")
        csv_block = format_variations_for_prompt(df_out)
    else:
        # CSV local
        config = get_market_config(args.market)
        print(f"🦜 PARROT — Informe {config.market_name} para {date_str} (CSV local)")
        print(f"📥 Leyendo CSV: {csv_arg}")
        df = load_market_csv(csv_arg)
        print(f"   {len(df)} instrumentos cargados.  Columnas: {list(df.columns)}")
        csv_block = format_csv_for_prompt(df, source_name=os.path.basename(csv_arg))

    # Generar informe
    answer, score, debug_session = run_generation(
        config=config,
        csv_block=csv_block,
        report_date=report_date,
        news_text=args.news,
        news_urls=args.urls or None,
        temperature=args.temperature,
        prompt_template=args.prompt,
    )

    # Guardar resultado
    out_path = args.out or f"informe_{config.market_id}_{date_str}.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(answer)
    print(f"\n📄 Informe guardado en {out_path}  (score={score:.2f})")

    # Debug
    if args.debug and debug_session.iterations:
        print(f"\n🔄 Iteraciones: {debug_session.total_iterations}")
        for it in debug_session.iterations:
            emoji = "✅" if it.eval_ok else "⚠️"
            print(f"  {emoji} #{it.iteration}  score={it.eval_score:.2f}  reason={it.eval_reason[:100]}")

    # Exit code
    if score < config.min_eval_score:
        print(f"\n⚠️  Score final {score:.2f} por debajo del umbral {config.min_eval_score}.")
        sys.exit(2)


if __name__ == "__main__":
    main()
