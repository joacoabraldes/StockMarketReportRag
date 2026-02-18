# evaluator.py
# Carga dataset JSONL, arma few-shot y evalúa respuesta comparando con referencia.
# Includes decimal-format normalisation to avoid false negatives when the LLM
# uses a comma as decimal separator vs a dot (or vice-versa).
import os, json, re
from typing import List, Dict, Optional, Tuple
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

THRESHOLD_DATASET_PATH = "./data/threshold_dataset.jsonl"


# ─── Decimal / format normalisation helpers ───────────────────────────

_PCT_RE = re.compile(
    r"[+\-−]?\s*\d{1,6}[.,]\d{1,4}\s*%"      # e.g. -0,84%  +1.31%  0,04%
)

def normalize_decimal(text: str) -> str:
    """Replace comma-decimal percentages with dot-decimal so comparisons
    between '0,84%' and '0.84%' don't flag false differences.
    Also strips the optional '+' sign from positive numbers.
    """
    def _fix(m: re.Match) -> str:
        s = m.group(0)
        s = s.replace(",", ".")          # comma → dot
        s = s.replace("−", "-")          # unicode minus → ascii minus
        s = s.replace(" ", "")           # strip inner spaces
        # strip leading '+' so +1.31% == 1.31%
        s = re.sub(r"^\+", "", s)
        # normalise to 2-decimal precision to avoid 1.3% vs 1.31% noise
        num_match = re.search(r"([+\-]?\d+\.?\d*)", s)
        if num_match:
            try:
                val = float(num_match.group(1))
                s = f"{val:.2f}%"
            except ValueError:
                pass
        return s
    return _PCT_RE.sub(_fix, text)


def texts_are_numerically_equal(a: str, b: str) -> bool:
    """Return True when two strings differ only in decimal formatting."""
    return normalize_decimal(a) == normalize_decimal(b)

def load_dataset(path: str = THRESHOLD_DATASET_PATH) -> List[Dict[str,str]]:
    if not os.path.exists(path):
        return []
    out = []
    # Try utf-8 first, fall back to latin-1 if there are encoding issues
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
            return out
        except UnicodeDecodeError:
            out = []
            continue
    return out

def extract_date_from_prompt(prompt: str) -> Optional[str]:
    """Extrae la fecha del prompt (formato YYYY-MM-DD al inicio)."""
    match = re.match(r"(\d{4}-\d{2}-\d{2})", prompt.strip())
    return match.group(1) if match else None

def find_reference_for_date(dataset: List[Dict], target_date: str) -> Optional[Dict]:
    """Busca una respuesta de referencia para la fecha dada."""
    for entry in dataset:
        entry_date = extract_date_from_prompt(entry.get("prompt", ""))
        if entry_date == target_date:
            return entry
    return None

def build_eval_prompt(few_shot_examples: List[Dict[str,str]], csv_data: str, generated: str, 
                      reference_response: Optional[str] = None,
                      iteration: int = 1,
                      previous_attempts: Optional[List[Dict]] = None) -> str:
    """
    Construye el prompt para el evaluador.
    El evaluador aprende de los ejemplos del dataset (con sus accuracy reales)
    y evalúa holísticamente, sin penalizaciones mecánicas.
    
    Args:
        few_shot_examples: Ejemplos del dataset con prompt, response y accuracy
        csv_data: Los datos del CSV formateados
        generated: La respuesta generada por el LLM
        reference_response: Respuesta de referencia del dataset (opcional)
        iteration: Número de iteración actual (1, 2, 3...)
        previous_attempts: Lista de intentos anteriores con {response, score, reason, mejoras}
    """
    # Extraer fecha del CSV y calcular el día de la semana real
    fecha_info = ""
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", csv_data)
    if date_match:
        from datetime import datetime
        try:
            fecha = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
            dia_semana = dias[fecha.weekday()]
            fecha_info = f"\n\n**FECHA CORRECTA:** {fecha.strftime('%d/%m/%Y')} es {dia_semana}.\n"
        except:
            pass
    
    # Mostrar TODOS los ejemplos del dataset con su accuracy como calibración
    fs = """=== EJEMPLOS DE CALIBRACIÓN DEL DATASET ===

Los siguientes ejemplos son informes REALES evaluados por un humano con su accuracy real.
Tu trabajo es APRENDER de estos ejemplos para entender:
- Qué hace que un informe sea excelente (100%) vs bueno (98%) vs mejorable (77%)
- Qué tickers son más relevantes según el contexto del mercado (no todos pesan igual)
- Cómo se integran las noticias y el contexto macro en un buen informe
- Qué nivel de detalle y estructura se espera
- Cuándo es importante mencionar after-hours, datos macro, sectores defensivos vs cíclicos

Estudiá cada ejemplo y su accuracy para calibrar tu evaluación.

"""
    for i, ex in enumerate(few_shot_examples, 1):
        acc = ex.get('accuracy', 'N/A')
        # Format accuracy as percentage for display
        if isinstance(acc, (int, float)):
            acc_display = f"{acc * 100:.0f}%" if acc <= 1.0 else f"{acc:.0f}%"
        else:
            acc_display = str(acc)
        fs += f"### EJEMPLO {i} — ACCURACY: {acc_display}\n"
        fs += f"DATOS (prompt):\n{ex['prompt'][:1200]}\n\n"
        fs += f"RESPUESTA:\n{ex['response']}\n\n"
        fs += f"ACCURACY ASIGNADA: {acc_display}\n"
        fs += "---\n\n"
    
    instruction = """
=== TU ROL COMO EVALUADOR ===

Sos un evaluador experto de informes de mercado financiero. Tu evaluación debe ser HOLÍSTICA
y basada en lo que aprendiste de los ejemplos anteriores, NO en una lista rígida de penalizaciones.

**PRINCIPIOS CLAVE (aprendidos del dataset):**

1. NO todos los tickers valen lo mismo. En una rueda risk-off con VIX +17%, importa más
   mencionar los grandes perdedores (MSFT -4.95%, AMZN -4.42%) que un ticker secundario
   con -0.19%. Evaluá según RELEVANCIA DE MERCADO, no por cantidad.

2. La NARRATIVA importa. Un buen informe no es una lista de tickers con variaciones,
   sino un relato que explica POR QUÉ el mercado se movió así, conectando datos macro,
   noticias corporativas, flujos sectoriales y sentimiento.

3. El CONTEXTO MACRO es clave. Datos como retail sales, job openings, consumer confidence,
   Treasury yields, etc., deben integrarse como CAUSAS de los movimientos, no como datos sueltos.

4. Los VALORES NUMÉRICOS de los tickers mencionados deben ser correctos respecto al CSV.
   Un valor incorrecto en un ticker importante es más grave que omitir un ticker menor.

5. La ESTRUCTURA debe ser profesional: párrafo(s) fluido(s), no bullet points ni listas.

6. Si el usuario pidió foco en algo específico (ej: "pone foco en AMZN"), la respuesta
   DEBE desarrollar ese tema con profundidad (after-hours, earnings, guidance, etc.).

**CÓMO ASIGNAR EL SCORE (aprendé del dataset):**
- Mirá los ejemplos: ¿a qué se parece más la respuesta evaluada?
- ¿Tiene la misma profundidad de análisis que el ejemplo de 100%?
- ¿Integra las noticias tan bien como el ejemplo de 98%?
- ¿Le falta narrativa o contexto como al ejemplo de 77%?
- Asigná un score que refleje la CALIDAD REAL comparada con los ejemplos.

**PRECISIÓN DEL SCORE — MUY IMPORTANTE:**
- Usá TODA la escala decimal disponible. NO redondees a 0.05 ni a 0.10.
- Scores válidos incluyen valores como 0.82, 0.87, 0.91, 0.73, 0.94, 0.68, 0.96, etc.
- Cada evaluación debe producir un score DISTINTO y PRECISO según la calidad real.
- NUNCA repitas el mismo score entre iteraciones si la respuesta cambió.

**NORMALIZACIÓN DECIMAL (aplicar ANTES de comparar valores):**
  - Coma = punto: -0,84% = -0.84%
  - Signo + opcional: +1,31% = 1,31% = 1.31%
  - Tolerancia de redondeo ≤ 0.05pp: 1.3% ≈ 1.31%
  - Guion largo (—/–) = guion corto (-)
  - Si tras normalizar los valores son iguales → NO es error

"""
    
    if reference_response:
        instruction += f"""
=== RESPUESTA DE REFERENCIA (misma fecha del dataset) ===
Esta respuesta del dataset es el benchmark para esta fecha específica.
Usala como punto de comparación directo:

{reference_response}

"""

    # Agregar contexto de iteraciones previas si existen
    iteration_context = ""
    if previous_attempts and len(previous_attempts) > 0:
        iteration_context = f"""
=== CONTEXTO DE ITERACIÓN (Intento {iteration} de la generación) ===

Esta es la iteración {iteration}. El escritor ya recibió feedback de intentos anteriores
y debería haber mejorado. Evaluá si EFECTIVAMENTE mejoró respecto a los intentos previos.

HISTORIAL DE INTENTOS ANTERIORES:
"""
        for prev in previous_attempts:
            iteration_context += f"""
--- Intento {prev.get('iteration', '?')} (score: {prev.get('score', 0):.2f}) ---
Respuesta anterior (extracto): {prev.get('response', '')[:400]}...
Feedback dado: {prev.get('reason', '')}
Mejoras pedidas: {', '.join(prev.get('mejoras', []))}
"""
        iteration_context += f"""
INSTRUCCIÓN CRÍTICA PARA ESTA ITERACIÓN:
- Compará la respuesta actual contra las anteriores: ¿mejoró, empeoró o se estancó?
- Si la respuesta incorporó las mejoras pedidas → el score DEBE subir.
- Si la respuesta NO incorporó las mejoras → explicá cuáles faltan y el score no sube.
- Si la respuesta empeoró en algún aspecto → el score DEBE bajar.
- NUNCA des el mismo score exacto que un intento anterior. Siempre hay diferencias.
"""
    elif iteration == 1:
        iteration_context = """
=== CONTEXTO DE ITERACIÓN ===
Este es el PRIMER intento del escritor. Evaluá sin comparación previa,
calibrando tu score contra los ejemplos del dataset.
"""
    
    prompt = fs + instruction + iteration_context
    prompt += f"""
=== DATOS CSV A EVALUAR ==={fecha_info}
{csv_data}

=== RESPUESTA GENERADA A EVALUAR ===
{generated}

=== TU EVALUACIÓN ===
Responde SOLO con JSON válido (sin texto adicional ni markdown).

Evaluá holísticamente basándote en lo que aprendiste de los ejemplos del dataset.
El campo "mejoras" debe listar las mejoras CONCRETAS más importantes que harían
subir el score, priorizadas por impacto (las más importantes primero).

RECORDÁ: el score debe ser PRECISO (ej: 0.82, 0.91, 0.74), NO redondeado a 0.05.
Si es iteración > 1, el score DEBE diferir del anterior según si hubo mejora o no.

{{"score": <decimal PRECISO entre 0 y 1, calibrado contra el dataset, sin redondear a .05>,
  "reason": "<análisis holístico: qué hace bien, qué le falta, y si es iteración >1 cómo compara con el intento anterior>",
  "datos_correctos": <true si los valores numéricos mencionados coinciden con el CSV / false>,
  "narrativa_quality": "<alta/media/baja — qué tan bien conecta causas y efectos>",
  "mejoras": [
    "<mejora concreta 1, la más importante>",
    "<mejora concreta 2>",
    "<mejora concreta 3>"
  ]}}
"""
    return prompt

def call_evaluator(prompt: str, openai_model: str = "gpt-4o-mini", temperature: float = 0.0) -> Tuple[Dict, str]:
    """Calls the evaluator LLM.
    Returns (parsed_dict, raw_response_text) so callers can log the raw output.
    """
    if OpenAI is None:
        return {"score": 0.5, "ok": False, "reason": "Fallback evaluator (OpenAI SDK no disponible)."}, ""
    client = OpenAI()
    raw_text = ""
    try:
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "Eres un evaluador que responde SOLO con JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        raw_text = resp.choices[0].message.content.strip()
        try:
            obj = json.loads(raw_text)
            return obj, raw_text
        except Exception:
            m = re.search(r"\{.*\}", raw_text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0)), raw_text
                except Exception:
                    pass
            return {"score": 0.0, "ok": False, "reason": "No parseable JSON from evaluator."}, raw_text
    except Exception as e:
        return {"score": 0.0, "ok": False, "reason": f"Error calling evaluator: {e}"}, raw_text
