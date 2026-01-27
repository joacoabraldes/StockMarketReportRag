# evaluator.py
# Carga dataset JSONL, arma few-shot y evalúa respuesta (usa OpenAI si disponible).
import os, json
from typing import List, Dict
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

THRESHOLD_DATASET_PATH = "./threshold_dataset.jsonl"

def load_dataset(path: str = THRESHOLD_DATASET_PATH) -> List[Dict[str,str]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def build_eval_prompt(few_shot_examples: List[Dict[str,str]], query: str, generated: str) -> str:
    # Simple evaluator instruction: return JSON with score in [0,1], ok boolean, reason
    fs = ""
    for ex in few_shot_examples:
        fs += f"### EJEMPLO PREGUNTA:\n{ex['prompt']}\n### EJEMPLO RESPUESTA:\n{ex['response']}\n\n"
    instruction = (
        "Eres un evaluador. Compara la RESPUESTA con la PREGUNTA y juzga si la respuesta cumple "
        "estrictamente los requisitos pedidos (fecha/día, un solo párrafo, uso de tickers, formato numerico, "
        "identificación de ganadores/rezagados, no introducir información sin fuente). "
        "Devuelve SOLO un JSON con campos: score (0..1), ok (true/false), reason (breve).\n\n"
    )
    prompt = fs + instruction + f"### PREGUNTA:\n{query}\n\n### RESPUESTA A EVALUAR:\n{generated}\n\n"
    prompt += "RESPONDE AHORA con un JSON válido, por ejemplo: {\"score\":0.85, \"ok\": true, \"reason\":\"...\"}"
    return prompt

def call_evaluator(prompt: str, openai_model: str = "gpt-4o-mini", temperature: float = 0.0) -> Dict:
    # Uses OpenAI SDK if available; expects JSON in response, parse and return dict
    if OpenAI is None:
        # fallback: simple heuristic
        # returns neutral score
        return {"score": 0.5, "ok": False, "reason": "Fallback evaluator (OpenAI SDK no disponible)."}
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "Eres un evaluador que responde SOLO con JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        text = resp.choices[0].message.content.strip()
        # try to find first JSON object in text
        try:
            obj = json.loads(text)
            return obj
        except Exception:
            # try to extract JSON substring
            import re
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {"score": 0.0, "ok": False, "reason": "No parseable JSON from evaluator."}
    except Exception as e:
        return {"score": 0.0, "ok": False, "reason": f"Error calling evaluator: {e}"}
