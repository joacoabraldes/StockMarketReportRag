# 🤖 AGENTS.md — Arquitectura multi-agente

Este proyecto usa dos agentes LLM que interactúan en un loop iterativo para generar informes de mercado de alta calidad.

## Agentes

### ✍️ Writer (Modelo Escritor)

**Rol**: Generar el informe de rueda bursátil.

**Modelo**: GPT-4o-mini (configurable)

**Input**:
- System prompt con reglas de formato, estilo y convenciones de mercado
- Datos CSV con tickers, precios de cierre y variaciones diarias (formato `|`-separated)
- Noticias del usuario (texto y/o URLs scrapeadas)
- Fecha y día de la semana del informe

**Output**: Un párrafo profesional en español con:
- Fecha y día al inicio
- Tickers más relevantes con variaciones exactas
- Narrativa causal (por qué se movió el mercado)
- Contexto macro integrado como causa
- ETFs sectoriales como indicadores

**Comportamiento en iteraciones**:
- **Iteración 1**: Genera desde cero con el prompt original
- **Iteración 2+**: Recibe su mejor respuesta anterior + feedback del evaluador, y la **edita** (no reescribe) aplicando mejoras concretas

---

### 🔍 Evaluator (Modelo Evaluador)

**Rol**: Evaluar holísticamente la calidad del informe generado.

**Modelo**: GPT-4o-mini (temperature=0.0)

**Input**:
- Ejemplos de calibración del dataset (few-shot con accuracy real: 100%, 98%, 77%)
- Datos CSV originales (ground truth para verificar valores numéricos)
- Respuesta generada por el writer
- Respuesta de referencia del dataset (si existe para esa fecha)
- Historial de intentos anteriores (scores, feedback, mejoras pedidas)

**Output** (JSON):
```json
{
  "score": 0.93,
  "reason": "Análisis holístico de la calidad...",
  "datos_correctos": true,
  "narrativa_quality": "alta",
  "mejoras": [
    "Mejora concreta 1 (la más importante)",
    "Mejora concreta 2",
    "Mejora concreta 3"
  ]
}
```

**Criterios de evaluación**:

| Criterio | Peso conceptual |
|---|---|
| Valores numéricos correctos vs CSV | Alto — errores en tickers importantes son graves |
| Narrativa causal (por qué, no solo qué) | Alto — conectar macro/noticias con movimientos |
| Cobertura de tickers relevantes | Medio — priorizar por relevancia de mercado, no cantidad |
| Estructura profesional | Medio — párrafos fluidos, sin bullets ni listas |
| Foco del usuario respetado | Alto — si pidió foco en un ticker, debe desarrollarlo |

**Escala de scores**:

| Rango | Nivel | Descripción |
|---|---|---|
| 0.93 – 1.00 | Excepcional | Cobertura completa, narrativa profesional, datos correctos |
| 0.85 – 0.92 | Muy bueno | Buena cobertura y narrativa, quizás falta un detalle menor |
| 0.75 – 0.84 | Bueno | Cubre los principales pero omite contexto o tiene algún error |
| 0.60 – 0.74 | Mejorable | Faltan tickers relevantes o narrativa superficial |
| < 0.60 | Deficiente | Omisiones graves, datos incorrectos |

---

## Loop de interacción

```
Iteración 1
  Writer  → genera informe desde cero
  Evaluator → evalúa, score=0.82, mejoras=["falta contexto macro", "agregar VIX"]

Iteración 2
  Writer  → recibe su respuesta anterior (0.82) + feedback + CSV
          → EDITA la respuesta aplicando mejoras (no reescribe)
  Evaluator → evalúa la versión mejorada, score=0.91, mejoras=["integrar yields"]

Iteración 3
  Writer  → recibe su mejor respuesta (0.91) + último feedback
          → aplica las mejoras restantes
  Evaluator → evalúa, score=0.94
          → plateau + good enough → ACEPTA ✅
```

### Mecanismos de control

| Mecanismo | Condición | Acción |
|---|---|---|
| **Score ideal** | `score ≥ 0.95` | Acepta inmediatamente |
| **Good enough** | `score ≥ 0.88` después de 3+ intentos | Acepta el mejor |
| **Plateau** | Mejora < 0.02 dos veces seguidas + score ≥ 0.88 | Acepta el mejor |
| **Max retries** | 5 intentos agotados | Usa el mejor score obtenido |

### Temperatura adaptiva

El writer comienza con la temperatura del usuario (default 0.0) y sube +0.03 por intento, hasta un máximo de +0.12. Esto introduce variabilidad gradual si la respuesta está "estancada".

---

## Trazabilidad (Debug)

Toda la conversación entre agentes se captura en un `DebugSession` con `IterationRecord`s:

- **Por cada iteración**: system prompt, user prompt, respuesta del writer, prompt del evaluator, respuesta raw del evaluator, score, temperatura
- **Visualización en Streamlit**: activar 🐛 Debug en sidebar para ver la timeline completa con métricas, prompts y respuestas de ambos agentes

---

## Diagrama de flujo

```
                    ┌──────────────┐
                    │   Usuario    │
                    │  (pregunta)  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Descarga    │
                    │  yfinance    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Embedding   │
                    │  + ranking   │
                    │  (noticias)  │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │    Writer (intento 1)   │
              │    temp=0.00            │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    Evaluator            │
              │    score=0.82 ❌        │
              └────────────┬────────────┘
                           │ feedback
              ┌────────────▼────────────┐
              │    Writer (intento 2)   │
              │    edita mejor resp.    │
              │    temp=0.03            │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    Evaluator            │
              │    score=0.91 ❌        │
              └────────────┬────────────┘
                           │ feedback
              ┌────────────▼────────────┐
              │    Writer (intento 3)   │
              │    edita mejor resp.    │
              │    temp=0.06            │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    Evaluator            │
              │    score=0.94 ✅        │
              │    (good enough @ 3)    │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │   Informe    │
                    │   final      │
                    └──────────────┘
```
