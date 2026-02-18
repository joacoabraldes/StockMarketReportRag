# 🦜 PARROT RAG — Generador de informes de rueda bursátil

Sistema multi-agente que genera informes profesionales de mercado financiero a partir de datos de Yahoo Finance y noticias, usando un loop **Writer → Evaluator** con auto-corrección iterativa.

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                     Usuario                             │
│        (Streamlit UI  ó  CLI generate_report.py)        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Descarga de datos (yfinance)                │
│     Batch download → fallback individual por ticker     │
│     Detección automática de fecha del mercado (moda)    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Loop Writer ↔ Evaluator                    │
│                                                         │
│  ┌──────────┐   respuesta    ┌────────────┐             │
│  │  Writer   │──────────────▶│ Evaluator  │             │
│  │ (GPT-4o) │◀──────────────│ (GPT-4o)   │             │
│  └──────────┘   feedback     └────────────┘             │
│                 + mejoras                               │
│                                                         │
│  Hasta 5 iteraciones, con:                              │
│  • Edición incremental (no reescritura)                 │
│  • Detección de plateau                                 │
│  • Umbral adaptivo (0.95 ideal / 0.88 good-enough)     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Informe final                              │
│  Párrafo profesional con tickers, variaciones,          │
│  contexto macro y narrativa causal                      │
└─────────────────────────────────────────────────────────┘
```

> Para más detalle sobre los agentes, ver [AGENTS.md](AGENTS.md).

## Estructura del proyecto

```
├── app_streamlit.py           # UI principal (Streamlit)
├── generate_report.py         # Generador headless (CLI)
├── requirements.txt           # Dependencias
├── config/
│   ├── __init__.py
│   └── market_config.py       # MarketConfig dataclass, configs US/AR
├── core/
│   ├── __init__.py
│   ├── compute_variations.py  # Descarga Yahoo Finance + cálculo variaciones
│   ├── debug_logger.py        # DebugSession / IterationRecord (trazabilidad)
│   ├── evaluator.py           # Evaluador holístico calibrado con dataset
│   └── utils.py               # Limpieza de texto, fetch URLs, formateo CSV
├── prompts/
│   ├── systemprompt_template.txt     # System prompt mercado US
│   └── systemprompt_template_ar.txt  # System prompt mercado AR
└── data/
    ├── threshold_dataset.jsonl       # Ejemplos curados para calibración
    └── history/
        └── chat_history.jsonl        # Historial de consultas
```

## Instalación

```bash
# Clonar el repo
git clone <repo-url>
cd StockMarketReportRag

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key
echo OPENAI_API_KEY=sk-... > .env
```

## Uso

### Streamlit (UI interactiva)

```bash
streamlit run app_streamlit.py
```

1. Activar **"Generar informe de la rueda"** en la sidebar
2. Opcionalmente seleccionar fecha y pegar noticias/URLs
3. Escribir el prompt (ej: `"Generá resumen para 17/02/2026"`)
4. El sistema descarga datos, genera el informe y lo evalúa iterativamente
5. Activar **🐛 Debug** para ver la conversación completa entre agentes

### CLI (headless)

```bash
# Informe US básico
python generate_report.py --date 2026-02-17

# Con noticias y mercado AR
python generate_report.py --date 2026-02-17 --market AR --news "El BCRA mantuvo la tasa..."

# Con URLs de noticias
python generate_report.py --date 2026-02-17 --news-urls https://wsj.com/... https://reuters.com/...

# Guardar en archivo específico
python generate_report.py --date 2026-02-17 --out informe_lunes.txt
```

## Configuración

### Variables de entorno (`.env`)

| Variable | Default | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | — | API key de OpenAI (requerida) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo para writer y evaluator |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings local |

### Mercados soportados

| Mercado | Tickers | Prompt |
|---|---|---|
| 🇺🇸 US | SPX, NDX, VIX, 7 ETFs sectoriales, 10 acciones tech + quántica | `systemprompt_template.txt` |
| 🇦🇷 AR | MERVAL, GGAL, YPF, BMA, y 12 ADRs argentinos | `systemprompt_template_ar.txt` |

## Dataset de evaluación

El archivo `data/threshold_dataset.jsonl` contiene ejemplos curados manualmente con accuracy asignada:

| Accuracy | Descripción |
|---|---|
| 100% | Informe excepcional: cubre todos los tickers relevantes, narrativa profesional, integra noticias y macro |
| 98% | Muy bueno: cubre la mayoría con buena narrativa, quizás falta un detalle menor |
| 77% | Mejorable: cubre los principales pero omite contexto importante o narrativa básica |

El evaluador usa estos ejemplos como **calibración few-shot** para aprender qué nivel de calidad merece cada score.

### Agregar ejemplos al dataset

Cuando un informe alcanza el score mínimo (0.95), aparece el botón **"➕ Agregar ejemplo al dataset"** en la sidebar. Esto permite enriquecer el dataset progresivamente.

## Parámetros del loop de evaluación

| Parámetro | Valor | Descripción |
|---|---|---|
| `max_eval_retries` | 5 | Intentos máximos de generación |
| `min_eval_score` | 0.95 | Score ideal para aceptar directamente |
| `GOOD_ENOUGH_SCORE` | 0.88 | Score aceptable si hay plateau o se exceden 3 intentos |
| `PLATEAU_THRESHOLD` | 0.02 | Mejora mínima para no considerar plateau |
| Temperatura | +0.03/intento | Incremento gradual para variabilidad (max +0.12) |

## Stack tecnológico

- **LLM**: OpenAI GPT-4o-mini (writer + evaluator)
- **Datos financieros**: Yahoo Finance via `yfinance`
- **Embeddings**: HuggingFace `sentence-transformers` (ranking local de relevancia)
- **UI**: Streamlit
- **Scraping de noticias**: `requests` + `BeautifulSoup`
