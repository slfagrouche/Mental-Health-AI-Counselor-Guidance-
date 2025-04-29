# TheraGuide AI

## Overview

**TheraGuide AI** helps mental‑health professionals craft real‑time, context‑aware replies during patient sessions. A FastAPI back‑end predicts an appropriate response style (*Empathetic Listening*, *Advice*, *Question*, *Validation*) and flags potential crises, while a React front‑end provides a streamlined UI. Retrieval‑augmented generation (RAG) powered by ChromaDB and OpenAI embeddings surfaces similar past exchanges, and a direct LLM call offers an additional suggestion.

---

## Core Components

- **Back‑End** `/backend`
  - `/suggest` endpoint for response generation
  - RandomForest (response‑type) & GradientBoosting (crisis‑detection) models
  - ChromaDB vector store with `text‑embedding‑ada‑002`
- **Front‑End** `/frontend` (React)
  - Form to enter patient text and view suggestions / flags
- **Notebooks** `/notebooks`
  - Data exploration, model training, endpoint testing

---

## Features

| Endpoint          | Description                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| **POST /suggest** | Returns `response_type` (4‑class), `crisis_flag`, `confidence`, `rag_suggestion`, and `direct_suggestion`. |

Additional highlights:

- **Crisis detection** tuned for high recall (automatic 988 escalation notice).
- **LangSmith tracing** for full audit trail.

---

## Prerequisites

| Tool    | Version                              |
| ------- | ------------------------------------ |
| Python  | 3.11.12                              |
| Node.js | Current LTS                          |
| OS      | macOS / Linux / Windows (≥ 8 GB RAM) |

`.env` variables:

```env
OPENAI_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=TheraGuideAI
```

---

## Installation

```bash
# 1. Clone
$ git clone https://github.com/your‑repo/theraguide‑ai.git
$ cd theraguide‑ai

# 2. Back‑end
$ python3 -m venv venv && source venv/bin/activate
$ pip install -r backend/requirements.txt
$ cp .env.example .env  # then paste your keys

# 3. Front‑end
$ cd frontend && npm install && npm start
```

### Dataset & Models

```bash
# from project root
$ python backend/retrain_models.py        # download + train
$ python backend/create_vector_db.py      # build ChromaDB
$ uvicorn backend.api_mental_health:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage

### cURL Example

```bash
curl -X POST http://localhost:8000/suggest \
     -H "Content-Type: application/json" \
     -d '{"context": "I feel so alone, nobody cares."}'
```

**Sample response**

```json
{
  "context": "I feel so alone, nobody cares.",
  "response_type": "Empathetic Listening",
  "crisis_flag": true,
  "confidence": 0.95,
  "rag_suggestion": "I'm here for you. Please consider calling 988 for immediate support.",
  "direct_suggestion": "Your feelings are valid. Please reach out to 988.",
  "escalation_required": true
}
```

### Front‑End

Open `http://localhost:3000`, enter a patient statement, and review suggestions plus crisis alerts.

---

## Project Structure

```
theraguide‑ai/
├── backend/
│   ├── api_mental_health.py
│   ├── create_vector_db.py
│   ├── retrain_models.py
│   ├── mental_health_model_artifacts/
│   └── requirements.txt
├── frontend/
│   └── src/ …
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── endpoint_testing.ipynb
└── README.md
```

---

## Technical Details

| Area       | Stack                                                 |
| ---------- | ----------------------------------------------------- |
| Framework  | FastAPI 0.115                                         |
| Embeddings | OpenAI `text‑embedding‑ada‑002`                       |
| LLM        | OpenAI `gpt‑4o‑mini`                                  |
| Models     | RandomForest (TF‑IDF + LDA + VADER), GradientBoosting |
| Vector DB  | ChromaDB                                              |
| Monitoring | LangSmith                                             |

---

## Limitations

- Kaggle dataset may lack demographic diversity.
- Crisis detector favors recall → potential false positives.

---

## Next Steps / Roadmap

- **Front‑End**: richer visuals for risk levels & explanations.
- **Responsible AI**: PII anonymization, SMOTE rebalancing, SHAP explainability, human‑in‑the‑loop escalation.
- **Deployment**: containerize & host on AWS (ECS or Lambda).
- **Data**: extend corpus with diverse demographic conversations.

---

© 2025 Said Lfagrouche — MIT License

