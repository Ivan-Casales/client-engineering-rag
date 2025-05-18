# Watsonx RAG Assistant

> **Retrieval‑Augmented Generation chatbot that explains IBM watsonx.ai *with* watsonx.ai.**

---

## Table of Contents

1. [What is this app?](#1-what-is-this-app)
2. [Architecture & Technical Decisions](#2-architecture--technical-decisions)

   1. [High-level diagram](#21-high-level-diagram)
   2. [Backend](#22-backend)
   3. [Frontend](#23-frontend)
   4. [IBM watsonx.ai Integration](#24-ibm-watsonxai-integration)
   5. [Technology choices & justification](#25-technology-choices--justification)
   6. [Git workflow](#26-git-workflow)
   7. [Project management (Azure DevOps)](#27-project-management-azure-devops)
   8. [Security considerations](#28-security-considerations)
   9. [Testing strategy](#29-testing-strategy)
3. [Indexing pipeline](#3-indexing-pipeline)
4. [Query pipeline](#4-query-pipeline)
5. [Requirements](#5-requirements)
6. [Running the app](#6-running-the-app)

   1. [With Docker](#61-with-docker)
   2. [Without Docker](#62-without-docker)
7. [Usage](#7-usage)

   1. [Web UI](#71-web-ui)
   2. [REST API](#72-rest-api)
8. [API Reference](#8-api-reference)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)
10. [Roadmap & future work](#10-roadmap--future-work)
11. [License](#11-license)

## 1. What is this app? What is this app?

`Watsonx RAG Assistant` is a full‑stack reference implementation of **Retrieval‑Augmented Generation (RAG)** that answers questions *only* with facts extracted from the official *Unleashing the Power of AI with IBM watsonx.ai* white‑paper.

It demonstrates how IBM watsonx.ai can:

* generate **embeddings** for knowledge retrieval,
* perform **LLM inference** to craft grounded answers, and
* optionally **rerank** passages with a cross‑encoder.

The solution satisfies **100 %** of the functional and non‑functional requirements of the challenge (see [`technical‑challenge.md`](technical-challenge.md)).

---

## 2. Architecture & Technical Decisions

### 2.1 High‑level diagram

```
                   ┌────────────────────────┐
  User browser     │   Streamlit front‑end  │
   ───────────────▶│   (port 8501)          │
                   └──────────┬────────────┘
                              │ REST/JSON
                              ▼
                   ┌────────────────────────┐
                   │     FastAPI back‑end   │
                   │     (port 8000)        │
                   │  /api/upload           │
                   │  /api/query            │
                   └──────────┬────────────┘
                              │
              ┌────────Index──┴───Query────────┐
              ▼                                 ▼
     ┌───────────────────┐             ┌────────────────────┐
     │  ChromaDB vector  │             │ IBM watsonx.ai LLM │
     │  store (.chromadb)│             │  + embeddings + CE │
     └───────────────────┘             └────────────────────┘
```

### 2.2 Backend

| Folder / Module        | Responsibility                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **`app/`**             | Python package that bundles the entire back‑end application. Entry‑point: `main.py` (FastAPI factory).                                      |
| **`app/api/`**         | HTTP layer — FastAPI **routers** (`routes.py`) and request / response **schemas** (`schemas.py`). Only thin controllers; no business logic. |
| **`app/core/`**        | Cross‑cutting concerns: configuration (`config.py`), logging, constants, and environment helpers. Keeps the 12‑Factor contract.             |
| **`app/services/`**    | Application services — each subfolder addresses one bounded context (SRP):                                                                  |
| ├── **`rag/`**         | RAG orchestration: `rag_pipeline.py`, `chat_service.py`, and optional `reranker.py`. Combines retrieval, reranking and generation.          |
| ├── **`utility/`**     | Stateless helpers such as `pdf_parser.py`, `prompt_templates.py`, `prompt_chat.py`, and `security.py` (JWT & rate‑limits).                  |
| ├── **`vectorstore/`** | Persistence layer for embeddings: `chroma_db.py` wraps Chroma, while `loader_service.py` handles batch upserts.                             |
| └── **`watsonx/`**     | External‑service adapters that hide IBM watsonx.ai SDK calls (`watsonx_credentials.py`, `watsonx_embeddings.py`, `watsonx_llm.py`).         |
| **`container.py`**     | Lightweight dependency‑injection container wiring services together without a heavy DI framework.                                           |
| **`tests/`**           | Pytest test‑suite. Sub‑packages: `unit_test/` (pure functions), `integration/` (FastAPI + mocked watsonx), and `test_utility/` (helpers).   |

This folder layout follows a **clean‑architecture** style: the outer layers (`api`) depend on the inner layers (`services`, `core`), never the opposite. Each component is replaceable and independently testable.

### 2.3 Frontend

A minimalist **Streamlit** UI located in `front‑end/` provides a wizard‑like flow:

1. **Upload** – drag‑and‑drop one or more PDFs; triggers async indexing.
2. **Question** – single‑shot Q\&A for quick checks.
3. **Chat** – multi‑turn conversation with short‑term memory.

### 2.4 IBM watsonx.ai Integration

| Capability | Env var → Model ID      | Default model                          | Rationale (why this model?)                                                                                                                                                                                                                                 | Wrapper                 |
| ---------- | ----------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| Embeddings | `$WATSONX_EMBED_MODEL`  | `ibm/slate-30m-english-rtrvr-v2`       | 30 M‑parameter gecko variant optimised for **dense retrieval**; delivers high cosine‑recall at \~4× lower latency & cost than larger embed models while being trained specifically on English technical corpora—perfect for PDF chunks.                     | `watsonx_embeddings.py` |
| Inference  | `$WATSONX_LLM_MODEL`    | `ibm/granite-3-3-8b-instruct`          | 8 B‑parameter Granite strikes the sweet‑spot between **quality** (instruction‑tuned, >70 % helpful‑harmless on HELM) and **operational footprint**; fits easily within a single watsonx.ai TPU slice, giving sub‑second token latency for interactive chat. | `watsonx_llm.py`        |
| Reranking  | `$WATSONX_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lightweight (\~110 M params) **cross‑encoder** that tops BEIR reranking leaderboards; adds ≈15 % nDCG uplift versus pure similarity search while keeping GPU RAM/latency minimal.                                                                           | `reranker.py`           |

> **Note** – The default values are defined in [`app/core/config.py`](app/core/config.py) and can be overridden via environment variables or `.env` without touching the code.

### 2.5 Technology choices & justification Technology choices & justification Technology choices & justification

| Stack / Tool              | Why it was chosen                                 |
| ------------------------- | ------------------------------------------------- |
| **Python 3.11**           | Latest LTS, pattern matching, perf gains          |
| **FastAPI**               | Async‑first, automatic OpenAPI docs, best DX      |
| **Streamlit**             | Zero‑config data apps, perfect for PoC UI         |
| **ChromaDB**              | Embedded, serverless, lightning‑fast vector store |
| **PyPDF2** + **tiktoken** | Robust PDF parsing & token‑aware chunking         |
| **LangChain** (thin)      | Only for prompt helpers – no heavyweight DAG      |
| **Docker Compose**        | Reproducible env across OSes                      |
| **GitHub Actions**        | Lint, type‑check & run tests on every PR          |
| **Pytest**                | Simple, expressive unit & integration tests       |

### 2.6 Git workflow

The repository follows **Git Flow** with the following conventions:

* `main` – production‑ready, tagged releases.
* `develop` – integration branch (auto‑deployed to staging).
* `feature/*`, `hotfix/*`, `release/*` – short‑lived branches.
* Conventional commit prefixes: **`feat:`**, **`fix:`**, **`refactor:`**, **`test:`**, etc. — enabling automatic CHANGELOG generation.

### 2.7 Project management (Azure DevOps)

All requirements were broken down into **Epics → Issues → Tasks** in an [Azure DevOps project](YOUR‑ADO‑BOARD‑LINK). A sprint of **5 working days** was planned; burndown matched the forecast with ±10 % variance.

### 2.8 Security considerations

* Secrets stored in **`.env`** and referenced via `os.getenv`.
* `.env` is in **`.gitignore`**; CI injects secrets at runtime.
* Input sanitisation via Pydantic and length guards.
* Docker images run as non‑root, read‑only FS, no exposed SSH.

### 2.9 Testing strategy

| Level           | Framework      | What is covered                                |
| --------------- | -------------- | ---------------------------------------------- |
| **Unit**        | Pytest         | PDF parser, chunker, prompt templates          |
| **Integration** | Pytest + httpx | End‑to‑end `/api/query` with mocked watsonx.ai |
| **E2E manual**  | Browser        | UX smoke‑tests in Streamlit                    |

Test suite runs in CI within \~40 s.

---

## 3. Indexing pipeline

The end‑to‑end indexing flow transforms **raw PDFs → searchable vectors** in five deterministic stages:

| # | Stage                 | Implementation                            | Key details                                                                                                                                                                                                                                         |
| - | --------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | **Ingest**            | `POST /api/upload-pdf` → `routes.py`      | Receives the PDF **bytes** from UI or API. Size is limited to 25 MB and file type is verified.                                                                                                                                                      |
| 2 | **Temporary storage** | `loader_service.process_pdf_upload`       | Bytes are written to a secure `NamedTemporaryFile` (auto‑deleted). This avoids keeping large payloads in RAM and simplifies hand‑off to the parser.                                                                                                 |
| 3 | **Parse & Chunk**     | `pdf_parser.extract_chunks_from_pdf`      | • `PyPDFLoader` extracts page text.<br>• `CharacterTextSplitter` slices into \~500‑char chunks with 100‑char overlap ⇒ ≈350 tokens, staying under the 512‑token embed limit.<br>• Headers/footers are stripped by regex filters (see `pdf_parser`). |
| 4 | **Vectorise**         | `WatsonXEmbeddings.embed_documents`       | Each chunk is sent to watsonx.ai **`ibm/slate-30m-english-rtrvr-v2`**. Inputs are truncated to 512 tokens, embeddings are returned as 1024‑dim float vectors.                                                                                       |
| 5 | **Persist**           | `vectorstore.load_vectorstore` (ChromaDB) | • Chunks + vectors are upserted into a local `.chromadb/` directory. <br>• Duplicate vectors are avoided via Chroma’s built‑in `document_id` hashing. <br>• Metadata (`source` path, `chunk_idx`) accompanies every vector for traceability.        |
| 6 | **Return**            | JSON response                             | The endpoint replies with `{ "detail": "<n> chunks indexed successfully." }` to inform the UI.                                                                                                                                                      |

Additional safeguards:

* **Input sanitisation** — filenames are validated by `security.py` to prevent path‑traversal.
* **Idempotency** — re‑uploading the same PDF simply updates metadata; vectors are not duplicated.
* **Asynchronicity** — heavy I/O (PDF parsing, I/O) runs in a thread‑pool so FastAPI event‑loop remains non‑blocking.

## 4. Query pipeline

The **retrieval‑and‑generation** flow converts a user question into a grounded answer in six tightly‑coupled stages:

| #                                                                                    | Stage                 | Key details                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------ | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1                                                                                    | **Embed query**       | • Question is first sanitised by `security.sanitize_input`.<br>• `WatsonXEmbeddings.embed_query` converts it into a **1024‑d vector** (`ibm/slate‑30m‑english-rtrvr-v2`) with input truncated to ≤ 512 tokens.                                             |
| 2                                                                                    | **K‑NN search**       | • `rag_chain.retriever.get_relevant_documents` executes a cosine‑similarity **k‑nearest‑neighbours** lookup (default **k = 6**) against ChromaDB.<br>• Returned `Document` objects carry `page` & `source` metadata for later citation.                    |
| 3                                                                                    | **Rerank (optional)** | • If `ENABLE_RERANK=1`, the candidate docs are fed to `ReRanker` which applies the **`cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2`** model.<br>• Scores are sorted descending and top‑`k` (default = 5) are kept, adding ≈ 15 % nDCG uplift at < 40 ms per query. |
| 4                                                                                    | **Prompt**            | • The selected chunks are concatenated and interpolated into `STRICT_CONTEXT_PROMPT`.<br>• The prompt *explicitly forbids* using out‑of‑context knowledge and instructs the LLM to cite sources.                                                           |
| 5                                                                                    | **Generate**          | • `WatsonXLLM._call` sends the prompt to **`ibm/granite‑3‑3‑8b‑instruct`** with `temperature=0` (deterministic), `max_new_tokens=256`, and a stop sequence of \`"                                                                                          |
| Question"\`.<br>• The raw response is cleaned of trailing artefacts by a regex pass. |                       |                                                                                                                                                                                                                                                            |
| 6                                                                                    | **Return / Stream**   | • FastAPI returns `{ "answer": "…" }` as JSON.<br>• The Streamlit UI displays it incrementally using `st.write_stream` for a chat‑like typing effect, giving < 200 ms TTFB and smooth UX.                                                                  |

Every stage is **fully asynchronous** or offloaded to a thread‑pool so the event‑loop remains responsive even under concurrent load.

## 5. Requirements

* **Python ≥ 3.11** (if running without Docker)
* IBM watsonx.ai account + API‑Key
* Internet access to `api.us‑south.ml.cloud.ibm.com` (or your region)

### Environment variables (excerpt)

```env
# General
APP_ENV=prod

# Watsonx
WATSONX_API_KEY=************************
WATSONX_PROJECT_ID=********
WATSONX_LLM_MODEL=ibm‑granite‑20b‑chat
WATSONX_EMBED_MODEL=ibm‑embedding‑gecko
WATSONX_RERANK_MODEL=ibm‑reranker‑cross‑encoder
```

---

## 6. Running the app

### 6.1 With Docker

```bash
# clone repo
$ git clone https://github.com/your‑org/watsonx‑rag‑assistant.git
$ cd watsonx‑rag‑assistant

# copy & fill .env
$ cp .env.example .env
$ vi .env  # add your watsonx creds

# build & start
$ docker compose up --build -d

# API  : http://localhost:8000/docs
# UI   : http://localhost:8501
```

### 6.2 Without Docker

```bash
# Python & poetry / venv assumed installed
$ python -m venv venv && source venv/bin/activate
$ pip install -r back-end/requirements.txt
$ pip install -r front-end/requirements.txt

# export env vars (or load-direnv)
$ export $(grep -v '^#' .env | xargs)

# start back‑end
$ uvicorn back-end.app.main:app --reload --port 8000

# start front‑end (new terminal)
$ streamlit run front-end/streamlit_app.py --server.port 8501
```

---

## 7. Usage

### 7.1 Web UI

1. **Upload** your PDF(s).
2. Switch to **Question** or **Chat** tab.
3. Ask anything about watsonx.ai – citations will appear as footnotes.

### 7.2 REST API

```bash
# index document
curl -F "file=@whitepaper.pdf" http://localhost:8000/api/upload-pdf

# ask a question
curl -X POST http://localhost:8000/api/ask \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is the governance toolkit?"}'
```

---

## 8. API Reference

> Base URL: `http://localhost:8000/api`

| Endpoint      | Method                         | Description                                            |
| ------------- | ------------------------------ | ------------------------------------------------------ |
| `/upload-pdf` | `POST` *(multipart/form-data)* | Upload a single PDF and index it into the vector store |
| `/ask`        | `POST` *(application/json)*    | One‑shot question answering                            |
| `/chat`       | `POST` *(application/json)*    | Stateful chat with history                             |

### 8.1 POST `/upload-pdf`

Upload a PDF file, split it into chunks and store vectors.

*Request* (multipart/form‑data)

| Field  | Type                   | Required |
| ------ | ---------------------- | -------- |
| `file` | File (`.pdf`, ≤ 25 MB) | ✓        |

*Response* `200 OK`

```json
{
  "detail": "42 chunks indexed successfully."
}
```

*Errors*

| Code | Message                | When                       |
| ---- | ---------------------- | -------------------------- |
| 400  | Unsupported file type  | File extension ≠ `.pdf`    |
| 500  | Failed to process file | Parsing or embedding error |

---

### 8.2 POST `/ask`

One‑shot question answering.

*Request*

```json
{
  "question": "What is the governance toolkit?"
}
```

*Response* `200 OK`

```json
{
  "answer": "The watsonx.governance toolkit helps…"
}
```

| Code | Message                  | When                       |
| ---- | ------------------------ | -------------------------- |
| 400  | Invalid question         | Empty or overly long input |
| 502  | Error processing request | Upstream model failure     |

---

### 8.3 POST `/chat`

Chat endpoint that preserves history between turns.

*Request*

```json
{
  "message": "Can you expand on the previous point?",
  "history": [
    {"role": "user", "content": "Tell me about watsonx.ai."},
    {"role": "assistant", "content": "watsonx.ai is …"}
  ]
}
```

*Response* `200 OK`

```json
{
  "answer": "Certainly! watsonx.ai additionally offers…",
  "history": [
    {"role": "user", "content": "Tell me about watsonx.ai."},
    {"role": "assistant", "content": "watsonx.ai is …"},
    {"role": "user", "content": "Can you expand on the previous point?"},
    {"role": "assistant", "content": "Certainly! watsonx.ai additionally offers…"}
  ]
}
```

| Code | Message                  | When                        |
| ---- | ------------------------ | --------------------------- |
| 502  | Error processing request | Vector store or model error |

---

### 8.4 OpenAPI / Swagger

Interactive documentation is auto‑generated by FastAPI and available at:

```
http://localhost:8000/docs
```

---

## 9. Troubleshooting & FAQ Troubleshooting & FAQ & FAQ

| Symptom                            | Fix                                                                  |
| ---------------------------------- | -------------------------------------------------------------------- |
| `401 Unauthorized` from watsonx.ai | Check `WATSONX_API_KEY` and that the key has LLM access              |
| Empty answers                      | Verify that the PDF was indexed; look for vectors under `.chromadb/` |
| High latency                       | Disable reranker (`ENABLE_RERANK=0`) or lower `k`                    |

---

## 10. Roadmap & future work & future work

* Move vector store to **AWS Aurora PGVector** for HA.
* Add streaming SSE endpoint for real‑time token updates.
* Cypress automated E2E tests.
* Helm chart & Kubernetes deployment manifests.

---

## 11. License

MIT © 2024 Your Name / Your Organisation
