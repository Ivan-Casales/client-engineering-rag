# Watsonx RAG Assistant

> **Strictly documentation-based RAG chatbot.**

---

## Table of Contents

1. [What is this app?](#1-what-is-this-app)
2. [Architecture & Technical Decisions](#2-architecture--technical-decisions)

   1. [Backend](#21-backend)
   2. [Frontend](#22-frontend)
   3. [IBM watsonx.ai Integration](#23-ibm-watsonxai-integration)
   4. [Technology choices & justification](#24-technology-choices--justification)
   5. [Git workflow](#25-git-workflow)
   6. [Project management (Azure DevOps)](#26-project-management-azure-devops)
   7. [Security considerations](#27-security-considerations)
   8. [Testing strategy](#28-testing-strategy)
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
9. [Author](#9-author)
---

## 1. What is this app?

`Watsonx RAG Assistant` is a full‑stack reference implementation of **Retrieval‑Augmented Generation (RAG)** that answers questions *only* with facts extracted from the official *Unleashing the Power of AI with IBM watsonx.ai* white‑paper.

It demonstrates how IBM watsonx.ai can:

* generate **embeddings** for knowledge retrieval,
* perform **LLM inference** to craft grounded answers, and
* **rerank** passages with a cross‑encoder.

---

## 2. Architecture & Technical Decisions

### 2.1 Backend

| Folder / Module| Responsibility|
| ---------------|---------------|
| **`app/`**     | Python package that bundles the entire back‑end application. Entry‑point: `main.py`.|
| **`app/api/`** | HTTP layer — FastAPI **routers** (`routes.py`) and request / response **schemas** (`schemas.py`). Only thin controllers; no business logic. |
| **`app/core/`**| Cross‑cutting concerns: configuration (`config.py`), models ids, constants, and environment helpers. Keeps the 12‑Factor contract.|
| **`app/services/`**| Application services — each subfolder addresses one bounded context (SRP):|
| ├── **`rag/`**| RAG orchestration: `rag_pipeline.py`, `chat_service.py`, and `reranker.py`. Combines retrieval, reranking and generation.|
| ├── **`utility/`**| Stateless helpers such as `pdf_parser.py`, `prompt_templates.py`, `prompt_chat.py`, and `security.py`.|
| ├── **`vectorstore/`** | Persistence layer for embeddings: `chroma_db.py` wraps Chroma, while `loader_service.py` handles batch upserts.|
| └── **`watsonx/`**| External‑service adapters that hide IBM watsonx.ai SDK calls (`watsonx_credentials.py`, `watsonx_embeddings.py`, `watsonx_llm.py`).|
| **`container.py`**| Lightweight dependency‑injection container wiring services together.|
| **`tests/`**| Pytest test‑suite. Sub‑packages: `unit_test/` (pure functions), `integration/` (FastAPI + mocked watsonx), and `test_utility/` (helpers).|

This folder layout follows a **clean‑architecture** style: the outer layers (`api`) depend on the inner layers (`services`, `core`), never the opposite.

### 2.2 Frontend

A minimalist **Streamlit** UI located in `front‑end/` provides a wizard‑like flow:

1. **Upload** – drag‑and‑drop one or more PDFs; triggers async indexing.
2. **Question** – single‑shot Q\&A for quick checks.
3. **Chat** – multi‑turn conversation with short‑term memory.

### 2.3 IBM watsonx.ai Integration

| Capability | Env var → Model ID | Default model| Wrapper|
| ---------- | ----------------------- | -------------------------------------- | ----------------------- |
| Embeddings | `$WATSONX_EMBED_MODEL`  | `ibm/slate-30m-english-rtrvr-v2`       | `watsonx_embeddings.py` |
| Inference  | `$WATSONX_LLM_MODEL`    | `ibm/granite-3-3-8b-instruct`          | `watsonx_llm.py`        |
| Reranking  | `$WATSONX_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `reranker.py`           |

**ibm/slate-30m-english-rtrvr-v2**: A small model (30 million parameters) built to quickly and accurately find information in English technical texts, like PDFs. It’s fast and cheap to use, but still very effective for search tasks.

**ibm/granite-3-3-8b-instruct**: A larger model (8 billion parameters) that understands and follows instructions well. It’s trained to be helpful and safe, and runs fast in chat settings without needing a lot of computing power.

**cross-encoder/ms-marco-MiniLM-L-6-v2**: A lightweight model (110 million parameters) that improves search results by re-ranking them. It boosts result quality without using much memory or slowing things down.

### 2.4 Technology choices & justification

| Stack / Tool       | Why it was chosen                                                        |
| ------------------ | ------------------------------------------------------------------------ |
| **Python 3.11**    | Offers better performance and useful new features like pattern matching. |
| **FastAPI**        | It's fast, easy to use, and auto-generates API docs.                     |
| **Streamlit**      | Lets you build simple UIs with no setup – great for quick prototypes.    |
| **ChromaDB**       | Super fast, serverless, and easy to use for handling vector data.        |
| **PyPDF2**         | Works reliably for extracting text from PDFs without hassle.             |
| **LangChain**      | Used it to simplify prompt logic and basic workflow handling.            |
| **Docker Compose** | Makes everything run the same across different operating systems.        |
| **GitHub Actions** | Automatically runs tests and checks on every pull request.               |
| **Pytest**         | Simple to use and supports both unit and integration tests.              |

### 2.5 Git workflow

The repository follows **Git Flow** with the following conventions:

* `main` – production‑ready, tagged releases.
* `develop` – integration branch.
* `feature/*`, `hotfix/*`, `release/*` – short‑lived branches.
* Conventional commit prefixes: **`feat:`**, **`fix:`**, **`refactor:`**, etc.

### 2.6 Project management (Azure DevOps)

All requirements were broken down into **Epics → Issues → Tasks** in an [Azure DevOps project](https://dev.azure.com/casalesivan/IBM%20Challenge/_boards/board/t/IBM%20Challenge%20Team/Epics). 

A sprint of **7 working days** was planned: [Planification](https://dev.azure.com/casalesivan/IBM%20Challenge/_wiki/wikis/IBM-Challenge.wiki/1/Planificaci%C3%B3n-de-trabajo).

The burndown chart stayed close to the plan, with only about 5% variation.

### 2.7 Security considerations

* Secrets stored in **`.env`** and referenced via `os.getenv`.
* `.env` is in **`.gitignore`**; CI injects secrets at runtime.
* Input sanitisation via Pydantic and length guards.
* Docker images run as non‑root, read‑only FS, no exposed SSH.

### 2.8 Testing strategy

| Level           | Framework      | What is covered                                |
| --------------- | -------------- | ---------------------------------------------- |
| **Unit**        | Pytest         | PDF parser, chunker, prompt templates          |
| **Integration** | Pytest + httpx | End‑to‑end `/api/query` with mocked watsonx.ai |
| **E2E manual**  | Browser        | Exploratory testing in Streamlit                    |

Inside the tests folder are the instructions to run it correctly (in ReadmeBeforeRunTest).

---

## 3. Indexing pipeline

The end‑to‑end indexing flow transforms **raw PDFs → searchable vectors** in five stages:

| # | Stage | Key details|
| - | ----- | -----------|
| 1 | **Ingest**| Receives the PDF **bytes** from UI or API.|
| 2 | **Temporary storage** | Bytes are written to a secure `NamedTemporaryFile` (auto‑deleted).|
| 3 | **Parse & Chunk**| - `PyPDFLoader` extracts page text.<br> - `CharacterTextSplitter` slices into \~500-character chunks with 100-character overlap ⇒ ~350 tokens, staying under the 512-token embed limit. |
| 4 | **Vectorise**| Each chunk is sent to watsonx.ai using **`ibm/slate-30m-english-rtrvr-v2`**.|
| 5 | **Persist**| Chunks + vectors are upserted into a local `.chromadb/` directory.|
| 6 | **Return**| The endpoint replies with `{ "detail": "<n> chunks indexed successfully." }` to inform the UI.|

---

## 4. Query pipeline

The **retrieval‑and‑generation** flow converts a user question into a grounded answer in six stages:

| # | Stage | Key details |
| - | ----- | ----------- |
| 1 | **Embed query** | - Question is first sanitised by `security.sanitize_input`.<br> - `WatsonXEmbeddings.embed_query` converts it into a **1024‑d vector** (`ibm/slate‑30m‑english-rtrvr-v2`) with input truncated to ≤ 512 tokens. |
| 2 | **K‑NN search** | - `rag_chain.retriever.get_relevant_documents` executes a cosine‑similarity **k‑nearest‑neighbours** lookup (default **k = 6**) against ChromaDB.<br> - Returned `Document` objects carry `page` & `source` metadata for later citation. |
| 3 | **Rerank (optional)** | - The candidate docs are fed to `ReRanker` which applies the **`cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2`** model.<br> - Scores are sorted descending and top‑`k` (default = 5) are kept. |
| 4 | **Prompt** | - The selected chunks are concatenated and interpolated into `STRICT_CONTEXT_PROMPT`.<br> - The prompt *explicitly forbids* using out‑of‑context knowledge and instructs the LLM to cite sources.|
| 5 | **Generate** | - `WatsonXLLM._call` sends the prompt to **`ibm/granite‑3‑3‑8b‑instruct`** with `temperature=0` (deterministic), `max_new_tokens=256`, and a stop sequences.<br> - The raw response is cleaned of trailing artefacts by a regex pass. |
| 6 | **Return / Stream** | - FastAPI returns `{ "answer": "…" }` as JSON.<br>• The Streamlit UI displays it incrementally using `st.write_stream` for a chat‑like typing effect.       

---

## 5. Requirements

* **Python ≥ 3.11** (& poetry / venv, if running without Docker)
* IBM watsonx.ai account + API‑Key
* Internet access to `api.us‑south.ml.cloud.ibm.com` (or your region)
* Docker (Optional run)

### Environment variables

Add a .env file in source/back-end/.env, then copy and complete:
```env
WATSONX_URL=''
WATSONX_PROJECT_ID=''
WATSONX_APIKEY=''
CHROMA_PERSIST_DIRECTORY=.chromadb
```
---

Add a .env file in source/front-end/.env, then copy and complete:
```env
API_BASE_URL=url
```

---

## 6. Running the app

### 6.1 With Docker

```bash
# clone repo
$ git clone https://github.com/your‑org/watsonx‑rag‑assistant.git
$ cd watsonx‑rag‑assistant

# create & fill .env (follow previous instructions)

# build & start
$ docker compose up --build -d

# API  : http://localhost:8000/docs
# UI   : http://localhost:8501
```

### 6.2 Without Docker

First create & fill .env (follow previous instructions)

**Back-end**:
Open a console in source/back-end/ and enter:
```bash
# Python & poetry / venv assumed installed
$ python -m venv venv
$ .\venv\Scripts\activate
$ pip install -r requirements.txt
$ uvicorn app.main:app –reload
```

---
**Front-end**:
Open a (different) console in source/front-end/ and enter:
```bash
# Python & poetry / venv assumed installed
$ python -m venv venv
$ .\venv\Scripts\activate
$ pip install -r requirements.txt
$ streamlit run streamlit_app.py

```

If you are using Linux enter (instead: source venv/bin/actívate):
- .\venv\Scripts\activate

---

## 7. Usage

### 7.1 Web UI

1. **Upload** your PDF(s) in Upload Page (Option on the left).
2. Switch to **Question** or **Chat** tab.
3. Ask anything about watsonx.ai.

### 7.2 REST API

```bash
# index document
http://localhost:8000/api/upload

# ask a question
curl -X POST http://localhost:8000/api/query \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is Watsonx?"}'
```

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

(The backend need to be active)

---

## 9. Author

### Iván Casales