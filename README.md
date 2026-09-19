# MythosAI: Local AI Fiction Co-Author with RAG-Powered Lorebook

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg?style=flat)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg?style=flat&logo=ollama&logoColor=white)](https://ollama.com/)
[![Docker Compose](https://img.shields.io/badge/Docker_Compose-Multi--Container-blue.svg?style=flat&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

**MythosAI** is a production-quality, local AI creative writing co-author and story development studio. It combines open-source Large Language Models (LLMs) running locally via **Ollama** with a persistent vector database (**ChromaDB**) and **Sentence-Transformers** to provide Retrieval-Augmented Generation (RAG) over a custom story lorebook.

All data, models, canon notes, and manuscripts remain **100% private and offline** on your machine.

---

## Key Features

- 🧠 **Local LLM Deployment**: Native orchestration with Ollama for models like `llama3.1:8b`, `mistral:7b`, or `llama3.2:1b`.
- 📜 **Persistent Lorebook & Semantic RAG**: Add worldbuilding notes, characters, artifacts, and factions into ChromaDB; relevant canon is dynamically retrieved via cosine vector similarity using `all-MiniLM-L6-v2` embeddings.
- 🎭 **Persona Steering & Prompt Templating**: Jinja2-powered dynamic prompt engine enforcing consistent narrative voice, tone, and pacing via `prompts/persona.md`.
- 🎛️ **Creative Hyperparameter Control**: Fine-tune `temperature`, `top_p`, `repeat_penalty`, and `max_tokens` with writing presets (Balanced Fantasy, Tight Mystery, Surreal/Mythic, Fast Dialogue).
- ⚡ **Real-Time Token Streaming**: Low-latency token-by-token streaming via Server-Sent Events (SSE) and asynchronous generators.
- 🎨 **Dark-Themed Fiction Studio Web UI**: Full-featured web interface with live manuscript metrics, RAG context inspector, lore manager, and parameter sliders.
- 🐳 **Docker Compose Orchestration**: Containerized multi-service setup (`app`, `ollama`, `chromadb`) with healthchecks and volume persistence.
- 🧪 **Comprehensive Automated Testing**: 100% contract coverage verifying API schemas, RAG keyword grounding (`Aethelgard`), and temperature variation.

---

## Architecture Diagram

```
                             +-----------------------------------+
                             |     Fiction Studio Web UI         |
                             |  - Manuscript Editor & Streamer   |
                             |  - Lorebook Management & Visuals  |
                             |  - Parameter Studio & Inspector   |
                             +-----------------+-----------------+
                                               | HTTP / SSE
                                               v
+-----------------------------------------------------------------------------------------+
|                               FastAPI Application Service                               |
|                                                                                         |
|   [/health]      [/api/lore]      [/api/generate]      [/api/generate/stream]           |
|                                                                                         |
|   +-----------------------+     +------------------------+     +--------------------+   |
|   |    Lorebook Manager   |     |    RAG Orchestrator    |     |  Ollama LLM Client |   |
|   |  - Chunking & Ingest  |     |  - SentenceTransformer |     |  - Persona Prompt  |   |
|   |  - Metadata Tagging   |     |  - Top-K Context Query |     |  - Param Steering  |   |
|   +-----------+-----------+     +-----------+------------+     +----------+---------+   |
|               |                             |                             |             |
+---------------|-----------------------------|-----------------------------|-------------+
                |                             |                             |
                v                             v                             v
+--------------------------------+                          +-----------------------------+
|      ChromaDB Vector Store     |                          |      Ollama LLM Engine      |
|    (chromadb/chroma:0.4.24)    |                          |    (ollama/ollama:latest)   |
|  - Story Canon Vector Space    |                          |  - Llama 3.1 / Mistral      |
|  - Cosine / L2 Similarity      |                          |  - Real-Time Token Stream   |
+--------------------------------+                          +-----------------------------+
```

---

## Project Structure

```
.
├── .env.example              # Documented environment variable template
├── .env                      # Local environment configuration
├── docker-compose.yml        # Multi-container orchestration (app, ollama, chromadb)
├── Dockerfile                # Python 3.11 application container
├── requirements.txt          # Python dependencies
├── prompts/
│   └── persona.md            # System persona prompt (>100 chars)
├── docs/
│   └── parameter_effects.md  # Empirical study on Temperature, Top-P, Repeat Penalty
├── src/
│   ├── __init__.py
│   ├── main.py               # FastAPI entrypoint, lifespan & static mounting
│   ├── config.py             # Pydantic Settings configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Pydantic contract schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py # SentenceTransformer vector embedding engine
│   │   ├── chroma_service.py    # ChromaDB client adapter & operations
│   │   ├── ollama_service.py    # Asynchronous Ollama inference client
│   │   └── rag_service.py       # RAG context retriever & Jinja2 prompt builder
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py         # /health endpoint with service status
│   │   └── api.py            # /api/lore, /api/generate, /api/models endpoints
│   └── static/               # Fiction Studio Frontend
│       ├── index.html        # Responsive Studio UI
│       ├── css/style.css     # Premium dark-mode styling
│       └── js/app.js         # Interactive client logic & streaming
└── tests/
    ├── __init__.py
    ├── conftest.py           # Test fixtures and mock services
    ├── test_api.py           # Contract verification & integration tests
    ├── test_rag.py           # RAG retrieval & embedding tests
    └── test_parameters.py    # Parameter bounds & schema validation tests
```

---

## Quickstart Guide

### Option 1: Running with Docker Compose (Recommended)

1. **Clone the repository and copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Launch all services**:
   ```bash
   docker-compose up -d --build
   ```

3. **Verify container health status**:
   ```bash
   docker-compose ps
   ```

4. **Pull the desired LLM model inside Ollama container** (first time only):
   ```bash
   docker-compose exec ollama ollama pull llama3.1:8b
   ```

5. **Open the Fiction Studio**:
   Navigate to [http://localhost:8080](http://localhost:8080) in your browser.

---

### Option 2: Running Locally (Standalone Mode)

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama** (if running locally):
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ```

3. **Run the application**:
   ```bash
   python -m uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
   ```

4. **Access the application**:
   - Web UI: [http://localhost:8080](http://localhost:8080)
   - Interactive API Docs (Swagger): [http://localhost:8080/docs](http://localhost:8080/docs)

---

## API Reference & Contracts

### 1. Ingest Lore Snippet
**`POST /api/lore`** (Status: `201 Created`)

**Request**:
```json
{
  "content": "The ancient sword is named 'Aethelgard' and it glows with a faint blue light.",
  "metadata": {
    "category": "Artifact",
    "rarity": "Legendary"
  }
}
```

**Response**:
```json
{
  "status": "success",
  "id": "e83921ab-7a63-4b61-9c88-c44d1872f90a"
}
```

---

### 2. Generate Story Segment (RAG-Augmented)
**`POST /api/generate`** (Status: `200 OK`)

**Request**:
```json
{
  "prompt": "The hero unsheathes his glowing blade as shadows stir.",
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.15,
    "max_tokens": 512
  },
  "top_k": 3
}
```

**Response**:
```json
{
  "story_segment": "Drawing Aethelgard from its scabbard, a faint azure luminescence bathed the damp stone walls...",
  "retrieved_lore": [
    {
      "id": "e83921ab-7a63-4b61-9c88-c44d1872f90a",
      "content": "The ancient sword is named 'Aethelgard' and it glows with a faint blue light.",
      "relevance_score": 0.89
    }
  ],
  "model_used": "llama3.1:8b"
}
```

---

### 3. Stream Story Generation (SSE / Chunked)
**`POST /api/generate/stream`** (Status: `200 OK`, `Content-Type: text/plain`)

Streams tokens asynchronously as they are generated by the model.

---

### 4. Health Check
**`GET /health`** (Status: `200 OK`)

```json
{
  "status": "healthy",
  "ollama_status": "connected",
  "chromadb_status": "connected",
  "ollama_connected": true,
  "chromadb_connected": true,
  "embedding_model": "all-MiniLM-L6-v2",
  "active_model": "llama3.1:8b",
  "lore_entries_count": 4
}
```

---

## Running Automated Tests

Run the full pytest suite to verify all API contracts, RAG keyword grounding, and parameter variance:

```bash
pytest -v
```

---

## License
MIT License. Built for local AI creative collaboration.
