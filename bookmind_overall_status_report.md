# BookMind Project Status & Progress Report (Phases 1–7)

**BookMind** is a domain-specialized Retrieval-Augmented Generation (RAG) platform engineered for deep book comprehension, bottom-up hierarchical knowledge extraction, evidence-grounded conversational inquiry, and reproducible empirical benchmarking.

---

## 1. Executive Summary & Current Status

| Component | Status | Details |
| :--- | :---: | :--- |
| **Backend Core & Ingestion** | ✅ Complete | FastAPI, PyMuPDF, layout extraction, OCR detection, chunking |
| **Indexing & Hybrid Retrieval** | ✅ Complete | BGE-M3 dense vectors, persistent Qdrant, Okapi BM25, RRF fusion ($k=60$) |
| **Hierarchical Knowledge** | ✅ Complete | 4 Knowledge Levels (Evidence, Section, Chapter, Book), crux extraction |
| **Conversational RAG API** | ✅ Complete | Crux-first answering, strict book isolation, structured citations, safety abstention |
| **Evaluation Framework** | ✅ Complete | 6 question types, human-in-the-loop review, multi-pipeline comparative benchmarking |
| **Complete Frontend UI** | ✅ Complete | Claude-inspired warm neutral UI (`#FAF9F5`), React 19, Vite, Tailwind CSS v4 |
| **Cloud & LLM Integration** | ✅ Complete | Production Groq (`openai/gpt-oss-120b`) & Qdrant Cloud cluster |
| **Automated Verification** | ✅ Passing | **54/54 backend pytest tests** & **12/12 frontend vitest tests** passing |

---

## 2. Phase-by-Phase Implementation Summary

### Phase 1: Architecture, Scaffolding & Health Monitoring
- Asynchronous FastAPI backend structure using Pydantic Settings.
- Dynamic `/api/health` monitoring latency, storage directories, and database connectivity.
- Configurable environment parsing with `.env` hierarchy.

### Phase 2: PDF Ingestion & Document Intelligence
- High-fidelity PDF processing using `pymupdf` (fitz).
- Heuristic font-size and TOC-based chapter/section heading detection.
- Scanned/image-only PDF detection with OCR warning heuristics (`OCR_THRESHOLD_MIN_CHARS_PER_PAGE=50`).
- Deterministic page-aware chunking preserving chunk index, page boundaries, and section metadata.
- Isolated filesystem storage in `data/uploads/books/{book_id}/`.

### Phase 3: Dense Vectors, Lexical BM25 & Hybrid Retrieval
- **Dense Embeddings**: BGE-M3 (1024 dimensions) with configurable device selection (`cpu`, `cuda`, `mps`, `auto`).
- **Vector Database**: Qdrant integration supporting both local directory storage and remote Qdrant Cloud clusters with deterministic UUIDv5 point IDs for idempotent re-indexing.
- **Lexical Search**: Custom Okapi BM25 engine with smoothed IDF and document length normalization.
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF, $k=60$) combining semantic and lexical ranks with exact score calculations.
- **Strict Isolation**: Guaranteed multi-tenant/book-level isolation preventing cross-book evidence leakage.

### Phase 4: Hierarchical Book Understanding (Four Knowledge Levels)
- **Level 1 (Evidence)**: Original leaf chunks with immutable chunk IDs and page spans.
- **Level 2 (Section)**: Concise summaries, definitions, key concepts, and reasoning chains.
- **Level 3 (Chapter)**: Chapter crux, central argument, supporting points, assumptions, examples, and conclusions.
- **Level 4 (Book)**: Central thesis, overall crux, essential takeaways, and chapter relationships.
- Bottom-up summarization architecture: leaf chunks synthesize sections, sections synthesize chapters, chapters synthesize the book.
- Separation of concerns: summaries stored in separate `hierarchy.json` without modifying original evidence.
- Idempotent and resumable execution with staged processing for chapters with $>8$ chunks.

### Phase 5: Conversational RAG API & Provenance
- `POST /api/books/{book_id}/chat` endpoint with query categorization (`factual`, `conceptual`, `chapter_summary`, `book_crux`, `cross_chapter_comparison`, `quiz`).
- **Crux-First Answering**: Synthesizes the core crux directly before elaborating with multi-passage evidence.
- **Structured Citations**: Every claim is anchored to verbatim `quoted_snippet`, chunk ID, and page numbers.
- **Abstention & Safety**: Built-in uncertainty detection for queries outside book evidence.
- **Prompt Injection Defense**: Explicit delimiters treating retrieved content strictly as passive data.
- **Thread Management**: Conversation lifecycle persisted in `conversations/{conversation_id}.json`.

### Phase 6: Reproducible RAG Evaluation Framework
- **Dataset Curation**: Versioned datasets covering 6 question types (`factual`, `conceptual`, `summarization`, `multi_hop`, `comparison`, `unanswerable`).
- **Human-in-the-Loop Review**: Mandatory transition from `draft` to `validated` / `rejected` ground truth.
- **Multi-Pipeline Runner**: Benchmarks Baseline RAG, Hybrid RAG, Hierarchical RAG, and Hierarchical + Reranking.
- **Empirical Metrics**:
  - Retrieval: Recall@k, Precision@k, MRR, Hit Rate@k.
  - Generation: Faithfulness, Unsupported Claim Rate, Answer Relevance, Answer Correctness, Abstention Quality, Citation Validity.
  - 5-Dimension Crux Rubric: Thesis Capture, Supporting Arguments, Key Concepts, Evidence Grounding, Conciseness & Clarity.
- **Failure Case Inspector**: Automatic diagnosis of retrieval misses, unfaithful generations, and false abstentions.
- **Exports**: Instant CSV and JSON export endpoints for individual runs and comparison matrices.

### Phase 7: Complete Frontend (Claude-Inspired Editorial UI)
- **Aesthetic**: Warm `#FAF9F5` neutral background, subtle shadows, `#ECE7DE` hairline borders, and Google Fonts *Newsreader* (serif) + *Inter* (sans).
- **Workspace**: Generous whitespace reading column (`max-w-3xl`), suggested inquiry chips, zero-dependency Markdown rendering, and expandable citation cards revealing verbatim quotes.
- **Evaluation Dashboard**: Dedicated view for dataset review, benchmark execution, side-by-side comparative matrices, non-collapsed Crux rubric cards, failure inspection, and CSV/JSON downloads.
- **Interactive Modals**: Multi-stage PDF upload modal with sequential progress tracking and 4-level knowledge inspector modal.
- **Mobile Responsive**: Sliding sidebar drawer with backdrop blur.

---

## 3. Active Credentials & Cloud Configuration

The application is configured in `backend/.env` with your active cloud services:
- **Groq API**:
  - Provider: `groq`
  - Model: `openai/gpt-oss-120b` *(verified active model with strict JSON-mode structured outputs)*
  - Status: **Live and verified** (sample inquiry answered in 3.3s with high confidence)
- **Qdrant Cloud**:
  - Endpoint: `https://41cc9bf5-45c0-4f88-8ed3-a8d19e10d1a9.us-west-1-0.aws.cloud.qdrant.io`
  - Collection: `bookmind_chunks`
  - Status: **Connected and verified**

---

## 4. Verification & Testing Metrics

### Automated Pytest Suite (Backend)
- Total tests: **54 passed (100%)**
- Execution time: ~9.3 seconds
- Coverage:
  - `test_books.py`: Ingestion, headings, OCR detection, deduplication, chunk pagination.
  - `test_indexing.py`: Vector indexing, idempotency, duplicate prevention.
  - `test_retrieval.py`: BM25, RRF fusion, strict book isolation.
  - `test_hierarchy.py`: Provenance, bottom-up synthesis, long chapters, uncertainty.
  - `test_chat.py`: Crux-first answering, structured citations, insufficient evidence.
  - `test_evaluation.py`: Metrics calculations, failure cases, dataset curation, CSV/JSON export.

### Automated Vitest Suite (Frontend)
- Total tests: **12 passed (100%)**
- Test files:
  - `MarkdownRenderer.test.tsx`: Headings, bold/italics, blockquotes, code blocks, lists.
  - `CitationsView.test.tsx`: Expandable page citations, verbatim quotes, chunk IDs.
  - `api.test.ts`: URL composition, health checks, error handling, export formatting.
- Production build: `npm run build` completed in **885ms with 0 errors**.

---

## 5. Active Development Services

| Service | Port / URL | State |
| :--- | :--- | :--- |
| **FastAPI Backend** | `http://127.0.0.1:8000` | Running (`uvicorn app.main:app`) |
| **Vite Frontend UI** | `http://localhost:5173` | Running (`vite --host 127.0.0.1 --port 5173`) |
| **API Documentation** | `http://127.0.0.1:8000/docs` | Swagger UI accessible |
