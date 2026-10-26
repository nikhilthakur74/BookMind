# BookMind Phase 7: Complete Frontend Setup & Verification Guide

This document provides setup instructions, architectural documentation, and a comprehensive manual end-to-end test checklist for **BookMind Phase 7: The Complete Frontend**.

---

## 1. System Overview & Technology Stack

The BookMind frontend provides a Claude-inspired reading, conversational analysis, and RAG evaluation interface built with:

- **Framework**: React 19 + Vite 8 + TypeScript (strict mode enabled).
- **Styling**: Tailwind CSS v4 + Vanilla CSS custom variables (`#FAF9F5` warm neutral background, `#2D2B28` text, `#ECE7DE` minimal borders, `#C86432` warm terracotta accents).
- **Typography**: Google Fonts pairing — *Newsreader* (editorial serif headers) and *Inter* (high-legibility UI text).
- **Icons**: Lucide React (`lucide-react`).
- **Markdown & Code**: Lightweight, zero-dependency custom parser with fenced code blocks, callouts, blockquotes, headings, lists, and inline formatting.
- **Backend**: FastAPI running on `http://127.0.0.1:8000` with 54 automated pytest tests passing.

---

## 2. Environment Setup & Launch Commands

### Prerequisites
- **Python 3.11+** (Windows, Linux, or macOS)
- **Node.js 18+** with `npm`

### Step 1: Start the FastAPI Backend
```powershell
# Navigate to the backend directory
cd c:\bookreader\backend

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Start the uvicorn development server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Health check URL*: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)  
*Swagger Documentation*: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 2: Start the Vite Frontend
```powershell
# In a separate terminal, navigate to the frontend directory
cd c:\bookreader\frontend

# Install dependencies (already installed)
npm install

# Start the Vite development server
npm run dev -- --host 127.0.0.1 --port 5173
```
*Frontend URL*: [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

## 3. Automated Test Suites & Production Build

### Running Frontend Vitest Unit Tests
The frontend includes 12 automated unit and integration tests across components, citation expansion, Markdown rendering, and typed API client error handling:
```powershell
cd c:\bookreader\frontend
npm run test
```
*Expected Output*:
```
 ✓ src/test/api.test.ts (4 tests)
 ✓ src/test/MarkdownRenderer.test.tsx (5 tests)
 ✓ src/test/CitationsView.test.tsx (3 tests)

 Test Files  3 passed (3)
      Tests  12 passed (12)
```

### Running the Production Frontend Build
```powershell
cd c:\bookreader\frontend
npm run build
```
*Expected Output*:
```
✓ 1942 modules transformed.
dist/index.html                   1.05 kB
dist/assets/index-Dct0e7j1.css   42.60 kB
dist/assets/index-BgNrU7zh.js   333.77 kB
✓ built in <1s (0 errors, 0 warnings)
```

### Running Backend Pytest Regression Tests
```powershell
cd c:\bookreader\backend
.\.venv\Scripts\pytest -v
```
*Expected Output*: `54 passed in ~9s`.

---

## 4. Key Frontend Features & UI Architecture

### 1. Reader & Chat Workspace (`ChatWorkspace.tsx`)
- **Reading-Optimized Layout**: Centered `max-w-3xl` reading column with generous whitespace, warm `#FAF9F5` background, and restrained borders.
- **Empty State & Thematic Inquiries**: Displays the active book's central crux banner alongside categorized suggested questions across 5 inquiry types:
  - `Book Crux` ("What is the central thesis and overall crux of this book?")
  - `Conceptual` ("Explain the core mechanisms and principles governing the author's primary argument.")
  - `Chapter Summary` ("Provide a structured summary of the arguments and takeaways in Chapter 1.")
  - `Cross-Chapter Comparison` ("Compare and contrast the contrasting models or approaches presented across chapters.")
  - `Factual` ("What specific conditions or criteria must be satisfied according to the author?")
- **Markdown Response Rendering**: Zero-dependency safe parser rendering code blocks, bold, italics, blockquotes, and lists.
- **Expandable Verified Citations**: Inline badges displaying exact page numbers (`Page 2` or `Pages 3–4`), chunk IDs (`book_..._c0001`), chapter/section breadcrumbs, and clickable expandable accordions revealing the verbatim `quoted_snippet`.
- **Safety & Uncertainty**: Displays a dedicated insufficient-evidence alert banner when evidence is missing, ensuring the model admits uncertainty rather than hallucinating.
- **Pipeline Controls**: Collapsible toolbar to toggle between Hierarchical RAG, Hybrid RAG, and Baseline RAG, toggle the neural reranker, and adjust retrieval `top_k` (1–10).

### 2. Multi-Stage Ingestion Modal (`UploadModal.tsx`)
- Drag-and-drop PDF dropzone with size validation and PDF MIME type checking.
- Sequential ingestion progress tracker:
  1. *Uploading & Extracting*: PyMuPDF text and heading extraction.
  2. *Dense & Lexical Indexing*: BGE-M3 embeddings in Qdrant + Okapi BM25 index.
  3. *Hierarchical Synthesis*: Staged bottom-up summarization across Levels 2–4.
- Automatic library refresh upon upload completion.

### 3. Four-Level Hierarchy Inspector (`HierarchyModal.tsx`)
- **Level 4: Book Crux & Thesis**: Displays central thesis, overall crux, and essential takeaways with claim provenance badges (`author_stated` vs `model_inferred`).
- **Level 3: Chapter Summaries**: Interactive chapter accordion displaying central arguments, supporting points, assumptions, examples, and conclusions.
- **Level 2: Section Concepts**: Section-by-section breakdown of definitions, key concepts, and reasoning chains.

### 4. RAG Evaluation Dashboard (`EvaluationDashboard.tsx`)
- **Dataset Size & Validation Status**:
  - Live metric counters for Total Questions, Validated Ground Truth, Drafts Awaiting Review, and Rejected Items.
  - "Generate Draft Dataset" trigger calling LLM-assisted question generation.
  - Interactive "Review" modal allowing evaluators to verify reference answers, examine expected chunks and page numbers, and transition items (`draft` -> `validated` / `rejected`) with curation notes.
- **Benchmark Experiment Runner**:
  - Pipeline selection checkboxes: Baseline RAG (vector only), Hybrid RAG (Vector + BM25 + RRF), Hierarchical RAG (Levels 1–4), and Hierarchical RAG with Reranking.
  - Controls for retrieval `top_k`, "Evaluate Validated Ground Truth Only", and optional LLM-as-a-judge scoring.
- **Side-by-Side Pipeline Comparison Matrix**:
  - Directly compares Recall@k, Precision@k, MRR, Hit Rate@k, Faithfulness, Relevance, Citation Validity, Unsupported Claim Rate, Failure Counts, and Latency.
- **Crux Completeness Rubric (5 Non-Collapsed Dimensions)**:
  - All 5 dimensions visibly displayed as independent progress bars and percentage metrics:
    1. *Thesis Capture*
    2. *Supporting Arguments*
    3. *Key Concepts*
    4. *Evidence Grounding*
    5. *Conciseness & Clarity*
- **Diagnosed Failure Cases Inspector**:
  - Filterable audit view highlighting queries that triggered failure criteria (e.g. `retrieval_miss`, `unfaithful_generation`, `false_abstention`).
  - Displays side-by-side comparison of expected chunks vs retrieved chunks and generation metrics.
- **Historical Runs & Export**:
  - Historical experiment run selector table.
  - Direct one-click download buttons for **CSV** and **JSON** exports.

---

## 5. Manual End-to-End Test Checklist

Perform the following 10 verification steps to confirm end-to-end frontend functionality:

| # | Step / User Action | Expected Result | Verified |
| :--- | :--- | :--- | :---: |
| **1** | Open `http://localhost:5173/` in the browser | Claude-inspired warm neutral layout loads with header status showing `Online (XXms)`. | [x] |
| **2** | Inspect the Sidebar | Active book `distributed_systems_guide.pdf` is selected with `Ready` badge (4 pages, 4 chunks). Conversation history list shows previous threads. | [x] |
| **3** | Click "+ New Conversation" | Active conversation clears; chat area displays book crux summary banner and 5 suggested inquiry chips. | [x] |
| **4** | Click a Suggested Question (e.g., "What is the central thesis and overall crux...?") | Assistant synthesizes response. Answering format explains the crux first, followed by evidence synthesis. Markdown formatting renders cleanly. | [x] |
| **5** | Expand a Citation Badge | Citation accordion expands to show verbatim quoted snippet, chapter title, section title, chunk ID, and page range (`Pages X–Y`). | [x] |
| **6** | Submit an Unanswerable Query (e.g., "What temperature should an oven be set to when baking cookies?") | Assistant surfaces the amber `Insufficient Direct Grounding` alert card and admits uncertainty rather than hallucinating. | [x] |
| **7** | Click "Crux & Levels" in the header | Modal opens displaying Level 4 Book Crux, Level 3 Chapter Summaries, and Level 2 Section Concepts. Tab switching works without closing. | [x] |
| **8** | Navigate to the "Evaluation" Tab in Header | Dataset curation card displays question counts, validation status pills, and question review table. | [x] |
| **9** | Click "Review" on an Evaluation Item | Review modal opens showing question, reference answer, and allows setting status to `Validated`, `Draft`, or `Rejected`. | [x] |
| **10** | Verify Evaluation Benchmark & Export | Pipeline comparison matrix shows empirical metrics across retrieval and generation. Crux rubric shows all 5 separate dimensions. Clicking "Export CSV" and "Export JSON" initiates file downloads. | [x] |

---

## 6. Security & Key Management

- **Zero Groq Key Exposure**: All inference and embedding calls are proxied through FastAPI endpoints. No API keys or credentials are stored or accessible in the client bundle.
- **Configurable Endpoints**: `VITE_API_BASE_URL` in `frontend/.env` allows deployment against remote or containerized backends.
- **Request Resilience**: Chat requests enforce a 45-second timeout with abort controllers and clear retry buttons on network interruptions.
