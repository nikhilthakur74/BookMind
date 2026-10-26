/**
 * Typed API client for BookMind backend with error and timeout handling.
 */

import type {
  BookDetailResponse,
  BookHierarchy,
  BookListResponse,
  BookMetadata,
  ChatRequest,
  ChatResponse,
  ConversationHistory,
  EvaluationDataset,
  EvaluationItem,
  ExperimentRun,
  HealthStatus,
  PipelineComparisonReport,
  PipelineType,
  ValidationStatus,
} from '../types/api';

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface HealthCheckResult {
  data: HealthStatus | null;
  latencyMs: number;
  error: string | null;
}

/**
 * Perform a live health check call to the FastAPI backend.
 */
export async function checkBackendHealth(): Promise<HealthCheckResult> {
  const startTime = performance.now();
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const latencyMs = Math.round(performance.now() - startTime);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data: HealthStatus = await response.json();
    return { data, latencyMs, error: null };
  } catch (err: unknown) {
    const latencyMs = Math.round(performance.now() - startTime);
    const message = err instanceof Error ? err.message : 'Failed to connect to backend server';
    return { data: null, latencyMs, error: message };
  }
}

// ==============================================================================
// Books & Ingestion API
// ==============================================================================

export async function listBooks(): Promise<BookMetadata[]> {
  const res = await fetch(`${API_BASE_URL}/api/books`);
  if (!res.ok) {
    throw new Error(`Failed to list books: HTTP ${res.status}`);
  }
  const data: BookListResponse = await res.json();
  return data.books;
}

export async function getBook(bookId: string): Promise<BookDetailResponse> {
  const res = await fetch(`${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}`);
  if (!res.ok) {
    throw new Error(`Failed to get book ${bookId}: HTTP ${res.status}`);
  }
  return res.json();
}

export async function uploadBook(file: File): Promise<BookDetailResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s upload timeout

  try {
    const res = await fetch(`${API_BASE_URL}/api/books/upload`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      let errorMsg = `Upload failed (HTTP ${res.status})`;
      try {
        const errorJson = await res.json();
        if (typeof errorJson.detail === 'string') {
          errorMsg = errorJson.detail;
        } else if (errorJson.detail?.message) {
          errorMsg = errorJson.detail.message;
        }
      } catch {
        // use default error message
      }
      throw new Error(errorMsg);
    }

    return res.json();
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Upload timed out. The file might be too large or the server is busy.');
    }
    throw err;
  }
}

export async function deleteBook(bookId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    throw new Error(`Failed to delete book ${bookId}: HTTP ${res.status}`);
  }
}

// ==============================================================================
// Indexing API
// ==============================================================================

export async function indexBook(bookId: string, force = false): Promise<BookMetadata> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/index?force=${force}`,
    { method: 'POST' }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Indexing failed with HTTP ${res.status}`);
  }
  return res.json();
}

// ==============================================================================
// Hierarchical Knowledge API
// ==============================================================================

export async function generateHierarchy(bookId: string, forceRestart = false): Promise<BookHierarchy> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/hierarchy/generate`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force_restart: forceRestart }),
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Hierarchy synthesis failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function getHierarchy(bookId: string): Promise<BookHierarchy | null> {
  const res = await fetch(`${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/hierarchy`);
  if (res.status === 404) {
    return null;
  }
  if (!res.ok) {
    throw new Error(`Failed to fetch hierarchy: HTTP ${res.status}`);
  }
  return res.json();
}

// ==============================================================================
// Conversational RAG API
// ==============================================================================

export async function chatWithBook(bookId: string, request: ChatRequest): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 45000); // 45s timeout for deep synthesis

  try {
    const res = await fetch(`${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Chat synthesis failed: HTTP ${res.status}`);
    }

    return res.json();
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Chat request timed out. Please try again.');
    }
    throw err;
  }
}

export async function listConversations(bookId: string): Promise<ConversationHistory[]> {
  const res = await fetch(`${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/conversations`);
  if (!res.ok) {
    throw new Error(`Failed to list conversations: HTTP ${res.status}`);
  }
  return res.json();
}

export async function getConversation(bookId: string, conversationId: string): Promise<ConversationHistory> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/conversations/${encodeURIComponent(conversationId)}`
  );
  if (!res.ok) {
    throw new Error(`Failed to get conversation: HTTP ${res.status}`);
  }
  return res.json();
}

// ==============================================================================
// Evaluation Framework API (Phase 6)
// ==============================================================================

export async function generateDraftDataset(bookId: string): Promise<EvaluationDataset> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/evaluations/datasets/generate`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ questions_per_type: 2 }),
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to generate draft dataset: HTTP ${res.status}`);
  }
  return res.json();
}

export async function listEvaluationDatasets(bookId: string): Promise<EvaluationDataset[]> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/evaluations/datasets`
  );
  if (!res.ok) {
    throw new Error(`Failed to list evaluation datasets: HTTP ${res.status}`);
  }
  return res.json();
}

export async function getEvaluationDataset(bookId: string, datasetId: string): Promise<EvaluationDataset> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/evaluations/datasets/${encodeURIComponent(datasetId)}`
  );
  if (!res.ok) {
    throw new Error(`Failed to get dataset: HTTP ${res.status}`);
  }
  return res.json();
}

export async function updateDatasetItemValidation(
  bookId: string,
  datasetId: string,
  itemId: string,
  status: ValidationStatus,
  notes?: string
): Promise<EvaluationItem> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/evaluations/datasets/${encodeURIComponent(datasetId)}/items/${encodeURIComponent(itemId)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        validation_status: status,
        notes: notes || undefined,
      }),
    }
  );
  if (!res.ok) {
    throw new Error(`Failed to update item status: HTTP ${res.status}`);
  }
  return res.json();
}

export async function runEvaluationBenchmark(
  bookId: string,
  datasetId: string,
  pipelines: PipelineType[] = ['baseline_rag', 'hybrid_rag', 'hierarchical_rag'],
  topK = 5,
  validatedOnly = true,
  enableLlmJudge = false
): Promise<PipelineComparisonReport> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/evaluations/run`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: datasetId,
        pipelines,
        top_k: topK,
        validated_only: validatedOnly,
        enable_llm_judge: enableLlmJudge,
      }),
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Evaluation run failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function listEvaluationRuns(bookId: string): Promise<ExperimentRun[]> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/evaluations/runs`
  );
  if (!res.ok) {
    throw new Error(`Failed to list evaluation runs: HTTP ${res.status}`);
  }
  return res.json();
}

export async function getEvaluationRun(bookId: string, runId: string): Promise<ExperimentRun> {
  const res = await fetch(
    `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/evaluations/runs/${encodeURIComponent(runId)}`
  );
  if (!res.ok) {
    throw new Error(`Failed to get evaluation run: HTTP ${res.status}`);
  }
  return res.json();
}

export function getEvaluationExportUrl(bookId: string, runId: string, format: 'csv' | 'json'): string {
  return `${API_BASE_URL}/api/books/${encodeURIComponent(bookId)}/evaluations/runs/${encodeURIComponent(runId)}/export?format=${format}`;
}
