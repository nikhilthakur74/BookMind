import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  checkBackendHealth,
  getEvaluationExportUrl,
  indexBook,
} from '../lib/api';

describe('API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('formats export URLs correctly for CSV and JSON', () => {
    const csvUrl = getEvaluationExportUrl('book_123', 'run_456', 'csv');
    expect(csvUrl).toContain('/api/books/book_123/evaluations/runs/run_456/export?format=csv');

    const jsonUrl = getEvaluationExportUrl('book_123', 'run_456', 'json');
    expect(jsonUrl).toContain('/api/books/book_123/evaluations/runs/run_456/export?format=json');
  });

  it('handles successful health check', async () => {
    const mockHealth = {
      status: 'healthy',
      version: '0.1.0',
      timestamp: '2026-10-20T12:00:00Z',
      services: {
        database: 'connected',
        embedding_model: 'loaded',
        vector_store: 'ready',
        llm: 'connected',
      },
      qdrant: {
        host: 'localhost',
        port: 6333,
        status: 'ready',
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    });

    const result = await checkBackendHealth();
    expect(result.data?.status).toBe('healthy');
    expect(result.error).toBeNull();
    expect(result.latencyMs).toBeGreaterThanOrEqual(0);
  });

  it('handles backend connection failure gracefully', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

    const result = await checkBackendHealth();
    expect(result.data).toBeNull();
    expect(result.error).toContain('Failed to fetch');
  });

  it('throws descriptive error on failed indexing request', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Book chunks have not been extracted yet' }),
    });

    await expect(indexBook('book_999')).rejects.toThrow('Book chunks have not been extracted yet');
  });
});
